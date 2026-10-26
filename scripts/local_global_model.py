"""One ResNet pass: native local triplets and study-wide contextual correction."""
import math
import torch
from local_global_data import MAX_WINDOWS, WindowBatch


class SliceModel(torch.nn.Module):
    def __init__(self, backbone, pretrained, dropout, targets, top_k_fraction=0.1):
        super().__init__()
        import torchvision
        if backbone not in {"resnet34", "resnet50"}:
            raise ValueError("Expected ResNet34 or ResNet50")
        self.backbone = getattr(torchvision.models, backbone)(weights="DEFAULT" if pretrained else None)
        old = self.backbone.conv1
        self.backbone.conv1 = torch.nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight.zero_()
            self.backbone.conv1.weight[:, :3].copy_(old.weight)
        features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
        self.head = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(features, targets))
        width = 256
        self.projection = torch.nn.Linear(features, width)
        self.plane = torch.nn.Embedding(4, width)
        self.fluid = torch.nn.Embedding(3, width)
        self.fat = torch.nn.Embedding(3, width)
        self.series = torch.nn.Embedding(4, width)
        self.position = torch.nn.Linear(8, width)
        # Acquisition metadata is a cue, not the dominant initial signal.
        # Default Embedding initialization has unit variance and would swamp
        # the projected image features before any useful contextual learning.
        for embedding in (self.plane, self.fluid, self.fat, self.series):
            torch.nn.init.normal_(embedding.weight, std=0.02)
        torch.nn.init.normal_(self.position.weight, std=0.02)
        torch.nn.init.zeros_(self.position.bias)
        self.input_norm = torch.nn.LayerNorm(width)
        self.context = torch.nn.TransformerEncoderLayer(width, 4, dim_feedforward=512,
                                                       dropout=0.1, batch_first=True, norm_first=True)
        self.queries = torch.nn.Parameter(torch.randn(targets, width) * 0.02)
        self.pool = torch.nn.MultiheadAttention(width, 4, dropout=0.1, batch_first=True)
        self.output_norm = torch.nn.LayerNorm(width)
        self.correction = torch.nn.Linear(width, 1)
        # Small, nonzero weights: contextual queries/attention receive gradients
        # from the first update, without swamping the retained local prediction.
        torch.nn.init.normal_(self.correction.weight, std=0.005)
        torch.nn.init.zeros_(self.correction.bias)
        self.top_k_fraction = float(top_k_fraction)
        if not 0 < self.top_k_fraction <= 1:
            raise ValueError("Invalid top-k fraction")
        self.to(memory_format=torch.channels_last)

    def forward(self, batch, study_of_slice, studies, fixed_windows_per_study=None, fixed_window_valid=None):
        if not isinstance(batch, WindowBatch) or batch.valid.shape != (studies, MAX_WINDOWS):
            raise ValueError("Native-triplet WindowBatch required")
        features = self.backbone(batch.images)
        logits = self.head(features)
        # Scatter using CPU-prepared indices: no boolean selection/nonzero CUDA
        # synchronization per study and no encoder work on padded windows.
        grouped = logits.new_zeros((studies * MAX_WINDOWS, logits.shape[-1])).index_copy(0, batch.padded_indices, logits)
        grouped = grouped.view(studies, MAX_WINDOWS, -1).masked_fill(~batch.valid[..., None], -torch.inf)
        max_k = max(1, math.ceil(MAX_WINDOWS * self.top_k_fraction))
        values = grouped.topk(max_k, dim=1).values
        counts = torch.ceil(batch.valid.sum(1) * self.top_k_fraction).long().clamp_min(1)
        use = torch.arange(max_k, device=logits.device)[None, :, None] < counts[:, None, None]
        local = values.masked_fill(~use, 0).sum(1) / counts[:, None]
        meta = batch.metadata
        discrete = meta[:, :4].long()
        tokens = (self.projection(features) + self.plane(discrete[:, 0]) + self.fluid(discrete[:, 1])
                  + self.fat(discrete[:, 2]) + self.series(discrete[:, 3]) + self.position(meta[:, 4:]))
        tokens = self.input_norm(tokens)
        grouped_tokens = tokens.new_zeros((studies * MAX_WINDOWS, tokens.shape[-1])).index_copy(0, batch.padded_indices, tokens)
        grouped_tokens = grouped_tokens.view(studies, MAX_WINDOWS, -1)
        padding = ~batch.valid
        context = self.context(grouped_tokens, src_key_padding_mask=padding)
        query = self.queries[None].expand(studies, -1, -1)
        pooled, _ = self.pool(query, context, context, key_padding_mask=padding, need_weights=False)
        delta = 0.1 * self.correction(self.output_norm(pooled)).squeeze(-1)
        return local + delta
