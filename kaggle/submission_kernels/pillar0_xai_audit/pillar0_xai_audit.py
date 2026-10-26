"""Inference-only XAI audit of the saved Pillar-0 target-query reader.

This kernel reads report-validation studies only. It does not load competition
labels, train a model, or export study/series identifiers.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
SCALES = ["fine_8x8x8", "native_8x8x8", "coarse_2x2x2"]
DIM = 384
TOKENS = (512, 512, 8)
ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/pillar0_xai")
SCRATCH = Path("/kaggle/temp/pillar0_xai_memmaps")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataset_dir(slug: str) -> Path:
    direct = ROOT / "datasets" / slug
    if direct.exists():
        return direct
    matches = list(ROOT.glob(f"**/{slug}"))
    matches = [p for p in matches if p.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one input dataset directory for {slug}; found {matches[:5]}")
    return matches[0]


def find_one(root: Path, name: str) -> Path:
    found = list(root.rglob(name))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one {name} below {root}, found {len(found)}")
    return found[0]


def read_labels(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    frame = pd.read_csv(path)
    id_col = "StudyInstanceUID" if "StudyInstanceUID" in frame else "study_id"
    missing = set(TARGETS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    result = {}
    for _, row in frame.iterrows():
        sid = str(row[id_col]).strip()
        target = np.full(12, 0.5, dtype=np.float32)
        weight = np.zeros(12, dtype=np.float32)
        for j, key in enumerate(TARGETS):
            value = pd.to_numeric(row[key], errors="coerce")
            if pd.notna(value) and np.isfinite(value) and 0 <= value <= 1:
                target[j], weight[j] = float(value), 1.0
        result[sid] = (target, weight)
    return result


def read_folds(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    id_col = "study_id" if "study_id" in frame else "StudyInstanceUID"
    return {str(row[id_col]).strip(): int(row["fold"]) for _, row in frame.iterrows()}


def plane_id(value: str) -> int:
    return {"axial": 1, "coronal": 2, "sagittal": 3, "oblique": 4}.get(value.strip().lower(), 0)


def bool_id(value: str) -> int:
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return 2
    if value in {"false", "0", "no", "n"}:
        return 1
    return 0


def text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def build_validation_store(cache: Path, folds: dict[str, int], labels: dict, fold: int = 0):
    """Expand only selected validation series into temporary quantized memmaps."""
    chunks = sorted(cache.glob("spatial_batch_*.npz"))
    if len(chunks) != 376:
        raise ValueError(f"Expected 376 cache batches, found {len(chunks)}")
    n_total = 0
    n_selected = 0
    for chunk in chunks:
        with np.load(chunk, allow_pickle=False) as z:
            n_total += len(z["study_uid"])
            n_selected += sum(
                text(sid) in labels and folds.get(text(sid)) == fold for sid in z["study_uid"]
            )
    if n_total != 24035:
        raise ValueError(f"Expected 24,035 tokenized series; found {n_total}")
    store = {
        "fine8": np.memmap(SCRATCH / "fine8.int8", mode="w+", dtype=np.int8, shape=(n_selected, 512, DIM)),
        "native8": np.memmap(SCRATCH / "native8.int8", mode="w+", dtype=np.int8, shape=(n_selected, 512, DIM)),
        "native2": np.memmap(SCRATCH / "native2.int8", mode="w+", dtype=np.int8, shape=(n_selected, 8, DIM)),
        "scales": np.memmap(SCRATCH / "scales.float32", mode="w+", dtype=np.float32, shape=(n_selected, 3, DIM)),
        "support8": np.memmap(SCRATCH / "support.uint8", mode="w+", dtype=np.uint8, shape=(n_selected, 512)),
    }
    records = []
    out = 0
    for k, chunk in enumerate(chunks, 1):
        with np.load(chunk, allow_pickle=False) as z:
            for i, raw_sid in enumerate(z["study_uid"]):
                sid = text(raw_sid)
                if sid not in labels or folds.get(sid) != fold:
                    continue
                for key in ("fine8", "native8", "native2", "support8"):
                    store[key][out] = z[key][i]
                store["scales"][out] = z["channel_scales"][i]
                records.append({
                    "index": out, "study_id": sid,
                    "series_id": text(z["series_uid"][i]),
                    "plane": plane_id(text(z["plane"][i])),
                    "fluid": bool_id(text(z["fluid_sensitive"][i])),
                    "fat": bool_id(text(z["fat_suppression"][i])),
                })
                out += 1
        if k % 64 == 0 or k == len(chunks):
            print(f"Indexed selected report-validation series from {k}/{len(chunks)} cache batches", flush=True)
    for arr in store.values():
        arr.flush()
    if out != n_selected:
        raise RuntimeError(f"Selected count mismatch: indexed {out}, expected {n_selected}")
    by_study = {}
    for rec in records:
        by_study.setdefault(rec["study_id"], []).append(rec)
    return store, by_study, records


def coordinates(side: int) -> np.ndarray:
    axis = (np.arange(side, dtype=np.float32) + 0.5) * (2.0 / side) - 1.0
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)


class Reader(nn.Module):
    """Checkpoint-compatible forward with optional attention trace."""
    def __init__(self, targets=12, width=128, heads=4, dropout=0.15, feature_dim=384):
        super().__init__()
        self.targets, self.width, self.heads = targets, width, heads
        self.head_dim = width // heads
        self.scale_norms = nn.ModuleList(nn.LayerNorm(feature_dim) for _ in range(3))
        self.scale_projections = nn.ModuleList(nn.Linear(feature_dim, width) for _ in range(3))
        self.scale_embeddings = nn.Embedding(3, width)
        self.position_projection = nn.Sequential(nn.Linear(18, width), nn.GELU(), nn.Linear(width, width))
        self.plane_embedding = nn.Embedding(5, 12, padding_idx=0)
        self.fluid_embedding = nn.Embedding(3, 6, padding_idx=0)
        self.fat_embedding = nn.Embedding(3, 6, padding_idx=0)
        self.protocol_projection = nn.Sequential(nn.Linear(24, width), nn.GELU(), nn.Linear(width, width))
        self.queries = nn.Parameter(torch.empty(targets, width))
        nn.init.normal_(self.queries, std=0.02)
        self.query_projection = nn.Linear(width, width, bias=False)
        self.key_projection = nn.Linear(width, width, bias=False)
        self.value_projection = nn.Linear(width, width, bias=False)
        self.output_projection = nn.Linear(width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(width)
        self.classifier = nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width // 2, 1))

    def forward(self, tokens, scale_ids, xyz, protocol_ids, support, padding_mask,
                return_attention=False, hard_mask=False):
        b, n, _ = tokens.shape
        x = tokens.new_zeros((b, n, self.width), dtype=torch.float32)
        ft, fs = tokens.reshape(-1, tokens.shape[-1]).float(), scale_ids.reshape(-1)
        fx = x.reshape(-1, self.width)
        for s in range(3):
            sel = fs == s
            if sel.any():
                fx[sel] = self.scale_projections[s](self.scale_norms[s](ft[sel])).float()
        x = x + self.scale_embeddings(scale_ids)
        frequencies = xyz.new_tensor((1.0, 2.0, 4.0)) * math.pi
        phase = xyz.float().unsqueeze(-1) * frequencies
        pos = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(start_dim=-2)
        plane = self.plane_embedding(protocol_ids[..., 0].long())
        fluid = self.fluid_embedding(protocol_ids[..., 1].long())
        fat = self.fat_embedding(protocol_ids[..., 2].long())
        protocol = self.protocol_projection(torch.cat((plane, fluid, fat), dim=-1))
        x = self.dropout(x + self.position_projection(pos) + protocol)
        q = self.query_projection(self.queries).view(self.targets, self.heads, self.head_dim)
        q = q.unsqueeze(0).expand(b, -1, -1, -1)
        k = self.key_projection(x).view(b, n, self.heads, self.head_dim)
        v = self.value_projection(x).view(b, n, self.heads, self.head_dim)
        logits = torch.einsum("bqhd,bnhd->bqhn", q, k).float() * (self.head_dim ** -0.5)
        logits = logits + support.float().clamp(min=0.05, max=1.0).log()[:, None, None, :]
        mask = padding_mask
        if hard_mask:
            mask = mask | (support <= 0.0)
        logits = logits.masked_fill(mask[:, None, None, :], -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        context = torch.einsum("bqhn,bnhd->bqhd", weights, v.float()).reshape(b, self.targets, self.width)
        target = self.attention_norm(self.queries.unsqueeze(0) + self.output_projection(context))
        target = self.output_norm(self.dropout(target))
        out = self.classifier(target).squeeze(-1)
        return (out, weights) if return_attention else out


class ValidationStudies(Dataset):
    def __init__(self, store, by_study, study_ids, labels, *, shuffle=False):
        self.store, self.by_study, self.study_ids, self.labels = store, by_study, study_ids, labels
        self.shuffle = shuffle
        self.c8, self.c2 = coordinates(8), coordinates(2)

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, i):
        sid = self.study_ids[i]
        toks, scales, xyzs, protos, supports = [], [], [], [], []
        for rec in self.by_study[sid]:
            j = rec["index"]
            q = self.store["scales"][j]
            blocks = [self.store["fine8"][j].astype(np.float32) * q[0],
                      self.store["native8"][j].astype(np.float32) * q[1],
                      self.store["native2"][j].astype(np.float32) * q[2]]
            if self.shuffle:
                rng = np.random.default_rng((j + 173205) % (2**32))
                blocks = [block[rng.permutation(len(block))] for block in blocks]
            toks.extend(blocks)
            scales.extend([np.zeros(512, np.int64), np.ones(512, np.int64), np.full(8, 2, np.int64)])
            xyzs.extend([self.c8, self.c8, self.c2])
            metadata = np.array([rec["plane"], rec["fluid"], rec["fat"]], dtype=np.int64)
            protos.extend([np.repeat(metadata[None], 512, 0), np.repeat(metadata[None], 512, 0), np.repeat(metadata[None], 8, 0)])
            s8 = self.store["support8"][j].astype(np.float32).reshape(8, 8, 8) / 255.0
            s2 = s8.reshape(2, 4, 2, 4, 2, 4).mean(axis=(1, 3, 5)).reshape(8)
            supports.extend([s8.reshape(512), s8.reshape(512), s2])
        return {
            "study_id": sid,
            "tokens": torch.from_numpy(np.concatenate(toks).astype(np.float16)),
            "scale_ids": torch.from_numpy(np.concatenate(scales)),
            "xyz": torch.from_numpy(np.concatenate(xyzs).astype(np.float16)),
            "protocol_ids": torch.from_numpy(np.concatenate(protos)),
            "support": torch.from_numpy(np.concatenate(supports).astype(np.float16)),
            "target": torch.from_numpy(self.labels[sid][0]),
            "weight": torch.from_numpy(self.labels[sid][1]),
        }


def collate(items):
    b, n = len(items), max(len(x["scale_ids"]) for x in items)
    tokens = torch.zeros((b, n, DIM), dtype=torch.float16)
    scales = torch.zeros((b, n), dtype=torch.long)
    xyz = torch.zeros((b, n, 3), dtype=torch.float16)
    protocol = torch.zeros((b, n, 3), dtype=torch.long)
    support = torch.zeros((b, n), dtype=torch.float16)
    pad = torch.ones((b, n), dtype=torch.bool)
    for i, item in enumerate(items):
        k = len(item["scale_ids"])
        for dst, key in ((tokens, "tokens"), (scales, "scale_ids"), (xyz, "xyz"),
                         (protocol, "protocol_ids"), (support, "support")):
            dst[i, :k] = item[key]
        pad[i, :k] = False
    return {"tokens": tokens, "scale_ids": scales, "xyz": xyz, "protocol_ids": protocol,
            "support": support, "padding_mask": pad,
            "target": torch.stack([x["target"] for x in items]),
            "weight": torch.stack([x["weight"] for x in items])}


def aucs(logits, target, weight):
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    result, vals = {}, []
    for j, name in enumerate(TARGETS):
        keep = (weight[:, j] > 0) & ((target[:, j] <= .3) | (target[:, j] >= .7))
        y = target[keep, j] >= .7
        if len(np.unique(y)) == 2:
            value = float(roc_auc_score(y, p[keep, j]))
            result[name] = value
            vals.append(value)
        else:
            result[name] = None
    return float(np.mean(vals)), result


def paired_bootstrap_auc(base_logits, alt_logits, target, weight, repeats=1000, seed=2609):
    """Paired study bootstrap; preserves row pairing across the two predictions."""
    rng = np.random.default_rng(seed)
    n = len(target)
    base_values, alt_values = [], []
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        base_auc, _ = aucs(base_logits[idx], target[idx], weight[idx])
        alt_auc, _ = aucs(alt_logits[idx], target[idx], weight[idx])
        if np.isfinite(base_auc) and np.isfinite(alt_auc):
            base_values.append(base_auc)
            alt_values.append(alt_auc)
    if len(base_values) < repeats * .8:
        print(f"Bootstrap warning: only {len(base_values)}/{repeats} replicates had valid per-target AUCs", flush=True)
    base_values = np.asarray(base_values)
    delta = np.asarray(alt_values) - base_values
    return {
        "base_auc_ci_low": float(np.quantile(base_values, .025)),
        "base_auc_ci_high": float(np.quantile(base_values, .975)),
        "delta_auc_ci_low": float(np.quantile(delta, .025)),
        "delta_auc_ci_high": float(np.quantile(delta, .975)),
    }


def move(batch):
    return tuple(batch[k].to(DEVICE, non_blocking=True) for k in
                 ("tokens", "scale_ids", "xyz", "protocol_ids", "support", "padding_mask"))


def run_predictions(model, loader, *, ablate=None, hard_mask=False, attention=False):
    ys, ws, ls, attentions = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = list(move(batch))
            tokens, scale_ids, xyz, proto, support, padding = inputs
            if ablate in {"fine", "native", "coarse", "all_visual"}:
                mask = (scale_ids == {"fine": 0, "native": 1, "coarse": 2}[ablate]) if ablate != "all_visual" else torch.ones_like(scale_ids, dtype=torch.bool)
                tokens = tokens.clone()
                tokens[mask] = 0
            if ablate == "protocol":
                proto = torch.zeros_like(proto)
            inputs = [tokens, scale_ids, xyz, proto, support, padding]
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                if attention:
                    logit, att = model(*inputs, return_attention=True, hard_mask=hard_mask)
                    attentions.append((att.float().cpu(), scale_ids.cpu(), proto.cpu(), support.float().cpu(), padding.cpu(), xyz.float().cpu()))
                else:
                    logit = model(*inputs, hard_mask=hard_mask)
            ls.append(logit.float().cpu().numpy())
            ys.append(batch["target"].numpy())
            ws.append(batch["weight"].numpy())
    logits, target, weight = np.concatenate(ls), np.concatenate(ys), np.concatenate(ws)
    cells = np.logaddexp(0, logits) - target * logits
    bce = float((cells * weight).sum() / max(weight.sum(), 1.0))
    auc, per_target = aucs(logits, target, weight)
    return {"bce": bce, "macro_auc": auc, "per_target_auc": per_target}, (logits, target, weight), attentions


def make_attention_profiles(attentions):
    rows = []
    for t, name in enumerate(TARGETS):
        mass = np.zeros(3, np.float64)
        count = np.zeros(3, np.float64)
        p_mass = np.zeros(5, np.float64); p_count = np.zeros(5, np.float64)
        s_mass = np.zeros(4, np.float64); s_count = np.zeros(4, np.float64)
        for att, scale, proto, support, pad, _xyz in attentions:
            # Average over attention heads; each study contributes equal total mass.
            a = att[:, t].mean(dim=1).numpy()
            sc, pr, su, pm = scale.numpy(), proto.numpy(), support.numpy(), pad.numpy()
            for b in range(len(a)):
                valid = ~pm[b]
                for s in range(3):
                    sel = valid & (sc[b] == s)
                    mass[s] += a[b, sel].sum(); count[s] += sel.sum()
                for p in range(1, 5):
                    sel = valid & (pr[b, :, 0] == p)
                    p_mass[p] += a[b, sel].sum(); p_count[p] += sel.sum()
                bins = np.select([su[b] <= 0.0, su[b] <= .05, su[b] <= .25],
                                 [0, 1, 2], default=3)
                for q in range(4):
                    sel = valid & (bins == q)
                    s_mass[q] += a[b, sel].sum(); s_count[q] += sel.sum()
        n_studies = len(attentions) * attentions[0][0].shape[0]
        mass /= max(n_studies, 1); p_mass /= max(n_studies, 1); s_mass /= max(n_studies, 1)
        total_tokens = count.sum()
        for s, label in enumerate(SCALES):
            rows.append({"target": name, "group": "scale", "category": label,
                         "attention_mass": mass[s], "token_share": count[s] / max(total_tokens, 1),
                         "enrichment": mass[s] / max(count[s] / max(total_tokens, 1), 1e-12)})
        for p, label in enumerate(["unknown", "axial", "coronal", "sagittal", "oblique"]):
            if p:
                rows.append({"target": name, "group": "plane", "category": label,
                             "attention_mass": p_mass[p], "token_share": p_count[p] / max(p_count.sum(), 1),
                             "enrichment": p_mass[p] / max(p_count[p] / max(p_count.sum(), 1), 1e-12)})
        for q, label in enumerate(["zero", "(0,.05]", "(.05,.25]", "(.25,1]"]):
            rows.append({"target": name, "group": "support", "category": label,
                         "attention_mass": s_mass[q], "token_share": s_count[q] / max(s_count.sum(), 1),
                         "enrichment": s_mass[q] / max(s_count[q] / max(s_count.sum(), 1), 1e-12)})
    return pd.DataFrame(rows)


def select_explanation_studies(study_ids, labels, limit=96):
    # Deterministic, label-aware greedy coverage of target/class strata without
    # exporting any identifiers. Validation is used only for explanation, never fitting.
    rng = np.random.default_rng(2609)
    by_stratum = {}
    for sid in study_ids:
        y, w = labels[sid]
        for t in range(12):
            if w[t] > 0 and (y[t] <= .3 or y[t] >= .7):
                by_stratum.setdefault((t, int(y[t] >= .7)), []).append(sid)
    for values in by_stratum.values():
        rng.shuffle(values)
    selected, cursor = [], {k: 0 for k in by_stratum}
    keys = list(by_stratum)
    while len(selected) < min(limit, len(study_ids)):
        changed = False
        for key in keys:
            if cursor[key] < len(by_stratum[key]):
                sid = by_stratum[key][cursor[key]]
                cursor[key] += 1
                if sid not in selected:
                    selected.append(sid); changed = True
                    if len(selected) >= limit:
                        break
        if not changed:
            break
    return selected


def gradient_input_profiles(model, loader):
    """Target-specific token Grad x Input, averaged per study and scale/grid cell."""
    scale_total = np.zeros((12, 3), np.float64)
    grid_total = np.zeros((12, 3, 8, 8, 8), np.float64)
    scale_n = np.zeros(12, np.int64)
    model.eval()
    for batch_i, batch in enumerate(loader, 1):
        inputs = list(move(batch))
        inputs[0] = inputs[0].detach().float().requires_grad_(True)
        tokens, scale_ids, xyz, proto, support, pad = inputs
        logits = model(tokens, scale_ids, xyz, proto, support, pad)
        valid = ~pad
        for target_i in range(12):
            grad = torch.autograd.grad(logits[:, target_i].sum(), tokens,
                                       retain_graph=target_i < 11)[0]
            attr = (grad * tokens).abs().sum(dim=-1)
            for b in range(tokens.shape[0]):
                if not valid[b].any():
                    continue
                row_attr = attr[b].detach().cpu().numpy()
                row_scale = scale_ids[b].cpu().numpy()
                row_xyz = xyz[b].cpu().numpy()
                row_valid = valid[b].cpu().numpy()
                total = row_attr[row_valid].sum()
                if total > 0:
                    row_attr = row_attr / total
                for s in range(3):
                    sel = row_valid & (row_scale == s)
                    scale_total[target_i, s] += row_attr[sel].sum()
                    ijk = np.clip(((row_xyz[sel] + 1.0) * 4).astype(int), 0, 7)
                    for val, (x, y, z) in zip(row_attr[sel], ijk):
                        grid_total[target_i, s, z, y, x] += val
                scale_n[target_i] += 1
        if batch_i % 4 == 0 or batch_i == len(loader):
            print(f"Grad×Input batches {batch_i}/{len(loader)}", flush=True)
    rows = []
    for t, name in enumerate(TARGETS):
        denom = max(int(scale_n[t]), 1)
        for s, label in enumerate(SCALES):
            rows.append({"target": name, "scale": label,
                         "normalized_grad_input_mass": scale_total[t, s] / denom})
    grid_denominator = np.maximum(scale_n, 1).reshape(12, 1, 1, 1, 1)
    return pd.DataFrame(rows), grid_total / grid_denominator


def main():
    if DEVICE.type != "cuda":
        raise RuntimeError("Attach a Kaggle T4 GPU for this inference-only XAI audit")
    WORK.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    print(f"Device: {torch.cuda.get_device_name(0)} | training: none | gold labels: not loaded", flush=True)
    cache = dataset_dir("rsna-pillar0-spatial-token-cache-v1")
    artifacts = dataset_dir("rsna-pillar0-target-query-reader-v1")
    labels_root = dataset_dir("rsna-knee-llm-report-labels")
    source = dataset_dir("rsna-cnn-topk-support-source-v1")
    labels = read_labels(find_one(labels_root, "llm_labels_v4_blend.csv"))
    folds = read_folds(find_one(source, "report_folds_v5.csv"))
    checkpoint_path = find_one(artifacts, "checkpoint_best.pt")
    cache_config = json.loads(find_one(cache, "run_config.json").read_text())
    if cache_config.get("encoder_token_gating_applied") is not False:
        raise ValueError("Expected ungated baseline contextual Pillar-0 cache")
    if cache_config.get("support_map_saved") is not True:
        raise ValueError("Support map missing from token cache")
    store, by_study, records = build_validation_store(cache, folds, labels, fold=0)
    study_ids = sorted(by_study)
    if len(study_ids) != 825:
        print(f"Notice: report fold contains {len(study_ids)} labeled studies (expected around 825)", flush=True)
    print(f"Validation-only data loaded: {len(study_ids)} studies, {len(records)} series; all other studies excluded from analysis", flush=True)
    model = Reader().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Checkpoint epoch={checkpoint['best_epoch']} macroAUC={checkpoint['best_auc']:.6f} trainable={n_params:,}", flush=True)

    base_data = ValidationStudies(store, by_study, study_ids, labels)
    loader = DataLoader(base_data, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)
    baseline, baseline_raw, attention = run_predictions(model, loader, attention=True)
    print("Baseline validation:", json.dumps(baseline), flush=True)
    baseline_logits, baseline_target, baseline_weight = baseline_raw
    baseline_ci = paired_bootstrap_auc(baseline_logits, baseline_logits, baseline_target, baseline_weight)
    results = [{"condition": "baseline", **{k: baseline[k] for k in ("bce", "macro_auc")},
                **{k: baseline_ci[k] for k in ("base_auc_ci_low", "base_auc_ci_high")}}]
    for condition, kwargs in [
        ("zero_fine_visual_tokens", {"ablate": "fine"}),
        ("zero_native_8cube_visual_tokens", {"ablate": "native"}),
        ("zero_coarse_visual_tokens", {"ablate": "coarse"}),
        ("zero_all_visual_tokens", {"ablate": "all_visual"}),
        ("unknown_protocol_metadata", {"ablate": "protocol"}),
        ("hard_mask_zero_support_tokens", {"hard_mask": True}),
    ]:
        result, raw, _ = run_predictions(model, loader, **kwargs)
        boot = paired_bootstrap_auc(baseline_logits, raw[0], baseline_target, baseline_weight,
                                    seed=2609 + len(results))
        results.append({"condition": condition, "bce": result["bce"], "macro_auc": result["macro_auc"],
                        "delta_macro_auc": result["macro_auc"] - baseline["macro_auc"], **boot})
        print(f"Ablation {condition}: BCE={result['bce']:.6f} macroAUC={result['macro_auc']:.6f} "
              f"paired ΔAUC CI=[{boot['delta_auc_ci_low']:.5f}, {boot['delta_auc_ci_high']:.5f}]", flush=True)
    shuffled_loader = DataLoader(ValidationStudies(store, by_study, study_ids, labels, shuffle=True),
                                 batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)
    shuffled, shuffled_raw, _ = run_predictions(model, shuffled_loader)
    boot = paired_bootstrap_auc(baseline_logits, shuffled_raw[0], baseline_target, baseline_weight,
                                seed=2609 + len(results))
    results.append({"condition": "within_series_feature_to_coordinate_shuffle", "bce": shuffled["bce"],
                    "macro_auc": shuffled["macro_auc"],
                    "delta_macro_auc": shuffled["macro_auc"] - baseline["macro_auc"], **boot})
    print(f"Spatial alignment shuffle: BCE={shuffled['bce']:.6f} macroAUC={shuffled['macro_auc']:.6f} "
          f"paired ΔAUC CI=[{boot['delta_auc_ci_low']:.5f}, {boot['delta_auc_ci_high']:.5f}]", flush=True)
    pd.DataFrame(results).to_csv(WORK / "counterfactual_ablation.csv", index=False)

    attention_frame = make_attention_profiles(attention)
    attention_frame.to_csv(WORK / "attention_profiles.csv", index=False)
    explanation_ids = select_explanation_studies(study_ids, labels, limit=96)
    explanation_loader = DataLoader(ValidationStudies(store, by_study, explanation_ids, labels),
                                    batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)
    saliency_frame, grid = gradient_input_profiles(model, explanation_loader)
    saliency_frame.to_csv(WORK / "gradient_input_scale_profiles.csv", index=False)
    np.savez_compressed(WORK / "gradient_input_spatial_grids.npz", grid=grid)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    ab = pd.DataFrame(results)
    axes[0].barh(ab["condition"], ab["macro_auc"] - baseline["macro_auc"])
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set(title="Macro AUC change under counterfactuals", xlabel="Δ report-validation macro AUC")
    mean_sal = saliency_frame.groupby("scale")["normalized_grad_input_mass"].mean().reindex(SCALES)
    mean_att = attention_frame[attention_frame.group == "scale"].groupby("category")["attention_mass"].mean().reindex(SCALES)
    x = np.arange(3); width = .36
    axes[1].bar(x - width/2, mean_att.values, width, label="Attention mass")
    axes[1].bar(x + width/2, mean_sal.values, width, label="|Gradient × input|")
    axes[1].set_xticks(x, ["Fine 8³", "Native 8³", "Coarse 2³"])
    axes[1].set(title="Where the reader routes and uses signal", ylabel="Mean mass per study")
    axes[1].legend(); axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(WORK / "pillar0_xai_summary.png"); plt.close(fig)

    summary = {
        "experiment": "inference_only_pillar0_reader_xai",
        "training_performed": False,
        "gold_labels_loaded": False,
        "validation_fold": 0,
        "validation_studies": len(study_ids),
        "validation_series": len(records),
        "checkpoint_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_macro_auc_at_training": float(checkpoint["best_auc"]),
        "trainable_parameters": n_params,
        "baseline_validation": baseline,
        "counterfactuals": results,
        "saliency_studies": len(explanation_ids),
        "cache_checkpoint_revision": cache_config.get("checkpoint_revision"),
        "outputs": [p.name for p in WORK.iterdir() if p.is_file()],
        "privacy": "No study or series identifiers exported; no raw DICOM accessed.",
    }
    (WORK / "xai_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print("XAI audit complete:", json.dumps(summary, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
