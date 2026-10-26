"""E4-ALL full v4 l4copy checkpoint -> slim fp16 global-head weights for the fast kernel (FAST-G).
Same fold as the F1 slim file (git d8ed14c slim_from_full): BN folded into each conv, global head only, crop branch
dropped. Checks the result has F1's slim keys, shapes and dtypes."""
import glob
import hashlib
import os

import torch

LAYERS = (3, 4, 6, 3)


def find(name):
    return next(p for p in glob.glob("/kaggle/input/**/" + name, recursive=True))


def slim_from_full(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    out = {}

    def fold(conv, bn, name):
        w = sd[f"backbone.{conv}.weight"].float()
        g, b = sd[f"backbone.{bn}.weight"].float(), sd[f"backbone.{bn}.bias"].float()
        m, v = sd[f"backbone.{bn}.running_mean"].float(), sd[f"backbone.{bn}.running_var"].float()
        s = g / torch.sqrt(v + 1e-5)
        out[name + ".w"] = (w * s.view(-1, 1, 1, 1)).half()
        out[name + ".b"] = (b - m * s).half()
    fold("conv1", "bn1", "stem")
    for li, n in enumerate(LAYERS, 1):
        for bi in range(n):
            p = f"layer{li}.{bi}"
            for c in (1, 2, 3):
                fold(f"{p}.conv{c}", f"{p}.bn{c}", f"{p}.c{c}")
            if f"backbone.{p}.downsample.0.weight" in sd:
                fold(f"{p}.downsample.0", f"{p}.downsample.1", f"{p}.ds")
    for k in ("norm.weight", "norm.bias", "att.0.weight", "att.0.bias", "att.3.weight", "att.3.bias", "clsW", "clsb"):
        out[k] = sd[k].float()
    return out


full = find("weights_swa21_25.pt")
sha = hashlib.sha256(open(full, "rb").read()).hexdigest()
assert sha.startswith("94d6ff12"), sha
W = slim_from_full(full)
ref = torch.load(find("f1_global_fp16.pt"), map_location="cpu", weights_only=True)
assert list(W) == list(ref), "key order differs from F1 slim"
for k in W:
    assert W[k].shape == ref[k].shape and W[k].dtype == ref[k].dtype, k
    assert torch.isfinite(W[k].float()).all(), k
diff = max((W[k].float() - ref[k].float()).abs().max().item() for k in W)
dst = "/kaggle/working/e4all_global_fp16.pt"
torch.save(W, dst)
print(f"full {os.path.basename(full)} sha {sha[:12]} | {len(W)} tensors, same keys/shapes/dtypes as F1 slim | "
      f"max|E4ALL-F1| {diff:.4f} (nonzero = different weights) | {dst} {os.path.getsize(dst)/2**20:.1f} MB "
      f"sha {hashlib.sha256(open(dst, 'rb').read()).hexdigest()[:12]}", flush=True)
