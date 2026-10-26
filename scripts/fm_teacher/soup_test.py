"""SOUP test (2026-10-22): are heliosli's R7 / R6 ConvNeXt-S fold checkpoints (E_lrprior_s0_f0..f3, all fine-tuned from
convnext_small_in22k_ft_in1k) in one basin, so that a weight average of their encoders is a usable start for an R7-D
student? No training: forward passes only.
Inputs: the first N studies of the competition d96 cache shard 0 (uint8 [96, 384, 384] per study), W windows per study
(3 consecutive slices of one slot as RGB), 320 centre crop, ImageNet normalisation (as R7). Features = global-average
pooled stage-4 maps (768-d, before head.norm).
Per model / soup: mean feature norm; linear CKA and mean per-sample cosine vs each member; members vs each other as the
reference. A soup in one basin keeps CKA / cosine to its members at least as high as members have to each other and its
feature norm near theirs; a broken soup collapses (norm drop, CKA well below the member-member level).
Also prints the non-encoder keys of one checkpoint (head structure).
Usage: python soup_test.py <dir with r7_f*.pt r6_f*.pt> <out dir> [N studies] [W windows]"""
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[3]) if len(sys.argv) > 3 else 48
W = int(sys.argv[4]) if len(sys.argv) > 4 else 12
dev = "cuda"
torch.backends.cudnn.benchmark = True


def enc_of(p):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    return {k[4:]: v.float() for k, v in sd.items() if k.startswith("enc.")}, sd


encs, raw = {}, None
for fam in ("r7", "r6"):
    for f in range(4):
        encs[f"{fam}_f{f}"], sd = enc_of(SRC / f"{fam}_f{f}.pt")
        raw = raw or sd
print("non-encoder keys (r7_f0):", {k: tuple(v.shape) for k, v in raw.items() if not k.startswith("enc.")}, flush=True)
keys = list(encs["r7_f0"])
assert all(list(e) == keys for e in encs.values()), "key sets differ"


def avg(names):
    return {k: torch.stack([encs[n][k] for n in names]).mean(0) for k in keys}


# weight-space distances: how far each fold moved from the others (relative L2 over all encoder tensors)
def rel(a, b):
    num = sum(((a[k] - b[k]) ** 2).sum() for k in keys); den = sum((b[k] ** 2).sum() for k in keys)
    return float((num / den).sqrt())


models = dict(encs)
models["soup_r7"] = avg([f"r7_f{f}" for f in range(4)])
models["soup_r6"] = avg([f"r6_f{f}" for f in range(4)])
models["soup_all8"] = avg(list(encs))
models["soup_r7_f01"] = avg(["r7_f0", "r7_f1"])
dist = {"r7_f0-r7_f1": rel(encs["r7_f0"], encs["r7_f1"]), "r7_f0-r6_f0": rel(encs["r7_f0"], encs["r6_f0"]),
        "r6_f0-r6_f1": rel(encs["r6_f0"], encs["r6_f1"])}
frozen = [k for k in keys if all(torch.equal(encs["r7_f0"][k], encs[f"r7_f{f}"][k]) for f in range(1, 4))]
print("relative weight distance:", {k: round(v, 4) for k, v in dist.items()}, "| tensors identical across R7 folds:",
      len(frozen), "of", len(keys), flush=True)

# inputs
vol = np.load("/data/d96/shard_0/vol_000.npy", mmap_mode="r")
rows = [json.loads(l) for l in open("/data/d96/shard_0/manifest.jsonl")][:N]
rng = np.random.default_rng(0); xs = []
for i, r in enumerate(rows):
    v = vol[i * 96:(i + 1) * 96]; ok = np.where(np.array(r["mask"]) > 0)[0]
    starts = [s for s in ok if s + 2 < 96 and r["mask"][s + 1] and r["mask"][s + 2]]
    for s in rng.choice(starts, size=min(W, len(starts)), replace=False):
        xs.append(np.asarray(v[s:s + 3, 32:352, 32:352]))
X = torch.from_numpy(np.stack(xs)).float().div_(255)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
X = (X - mean) / std
print("inputs", tuple(X.shape), flush=True)

net = timm.create_model("convnext_small", pretrained=False, num_classes=0).to(dev).eval()
hn = net.state_dict()
feats = {}
for name, e in models.items():
    e = dict(e); e["head.norm.weight"] = torch.ones_like(hn["head.norm.weight"]); e["head.norm.bias"] = torch.zeros_like(hn["head.norm.bias"])
    net.load_state_dict(e, strict=True)
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, len(X), 64):
            out.append(net.forward_features(X[b:b + 64].to(dev)).float().mean((2, 3)).cpu())
    feats[name] = torch.cat(out).numpy().astype(np.float64)


def cka(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return float(np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def cos(a, b):
    return float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))


norm = {k: float(np.linalg.norm(v, axis=1).mean()) for k, v in feats.items()}
res = {"inputs": list(X.shape), "weight_distance": dist, "identical_tensors_r7": len(frozen), "n_tensors": len(keys),
       "feature_norm": norm, "pairs": {}}
pairs = [("r7_f0", "r7_f1"), ("r7_f0", "r7_f2"), ("r7_f2", "r7_f3"), ("r6_f0", "r6_f1"), ("r7_f0", "r6_f0"),
         ("soup_r7_f01", "r7_f0"), ("soup_r7_f01", "r7_f1")] + [("soup_r7", f"r7_f{f}") for f in range(4)] \
        + [("soup_r6", f"r6_f{f}") for f in range(4)] + [("soup_all8", m) for m in ("r7_f0", "r7_f2", "r6_f0", "r6_f2", "soup_r7", "soup_r6")]
print(f"\n{'pair':28s} {'CKA':>6s} {'cos':>6s}")
for a, b in pairs:
    c, s = cka(feats[a], feats[b]), cos(feats[a], feats[b]); res["pairs"][f"{a}|{b}"] = {"cka": c, "cos": s}
    print(f"{a + ' vs ' + b:28s} {c:6.3f} {s:6.3f}")
print("\nmean feature norm:", {k: round(v, 2) for k, v in norm.items()})
(OUT / "soup_test.json").write_text(json.dumps(res, indent=1))
print("wrote", OUT / "soup_test.json")
