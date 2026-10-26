"""Score experiment 1 (CoAtNet d96, fold 0) against the pre-registered rule.

Fold-0 ruler: Steven-v4, confident cells (<= 0.3 / >= 0.7), 825 studies. Baseline: d2_224 (d2_ep15, no TTA,
0.8963) and its 9-view shift TTA (0.8996), both from the r50 TTA kernel output.
Primary: val_swa.json. Accept if macro AUC - 0.8963 >= +0.005 with >= 90% of 2,000 paired study bootstraps
better than the baseline. Also: per target, secondary variants (mirror TTA, best epoch), each epoch,
the quad with and without the new model added, and gold-58 AUC (development evidence only).

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_coatnet_d96.py /proj repository/data/coatnet_d96_fold0/run
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, RUN = Path(sys.argv[1]), Path(sys.argv[1]) / sys.argv[2]
D = "repository/data/"
REF = D + "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json"
BASE = {"d2_224": D + "r50_tta/kaggle_output/d2_ep15__none.json",
        "d2_224_tta9": D + "r50_tta/kaggle_output/d2_ep15__tta9.json"}
QUAD = [D + "r50_tta/kaggle_output/d2_ep15__none.json", D + "r50_tta/kaggle_output/hpo_v1_ep17__none.json",
        D + "highres336/results/oof_predictions.json", D + "efficientnet_v31_d2/results/oof_predictions.json"]
ref = json.loads((R / REF).read_text()); T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7


def load(p, order=ids):
    d = json.loads(Path(p).read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
    tg = d.get("targets_in_order", d.get("targets", T)); a = np.array([m[s] for s in order], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, yy):
    r = pd.Series(s).rank().to_numpy(); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per(S, idx, m=mask, t=truth):
    return np.array([auc(S[idx, k][m[idx, k]], t[idx, k][m[idx, k]]) for k in range(len(T))])


rng = np.random.default_rng(0); n = len(ids)
boots = [rng.integers(0, n, n) for _ in range(2000)]
full = np.arange(n)


def compare(A, B):
    d = np.array([np.nanmean(per(A, b)) - np.nanmean(per(B, b)) for b in boots])
    return {"delta": np.nanmean(per(A, full)) - np.nanmean(per(B, full)), "p2.5": np.percentile(d, 2.5),
            "p97.5": np.percentile(d, 97.5), "share_better": (d > 0).mean()}


base = {k: load(R / v) for k, v in BASE.items()}
models = {k: load(RUN / f"val_{k}.json") for k in ("swa", "swa_mirror", "best_epoch") if (RUN / f"val_{k}.json").is_file()}
rows, pt = [], {}
for name, S in {**base, **models}.items():
    pt[name] = per(S, full)
    row = {"model": name, "macro": pt[name].mean()}
    if name in models:
        row.update({f"vs_d2_224_{k}": v for k, v in compare(S, base["d2_224"]).items()})
    rows.append(row)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\nPer target:")
print(pd.DataFrame(pt, index=T).round(4).to_string())

epochs = sorted(RUN.glob("val_epoch*.json"))
if epochs:
    print("\nPer epoch (no TTA):", " ".join(f"{p.stem[-2:]}:{per(load(p), full).mean():.4f}" for p in epochs))

if "swa" in models:
    r = compare(models["swa"], base["d2_224"])
    ok = r["delta"] >= 0.005 and r["share_better"] >= 0.90
    print(f"\nPRE-REGISTERED (swa vs d2_224 0.8963): delta {r['delta']:+.4f} [{r['p2.5']:+.4f}, {r['p97.5']:+.4f}], "
          f"{r['share_better']:.1%} better -> {'PASS' if ok else 'FAIL'}")
    quad = np.mean([load(R / p) for p in QUAD], 0)
    print("Quad", round(np.nanmean(per(quad, full)), 4), "| quad + swa (equal mean)",
          {k: round(v, 4) for k, v in compare(np.mean([load(R / p) for p in QUAD] + [models["swa"]], 0), quad).items()})

gold_rows = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8"))
             if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in gold_rows]
gy = np.array([[int(r[t]) for t in T] for r in gold_rows]).astype(bool)
for k in ("swa", "swa_mirror", "best_epoch"):
    p = RUN / f"gold_{k}.json"
    if p.is_file():
        G = load(p, gids)
        g = np.array([auc(G[:, j], gy[:, j]) if 0 < gy[:, j].sum() < len(gy) else np.nan for j in range(len(T))])
        print(f"gold-58 {k}: macro {np.nanmean(g):.4f} (development evidence only) "
              + " ".join(f"{t[:6]}={v:.3f}" for t, v in zip(T, g)))
