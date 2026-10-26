"""Clear-band screen (6 targets with >= 20 report-moderate+ positives) for every fold-0 candidate, its fit to the public
LB over the submitted d96 singles, and what the screen says about candidates not yet submitted (incl. 2-seed means
and mirror TTA). Development evidence only; 9 LB points with +-0.002 noise.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds python -I /proj/repository/scripts/audit_clear_band_lb.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
clear = np.zeros((len(ids), 12), bool)
for i, s in enumerate(ids):
    if s in st.index:
        for k, t in enumerate(T):
            clear[i, k] = st.at[s, f"{key(t)}__status"] == "present" and st.at[s, f"{key(t)}__severity"] in (
                "moderate", "moderate_to_severe", "severe")
CB = [k for k in range(12) if (mask[:, k] & truth[:, k] & clear[:, k]).sum() >= 20]


def load(p):
    d = json.loads(Path(p).read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
    tg = d.get("targets_in_order", d.get("targets", T)); a = np.array([m[s] for s in ids], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def scores(S, idx=None):
    idx = np.arange(len(ids)) if idx is None else idx
    c6 = []
    for k in CB:
        sel = idx[mask[idx, k] & (~truth[idx, k] | clear[idx, k])]
        c6.append(auc(S[sel, k], truth[sel, k]))
    m10 = np.mean([auc(S[idx[mask[idx, k]], k], truth[idx[mask[idx, k]], k]) for k in TEN])
    return np.mean(c6), m10


run = lambda d, f="val_swa.json": C / d / f
M = {"1b": run("run_continue"), "1c": run("run_r50"), "1e": run("run_r50_pertarget"), "1e_s27": run("run_r50_pertarget_seed"),
     "3a": run("run_r50_border"), "3b": run("run_r50_state"), "2d": run("run_r50_plane_pertarget"), "v3": run("run_r50_spatial"),
     "2a": run("run_r50_ctx"), "2c": run("run_r50_plane"), "4b": run("run_r50_native_pertarget"), "5a": run("run_r50_ssl_pertarget"),
     "6a": run("run_r50_swa_cyclic"), "6b": run("run_r50_series_alloc"), "C0": run("run_r50_os16_gap"),
     "v3_os32": run("run_r50_spatial_os32"), "v3.1": run("run_r50_spatial_v31b"),
     "1e_mirror": run("run_r50_pertarget", "val_swa_mirror.json"), "1e_s27_mirror": run("run_r50_pertarget_seed", "val_swa_mirror.json"),
     "v3_os32_mirror": run("run_r50_spatial_os32", "val_swa_mirror.json"),
     "1e_best_epoch": run("run_r50_pertarget", "val_best_epoch.json"),
     "d2_224": D / "r50_tta/kaggle_output/d2_ep15__none.json",
     "B+D": run("run_r50_dual_guided"), "B+D_mirror": run("run_r50_dual_guided", "val_swa_mirror.json"),
     "B+D_head1": run("run_r50_dual_guided", "val_swa_h1.json"), "B+D_head2": run("run_r50_dual_guided", "val_swa_h2.json")}
LB = {"1b": .933, "1c": .933, "1e": .936, "1e_s27": .934, "3a": .926, "3b": .934, "2d": .924, "v3": .933}
P = {k: load(v) for k, v in M.items() if Path(v).is_file()}
P["mean(1e, 1e_s27)"] = (P["1e"] + P["1e_s27"]) / 2
P["mean(1e, v3_os32)"] = (P["1e"] + P["v3_os32"]) / 2
P["L4"] = (P["d2_224"] + P["1b"] + P["2c"] + P["1e"]) / 4
LB["L4"] = .943
if "B+D" in P:
    P["L4: 1e -> B+D"] = (P["d2_224"] + P["1b"] + P["2c"] + P["B+D"]) / 4
rows = {k: scores(S) for k, S in P.items()}
x = np.array([rows[k][0] for k in LB if k != "L4"]); yl = np.array([LB[k] for k in LB if k != "L4"])
b, a = np.polyfit(x, yl, 1)
r = np.corrcoef(x, yl)[0, 1]
print(f"clear-band targets: {[T[k] for k in CB]}")
print(f"fit over {len(x)} submitted singles: LB = {a:.3f} + {b:.3f} * clear6   (Pearson {r:+.2f}; residual sd "
      f"{np.std(yl - (a + b * x), ddof=2):.4f})\n")
print(f"{'model':20s} {'clear6':>7s} {'10-tgt':>7s} {'fit LB':>7s} {'LB':>6s}")
for k, (c6, m10) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
    print(f"{k:20s} {c6:7.4f} {m10:7.4f} {a + b * c6:7.3f} {LB.get(k, float('nan')):6.3f}")
rng = np.random.default_rng(0); n = len(ids)
print("\nPaired bootstrap of clear6 (2,000), vs 1e:")
boots = [rng.integers(0, n, n) for _ in range(2000)]
base = np.array([scores(P["1e"], i)[0] for i in boots])
for k in ("1e_s27", "mean(1e, 1e_s27)", "1e_mirror", "v3_os32", "mean(1e, v3_os32)", "L4", "B+D", "B+D_mirror",
          "B+D_head1", "B+D_head2", "L4: 1e -> B+D"):
    d = np.array([scores(P[k], i)[0] for i in boots]) - base
    print(f"  {k:20s} {d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] {(d > 0).mean():.0%} better")
