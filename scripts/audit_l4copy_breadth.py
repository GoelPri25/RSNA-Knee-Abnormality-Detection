"""Exp L4C-B (ledger row L4C-B, pre-registered 2026-10-18): is v4 l4copy's fold-0 gain broad or carried by one or two
findings? Fold 0, label-v2 confident cells, 12-target mean AUC, 2,000 paired study bootstraps. CPU only, existing files.
Control: our 3 label-v2 r50 retrains (lv2_s2027..s2029, same node and code, no crops); per study the seed-mean AUC is the
mean of the three per-model AUCs (not of averaged probabilities, which would be an ensemble).
  BROAD if  l4copy - label-v2 seed mean > 0 on >= 8 of 12 findings
       and  with the 2 largest per-finding gains removed, the 10-finding mean gain is >= +0.002 in >= 90% of resamples
            (the 2 findings are chosen on the full set and held fixed in the bootstrap).
  Reported for l4copy global (primary: the hypothesis is that crops in training help the shared stem-layer3) and fused.
  Descriptive: same table for shared (3 seeds, mean) and adapter; per-finding label-v2 seed SD; gain / SD.
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_l4copy_breadth.py <project root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(sys.argv[1]) / "repository/data/coatnet_d96_fold0"; RUN = C / "run_r50_gl4_mlx"
lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text()); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
LV2 = ["lv2_s2027", "lv2_s2028", "lv2_s2029"]
SHARED = ["v4_shared_s2026", "v4_shared_s2027", "v4_shared_s2028"]


def load(p):
    d = json.loads(p.read_text()); assert d["targets_in_order"] == T
    return {x["study_id"]: np.array(x["probabilities"]) for x in d["predictions"]}


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


P = {s: load(RUN / s / "val_swa.json") for s in LV2}
for d in SHARED + ["v4_adapter_s2026", "v4_l4copy_s2026"]:
    for h in ("global", "fused"):
        P[f"{d}:{h}"] = load(RUN / d / f"val_swa_{h}.json")
P["parent"] = load(C / "kaggle_fold0_labelv2/val_swa.json")
ids = sorted(P["parent"]); assert all(set(p) == set(ids) for p in P.values())
X = {k: np.array([v[s] for s in ids]) for k, v in P.items()}
Y = np.array([lab[s][0] for s in ids]); W = np.array([lab[s][1] for s in ids])
M = (W > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7


def per(x, i):
    return np.array([auc(x[i, k][M[i, k]], TT[i, k][M[i, k]]) for k in range(12)])


def group(keys, i):
    return np.mean([per(X[k], i) for k in keys], axis=0)


lines = []
say = lambda s="": (print(s, flush=True), lines.append(s))
al = np.arange(len(ids))
base = group(LV2, al); sd = np.std([per(X[k], al) for k in LV2], axis=0, ddof=1)
cand = {"l4copy global": ["v4_l4copy_s2026:global"], "l4copy fused": ["v4_l4copy_s2026:fused"],
        "adapter global": ["v4_adapter_s2026:global"], "adapter fused": ["v4_adapter_s2026:fused"],
        "shared global (3-seed)": [f"{d}:global" for d in SHARED], "shared fused (3-seed)": [f"{d}:fused" for d in SHARED]}
full = {nm: group(k, al) for nm, k in cand.items()}
say(f"L4C-B: fold 0, label-v2 confident cells ({len(ids)} studies); control = label-v2 r50 retrains {LV2} (seed mean of AUCs)")
tab = pd.DataFrame({"lv2 mean": base, "lv2 seed SD": sd, "parent": per(X["parent"], al)}, index=T)
for nm, v in full.items():
    tab[nm + " - lv2"] = v - base
say(tab.round(4).to_string())
say("mean: " + ", ".join(f"{c} {tab[c].mean():+.4f}" if "- lv2" in c else f"{c} {tab[c].mean():.4f}" for c in tab.columns))
say("gain / label-v2 seed SD (per finding), l4copy global: " + ", ".join(
    f"{t} {g / s:+.1f}" for t, g, s in zip(T, full["l4copy global"] - base, sd)))

rng = np.random.default_rng(2026); B = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
bt_base = np.array([group(LV2, i) for i in B])
for nm, k in cand.items():
    g = full[nm] - base; pos = int((g > 0).sum()); top2 = list(np.argsort(-g)[:2]); keep = [j for j in range(12) if j not in top2]
    bt = np.array([group(k, i) for i in B]) - bt_base
    m12 = np.nanmean(bt, axis=1); m10 = np.nanmean(bt[:, keep], axis=1)
    say(f"\n{nm}: 12-target gain {np.nanmean(g):+.4f} [{np.percentile(m12, 2.5):+.4f}, {np.percentile(m12, 97.5):+.4f}] "
        f"{np.mean(m12 > 0):.0%} > 0; findings > 0: {pos}/12")
    say(f"  top-2 removed ({T[top2[0]]} {g[top2[0]]:+.4f}, {T[top2[1]]} {g[top2[1]]:+.4f}): 10-finding gain {np.nanmean(g[keep]):+.4f} "
        f"[{np.percentile(m10, 2.5):+.4f}, {np.percentile(m10, 97.5):+.4f}], {np.mean(m10 >= 0.002):.0%} of resamples >= +0.002, "
        f"{np.mean(m10 > 0):.0%} > 0")
    if nm.startswith("l4copy"):
        ok = pos >= 8 and np.mean(m10 >= 0.002) >= 0.9
        say(f"  VERDICT ({nm}): {'BROAD' if ok else 'NARROW'} (rule: >= 8/12 findings > 0 and top-2-removed gain >= +0.002 in >= 90%)")
(RUN / "audit_l4copy_breadth.txt").write_text("\n".join(lines) + "\n")
