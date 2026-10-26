"""Exp r50-GL scorer (ledger, pre-registered 2026-10-18). Fold 0, label-v2 confident cells (weight > 0, y <= 0.3 / >= 0.7),
12-target mean AUC, paired study bootstrap (2,000).
  Bars side by side: parent = label-v2 r50 (kaggle_fold0_labelv2/val_swa.json, 0.9425, LB 0.937, the best single);
                     E3 = label-v2 r50 + 4-network + CoAtNet 1b + P1-cnx (0.9485, LB 0.944).
  BEST SINGLE:  fused - label-v2 r50 >= +0.003 and 95% lower bound > 0           -> single LB submission
  ATTRIBUTION:  fused - own global   >= +0.002 and 95% lower bound > 0           (required together with best single)
  ENSEMBLE:     E3 with label-v2 r50 swapped for r50-GL fused >= E3 + 0.002, >= 90% of bootstraps better
                                                                                -> ensemble LB submission
  Descriptive: global / local (10 cropped findings) / fused without crop tokens / top-1 crops / multi-pass, per
  finding; Steven clear6 / 10-target (scripts/audit_phase1.py definitions) for the LB-fit check.
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_r50_gl.py <project root> [run dir, default run_r50_gl] [tag, default swa]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
RUN = C / (sys.argv[2] if len(sys.argv) > 2 else "run_r50_gl"); TAG = sys.argv[3] if len(sys.argv) > 3 else "swa"
OUT = RUN / f"audit_r50_gl_{TAG}.txt"; lines = []
say = lambda s="": (print(s, flush=True), lines.append(s))
lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text()); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
CROPPED = [k for k, t in enumerate(T) if t not in ("Effusion", "Synovitis")]


def load(p):
    d = json.loads(p.read_text()); assert d["targets_in_order"] == T
    return {x["study_id"]: np.array(x["probabilities"]) for x in d["predictions"]}


name = lambda nm: RUN / f"val_{TAG}_{nm}.json"
TM = D / "teammate_pp4/fold0_predictions"
P = {"lv2": load(C / "kaggle_fold0_labelv2/val_swa.json"), "4net": load(TM / "four_network_fold0_val_swa.json"),
     "1b": load(C / "run_continue/val_swa.json"), "cnx": load(C / "run_cnx_pertarget/val_swa.json"),
     **{nm: load(name(nm)) for nm in ("fused", "global", "local")},
     **{nm: load(name(nm)) for nm in ("fused_nocrop", "fused_top1", "fused_multi6") if name(nm).is_file()}}
V3 = not name("fused_nocrop").is_file()                       # r50-GL v3 (gated): no crop-token ablation
if V3:
    P["v2 local"] = load(C / "run_r50_gl_v2/val_swa_local.json")
ids = sorted(P["lv2"]); assert all(set(p) == set(ids) for p in P.values())
X = {k: np.array([v[s] for s in ids]) for k, v in P.items()}
E3M = ["lv2", "4net", "1b", "cnx"]
X["E3"] = np.mean([X[m] for m in E3M], 0)
X["E3 swap lv2->GL"] = np.mean([X[m] for m in E3M[1:]] + [X["fused"]], 0)
X["E3+GL"] = np.mean([X[m] for m in E3M] + [X["fused"]], 0)
if V3:                                                         # fixed beta = 0.5 in logit space (local = global off-region)
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / np.clip(1 - q, 1e-6, 1))
    X["halfgate"] = 1 / (1 + np.exp(-(lg(X["global"]) + lg(X["local"])) / 2))
    X["lv2 + v2 local"] = np.where(np.isclose(X["v2 local"], X["lv2"]), X["lv2"], (X["lv2"] + X["v2 local"]) / 2)
Y = np.array([lab[s][0] for s in ids]); W = np.array([lab[s][1] for s in ids])
M = (W > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(x, idx=None):
    idx = np.arange(len(ids)) if idx is None else idx
    return np.array([auc(x[idx, k][M[idx, k]], TT[idx, k][M[idx, k]]) for k in range(12)])


cols = ["lv2", "fused", "global", "local"] + [c for c in ("fused_nocrop", "fused_top1", "fused_multi6", "halfgate",
                                                         "v2 local") if c in X] + ["E3", "E3 swap lv2->GL", "E3+GL"]
tab = pd.DataFrame({k: per(X[k]) for k in cols}, index=T)
tab["fused-lv2"] = tab["fused"] - tab["lv2"]; tab["fused-global"] = tab["fused"] - tab["global"]
tab["global-lv2"] = tab["global"] - tab["lv2"]
say(f"r50-GL {TAG}: fold 0 AUC per finding, label-v2 confident cells ({len(ids)} studies)")
say(tab.round(4).to_string())
say("mean (12): " + ", ".join(f"{c} {tab[c].mean():+.4f}" if "-" in c and "swap" not in c else f"{c} {tab[c].mean():.4f}"
                              for c in tab))
say(f"local head on its 10 cropped findings: {tab['local'].iloc[CROPPED].mean():.4f} "
    f"(own global {tab['global'].iloc[CROPPED].mean():.4f}, label-v2 r50 {tab['lv2'].iloc[CROPPED].mean():.4f})")

rng = np.random.default_rng(2026); B = 2000
pairs = {"fused-lv2": ("fused", "lv2"), "fused-global": ("fused", "global"), "global-lv2": ("global", "lv2"),
         "swap-E3": ("E3 swap lv2->GL", "E3"), "E3+GL-E3": ("E3+GL", "E3"),
         **({"fused-halfgate": ("fused", "halfgate"), "local-v2 local": ("local", "v2 local")} if V3 else {})}
need = sorted({a for p in pairs.values() for a in p})
bt = {k: [] for k in pairs}
for _ in range(B):
    i = rng.integers(0, len(ids), len(ids)); a = {k: np.nanmean(per(X[k], i)) for k in need}
    for k, (u, v) in pairs.items():
        bt[k].append(a[u] - a[v])
say("")
for k, (u, v) in pairs.items():
    lo, hi = np.percentile(bt[k], [2.5, 97.5])
    say(f"{k:13s}: {tab[u].mean() - tab[v].mean():+.4f}  bootstrap mean {np.mean(bt[k]):+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], "
        f"{np.mean(np.array(bt[k]) > 0):.0%} > 0")
lo = lambda k: np.percentile(bt[k], 2.5)
g_best = tab["fused-lv2"].mean() >= .003 and lo("fused-lv2") > 0
g_attr = tab["fused-global"].mean() >= .002 and lo("fused-global") > 0
g_ens = tab["E3 swap lv2->GL"].mean() - tab["E3"].mean() >= .002 and np.mean(np.array(bt["swap-E3"]) > 0) >= .9
say("")
say(f"BEST SINGLE (fused vs label-v2 r50 >= +0.003, CI > 0): {'PASS' if g_best else 'FAIL'}")
say(f"ATTRIBUTION (fused vs own global >= +0.002, CI > 0): {'PASS' if g_attr else 'FAIL'}")
say(f"  -> single LB submission: {'YES' if g_best and g_attr else 'NO'}")
say(f"ENSEMBLE (E3 swap label-v2 -> r50-GL >= E3 + 0.002, >= 90% bootstraps): {'PASS -> ensemble LB submission' if g_ens else 'FAIL'}")
if V3:
    say(f"V3 PRIMARY (fused >= label-v2 r50 + v2 local {per(X['lv2 + v2 local']).mean():.4f}, pre-registered 0.9468): "
        f"{'PASS' if tab['fused'].mean() >= .9468 else 'FAIL'} ({tab['fused'].mean():.4f})")
    say(f"V3 mechanism: learned gate vs fixed 0.5 {tab['fused'].mean() - tab['halfgate'].mean():+.4f}; "
        f"local vs v2 local (10 cropped) {tab['local'].iloc[CROPPED].mean() - tab['v2 local'].iloc[CROPPED].mean():+.4f}; "
        f"Baker's local {tab.loc[T[9], 'local']:.4f} (v2 {tab.loc[T[9], 'v2 local']:.4f})")

# secondary: Steven clear6 / 10-target, scripts/audit_phase1.py definitions (LB-fit check, no selection)
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
assert ref["targets_in_order"] == T
sid = [r["study_id"] for r in ref["labels"]]; assert set(sid) == set(ids)
ys = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
ws = np.array([r["weights"] for r in ref["labels"]]); ms = (ws > 0) & ((ys <= .3) | (ys >= .7)); ts = ys >= .7
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
clear = np.zeros((len(sid), 12), bool)
for i, s in enumerate(sid):
    if s in st.index:
        for k, t in enumerate(T):
            clear[i, k] = st.at[s, f"{key(t)}__status"] == "present" and st.at[s, f"{key(t)}__severity"] in (
                "moderate", "moderate_to_severe", "severe")
CB = [k for k in range(12) if (ms[:, k] & ts[:, k] & clear[:, k]).sum() >= 20]
order = [ids.index(s) for s in sid]
say("\nSecondary (Steven validation labels): clear6 / 10-target")
for k in ["lv2", "fused", "global", "E3", "E3 swap lv2->GL"]:
    S = X[k][order]
    c6 = np.mean([auc(S[sel, j], ts[sel, j]) for j in CB for sel in [np.where(ms[:, j] & (~ts[:, j] | clear[:, j]))[0]]])
    m10 = np.mean([auc(S[ms[:, j], j], ts[ms[:, j], j]) for j in TEN])
    say(f"  {k:16s} clear6 {c6:.4f}  10-target {m10:.4f}")
OUT.write_text("\n".join(lines) + "\n")
