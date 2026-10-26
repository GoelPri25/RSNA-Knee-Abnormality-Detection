"""Exp r50-GL4 (v4) scorer (ledger row r50-GL4, pre-registered 2026-10-18). Fold 0, label-v2 confident cells (weight > 0,
y <= 0.3 / >= 0.7), 12-target mean AUC, 2,000 study bootstraps (each resample scores every model on the same studies).
  PRIMARY:    mean over v4-shared seeds of per-seed fused SWA AUC >= 0.9425 + 0.003 (point), and >= 90% of resamples
              with that seed mean > 0.9425 (fixed bar, exp_v4_review_round2_methods.md F5). Per-seed shares descriptive.
  PROTECTION: mean v4-shared global SWA AUC >= 0.9405.
  EARLY STOP: (post hoc, shared s2026, epochs 10-12) read from history.json, reported only.
  SECONDARY:  E3 with v4 shared s2026 fused replacing label-v2 r50 >= E3 + 0.002 at >= 90% of resamples;
              adapter / l4copy vs shared s2026 (one seed: only a gap > the label-v2 seed spread counts);
              v4-shared seeds vs our label-v2 r50 seeds (same node, same cache), per-epoch and SWA; Steven clear6 / 10-target.
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_r50_gl4.py <project root> [run dir, default run_r50_gl4_mlx]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
RUN = C / (sys.argv[2] if len(sys.argv) > 2 else "run_r50_gl4_mlx")
OUT = RUN / "audit_r50_gl4.txt"; lines = []
say = lambda s="": (print(s, flush=True), lines.append(s))
lv = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text()); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
CROPPED = [k for k, t in enumerate(T) if t not in ("Effusion", "Synovitis")]
BAR, PARENT = .9455, .9425


def load(p):
    d = json.loads(p.read_text()); assert d["targets_in_order"] == T
    return {x["study_id"]: np.array(x["probabilities"]) for x in d["predictions"]}


TM = D / "teammate_pp4/fold0_predictions"
P = {"lv2": load(C / "kaggle_fold0_labelv2/val_swa.json"), "4net": load(TM / "four_network_fold0_val_swa.json"),
     "1b": load(C / "run_continue/val_swa.json"), "cnx": load(C / "run_cnx_pertarget/val_swa.json")}
SHARED, ARMS, LV2 = [], [], []
for d in sorted(RUN.glob("v4_*")):
    if (d / "val_swa_fused.json").is_file():
        for h in ("fused", "global", "local"):
            P[f"{d.name}:{h}"] = load(d / f"val_swa_{h}.json")
        (SHARED if d.name.startswith("v4_shared") else ARMS).append(d.name)
for d in sorted(RUN.glob("lv2_*")):
    if (d / "val_swa.json").is_file():
        P[d.name] = load(d / "val_swa.json"); LV2.append(d.name)
assert SHARED, "no finished v4-shared run"
ids = sorted(P["lv2"]); assert all(set(p) == set(ids) for p in P.values())
X = {k: np.array([v[s] for s in ids]) for k, v in P.items()}
Y = np.array([lab[s][0] for s in ids]); W = np.array([lab[s][1] for s in ids])
M = (W > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7
E3M = ["lv2", "4net", "1b", "cnx"]
X["E3"] = np.mean([X[m] for m in E3M], 0)
CAND = "v4_shared_s2026" if "v4_shared_s2026" in SHARED else SHARED[0]
X["E3 swap"] = np.mean([X[m] for m in E3M[1:]] + [X[f"{CAND}:fused"]], 0)


def auc(s, t):
    rk = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (rk[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(x, idx=None):
    idx = np.arange(len(ids)) if idx is None else idx
    return np.array([auc(x[idx, k][M[idx, k]], TT[idx, k][M[idx, k]]) for k in range(12)])


tab = pd.DataFrame({k: per(v) for k, v in X.items()}, index=T)
say(f"r50-GL4 (v4): fold 0, label-v2 confident cells ({len(ids)} studies); finished v4: {SHARED + ARMS}; label-v2 seeds: {LV2}")
say("\nSWA 12-target mean AUC (local: 12-mean with g on effusion / synovitis is not defined -> 10 cropped findings)")
for d in SHARED + ARMS:
    say(f"  {d:18s} fused {tab[d + ':fused'].mean():.4f}  global {tab[d + ':global'].mean():.4f}  "
        f"local10 {tab[d + ':local'].iloc[CROPPED].mean():.4f}  fused-global {tab[d + ':fused'].mean() - tab[d + ':global'].mean():+.4f}")
for k in ["lv2"] + LV2 + ["E3", "E3 swap"]:
    say(f"  {k:18s} {tab[k].mean():.4f}")
if LV2:
    a = [tab[k].mean() for k in LV2]
    say(f"  our label-v2 r50 seeds: mean {np.mean(a):.4f}, range {min(a):.4f}-{max(a):.4f} (spread {max(a) - min(a):.4f}); "
        f"Kaggle parent 0.9425")
sh_f = np.mean([tab[d + ":fused"].mean() for d in SHARED]); sh_g = np.mean([tab[d + ":global"].mean() for d in SHARED])
say("\nPer finding: shared-seed mean fused / global, parent label-v2 r50")
pf = pd.DataFrame({"fused": np.mean([tab[d + ":fused"] for d in SHARED], 0), "global": np.mean([tab[d + ":global"] for d in SHARED], 0),
                   "lv2": tab["lv2"]}, index=T)
pf["fused-lv2"] = pf["fused"] - pf["lv2"]; pf["fused-global"] = pf["fused"] - pf["global"]
say(pf.round(4).to_string())

rng = np.random.default_rng(2026); B = 2000
need = ["lv2", "E3", "E3 swap"] + [f"{d}:{h}" for d in SHARED + ARMS for h in ("fused", "global")] + LV2
bt = {k: [] for k in need}
for _ in range(B):
    i = rng.integers(0, len(ids), len(ids))
    for k in need:
        bt[k].append(np.nanmean(per(X[k], i)))
bt = {k: np.array(v) for k, v in bt.items()}
seedmean = np.mean([bt[d + ":fused"] for d in SHARED], 0)
share = np.mean(seedmean > PARENT)
say("\nPRIMARY")
say(f"  v4-shared seed-mean fused {sh_f:.4f} ({len(SHARED)} seeds) - 0.9425 = {sh_f - PARENT:+.4f} (need >= +0.003); "
    f"resamples with seed mean > 0.9425: {share:.1%} (need >= 90%)")
for d in SHARED:
    say(f"    {d}: fused {tab[d + ':fused'].mean():.4f}, resamples > 0.9425 {np.mean(bt[d + ':fused'] > PARENT):.1%}, "
        f"paired vs parent {np.mean(bt[d + ':fused'] - bt['lv2']):+.4f} [{np.percentile(bt[d + ':fused'] - bt['lv2'], 2.5):+.4f}, "
        f"{np.percentile(bt[d + ':fused'] - bt['lv2'], 97.5):+.4f}]")
g_primary = sh_f - PARENT >= .003 and share >= .9
g_prot = sh_g >= .9405
say(f"  PRIMARY: {'PASS' if g_primary else 'FAIL'}" + ("" if len(SHARED) == 3 else f"  (PROVISIONAL: {len(SHARED)} of 3 seeds)"))
say(f"  PROTECTION (shared-seed mean global {sh_g:.4f} >= 0.9405): {'PASS' if g_prot else 'FAIL'}")
dE = bt["E3 swap"] - bt["E3"]
g_ens = tab["E3 swap"].mean() - tab["E3"].mean() >= .002 and np.mean(dE > 0) >= .9
say(f"\nSECONDARY E3 swap ({CAND} fused for label-v2 r50): {tab['E3 swap'].mean() - tab['E3'].mean():+.4f}, "
    f"95% CI [{np.percentile(dE, 2.5):+.4f}, {np.percentile(dE, 97.5):+.4f}], {np.mean(dE > 0):.0%} > 0 -> {'PASS' if g_ens else 'FAIL'}")
if LV2:
    sp = max(tab[k].mean() for k in LV2) - min(tab[k].mean() for k in LV2)
    lvm = np.mean([bt[k] for k in LV2], 0)
    say(f"v4-shared seed mean vs our label-v2 seed mean (like-for-like): {sh_f - np.mean([tab[k].mean() for k in LV2]):+.4f}, "
        f"95% CI [{np.percentile(seedmean - lvm, 2.5):+.4f}, {np.percentile(seedmean - lvm, 97.5):+.4f}]")
    for d in ARMS:
        g = tab[d + ":fused"].mean() - tab[CAND + ":fused"].mean()
        say(f"  arm {d} vs {CAND}: fused {g:+.4f} ({'beyond' if abs(g) > sp else 'within'} label-v2 seed spread {sp:.4f})")

# per-epoch: v4 heads vs our label-v2 seeds (same node, same cache, scored every epoch)
say("\nPer-epoch 12-target mean AUC (label-v2 confident cells)")
cols = {}
for d in sorted(RUN.glob("v4_*")) + sorted(RUN.glob("lv2_*")):
    for h in (("fused", "global") if d.name.startswith("v4") else ("",)):
        es = {}
        for f in sorted(d.glob(f"val_epoch*{'_' + h if h else ''}.json")):
            e = int(f.stem[9:11]); p = load(f); es[e] = per(np.array([p[s] for s in ids])).mean()
        if es:
            cols[f"{d.name}{':' + h if h else ''}"] = es
if cols:
    ep = pd.DataFrame(cols).sort_index()
    say(ep.loc[[e for e in ep.index if e % 5 == 0 or e >= 21]].round(4).to_string())

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
for k in ["lv2"] + LV2 + [f"{d}:{h}" for d in SHARED + ARMS for h in ("fused", "global")] + ["E3", "E3 swap"]:
    S = X[k][order]
    c6 = np.mean([auc(S[sel, j], ts[sel, j]) for j in CB for sel in [np.where(ms[:, j] & (~ts[:, j] | clear[:, j]))[0]]])
    m10 = np.mean([auc(S[ms[:, j], j], ts[ms[:, j], j]) for j in TEN])
    say(f"  {k:24s} clear6 {c6:.4f}  10-target {m10:.4f}")
OUT.write_text("\n".join(lines) + "\n")
