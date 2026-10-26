"""KD-GAP, CPU part: why no student matched its teacher (read-only; no training, validation labels unchanged).

Facts to explain: Raptor teacher (nartaa CoAtNet-2; 384 alone = their 0.949 notebook, our 384+224 port LB 0.949) and the
full public blend (0.958) vs our r50 l4copy students: ra 0.943 (Raptor targets), B / E4 0.946 (full-blend family),
v4 l4copy (pre-KD parent) 0.939.
Caveat throughout: the teachers trained on all competition data, so on fold 0 they are IN-SAMPLE (their fold-0 outputs
carry memorised fold-0 labels a student cannot copy). The fastMRI holdout (500 patients, no labels; neither teacher nor
student trained on them) is the clean place to measure copying.

S1 copy fit     train KL (train BCE vs targets minus the targets' entropy, epochs 21-25) vs fold-0 KL to the same targets
S2 fidelity     Spearman student vs teacher per finding: fold 0 (in-sample teacher) and fastMRI holdout (clean), next to
                the teacher's own 384-vs-224 agreement (how close a second teacher model is)
S3 shrinkage    fastMRI holdout: slope of logit(student) on logit(teacher) per finding, |error| by teacher-confidence bin
S4 branches     global / local / fused: neutral LB-ruler metric and fidelity (does the crop branch help a KD student?)
S5 dynamics     neutral metric and fidelity by epoch (plateau or still rising?)
S6 per finding  Steven-confident AUC per finding: v4 -> students -> teacher (in-sample), share of the gap closed
S7 data         fidelity / predicted LB vs training-set size (ra: 3.5k studies; F1: + 9.2k fastMRI)
Usage: LBP_OUT=data/lb_predictor_v2 python3 -I scripts/audit_kd_gap.py  -> data/coatnet_d96_fold0/kd_gap_analysis/kd_gap_cpu.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_predictor as L                                   # noqa: E402

T, N, IDS, C = L.T, L.N, L.IDS, L.C
A = C / "kd_gap_analysis"; P = C / "pubkd"; S = P / "students_full"; SF = C / "surpass_f"
NEU = json.loads((L.OUT / "fit.json").read_text())["neutral"]; FIT = json.loads((L.OUT / "fit.json").read_text())["fits"][NEU]
lines = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)


sp1 = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())
spm = lambda X, Y: float(np.nanmean([sp1(X[:, j], Y[:, j]) for j in range(12)]))
H = lambda t: -(t * np.log(np.clip(t, 1e-7, 1)) + (1 - t) * np.log(np.clip(1 - t, 1e-7, 1)))
bce = lambda t, p: -(t * np.log(np.clip(p, 1e-7, 1)) + (1 - t) * np.log(np.clip(1 - p, 1e-7, 1)))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
met = lambda X: L.metric(NEU, X)
lb_of = lambda m: 0.946 + FIT["paired_b"] * (m - MB)        # paired, anchored on B's actual LB


def targets(name):
    d = json.loads((P / f"pubkd_targets_{name}.json").read_text()); assert d["targets_in_order"] == T
    return {r["study_id"]: np.array(r["probabilities"], float) for r in d["labels"]}


LANES = {  # lane: (target file, student dir, LB if submitted)
    "ra": ("raptor_pure", P / "students_r/pubkd_kd_ra", .943), "A": ("pure", S / "pubkd_kd_a", None),
    "B": ("mix73", S / "pubkd_kd_b", .946), "C": ("pure", S / "pubkd_kd_c", None), "D": ("mix73", S / "pubkd_kd_d", None),
    "E1": ("fullorder_rapvalues", S / "pubkd_kd_e1", None), "E4": ("fullorder_rapvalues_mix73", S / "pubkd_kd_e4", .946)}
TG = {k: targets(v[0]) for k, v in LANES.items()}
T0 = {k: np.array([TG[k][s] for s in IDS]) for k in LANES}
PB = L.public_rulers()
V4 = L.load_pred(L.F["v4"])


def branch(k, b):
    f = LANES[k][1] / f"val_swa_{b}.json"
    return L.load_pred(f) if f.exists() else None


MB = met(branch("B", "fused"))
pr(f"ruler {NEU}; LB(pred) = 0.946 + {FIT['paired_b']:.2f} x (metric - metric(B fused) {MB:.4f})")

# ------------------------------------------------------------------ S1
pr("\n== S1 copy fit: KL to the lane's own targets (nats per cell, global head)")
pr("lane  train KL ep21-25 | fold-0 KL (SWA) | ratio  (train KL from train BCE under augmentation, an upper bound)")
for k in LANES:
    hf = A / "hist" / f"pubkd_kd_{k.lower()}.json"
    if not hf.exists():
        continue
    h = json.loads(hf.read_text()); tr = np.mean([e["train_loss"]["global"] for e in h[-5:]])
    tr_ids = [s for s in TG[k] if s not in set(IDS)]
    Htr = float(np.mean([H(TG[k][s]).mean() for s in tr_ids]))
    g = branch(k, "global")
    vkl = float((bce(T0[k], g) - H(T0[k])).mean())
    pr(f"{k:4s}  {tr - Htr:.4f} (BCE {tr:.4f} - H {Htr:.4f}) | {vkl:.4f} | x{vkl / max(tr - Htr, 1e-6):.1f}")

# ------------------------------------------------------------------ S2 / S3 on the fastMRI holdout
pr("\n== S2 fidelity (mean per-finding Spearman to the teacher)")
fmt = lambda p: {r["study_id"]: np.array(r["probabilities"], float)
                 for r in json.loads(Path(p).read_text())["predictions"]}
TH = fmt(SF / "holdout/fm_holdout_teacher.json"); HID = sorted(TH)
tm = {}
for m in ("384", "224"):
    d = {}
    for f in sorted((A / "fm_teacher").glob(f"raptor{m}_*.json")):
        for s, v in json.loads(f.read_text())["pred"].items():
            d[s] = 0.5 * (np.array(v["plain"]) + np.array(v["mirror"]))
    tm[m] = np.array([d[s] for s in HID])
THX = np.array([TH[s] for s in HID])
FM = {"ra (raeval)": SF / "raeval/fm_holdout_global.json", "F1 +fastMRI": SF / "f1/fm_holdout_global.json",
      "F2 +noise": SF / "f2/fm_holdout_global.json", "F4pre fm-only": SF / "f4pre/fm_holdout_global.json",
      "F4fine": SF / "f4fine/fm_holdout_global.json"}
FMX = {k: np.array([fmt(v)[s] for s in HID]) for k, v in FM.items() if v.exists()}
pr(f"fastMRI holdout ({len(HID)} studies, clean): teacher 384 vs 224 {spm(tm['384'], tm['224']):.3f}; "
   f"384 vs blend {spm(tm['384'], THX):.3f}; 224 vs blend {spm(tm['224'], THX):.3f}")
pr(f"{'student':14s} " + " ".join(f"{t[:7]:>7s}" for t in T) + "   mean")
for k, X in FMX.items():
    pr(f"{k:14s} " + " ".join(f"{sp1(X[:, j], THX[:, j]):7.3f}" for j in range(12)) + f"  {spm(X, THX):.3f}")
pr(f"{'224 vs 384':14s} " + " ".join(f"{sp1(tm['224'][:, j], tm['384'][:, j]):7.3f}" for j in range(12))
   + f"  {spm(tm['224'], tm['384']):.3f}")
pr(f"\nfold 0 (teacher in-sample): Raptor 384 vs 224 {spm(PB['r384'], PB['r224']):.3f}")
for k in LANES:
    g, f = branch(k, "global"), branch(k, "fused")
    pr(f"  {k:4s} vs own targets: global {spm(g, T0[k]) if g is not None else float('nan'):.3f}  fused {spm(f, T0[k]):.3f}"
       f" | fused vs Raptor 384 {spm(f, PB['r384']):.3f}")
pr(f"  v4 (no KD) vs Raptor 384 {spm(V4, PB['r384']):.3f}")

pr("\n== S3 shrinkage on the fastMRI holdout: slope of logit(student) on logit(teacher) (1 = same confidence)")
pr(f"{'student':14s} " + " ".join(f"{t[:7]:>7s}" for t in T))
for k, X in list(FMX.items()) + [("224 vs blend", tm["224"]), ("384 vs blend", tm["384"])]:
    pr(f"{k:14s} " + " ".join(f"{np.polyfit(logit(THX[:, j]), logit(X[:, j]), 1)[0]:7.2f}" for j in range(12)))
pr("mean |student - teacher| by teacher-probability bin (all findings pooled)")
bins = [0, .1, .3, .5, .7, .9, 1.01]
pr(f"{'student':14s} " + " ".join(f"{a:.1f}-{b:.1f}".rjust(9) for a, b in zip(bins[:-1], bins[1:])))
cnt = [((THX >= a) & (THX < b)).sum() for a, b in zip(bins[:-1], bins[1:])]
pr(f"{'(cells)':14s} " + " ".join(f"{c:9d}" for c in cnt))
for k, X in list(FMX.items()) + [("224 vs blend", tm["224"])]:
    pr(f"{k:14s} " + " ".join(f"{np.abs(X - THX)[(THX >= a) & (THX < b)].mean():9.3f}" for a, b in zip(bins[:-1], bins[1:])))

# ------------------------------------------------------------------ S4
pr("\n== S4 branch ablation (fold 0): neutral metric -> predicted LB, and Spearman to own targets")
for k in LANES:
    row = []
    for b in ("global", "local", "fused"):
        X = branch(k, b)
        if X is None:
            row.append(f"{b} -"); continue
        m = met(X); row.append(f"{b} {m:.4f} ({lb_of(m):.4f}, rho {spm(X, T0[k]):.3f})")
    pr(f"{k:4s} " + "  ".join(row) + (f"  | LB {LANES[k][2]}" if LANES[k][2] else ""))

# ------------------------------------------------------------------ S5
pr("\n== S5 dynamics: neutral metric / Spearman to own targets by epoch (fused)")
for k in ("A", "B", "E4"):
    eps = sorted(LANES[k][1].glob("val_epoch*_fused.json"))
    pr(f"{k}: " + "  ".join(f"ep{int(p.stem[9:11])} {met(L.load_pred(p)):.4f}/{spm(L.load_pred(p), T0[k]):.3f}" for p in eps)
       + f"  SWA {met(branch(k, 'fused')):.4f}/{spm(branch(k, 'fused'), T0[k]):.3f}")
eps = sorted((S / "ra_epochs").glob("val_epoch*_fused.json"))
pr("ra: " + "  ".join(f"ep{int(p.stem[9:11])} {met(L.load_pred(p)):.4f}/{spm(L.load_pred(p), T0['ra']):.3f}" for p in eps))
for k in ("pubkd_kd_e4", "pubkd_kd_ra"):
    h = json.loads((A / "hist" / f"{k}.json").read_text())
    pr(f"{k} history: " + " ".join(f"ep{e['epoch']} tr {e['train_loss']['global']:.4f} lv2 {e.get('lv2_fused', float('nan')):.4f}"
                                     for e in h if e["epoch"] in (5, 10, 15, 20, 25)))

# ------------------------------------------------------------------ S6
pr("\n== S6 per finding, Steven-v4 confident cells (fold 0; teacher in-sample = upper reference only)")
t0, m0 = L.RL["steven"]
cols = {"v4": V4, "ra": branch("ra", "fused"), "B": branch("B", "fused"), "E4": branch("E4", "fused"),
        "D(in-s)": branch("D", "fused"), "R384(in-s)": PB["r384"]}
pr(f"{'finding':17s} {'pos':>4s} " + " ".join(f"{k:>10s}" for k in cols) + "  E4 share of v4->R384 gap")
for j, t in enumerate(T):
    sel = m0[:, j]
    if (sel & t0[:, j]).sum() < 5:
        continue
    a = {k: L.auc(X[sel, j], t0[sel, j]) for k, X in cols.items()}
    gap = a["R384(in-s)"] - a["v4"]
    pr(f"{t:17s} {int((sel & t0[:, j]).sum()):4d} " + " ".join(f"{a[k]:10.4f}" for k in cols)
       + (f"  {(a['E4'] - a['v4']) / gap:+.2f}" if abs(gap) > .003 else "  (gap < .003)"))

# ------------------------------------------------------------------ S7
pr("\n== S7 more data: ra recipe on 3.5k competition studies vs + 9.2k fastMRI (F1)")
f1 = L.load_pred(SF / "f1/val_swa_fused.json"); ra = branch("ra", "fused")
pr(f"fastMRI-holdout fidelity ra {spm(FMX['ra (raeval)'], THX):.3f} -> F1 {spm(FMX['F1 +fastMRI'], THX):.3f}; "
   f"fold-0 metric ra {met(ra):.4f} ({lb_of(met(ra)):.4f}, LB 0.943) -> F1 {met(f1):.4f} ({lb_of(met(f1)):.4f})")
(A / "kd_gap_cpu.txt").write_text("\n".join(lines) + "\n"); print(f"\nwrote {A / 'kd_gap_cpu.txt'}")
