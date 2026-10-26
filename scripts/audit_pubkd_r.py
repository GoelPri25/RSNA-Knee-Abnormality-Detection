"""Score PUB-KD-R (ledger row): v4 l4copy r50 students of the Raptor-only public teacher vs v4 l4copy of the same seed.
SWA fused fold-0 predictions, label-v2, clear6 / 12t as audit_bbf.py; fidelity = mean per-finding Spearman with the
teacher target on fold 0 (teacher in-sample there: descriptive).
Usage: python3 -I scripts/audit_pubkd_r.py <project root> <dir with pubkd_kd_r*> <run_r50_gl4_mlx dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, X, V = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
TGT = R / "repository/data/coatnet_d96_fold0/pubkd/pubkd_targets_raptor_pure.json"
g = {"__name__": "x"}; argv = sys.argv; sys.argv = [argv[0], str(R)]
exec((R / "repository/scripts/audit_clear_band_lb.py").read_text().split("run = lambda")[0], g); sys.argv = argv
ids, T, scores, auc, load = g["ids"], g["T"], g["scores"], g["auc"], g["load"]
A = R / "repository/data/coatnet_d96_fold0/assets"
lv = json.load(open(A / "soft_labels_final_per_target_latD2.json")); assert lv["targets_in_order"] == T
lab = {r["study_id"]: np.array([.5 if p is None else p for p in r["probabilities"]]) for r in lv["labels"]}
Y = np.array([lab[s] for s in ids]); M = (Y <= .3) | (Y >= .7); TT = Y >= .7
rng = np.random.default_rng(2026); BS = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
rk = lambda S: pd.DataFrame(S).rank(pct=True).to_numpy()
fit = lambda c: -0.687 + 1.671 * c
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


def m12(S, idx=None):
    idx = np.arange(len(ids)) if idx is None else idx
    return np.mean([auc(S[idx[M[idx, k]], k], TT[idx[M[idx, k]], k]) for k in range(12)])


def cmp(S, b, tag):
    c, c0 = scores(S)[0], scores(b)[0]; a, a0 = m12(S), m12(b)
    dc = np.array([scores(S, i)[0] - scores(b, i)[0] for i in BS]); da = np.array([m12(S, i) - m12(b, i) for i in BS])
    pr(f"  {tag}: clear6 {c:.4f} d {c - c0:+.4f} [{np.percentile(dc, 2.5):+.4f}, {np.percentile(dc, 97.5):+.4f}] P>0 {100 * (dc > 0).mean():.0f}% "
       f"fit {fit(c):.3f} | 12t {a:.4f} d {a - a0:+.4f} [{np.percentile(da, 2.5):+.4f}, {np.percentile(da, 97.5):+.4f}]")
    return c - c0, (dc > 0).mean(), a - a0


def per_target(S, b):
    return " ".join(f"{T[k][:8]} {auc(S[M[:, k], k], TT[M[:, k], k]) - auc(b[M[:, k], k], TT[M[:, k], k]):+.3f}" for k in range(12))


def curve(d):
    h = json.load(open(d / "history.json"))
    return " ".join(f"{e['epoch']}:{e['lv2_bce_fused']:.3f}/{e['lv2_fused']:.4f}/{e['gap_global']:.3f}" for e in h if e["epoch"] % 4 == 1 or e["epoch"] == 25)




def spear(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def fidelity(S, Tt):
    """mean over findings of the fold-0 Spearman between a model and the teacher."""
    return np.mean([spear(S[:, k], Tt[:, k]) for k in range(12)])


tg = {r["study_id"]: np.array(r["probabilities"]) for r in json.load(open(TGT))["labels"]}
Tt = np.array([tg[s] for s in ids])
base = {s: load(V / f"v4_l4copy_{s}" / "val_swa_fused.json") for s in ("s2026", "s2027")}
pr("## references (fold 0, label-v2)")
for s, b in base.items():
    pr(f"  v4 l4copy {s}: clear6 {scores(b)[0]:.4f} (fit {fit(scores(b)[0]):.3f}) 12t {m12(b):.4f} | fidelity to teacher {fidelity(b, Tt):.3f}")
pr(f"  Raptor teacher (in-sample on fold 0, descriptive): clear6 {scores(Tt)[0]:.4f} 12t {m12(Tt):.4f}")
v4pair = (rk(base["s2026"]) + rk(base["s2027"])) / 2
pr(f"  v4 2-seed rank-mean: clear6 {scores(v4pair)[0]:.4f} 12t {m12(v4pair):.4f}")

LANES = {"ra": ("pure", "s2026"), "rb": ("mix73", "s2026"), "rc": ("pure", "s2027"), "rd": ("mix73", "s2027")}
res, S_ = {}, {}
pr("\n## students vs v4 l4copy of the same seed (SWA fused, 1,000 paired bootstraps)")
for lane, (rec, seed) in LANES.items():
    d = X / f"pubkd_kd_{lane}"
    if not (d / "val_swa_fused.json").exists():
        pr(f"{lane} ({rec} {seed}): not finished"); continue
    S = load(d / "val_swa_fused.json"); S_[lane] = S
    pr(f"{lane} ({rec} {seed}): fidelity to teacher {fidelity(S, Tt):.3f}")
    res[lane] = cmp(S, base[seed], "vs v4")
    pr("  per-target 12t d: " + per_target(S, base[seed]))
    pr("  epochs (val BCE fused / lv2 fused / gap global): " + curve(d))
pr("\n## pre-registered readout (per recipe: both seeds clear6 d >= +0.0015 and mean >= +0.0020 -> LB candidate)")
for rec, lanes in (("pure", ("ra", "rc")), ("mix73", ("rb", "rd"))):
    got = [res[l] for l in lanes if l in res]
    if len(got) < 2:
        pr(f"{rec}: {len(got)}/2 seeds finished"); continue
    m = np.mean([g[0] for g in got]); ok = all(g[0] >= 0.0015 for g in got) and m >= 0.0020
    pr(f"{rec}: clear6 d {got[0][0]:+.4f} / {got[1][0]:+.4f} (mean {m:+.4f}; P>0 {100 * got[0][1]:.0f}% / {100 * got[1][1]:.0f}%), "
       f"12t d {got[0][2]:+.4f} / {got[1][2]:+.4f} -> {'LB CANDIDATE (ask before submitting)' if ok else 'not an LB candidate'}")
    pair = (rk(S_[lanes[0]]) + rk(S_[lanes[1]])) / 2
    cmp(pair, v4pair, f"{rec} 2-seed rank-mean vs v4 2-seed rank-mean (descriptive)")
(X / "audit_pubkd_r.txt").write_text("\n".join(L) + "\n")
