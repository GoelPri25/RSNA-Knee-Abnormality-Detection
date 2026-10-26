"""Score V4-IMP (docs/v4_improve_spec.md v2, ledger V4-IMP) with the pre-registered gates.
Per lane: SWA fused fold-0 predictions vs v4 l4copy of the same seed, 1,000 paired study bootstraps.
  clear6     audit_clear_band_lb.py definitions (LB fit -0.687 + 1.671 x clear6)
  12-target  label-v2 confident cells (p <= .3 or >= .7)
  mild band  descriptive: label-v2 positives whose report severity is mild (vs all confident negatives)
  SIL split  descriptive: label-v2 confident AUC with negatives restricted to report-silent (SIL) vs report-negative cells
Usage (container): python -I audit_v4imp.py <project root> <lane root: imp_* dirs> <v4 root: v4_l4copy_s2026 / _s2027 dirs>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

R, X, V = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
g = {"__name__": "x"}; argv = sys.argv; sys.argv = [argv[0], str(R)]
exec((R / "repository/scripts/audit_clear_band_lb.py").read_text().split("run = lambda")[0], g); sys.argv = argv
ids, T, scores, auc, load, st = g["ids"], g["T"], g["scores"], g["auc"], g["load"], g["st"]
A = R / "repository/data/coatnet_d96_fold0/assets"
lv = json.load(open(A / "soft_labels_final_per_target_latD2.json")); assert lv["targets_in_order"] == T
lab = {r["study_id"]: np.array([.5 if p is None else p for p in r["probabilities"]]) for r in lv["labels"]}
Y = np.array([lab[s] for s in ids]); M = (Y <= .3) | (Y >= .7); TT = Y >= .7
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
mild = np.zeros_like(TT)
for i, s in enumerate(ids):
    if s in st.index:
        for k, t in enumerate(T):
            mild[i, k] = st.at[s, f"{key(t)}__status"] == "present" and st.at[s, f"{key(t)}__severity"] in ("mild", "mild_to_moderate")
rs = {x["StudyInstanceUID"]: [x[t] for t in T] for x in csv.DictReader((A / "rule_states.csv").open())}
STATE = np.array([rs.get(s, ["?"] * 12) for s in ids])
rng = np.random.default_rng(2026); BS = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)


def m12(S, idx=None):
    idx = np.arange(len(ids)) if idx is None else idx
    return np.mean([auc(S[idx[M[idx, k]], k], TT[idx[M[idx, k]], k]) for k in range(12)])


def band(S, neg):
    out = []
    for k in range(12):
        sel = M[:, k] & (TT[:, k] | neg[:, k])
        if TT[sel, k].sum() >= 10 and (~TT[sel, k]).sum() >= 10:
            out.append(auc(S[sel, k], TT[sel, k]))
    return np.mean(out)


def mildband(S):
    out = []
    for k in range(12):
        sel = M[:, k] & (~TT[:, k] | mild[:, k])
        if (TT[sel, k]).sum() >= 10:
            out.append(auc(S[sel, k], TT[sel, k]))
    return np.mean(out)


res, base = {}, {}
for seed in ("s2026", "s2027"):
    p = V / f"v4_l4copy_{seed}" / "val_swa_fused.json"
    if p.exists():
        b = load(p); base[seed] = b
        pr(f"## v4 l4copy {seed}: clear6 {scores(b)[0]:.4f} (fit {-0.687 + 1.671 * scores(b)[0]:.3f}) 12t {m12(b):.4f} "
           f"mild {mildband(b):.4f} | SIL-neg {band(b, STATE == 'SIL'):.4f} NEG-neg {band(b, STATE == 'NEG'):.4f}")
for d in sorted(X.glob("imp_*_s20*")):
    if "smoke" in d.name or not (d / "val_swa_fused.json").exists():
        continue
    lane, seed = d.name[4:].rsplit("_", 1); b = base[seed]; S = load(d / "val_swa_fused.json")
    c, c0 = scores(S)[0], scores(b)[0]; a, a0 = m12(S), m12(b)
    dc = np.array([scores(S, i)[0] - scores(b, i)[0] for i in BS]); da = np.array([m12(S, i) - m12(b, i) for i in BS])
    res[(lane, seed)] = (c - c0, (dc > 0).mean(), a - a0)
    pr(f"\n## {lane} {seed}: clear6 {c:.4f} d {c - c0:+.4f} [{np.percentile(dc, 2.5):+.4f}, {np.percentile(dc, 97.5):+.4f}] "
       f"P>0 {100 * (dc > 0).mean():.0f}% fit {-0.687 + 1.671 * c:.3f} | 12t {a:.4f} d {a - a0:+.4f} "
       f"[{np.percentile(da, 2.5):+.4f}, {np.percentile(da, 97.5):+.4f}] | mild {mildband(S):.4f} d {mildband(S) - mildband(b):+.4f}")
    pr(f"   SIL-neg AUC {band(S, STATE == 'SIL'):.4f} (v4 {band(b, STATE == 'SIL'):.4f}) | NEG-neg {band(S, STATE == 'NEG'):.4f} "
       f"(v4 {band(b, STATE == 'NEG'):.4f})")
    pr("   per-target 12t d: " + " ".join(f"{T[k][:8]} {auc(S[M[:, k], k], TT[M[:, k], k]) - auc(b[M[:, k], k], TT[M[:, k], k]):+.3f}" for k in range(12)))
    h = json.load(open(d / "history.json"))
    pr("   epochs (val BCE fused / lv2 fused / gap global): " + " ".join(
        f"{e['epoch']}:{e['lv2_bce_fused']:.3f}/{e['lv2_fused']:.4f}/{e['gap_global']:.3f}" for e in h if e["epoch"] % 4 == 1 or e["epoch"] == 25))

pr("\n## pre-registered decision")
mc = [res.get(("mcpair", s)) for s in ("s2026", "s2027")]
if None not in mc:
    ok = all(d >= 0.0015 and p >= 0.80 and a >= -0.0020 for d, p, a in mc) and np.mean([m[0] for m in mc]) >= 0.0025
    pr("mcpair: " + " | ".join(f"{s} clear6 {d:+.4f} ({100 * p:.0f}%) 12t {a:+.4f}" for s, (d, p, a) in zip(("s2026", "s2027"), mc))
       + f" mean {np.mean([m[0] for m in mc]):+.4f} -> {'PASS (submission pairing, then LB)' if ok else 'FAIL'}")
for lane in ("silw", "hill"):
    if (lane, "s2026") in res:
        d, p, a = res[(lane, "s2026")]
        pr(f"{lane}: clear6 {d:+.4f} 12t {a:+.4f} -> {'LB (floor met)' if d >= -0.0050 else 'STOP (below catastrophe floor)'}")
(X / "audit_v4imp.txt").write_text("\n".join(L) + "\n")
