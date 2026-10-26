"""Score L4C-NF (ledger row L4C-NF): OA crops partly on the sagittal non-fluid series, inference only.
Per model: self vs nf1c / nf1o / nf2c, all with PF OA + fracture global-only (submission A's patch).
  clear band   audit_clear_band_lb.py definitions (6 targets, moderate+ positives), LB fit -0.687 + 1.671 x clear6
  12-target    label-v2 confident cells (w > 0, p <= .3 or >= .7)
1,000 paired study bootstraps. Pre-registered adoption: best variant on the clear band with, on s2026 AND s2027, clear6 delta
>= +0.0010 (>= 80% > 0) and 12-target delta >= -0.0005.
Also scores L4C-LOC (scripts/l4c_loc_runner.py: wp / wm / w / qm) with its pre-registered decision.
Usage (container): python -I audit_l4c_nf.py <project root> <output dir> [more dirs: per-seed L4C-LOC lanes]
"""
import json
import sys
from pathlib import Path

import numpy as np

R, XS = Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]]; X = XS[0]
g = {"__name__": "x"}; argv = sys.argv; sys.argv = [argv[0], str(R)]
exec((R / "repository/scripts/audit_clear_band_lb.py").read_text().split("run = lambda")[0], g); sys.argv = argv
ids, T, scores, auc, CB = g["ids"], g["T"], g["scores"], g["auc"], g["CB"]
lv = json.load(open(R / "repository/data/coatnet_d96_fold0/assets/soft_labels_final_per_target_latD2.json"))
assert lv["targets_in_order"] == T
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
Y = np.array([lab[s][0] for s in ids]); W = np.array([lab[s][1] for s in ids]); M = (W > 0) & ((Y <= .3) | (Y >= .7)); TT = Y >= .7
CF = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]; GO = [T.index("PF OA"), T.index("Fracture")]; OA = [T.index(t) for t in ("Medial OA", "Lateral OA", "PF OA")]
sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(2026); BS = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)


def m12(S, idx=None):
    idx = np.arange(len(ids)) if idx is None else idx
    return np.mean([auc(S[idx[M[idx, k]], k], TT[idx[M[idx, k]], k]) for k in range(12)])


def per_oa(S):
    out = []
    for k in OA:
        sel = g["mask"][:, k] & (~g["truth"][:, k] | g["clear"][:, k])
        out.append(auc(S[sel, k], g["truth"][sel, k]) if k in CB else np.nan)
    return out


res = {}
for tag in ("s2026", "s2027", "r1self"):
    p = next((d / f"xai_arrays_{tag}.npz" for d in XS if (d / f"xai_arrays_{tag}.npz").exists()), None)
    if p is None:
        continue
    z = np.load(p); zi = list(z["ids"]); ix = [zi.index(s) for s in ids]
    gl = z["glob"][ix]

    def fused(key):
        f = gl.copy(); f[:, CF] = 0.5 * (gl[:, CF] + z[key][ix][:, CF]); f[:, GO] = gl[:, GO]; return sig(f)
    base = fused("loc_self_100"); c0, _ = scores(base); a0 = m12(base)
    pr(f"\n## {tag}: self+patch clear6 {c0:.4f} (fit {-0.687 + 1.671 * c0:.3f}) 12-target {a0:.4f}  OA clear "
       + " ".join(f"{T[k][:7]} {v:.3f}" for k, v in zip(OA, per_oa(base))))
    nt = z["nf_tab"][ix] if "nf_tab" in z.files else None
    if nt is not None:
        moved = [(np.isin(nt[:, v][:, j], np.arange(27, 47))).any(1).mean() for v in range(2) for j in (4, 5, 6)]
        pr("studies with >= 1 SagNF OA crop (nf1c medOA latOA PF | nf2c ...): " + " ".join(f"{100 * x:.0f}%" for x in moved))
    for v in ("nf1c", "nf1o", "nf2c", "selfu", "qm", "qs", "wp", "wm", "w", "ms1"):
        key = f"loc_{v}_100"
        if key not in z.files:
            continue
        S = fused(key); c, _ = scores(S); a = m12(S)
        dc = np.array([scores(S, b)[0] - scores(base, b)[0] for b in BS]); da = np.array([m12(S, b) - m12(base, b) for b in BS])
        res[(tag, v)] = (c - c0, (dc > 0).mean(), a - a0)
        pr(f"{v:5s} clear6 {c:.4f} d {c - c0:+.4f} [{np.percentile(dc, 2.5):+.4f}, {np.percentile(dc, 97.5):+.4f}] P>0 {100 * (dc > 0).mean():.0f}%"
           f" fit {-0.687 + 1.671 * c:.3f} | 12-target {a:.4f} d {a - a0:+.4f} [{np.percentile(da, 2.5):+.4f}, {np.percentile(da, 97.5):+.4f}]"
           f" | OA clear " + " ".join(f"{T[k][:7]} {x:.3f}" for k, x in zip(OA, per_oa(S))))

pr("\n## pre-registered decision (s2026 and s2027; ms1 / selfu are references, not candidates)")
pr("L4C-NF: nf1c / nf1o / nf2c, bar clear6 >= +0.0010. L4C-LOC zero-training Q: qm primary (bar +0.0010), qs +0.0015 (best-of-n margin)")
ok = []
BAR = {"nf1c": 0.0010, "nf1o": 0.0010, "nf2c": 0.0010, "qm": 0.0010, "qs": 0.0015}
for v in ("nf1c", "nf1o", "nf2c", "qm", "qs"):
    rows = [res.get((t, v)) for t in ("s2026", "s2027")]
    if None in rows:
        continue
    passed = all(d >= BAR[v] and p >= 0.80 and a >= -0.0005 for d, p, a in rows)
    pr(f"{v}: " + " | ".join(f"{t} clear6 {d:+.4f} ({100 * p:.0f}%) 12t {a:+.4f}" for t, (d, p, a) in zip(("s2026", "s2027"), rows))
       + f" -> {'PASS' if passed else 'fail'}")
    if passed:
        ok.append((np.mean([r[0] for r in rows]), v))
pr("ADOPT " + (max(ok)[1] if ok else "none"))
if all((t, "wp") in res for t in ("s2026", "s2027")):
    pr("\n## L4C-LOC pre-registered decision (wp primary +0.0010; wm / w +0.0015; qm replaces wp if within 0.0005 clear6)")
    pas = {}
    for v, bar in (("wp", 0.0010), ("wm", 0.0015), ("w", 0.0015), ("qm", 0.0010)):
        rows = [res.get((t, v)) for t in ("s2026", "s2027")]
        if None in rows:
            continue
        pas[v] = all(d >= bar and p_ >= 0.80 and a >= -0.0005 for d, p_, a in rows)
        pr(f"{v}: " + " | ".join(f"{t} clear6 {d:+.4f} ({100 * p_:.0f}%) 12t {a:+.4f}" for t, (d, p_, a) in zip(("s2026", "s2027"), rows))
           + f" -> {'PASS' if pas[v] else 'fail'}")
    if pas.get("wp"):
        gap = np.mean([res[(t, "wp")][0] - res[(t, "qm")][0] for t in ("s2026", "s2027")]) if ("s2026", "qm") in res else 1.0
        pr("L4C-LOC: PASS -> " + ("qm (zero training, within 0.0005 of wp)" if gap <= 0.0005 and pas.get("qm") else "wp"))
    else:
        pr("L4C-LOC: wp FAIL" + (" (wm / w passed their higher bar: " + ",".join(v for v in ("wm", "w") if pas.get(v)) + ")" if any(pas.get(v) for v in ("wm", "w")) else ""))
(X / "audit_l4c_nf.txt").write_text("\n".join(L) + "\n")
