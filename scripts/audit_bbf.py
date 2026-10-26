"""Score BB-CMP and V4-FUSE (ledger rows) with the pre-registered readouts. SWA fused fold-0 predictions, 1,000 paired
study bootstraps; clear6 / 12-target as audit_v4imp.py (audit_clear_band_lb.py definitions, LB fit -0.687 + 1.671 x clear6).
  v4 seed spread: v4 l4copy s2026 / s2027 (run_r50_gl4_mlx) and s2028 (this round).
  BB-CMP (descriptive, vs v4 s2026): X >= +0.0015 clear6 -> LB single candidate; within +-0.003 -> ensemble-partner
    candidate if rank-mean(v4 s2026, X) - v4 s2026 >= +0.002 and > rank-mean(v4 s2026, v4 s2027) - v4 s2026.
  V4-FUSE (vs v4 same seed): PASS if both seeds clear6 >= +0.0015, mean >= +0.0025, each P>0 >= 80%, 12t >= -0.0020;
    mechanism: fuse_ratio over epochs 21-25 (logged mean; outlier-sensitive), with-partner - no-partner.
Usage: python3 -I audit_bbf.py <project root> <bbf lane root: bbf_* dirs> <v4 root: v4_l4copy_s2026 / _s2027 dirs>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, X, V = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
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


base = {}
for s in ("s2026", "s2027"):
    base[s] = load(V / f"v4_l4copy_{s}" / "val_swa_fused.json")
if (X / "bbf_v4_s2028" / "val_swa_fused.json").exists():
    base["s2028"] = load(X / "bbf_v4_s2028" / "val_swa_fused.json")
pr("## v4 l4copy seed spread")
for s, b in base.items():
    pr(f"  v4 {s}: clear6 {scores(b)[0]:.4f} (fit {fit(scores(b)[0]):.3f}) 12t {m12(b):.4f}")
cs = [scores(b)[0] for b in base.values()]; ts = [m12(b) for b in base.values()]
pr(f"  spread: clear6 range {max(cs) - min(cs):.4f} sd {np.std(cs, ddof=1) if len(cs) > 1 else float('nan'):.4f} | 12t range {max(ts) - min(ts):.4f}")
if "s2028" in base:
    cmp(base["s2028"], base["s2026"], "v4 s2028 vs v4 s2026 (seed-only difference)")
b26 = base["s2026"]
seedpair = (rk(b26) + rk(base["s2027"])) / 2
sp_c = np.array([scores(seedpair, i)[0] - scores(b26, i)[0] for i in BS])
pr(f"  seed-ensemble control rank-mean(v4 s2026, s2027) - v4 s2026: clear6 {scores(seedpair)[0] - scores(b26)[0]:+.4f}, 12t {m12(seedpair) - m12(b26):+.4f}")

pr("\n## BB-CMP (single seed s2026, vs v4 l4copy s2026)")
dec = []
for bb in ("cnx", "coat0", "effv2s", "dino2s", "dino3s"):
    d = X / f"bbf_{bb}_s2026"
    if not (d / "val_swa_fused.json").exists():
        pr(f"{bb}: not finished"); continue
    S = load(d / "val_swa_fused.json"); pr(f"{bb}:")
    dc, p, da = cmp(S, b26, "single")
    pair = (rk(b26) + rk(S)) / 2
    pc = np.array([scores(pair, i)[0] - scores(b26, i)[0] for i in BS])
    ec = scores(pair)[0] - scores(b26)[0]; vs_seed = np.array([scores(pair, i)[0] - scores(seedpair, i)[0] for i in BS])
    pr(f"  pair rank-mean(v4 s2026, {bb}) - v4 s2026: clear6 {ec:+.4f} [{np.percentile(pc, 2.5):+.4f}, {np.percentile(pc, 97.5):+.4f}] 12t {m12(pair) - m12(b26):+.4f} | "
       f"vs seed-ensemble control {scores(pair)[0] - scores(seedpair)[0]:+.4f} [{np.percentile(vs_seed, 2.5):+.4f}, {np.percentile(vs_seed, 97.5):+.4f}] P>0 {100 * (vs_seed > 0).mean():.0f}%")
    pr("  per-target 12t d: " + per_target(S, b26))
    pr("  epochs (val BCE fused / lv2 fused / gap global): " + curve(d))
    if dc >= 0.0015:
        v = "LB single candidate (port + ask)"
    elif abs(dc) <= 0.003 and ec >= 0.002 and scores(pair)[0] > scores(seedpair)[0]:
        v = "ensemble-partner candidate"
    else:
        v = "no (below single bar; ensemble rule not met)"
    dec.append(f"{bb}: single {dc:+.4f}, pair {ec:+.4f} (seed control {scores(seedpair)[0] - scores(b26)[0]:+.4f}) -> {v}")

pr("\n## V4-FUSE (vs v4 l4copy same seed)")
fz = {}
for s in ("s2026", "s2027"):
    d = X / f"bbf_fuse_{s}"
    if not (d / "val_swa_fused.json").exists():
        pr(f"fuse {s}: not finished"); continue
    S = load(d / "val_swa_fused.json"); pr(f"fuse {s}:")
    fz[s] = cmp(S, base[s], "with partner")
    if (d / "val_swa_fused_nopartner.json").exists():
        N = load(d / "val_swa_fused_nopartner.json")
        cmp(S, N, "with partner vs no partner (same model)")
    h = json.load(open(d / "history.json"))
    pr("  fuse_ratio (global, crop) epochs 21-25: " + " ".join(f"{e['epoch']}:{e.get('fuse_ratio')}" for e in h if e["epoch"] >= 21))
    pr("  per-target 12t d: " + per_target(S, base[s]))
    pr("  epochs (val BCE fused / lv2 fused / gap global): " + curve(d))
if len(fz) == 2:
    ok = all(dc >= 0.0015 and p >= 0.80 and da >= -0.0020 for dc, p, da in fz.values()) and np.mean([v[0] for v in fz.values()]) >= 0.0025
    dec.append("fuse: " + " | ".join(f"{s} clear6 {v[0]:+.4f} ({100 * v[1]:.0f}%) 12t {v[2]:+.4f}" for s, v in fz.items())
               + f" mean {np.mean([v[0] for v in fz.values()]):+.4f} -> {'PASS (submission pairing, then LB)' if ok else 'FAIL'}")
pr("\n## pre-registered readout")
for x in dec:
    pr(x)
(X / "audit_bbf.txt").write_text("\n".join(L) + "\n")
