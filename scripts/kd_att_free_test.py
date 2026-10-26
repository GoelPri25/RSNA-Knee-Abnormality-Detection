"""KD-ATT-T: free test before window-attention distillation (CPU, read-only; pre-registered in the ledger row KD-ATT-T).
Does a student's shortfall vs its teacher (Raptor 384) sit in the studies where the teacher reads non-sagittal series?
Reliance R[i,j] = teacher |logit change| when Ax, CorF and CorNF are each dropped (summed; kd_gap_part1.npz drop logits);
secondary = teacher attention mass on non-sagittal windows. Cells = Steven-confident fold-0 cells. Shortfall per cell =
teacher percentile - student percentile on positives, reversed on negatives. share_hi = share of the total shortfall in
the top reliance tercile (per finding, then mean; 1/3 = no concentration). Control = Raptor 224 (the teacher's sibling).
Usage: LBP_OUT=data/lb_predictor_v2 python3 -I scripts/kd_att_free_test.py <kd_gap_xai dir> <cv1 val_swa_fused.json>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_predictor as L                                   # noqa: E402

X = Path(sys.argv[1]); OUT = X.parent / "kd_att_free_test.txt"; lines = []
T = L.T; BOUNDS = np.cumsum([0, 26, 22, 18, 12, 18]); slot_of = lambda c: np.searchsorted(BOUNDS, c, side="right") - 1
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)


row = lambda name, v, f="7.3f": pr(f"{name:24s} " + " ".join(f"{x:{f}}" for x in v) + f"  | mean {np.nanmean(v):{f}}")
hdr = lambda name: pr(f"{name:24s} " + " ".join(f"{s:>7s}" for s in [t[:7] for t in T]))

z = np.load(X / "kd_gap_part1.npz"); ids = list(z["ids"]); n = len(ids)
o = np.array([{s: i for i, s in enumerate(L.IDS)}[s] for s in ids])
t0, m0 = L.RL["steven"]; t0, m0 = t0[o], m0[o]
TP = z["teacher_prob"].astype(np.float64)
S = {m: z[f"{m}_prob"].astype(np.float64) for m in ("E4", "ra", "B", "v4")}
S["cv1"] = L.load_pred(sys.argv[2])[o]
S["r224"] = L.PUB["r224"][o]
assert not np.isnan(S["r224"]).any()
pr(f"teacher npz vs lb_predictor r384 max|d| {np.abs(TP - L.PUB['r384'][o]).max():.4f} (same mirror-TTA teacher)")

drop = z["teacher_drop"].astype(np.float64)                               # (n, 5 slots, 12) logits with the slot dropped
dl = np.abs(drop - lg(TP)[:, None]); dl = np.nan_to_num(dl, nan=0.0)        # NaN = slot empty in that study -> no reliance
REL = {"drop non-sag": dl[:, 2:].sum(1)}
sl = slot_of(z["cs"]); att = z["teacher_att"].astype(np.float64)
REL["attn non-sag"] = (att * (sl >= 2)[:, :, None]).sum(1)

pct = lambda P: np.column_stack([pd.Series(P[:, j]).rank(pct=True).to_numpy() for j in range(12)])
PT = pct(TP); sgn = np.where(t0, 1.0, -1.0)
SH = {m: (PT - pct(P)) * sgn for m, P in S.items()}                          # >0 = student worse than the teacher


def terc(R):
    q = np.column_stack([pd.Series(R[:, j]).rank(pct=True).to_numpy() for j in range(12)])
    return np.where(q > 2 / 3, 2, np.where(q > 1 / 3, 1, 0))


def share_hi(sh, tc, idx):
    v = []
    for j in range(12):
        c = idx[m0[idx, j]]; tot = sh[c, j].sum()
        v.append(sh[c[tc[c, j] == 2], j].sum() / tot if tot > 0 else np.nan)
    return np.array(v)


rng = np.random.default_rng(2026); B = 2000; boots = [rng.integers(0, n, n) for _ in range(B)]
for rn, R in REL.items():
    tc = terc(R); full = np.arange(n)
    pr(f"\n== reliance = {rn}: share of each model's shortfall vs the teacher in the top reliance tercile (1/3 = none)")
    hdr("")
    for m in ("E4", "cv1", "ra", "B", "v4", "r224"):
        row(m, share_hi(SH[m], tc, full))
    pr("mean shortfall per Steven cell, tercile lo / mid / hi (percentile points x100):")
    for m in ("E4", "cv1", "r224"):
        pr(f"  {m:5s} " + " / ".join(f"{100 * np.nanmean([SH[m][m0[:, j] & (tc[:, j] == k), j].mean() for j in range(12)]):5.2f}"
                                     for k in (0, 1, 2)))
    for m in ("E4", "cv1"):
        d0 = share_hi(SH[m], tc, full) - share_hi(SH["r224"], tc, full)
        bd = np.array([np.nanmean(share_hi(SH[m], tc, b) - share_hi(SH["r224"], tc, b)) for b in boots])
        lo, hi = np.percentile(bd, [5, 95])
        pr(f"  {m} - r224 share_hi: {np.nanmean(d0):+.3f} (90% CI {lo:+.3f} .. {hi:+.3f}); findings > 0: "
           f"{int((d0 > 0).sum())}/12; {m} share_hi {np.nanmean(share_hi(SH[m], tc, full)):.3f}")
    pr("AUC on Steven cells by reliance tercile (teacher in-sample; gap = teacher - model), mean over findings:")
    for k, nm in ((0, "lo"), (2, "hi")):
        out = {}
        for m, P in [("teacher", TP)] + [(m, S[m]) for m in ("E4", "cv1", "r224")]:
            v = []
            for j in range(12):
                c = m0[:, j] & (tc[:, j] == k)
                if (c & t0[:, j]).sum() >= 5 and (c & ~t0[:, j]).sum() >= 5:
                    v.append(L.auc(P[c, j], t0[c, j]))
                else:
                    v.append(np.nan)
            out[m] = np.array(v)
        pr(f"  {nm}: teacher {np.nanmean(out['teacher']):.4f}  " + "  ".join(
            f"{m} {np.nanmean(out[m]):.4f} (gap {np.nanmean(out['teacher'] - out[m]):+.4f})" for m in ("E4", "cv1", "r224")))

OUT.write_text("\n".join(lines) + "\n"); print(f"\nwrote {OUT}")

# ------------------------------------------------------------------ post-hoc (added after the pre-registered readout:
# the r224 control's total shortfall is ~0 in-sample, so share_hi is undefined for it). Label-free: mean |logit gap| to
# the teacher by reliance tercile, all 825 studies; hi / lo ratio vs r224's (r224 reads like the teacher, so its rise =
# the volatile-study artefact).
lines.append("\n== POST-HOC (not pre-registered): label-free |logit(model) - logit(teacher)| by reliance tercile, all studies")
print(lines[-1])
for rn, R in REL.items():
    tc = terc(R)
    pr(f"-- reliance = {rn}: lo / mid / hi, hi/lo ratio; per-finding hi/lo ratio")
    for m in ("E4", "cv1", "ra", "B", "v4", "r224"):
        g = np.abs(lg(S[m]) - lg(TP))
        lvl = [[g[tc[:, j] == k, j].mean() for j in range(12)] for k in (0, 1, 2)]
        rat = np.array(lvl[2]) / np.array(lvl[0])
        pr(f"  {m:5s} " + " / ".join(f"{np.mean(v):.3f}" for v in lvl) + f"  ratio {np.mean(rat):.2f} | " +
           " ".join(f"{x:4.2f}" for x in rat))
    g4, g2 = np.abs(lg(S["E4"]) - lg(TP)), np.abs(lg(S["r224"]) - lg(TP))
    def ex(idx):
        r4 = [g4[idx][tc[idx, j] == 2, j].mean() / g4[idx][tc[idx, j] == 0, j].mean() for j in range(12)]
        r2 = [g2[idx][tc[idx, j] == 2, j].mean() / g2[idx][tc[idx, j] == 0, j].mean() for j in range(12)]
        return np.mean(r4) - np.mean(r2)
    bd = np.array([ex(b) for b in boots[:500]]); lo, hi = np.percentile(bd, [5, 95])
    pr(f"  E4 ratio - r224 ratio {ex(np.arange(n)):+.2f} (90% CI {lo:+.2f} .. {hi:+.2f})")
OUT.write_text("\n".join(lines) + "\n"); print(f"\nwrote {OUT}")
