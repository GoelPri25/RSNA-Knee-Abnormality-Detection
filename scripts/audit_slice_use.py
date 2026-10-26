"""MS1-L1 scorer (ledger, pre-registered 2026-10-17): does MS1 use the neighbouring slices? CPU, read-only.

Inputs: run_sig_lv2/l1_slice_use/{ms1,lv2r50}_{normal,grey,wide}.json (eval_slice_use_gpu.py). Primary (user: label-v2
as judge): 12-target macro AUC on label-v2 confident cells; drop = mode - normal, 1,000 paired bootstraps. Secondary:
thin4 (ACL, MCL, PF OA, Lateral OA) on label-v2, Steven clear6 / 10-target, ranking change 1 - rank corr(normal, mode).
Sanity: normal reproduces each model's val_swa.json (per-target AUC |delta| <= 0.002).
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_slice_use.py <project root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
src = (R / "repository/scripts/audit_phase1.py").read_text()
sys.argv = [sys.argv[0], str(R)]
exec(src.split("def per_target")[0])                       # ids, T, mask, truth, clear, CB, TEN, load, auc, scores (Steven)
Q = C / "run_sig_lv2/l1_slice_use"; lines = []
say = lambda s="": (print(s), lines.append(s))
lv = json.loads((D / "teammate_labelv2/soft_labels_final_per_target_latD2.json").read_text())
m = {r["study_id"]: r for r in lv["labels"]}
yv = np.array([[.5 if p is None else p for p in m[s]["probabilities"]] for s in ids]); wv = np.array([m[s]["weights"] for s in ids])
mv = (wv > 0) & ((yv <= .3) | (yv >= .7)); tv = yv >= .7
THIN = [T.index(t) for t in ("ACL", "MCL", "PF OA", "Lateral OA")]
REF = {"ms1": C / "run_sig_lv2/val_swa.json", "lv2r50": C / "kaggle_fold0_labelv2/val_swa.json"}
MODES = ("normal", "grey", "wide")
P = {(t, md): load(Q / f"{t}_{md}.json") for t in REF for md in MODES}


def per(V, i):
    return np.array([auc(V[i[mv[i, k]], k], tv[i[mv[i, k]], k]) for k in range(12)])


def met(V, i):
    p = per(V, i)
    return (p.mean(), p[THIN].mean(), *scores(V, i))


al = np.arange(len(ids)); rng = np.random.default_rng(0)
boots = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
say("MS1-L1 slice-use test (fold 0, SWA, 94 windows, no TTA)\n")
for t, p in REF.items():
    d = np.abs(per(P[(t, "normal")], al) - per(load(p), al)).max()
    rc = min(pd.Series(P[(t, "normal")][:, k]).rank().corr(pd.Series(load(p)[:, k]).rank()) for k in range(12))
    say(f"sanity {t}: normal vs val_swa max per-target AUC |delta| {d:.4f}, min rank corr {rc:.4f}  {'OK' if d <= 0.002 else 'CHECK'}")
say(f"\n{'model':7s} {'input':7s} lv2-12   thin4   clear6  10-tgt  rank change")
A, B = {}, {}
for t in REF:
    for md in MODES:
        A[(t, md)] = met(P[(t, md)], al); B[(t, md)] = np.array([met(P[(t, md)], b) for b in boots])
        ch = np.mean([1 - pd.Series(P[(t, "normal")][:, k]).rank().corr(pd.Series(P[(t, md)][:, k]).rank()) for k in range(12)])
        say(f"{t:7s} {md:7s} " + "  ".join(f"{v:.4f}" for v in A[(t, md)]) + f"  {ch:.4f}")
say("\ndrop vs normal (Δ [95%]):        lv2-12                        thin4")
drop = {}
for t in REF:
    for md in ("grey", "wide"):
        d = B[(t, md)] - B[(t, "normal")]; dd = np.subtract(A[(t, md)], A[(t, "normal")]); drop[(t, md)] = dd
        say(f"  {t:7s} {md:5s} " + "   ".join(f"{dd[j]:+.4f} [{np.percentile(d[:, j], 2.5):+.4f}, {np.percentile(d[:, j], 97.5):+.4f}]"
                                           for j in (0, 1)))
say("\nper finding, label-v2 AUC (normal / grey / wide):")
for k, t_ in enumerate(T):
    say(f"  {t_:17s} " + "    ".join(f"{t}: " + " / ".join(f"{per(P[(t, md)], al)[k]:.3f}" for md in MODES) for t in REF))
dm, dr = -drop[("ms1", "grey")][0], -drop[("lv2r50", "grey")][0]
if dr < 0.005:
    v = "NOT INFORMATIVE: the r50 control barely uses the neighbours either"
elif drop[("ms1", "grey")][0] >= 0.003:
    v = "MS1 BETTER ON GREY: the slice packing hurts MS1 -> grey / centre-init input is the next run"
elif dm <= 0.003 or dm <= dr / 3:
    v = "DEPTH UNUSED BY MS1 -> centre-init patch embedding with its own LR (T1+T2) is the next run"
elif dm >= 2 * dr / 3:
    v = "DEPTH USED BY MS1: lead 1 dropped -> lead 2 (layer probe)"
else:
    v = "INCONCLUSIVE: read thin4 and the ranking change"
say(f"\nVERDICT (grey drop, lv2-12: MS1 {dm:+.4f}, r50 {dr:+.4f}): {v}")
(Q / "audit_l1.txt").write_text("\n".join(lines) + "\n")
