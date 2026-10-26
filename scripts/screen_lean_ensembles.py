"""Lean-ensemble screen (CPU, fold 0, 2026-10-16): every equal-probability-mean subset of <= 4 members from the existing
fold-0 checkpoints, vs L4 (LB 0.943, 4 members) and E2 (L4 + B2, LB 0.944, 5 members). No training, no submission.

Pools: STRICT = singles with a public LB >= 0.93 (1e 0.936, label-v2 0.937, 1e-s2 0.934, 3b 0.934, 1b 0.933, 1c 0.933,
7a 0.931); FULL = STRICT + the L4 / E2 members and partners without such a single score (d2_224 0.925, 2c, ConvNeXt
P1-cnx, Swin3D B2). Metrics exactly as scripts/audit_phase1.py (clear6, 10-target on the Steven-v4 ruler); LB guess =
the clear-band fit over 8 submitted singles (LB = -0.687 + 1.671 x clear6; held-out L4 0.941 predicted vs 0.943).
Known blind spots: fold 0 cannot read label changes (label-v2 / 3b / 7a), and underrates Swin3D on the LB by 0.006-0.013.
Selection optimism: the best subset of hundreds is flattered; 200 random half splits pick the best subset by clear6 on
one half and score it against L4 on the other half.

Usage (CPU, single-threaded): python -I screen_lean_ensembles.py <project root>
"""
import itertools
import sys
from pathlib import Path

src = Path(sys.argv[1] + "/repository/scripts/audit_phase1.py").read_text()
exec(src.split("def per_target")[0].replace("sys.argv[2:]", "[]"))
import numpy as np

LB = {"1e": .936, "lv2": .937, "1e_s2": .934, "3b": .934, "1b": .933, "1c": .933, "7a": .931, "d2_224": .925,
      "2c": None, "cnx": None, "B2": None, "4net": .936, "4net_ds": .932, "v3w": .933, "fine": .931}
P = {"1e": load(C / "run_r50_pertarget/val_swa.json"), "lv2": load(C / "kaggle_fold0_labelv2/val_swa.json"),
     "1e_s2": load(C / "run_r50_pertarget_seed/val_swa.json"), "3b": load(C / "run_r50_state/val_swa.json"),
     "1b": load(C / "run_continue/val_swa.json"), "1c": load(C / "run_r50/val_swa.json"),
     "7a": load(C / "run_r50_rulecap/val_swa.json"), "d2_224": load(D / "r50_tta/kaggle_output/d2_ep15__none.json"),
     "2c": load(C / "run_r50_plane/val_swa.json"), "cnx": load(C / "run_cnx_pertarget/val_swa.json"),
     "B2": load(C / "run_swin3d_b2/val_swa.json")}
TM = D / "teammate_pp4/fold0_predictions"                      # teammate's models (2026-10-16), sha256-checked
P.update({"4net": load(TM / "four_network_fold0_val_swa.json"),
          "4net_ds": load(TM / "four_network_deep_supervision_fold0_val_swa.json"),
          "v3w": load(TM / "label_v3w_fold0_val_swa.json"), "fine": load(TM / "fine_branch_fold0_val_swa.json")})
SPLITS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
STRICT = [m for m, v in LB.items() if v and v >= .93]
FULL = list(P)
L4 = ("d2_224", "1b", "2c", "1e"); E2 = L4 + ("B2",)
idx = np.arange(len(ids))
lb_fit = lambda c6: -0.687 + 1.671 * c6
mean = lambda ms: np.mean([P[m] for m in ms], 0)


def table(pool):
    subs = [s for k in (1, 2, 3, 4) for s in itertools.combinations(pool, k)]
    return {s: scores(mean(s), idx)[:2] for s in subs}


print("singles: clear6 / 10-target / LB")
for m in FULL:
    c6, m10 = scores(P[m], idx)[:2]
    print(f"  {m:7s} {c6:.4f}  {m10:.4f}  LB {LB[m] if LB[m] else '-'}  (fit {lb_fit(c6):.3f})")
ref = {n: scores(mean(s), idx)[:2] for n, s in (("L4", L4), ("E2", E2))}
for n, (c6, m10) in ref.items():
    print(f"{n}: clear6 {c6:.4f} 10-target {m10:.4f} (fit {lb_fit(c6):.3f})")

rng = np.random.default_rng(0); boots = [rng.choice(idx, len(idx)) for _ in range(1000)]
bl4 = [scores(mean(L4), b)[:2] for b in boots]
for name, pool in (("STRICT (LB >= 0.93 singles)", STRICT), ("FULL", FULL)):
    T = table(pool)
    print(f"\n=== {name}: {len(T)} subsets of 1-4 members")
    for k in (2, 3, 4):
        best = sorted([s for s in T if len(s) == k], key=lambda s: -T[s][0])[:4]
        for s in best:
            c6, m10 = T[s]; S = mean(s)
            d = np.array([np.subtract(scores(S, b)[:2], x) for b, x in zip(boots[:300], bl4[:300])])
            print(f"  {k}  {'+'.join(s):30s} clear6 {c6:.4f} 10t {m10:.4f} fit {lb_fit(c6):.3f} | vs L4 clear6 "
                  f"{c6 - ref['L4'][0]:+.4f} [{np.percentile(d[:, 0], 2.5):+.4f},{np.percentile(d[:, 0], 97.5):+.4f}] "
                  f"10t {m10 - ref['L4'][1]:+.4f}")
    # selection optimism: choose on half A by clear6, score on half B
    gains, inbest = [], 0
    r2 = np.random.default_rng(1)
    cand = [s for s in T if len(s) >= 2]
    for _ in range(SPLITS):
        perm = r2.permutation(idx); a, b = np.sort(perm[:len(idx) // 2]), np.sort(perm[len(idx) // 2:])
        sel = max(cand, key=lambda s: scores(mean(s), a)[0])
        gains.append(np.subtract(scores(mean(sel), b)[:2], scores(mean(L4), b)[:2]))
        inbest += sel == max(cand, key=lambda s: T[s][0])
    g = np.array(gains)
    print(f"  honest check ({SPLITS} half splits, pick best clear6 on half A, score on half B vs L4): clear6 "
          f"{g[:, 0].mean():+.4f} (sd {g[:, 0].std():.4f}, > 0 in {(g[:, 0] > 0).mean():.0%}); 10-target "
          f"{g[:, 1].mean():+.4f}; full-data winner re-picked in {100 * inbest / SPLITS:.0f}% of splits")
