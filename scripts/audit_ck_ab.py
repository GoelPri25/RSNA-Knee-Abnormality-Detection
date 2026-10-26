"""Exp CK-A / CK-B audit (pre-registered in docs/experiment_ledger.md row CK-A / CK-B): stronger ConvNeXt-T checkpoints
vs P1-cnx on the identical 1e recipe.

Rules (fixed before launch): a CK run replaces P1-cnx as the ConvNeXt teacher / member / J2 branch if own 10-target
>= 0.9094 AND clear6 >= 0.9721 AND mean(CK, 1e) clear6 >= 0.9758; if both qualify, the higher own 10-target wins
(within 0.001: the higher partner clear6). Single-model PASS: clear6 >= 0.9745 AND 10-target >= 0.9053.
Diagnostics (no selection): paired study bootstrap vs P1-cnx (2,000, seed 0), per target, val12 / val BCE / train-loss
curves vs P1-cnx, E3 with CK in place of P1-cnx, mean(CK, label-v2), gold-58 (development evidence only).

Usage (CPU, single-threaded): python -I audit_ck_ab.py <project root> [run_cnx_in12k run_cnx_dinov3]
"""
import csv
import sys
from pathlib import Path

src = (Path(sys.argv[1]) / "repository/scripts/screen_lean_ensembles.py").read_text()
argv, sys.argv = sys.argv, sys.argv[:1] + [sys.argv[1]]
exec(src.split("def table(pool)")[0])
RUNS = argv[2:] or ["run_cnx_in12k", "run_cnx_dinov3"]
RUNS = [r for r in RUNS if (C / r / "val_swa.json").is_file()]
P.update({r: load(C / r / "val_swa.json") for r in RUNS})
E3 = ("lv2", "4net", "1b", "cnx")
rng = np.random.default_rng(0); boots = [rng.choice(idx, len(idx)) for _ in range(2000)]


def ci(A, B):
    d = np.array([np.subtract(scores(A, b), scores(B, b)) for b in boots])
    lo, hi = np.percentile(d, [2.5, 97.5], 0)
    return lo, hi


def per_target(S):
    return np.array([auc(S[mask[:, k], k], truth[mask[:, k], k]) for k in range(12)])


print("CK-A / CK-B audit: clear6 / 10-target on the Steven-v4 ruler (fold 0, SWA 21-25, no TTA)\n")
ref = scores(P["cnx"], idx); ref_pair = scores(mean(("cnx", "1e")), idx)
print(f"P1-cnx                 {ref[0]:.4f} / {ref[1]:.4f}   mean(P1-cnx, 1e) {ref_pair[0]:.4f} / {ref_pair[1]:.4f}")
qual = {}
for r in RUNS:
    s = scores(P[r], idx); pair = scores(mean((r, "1e")), idx); lo, hi = ci(P[r], P["cnx"])
    replace = s[1] >= 0.9094 and s[0] >= 0.9721 and pair[0] >= 0.9758
    single = s[0] >= 0.9745 and s[1] >= 0.9053
    if replace:
        qual[r] = (s[1], pair[0])
    print(f"\n{r}: {s[0]:.4f} / {s[1]:.4f}   vs P1-cnx {s[0] - ref[0]:+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}] / "
          f"{s[1] - ref[1]:+.4f} [{lo[1]:+.4f},{hi[1]:+.4f}]")
    print(f"  mean({r}, 1e) {pair[0]:.4f} / {pair[1]:.4f}  (P1-cnx pair {ref_pair[0]:.4f} / {ref_pair[1]:.4f})")
    print(f"  REPLACE P1-cnx: {'YES' if replace else 'NO'} (10t {s[1]:.5f} >= 0.9094: {s[1] >= 0.9094}; clear6 {s[0]:.5f} "
          f">= 0.9721: {s[0] >= 0.9721}; pair clear6 {pair[0]:.5f} >= 0.9758: {pair[0] >= 0.9758})")
    print(f"  SINGLE-MODEL: {'PASS' if single else 'FAIL'} (clear6 >= 0.9745, 10-target >= 0.9053)")
    e3, e3r = scores(mean(E3), idx), scores(mean(("lv2", "4net", "1b", r)), idx)
    lv = scores(mean(("lv2", r)), idx); lvc = scores(mean(("lv2", "cnx")), idx)
    print(f"  E3 {e3[0]:.4f} / {e3[1]:.4f} -> with {r} for P1-cnx {e3r[0]:.4f} / {e3r[1]:.4f}  (LB fit {lb_fit(e3r[0]):.3f})")
    print(f"  mean(label-v2, {r}) {lv[0]:.4f} / {lv[1]:.4f}  (mean(label-v2, P1-cnx) {lvc[0]:.4f} / {lvc[1]:.4f})")
    d = per_target(P[r]) - per_target(P["cnx"]); d1 = per_target(P[r]) - per_target(P["1e"])
    print("  per target vs P1-cnx: " + " ".join(f"{t[:8]} {v:+.3f}" for t, v in zip(T, d)))
    print("  per target vs 1e:     " + " ".join(f"{t[:8]} {v:+.3f}" for t, v in zip(T, d1)))
if len(qual) == 2:
    (a, va), (b, vb) = qual.items()
    win = (a if va[1] >= vb[1] else b) if abs(va[0] - vb[0]) < 0.001 else (a if va[0] > vb[0] else b)
    print(f"\nBoth qualify -> {win} (rule: higher own 10-target; within 0.001 the higher partner clear6)")
elif qual:
    print(f"\nReplacement: {next(iter(qual))}")
else:
    print("\nNo replacement: P1-cnx stays the ConvNeXt member / teacher")

print("\nCurves (epoch: train loss / val12 [/ val BCE]):")
for r in ["run_cnx_pertarget"] + RUNS:
    h = json.loads((C / r / "history.json").read_text())
    print(f"  {r}: " + " ".join(f"{e['epoch']}:{e['train_loss']:.3f}/{e['val_macro_auc']:.4f}"
                                + (f"/{e['val_bce']:.3f}" if e.get("val_bce") is not None else "") for e in h))

rows = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in rows]; gy = np.array([[int(r[t]) for t in T] for r in rows]).astype(bool)
print(f"\nGold-{len(gids)} macro AUC (development evidence only):")
for r in ["run_cnx_pertarget", "run_r50_pertarget"] + RUNS:
    d = json.loads((C / r / "gold_swa.json").read_text()); m = {x["study_id"]: x["probabilities"] for x in d["predictions"]}
    G = np.array([m[s] for s in gids])[:, [d["targets_in_order"].index(t) for t in T]]
    g = [auc(G[:, k], gy[:, k]) if 0 < gy[:, k].sum() < len(gids) else np.nan for k in range(12)]
    print(f"  {r:22s} {np.nanmean(g):.3f}   " + " ".join(f"{t[:8]} {v:.3f}" for t, v in zip(T, g)))
