"""P3 v2 audit (ledger P3v2-T / P3v2-S): teacher sanity and student fold-0 screen vs label-v2.

teachers: each CK-A teacher's fold-0 SWA clear6 / 10-target (sanity floor 10-target >= 0.895), out-of-fold coverage of
its split, and the fold-0 mean of the available teachers (free ensemble readout).
students: clear6 / 10-target vs label-v2 (paired study bootstrap 2,000, seed 0), the pre-registered fold-0 screen
(clear6 >= 0.9700 AND 10-target >= 0.9000 -> submit to the LB), per target, mild band (report mild / hedged positives
vs all negatives) and clear band, clear-band LB fit, gold-58 (development evidence only), train-loss / val BCE curve.

Usage (CPU, single-threaded): python3 -I audit_p3v2.py <project root> teachers|students
"""
import csv
import sys
from pathlib import Path

root, MODE = sys.argv[1], sys.argv[2]
src = (Path(root) / "repository/scripts/screen_lean_ensembles.py").read_text()
sys.argv = sys.argv[:2]
exec(src.split("def table(pool)")[0])
rng = np.random.default_rng(0); boots = [rng.choice(idx, len(idx)) for _ in range(2000)]
mild = mask & truth & ~clear                                     # report positive, not moderate+


def band(S, sel_pos):
    out = []
    for k in range(12):
        pos = np.where(sel_pos[:, k])[0]; neg = np.where(mask[:, k] & ~truth[:, k])[0]
        if len(pos) >= 10:
            out.append(auc(S[np.r_[pos, neg], k], np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]))
    return np.mean(out)


per = lambda S: np.array([auc(S[mask[:, k], k], truth[mask[:, k], k]) for k in range(12)])

if MODE == "teachers":
    split = {r["study_id"]: int(r["teacher_split"]) for r in csv.DictReader((C / "assets/teacher_splits.csv").open())}
    have = []
    for k in range(4):
        run = C / f"run_cnxa_teacher{k}"
        if not (run / "val_swa.json").is_file():
            print(f"teacher {k}: not available"); continue
        S = load(run / "val_swa.json"); c6, t10 = scores(S, idx); have.append(S)
        o = json.loads((run / "oof_swa.json").read_text())["predictions"]
        ok = sorted(r["study_id"] for r in o) == sorted(s for s, v in split.items() if v == k)
        print(f"teacher {k}: fold-0 clear6 {c6:.4f} / 10-target {t10:.4f}  SANITY {'PASS' if t10 >= 0.895 else 'FAIL'} "
              f"(>= 0.895); oof {len(o)} studies, covers split {k} exactly: {ok}")
    if len(have) > 1:
        c6, t10 = scores(np.mean(have, 0), idx); print(f"mean of {len(have)} teachers on fold 0: {c6:.4f} / {t10:.4f}")
    sys.exit()

ref = P["lv2"]; r6, r10 = scores(ref, idx)
print(f"label-v2 (reference, LB 0.937): {r6:.4f} / {r10:.4f}; mild band {band(ref, mild):.4f}; LB fit {lb_fit(r6):.3f}\n")
for run in ("run_r50_student_state", "run_r50_student_a050"):
    if not (C / run / "val_swa.json").is_file():
        print(f"{run}: not available\n"); continue
    S = load(C / run / "val_swa.json"); c6, t10 = scores(S, idx)
    d = np.array([np.subtract(scores(S, b), scores(ref, b)) for b in boots]); lo, hi = np.percentile(d, [2.5, 97.5], 0)
    print(f"{run}: clear6 {c6:.4f} / 10-target {t10:.4f}  vs label-v2 {c6 - r6:+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}] / "
          f"{t10 - r10:+.4f} [{lo[1]:+.4f},{hi[1]:+.4f}]")
    print(f"  SCREEN: {'SUBMIT' if c6 >= 0.97 and t10 >= 0.90 else 'HOLD (inspect)'} (clear6 >= 0.9700, 10-target >= 0.9000); "
          f"LB fit {lb_fit(c6):.3f}; mild band {band(S, mild):.4f} (label-v2 {band(ref, mild):.4f})")
    print("  per target vs label-v2: " + " ".join(f"{t[:8]} {v:+.3f}" for t, v in zip(T, per(S) - per(ref))))
    h = json.loads((C / run / "history.json").read_text())
    print("  curve (epoch: train loss / val12 / val BCE): " + " ".join(
        f"{e['epoch']}:{e['train_loss']:.3f}/{e['val_macro_auc']:.4f}/{e.get('val_bce') or 0:.3f}" for e in h if e["epoch"] % 5 == 0))
    rows = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
    gids = [r["StudyInstanceUID"] for r in rows]; gy = np.array([[int(r[t]) for t in T] for r in rows]).astype(bool)
    g = json.loads((C / run / "gold_swa.json").read_text()); m = {x["study_id"]: x["probabilities"] for x in g["predictions"]}
    G = np.array([m[s] for s in gids])[:, [g["targets_in_order"].index(t) for t in T]]
    print(f"  gold-{len(gids)} macro {np.nanmean([auc(G[:, k], gy[:, k]) if 0 < gy[:, k].sum() < len(gids) else np.nan for k in range(12)]):.3f}\n")
