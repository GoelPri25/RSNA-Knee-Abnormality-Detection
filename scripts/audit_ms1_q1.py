"""MS1-Q1 scorer (ledger, pre-registered 2026-10-17): MS1 re-scored at 448 / 560 / 672 px. CPU, read-only.

Sanity: 448 reproduces run_sig_lv2/val_swa.json (max |delta| < 1e-3). Thin-structure score = mean fold-0 AUC of ACL, MCL,
PF OA, Lateral OA (confident cells, Steven ruler). Lever if 560 or 672 >= 448 + 0.010 with clear6 not below 448 by > 0.002;
flat if |delta| < 0.005; else inconclusive. Paired bootstraps (1,000) of the thin score and clear6 vs 448.

Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_ms1_q1.py <project root>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
src = (R / "repository/scripts/audit_phase1.py").read_text()
sys.argv = [sys.argv[0], str(R)]
exec(src.split("def per_target")[0])                       # ids, T, mask, truth, clear, load, auc, scores
Q = C / "run_sig_lv2/q1_resolution"; lines = []
say = lambda s="": (print(s), lines.append(s))
RES = (448, 560, 672)
P = {r: load(Q / f"val_r{r}.json") for r in RES}
ref = load(C / "run_sig_lv2/val_swa.json")
THIN = [T.index(t) for t in ("ACL", "MCL", "PF OA", "Lateral OA")]


def per(S, ix):
    return np.array([auc(S[ix[mask[ix, k]], k], truth[ix[mask[ix, k]], k]) for k in range(12)])


def thin(S, ix):
    return per(S, ix)[THIN].mean()


al = np.arange(len(ids))
say(f"sanity: 448 vs val_swa.json max |delta| {np.abs(P[448] - ref).max():.2e}")
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
base = {r: (thin(P[r], al), *scores(P[r], al)) for r in RES}
say(f"\n{'res':>4s}  thin4    clear6   10-target   " + "  ".join(t[:8] for t in T))
for r in RES:
    say(f"{r:4d}  {base[r][0]:.4f}  {base[r][1]:.4f}  {base[r][2]:.4f}     " + "  ".join(f"{v:8.3f}" for v in per(P[r], al)))
verdicts = []
for r in (560, 672):
    d = np.array([[thin(P[r], b) - thin(P[448], b), scores(P[r], b)[0] - scores(P[448], b)[0]] for b in boots])
    dt, dc = base[r][0] - base[448][0], base[r][1] - base[448][1]
    say(f"\n{r} vs 448: thin4 {dt:+.4f} [{np.percentile(d[:, 0], 2.5):+.4f}, {np.percentile(d[:, 0], 97.5):+.4f}] "
        f"{(d[:, 0] > 0).mean():.0%} better;  clear6 {dc:+.4f} [{np.percentile(d[:, 1], 2.5):+.4f}, {np.percentile(d[:, 1], 97.5):+.4f}]")
    verdicts.append("LEVER" if dt >= 0.010 and dc >= -0.002 else "FLAT" if abs(dt) < 0.005 else "INCONCLUSIVE" if dt > 0 else "LOSS")
say(f"\nVERDICT 560: {verdicts[0]}   672: {verdicts[1]}")

rows_g = [x for x in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(x[t] in ("0", "1") for t in T)]
gids = [x["StudyInstanceUID"] for x in rows_g]; gy = np.array([[int(x[t]) for t in T] for x in rows_g]).astype(bool)
for r in RES:
    g = json.loads((Q / f"gold_r{r}.json").read_text()); m = {x["study_id"]: x["probabilities"] for x in g["predictions"]}
    G = np.array([m[s] for s in gids])[:, [g["targets_in_order"].index(t) for t in T]]
    say(f"gold-58 {r}: macro {np.mean([auc(G[:, k], gy[:, k]) for k in range(12)]):.3f}  thin4 "
        f"{np.mean([auc(G[:, k], gy[:, k]) for k in THIN]):.3f}")
log = json.loads((Q / "q1_log.json").read_text())
say("seconds (fold 0 + gold): " + "  ".join(f"{r}: {log[str(r)]['seconds']}" for r in RES))
(Q / "audit_q1.txt").write_text("\n".join(lines) + "\n")
