"""SER-0 (ledger): which series group each v4 finding draws on. Kaggle kernel rsna-v4-series-ablation-fold0 outputs
(ser0_{config}_fused.json, self-located crops, the LB pipeline) on fold 0; per config vs "all": per-target AUC on
label-v2 confident cells and clear6 (audit_clear_band_lb.py), 1,000 paired study bootstraps on the studies each config
kept. Pre-registered rules (ledger SER-0):
  (a) a finding ignores its decisive evidence if removing its decisive group costs < 0.005 AUC with the 95% interval
      including 0, while removing some other group costs >= 0.010;
  (b) fracture and the 3 OA findings losing < 0.005 without non-FS -> T1 evidence unused (supports FS <-> NF fusion);
      >= 0.010 -> already used.
Usage (container): python -I audit_series_ablation.py <project root> <kernel output dir> <out dir> [<MS1-table val json>]
"""
import json
import sys
from pathlib import Path

import numpy as np

R, K, X = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); X.mkdir(parents=True, exist_ok=True)
g = {"__name__": "x"}; argv = sys.argv; sys.argv = [argv[0], str(R)]
exec((R / "repository/scripts/audit_clear_band_lb.py").read_text().split("run = lambda")[0], g); sys.argv = argv
ids, T, scores, auc, CB = g["ids"], g["T"], g["scores"], g["auc"], g["CB"]
lv = json.load(open(R / "repository/data/coatnet_d96_fold0/assets/soft_labels_final_per_target_latD2.json"))
assert lv["targets_in_order"] == T
lab = {r["study_id"]: np.array([.5 if p is None else p for p in r["probabilities"]]) for r in lv["labels"]}
Y = np.array([lab[s] for s in ids]); M = (Y <= .3) | (Y >= .7); TT = Y >= .7
n = len(ids); L = []
CONFIGS = ["all", "no_nonfs", "no_fs", "no_axial", "no_sag", "no_cor", "sagf_only"]
DECISIVE = {"ACL": ["no_sag"], "MCL": ["no_cor", "no_fs"], "Medial Meniscus": ["no_sag", "no_cor"],
            "Lateral Meniscus": ["no_sag", "no_cor"], "Fracture": ["no_nonfs"], "Baker's": ["no_axial"],
            "Effusion": ["no_sag", "no_axial"], "Medial OA": ["no_fs", "no_nonfs"], "Lateral OA": ["no_fs", "no_nonfs"],
            "PF OA": ["no_fs", "no_nonfs"], "Contusion": ["no_fs"]}


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


def loadc(c, head="fused"):
    d = json.loads((K / f"ser0_{c}_{head}.json").read_text()); assert d["targets_in_order"] == T
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]}
    return np.array([m.get(s, [np.nan] * 12) for s in ids], float)


def a12(S, idx):
    return np.array([auc(S[idx[M[idx, k]], k], TT[idx[M[idx, k]], k]) for k in range(12)])


P = {c: loadc(c) for c in CONFIGS}
log = json.loads((K / "ser0_log.json").read_text())
pr(f"studies {log['studies']} | decode errors {log['stats']['decode_errors']} | fp32 retries {log['stats']['fp32_retries']} "
   f"| skipped per config {log['stats']['skipped']} | {log['device']} {log['seconds']:.0f}s")
pr("fold-0 studies naturally lacking each slot (SagF, SagNF, CorF, CorNF, Ax): " + str(log["stats"]["slot_absent"]))
base = np.where(np.isfinite(P["all"]).all(1))[0]
c6, t12 = scores(P["all"], base)[0], a12(P["all"], base).mean()
pr(f"\nall (self crops): clear6 {c6:.4f} (LB fit {-0.687 + 1.671 * c6:.3f}) | 12t label-v2 {t12:.4f} (SL-0v4 self 0.9453)")
if len(sys.argv) > 4:
    ms = g["load"](sys.argv[4]); c6m = scores(ms)[0]
    pr(f"MS1-table run (MeluXina val_swa_fused): clear6 {c6m:.4f} 12t {a12(ms, np.arange(n)).mean():.4f} -> selfloc - MS1 clear6 {c6 - c6m:+.4f}")
rng = np.random.default_rng(2026); BS = [rng.integers(0, n, n) for _ in range(1000)]
res = {}
pr("\nper config - all: clear6 d [95%] | 12t d | per target d [95%] (label-v2 confident cells, studies kept by the config)")
for c in CONFIGS[1:]:
    idx = np.where(np.isfinite(P[c]).all(1) & np.isfinite(P["all"]).all(1))[0]
    keep = set(idx.tolist())
    bs = [b[np.isin(b, idx)] for b in BS]
    dc = np.array([scores(P[c], b)[0] - scores(P["all"], b)[0] for b in bs])
    dk = np.array([a12(P[c], b) - a12(P["all"], b) for b in bs])               # [1000, 12]
    d0 = a12(P[c], idx) - a12(P["all"], idx)
    lo, hi = np.percentile(dk, 2.5, 0), np.percentile(dk, 97.5, 0)
    res[c] = (d0, lo, hi)
    pr(f"\n## {c} ({len(keep)} studies): clear6 {scores(P[c], idx)[0] - scores(P['all'], idx)[0]:+.4f} "
       f"[{np.percentile(dc, 2.5):+.4f}, {np.percentile(dc, 97.5):+.4f}] | 12t {d0.mean():+.4f}")
    pr("   " + " | ".join(f"{T[k][:8]} {d0[k]:+.3f} [{lo[k]:+.3f}, {hi[k]:+.3f}]" for k in range(12)))

pr("\n## summary table: AUC change per finding when a group is removed (negative = the model uses that group)")
pr(f"{'finding':17s} " + " ".join(f"{c:>10s}" for c in CONFIGS[1:]))
for k, t in enumerate(T):
    pr(f"{t:17s} " + " ".join(f"{res[c][0][k]:+10.3f}" for c in CONFIGS[1:]))

pr("\n## rule (a): decisive evidence ignored?")
for t, dec in DECISIVE.items():
    k = T.index(t)
    worst_dec = min(dec, key=lambda c: res[c][0][k])                           # the decisive group the model leans on most
    d, lo, hi = (res[worst_dec][i][k] for i in range(3))
    other = [c for c in ("no_nonfs", "no_fs", "no_axial", "no_sag", "no_cor") if c not in dec]
    oc = min(other, key=lambda c: res[c][0][k]); od = res[oc][0][k]
    fire = (-d < 0.005 and lo <= 0 <= hi) and -od >= 0.010
    pr(f"{t:17s} decisive {worst_dec:9s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | largest other {oc:9s} {od:+.4f} -> "
       f"{'IGNORES DECISIVE EVIDENCE' if fire else 'no'}")
pr("\n## rule (b): non-FS (T1) evidence for fracture / OA")
fb = []
for t in ("Fracture", "Medial OA", "Lateral OA", "PF OA"):
    k = T.index(t); d = res["no_nonfs"][0][k]; fb.append(d)
    pr(f"{t:12s} no_nonfs {d:+.4f} [{res['no_nonfs'][1][k]:+.4f}, {res['no_nonfs'][2][k]:+.4f}] -> "
       + ("unused (< 0.005)" if -d < 0.005 else "used (>= 0.010)" if -d >= 0.010 else "between"))
pr("Caveat: removing series at test time is partly out of distribution (the slot-absence counts above show how often "
   "training saw each slot missing).")
(X / "audit_series_ablation.txt").write_text("\n".join(L) + "\n")
