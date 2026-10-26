"""AM-1 (ledger, pre-registered 2026-10-17): whole-score impact of the AM-0 evidence score (total positive CAM mass). CPU.

Primary: 12-target macro AUC on label-v2 confident cells, paired bootstraps vs the unchanged model. Secondary: clear6 /
10-target on Steven-v4 (audit_phase1 yardstick) and the LB fit. Variants V1 / V2 / V3 as in the ledger row.
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_attention_map_impact.py <project root>
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
OUT = D / "attention_side_audit/attention_map_impact.txt"; lines = []
say = lambda s="": (print(s), lines.append(s))

z = np.load(D / "attention_side_audit/kaggle_output/attention_side_fold0.npz")
pos = {s: i for i, s in enumerate(z["ids"])}; ix = np.array([pos[s] for s in ids])
assert z["ok"][ix].all()
lg, mass = z["logits"].astype(float)[ix], z["camx"].astype(np.float64).sum((1, 3))[ix]
lv = json.loads((D / "teammate_labelv2/soft_labels_final_per_target_latD2.json").read_text())
assert lv["targets_in_order"] == T
m = {r["study_id"]: r for r in lv["labels"]}
yv = np.array([[.5 if p is None else p for p in m[s]["probabilities"]] for s in ids]); wv = np.array([m[s]["weights"] for s in ids])
mv = (wv > 0) & ((yv <= .3) | (yv >= .7)); tv = yv >= .7

TM = D / "teammate_pp4/fold0_predictions"
P1e = load(C / "run_r50_pertarget/val_swa.json")
say(f"sanity: Spearman(1e val_swa, npz logit) min over findings "
    f"{min(pd.Series(P1e[:, k]).rank().corr(pd.Series(lg[:, k]).rank()) for k in range(12)):.4f}")
E3 = np.mean([load(C / "kaggle_fold0_labelv2/val_swa.json"), load(TM / "four_network_fold0_val_swa.json"),
              load(C / "run_continue/val_swa.json"), load(C / "run_cnx_pertarget/val_swa.json")], 0)
rk = lambda V: np.column_stack([pd.Series(V[:, k]).rank(pct=True).to_numpy() for k in range(12)])
WIN = [T.index(t) for t in ("Medial Meniscus", "Lateral Meniscus", "Lateral OA", "Baker's", "Contusion", "ACL", "MCL")]
V1 = (rk(P1e) + rk(mass)) / 2
V2 = rk(P1e).copy(); V2[:, WIN] = V1[:, WIN]
S = {"1e": P1e, "V1 1e+mass all": V1, "V2 1e+mass 7 winners": V2, "mass alone": mass, "E3": E3}
for w_ in (0.25, 0.5):
    S[f"V3 E3+mass w{w_}"] = (1 - w_) * rk(E3) + w_ * rk(mass)


def lv12(V, i):
    return np.mean([auc(V[i[mv[i, k]], k], tv[i[mv[i, k]], k]) for k in range(12)])


def all3(V, i):
    return (lv12(V, i), *scores(V, i))


al = np.arange(len(ids)); rng = np.random.default_rng(0)
boots = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
B = {n: np.array([all3(V, b) for b in boots]) for n, V in S.items()}
A = {n: all3(V, al) for n, V in S.items()}
say(f"\n{'score':24s} label-v2 12-target   Steven clear6  10-target   LB fit")
for n in S:
    say(f"  {n:22s} {A[n][0]:.4f}             {A[n][1]:.4f}        {A[n][2]:.4f}     {-0.687 + 1.671 * A[n][1]:.3f}")
say("\npaired vs base (Δ [95%] % better): label-v2 12-target | clear6 | 10-target")
for n, base in [("V1 1e+mass all", "1e"), ("V2 1e+mass 7 winners", "1e"), ("V3 E3+mass w0.25", "E3"), ("V3 E3+mass w0.5", "E3")]:
    d = B[n] - B[base]; dd = np.subtract(A[n], A[base])
    say(f"  {n:22s} vs {base:3s}: " + " | ".join(
        f"{dd[j]:+.4f} [{np.percentile(d[:, j], 2.5):+.4f}, {np.percentile(d[:, j], 97.5):+.4f}] {(d[:, j] > 0).mean():.0%}" for j in range(3)))
say("\nper finding, label-v2 AUC: 1e / V1 / E3 / V3 w0.25 / V3 w0.5")
for k, t in enumerate(T):
    f = lambda V: auc(V[mv[:, k], k], tv[mv[:, k], k])
    say(f"  {t:17s} {f(P1e):.3f} / {f(V1):.3f} / {f(E3):.3f} / {f(S['V3 E3+mass w0.25']):.3f} / {f(S['V3 E3+mass w0.5']):.3f}")
d = B["V3 E3+mass w0.25"] - B["E3"]; dd = np.subtract(A["V3 E3+mass w0.25"], A["E3"])
d5 = B["V3 E3+mass w0.5"] - B["E3"]; dd5 = np.subtract(A["V3 E3+mass w0.5"], A["E3"])
go = [n for n, a_, b_ in (("w0.25", dd, d), ("w0.5", dd5, d5)) if a_[0] >= 0.003 and (b_[:, 0] > 0).mean() >= 0.9 and a_[1] >= -0.002]
say("\nVERDICT: " + ("worth a submission test: V3 " + ", ".join(go) if go else "NO: does not lift the E3 whole score enough"))
OUT.write_text("\n".join(lines) + "\n")
