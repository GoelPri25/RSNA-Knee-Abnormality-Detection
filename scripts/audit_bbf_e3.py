"""BB-CMP / V4-FUSE against the project goal (gates-vs-project-goal): does a new model improve E3 (LB 0.944) on fold 0?
E3 = probability mean of label-v2 r50, 4-network, 1b, cnx (P1-cnx) as audit_l4copy_clear6.py. Options per new model X
(SWA fused, MS1-table crops; the deployed v4 uses self-located crops, clear6 -0.0011): E3 + X, E3 with X replacing
P1-cnx, E3 + v4 s2026 (reference), E3 + v4 s2026 + X. 1,000 paired study bootstraps vs E3, clear6 and 12-target.
Usage: python3 -I audit_bbf_e3.py <project root> <bbf lane root> <v4 root>
"""
import json
import sys
from pathlib import Path

import numpy as np

R, X, V = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
g = {"__name__": "x"}; argv = sys.argv; sys.argv = [argv[0], str(R)]
exec((R / "repository/scripts/audit_clear_band_lb.py").read_text().split("run = lambda")[0], g); sys.argv = argv
ids, T, scores, auc, load = g["ids"], g["T"], g["scores"], g["auc"], g["load"]
lv = json.load(open(R / "repository/data/coatnet_d96_fold0/assets/soft_labels_final_per_target_latD2.json"))
lab = {r["study_id"]: np.array([.5 if p is None else p for p in r["probabilities"]]) for r in lv["labels"]}
Y = np.array([lab[s] for s in ids]); M = (Y <= .3) | (Y >= .7); TT = Y >= .7
rng = np.random.default_rng(2026); BS = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
C = R / "repository/data/coatnet_d96_fold0"; TM = R / "repository/data/teammate_pp4/fold0_predictions"
E = {"lv2": C / "kaggle_fold0_labelv2/val_swa.json", "1b": C / "run_continue/val_swa.json", "cnx": C / "run_cnx_pertarget/val_swa.json",
     "4net": TM / "four_network_fold0_val_swa.json"}
E = {k: load(p) for k, p in E.items()}
v4 = load(V / "v4_l4copy_s2026/val_swa_fused.json")
NEW = {n: load(X / f"bbf_{n}" / "val_swa_fused.json") for n in
       ("cnx_s2026", "coat0_s2026", "effv2s_s2026", "dino2s_s2026", "dino3s_s2026", "fuse_s2026", "fuse_s2027", "v4_s2028")
       if (X / f"bbf_{n}" / "val_swa_fused.json").exists()}
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


def m12(S, idx=None):
    idx = np.arange(len(ids)) if idx is None else idx
    return np.mean([auc(S[idx[M[idx, k]], k], TT[idx[M[idx, k]], k]) for k in range(12)])


ens = lambda ms: np.mean(ms, 0)
e3 = ens(list(E.values()))
pr(f"E3: clear6 {scores(e3)[0]:.4f} 12t {m12(e3):.4f} | v4 s2026 alone clear6 {scores(v4)[0]:.4f}")
opts = {"E3 + v4 s2026": ens(list(E.values()) + [v4])}
for n, S in NEW.items():
    opts[f"E3 + {n}"] = ens(list(E.values()) + [S])
    opts[f"E3 (P1-cnx -> {n})"] = ens([E[k] for k in ("lv2", "1b", "4net")] + [S])
    opts[f"E3 + v4 s2026 + {n}"] = ens(list(E.values()) + [v4, S])
c0, a0 = scores(e3)[0], m12(e3)
for k, S in opts.items():
    d = np.array([scores(S, i)[0] - scores(e3, i)[0] for i in BS])
    pr(f"{k:36s} clear6 {scores(S)[0]:.4f} d {scores(S)[0] - c0:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] "
       f"P>0 {100 * (d > 0).mean():.0f}% | 12t d {m12(S) - a0:+.4f}")
(X / "audit_bbf_e3.txt").write_text("\n".join(L) + "\n")
