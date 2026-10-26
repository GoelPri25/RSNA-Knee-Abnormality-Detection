"""FB-F gate G2 (pre-registered in the ledger): do R6 and the 2.5D reader agree with Raptor-384 on fastMRI exams as much as on
competition studies? Per finding, Spearman(R6, Raptor-384) and Spearman(reader, Raptor-384) on the competition train
studies (Kaggle teacher shards) and on the fastMRI exams scored so far (Colab chunks). A finding whose fastMRI agreement is
more than 0.10 below its competition agreement is flagged: its fastMRI R6 / reader values are then not used (Raptor only).
Raptor-384 = mean of plain and mirror, as build_pubkd_targets.py.
Usage: python3 -I fbf_port_gate.py <competition pubkd dir> <fastMRI Raptor dir> <chunk root> [<chunk root> ...]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
CP, FR, ROOTS = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]


def r384(d):
    out = {}
    for f in sorted(d.glob("raptor384_*.json")):
        j = json.loads(f.read_text()); assert j["targets"] == T
        out.update({s: 0.5 * (np.array(v["plain"]) + np.array(v["mirror"])) for s, v in j["pred"].items()})
    return pd.DataFrame.from_dict(out, orient="index", columns=T)


def csvs(files):
    return pd.concat([pd.read_csv(f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[T] for f in files])


def spear(a, b):
    return a.rank().corr(b.rank())


comp = {"r6": csvs([CP / "r6" / "r6_train_all.csv"]), "reader": csvs([CP / "r6" / "reader_train_all.csv"])}
done = [c for r in ROOTS for c in sorted(r.glob("c[0-9][0-9][0-9]")) if (c / "DONE").exists()]
fm = {"r6": csvs([c / "r6_train_shard0.csv" for c in done]), "reader": csvs([c / "reader_train_shard0.csv" for c in done])}
RC, RF = r384(CP), r384(FR)
rows, flagged = [], []
for k in ("r6", "reader"):
    ic = sorted(set(comp[k].index) & set(RC.index)); iF = sorted(set(fm[k].index) & set(RF.index))
    for c in T:
        a = spear(comp[k].loc[ic, c], RC.loc[ic, c]); b = spear(fm[k].loc[iF, c], RF.loc[iF, c])
        rows.append({"member": k, "finding": c, "competition": round(a, 3), "fastmri": round(b, 3), "diff": round(b - a, 3),
                     "n_comp": len(ic), "n_fm": len(iF)})
        if b < a - 0.10:
            flagged.append((k, c))
df = pd.DataFrame(rows)
print(f"chunks {len(done)}; fastMRI exams with Raptor: r6 {rows[0]['n_fm']}, reader {rows[-1]['n_fm']}")
print(df.to_string(index=False))
print("mean agreement competition / fastMRI:", {k: (round(df[df.member == k].competition.mean(), 3),
                                                     round(df[df.member == k].fastmri.mean(), 3)) for k in ("r6", "reader")})
print("G2 FLAGGED (fastMRI agreement > 0.10 below competition):", flagged or "none")
missing = sorted(set(fm["r6"].index) - set(RF.index))
print(f"fastMRI exams without a Raptor prediction: {len(missing)}", missing[:5])
