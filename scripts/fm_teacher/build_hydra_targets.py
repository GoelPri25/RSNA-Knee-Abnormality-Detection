"""HYDRA (run C, 2026-10-22): per-member teacher probabilities for the student's member heads, one head per teacher.
Members (all four cover every pool study, competition + fastMRI): r384 and r224 (Raptor CoAtNet-rmlp-2, mean of plain and
mirror probabilities), r6 (heliosli ConvNeXt), reader (2.5D reader). The KF team is NOT a member: KF-1..KF-5 trained on
fold 0, so their predictions would carry fold-0 labels into the student and inflate its fold-0 read.
Output: {"targets_in_order", "members", "source", "studies": {study_id: [[12 probabilities] x members]}}.
Usage: python3 -I build_hydra_targets.py <pool dir (raptor384_0.json, raptor224_0.json, r6/)> <main targets json> <out>"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

POOL, MAIN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
main = json.loads(MAIN.read_text()); T = main["targets_in_order"]; ids = [r["study_id"] for r in main["labels"]]
mem = {}
for m in ("384", "224"):
    d = json.loads((POOL / f"raptor{m}_0.json").read_text()); assert d["targets"] == T
    mem[f"r{m}"] = {s: (np.array(v["plain"]) + np.array(v["mirror"])) / 2 for s, v in d["pred"].items()}
for m, f in (("r6", "r6_train_all.csv"), ("reader", "reader_train_all.csv")):
    rows = list(csv.DictReader((POOL / "r6" / f).open())); assert list(rows[0])[1:] == T
    mem[m] = {r["StudyInstanceUID"]: np.array([float(r[t]) for t in T]) for r in rows}
for m, d in mem.items():
    a = np.stack(list(d.values())); assert a.min() >= 0 and a.max() <= 1, m
    assert set(ids) <= set(d), f"{m} misses {len(set(ids) - set(d))} studies"
names = list(mem)
out = {"targets_in_order": T, "members": names,
       "source": f"build_hydra_targets.py; studies of {MAIN.name} (sha {hashlib.sha256(MAIN.read_bytes()).hexdigest()[:12]})",
       "studies": {s: [np.round(mem[m][s], 6).tolist() for m in names] for s in ids}}
OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out))
a = np.array(list(out["studies"].values()))                                     # [n, M, 12]
print(f"{len(ids)} studies ({sum(s.startswith('fm_') for s in ids)} fastMRI), members {names}; mean prob per member",
      a.mean((0, 2)).round(3).tolist(), "| mean |member - mean of members|", np.abs(a - a.mean(1, keepdims=True)).mean((0, 2)).round(3).tolist())
print("wrote", OUT, hashlib.sha256(OUT.read_bytes()).hexdigest())
