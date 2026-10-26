"""Write the Exp 3a label set as a file: HPO-v3.1-D2 with borderline cells (0.3 < p < 0.7) set to 0.85 for
ACL, MCL, Medial OA, Lateral OA, PF OA, Effusion, Baker's. Same rule as border_labels() in coatnet_d96_runner.py.

Usage: python3 scripts/build_d2_border_labels.py   (numpy-free; reads/writes project data only)
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/coatnet_d96_fold0/assets/soft_labels_hpo_v3_1_d2_structured.json"
SRC_SHA = "38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d"
OUT = ROOT / "data/label_releases/hpo_v31_d2_border085"
TARGETS = ["ACL", "MCL", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Baker's"]
LO, HI, TO = 0.3, 0.7, 0.85

raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA
d = json.loads(raw)
cols = [d["targets_in_order"].index(t) for t in TARGETS]
moved = {t: 0 for t in TARGETS}
for r in d["labels"]:
    for t, k in zip(TARGETS, cols):
        p = r["probabilities"][k]
        if p is not None and LO < p < HI:
            r["probabilities"][k] = TO; moved[t] += 1
d["version"] = "hpo_v3_1_d2_border085"
d["note"] = (f"Derived from hpo_v3_1_d2_structured (sha256 {SRC_SHA}): cells with {LO} < p < {HI} set to {TO} for "
             f"{', '.join(TARGETS)}; weights and other targets unchanged. Fixed rule on text-derived labels, no model. "
             f"Cells moved (all {len(d['labels'])} studies): {json.dumps(moved)}. Original note: " + d["note"])
d["derived_from_sha256"] = SRC_SHA
OUT.mkdir(parents=True, exist_ok=True)
f = OUT / "soft_labels_hpo_v3_1_d2_border085.json"
f.write_text(json.dumps(d) + "\n")
print(f, hashlib.sha256(f.read_bytes()).hexdigest(), moved)
