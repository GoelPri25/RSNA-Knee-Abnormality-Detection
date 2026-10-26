"""Write the Exp 3b label set: HPO-v3.1-D2 with cells re-valued by D2's own rule-extractor state (fixed rule, no model).

  "mild" (BORD)               -> 0.85 for Effusion, MCL, Medial OA
  trace / below threshold (BELOW) -> 0.15 for Effusion, MCL, ACL, Contusion

Why: Exp 3a (0.3 < p < 0.7 -> 0.85 on 7 targets) scored LB 0.926 vs 1c 0.933. The 0.3-0.7 band mixes BORD cells
(gold-58 mostly positive) and BELOW cells (Steven-v4 positive, gold-58 mostly negative). Values chosen before any run.
Weights and all other cells unchanged. Rule states: label_datasets/hpo_v3_1_d2/rules/s5_fx_site (the D2 builder's
final step).

Usage: python3 scripts/build_d2_state_labels.py   (numpy-free; reads/writes project data only)
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/coatnet_d96_fold0/assets/soft_labels_hpo_v3_1_d2_structured.json"
SRC_SHA = "38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d"
STATES = ROOT / "data/label_datasets/hpo_v3_1_d2/rules/s5_fx_site/rule_states.csv"
STATES_SHA = "ce4473c71f3ea100db021805750c6bb9038bc7b0e56a143c81516edf2d29e77b"
OUT = ROOT / "data/label_releases/hpo_v31_d2_state_rule"
RULE = {"BORD": (0.85, ["Effusion", "MCL", "Medial OA"]),
        "BELOW": (0.15, ["Effusion", "MCL", "ACL", "Contusion"])}

raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA
assert hashlib.sha256(STATES.read_bytes()).hexdigest() == STATES_SHA
d = json.loads(raw)
T = d["targets_in_order"]
states = {r["StudyInstanceUID"]: r for r in csv.DictReader(STATES.open(encoding="utf-8"))}
moved = {f"{s}->{v} {t}": 0 for s, (v, ts) in RULE.items() for t in ts}
missing = 0
for r in d["labels"]:
    st = states.get(r["study_id"])
    if st is None:
        missing += 1; continue
    for s, (v, ts) in RULE.items():
        for t in ts:
            if st[t] == s:
                r["probabilities"][T.index(t)] = v; moved[f"{s}->{v} {t}"] += 1
assert missing == 0, missing
d["version"] = "hpo_v3_1_d2_state_rule"
d["note"] = (f"Derived from hpo_v3_1_d2_structured (sha256 {SRC_SHA}) using D2 rule states (s5_fx_site, sha256 "
             f"{STATES_SHA}): {json.dumps({s: {'to': v, 'targets': ts} for s, (v, ts) in RULE.items()})}; weights and "
             f"other cells unchanged. Fixed rule on text-derived labels, no model. Cells moved (all {len(d['labels'])} "
             f"studies): {json.dumps(moved)}. Original note: " + d["note"])
d["derived_from_sha256"] = SRC_SHA
OUT.mkdir(parents=True, exist_ok=True)
f = OUT / "soft_labels_hpo_v3_1_d2_state_rule.json"
f.write_text(json.dumps(d) + "\n")
print(f, hashlib.sha256(f.read_bytes()).hexdigest(), json.dumps(moved, indent=1))
