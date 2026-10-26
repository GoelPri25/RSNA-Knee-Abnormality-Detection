"""P3 v2 student labels (CPU; ledger P3v2-S): label-v2 mixed with cross-fitted CK-A ConvNeXt teacher predictions.

For every labelled study and target with weight > 0:  y' = (1 - alpha) * y + alpha * t,  weights unchanged; cells
with weight 0 are left exactly as they are. alpha is either one number (control) or set per cell by the report rule
state (HPO-v3.1 s5_fx_site rule_states.csv): SIL 0.7 (report silent: label is a prior, the image knows more);
BELOW / BORD / HEDGE / LIKELY 0.5 (uncertain wording); POS / NEG 0.2 (report explicit). t = the CK-A teacher
(run_cnxa_teacher{k}, label-v2, split k held out):
  - training studies (folds 1-5): the one teacher whose held-out split holds the study (oof_swa.json);
  - fold-0 studies: the mean of the 4 teachers' val_swa.json (no teacher trained on fold 0). These rows are only
    trained on by an all-data student; a fold-0-held-out student never reads them. validation_labels.json, the rulers
    and gold are separate files and are not touched.
Checks: label, split and rule-state hashes; every training study in exactly one oof file, in its own split; every
labelled study has a rule state for every target; teacher predictions finite and in [0, 1].
Writes repository/data/student_labels/soft_labels_student_v2_{state|aNNN}.json + summary_v2_{...}.txt.

Usage (CPU): python3 -I build_student_labels_v2.py <project root> state|<alpha>
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1]) / "repository/data"; MODE = sys.argv[2]
A = R / "coatnet_d96_fold0/assets"; C = R / "coatnet_d96_fold0"; OUT = R / "student_labels"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
LAB = A / "soft_labels_final_per_target_latD2.json"; SPL = A / "teacher_splits.csv"
RS = R / "label_datasets/hpo_v3_1_d2/rules/s5_fx_site/rule_states.csv"
assert sha(LAB).startswith("dd0ab9ab") and sha(SPL).startswith("7cb597a7") and sha(RS).startswith("ce4473c7")
STATE_ALPHA = {"SIL": 0.7, "BELOW": 0.5, "BORD": 0.5, "HEDGE": 0.5, "LIKELY": 0.5, "POS": 0.2, "NEG": 0.2}
lab = json.loads(LAB.read_text()); T = lab["targets_in_order"]
fold = {r["study_id"]: r["fold"] for r in csv.DictReader((A / "folds.csv").open())}
split = {r["study_id"]: int(r["teacher_split"]) for r in csv.DictReader(SPL.open())}
state = {r["StudyInstanceUID"]: [r[t] for t in T] for r in csv.DictReader(RS.open())}
assert all(s in state and all(v in STATE_ALPHA for v in state[s]) for s in (r["study_id"] for r in lab["labels"]))


def load(p):
    d = json.loads(p.read_text()); ix = [d["targets_in_order"].index(t) for t in T]
    out = {r["study_id"]: np.array(r["probabilities"], float)[ix] for r in d["predictions"]}
    assert all(np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1 for v in out.values()), p
    return out


teach, f0 = {}, []
for k in range(4):
    run = C / f"run_cnxa_teacher{k}"
    o = load(run / "oof_swa.json")
    assert all(split[s] == k for s in o) and not set(o) & set(teach), f"teacher {k}: split mismatch / overlap"
    teach.update(o); f0.append(load(run / "val_swa.json"))
assert set(teach) == set(split), f"oof covers {len(teach)} of {len(split)} training studies"
assert all(set(f) == set(f0[0]) for f in f0)
teach.update({s: np.mean([f[s] for f in f0], 0) for s in f0[0]})

alpha_of = (lambda s, k: STATE_ALPHA[state[s][k]]) if MODE == "state" else (lambda s, k, a=float(MODE): a)
tag = "state" if MODE == "state" else f"a{int(round(float(MODE) * 100)):03d}"
rows, delta, n = [], np.zeros(len(T)), np.zeros(len(T))
by_state = {k: [0.0, 0, 0.0] for k in STATE_ALPHA}                 # sum |y' - y|, cells, sum (t - y) (training rows)
for r in lab["labels"]:
    s = r["study_id"]; t = teach[s]; probs = list(r["probabilities"])
    for k, (p, w) in enumerate(zip(probs, r["weights"])):
        if p is not None and w > 0:
            a = alpha_of(s, k); probs[k] = float((1 - a) * p + a * t[k])
            delta[k] += abs(probs[k] - p); n[k] += 1
            if fold[s] != "0":
                b = by_state[state[s][k]]; b[0] += abs(probs[k] - p); b[1] += 1; b[2] += t[k] - p
    rows.append({**r, "probabilities": probs})
assert len(rows) == len(lab["labels"]) and {r["study_id"] for r in rows} <= set(fold)
OUT.mkdir(parents=True, exist_ok=True)
name = f"soft_labels_student_v2_{tag}.json"
rule = "alpha by rule state " + json.dumps(STATE_ALPHA) if MODE == "state" else f"alpha {MODE}"
doc = {**{k: v for k, v in lab.items() if k != "labels"}, "targets_in_order": T,
       "note": f"P3 v2 student labels: label-v2 (sha dd0ab9ab...) mixed with cross-fitted CK-A ConvNeXt teachers "
               f"(splits sha 7cb597a7..., rule states sha ce4473c7...), {rule}; weights unchanged", "labels": rows}
(OUT / name).write_text(json.dumps(doc) + "\n")
lines = [f"{name}: sha256 {sha(OUT / name)}", f"{rule}; studies {len(rows)} "
         f"(training {sum(fold[r['study_id']] != '0' for r in rows)}, fold 0 {sum(fold[r['study_id']] == '0' for r in rows)})",
         "mean |y' - y| per target: " + ", ".join(f"{t} {d / max(c, 1):.3f}" for t, d, c in zip(T, delta, n)),
         "training cells by state (cells, mean |y' - y|, mean teacher - label): " + ", ".join(
             f"{k} {b[1]} {b[0] / max(b[1], 1):.3f} {b[2] / max(b[1], 1):+.3f}" for k, b in by_state.items())]
(OUT / f"summary_v2_{tag}.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
