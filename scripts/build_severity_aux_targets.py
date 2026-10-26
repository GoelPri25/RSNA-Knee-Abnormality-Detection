"""Exp P1-sev asset (CPU, 2026-10-14): cumulative severity targets for the auxiliary heads, training folds only.

For each of the 12 findings, two extra binary targets from the structured report extraction
(report_structured_labels.csv, `<finding>__status` / `<finding>__severity`):
  >=moderate: 1 = present & severity in {moderate, moderate_to_severe, severe}
              0 = absent / not_mentioned, or present & mild
              masked (weight 0) = present & {mild_to_moderate, unspecified, none, ''}, uncertain, missing
  >=severe:   1 = present & severe
              0 = absent / not_mentioned, or present & {mild, mild_to_moderate, moderate}
              masked = present & {moderate_to_severe, unspecified, none, ''}, uncertain, missing
Only studies in folds.csv with fold != 0 are written; fold 0 (validation) and gold studies get no row, so the runner
gives them weight 0. The validation labels and the main training labels are untouched.

Usage (container):
  podman run --rm --network=none -v "$P":/proj localhost/rsna-label-ds python -I \
      /proj/repository/scripts/build_severity_aux_targets.py /proj
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(sys.argv[1]); A = R / "repository/data/coatnet_d96_fold0/assets"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
fold = {r["study_id"]: r["fold"] for r in csv.DictReader((A / "folds.csv").open())}
rows = {r["StudyInstanceUID"]: r for r in csv.DictReader(
    (R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv").open(encoding="utf-8"))}
MOD = {"moderate", "moderate_to_severe", "severe"}


def cell(status, sev, level):
    if status in ("absent", "not_mentioned"):
        return 0.0
    if status != "present":
        return None
    if level == "mod":
        return 1.0 if sev in MOD else 0.0 if sev == "mild" else None
    return 1.0 if sev == "severe" else 0.0 if sev in ("mild", "mild_to_moderate", "moderate") else None


out, n_pos, n_obs = [], Counter(), Counter()
train_ids = sorted(s for s, f in fold.items() if f != "0")
for s in train_ids:
    r = rows.get(s)
    vals = [None] * 24
    if r is not None:
        for k, t in enumerate(T):
            st, sv = r[f"{key(t)}__status"], r[f"{key(t)}__severity"]
            vals[k], vals[12 + k] = cell(st, sv, "mod"), cell(st, sv, "sev")
    for j, v in enumerate(vals):
        if v is not None:
            n_obs[j] += 1; n_pos[j] += int(v == 1.0)
    out.append({"study_id": s, "probabilities": vals, "weights": [0.0 if v is None else 1.0 for v in vals]})

names = [f"{t} >=moderate" for t in T] + [f"{t} >=severe" for t in T]
doc = {"targets_in_order": names, "source": "report_structured_labels.csv status/severity, training folds only",
       "labels": out}
p = A / "severity_aux_targets.json"
p.write_text(json.dumps(doc) + "\n")
print(f"training studies {len(out)} (missing structured row: {sum(s not in rows for s in train_ids)})")
for j, nm in enumerate(names):
    print(f"  {nm:28s} observed {n_obs[j]:5d}  positive {n_pos[j]:4d}")
print("sha256", hashlib.sha256(p.read_bytes()).hexdigest())
