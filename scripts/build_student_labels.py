"""Phase 3 student labels (CPU): mix the per-target training labels with cross-fitted teacher predictions.

For every labelled study and target with weight > 0:  y' = (1 - alpha) * y + alpha * t,  weights unchanged; cells
with weight 0 (or no label) are left exactly as they are. t = mean over the teacher architectures (r50, cnx) of
  - training studies (folds 1-5): the one teacher of that architecture whose held-out split holds the study
    (run_{arch}_teacher{k}/oof_swa.json; k must match data/teacher_splits/teacher_splits.csv);
  - fold-0 studies: the mean of that architecture's 4 teachers' val_swa.json (no teacher trained on fold 0). These rows
    are only trained on by an all-data student; a fold-0-held-out student never reads them. The fold-0 reference
    (validation_labels.json), the rulers and gold are separate files and are not touched.
Checks: label-file and split-file hashes; every training study in exactly one oof file per architecture, in its own
split; teacher predictions finite and in [0, 1]. Writes repository/data/student_labels/soft_labels_student_a{alpha}.json
(same schema as the input) + summary.txt with sha256 and per-target mean |y' - y|.

Usage (container):
  podman run --rm --network=none -v "$P":/proj localhost/rsna-label-ds python -I /proj/repository/scripts/build_student_labels.py /proj 0.5
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1]) / "repository/data"; ALPHA = float(sys.argv[2])
A = R / "coatnet_d96_fold0/assets"; C = R / "coatnet_d96_fold0"; OUT = R / "student_labels"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
LAB = A / "soft_labels_final_per_target_any_model.json"; SPL = A / "teacher_splits.csv"
assert sha(LAB).startswith("d40803d0") and sha(SPL).startswith("7cb597a7")
lab = json.loads(LAB.read_text()); T = lab["targets_in_order"]
fold = {r["study_id"]: r["fold"] for r in csv.DictReader((A / "folds.csv").open())}
split = {r["study_id"]: int(r["teacher_split"]) for r in csv.DictReader(SPL.open())}


def load(p):
    d = json.loads(p.read_text()); ix = [d["targets_in_order"].index(t) for t in T]
    out = {r["study_id"]: np.array(r["probabilities"], float)[ix] for r in d["predictions"]}
    assert all(np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1 for v in out.values()), p
    return out


teach = {}
for arch in ("r50", "cnx"):
    oof, f0 = {}, []
    for k in range(4):
        run = C / f"run_{arch}_teacher{k}"
        o = load(run / "oof_swa.json")
        assert all(split[s] == k for s in o) and not set(o) & set(oof), f"{arch} teacher {k}: split mismatch / overlap"
        oof.update(o); f0.append(load(run / "val_swa.json"))
    assert set(oof) == set(split), f"{arch}: oof covers {len(oof)} of {len(split)} training studies"
    assert all(set(f) == set(f0[0]) for f in f0)
    oof.update({s: np.mean([f[s] for f in f0], 0) for s in f0[0]})
    teach[arch] = oof

rows, delta, n = [], np.zeros(len(T)), np.zeros(len(T))
for r in lab["labels"]:
    s = r["study_id"]; t = (teach["r50"][s] + teach["cnx"][s]) / 2
    probs = list(r["probabilities"])
    for k, (p, w) in enumerate(zip(probs, r["weights"])):
        if p is not None and w > 0:
            probs[k] = float((1 - ALPHA) * p + ALPHA * t[k]); delta[k] += abs(probs[k] - p); n[k] += 1
    rows.append({**r, "probabilities": probs})
assert len(rows) == len(lab["labels"]) and {r["study_id"] for r in rows} <= set(fold)
OUT.mkdir(parents=True, exist_ok=True)
name = f"soft_labels_student_a{int(round(ALPHA * 100)):03d}.json"
doc = {**{k: v for k, v in lab.items() if k != "labels"}, "targets_in_order": T,
       "note": f"Phase 3 student labels: (1 - {ALPHA}) * per-target labels (sha d40803d0...) + {ALPHA} * mean(r50, cnx) "
               f"cross-fitted teacher predictions (teacher splits sha 7cb597a7...); weights unchanged", "labels": rows}
(OUT / name).write_text(json.dumps(doc) + "\n")
lines = [f"{name}: sha256 {sha(OUT / name)}", f"alpha {ALPHA}; studies {len(rows)} "
         f"(training {sum(fold[r['study_id']] != '0' for r in rows)}, fold 0 {sum(fold[r['study_id']] == '0' for r in rows)})",
         "mean |y' - y| per target (weighted cells): " + ", ".join(f"{t} {d / max(c, 1):.3f}" for t, d, c in zip(T, delta, n)),
         "teacher agreement r50 vs cnx on training studies (Spearman per target): " + ", ".join(
             f"{t} {np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]:.2f}" for t, a, b in zip(
                 T, *[np.array([teach[x][s] for s in sorted(split)]).T for x in ("r50", "cnx")]))]
(OUT / f"summary_a{int(round(ALPHA * 100)):03d}.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
