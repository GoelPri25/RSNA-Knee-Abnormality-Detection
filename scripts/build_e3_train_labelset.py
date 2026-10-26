"""E3 labelset for the teammate (user request 2026-10-17, choice "E3 in-sample, all studies"). CPU, numpy/pandas only.
Merges the two Kaggle shard outputs of kaggle/submission_kernels/e3_train_labelset_shard{0,1} (the LB 0.944 E3 kernel run on
the training DICOMs) into one file per study and flags which predictions are out of fold.

E3's four members trained on every labelled study with fold != 0 (folds 1-5 of folds.csv) and validated on fold 0, so:
  out_of_fold = 1  fold 0 (825) and the 58 gold studies (not in folds.csv, never trained on)
  out_of_fold = 0  folds 1-5: in-sample, the members saw these studies' labels; NOT valid as training labels under the
                   out-of-fold rule, and close to a copy of the training labels
Contains model probabilities only: no report text and no report-derived labels.

Usage: python3 -I scripts/build_e3_train_labelset.py <project root>
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

R = Path(sys.argv[1]); REPO = R / "repository"
SRC = [REPO / f"data/kaggle_runs/e3_train_labelset_shard{k}" for k in (0, 1)]
OUT = REPO / "data/e3_train_labelset"; OUT.mkdir(parents=True, exist_ok=True)
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

parts, members, summaries = [], {}, []
for d in SRC:
    parts.append(pd.read_csv(d / "e3_train_predictions.csv", dtype={"StudyInstanceUID": str}))
    for name, preds in json.loads((d / "individual_predictions.json").read_text()).items():
        members.setdefault(name, {}).update(preds)
    summaries.append(json.loads((d / "inference_summary.json").read_text()))
e3 = pd.concat(parts, ignore_index=True)
assert e3.StudyInstanceUID.is_unique and list(e3.columns[1:13]) == TARGETS
assert e3[TARGETS].ge(0).all().all() and e3[TARGETS].le(1).all().all()
assert all(s["studies_done"] == s["d96"]["studies"] for s in summaries), "a shard did not finish"

series = pd.read_csv(REPO / "data/competition_metadata/train_series.csv", dtype=str)
fold = pd.read_csv(REPO / "data/coatnet_d96_fold0/assets/folds.csv", dtype=str).set_index("study_id").fold
e3["fold"] = e3.StudyInstanceUID.map(fold).fillna("gold")
e3["out_of_fold"] = e3.fold.isin(["0", "gold"]).astype(int)
missing = sorted(set(series.StudyInstanceUID) - set(e3.StudyInstanceUID))

e3.to_csv(OUT / "e3_train_labelset.csv", index=False, float_format="%.8f")
(OUT / "e3_train_labelset_members.json").write_text(json.dumps(members) + "\n")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
info = {"what": "E3 (LB 0.944) equal probability mean of label-v2 r50, teammate 4-network r50, CoAtNet 1b, P1-cnx ConvNeXt-T",
        "kernels": [f"tumis001/rsna-e3-train-labelset-shard{k}" for k in (0, 1)],
        "weights_sha256": summaries[0]["weights_sha256"], "targets": TARGETS,
        "studies": len(e3), "train_series_studies": int(series.StudyInstanceUID.nunique()),
        "not_predicted": missing, "by_fold": e3.fold.value_counts().sort_index().to_dict(),
        "out_of_fold_studies": int(e3.out_of_fold.sum()),
        "warning": "folds 1-5 are in-sample predictions (members trained on them); only fold 0 and gold are out of fold",
        "shard_stats": [s["d96"] for s in summaries],
        "sha256": {p.name: sha(p) for p in (OUT / "e3_train_labelset.csv", OUT / "e3_train_labelset_members.json")}}
(OUT / "e3_train_labelset_info.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps({k: v for k, v in info.items() if k != "shard_stats"}, indent=2))
