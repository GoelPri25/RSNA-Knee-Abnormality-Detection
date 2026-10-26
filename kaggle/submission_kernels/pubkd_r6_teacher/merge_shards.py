"""Merge PUB-KD teacher shards (R6 + 2.5D reader on train) into r6_train_all.csv / reader_train_all.csv.
Output: StudyInstanceUID + the 12 TARGETS columns (probabilities). Checks coverage of every train.csv study,
no duplicates, every value finite and in [0, 1]; prints fallbacks and timings from each shard receipt.
Usage: python3 -I merge_shards.py <r6 dir with shard{k}/> <train.csv> [nshard=3] [--partial]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D, TRAIN = Path(sys.argv[1]), Path(sys.argv[2])
NSHARD = int(next((a for a in sys.argv[3:] if a.isdigit()), 3))
PARTIAL = "--partial" in sys.argv
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]
ids = pd.read_csv(TRAIN, usecols=["StudyInstanceUID"])["StudyInstanceUID"]
want = set(ids)
assert len(want) == len(ids), "duplicate ids in train.csv"
ok = True
for name in ("r6", "reader"):
    parts = []
    for k in range(NSHARD):
        f = D / f"shard{k}" / f"{name}_train_shard{k}.csv"
        if not f.exists():
            print(f"{name}: missing {f}"); ok = False; continue
        p = pd.read_csv(f)[["StudyInstanceUID"] + TARGETS]
        expect = set(sorted(want)[k::NSHARD])
        print(f"{name} shard{k}: {len(p)} rows, {len(expect & set(p.StudyInstanceUID))}/{len(expect)} of its shard, "
              f"{len(set(p.StudyInstanceUID) - expect)} outside it")
        parts.append(p)
    if not parts:
        continue
    a = pd.concat(parts, ignore_index=True)
    dup = int(a.StudyInstanceUID.duplicated().sum())
    v = a[TARGETS].to_numpy(float)
    nonfinite = int((~np.isfinite(v)).sum()); outside = int(((v < 0) | (v > 1)).sum())
    miss = sorted(want - set(a.StudyInstanceUID)); extra = sorted(set(a.StudyInstanceUID) - want)
    print(f"{name}: {len(a)} rows, {a.StudyInstanceUID.nunique()} unique / {len(want)} train; missing {len(miss)}, "
          f"extra {len(extra)}, duplicates {dup}, non-finite {nonfinite}, outside [0,1] {outside}")
    if miss[:5]:
        print(f"  missing (first 5): {miss[:5]}")
    good = not (miss or extra or dup or nonfinite or outside)
    ok &= good
    if good or PARTIAL:
        a = a.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(
            [s for s in ids if s in set(a.StudyInstanceUID)]).reset_index()
        out = D / f"{name}_train_all{'_partial' if not good else ''}.csv"
        a.to_csv(out, index=False, float_format="%.9g"); print(f"  wrote {out}")
for k in range(NSHARD):
    f = D / f"shard{k}" / f"receipt_shard{k}.json"
    if f.exists():
        r = json.load(open(f))
        print(f"receipt shard{k}: studies {r['studies']} r6_scored {r['r6_scored']} fallback {len(r['r6_fallback_ids'])} "
              f"{r['r6_fallback_ids'][:5]} r6 {r['r6_seconds'] / 3600:.2f} h ({r['r6_s_per_study_wall']:.1f} s/study wall, "
              f"{r['gpus']} GPU) reader {r['reader']} {r['reader_seconds'] / 60:.1f} min")
print("ALL CHECKS PASS" if ok else "CHECKS FAILED")
sys.exit(0 if ok else 1)
