"""Label v3a (2026-10-22): teacher v1 (FB-F mix73 = 0.7 x full-blend-order / Raptor-value teacher + 0.3 x label-v2 where
label-v2 has weight > 0) with the 0.3 report mix removed on competition cells where gold-58 shows the report category is
unreliable for the host label: status uncertain, or present with severity mild / mild_to_moderate. Those cells take the
teacher-only value (pubkd_targets_fullorder_rapvalues.json); every other cell (and every fastMRI exam) keeps teacher v1.
QC: the mix identity v1 = 0.7 teacher + 0.3 label-v2 is asserted on every mixed cell; counts and |change| printed.
Report categories are read locally (report_structured_labels.csv, status / severity columns only; no report text).
Usage: python3 -I build_v3a_targets.py <pool dir> <label-v2 json> <report_structured_labels.csv> <out json>"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
P, LV, ST, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
load = lambda f: {r["study_id"]: np.array(r["probabilities"], float) for r in json.loads((P / f).read_text())["labels"]}
v1 = json.loads((P / "pubkd_targets_fullorder_rapvalues_mix73.json").read_text())
assert v1["targets_in_order"] == T
pure = load("pubkd_targets_fullorder_rapvalues.json")
lab = {r["study_id"]: (np.array([.5 if v is None else v for v in r["probabilities"]], float), np.array(r["weights"], float))
       for r in json.loads(LV.read_text())["labels"]}
st = {r["StudyInstanceUID"]: {t: (r[f"{key(t)}__status"], r[f"{key(t)}__severity"]) for t in T}
      for r in csv.DictReader(ST.open(encoding="utf-8"))}
rows, n, ch, d, worst = [], 0, 0, [], 0.0
per = np.zeros(12, int)
for r in v1["labels"]:
    s, v = r["study_id"], np.array(r["probabilities"], float)
    if s in lab and not s.startswith("fm_"):
        yl, wl = lab[s]; mixed = wl > 0
        worst = max(worst, float(np.abs(v[mixed] - (0.7 * pure[s][mixed] + 0.3 * yl[mixed])).max(initial=0)))
        if s in st:
            for j, t in enumerate(T):
                n += 1; stt, sv = st[s][t]
                if mixed[j] and (stt == "uncertain" or (stt == "present" and sv in ("mild", "mild_to_moderate"))):
                    d.append(abs(pure[s][j] - v[j])); v[j] = pure[s][j]; ch += 1; per[j] += 1
    rows.append({"study_id": s, "probabilities": [round(float(x), 6) for x in v], "weights": r["weights"]})
assert worst < 1e-4, f"mix identity off by {worst}"
d = np.array(d)
print(f"mix identity max error {worst:.2e}; competition cells {n}, changed {ch} ({ch / n:.1%}); |change| mean {d.mean():.3f}, "
      f"median {np.median(d):.3f}, >0.1 {(d > 0.1).mean():.1%}")
print("changed per finding " + json.dumps(dict(zip(T, per.tolist()))))
OUT.write_text(json.dumps({"targets_in_order": T, "labels": rows,
                           "source": "label v3a (scripts/fm_teacher/build_v3a_targets.py): teacher v1 mix73, report mix removed on uncertain / mild / mild_to_moderate cells"}))
print(f"wrote {OUT} {len(rows)} studies sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
