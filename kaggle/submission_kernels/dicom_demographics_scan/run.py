"""DICOM demographics / acquisition header scan (CPU, headers only, 2026-10-13).

Question: do the competition DICOMs (train AND test) keep patient age, sex, size/weight and acquisition tags? If age
survives in test, it is a strong prior for OA, degenerative meniscal tears and Baker's cysts that no model uses yet.

Reads one header per series (stop_before_pixels) for every train and test series. Writes per-study values of a fixed
non-identifying tag list (no names, IDs, dates of birth) and presence rates per split.
"""
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pydicom


def pick(c):
    return next(p for p in c if p.exists())


DATA = pick([Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
             Path("/kaggle/input/rsna-knee-abnormality-detection")])
OUT = Path("/kaggle/working")
TAGS = ["PatientAge", "PatientSex", "PatientSize", "PatientWeight", "Manufacturer", "ManufacturerModelName",
        "MagneticFieldStrength", "BodyPartExamined", "Laterality", "ImageLaterality", "StudyDescription",
        "SeriesDescription", "InstitutionName", "StationName", "SoftwareVersions", "ScanningSequence",
        "SequenceVariant", "RepetitionTime", "EchoTime", "SliceThickness"]


def header(args):
    split, study, series = args
    files = sorted((DATA / f"{split}_series" / study / series).glob("*.dcm"))
    if not files:
        return split, study, series, None
    try:
        h = pydicom.dcmread(files[0], stop_before_pixels=True)
    except Exception:
        return split, study, series, None
    out = {}
    for t in TAGS:
        v = getattr(h, t, None)
        out[t] = "" if v is None else str(v).strip()
    return split, study, series, out


jobs = []
for split in ("train", "test"):
    p = DATA / f"{split}_series.csv"
    if p.exists():
        with p.open(newline="", encoding="utf-8-sig") as f:
            jobs += [(split, r["StudyInstanceUID"], r["SeriesInstanceUID"]) for r in csv.DictReader(f)]
with ThreadPoolExecutor(16) as pool:
    res = list(pool.map(header, jobs))

per_study = defaultdict(dict)
for split, study, series, h in res:
    if h is None:
        continue
    d = per_study[(split, study)]
    for t, v in h.items():
        d.setdefault(t, Counter())[v] += 1
rows = []
for (split, study), d in per_study.items():
    row = {"split": split, "StudyInstanceUID": study}
    for t in TAGS:
        c = d.get(t, Counter()); nonempty = [(v, n) for v, n in c.most_common() if v]
        row[t] = nonempty[0][0] if nonempty else ""
    rows.append(row)
with (OUT / "study_header_tags.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, ["split", "StudyInstanceUID"] + TAGS); w.writeheader(); w.writerows(rows)

summary = {}
for split in ("train", "test"):
    rs = [r for r in rows if r["split"] == split]
    summary[split] = {"studies": len(rs),
                      "present_share": {t: round(sum(bool(r[t]) for r in rs) / max(len(rs), 1), 3) for t in TAGS},
                      "top_values": {t: Counter(r[t] for r in rs).most_common(8) for t in
                                     ("PatientAge", "PatientSex", "Manufacturer", "MagneticFieldStrength", "BodyPartExamined")}}
(OUT / "header_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
