"""Exp 2a: anatomical position of every d96 slice, from the DICOM geometry the cache builder recorded (numpy only).

Per slot, the picked slices' ImagePositionPatient coordinate along the plane's patient axis (LPS: x = patient left,
y = posterior, z = superior) is rescaled to [0, 1] over the slot's picks and oriented anatomically:
  sagittal: 0 = medial edge, 1 = lateral edge (needs the knee side: right knee medial = +x, left knee medial = -x);
  coronal:  0 = anterior, 1 = posterior;      axial: 0 = superior, 1 = inferior.
A slice gets known = 0 (position 0.5) when its slot is empty, its geometry is missing or degenerate (< 5 mm span),
or it is sagittal and the knee side is unknown. Mirror augmentation does not change these values: it flips pixels,
not anatomy.
"""
from __future__ import annotations

import numpy as np

MAXS = 96
AXIS = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def knee_side(rec, fallback: bool = True) -> str | None:
    """'R' / 'L' from the series Laterality / ImageLaterality tags (first non-empty, all slots). Untagged (50%):
    the sagittal slot's mean x, if at least 20 mm off centre (x < 0 = patient right). On the 2,201 tagged studies
    that rule agrees with the tag 98.5% of the time at 97% coverage (scripts/audit_d96_geometry.py)."""
    for slot in rec["slots"]:
        lat = slot.get("laterality") or {}
        for key in ("Laterality", "ImageLaterality"):
            v = str(lat.get(key, "")).strip().upper()[:1]
            if v in ("R", "L"):
                return v
    if fallback:
        for slot in rec["slots"]:
            geo = slot.get("geometry")
            if slot["plane"] == "Sagittal" and geo and all(g and g.get("ipp") for g in geo):
                x = float(np.mean([g["ipp"][0] for g in geo]))
                return None if abs(x) < 20.0 else ("R" if x < 0 else "L")
    return None


def positions(rec, side: str | None = "auto"):
    """-> pos float32 [96] in [0, 1], known float32 [96] in {0, 1}, for one manifest record."""
    side = knee_side(rec) if side == "auto" else side
    pos = np.full(MAXS, 0.5, np.float32)
    known = np.zeros(MAXS, np.float32)
    for slot in rec["slots"]:
        geo, start, k = slot.get("geometry"), slot["row_start"], slot["k"]
        if not geo or len(geo) != k or any(g is None or g.get("ipp") is None for g in geo):
            continue
        ax = AXIS[slot["plane"]]
        v = np.array([g["ipp"][ax] for g in geo], float)
        span = v.max() - v.min()
        if span < 5.0:
            continue
        p = (v - v.min()) / span                                   # 0 at the most -axis slice
        if slot["plane"] == "Sagittal":
            if side is None:
                continue
            p = 1 - p if side == "R" else p                      # right knee: medial = +x -> 0
        elif slot["plane"] == "Axial":
            p = 1 - p                                              # superior (+z) -> 0
        pos[start:start + k] = p
        known[start:start + k] = 1
    return pos, known
