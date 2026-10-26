"""Exp 2a pre-launch check: how many d96 slices get an anatomical position, and whether the orientation is sane.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_geometry.py /proj/repository
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

R = Path(sys.argv[1])
sys.path.insert(0, str(R / "scripts"))
from d96_positions import AXIS, knee_side, positions  # noqa: E402

rows = [json.loads(l) for i in range(4)
        for l in (R / f"data/coatnet_d96_fold0/manifests/manifest_{i}.jsonl").read_text().splitlines()]
NAMES = ["Sag fluid", "Sag non-fluid", "Cor fluid", "Cor non-fluid", "Axial"]
print(f"{len(rows)} studies")

tags = Counter(knee_side(r, fallback=False) for r in rows)
sides = Counter(knee_side(r) for r in rows)
print(f"knee side from tags: {dict(tags)}  ({tags[None] / len(rows):.1%} unknown)")
print(f"knee side, tags + x fallback: {dict(sides)}  ({sides[None] / len(rows):.1%} unknown)")

known = np.stack([positions(r)[1] for r in rows])
mask = np.stack([np.array(r["mask"], bool) for r in rows])
start = np.cumsum([0, 26, 22, 18, 12])
for s, (a, k) in enumerate(zip(start, [26, 22, 18, 12, 18])):
    m = mask[:, a:a + k]
    print(f"[{NAMES[s]}] real slices with a position: {(known[:, a:a + k] * m).sum() / max(m.sum(), 1):.1%}")
print(f"all real slices with a position: {(known * mask).sum() / mask.sum():.1%}")

# Plane sanity: the slice normal should point along the slot's patient axis.
off = Counter()
for r in rows:
    for slot in r["slots"]:
        g = (slot.get("geometry") or [None])[0]
        if g and g.get("iop"):
            n = np.cross(g["iop"][:3], g["iop"][3:])
            off[(slot["plane"], int(np.argmax(np.abs(n))) == AXIS[slot["plane"]])] += 1
print("slice normal along the slot's patient axis:", {f"{p} {ok}": n for (p, ok), n in sorted(off.items())})

# Side vs sagittal x: does the tagged side agree with where the knee sits (x < 0 = patient right)?
agree = Counter()
for r in rows:
    side, g = knee_side(r, fallback=False), (r["slots"][0].get("geometry") or [])
    if side and g and all(x and x.get("ipp") for x in g):
        x = np.mean([x["ipp"][0] for x in g])
        agree[(side, "x<0" if x < 0 else "x>0")] += 1
print("tagged side vs mean sagittal x:", dict(sorted(agree.items())))
