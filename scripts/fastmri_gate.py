"""SURPASS-F Job 1 gate (revised before any full-build data was seen, coordinator 2026-10-20: the original ">= 90% fill
all 5 slots" asked more than our own cache has, which fills all 5 slots in only 71.8% of studies).
(a) decode errors < 1% of picked slices (from the builder audits);
(b) >= 95% of fastMRI studies fill the sagittal-fluid slot and >= 90% have >= 3 non-empty slots;
(c) per-slot fill rates printed next to the competition cache's (descriptive);
(d) montage of 4 random studies (middle row of each slot) -> <fm root>/gate_montage.png, for a look (not a stop rule).
Exit 1 if (a) or (b) fails, so the job stops before the teacher.
Usage: python -I fastmri_gate.py <competition d96 root> <n comp shards> <fastMRI d96 root> <n fm shards> <audit tags...>
"""
import json
import os
import random
import sys
from pathlib import Path

if os.environ.get("FM_PYLIBS"):
    sys.path.insert(0, os.environ["FM_PYLIBS"])
import cv2  # noqa: E402
import numpy as np  # noqa: E402

SLOTS = [(0, 26), (26, 22), (48, 18), (66, 12), (78, 18)]
NAMES = ["sag fluid", "sag non-fluid", "cor fluid", "cor non-fluid", "axial"]


def records(root, nshard):
    out = []
    for k in range(nshard):
        d = Path(root) / f"shard_{k}"
        out += [(d, json.loads(line)) for line in (d / "manifest.jsonl").read_text().splitlines()]
    return out


def fill(recs):
    f = np.array([[r["slots"][j]["series"] is not None for j in range(5)] for _, r in recs])
    return f.mean(0), (f.sum(1) >= 3).mean(), f.all(1).mean()


def main():
    comp = records(sys.argv[1], int(sys.argv[2]))
    fm = records(sys.argv[3], int(sys.argv[4]))
    audits = [json.loads((Path(sys.argv[3]) / f"audit_{t}.json").read_text()) for t in sys.argv[5:]]
    n = sum(a["studies"] for a in audits)
    de = sum(a["decode_errors"] for a in audits)
    sl = sum(a["slices"] for a in audits)
    de_frac = de / max(sl, 1)
    failed = sum(len(a["failed"]) for a in audits)
    cs, c3, c5 = fill(comp)
    fs, f3, f5 = fill(fm)
    print(f"studies: fastMRI manifests {len(fm)} (audits {n}, failed {failed}), competition {len(comp)}")
    print(f"{'slot':16s} {'fastMRI':>8s} {'comp':>8s}")
    for j in range(5):
        print(f"{NAMES[j]:16s} {fs[j]:8.4f} {cs[j]:8.4f}")
    print(f"{'>= 3 slots':16s} {f3:8.4f} {c3:8.4f}\n{'all 5 slots':16s} {f5:8.4f} {c5:8.4f}")
    ok_a = de_frac < 0.01
    ok_b = fs[0] >= 0.95 and f3 >= 0.90
    print(f"GATE (a) decode errors {de} / {sl} picked slices frac {de_frac:.5f} < 0.01: {'PASS' if ok_a else 'FAIL'}")
    print(f"GATE (b) sag-fluid fill {fs[0]:.4f} >= 0.95 and >= 3 slots {f3:.4f} >= 0.90: {'PASS' if ok_b else 'FAIL'}")

    random.seed(0)
    mm, rows = {}, []
    for d, r in random.sample(fm, min(4, len(fm))):
        f = d / r["part"]
        if f not in mm:
            mm[f] = np.load(f, mmap_mode="r")
        tiles = []
        for j, (a, k) in enumerate(SLOTS):
            t = np.asarray(mm[f][r["row_start"] + a + k // 2]).copy()
            desc = str(r["slots"][j]["series"])[:30]
            cv2.putText(t, f"{NAMES[j]}: {desc}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
            tiles.append(t)
        row = np.concatenate(tiles, 1)
        cv2.putText(row, r["study_id"], (4, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        rows.append(row)
    cv2.imwrite(str(Path(sys.argv[3]) / "gate_montage.png"), np.concatenate(rows, 0))
    print(f"montage: {Path(sys.argv[3]) / 'gate_montage.png'}", flush=True)
    sys.exit(0 if ok_a and ok_b and len(fm) > 0 else 1)


if __name__ == "__main__":
    main()
