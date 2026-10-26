"""PUB-KD-E targets: teacher ORDER from one public teacher, VALUE scale from the other (labelsets audit 2026-10-20).
The Raptor targets (ra) are soft (5% of cells < 0.05 or > 0.95), the full-blend targets (A) near-hard (73%); their
orders agree at Spearman 0.95. Swapping the value scale separates "better teacher order" from "target sharpness":
  fullorder_rapvalues   per finding, the k-th lowest full-blend target gets the k-th lowest Raptor target value
  raporder_fullvalues   the reverse
  fullorder_rapvalues_mix73   fullorder_rapvalues, 0.7 / 0.3 with label-v2 where label-v2 has weight > 0 (as B's mix73)
Ties in the order source are broken by the study's position in the source file (stable), as in build_pubkd_targets.py.
QC: per finding, Spearman with the order source = 1 and value multiset = the value source's (sorted values identical).
Usage: python3 -I build_pubkd_swap_targets.py <pubkd dir: pubkd_targets_{pure,raptor_pure}.json> <label-v2 json>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

D, LV = Path(sys.argv[1]), Path(sys.argv[2])
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


def load(p):
    d = json.loads(p.read_text()); assert d["targets_in_order"] == T
    return [r["study_id"] for r in d["labels"]], np.array([r["probabilities"] for r in d["labels"]], float)


ids, full = load(D / "pubkd_targets_pure.json")
ids_r, rap = load(D / "pubkd_targets_raptor_pure.json")
assert ids == ids_r, "the two target files list different studies / order"
rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable")


def swap(order_src, value_src):
    out = np.empty_like(order_src)
    for j in range(12):
        out[np.argsort(order_src[:, j], kind="stable"), j] = np.sort(value_src[:, j])
        assert np.corrcoef(rank(out[:, j]), rank(order_src[:, j]))[0, 1] > 0.9999   # ties in the values only
        assert np.array_equal(np.sort(out[:, j]), np.sort(value_src[:, j]))
    return out


lab = {r["study_id"]: (np.array([.5 if v is None else v for v in r["probabilities"]], float), np.array(r["weights"], float))
       for r in json.loads(LV.read_text())["labels"]}
sharp = lambda y: f"extreme (< .05 or > .95) {((y < .05) | (y > .95)).mean():.3f}, mean |y - .5| {np.abs(y - .5).mean():.3f}"
pr(f"studies {len(ids)}; full: {sharp(full)}; raptor: {sharp(rap)}")
for name, y, mix in (("fullorder_rapvalues", swap(full, rap), None), ("raporder_fullvalues", swap(rap, full), None),
                     ("fullorder_rapvalues_mix73", swap(full, rap), 0.3)):
    rows = []
    for i, s in enumerate(ids):
        v = y[i].copy()
        if mix and s in lab:
            yl, wl = lab[s]; v = np.where(wl > 0, (1 - mix) * v + mix * yl, v)
        rows.append({"study_id": s, "probabilities": [round(float(x), 6) for x in v], "weights": [1.0] * 12})
    Y = np.array([r["probabilities"] for r in rows])
    sp_f = np.mean([np.corrcoef(rank(Y[:, j]), rank(full[:, j]))[0, 1] for j in range(12)])
    sp_r = np.mean([np.corrcoef(rank(Y[:, j]), rank(rap[:, j]))[0, 1] for j in range(12)])
    out = D / f"pubkd_targets_{name}.json"
    out.write_text(json.dumps({"targets_in_order": T, "source": f"PUB-KD-E {name} (scripts/build_pubkd_swap_targets.py)",
                               "labels": rows}))
    pr(f"wrote {out.name}: {len(rows)} studies, {sharp(Y)}; mean Spearman vs full {sp_f:.4f}, vs raptor {sp_r:.4f}; "
       f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
(D / "build_pubkd_swap_targets.txt").write_text("\n".join(L) + "\n")
