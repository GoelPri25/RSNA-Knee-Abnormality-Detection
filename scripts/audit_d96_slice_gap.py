"""Does through-plane subsampling hurt the targets that depend on that plane? (fold 0, CPU only, correlational)

Per fold-0 study: gap between the picked slices of the coronal-fluid slot and of the axial slot (mm, from the d96
manifest geometry). Studies split into gap tertiles; AUC per tertile for 1c and 1e on Steven-v4 confident cells.
Test targets: coronal-dependent per the series ablation (MCL, fracture, contusion) and axial-candidate (PF OA, Baker's).
Controls: sagittal-driven targets (ACL, menisci, lateral OA). If test targets fall with the gap and controls don't,
subsampling is a plausible cause; if everything falls, it's more likely site / scanner quality.
Bootstrap (2,000 study resamples within tertiles) for AUC(low tertile) - AUC(high tertile).

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_slice_gap.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
rows = {json.loads(l)["study_id"]: json.loads(l) for i in range(4)
        for l in (C / f"manifests/manifest_{i}.jsonl").read_text().splitlines()}


def gap(slot):
    g = [x for x in slot.get("geometry", []) if x and x["ipp"] and x["iop"]]
    if not slot["series"] or len(g) < 2:
        return np.nan
    nrm = np.cross(np.array(g[0]["iop"][:3]), np.array(g[0]["iop"][3:]))
    d = np.abs(np.diff([np.dot(x["ipp"], nrm) for x in g])); d = d[d > 1e-3]
    return float(np.median(d)) if len(d) else np.nan


G = {"coronal fluid": np.array([gap(rows[s]["slots"][2]) for s in ids]),
     "axial": np.array([gap(rows[s]["slots"][4]) for s in ids])}
NF = {"coronal fluid": np.array([rows[s]["slots"][2]["n_files"] for s in ids]),
      "axial": np.array([rows[s]["slots"][4]["n_files"] for s in ids])}


def load(p):
    d = json.loads(Path(p).read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
    tg = d.get("targets_in_order", T); a = np.array([m[s] for s in ids], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


P = {"1c": load(C / "run_r50/val_swa.json"), "1e": load(C / "run_r50_pertarget/val_swa.json")}
TESTS = {"coronal fluid": ["MCL", "Fracture", "Contusion"], "axial": ["PF OA", "Baker's", "Effusion"]}
CTRL = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Lateral OA"]
rng = np.random.default_rng(0)
for plane, g in G.items():
    ok = np.isfinite(g); q = np.nanpercentile(g, [100 / 3, 200 / 3])
    grp = np.where(~ok, -1, np.digitize(g, q))
    print(f"\n==== {plane}: gap between picked slices, mm: tertile cuts {q.round(1)}; median per tertile "
          f"{[round(float(np.median(g[grp == k])), 1) for k in range(3)]}; native slices median per tertile "
          f"{[int(np.median(NF[plane][grp == k])) for k in range(3)]}; studies {[(grp == k).sum() for k in range(3)]}")
    out = []
    for t in TESTS[plane] + CTRL:
        k = T.index(t)
        for mname, M in P.items():
            a = []
            for gi in range(3):
                idx = np.where((grp == gi) & mask[:, k])[0]
                a.append(auc(M[idx, k], truth[idx, k]))
            lo_idx = np.where((grp == 0) & mask[:, k])[0]; hi_idx = np.where((grp == 2) & mask[:, k])[0]
            bd = []
            for _ in range(2000):
                l = rng.choice(lo_idx, len(lo_idx)); h = rng.choice(hi_idx, len(hi_idx))
                bd.append(auc(M[l, k], truth[l, k]) - auc(M[h, k], truth[h, k]))
            bd = np.array(bd)
            out.append({"target": t, "role": "TEST" if t in TESTS[plane] else "control", "model": mname,
                        "pos low/mid/high": "/".join(str(int(truth[(grp == gi) & mask[:, k], k].sum())) for gi in range(3)),
                        "AUC low gap": a[0], "mid": a[1], "high gap": a[2], "low-high": a[0] - a[2],
                        "95% CI": f"[{np.nanpercentile(bd, 2.5):+.3f}, {np.nanpercentile(bd, 97.5):+.3f}]"})
    print(pd.DataFrame(out).round(3).to_string(index=False))
    # does the gap track site / scanner? correlation of gap with native in-plane spacing via the other slots' gaps
    sg = np.array([gap(rows[s]["slots"][0]) for s in ids])
    print(f"Spearman(gap {plane}, sagittal-fluid gap) = {pd.Series(g).corr(pd.Series(sg), method='spearman'):.2f} "
          "(high = gap mostly reflects the protocol / site, not our subsampling alone)")
