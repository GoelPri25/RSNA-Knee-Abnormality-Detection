"""Teacher v2 targets (KF-CNXS ledger row, 2026-10-22): teacher v1 (FB-F: full public blend ORDER, Raptor VALUE scale,
0.7 / 0.3 with label-v2 where label-v2 has weight > 0) with the order source changed to
    order = (1 - w) * percentile(full public blend) + w * percentile(team)
per finding over the FB-F pool (competition + fastMRI). Team = our out-of-fold ConvNeXt-S members: competition study in
fold k -> the member that held fold k out (exactly one per study, asserted); fastMRI exam -> mean probability of the
fastMRI member files given (no competition labels involved there). Studies without a team value (gold, missing
fastMRI) keep percentile(full). Values, ties and the label mix as build_pubkd_swap_targets.py (Raptor values assigned by
the new order; stable ties by pool position).
QC: per finding Spearman vs teacher v1 (fbf mix73) and vs team; team coverage per source.
Usage: python3 -I build_teacher_v2.py <pool dir> <label-v2 json> <out json> <w> --oof f1.json ... --fm m1.json ...
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
a = sys.argv[1:]
P, LV, OUT, W = Path(a[0]), Path(a[1]), Path(a[2]), float(a[3])
oof = a[a.index("--oof") + 1:a.index("--fm")] if "--fm" in a else a[a.index("--oof") + 1:]
fm = a[a.index("--fm") + 1:] if "--fm" in a else []
assert 0 <= W <= 1 and oof


def load(p, key):
    d = json.loads(Path(p).read_text()); assert d["targets_in_order"] == T, p
    return {r["study_id"]: np.array(r["probabilities"], float) for r in d[key]}


full = json.loads((P / "pubkd_targets_pure.json").read_text())["labels"]
ids = [r["study_id"] for r in full]; full = np.array([r["probabilities"] for r in full], float)
rp = json.loads((P / "pubkd_targets_raptor_pure.json").read_text())["labels"]
assert [r["study_id"] for r in rp] == ids; rap = np.array([r["probabilities"] for r in rp], float)
v1 = load(P / "pubkd_targets_fullorder_rapvalues_mix73.json", "labels")
team = {}
for f in oof:
    for s, v in load(f, "predictions").items():
        assert s not in team and not s.startswith("fm_"), f"{s} in two out-of-fold files"
        team[s] = v
fms = [load(f, "predictions") for f in fm]
for s in set().union(*fms) if fms else []:
    vs = [m[s] for m in fms if s in m]; team[s] = np.mean(vs, 0)
N = len(ids); has = np.array([s in team for s in ids]); isfm = np.array([s.startswith("fm_") for s in ids])
print(f"pool {N} (fastMRI {isfm.sum()}); team values: competition {has[~isfm].sum()}/{(~isfm).sum()}, "
      f"fastMRI {has[isfm].sum()}/{isfm.sum()}; w {W}", flush=True)
rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable")
tv = np.array([team.get(s, np.full(12, np.nan)) for s in ids])
y = np.empty_like(full); sp = []
for j in range(12):
    order = rank(full[:, j]) / (N - 1)
    if W > 0:
        pt = np.full(N, np.nan); k = np.where(has)[0]
        pt[k] = rank(tv[k, j]) / max(1, len(k) - 1)
        order = np.where(has, (1 - W) * order + W * pt, order)
    y[np.argsort(order, kind="stable"), j] = np.sort(rap[:, j])
lab = {r["study_id"]: (np.array([.5 if v is None else v for v in r["probabilities"]], float), np.array(r["weights"], float))
       for r in json.loads(LV.read_text())["labels"]}
rows = []
for i, s in enumerate(ids):
    v = y[i].copy()
    if s in lab:
        yl, wl = lab[s]; v = np.where(wl > 0, 0.7 * v + 0.3 * yl, v)
    rows.append({"study_id": s, "probabilities": [round(float(x), 6) for x in v], "weights": [1.0] * 12})
Y = np.array([r["probabilities"] for r in rows]); V1 = np.array([v1[s] for s in ids])
sp_v1 = [np.corrcoef(rank(Y[:, j]), rank(V1[:, j]))[0, 1] for j in range(12)]
k = np.where(has)[0]
sp_tm = [np.corrcoef(rank(Y[k, j]), rank(tv[k, j]))[0, 1] for j in range(12)]
sp_v1t = [np.corrcoef(rank(V1[k, j]), rank(tv[k, j]))[0, 1] for j in range(12)]
print("Spearman per finding: v2 vs v1 " + json.dumps({t: round(x, 4) for t, x in zip(T, sp_v1)}), flush=True)
print(f"mean Spearman: v2 vs v1 {np.mean(sp_v1):.4f}; on team-covered studies: v2 vs team {np.mean(sp_tm):.4f}, "
      f"v1 vs team {np.mean(sp_v1t):.4f}; max |v2 - v1| {np.abs(Y - V1).max():.4f}, mean {np.abs(Y - V1).mean():.4f}", flush=True)
OUT.write_text(json.dumps({"targets_in_order": T, "labels": rows,
                           "source": f"teacher v2 (scripts/fm_teacher/build_teacher_v2.py) w {W}, oof {len(oof)} files, fm {len(fm)} files"}))
print(f"wrote {OUT} {len(rows)} studies sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}", flush=True)
