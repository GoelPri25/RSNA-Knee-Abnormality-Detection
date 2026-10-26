"""SURPASS-F joint teacher targets: the PUB-KD-R Raptor teacher (0.65 pct(384) + 0.35 pct(224), order; logit blend,
values) over the competition studies AND the fastMRI studies, ranked in one pool, so both sources share one target scale.
Left out of the pool: fastMRI duplicates of competition studies (fastmri_dedupe.py), the studies listed in the optional
FM_EXCLUDE json ({"study_ids": [...]}, e.g. fm_misslot.json: series slotted under the wrong fat-sat flag) and the holdout.
Holdout (never trained): whole patients (patient dir = first part of the manifest's "source"), taken in sha256 order
of the patient dir until >= HOLD exams; its teacher values (logit blend, no ranks) are written for the fidelity test.
Outputs (label-file layout, weights 1 on all 12 cells) in <out dir>:
  fm_targets_raptor_pure.json   competition + fastMRI pool targets (pure teacher, as PUB-KD-R ra)
  fm_holdout.json               the holdout's study ids
  fm_holdout_teacher.json       the holdout's teacher probabilities (save_predictions layout)
QC: per finding Spearman of the pooled targets vs ra's targets on the competition studies (expected >= 0.999),
target distribution per source.
Usage: python3 -I build_fm_targets.py <competition teacher dir: raptor{384,224}_*.json> <ra targets json>
       <fastMRI teacher dir> <fastMRI cache dir> <n shards> <dupes json> <out dir> [holdout exams, default 500]
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

CT, RA, FT, FC, NS, DU, O = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]),
                             Path(sys.argv[6]), Path(sys.argv[7]))
HOLD = int(sys.argv[8]) if len(sys.argv) > 8 else 500
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
A224 = 0.35
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


def raptor(d, model):
    out = {}
    for f in sorted(d.glob(f"raptor{model}_*.json")):
        j = json.loads(f.read_text()); assert j["targets"] == T
        for s, v in j["pred"].items():
            assert s not in out
            out[s] = 0.5 * (np.array(v["plain"]) + np.array(v["mirror"]))   # mirror TTA, as PUB-KD-R
    return out


def rank(v):
    """1-based ranks, ties averaged (pandas rank(method="average")); numpy only: the training image has no pandas."""
    v = np.asarray(v, float); o = np.argsort(v, kind="stable"); sv = v[o]
    first = np.r_[True, sv[1:] != sv[:-1]]; grp = np.cumsum(first) - 1
    start = np.flatnonzero(first); end = np.r_[start[1:], len(v)]
    out = np.empty(len(v)); out[o] = ((start + 1 + end) / 2)[grp]
    return out


pct = lambda v: rank(v) / len(v)
spear = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

r384 = {**raptor(CT, "384"), **raptor(FT, "384")}
r224 = {**raptor(CT, "224"), **raptor(FT, "224")}
assert set(r384) == set(r224), "384 / 224 cover different studies"
src = {}
for i in range(NS):
    for line in (FC / f"shard_{i}" / "manifest.jsonl").read_text().splitlines():
        rec = json.loads(line); src[rec["study_id"]] = rec["source"]
fm = sorted(s for s in r384 if s.startswith("fm_"))
comp = sorted(s for s in r384 if not s.startswith("fm_"))
assert set(fm) == set(src), f"teacher / manifest differ: {len(set(fm) ^ set(src))}"
dupes = set(json.loads(DU.read_text())["dupes"])
assert dupes <= set(fm)
EX = os.environ.get("FM_EXCLUDE")
excl = set(json.loads(Path(EX).read_text())["study_ids"]) if EX else set()
assert excl <= set(fm)
pr(f"FM_EXCLUDE {EX} ({sha(Path(EX))[:16] if EX else '-'}): {len(excl)} studies, {len(excl - dupes)} not already dupes")
dupes |= excl                                             # left out exactly like dupes: no targets, not in the holdout
pat = {}
for s in fm:
    if s not in dupes:
        pat.setdefault(src[s].split("/")[0], []).append(s)
hold = []
for p in sorted(pat, key=lambda p: hashlib.sha256(p.encode()).hexdigest()):
    if len(hold) >= HOLD:
        break
    hold += pat[p]
hold = sorted(hold)
pool = comp + sorted(s for s in fm if s not in dupes and s not in set(hold))
pr(f"competition {len(comp)}; fastMRI {len(fm)} ({len(pat)} patients): dupes + excluded {len(dupes)}, holdout {len(hold)} exams "
   f"({sum(1 for p in pat if pat[p][0] in set(hold))} patients), pooled {len(pool) - len(comp)}; pool {len(pool)}")

P384, P224 = np.array([r384[s] for s in pool]), np.array([r224[s] for s in pool])
assert np.isfinite(P384).all() and np.isfinite(P224).all()
tgt = np.zeros_like(P384)
for j in range(12):                                      # pooled rank order, logit-blend values (PUB-KD-R recipe)
    final = pct((1 - A224) * pct(P384[:, j]) + A224 * pct(P224[:, j]))
    plog = 1 / (1 + np.exp(-((1 - A224) * logit(P384[:, j]) + A224 * logit(P224[:, j]))))
    tgt[np.argsort(final, kind="stable"), j] = np.sort(plog)

ra = {r["study_id"]: np.array(r["probabilities"]) for r in json.loads(RA.read_text())["labels"]}
ci = [i for i, s in enumerate(pool) if s in ra]
assert len(ci) == len(comp) == len(ra), (len(ci), len(comp), len(ra))
rho = [spear(tgt[ci, j], [ra[pool[i]][j] for i in ci]) for j in range(12)]
pr("competition: pooled vs ra targets spearman " + " ".join(f"{T[j][:6]} {rho[j]:.4f}" for j in range(12))
   + f"; mean |diff| {np.mean([np.abs(tgt[i] - ra[pool[i]]).mean() for i in ci]):.4f}")
fi = [i for i, s in enumerate(pool) if s.startswith("fm_")]
for nm, ix in (("competition", ci), ("fastMRI", fi)):
    pr(f"{nm:11s} target mean " + " ".join(f"{T[j][:6]} {tgt[ix, j].mean():.3f}" for j in range(12)))
    pr(f"{nm:11s} extreme cells (<0.05 or >0.95) {((tgt[ix] < .05) | (tgt[ix] > .95)).mean():.3f}")

O.mkdir(parents=True, exist_ok=True)
out = O / "fm_targets_raptor_pure.json"
out.write_text(json.dumps({"targets_in_order": T, "source": "SURPASS-F Raptor 384 + 224 teacher, competition + fastMRI pool (pure)",
                           "labels": [{"study_id": s, "probabilities": [round(float(x), 6) for x in tgt[i]],
                                       "weights": [1.0] * 12} for i, s in enumerate(pool)]}))
pr(f"wrote {out.name}: {len(pool)} studies, sha256 {sha(out)}")
(O / "fm_holdout.json").write_text(json.dumps(hold) + "\n")
ht = {s: [round(float(1 / (1 + np.exp(-((1 - A224) * logit(r384[s][j]) + A224 * logit(r224[s][j]))))), 6) for j in range(12)]
      for s in hold}
(O / "fm_holdout_teacher.json").write_text(json.dumps({"model": "Raptor teacher", "note": "logit blend 0.65 / 0.35, fastMRI holdout",
                                                       "targets_in_order": T, "predictions": [
                                                           {"study_id": s, "probabilities": ht[s]} for s in hold]}) + "\n")
pr(f"wrote fm_holdout.json / fm_holdout_teacher.json: {len(hold)} studies, sha256 {sha(O / 'fm_holdout.json')}")
(O / "build_fm_targets.txt").write_text("\n".join(L) + "\n")
