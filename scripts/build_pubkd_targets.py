"""PUB-KD teacher targets: the public 0.958 blend (goodpjw2008 v25/27) re-computed on our studies, as training targets.
Members (per study, 12 probabilities): Raptor 384 and 224 (mirror-averaged, pubkd_raptor_teacher.py), R6 (flip TTA) and
the 2.5D reader (3 folds) from the Kaggle teacher kernel. The blend, exactly theirs, per finding, ranks over all studies:
  raptor = 0.65 pct(384) + 0.35 pct(224)
  stack  = w pct(raptor) ... w = 0.5 except Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35 (Raptor weight);
           stack = w raptor + (1 - w) pct(R6)       (Part A3: raptor already a rank mix, not re-ranked)
  final  = pct((1 - v) pct(stack) + v pct(reader)), v = their per-finding reader weights (Part B, mode "end")
A rank is not a probability, so the target keeps the blend's ORDER exactly and takes its VALUES from the same blend in
logit space (sigmoid of the members' logits under the same effective weights): per finding, the k-th lowest final rank
gets the k-th lowest logit-blend probability. Outputs (label-file layout, weights 1 on all 12 cells):
  pubkd_targets_pure.json   y = teacher
  pubkd_targets_mix73.json  y = 0.7 teacher + 0.3 label-v2 where label-v2 has weight > 0, else teacher
QC printed and saved: each member's and the blend's AUC on gold-58 and on fold-0 label-v2 confident cells (both
in-sample for these all-data models: sanity only), and the members' rank correlations.
Usage: python3 -I build_pubkd_targets.py <project root> <pubkd dir: raptor384_*.json, raptor224_*.json, r6/*_all.csv> [raptor]
  raptor: interim teacher (PUB-KD-R) = the Raptor half only, raptor = 0.65 pct(384) + 0.35 pct(224), logit values
          0.65 / 0.35; no R6 / reader needed; outputs pubkd_targets_raptor_{pure,mix73}.json.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, D = Path(sys.argv[1]), Path(sys.argv[2])
RONLY = len(sys.argv) > 3 and sys.argv[3] == "raptor"
TAG = "raptor_" if RONLY else ""
A = R / "repository/data/coatnet_d96_fold0/assets"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
W_RAPTOR = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35}
W_READER = {"Baker's": 0.035, "Contusion": 0.03, "Medial OA": 0.025, "ACL": 0.025, "Medial Meniscus": 0.02, "MCL": 0.02,
            "Lateral Meniscus": 0.008, "Fracture": 0.008, "Lateral OA": 0.008, "PF OA": 0.008, "Effusion": 0.004,
            "Synovitis": 0.003}
A224 = 0.35
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


def raptor(model):
    out = {}
    for f in sorted(D.glob(f"raptor{model}_*.json")):
        d = json.loads(f.read_text()); assert d["targets"] == T
        for s, v in d["pred"].items():
            assert s not in out
            out[s] = 0.5 * (np.array(v["plain"]) + np.array(v["mirror"]))   # their infer_pair_probs
    return pd.DataFrame.from_dict(out, orient="index", columns=T)


def kaggle(name):
    df = pd.read_csv(D / "r6" / name, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    return df[T].astype(float)


pct = lambda v: pd.Series(v).rank(method="average", pct=True).to_numpy()
spear = lambda a, b: pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank())   # Spearman without scipy
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

M = {"r384": raptor("384"), "r224": raptor("224")}
if not RONLY:
    M.update({"r6": kaggle("r6_train_all.csv"), "reader": kaggle("reader_train_all.csv")})
ids = sorted(M["r384"].index)
for k, df in M.items():
    assert set(df.index) >= set(ids), f"{k} misses {len(set(ids) - set(df.index))} studies"
    M[k] = df.loc[ids]
    assert np.isfinite(M[k].to_numpy()).all() and (M[k].nunique() > 1).all(), k
pr(f"studies {len(ids)}; members {list(M)}")

final, plog = np.zeros((len(ids), 12)), np.zeros((len(ids), 12))
for j, c in enumerate(T):
    rr = (1 - A224) * pct(M["r384"][c]) + A224 * pct(M["r224"][c])
    if RONLY:                                            # PUB-KD-R: the Raptor half alone
        final[:, j] = pct(rr)
        plog[:, j] = 1 / (1 + np.exp(-((1 - A224) * logit(M["r384"][c]) + A224 * logit(M["r224"][c]))))
        continue
    w, v = W_RAPTOR.get(c, 0.5), W_READER[c]
    stack = w * rr + (1 - w) * pct(M["r6"][c])
    final[:, j] = pct((1 - v) * pct(stack) + v * pct(M["reader"][c]))
    z = (1 - v) * (w * ((1 - A224) * logit(M["r384"][c]) + A224 * logit(M["r224"][c])) + (1 - w) * logit(M["r6"][c])) \
        + v * logit(M["reader"][c])
    plog[:, j] = 1 / (1 + np.exp(-z))
tgt = np.zeros_like(final)
for j in range(12):                                      # blend's order, logit blend's values
    tgt[np.argsort(final[:, j], kind="stable"), j] = np.sort(plog[:, j])
rho = [spear(tgt[:, j], final[:, j]) for j in range(12)]
assert min(rho) > 0.9999, rho
pr("target order = blend order (spearman per finding >= 0.9999); logit-blend vs rank-blend spearman: "
   + " ".join(f"{T[j][:6]} {spear(plog[:, j], final[:, j]):.3f}" for j in range(12)))

# ---- QC (in-sample for these all-data models: sanity, not evidence)
lv = json.loads((A / "soft_labels_final_per_target_latD2.json").read_text()); assert lv["targets_in_order"] == T
lab = {r["study_id"]: (np.array([.5 if p is None else p for p in r["probabilities"]]), np.array(r["weights"])) for r in lv["labels"]}
fold = {x["study_id"]: x["fold"] for x in csv.DictReader((A / "folds.csv").open())}
gold = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
pos = {s: i for i, s in enumerate(ids)}


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


gi = [pos[r["StudyInstanceUID"]] for r in gold if r["StudyInstanceUID"] in pos]
gy = np.array([[int(r[t]) for t in T] for r in gold if r["StudyInstanceUID"] in pos]).astype(bool)
f0 = [s for s in ids if fold.get(s) == "0" and s in lab]
fi = [pos[s] for s in f0]; Y = np.array([lab[s][0] for s in f0]); C = (Y <= .3) | (Y >= .7)
cols = {**{k: M[k].to_numpy() for k in M}, "blend_rank": final, "target": tgt}
for k, S in cols.items():
    g = np.nanmean([auc(S[gi, j], gy[:, j]) for j in range(12)])
    f = np.nanmean([auc(S[fi, j][C[:, j]], Y[C[:, j], j] >= .7) for j in range(12)])
    pr(f"QC {k:11s} gold-58 macro {g:.4f} ({len(gi)} studies) | fold-0 label-v2 confident 12t {f:.4f}")
pr("rank corr with r384 (mean over findings): " + " ".join(
    f"{k} {np.mean([spear(M[k][c], M['r384'][c]) for c in T]):.3f}" for k in M))

# ---- outputs
lv_ids = set(lab)
miss = sorted(s for s in lv_ids if fold.get(s) not in (None, "0") and s not in pos)
assert not miss, f"{len(miss)} training studies without teacher predictions"
for name, mix in (("pure", None), ("mix73", 0.3)):
    rows = []
    for s in ids:
        y = tgt[pos[s]].copy()
        if mix and s in lab:
            yl, wl = lab[s]
            y = np.where(wl > 0, (1 - mix) * y + mix * yl, y)
        rows.append({"study_id": s, "probabilities": [round(float(x), 6) for x in y], "weights": [1.0] * 12})
    out = D / f"pubkd_targets_{TAG}{name}.json"
    out.write_text(json.dumps({"targets_in_order": T, "source": f"PUB-KD {'Raptor 384 + 224 rank blend (interim)' if RONLY else 'public 0.958 blend'} teacher ({name})",
                               "labels": rows}))
    pr(f"wrote {out.name}: {len(rows)} studies, sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
(D / f"build_pubkd_targets{'_raptor' if RONLY else ''}.txt").write_text("\n".join(L) + "\n")
