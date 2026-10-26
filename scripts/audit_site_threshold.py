"""Site-threshold audit (CPU, correlational, 2026-10-13): do reporting thresholds differ by site (report language), and does
the image model inherit them?

Hypothesis: the host labels come from one MSK panel with one guideline; the training labels come from hundreds of local
radiologists in ~10 countries who grade (or don't grade) borderline findings differently (e.g. "Derrame." with no amount).
That noise depends on the site, and the site is visible in the image (scanner, protocol), so the model can learn
"site X looks positive" -> a cross-site ranking error under the host's pooled AUC. If so, conditioning training on the
site (an explicit offset per language x target, dropped at test) would remove it.

Measured here, with report language as the site proxy:
  (1) per language: report-present rate and share of present findings with no severity grade;
  (2) fold 0 (825, 1e out-of-fold): logistic ruler ~ 1e logit + language offsets per target (offsets beyond what the
      image model already explains; a lower bound if the model has absorbed site from image style);
  (3) fold 0: per-language mean 1e probability vs ruler rate (does the model reproduce site reporting rates?);
  (4) gold-58 (development evidence only): host-positive rate of report-present cells by language.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds python -I /proj/repository/scripts/audit_site_threshold.py /proj
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"
SW = {"en": "the of and with is no there are knee".split(), "es": "de la el con sin rodilla los menisco derrame".split(),
      "de": "der die und mit kein keine ist knie meniskus".split(), "fr": "de la le avec sans genou ménisque épanchement".split(),
      "tr": "ve ile yok diz menisküs".split(), "nl": "de het en met geen knie meniscus".split(),
      "hr": "je se u i na sa bez koljena prati".split(), "pt": "do da com sem joelho não".split()}


def lang(t):
    if len(re.findall(r"[Ͱ-Ͽ]", t)) > 20:
        return "el"
    if len(re.findall(r"[Ѐ-ӿ]", t)) > 20:
        return "cyr"
    w = re.findall(r"\w+", t.lower()); c = collections.Counter(w)
    sc = {l: sum(c[x] for x in s) / (len(w) + 1) for l, s in SW.items()}
    return max(sc, key=sc.get) if max(sc.values()) > 0.02 else "other"


rows = list(csv.DictReader(open(R / "reports_raw/train.csv", encoding="utf-8-sig")))
LANG = {r["StudyInstanceUID"]: lang(r["Report"]) for r in rows}
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
st["lang"] = [LANG.get(s, "?") for s in st.index]
langs = [l for l, n in collections.Counter(st["lang"]).most_common() if n >= 60]
print("studies per language:", dict(collections.Counter(st["lang"]).most_common()), "\n")

print("(1) report-present rate | share of present with no severity grade ('unspecified'), by language")
GR = ["ACL", "MCL", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Baker's", "Synovitis"]
print(f"{'target':12s} " + " ".join(f"{l:>13s}" for l in langs))
for t in GR:
    cells = []
    for l in langs:
        s = st[st["lang"] == l]; pres = s[f"{key(t)}__status"] == "present"
        cells.append(f"{pres.mean():5.2f} | {(s.loc[pres, f'{key(t)}__severity'] == 'unspecified').mean():4.2f}")
    print(f"{t:12s} " + " ".join(f"{c:>13s}" for c in cells))

y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]]); mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
d = json.loads((D / "coatnet_d96_fold0/run_r50_pertarget/val_swa.json").read_text())
m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
tg = d.get("targets_in_order", d.get("targets", T))
p = np.array([m[s] for s in ids], float)[:, [list(tg).index(t) for t in T]]
lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
L0 = np.array([LANG.get(s, "?") for s in ids]); fl = [l for l in langs if (L0 == l).sum() >= 30]
print(f"\nfold-0 studies per language: { {l: int((L0 == l).sum()) for l in fl} }")


def fit(X, t, lam=1e-3, iters=60):
    A = np.hstack([np.ones((len(X), 1)), X]); b = np.zeros(A.shape[1]); Lm = lam * np.eye(A.shape[1]); Lm[0, 0] = 0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-A @ b)); H = A.T @ (A * (q * (1 - q))[:, None]) + Lm
        b -= np.linalg.solve(H, A.T @ (q - t) + Lm @ b)
    q = 1 / (1 + np.exp(-A @ b)); ll = (t * np.log(q + 1e-12) + (1 - t) * np.log(1 - q + 1e-12)).sum()
    return b, ll, np.sqrt(np.diag(np.linalg.inv(A.T @ (A * (q * (1 - q))[:, None]) + Lm)))


print("\n(2) ruler ~ 1e logit + language offsets (reference = en); offset in logits (z), LR chi2 / df")
print(f"{'target':16s} " + " ".join(f"{l:>12s}" for l in fl[1:]) + "   chi2/df")
for k, t in enumerate(T):
    sel = mask[:, k] & np.isin(L0, fl)
    D1 = np.stack([(L0[sel] == l).astype(float) for l in fl[1:]], 1)
    b0, ll0, _ = fit(lg[sel, k:k + 1], truth[sel, k].astype(float))
    b1, ll1, se = fit(np.hstack([lg[sel, k:k + 1], D1]), truth[sel, k].astype(float))
    print(f"{t:16s} " + " ".join(f"{b1[2 + j]:+5.2f} ({b1[2 + j] / se[2 + j]:+4.1f})" for j in range(len(fl) - 1))
          + f"   {2 * (ll1 - ll0):5.1f}/{len(fl) - 1}")

print("\n(3) fold 0 per language: ruler positive rate / mean 1e probability (confident cells)")
print(f"{'target':16s} " + " ".join(f"{l:>11s}" for l in fl))
for k, t in enumerate(T):
    print(f"{t:16s} " + " ".join(f"{truth[mask[:, k] & (L0 == l), k].mean():.2f}/{p[mask[:, k] & (L0 == l), k].mean():.2f}"
                                  .rjust(11) for l in fl))

g = json.loads((R / "colab_bundle/prepared/gold_labels_and_split.json").read_text())
G = g["labels"]
gid = list(G) if isinstance(G, dict) else [r.get("study_id") for r in G]
print(f"\n(4) gold-58: {len(gid)} studies; languages {dict(collections.Counter(LANG.get(s, '?') for s in gid))}")
gl = (lambda s: G[s]) if isinstance(G, dict) else (lambda s: next(r for r in G if r.get("study_id") == s))
print(f"    example gold record: {str(gl(gid[0]))[:200]}")

# (5) size in ranking units: cross-fitted pooled AUC vs the ruler of the 1e logit alone vs logit + language offsets
#     (offsets fitted on 4/5 of fold 0, applied to the held-out 1/5)


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(ids))
print("\n(5) cross-fitted pooled AUC vs ruler: 1e logit -> + language offsets")
gains = []
for k, t in enumerate(T):
    sel = np.where(mask[:, k])[0]; Lk = np.where(np.isin(L0, fl), L0, "other")
    X = np.stack([(Lk == l).astype(float) for l in fl[1:] + ["other"]], 1)
    out = np.full(len(ids), np.nan)
    for j in range(5):
        tr, te = sel[fold[sel] != j], sel[fold[sel] == j]
        b, _, _ = fit(np.hstack([lg[tr, k:k + 1], X[tr]]), truth[tr, k].astype(float), lam=1.0)
        out[te] = b[1] * lg[te, k] + X[te] @ b[2:]
    a0, a1 = auc(lg[sel, k], truth[sel, k]), auc(out[sel], truth[sel, k]); gains.append(a1 - a0)
    print(f"  {t:16s} {a0:.4f} -> {a1:.4f}  {a1 - a0:+.4f}")
print(f"  macro {np.mean(gains):+.4f}")
