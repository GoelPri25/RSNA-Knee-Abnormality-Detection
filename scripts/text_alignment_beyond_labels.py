"""Ledger row T-pre2: is there finding content in the report embedding beyond what the labels already say?

Reuses T-pre's cached e5 report embeddings and language calls (scripts/text_alignment_prechecks.py must have run).
Training folds 1-5 only (fold 0 and gold-58 are never read). No model is trained on images and no label is changed.

R1 variance: fold-CV R^2 of the embedding from language, from the known label side, and from both (+ interactions);
   residual = embedding minus the in-sample linear fit on all of them.
R2 beyond-label content: residual -> extra-finding tags (outside the 12 targets) from a hand-written lexicon in
   en / es / de / fr / nl, with a simple preceding-negation rule.
R3 within-language site channel: residual -> scanner model, one-vs-rest AUC per (language, model).

Usage (CPU only): CUDA_VISIBLE_DEVICES="" python -I text_alignment_beyond_labels.py <repository root>
"""
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(sys.argv[1]).resolve()
PRE = ROOT / "data/text_alignment_prechecks"
OUT = PRE / "beyond_labels"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = ROOT / "data/coatnet_d96_fold0/assets"
STATES = ["SIL", "NEG", "BELOW", "POS", "BORD", "HEDGE", "LIKELY"]
TAG_LANGS = ["en", "es", "de", "fr", "nl"]
MIN_POS = 30
csv.field_size_limit(10 ** 9)
log_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_lines.append(s)


# ---------------------------------------------------------------- data (same order as T-pre)
folds = {r["study_id"]: int(r["fold"]) for r in csv.DictReader(open(ASSETS / "folds.csv"))}
lab = json.load(open(ASSETS / "soft_labels_final_per_target_latD2.json"))
TARGETS = lab["targets_in_order"]
labels = {r["study_id"]: np.array(r["probabilities"]) for r in lab["labels"] if folds.get(r["study_id"], 0) != 0}
ids = sorted(labels)
assert len(ids) == 3524
reports = {r["StudyInstanceUID"]: r["Report"] for r in csv.DictReader(open(ROOT.parent / "reports_raw/train.csv"))
           if r["StudyInstanceUID"] in labels}
states = {r["StudyInstanceUID"]: r for r in csv.DictReader(
    open(ROOT / "data/label_datasets/hpo_v3_1_d2/rules/s5_fx_site/rule_states.csv")) if r["StudyInstanceUID"] in labels}
hdr = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(ROOT / "data/dicom_demographics/study_header_tags.csv"))}
LANG = json.load(open(PRE / "languages.json"))
E = np.load(PRE / "emb_e5_report.npy")
assert E.shape == (3524, 768)
F = np.array([folds[s] for s in ids])
LG = np.array([LANG[s] for s in ids])
LANGS = sorted(set(LG.tolist()))
Y = np.stack([labels[s] for s in ids])

# known label side: label-v2 probabilities + rule-state one-hots; language one-hot; language x label interactions
S1h = np.array([[float(states[s][t] == st) for t in TARGETS for st in STATES] for s in ids])
L1h = np.array([[float(l == x) for x in LANGS] for l in LG])
X_lab = np.hstack([Y, S1h])
X_all = np.hstack([X_lab, L1h, (L1h[:, :, None] * Y[:, None, :]).reshape(len(ids), -1)])
log(f"studies {len(ids)}; known-label features {X_lab.shape[1]}, all features {X_all.shape[1]}")


# ---------------------------------------------------------------- R1 variance shares
def cv_r2(X):
    pred = np.zeros_like(E)
    for k in [1, 2, 3, 4]:
        tr, te = F != k, F == k
        pred[te] = Ridge(alpha=1.0).fit(X[tr], E[tr]).predict(X[te])
    m = F != 5
    return float(1 - ((E[m] - pred[m]) ** 2).sum() / ((E[m] - E[m].mean(0)) ** 2).sum())


r1 = {"language": cv_r2(L1h), "labels_and_states": cv_r2(X_lab), "all_with_interactions": cv_r2(X_all)}
log("R1 fold-CV R^2 of e5 embedding:", {k: round(v, 3) for k, v in r1.items()})
A = np.hstack([X_all, np.ones((len(ids), 1))])
coef, *_ = np.linalg.lstsq(A, E, rcond=None)
R = E - A @ coef
r1["residual_share_in_sample"] = float((R ** 2).sum() / ((E - E.mean(0)) ** 2).sum())
log(f"R1 in-sample residual variance share {r1['residual_share_in_sample']:.3f}")


# ---------------------------------------------------------------- R2 extra-finding tags (outside the 12 targets)
def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


TAGS = {
    "posterior_horn": r"posterior horn|cuerno posterior|asta posterior|hinterhorn|corne posterieure|achterhoorn",
    "bucket_handle": r"bucket[- ]?handle|asa de cubo|asa de balde|korbhenkel|anse de seau|emmerhengsel",
    "chondromalacia": r"chondromala[cz]|condromala[cz]",
    "osteophyte": r"osteophyt|osteofit|osteofyt",
    "tendinopathy": r"tendinopat|tendinopath|tendinos|tendinit",
    "loose_body": r"loose bod|cuerpos? libres?|freie[nrs]? (gelenk)?korper|gelenkmaus|corps (etrangers?|libres?)"
                  r"|(losse|vrije) (corpus|corpora|lichaam|lichamen)|corp(us|ora) liber",
    "meniscal_extrusion": r"extrus|extrudi",
    "parameniscal_cyst_ganglion": r"paramenisc|paramenisk|ganglion|meniscal cyst|quiste meniscal",
    "hoffa": r"hoffa",
    "plica": r"\bplica|\bplika",
}
NEG = re.compile(r"\b(no|not|without|negative for|free of|absence of|sin|ausencia de|ni|kein|keine|keinen|keiner"
                 r"|ohne|nicht|pas de|sans|aucun|aucune|geen|zonder|niet)\b")


def tagged(text, pat):
    for m in re.finditer(pat, text):
        pre = text[max(0, m.start() - 40):m.start()]
        pre = re.split(r"[.;:\n]", pre)[-1]
        if not NEG.search(pre):
            return 1
    return 0


tag_rows = np.where(np.isin(LG, TAG_LANGS))[0]
TXT = [norm(reports[ids[i]]) for i in tag_rows]
T = {k: np.array([tagged(t, p) for t in TXT]) for k, p in TAGS.items()}
for k, v in T.items():
    per = {l: int(v[LG[tag_rows] == l].sum()) for l in TAG_LANGS}
    log(f"R2 tag {k:28s} positives {int(v.sum()):4d} / {len(tag_rows)}  by language {per}")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=3, scoring="roc_auc",
                                                                max_iter=3000))


def cv_auc(X, y, f):
    pred = np.full(len(y), np.nan)
    for k in [1, 2, 3, 4]:
        tr, te = f != k, f == k
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        pred[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    m = ~np.isnan(pred)
    if min(y[m].sum(), (1 - y[m]).sum()) < 10:
        return None
    return float(roc_auc_score(y[m], pred[m]))


feats = {"residual": R[tag_rows], "label_side_plus_language": np.hstack([X_lab, L1h])[tag_rows], "raw_e5": E[tag_rows]}
r2 = {}
for k, y in T.items():
    if y.sum() < MIN_POS:
        r2[k] = None
        continue
    r2[k] = {name: cv_auc(X, y, F[tag_rows]) for name, X in feats.items()}
    log(f"R2 {k:28s} AUC " + "  ".join(f"{n} {v:.3f}" for n, v in r2[k].items()))
ev = [k for k, v in r2.items() if v is not None and v["residual"] is not None]
r2_macro = {n: float(np.mean([r2[k][n] for k in ev])) for n in feats} if ev else {}
log(f"R2 evaluable tags {len(ev)}; macro AUC", {n: round(v, 3) for n, v in r2_macro.items()})

# ---------------------------------------------------------------- R3 within-language site channel
MODEL = np.array([(hdr.get(s, {}).get("ManufacturerModelName") or "NA").strip() or "NA" for s in ids])
r3 = {}
for l in LANGS:
    rows = np.where(LG == l)[0]
    models = [m for m, n in Counter(MODEL[rows].tolist()).items() if n >= 30 and m != "NA"]
    if len(models) < 2:
        continue
    for m in models:
        y = (MODEL[rows] == m).astype(int)
        if (1 - y).sum() < 30:
            continue
        r3[f"{l}:{m}"] = {"n": int(y.sum()), "of": len(rows),
                          "residual": cv_auc(R[rows], y, F[rows]),
                          "label_side": cv_auc(X_lab[rows], y, F[rows]),
                          "raw_e5": cv_auc(E[rows], y, F[rows])}
        log(f"R3 {l}:{m:28s} n {y.sum():4d}/{len(rows):4d}  " +
            "  ".join(f"{n} {r3[f'{l}:{m}'][n]:.3f}" for n in ["residual", "label_side", "raw_e5"]
                      if r3[f"{l}:{m}"][n] is not None))
ok3 = [v for v in r3.values() if v["residual"] is not None]
r3_macro = {n: float(np.mean([v[n] for v in ok3 if v[n] is not None])) for n in ["residual", "label_side", "raw_e5"]}
log(f"R3 (language, model) pairs {len(ok3)}; macro AUC", {n: round(v, 3) for n, v in r3_macro.items()})

# ---------------------------------------------------------------- pre-registered decision
g2 = len(ev) >= 4 and r2_macro.get("residual", 0) >= 0.80
g3 = bool(ok3) and r3_macro["residual"] <= 0.75
if g2 and g3:
    decision = "GO (GPU spec for option 1 next; user still chooses vs Phase 3)"
elif g2:
    decision = "NO-GO for now: content beyond labels exists, but a within-language site channel survives"
else:
    decision = "NO-GO: no finding content beyond the labels; text alignment closed"
log("gates", {"R2 residual macro >= 0.80 over >= 4 tags": g2, "R3 residual site macro <= 0.75": g3})
log("DECISION:", decision)
summary = {"ledger_row": "T-pre2", "R1": r1, "R2_per_tag": r2, "R2_macro": r2_macro, "R2_evaluable": ev,
           "tag_positives": {k: int(v.sum()) for k, v in T.items()}, "R3": r3, "R3_macro": r3_macro,
           "gates": {"R2": g2, "R3": g3}, "decision": decision}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUT / "summary.txt").write_text("\n".join(log_lines) + "\n")
