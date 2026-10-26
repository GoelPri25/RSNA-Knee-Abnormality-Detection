"""Ledger row T-pre3: site-stripped report-embedding target (arm B) and extra-finding tag quality (arm A).

Reuses T-pre's cached e5 report embeddings and language calls. Training folds 1-5 only (fold 0 and gold-58 are never
read). Prints counts and scores only, never report text. No model is trained on images and no label is changed.

Site regressors: language + (language x scanner model) + manufacturer + field strength one-hots.
T_full   = embedding minus its in-sample linear fit on the site regressors (arm-B target candidate).
T_beyond = embedding minus the fit on site regressors + label side (labels, rule states, language x labels).
S-R3: within-language scanner-model AUC (linear and kNN probes) on raw / T_full / T_beyond.
S-R2: lexicon tags from T_beyond (5 languages, and English only).
A-tags: positives, negated share, lift on related targets, tag x site Cramer's V.

Usage (CPU only): CUDA_VISIBLE_DEVICES="" python -I text_alignment_site_stripped.py <repository root>
"""
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(sys.argv[1]).resolve()
PRE = ROOT / "data/text_alignment_prechecks"
OUT = PRE / "site_stripped"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = ROOT / "data/coatnet_d96_fold0/assets"
STATES = ["SIL", "NEG", "BELOW", "POS", "BORD", "HEDGE", "LIKELY"]
TAG_LANGS = ["en", "es", "de", "fr", "nl"]
csv.field_size_limit(10 ** 9)
log_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_lines.append(s)


# ---------------------------------------------------------------- data (same order as T-pre / T-pre2)
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
n = len(ids)


def h(s, col):
    return (hdr.get(s, {}).get(col) or "NA").strip() or "NA"


MODEL = np.array([h(s, "ManufacturerModelName") for s in ids])
MANU = np.array([h(s, "Manufacturer") for s in ids])
FIELD = np.array([h(s, "MagneticFieldStrength") for s in ids])
combo_raw = np.array([f"{l}|{m}" for l, m in zip(LG, MODEL)])
cnt = Counter(combo_raw.tolist())
COMBO = np.array([c if cnt[c] >= 10 else c.split("|")[0] + "|other" for c in combo_raw])


def onehot(v):
    cats = sorted(set(v.tolist()))
    return np.array([[float(x == c) for c in cats] for x in v])


L1h = onehot(LG)
X_site = np.hstack([L1h, onehot(COMBO), onehot(MANU), onehot(FIELD)])
S1h = np.array([[float(states[s][t] == st) for t in TARGETS for st in STATES] for s in ids])
X_lab = np.hstack([Y, S1h, (L1h[:, :, None] * Y[:, None, :]).reshape(n, -1)])
log(f"studies {n}; site regressors {X_site.shape[1]} ({len(set(COMBO.tolist()))} language x model combos); "
    f"label-side regressors {X_lab.shape[1]}")


def strip(X):
    A = np.hstack([X, np.ones((n, 1))])
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    return E - A @ coef


T_full = strip(X_site)
T_beyond = strip(np.hstack([X_site, X_lab]))
tot = ((E - E.mean(0)) ** 2).sum()
log(f"variance share kept: T_full {(T_full ** 2).sum() / tot:.3f}, T_beyond {(T_beyond ** 2).sum() / tot:.3f}")


# ---------------------------------------------------------------- probes
def lin():
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=3, scoring="roc_auc",
                                                                max_iter=3000))


def knn():
    return KNeighborsClassifier(n_neighbors=25, metric="cosine")


def cv_auc(make, X, y, f):
    pred = np.full(len(y), np.nan)
    for k in [1, 2, 3, 4]:
        tr, te = f != k, f == k
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        pred[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    m = ~np.isnan(pred)
    if min(y[m].sum(), (1 - y[m]).sum()) < 10:
        return None
    return float(roc_auc_score(y[m], pred[m]))


# ---------------------------------------------------------------- S-R3 within-language site channel
reps = {"raw_e5": E, "T_full": T_full, "T_beyond": T_beyond}
pairs = []
for l in LANGS:
    rows = np.where(LG == l)[0]
    models = [m for m, c in Counter(MODEL[rows].tolist()).items() if c >= 30 and m != "NA"]
    if len(models) < 2:
        continue
    for m in models:
        y = (MODEL[rows] == m).astype(int)
        if (1 - y).sum() >= 30:
            pairs.append((l, m, rows, y))
sr3 = {}
for rep, X in reps.items():
    for pname, make in [("linear", lin), ("knn", knn)]:
        vals = [cv_auc(make, X[rows], y, F[rows]) for _, _, rows, y in pairs]
        vals = [v for v in vals if v is not None]
        sr3[f"{rep}/{pname}"] = float(np.mean(vals))
        log(f"S-R3 site macro AUC {rep:9s} {pname:6s} {np.mean(vals):.3f} over {len(vals)} pairs")


# ---------------------------------------------------------------- tags (same lexicon as T-pre2)
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
RELATED = {"posterior_horn": ["Medial Meniscus", "Lateral Meniscus"],
           "bucket_handle": ["Medial Meniscus", "Lateral Meniscus"],
           "parameniscal_cyst_ganglion": ["Medial Meniscus", "Lateral Meniscus"],
           "meniscal_extrusion": ["Medial Meniscus", "Medial OA"],
           "chondromalacia": ["PF OA"],
           "osteophyte": ["Medial OA", "Lateral OA", "PF OA"],
           "loose_body": ["Medial OA", "Lateral OA", "PF OA"]}


def scan(text, pat):
    """(affirmed, negated) match flags for one report."""
    aff = neg = 0
    for m in re.finditer(pat, text):
        pre = re.split(r"[.;:\n]", text[max(0, m.start() - 40):m.start()])[-1]
        if NEG.search(pre):
            neg = 1
        else:
            aff = 1
    return aff, neg


tag_rows = np.where(np.isin(LG, TAG_LANGS))[0]
TXT = [norm(reports[ids[i]]) for i in tag_rows]
SC = {k: np.array([scan(t, p) for t in TXT]) for k, p in TAGS.items()}
T = {k: v[:, 0] for k, v in SC.items()}
en_rows = LG[tag_rows] == "en"

# S-R2 content beyond labels, after site stripping
sr2, sr2_en = {}, {}
for k, y in T.items():
    if y.sum() >= 30:
        sr2[k] = cv_auc(lin, T_beyond[tag_rows], y, F[tag_rows])
    if y[en_rows].sum() >= 30:
        sr2_en[k] = cv_auc(lin, T_beyond[tag_rows][en_rows], y[en_rows], F[tag_rows][en_rows])
    log(f"S-R2 {k:28s} T_beyond AUC all-5 {sr2.get(k) or float('nan'):.3f}  English-only "
        f"{sr2_en.get(k) or float('nan'):.3f}")
ok2 = [v for v in sr2.values() if v is not None]
ok2en = [v for v in sr2_en.values() if v is not None]
sr2_macro, sr2_en_macro = float(np.mean(ok2)), float(np.mean(ok2en)) if ok2en else float("nan")
log(f"S-R2 macro all-5 {sr2_macro:.3f} over {len(ok2)} tags; English-only {sr2_en_macro:.3f} over {len(ok2en)} tags")

# A-tags quality
POSI = {t: Y[tag_rows, j] >= 0.7 for j, t in enumerate(TARGETS)}
combo_tag = COMBO[tag_rows]
big = [c for c, k in Counter(combo_tag.tolist()).items() if k >= 30 and not c.endswith("|other")]
in_big = np.isin(combo_tag, big)


def cramers_v_binary(y, g):
    cats = sorted(set(g.tolist()))
    tab = np.array([[np.sum((g == c) & (y == v)) for v in [0, 1]] for c in cats], float)
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    chi2 = ((tab - exp) ** 2 / np.where(exp > 0, exp, 1)).sum()
    return float(np.sqrt(chi2 / tab.sum()))


atags = {}
for k, y in T.items():
    matches = SC[k].max(1).sum()
    negated_share = float(SC[k][:, 1].sum() / max(1, matches))
    lift = None
    if k in RELATED:
        rel = np.any([POSI[t] for t in RELATED[k]], axis=0)
        lift = float(rel[y == 1].mean() / rel.mean()) if y.sum() else None
    v = cramers_v_binary(y[in_big], combo_tag[in_big])
    kept = bool(y.sum() >= 40 and (lift is None or lift >= 1.3) and v < 0.40)
    atags[k] = {"positives": int(y.sum()), "negated_share": negated_share, "lift": lift, "site_cramers_v": v,
                "kept": kept}
    log(f"A-tag {k:28s} pos {int(y.sum()):4d}  negated {negated_share:.2f}  lift "
        f"{'  n/a' if lift is None else f'{lift:5.2f}'}  site V {v:.3f}  {'KEPT' if kept else 'dropped'}")

# ---------------------------------------------------------------- pre-registered decisions
gB = {"T_full site linear <= 0.65": sr3["T_full/linear"] <= 0.65,
      "T_full site kNN <= 0.65": sr3["T_full/knn"] <= 0.65,
      "T_beyond tags all-5 >= 0.80 over >= 4": len(ok2) >= 4 and sr2_macro >= 0.80,
      "T_beyond tags English >= 0.75": bool(ok2en) and sr2_en_macro >= 0.75}
kept = [k for k, v in atags.items() if v["kept"]]
gA = len(kept) >= 5
log("arm B gates", gB)
log(f"DECISION arm B: {'GO' if all(gB.values()) else 'NO-GO'}")
log(f"DECISION arm A: {'GO' if gA else 'NO-GO'} ({len(kept)} tags kept: {kept})")
summary = {"ledger_row": "T-pre3", "S_R3": sr3, "S_R2": sr2, "S_R2_english": sr2_en, "S_R2_macro": sr2_macro,
           "S_R2_english_macro": sr2_en_macro, "A_tags": atags, "gates_B": gB, "arm_B_GO": all(gB.values()),
           "arm_A_kept": kept, "arm_A_GO": gA}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUT / "summary.txt").write_text("\n".join(log_lines) + "\n")
