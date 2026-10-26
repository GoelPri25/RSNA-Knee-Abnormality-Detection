"""Ledger row T-enc: score each encoder's report embeddings with the T-pre / T-pre2 / T-pre3 checks.

Embeddings come from the Kaggle kernel tumis001/rsna-text-encoder-bakeoff (downloaded output dir, untrusted:
np.load with allow_pickle=False). Training folds 1-5 only (fold 0 and gold-58 never read). Prints scores only.

Per encoder: between-language variance share; label probe macro; leave-one-language-out (centred) macro;
T_full site macro (linear, kNN); T_beyond extra-finding tag macro (5 languages, English only); arm-B gates.

Usage (CPU only): CUDA_VISIBLE_DEVICES="" python -I text_encoder_bakeoff.py <repository root> <kernel output dir> [...]
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
KOUTS = [Path(a).resolve() for a in sys.argv[2:]]
PRE = ROOT / "data/text_alignment_prechecks"
OUT = PRE / "encoder_bakeoff"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = ROOT / "data/coatnet_d96_fold0/assets"
STATES = ["SIL", "NEG", "BELOW", "POS", "BORD", "HEDGE", "LIKELY"]
TAG_LANGS = ["en", "es", "de", "fr", "nl"]
POS, NEG_T, MIN_CLASS = 0.7, 0.15, 30
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
n = len(ids)
reports = {r["StudyInstanceUID"]: r["Report"] for r in csv.DictReader(open(ROOT.parent / "reports_raw/train.csv"))
           if r["StudyInstanceUID"] in labels}
states = {r["StudyInstanceUID"]: r for r in csv.DictReader(
    open(ROOT / "data/label_datasets/hpo_v3_1_d2/rules/s5_fx_site/rule_states.csv")) if r["StudyInstanceUID"] in labels}
hdr = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(ROOT / "data/dicom_demographics/study_header_tags.csv"))}
LANG = json.load(open(PRE / "languages.json"))
F = np.array([folds[s] for s in ids])
LG = np.array([LANG[s] for s in ids])
LANGS = sorted(set(LG.tolist()))
Y = np.stack([labels[s] for s in ids])


def h(s, col):
    return (hdr.get(s, {}).get(col) or "NA").strip() or "NA"


MODEL = np.array([h(s, "ManufacturerModelName") for s in ids])
combo_raw = np.array([f"{l}|{m}" for l, m in zip(LG, MODEL)])
cnt = Counter(combo_raw.tolist())
COMBO = np.array([c if cnt[c] >= 10 else c.split("|")[0] + "|other" for c in combo_raw])


def onehot(v):
    cats = sorted(set(v.tolist()))
    return np.array([[float(x == c) for c in cats] for x in v])


L1h = onehot(LG)
X_site = np.hstack([L1h, onehot(COMBO), onehot(np.array([h(s, "Manufacturer") for s in ids])),
                    onehot(np.array([h(s, "MagneticFieldStrength") for s in ids]))])
S1h = np.array([[float(states[s][t] == st) for t in TARGETS for st in STATES] for s in ids])
X_lab = np.hstack([Y, S1h, (L1h[:, :, None] * Y[:, None, :]).reshape(n, -1)])


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


TAGS = {  # same lexicon as T-pre2 / T-pre3
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
        if not NEG.search(re.split(r"[.;:\n]", text[max(0, m.start() - 40):m.start()])[-1]):
            return 1
    return 0


tag_rows = np.where(np.isin(LG, TAG_LANGS))[0]
TXT = [norm(reports[ids[i]]) for i in tag_rows]
TAGV = {k: np.array([tagged(t, p) for t in TXT]) for k, p in TAGS.items()}
en_rows = LG[tag_rows] == "en"
pairs = []
for l in LANGS:
    rows = np.where(LG == l)[0]
    models = [m for m, c in Counter(MODEL[rows].tolist()).items() if c >= 30 and m != "NA"]
    if len(models) >= 2:
        pairs += [(rows, (MODEL[rows] == m).astype(int)) for m in models if (MODEL[rows] != m).sum() >= 30]


def lin():
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=3, scoring="roc_auc",
                                                                max_iter=3000, n_jobs=-1))


def knn():
    return KNeighborsClassifier(n_neighbors=25, metric="cosine")


def cv_auc(make, X, y, groups, test_groups=(1, 2, 3, 4)):
    pred = np.full(len(y), np.nan)
    for g in test_groups:
        tr, te = groups != g, groups == g
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        pred[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    m = ~np.isnan(pred)
    if min(y[m].sum(), (1 - y[m]).sum()) < 10:
        return None
    return float(roc_auc_score(y[m], pred[m]))


def label_probe(X, groups=None, test_groups=(1, 2, 3, 4)):
    g = F if groups is None else groups
    vals = []
    for j in range(len(TARGETS)):
        y = Y[:, j]
        keep = (y >= POS) | (y <= NEG_T)
        yb = (y >= POS).astype(int)
        if min(yb[keep].sum(), (1 - yb[keep]).sum()) < MIN_CLASS:
            continue
        v = cv_auc(lin, X[keep], yb[keep], g[keep], test_groups)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals))


def strip(E, X):
    A = np.hstack([X, np.ones((n, 1))])
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    return E - A @ coef


def score(name, E):
    r = {}
    mu = E.mean(0)
    r["language_variance_share"] = float(sum((LG == l).sum() * ((E[LG == l].mean(0) - mu) ** 2).sum()
                                             for l in LANGS) / ((E - mu) ** 2).sum())
    r["label_probe"] = label_probe(E)
    Ec = E.copy()
    for l in LANGS:
        Ec[LG == l] -= E[LG == l].mean(0)
    langs = [l for l, c in Counter(LG.tolist()).items() if c >= 50]
    r["lolo_centred"] = label_probe(Ec, groups=LG, test_groups=langs)
    T_full = strip(E, X_site)
    T_beyond = strip(E, np.hstack([X_site, X_lab]))
    for pname, make in [("linear", lin), ("knn", knn)]:
        vals = [cv_auc(make, T_full[rows], y, F[rows]) for rows, y in pairs]
        r[f"T_full_site_{pname}"] = float(np.mean([v for v in vals if v is not None]))
    t5 = [cv_auc(lin, T_beyond[tag_rows], y, F[tag_rows]) for y in TAGV.values() if y.sum() >= 30]
    ten = [cv_auc(lin, T_beyond[tag_rows][en_rows], y[en_rows], F[tag_rows][en_rows])
           for y in TAGV.values() if y[en_rows].sum() >= 30]
    t5, ten = [v for v in t5 if v is not None], [v for v in ten if v is not None]
    r["T_beyond_tags_all5"], r["n_tags_all5"] = float(np.mean(t5)), len(t5)
    r["T_beyond_tags_en"], r["n_tags_en"] = float(np.mean(ten)), len(ten)
    r["gates"] = {"site_linear<=0.65": r["T_full_site_linear"] <= 0.65, "site_knn<=0.65": r["T_full_site_knn"] <= 0.65,
                  "tags_all5>=0.80": r["n_tags_all5"] >= 4 and r["T_beyond_tags_all5"] >= 0.80,
                  "tags_en>=0.75": r["T_beyond_tags_en"] >= 0.75}
    r["eligible"] = all(r["gates"].values())
    log(f"{name:14s} lang-var {r['language_variance_share']:.3f}  label {r['label_probe']:.4f}  LOLO "
        f"{r['lolo_centred']:.4f}  site lin {r['T_full_site_linear']:.3f} kNN {r['T_full_site_knn']:.3f}  tags "
        f"{r['T_beyond_tags_all5']:.3f} ({r['n_tags_all5']}) en {r['T_beyond_tags_en']:.3f} ({r['n_tags_en']})  "
        f"{'ELIGIBLE' if r['eligible'] else 'not eligible'}")
    return r


# ---------------------------------------------------------------- encoders
meta, src = {}, {}
for d in KOUTS:  # later dirs win (e.g. v3 re-runs an encoder that failed in v2)
    kids = json.loads((d / "study_ids.json").read_text())
    pos = {s: i for i, s in enumerate(kids)}
    assert all(s in pos for s in ids), f"{d}: kernel ids do not cover the training folds"
    for name, m in json.loads((d / "meta.json").read_text()).items():
        if "error" not in m or name not in meta:
            meta[name], src[name] = m, (d, np.array([pos[s] for s in ids]))
log("kernel meta", {k: {kk: v.get(kk) for kk in ("dim", "truncated_reports", "seconds", "error")} for k, v in meta.items()})
results = {}
local_e5 = np.load(PRE / "emb_e5_report.npy", allow_pickle=False)
for name in meta:
    d, sel = src[name]
    p = d / f"emb_{name}.npy"
    if "error" in meta[name] or not p.exists():
        log(f"{name}: no embeddings ({meta[name].get('error')})")
        continue
    E = np.load(p, allow_pickle=False)[sel].astype(np.float64)
    if name == "e5_base":
        cos = float((E * local_e5).sum(1).mean())
        results["parity_e5_base_mean_cosine"] = cos
        log(f"parity e5_base vs local T-pre cache: mean cosine {cos:.4f} ({'OK' if cos >= 0.99 else 'FAIL: bake-off void'})")
    results[name] = score(name, E)

eligible = {k: v for k, v in results.items() if isinstance(v, dict) and v.get("eligible")}
if results.get("parity_e5_base_mean_cosine", 0) < 0.99:
    decision = "VOID: e5_base parity failed"
elif not eligible:
    decision = "arm B NO-GO for all encoders"
else:
    top = max(v["T_beyond_tags_all5"] for v in eligible.values())
    near = {k: v for k, v in eligible.items() if v["T_beyond_tags_all5"] >= top - 0.005}
    decision = f"arm-B encoder: {max(near, key=lambda k: near[k]['lolo_centred'])}"
results["decision"] = decision
log("DECISION:", decision)
(OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
(OUT / "summary.txt").write_text("\n".join(log_lines) + "\n")
