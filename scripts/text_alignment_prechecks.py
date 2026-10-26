"""Ledger row T-pre: CPU pre-checks for a training-only report-text alignment loss.

Does a frozen report encoder carry finding information that an image encoder could usefully align to, and is it
language-bound (language ~ site)? Training folds 1-5 only (fold 0 and gold-58 are never read). No model is trained
on images and no label is changed.

C1 coverage: per target, share of studies with a non-empty HPO-v3.1 rule-evidence span.
C2 whole-report probe: logistic-regression macro AUC on label-v2 confident cells, per encoder (TF-IDF, e5, RadBERT).
C3 language shortcut: between-language variance share, leave-one-language-out probe, language x manufacturer.
C4 per-finding span probe: e5 embedding of the evidence span, POS vs NEG among mentioned cells.

Usage (CPU only): CUDA_VISIBLE_DEVICES="" python -I text_alignment_prechecks.py <repository root>
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from langdetect import DetectorFactory, detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT / "data/text_alignment_prechecks"
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = ROOT / "data/coatnet_d96_fold0/assets"
E5 = ROOT / "data/pretrained_text/multilingual-e5-base"
RADBERT = ROOT / "data/pretrained_text/radbert-roberta-4m"
POS, NEG, MIN_CLASS = 0.7, 0.15, 30
csv.field_size_limit(10 ** 9)
torch.set_num_threads(max(1, torch.get_num_threads()))
assert not torch.cuda.is_available(), "CPU only: run with CUDA_VISIBLE_DEVICES=''"
log_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_lines.append(s)


# ---------------------------------------------------------------- data (training folds 1-5 only)
folds = {r["study_id"]: int(r["fold"]) for r in csv.DictReader(open(ASSETS / "folds.csv"))}
lab = json.load(open(ASSETS / "soft_labels_final_per_target_latD2.json"))
TARGETS = lab["targets_in_order"]
labels = {r["study_id"]: np.array(r["probabilities"]) for r in lab["labels"] if folds.get(r["study_id"], 0) != 0}
reports = {r["StudyInstanceUID"]: r["Report"].strip()
           for r in csv.DictReader(open(ROOT.parent / "reports_raw/train.csv")) if r["StudyInstanceUID"] in labels}
evid = {r["StudyInstanceUID"]: r for r in csv.DictReader(
    open(ROOT / "data/label_datasets/hpo_v3_1_d2/rules/s5_fx_site/rule_evidence.csv")) if r["StudyInstanceUID"] in labels}
hdr = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(ROOT / "data/dicom_demographics/study_header_tags.csv"))}
ids = sorted(labels)
assert len(ids) == 3524 and all(s in reports for s in ids) and all(s in evid for s in ids), (len(ids),)
Y = np.stack([labels[s] for s in ids])
F = np.array([folds[s] for s in ids])
TEXT = [reports[s] for s in ids]
log(f"studies {len(ids)} (folds 1-5), fold counts {dict(sorted(Counter(F.tolist()).items()))}")

lang_path = OUT / "languages.json"
if lang_path.exists():
    LANG = json.load(open(lang_path))
else:
    DetectorFactory.seed = 0
    LANG = {s: detect(reports[s]) for s in ids}
    lang_path.write_text(json.dumps(LANG) + "\n")
LG = np.array([LANG[s] for s in ids])
log("languages", dict(Counter(LG.tolist()).most_common()))


# ---------------------------------------------------------------- encoders
def embed(model_dir, texts, prefix="", bs=16, radbert=False):
    tok = AutoTokenizer.from_pretrained(model_dir)
    if radbert:  # pickle checkpoint: build from config, load tensors only
        model = AutoModel.from_config(AutoConfig.from_pretrained(model_dir))
        sd = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        sd = {k[len("roberta."):] if k.startswith("roberta.") else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"radbert load: missing {len(missing)} {missing[:4]} unexpected {len(unexpected)} {unexpected[:4]}")
    else:
        model = AutoModel.from_pretrained(model_dir, use_safetensors=True)
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            b = tok([prefix + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=512,
                    return_tensors="pt")
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
            if i % (bs * 40) == 0:
                print(f"  {model_dir.name} {i}/{len(texts)}", flush=True)
    return np.concatenate(out)


def cached(name, fn):
    p = OUT / f"emb_{name}.npy"
    if p.exists():
        return np.load(p)
    x = fn()
    np.save(p, x)
    return x


E_e5 = cached("e5_report", lambda: embed(E5, TEXT, prefix="passage: "))
en = np.where(LG == "en")[0]
E_rb = cached("radbert_report_en", lambda: embed(RADBERT, [TEXT[i] for i in en], radbert=True))
TF = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)
X_tf = TF.fit_transform(TEXT)  # unsupervised, fitted on training-fold text only
log(f"embeddings: e5 {E_e5.shape}, radbert(en) {E_rb.shape}, tfidf {X_tf.shape}")


# ---------------------------------------------------------------- probes
def clf(sparse):
    lr = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=3, scoring="roc_auc", max_iter=3000,
                              solver="liblinear" if sparse else "lbfgs")
    return lr if sparse else make_pipeline(StandardScaler(), lr)


def probe(X, y, groups, sparse=False, test_groups=None):
    """Out-of-group predictions for confident cells; returns AUC or None. groups: test group per row."""
    keep = (y >= POS) | (y <= NEG)
    yb = (y >= POS).astype(int)
    if min(yb[keep].sum(), (1 - yb[keep]).sum()) < MIN_CLASS:
        return None
    pred = np.full(len(y), np.nan)
    for g in (test_groups if test_groups is not None else sorted(set(groups.tolist()))):
        te = keep & (groups == g)
        tr = keep & (groups != g)
        if te.sum() == 0 or len(set(yb[tr])) < 2:
            continue
        pred[te] = clf(sparse).fit(X[tr], yb[tr]).predict_proba(X[te])[:, 1]
    m = keep & ~np.isnan(pred)
    if min(yb[m].sum(), (1 - yb[m]).sum()) < 10:
        return None
    return float(roc_auc_score(yb[m], pred[m]))


def probe_all(X, rows, sparse=False, groups=None, test_groups=None):
    g = F[rows] if groups is None else groups
    tg = [1, 2, 3, 4] if groups is None else test_groups
    res = {t: probe(X, Y[rows, j], g, sparse, tg) for j, t in enumerate(TARGETS)}
    vals = [v for v in res.values() if v is not None]
    return {"per_target": res, "macro": float(np.mean(vals)) if vals else None, "n_targets": len(vals)}


summary = {"ledger_row": "T-pre", "n_studies": len(ids), "languages": dict(Counter(LG.tolist()).most_common())}

# C1 coverage
cov = {t: float(np.mean([bool(evid[s][t].strip()) for s in ids])) for t in TARGETS}
summary["C1_coverage"] = cov
log("C1 coverage", {t: round(v, 3) for t, v in cov.items()})
log(f"C1 targets with coverage >= 0.30: {sum(v >= 0.30 for v in cov.values())}/12")

# C2 whole-report probes
allr = np.arange(len(ids))
c2 = {"tfidf": probe_all(X_tf, allr, sparse=True), "e5": probe_all(E_e5, allr)}
c2_en = {"tfidf": probe_all(X_tf[en], en, sparse=True), "e5": probe_all(E_e5[en], en),
         "radbert": probe_all(E_rb, en)}
summary["C2_all"], summary["C2_english"] = c2, c2_en
for k, v in c2.items():
    log(f"C2 all  {k:8s} macro {v['macro']:.4f} ({v['n_targets']} targets)")
for k, v in c2_en.items():
    log(f"C2 en   {k:8s} macro {v['macro']:.4f} ({v['n_targets']} targets)")

# C3 language shortcut
mu = E_e5.mean(0)
tot = ((E_e5 - mu) ** 2).sum()
between = sum((LG == l).sum() * ((E_e5[LG == l].mean(0) - mu) ** 2).sum() for l in set(LG.tolist()))
summary["C3_between_language_variance_share_e5"] = float(between / tot)
log(f"C3 e5 between-language variance share {between / tot:.3f}")
E_c = E_e5.copy()
for l in set(LG.tolist()):
    E_c[LG == l] -= E_e5[LG == l].mean(0)
langs = [l for l, n in Counter(LG.tolist()).items() if n >= 50]
c3 = {"e5_centred_in_language_cv": probe_all(E_c, allr),
      "e5_centred_lolo": probe_all(E_c, allr, groups=LG, test_groups=langs),
      "e5_raw_lolo": probe_all(E_e5, allr, groups=LG, test_groups=langs),
      "tfidf_lolo": probe_all(X_tf, allr, sparse=True, groups=LG, test_groups=langs)}
summary["C3_probes"] = c3
for k, v in c3.items():
    log(f"C3 {k:28s} macro {v['macro']:.4f} ({v['n_targets']} targets)")


def cramers_v(a, b):
    ca, cb = sorted(set(a)), sorted(set(b))
    tab = np.array([[sum(1 for x, y in zip(a, b) if x == i and y == j) for j in cb] for i in ca], float)
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    chi2 = ((tab - exp) ** 2 / np.where(exp > 0, exp, 1)).sum()
    return float(np.sqrt(chi2 / tab.sum() / max(1, min(len(ca), len(cb)) - 1)))


have = [i for i, s in enumerate(ids) if s in hdr]
for col in ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength"]:
    vals = [(hdr[ids[i]].get(col) or "NA").strip() or "NA" for i in have]
    v = cramers_v(LG[have].tolist(), vals)
    summary[f"C3_cramers_v_language_x_{col}"] = v
    log(f"C3 Cramer's V language x {col}: {v:.3f} (n {len(have)})")

# C4 per-finding span probe (mentioned cells only)
spans, owner = [], []
for i, s in enumerate(ids):
    for j, t in enumerate(TARGETS):
        if evid[s][t].strip():
            spans.append(evid[s][t].strip())
            owner.append((i, j))
E_sp = cached("e5_spans", lambda: embed(E5, spans, prefix="passage: ", bs=64))
owner = np.array(owner)
c4 = {}
for j, t in enumerate(TARGETS):
    r = np.where(owner[:, 1] == j)[0]
    if len(r) == 0:
        c4[t] = None
        continue
    rows = owner[r, 0]
    c4[t] = probe(E_sp[r], Y[rows, j], F[rows], test_groups=[1, 2, 3, 4])
vals = [v for v in c4.values() if v is not None]
summary["C4_span_probe_e5"] = {"per_target": c4, "macro": float(np.mean(vals)) if vals else None,
                               "n_targets": len(vals)}
log(f"C4 e5 span probe macro {summary['C4_span_probe_e5']['macro']:.4f} ({len(vals)} targets)",
    {t: (round(v, 3) if v is not None else None) for t, v in c4.items()})

# ---------------------------------------------------------------- pre-registered decision
e5, tf = c2["e5"]["macro"], c2["tfidf"]["macro"]
lolo = c3["e5_centred_lolo"]["macro"]
n_cov = sum(v >= 0.30 for v in cov.values())
span = summary["C4_span_probe_e5"]["macro"]
gates = {"C2 e5 >= 0.90": e5 >= 0.90, "C3 e5 centred LOLO >= 0.85": lolo >= 0.85,
         "C2 e5 >= tfidf - 0.01": e5 >= tf - 0.01}
go = all(gates.values())
variant = "B" if (n_cov >= 6 and span is not None and span >= 0.90) else "A"
summary["gates"], summary["GO"], summary["variant_if_go"] = gates, go, variant
log("gates", gates)
log(f"DECISION: {'GO (GPU spec next; user still chooses vs Phase 3)' if go else 'NO-GO'}; variant if GO: {variant} "
    f"(coverage>=0.30 on {n_cov}/12, span probe {span:.4f})")
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUT / "summary.txt").write_text("\n".join(log_lines) + "\n")
