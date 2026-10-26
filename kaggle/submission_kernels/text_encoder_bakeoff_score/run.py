"""Ledger row T-enc (amended): embed reports with five encoders and score them with the T-pre / T-pre3 checks on a T4.

Reads the competition train.csv inside Kaggle (report text never leaves the kernel; internet only downloads encoder
weights). Derived inputs come from the private dataset tumis001/rsna-text-bakeoff-inputs. Training folds 1-5 only
(fold 0 and gold-58 are never used). Probes are the scikit-learn protocol re-implemented in PyTorch:
standardise on the outer-train split; L2 logistic regression (sum log-loss + ||w||^2 / 2C, intercept unpenalised),
C in {0.01, 0.1, 1, 10} picked by 3-fold StratifiedKFold inner CV on AUC, then refit; kNN k=25 cosine, uniform vote.
Parity: the local e5-base cache is scored too and must match the CPU values; in-kernel e5-base must match the cache.
Output: /kaggle/working/summary.json (scores only).
"""
import csv
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
    from sentence_transformers import SentenceTransformer

DEV = "cuda"
OUT = Path("/kaggle/working")
ROOTS = (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
         Path("/kaggle/input/rsna-knee-abnormality-detection"))
INP = next(p.parent for p in Path("/kaggle/input").rglob("emb_e5_base_local.npy"))
ENCODERS = {  # name -> (hub id, max tokens, prefix, batch, fp32)
    "e5_base": ("intfloat/multilingual-e5-base", 512, "passage: ", 16, False),
    "labse": ("sentence-transformers/LaBSE", 512, "", 16, False),
    "e5_large": ("intfloat/multilingual-e5-large", 512, "passage: ", 16, False),
    "bge_m3": ("BAAI/bge-m3", 2048, "", 16, False),
    "qwen3_emb_06b": ("Qwen/Qwen3-Embedding-0.6B", 2048, "", 2, True),
}
CPU_REF = {"label_probe": 0.9435, "lolo_centred": 0.8407, "language_variance_share": 0.460,
           "raw_site_linear": 0.815, "raw_site_knn": 0.809}  # T-pre / T-pre3 scikit-learn values, local e5-base
STATES = ["SIL", "NEG", "BELOW", "POS", "BORD", "HEDGE", "LIKELY"]
TAG_LANGS = ["en", "es", "de", "fr", "nl"]
POS, NEG_T, MIN_CLASS = 0.7, 0.15, 30
CS = [0.01, 0.1, 1.0, 10.0]
csv.field_size_limit(10 ** 8)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- data (same selection and order as the CPU scripts)
folds = {r["study_id"]: int(r["fold"]) for r in csv.DictReader(open(INP / "folds.csv"))}
lab = json.load(open(INP / "soft_labels_final_per_target_latD2.json"))
TARGETS = lab["targets_in_order"]
labels = {r["study_id"]: np.array(r["probabilities"]) for r in lab["labels"] if folds.get(r["study_id"], 0) != 0}
ids = sorted(labels)
assert len(ids) == 3524
n = len(ids)
root = next(p for p in ROOTS if (p / "train.csv").is_file())
reports = {r["StudyInstanceUID"]: r["Report"] for r in csv.DictReader(open(root / "train.csv", encoding="utf-8"))}
assert all(s in reports for s in ids)
states = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(INP / "rule_states.csv"))}
hdr = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(INP / "header_tags.csv"))}
LANG = json.load(open(INP / "languages.json"))
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
log("tag positives", {k: int(v.sum()) for k, v in TAGV.items()})
pairs = []
for l in LANGS:
    rows = np.where(LG == l)[0]
    models = [m for m, c in Counter(MODEL[rows].tolist()).items() if c >= 30 and m != "NA"]
    if len(models) >= 2:
        pairs += [(rows, (MODEL[rows] == m).astype(int)) for m in models if (MODEL[rows] != m).sum() >= 30]
log(f"studies {n}; site pairs {len(pairs)}")


# ---------------------------------------------------------------- GPU probes
def fit_lr(X, y, Cs):
    """L2 logistic regression for several C at once (independent columns). X (n,d), y (n,) tensors on GPU."""
    d = X.shape[1]
    W = torch.zeros(d, len(Cs), device=DEV, requires_grad=True)
    b = torch.zeros(len(Cs), device=DEV, requires_grad=True)
    pen = 0.5 / torch.tensor(Cs, device=DEV)
    Yk = y[:, None].expand(-1, len(Cs))
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=1000, tolerance_grad=1e-5, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ W + b
        loss = Fn.binary_cross_entropy_with_logits(z, Yk, reduction="sum") + (pen * (W * W).sum(0)).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def lr_predict(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0, unbiased=False)
    sd = torch.where(sd > 0, sd, torch.ones_like(sd))
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    yn = ytr.cpu().numpy()
    scores = np.zeros(len(CS))
    for itr, iva in StratifiedKFold(n_splits=3).split(np.zeros(len(yn)), yn):
        W, b = fit_lr(Xtr[itr], ytr[itr], CS)
        z = (Xtr[iva] @ W + b).cpu().numpy()
        scores += [roc_auc_score(yn[iva], z[:, j]) for j in range(len(CS))]
    best = CS[int(np.argmax(scores))]
    W, b = fit_lr(Xtr, ytr, [best])
    return torch.sigmoid(Xte @ W + b)[:, 0].cpu().numpy()


def knn_predict(Xtr, ytr, Xte, k=25):
    sim = Fn.normalize(Xte, dim=1) @ Fn.normalize(Xtr, dim=1).T
    return ytr[sim.topk(k, dim=1).indices].mean(1).cpu().numpy()


def cv_auc(kind, X, y, groups, test_groups=(1, 2, 3, 4)):
    Xg = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    yg = torch.as_tensor(y, dtype=torch.float32, device=DEV)
    pred = np.full(len(y), np.nan)
    for g in test_groups:
        tr, te = groups != g, groups == g
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        f = lr_predict if kind == "linear" else knn_predict
        pred[te] = f(Xg[torch.as_tensor(tr, device=DEV)], yg[torch.as_tensor(tr, device=DEV)],
                     Xg[torch.as_tensor(te, device=DEV)])
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
        v = cv_auc("linear", X[keep], yb[keep], g[keep], test_groups)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals))


def strip(E, X):
    A = np.hstack([X, np.ones((n, 1))])
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    return E - A @ coef


def site(X, kind):
    vals = [cv_auc(kind, X[rows], y, F[rows]) for rows, y in pairs]
    return float(np.mean([v for v in vals if v is not None]))


def score(name, E, raw_site=False):
    t = time.time()
    r = {}
    mu = E.mean(0)
    r["language_variance_share"] = float(sum((LG == l).sum() * ((E[LG == l].mean(0) - mu) ** 2).sum()
                                             for l in LANGS) / ((E - mu) ** 2).sum())
    r["label_probe"] = label_probe(E)
    Ec = E.copy()
    for l in LANGS:
        Ec[LG == l] -= E[LG == l].mean(0)
    r["lolo_centred"] = label_probe(Ec, groups=LG, test_groups=[l for l, c in Counter(LG.tolist()).items() if c >= 50])
    if raw_site:
        r["raw_site_linear"], r["raw_site_knn"] = site(E, "linear"), site(E, "knn")
    T_full = strip(E, X_site)
    T_beyond = strip(E, np.hstack([X_site, X_lab]))
    r["T_full_site_linear"], r["T_full_site_knn"] = site(T_full, "linear"), site(T_full, "knn")
    t5 = [cv_auc("linear", T_beyond[tag_rows], y, F[tag_rows]) for y in TAGV.values() if y.sum() >= 30]
    ten = [cv_auc("linear", T_beyond[tag_rows][en_rows], y[en_rows], F[tag_rows][en_rows])
           for y in TAGV.values() if y[en_rows].sum() >= 30]
    t5, ten = [v for v in t5 if v is not None], [v for v in ten if v is not None]
    r["T_beyond_tags_all5"], r["n_tags_all5"] = float(np.mean(t5)), len(t5)
    r["T_beyond_tags_en"], r["n_tags_en"] = float(np.mean(ten)), len(ten)
    r["gates"] = {"site_linear<=0.65": r["T_full_site_linear"] <= 0.65, "site_knn<=0.65": r["T_full_site_knn"] <= 0.65,
                  "tags_all5>=0.80": r["n_tags_all5"] >= 4 and r["T_beyond_tags_all5"] >= 0.80,
                  "tags_en>=0.75": r["T_beyond_tags_en"] >= 0.75}
    r["eligible"] = all(r["gates"].values())
    r["seconds"] = round(time.time() - t, 1)
    log(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    return r


# ---------------------------------------------------------------- parity: local e5-base cache through the GPU probes
results = {}
local = np.load(INP / "emb_e5_base_local.npy", allow_pickle=False).astype(np.float64)
results["e5_base_local_cache"] = score("e5_base_local_cache", local, raw_site=True)
par = {k: round(results["e5_base_local_cache"][k] - v, 4) for k, v in CPU_REF.items()}
results["probe_parity_vs_cpu"] = par
log("probe parity (GPU - CPU)", par, "OK" if all(abs(v) <= 0.005 for v in par.values()) else "FAIL")

# ---------------------------------------------------------------- embed + score each encoder
texts = [reports[s].strip() for s in ids]
for name, (hub, max_len, prefix, bs, fp32) in ENCODERS.items():
    t = time.time()
    try:
        model = SentenceTransformer(hub, device=DEV, trust_remote_code=False)
        model.max_seq_length = max_len
        if not fp32:
            model.half()
        E = model.encode([prefix + x for x in texts], batch_size=bs, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype(np.float64)
        assert np.isfinite(E).all()
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        results[name] = {"hub": hub, "error": f"{type(e).__name__}: {e}"[:400]}
        log(name, results[name])
        continue
    log(f"{name} embedded in {time.time() - t:.0f}s, dim {E.shape[1]}")
    if name == "e5_base":
        results["embed_parity_e5_base_mean_cosine"] = float((E * local).sum(1).mean())
        log("embed parity e5_base vs local cache", round(results["embed_parity_e5_base_mean_cosine"], 4))
    results[name] = score(name, E)
    results[name]["hub"] = hub
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")

ok_probe = all(abs(v) <= 0.005 for v in par.values())
ok_embed = results.get("embed_parity_e5_base_mean_cosine", 0) >= 0.99
elig = {k: v for k, v in results.items() if k in ENCODERS and isinstance(v, dict) and v.get("eligible")}
if not (ok_probe and ok_embed):
    decision = "VOID: parity failed (CPU script decides)"
elif not elig:
    decision = "arm B NO-GO for all encoders"
else:
    top = max(v["T_beyond_tags_all5"] for v in elig.values())
    near = {k: v for k, v in elig.items() if v["T_beyond_tags_all5"] >= top - 0.005}
    decision = f"arm-B encoder: {max(near, key=lambda k: near[k]['lolo_centred'])}"
results["decision"] = decision
log("DECISION:", decision)
(OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
