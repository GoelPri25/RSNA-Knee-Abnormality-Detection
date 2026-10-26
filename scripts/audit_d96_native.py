"""Exp 4b: native-capped input (d96n) + 1e per-target labels vs Exp 1e (two seeds). Pre-registered primary:
SWA 21-25, no TTA, 10-target fold 0, mean delta vs the two 1e seeds >= +0.003 and >= 90% of 2,000 paired bootstraps
better against each seed. Secondary: 12-target, per target, mirror TTA, report-severity band split, L4 swap, gold-58.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_native.py /proj [run_dir name]
Optional run_dir / name score another 1e-recipe variant the same way (Exp 5a: run_r50_ssl_pertarget 5a); the
clear-band screen (2-band, targets with >= 20 clear positives) is printed for every run.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
RUN = sys.argv[2] if len(sys.argv) > 2 else "run_r50_native_pertarget"; NAME = sys.argv[3] if len(sys.argv) > 3 else "4b"
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]; ALL = list(range(12))


def load(p, order=ids):
    d = json.loads(Path(p).read_text())
    m = {r["study_id"]: r["probabilities"] for r in d["predictions"]} if "predictions" in d else dict(zip(d["study_ids"], d["scores"]))
    tg = d.get("targets_in_order", d.get("targets", T)); a = np.array([m[s] for s in order], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(S, idx):
    return np.array([auc(S[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(12)])


rng = np.random.default_rng(0); n = len(ids); full = np.arange(n)
boots = [rng.integers(0, n, n) for _ in range(2000)]


def compare(A, B, cols):
    d = np.array([per(A, b)[cols].mean() - per(B, b)[cols].mean() for b in boots])
    return per(A, full)[cols].mean() - per(B, full)[cols].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()



S1, S2 = load(C / "run_r50_pertarget/val_swa.json"), load(C / "run_r50_pertarget_seed/val_swa.json")
N = load(C / RUN / "val_swa.json")
print(f"PRIMARY (pre-registered): {NAME} SWA 21-25 vs each 1e seed, 10-target; PASS if mean delta >= +0.003 and >= 90% better vs each")
res = {}
for lab, cols in (("10-target", TEN), ("12-target", ALL)):
    for sl, S in (("s2026", S1), ("s2027", S2)):
        d, lo, hi, sh = compare(N, S, cols); res[(lab, sl)] = (d, sh)
        print(f"  {lab} {NAME} {per(N, full)[cols].mean():.4f} vs 1e {sl} {per(S, full)[cols].mean():.4f}: "
              f"delta {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")
m10 = (res[("10-target", "s2026")][0] + res[("10-target", "s2027")][0]) / 2
ok = m10 >= 0.003 and min(res[("10-target", s)][1] for s in ("s2026", "s2027")) >= 0.9
print(f"  mean 10-target delta vs the two seeds {m10:+.4f} -> {'PASS' if ok else 'FAIL'}")
MN = load(C / RUN / "val_swa_mirror.json"); M1 = load(C / "run_r50_pertarget/val_swa_mirror.json")
d, lo, hi, sh = compare(MN, M1, TEN); print(f"mirror TTA 10-target vs 1e s2026 mirror: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
hn = {r["epoch"]: r["val_macro_auc"] for r in json.loads((C / RUN / "history.json").read_text())}
h1 = {r["epoch"]: r["val_macro_auc"] for r in json.loads((C / "run_r50_pertarget/history.json").read_text())}
h2 = {r["epoch"]: r["val_macro_auc"] for r in json.loads((C / "run_r50_pertarget_seed/history.json").read_text())}
print(f"per epoch 12-target ({NAME} / s2026 / s2027): " + "; ".join(f"{e}: {hn[e]:.4f}/{h1[e]:.4f}/{h2[e]:.4f}" for e in sorted(hn) if e % 4 == 1 or e >= 20))
avg = (S1 + S2) / 2
pt = pd.DataFrame({NAME: per(N, full), "1e s2026": per(S1, full), "1e s2027": per(S2, full)}, index=T)
pt[f"{NAME} - seed mean"] = pt[NAME] - (pt["1e s2026"] + pt["1e s2027"]) / 2
pt["seed |diff|"] = (pt["1e s2026"] - pt["1e s2027"]).abs()
pt.loc["MACRO12"] = pt.iloc[:12].mean(); pt.loc["MACRO10"] = pt.iloc[TEN].mean()
print("\nPer target:"); print(pt.round(4).to_string())

# band split: clear (moderate+) positives vs mild / hedged / reader-absent positives, each against all negatives
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
band = np.full((len(ids), 12), "", object)
for i, s in enumerate(ids):
    for k, t in enumerate(T):
        if s in st.index:
            stt, sev = st.at[s, f"{key(t)}__status"], st.at[s, f"{key(t)}__severity"]
            band[i, k] = ("clear" if stt == "present" and sev in ("moderate", "moderate_to_severe", "severe") else
                          "unspec" if stt == "present" and sev in ("unspecified", "none") else "mild")


def band_auc(S, which, cols):
    out = []
    for k in cols:
        sel = mask[:, k] & (~truth[:, k] | (band[:, k] == which))
        out.append(auc(S[sel, k], truth[sel, k]) if (truth[sel, k]).sum() >= 5 else np.nan)
    return np.nanmean(out)


print("\nBand split (10-target mean; positives of the band vs all confident negatives):")
for which in ("clear", "mild", "unspec"):
    print(f"  {which:7s} {NAME} {band_auc(N, which, TEN):.4f}  1e s2026 {band_auc(S1, which, TEN):.4f}  "
          f"s2027 {band_auc(S2, which, TEN):.4f}")
# 2-band screen (pre-registered gate for 5a): clear-band AUC on the targets with >= 20 clear positives
cb = [k for k in range(12) if (mask[:, k] & truth[:, k] & (band[:, k] == "clear")).sum() >= 20]
print(f"Clear-band screen ({[T[k] for k in cb]}): {NAME} {band_auc(N, 'clear', cb):.4f}  1e s2026 "
      f"{band_auc(S1, 'clear', cb):.4f}  s2027 {band_auc(S2, 'clear', cb):.4f}  (5a gate >= 0.9690)")
rk = lambda a: pd.DataFrame(a).rank().to_numpy()
print(f"rank corr {NAME} vs 1e s2026:", round(float(np.mean([np.corrcoef(rk(S1)[:, k], rk(N)[:, k])[0, 1] for k in range(12)])), 3))

P = {"d2_224": load(D / "r50_tta/kaggle_output/d2_ep15__none.json"), "1b": load(C / "run_continue/val_swa.json"),
     "2c": load(C / "run_r50_plane/val_swa.json")}
L4 = np.mean([P["d2_224"], P["1b"], P["2c"], S1], 0)
for label, M in ((f"L4, 1e -> {NAME}", np.mean([P["d2_224"], P["1b"], P["2c"], N], 0)),
                 (f"L4 + {NAME} (5 equal)", np.mean([P["d2_224"], P["1b"], P["2c"], S1, N], 0))):
    d, lo, hi, sh = compare(M, L4, TEN)
    print(f"{label:22s} vs L4 (10-target): {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {sh:.1%} better")

rows = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in rows]; gy = np.array([[int(r[t]) for t in T] for r in rows]).astype(bool)
g = {lab: [auc(load(C / run / "gold_swa.json", gids)[:, k], gy[:, k]) for k in range(12)]
     for lab, run in ((NAME, RUN), ("1e s2026", "run_r50_pertarget"), ("1e s2027", "run_r50_pertarget_seed"))}
g = pd.DataFrame(g, index=T); g.loc["MACRO"] = g.mean()
print("\nGold-58 (development only; per-target noise about +-0.04):"); print(g.round(3).to_string())
