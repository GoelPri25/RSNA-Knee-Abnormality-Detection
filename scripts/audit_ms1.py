"""MS1 (ledger): MedSigLIP-448 fine-tuned on label-v2, pre-registered fold-0 readout + learning diagnostics. CPU, read-only.

Primary (SWA 21-25, 94 windows, no TTA; clear6 / 10-target exactly as audit_clear_band_lb.py via audit_phase1.py):
  (1) single submit if clear6 >= 0.9739 or 10-target >= 0.9124 (+0.005 over label-v2 with >= 90% paired bootstraps)
  (2) ensemble: E3 + MS1 and the best 4-member swap; submit the leaner if clear6 >= E3 + 0.002 with >= 90% bootstraps
Diagnostics (no selection): per target vs label-v2 / CK-A / E3, mild band, gold-58, epoch curve, error correlation of MS1
with each E3 member (does it make different mistakes?), pair mean(label-v2, MS1).

Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_ms1.py <project root>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
src = (R / "repository/scripts/audit_phase1.py").read_text()
sys.argv = [sys.argv[0], str(R)]
exec(src.split("def per_target")[0])                       # ids, T, mask, truth, clear, CB, TEN, load, auc, scores
OUT = C / "run_sig_lv2"; lines = []
say = lambda s="": (print(s), lines.append(s))
TM = D / "teammate_pp4/fold0_predictions"
P = {"MS1": load(OUT / "val_swa.json"), "lv2": load(C / "kaggle_fold0_labelv2/val_swa.json"),
     "4net": load(TM / "four_network_fold0_val_swa.json"), "1b": load(C / "run_continue/val_swa.json"),
     "cnx": load(C / "run_cnx_pertarget/val_swa.json"), "ckA": load(C / "run_cnx_in12k/val_swa.json"),
     "1e": load(C / "run_r50_pertarget/val_swa.json")}
E3M = ["lv2", "4net", "1b", "cnx"]
P["E3"] = np.mean([P[m] for m in E3M], 0)
P["E3+MS1"] = np.mean([P[m] for m in E3M + ["MS1"]], 0)
for m in E3M:
    P[f"E3 swap {m}->MS1"] = np.mean([P[x] for x in E3M if x != m] + [P["MS1"]], 0)
P["lv2+MS1"] = np.mean([P["lv2"], P["MS1"]], 0)
P["lv2+ckA"] = np.mean([P["lv2"], P["ckA"]], 0)

al = np.arange(len(ids))
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
S0 = {n: scores(S, al) for n, S in P.items()}


def paired(a, b):
    d = np.array([np.subtract(scores(P[a], i), scores(P[b], i)) for i in boots])
    return d


say("MS1 MedSigLIP-448 on label-v2: fold-0 readout (SWA 21-25, Steven-v4 ruler)\n")
say(f"{'model':26s} clear6   10-target")
for n in ["MS1", "lv2", "ckA", "1e", "E3", "E3+MS1"] + [f"E3 swap {m}->MS1" for m in E3M] + ["lv2+MS1", "lv2+ckA"]:
    say(f"  {n:24s} {S0[n][0]:.4f}   {S0[n][1]:.4f}")

say("\nPRIMARY (1) single vs label-v2 (gates: clear6 >= 0.9739 or 10-target >= 0.9124)")
d = paired("MS1", "lv2")
for j, nm in enumerate(("clear6", "10-target")):
    say(f"  {nm:9s} {S0['MS1'][j] - S0['lv2'][j]:+.4f} [{np.percentile(d[:, j], 2.5):+.4f}, {np.percentile(d[:, j], 97.5):+.4f}] "
        f"{(d[:, j] > 0).mean():.0%} better")
single = S0["MS1"][0] >= 0.9739 or S0["MS1"][1] >= 0.9124
say(f"  -> single {'PASS' if single else 'FAIL'}")

say("\nPRIMARY (2) ensembles vs E3 (gate: clear6 >= E3 + 0.002 with >= 90% bootstraps)")
best = None
for n in ["E3+MS1"] + [f"E3 swap {m}->MS1" for m in E3M]:
    d = paired(n, "E3")
    ok = (S0[n][0] - S0["E3"][0] >= 0.002) and (d[:, 0] > 0).mean() >= 0.9
    say(f"  {n:24s} clear6 {S0[n][0] - S0['E3'][0]:+.4f} [{np.percentile(d[:, 0], 2.5):+.4f}, {np.percentile(d[:, 0], 97.5):+.4f}] "
        f"{(d[:, 0] > 0).mean():.0%};  10-target {S0[n][1] - S0['E3'][1]:+.4f} [{np.percentile(d[:, 1], 2.5):+.4f}, "
        f"{np.percentile(d[:, 1], 97.5):+.4f}] {(d[:, 1] > 0).mean():.0%}  {'PASS' if ok else ''}")
say("  diagnostic pair: lv2+MS1 vs lv2+ckA (is MedSigLIP a better partner than a ConvNeXt?)")
d = np.array([np.subtract(scores(P["lv2+MS1"], i), scores(P["lv2+ckA"], i)) for i in boots])
say(f"    clear6 {S0['lv2+MS1'][0] - S0['lv2+ckA'][0]:+.4f} [{np.percentile(d[:, 0], 2.5):+.4f}, {np.percentile(d[:, 0], 97.5):+.4f}];"
    f"  10-target {S0['lv2+MS1'][1] - S0['lv2+ckA'][1]:+.4f} [{np.percentile(d[:, 1], 2.5):+.4f}, {np.percentile(d[:, 1], 97.5):+.4f}]")

say("\nPER TARGET (fold 0, all confident cells) and mild band (ruler-positive, report not moderate+)")
mild = mask & truth & ~clear


def pt(S, sel=None):
    out = []
    for k in range(12):
        neg = np.where(mask[:, k] & ~truth[:, k])[0]
        pos = np.where((mask[:, k] & truth[:, k]) if sel is None else sel[:, k])[0]
        out.append(auc(S[np.r_[pos, neg], k], np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]) if len(pos) >= 10 else np.nan)
    return out


tab = pd.DataFrame({n: pt(P[n]) for n in ["MS1", "lv2", "ckA", "E3", "E3+MS1"]}, index=T)
tab["MS1-lv2"] = tab.MS1 - tab.lv2; tab["MS1-ckA"] = tab.MS1 - tab.ckA
tab["mild MS1"] = pt(P["MS1"], mild); tab["mild lv2"] = pt(P["lv2"], mild)
say(tab.round(3).to_string())

say("\nERROR CORRELATION: Spearman correlation of (prediction - ruler) residuals on confident cells, per target, averaged")
res = {n: P[n] - truth.astype(float) for n in ["MS1", "lv2", "4net", "1b", "cnx", "ckA", "1e"]}


def rcorr(a, b):
    v = []
    for k in range(12):
        m = mask[:, k]
        v.append(pd.Series(res[a][m, k]).rank().corr(pd.Series(res[b][m, k]).rank()))
    return np.mean(v)


names = list(res)
cm = pd.DataFrame([[rcorr(a, b) for b in names] for a in names], index=names, columns=names)
say(cm.round(3).to_string())

say("\nEPOCH CURVE (runner log, 12-target val macro, val BCE, train loss)")
h = json.loads((OUT / "history.json").read_text())
for e in h:
    if e.get("val_macro_auc"):
        say(f"  ep {e['epoch']:2d}  val {e['val_macro_auc']:.4f}  bce {e.get('val_bce', float('nan')):.4f}  train {e['train_loss']:.4f}")

rows_g = [r for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8")) if all(r[t] in ("0", "1") for t in T)]
gids = [r["StudyInstanceUID"] for r in rows_g]; gy = np.array([[int(r[t]) for t in T] for r in rows_g]).astype(bool)


def gload(p):
    g = json.loads(Path(p).read_text()); m = {x["study_id"]: x["probabilities"] for x in g["predictions"]}
    return np.array([m[s] for s in gids])[:, [g["targets_in_order"].index(t) for t in T]]


G = {"MS1": gload(OUT / "gold_swa.json"), "ckA": gload(C / "run_cnx_in12k/gold_swa.json"),
     "1e": gload(C / "run_r50_pertarget/gold_swa.json"), "1b": gload(C / "run_continue/gold_swa.json"),
     "cnx": gload(C / "run_cnx_pertarget/gold_swa.json")}
G["E3*"] = np.mean([G["1e"], G["1b"], G["cnx"]], 0); G["E3*+MS1"] = np.mean([G["1e"], G["1b"], G["cnx"], G["MS1"]], 0)
say("\nGOLD-58 12-target macro (development evidence only; E3* = 1e stands in for label-v2, no 4-network gold file)")
say("  " + "  ".join(f"{n} {np.nanmean([auc(S[:, k], gy[:, k]) for k in range(12)]):.3f}" for n, S in G.items()))
(OUT / "audit_ms1.txt").write_text("\n".join(lines) + "\n")
