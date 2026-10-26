import csv, json
from pathlib import Path
import numpy as np, pandas as pd
D = Path("data/coatnet_d96_fold0"); A = D / "assets"; P = D / "pubkd"
lv = json.loads((A / "soft_labels_final_per_target_latD2.json").read_text()); T = lv["targets_in_order"]
lab = {r["study_id"]: (np.array([np.nan if p is None else p for p in r["probabilities"]], float), np.array(r["weights"], float)) for r in lv["labels"]}
fold = {x["study_id"]: x["fold"] for x in csv.DictReader((A / "folds.csv").open())}
tg = {r["study_id"]: np.array(r["probabilities"]) for r in json.loads((P / "pubkd_targets_raptor_pure.json").read_text())["labels"]}
def mem(m):
    out = {}
    for f in sorted(P.glob(f"raptor{m}_*.json")):
        for s, v in json.loads(f.read_text())["pred"].items(): out[s] = (np.array(v["plain"]) + np.array(v["mirror"])) / 2
    return out
r384, r224 = mem("384"), mem("224")
H = lambda p: -(p * np.log(np.clip(p, 1e-7, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-7, 1)))
tr = [s for s in tg if fold.get(s) not in (None, "0") and s in lab]
Yt = np.array([tg[s] for s in tr]); Yl = np.array([lab[s][0] for s in tr]); Wl = np.array([lab[s][1] for s in tr]); m = (Wl > 0) & np.isfinite(Yl)
print("## 1 how soft are the targets (training folds)")
print(f"  teacher pure: mean entropy {H(Yt).mean():.3f} nats; cells <0.05 or >0.95: {np.mean((Yt < .05) | (Yt > .95)):.1%}; in [0.2, 0.8]: {np.mean((Yt > .2) & (Yt < .8)):.1%}")
print(f"  label-v2 (weighted cells): mean entropy {H(Yl[m]).mean():.3f}; cells <0.05 or >0.95: {np.mean((Yl[m] < .05) | (Yl[m] > .95)):.1%}; in [0.2, 0.8]: {np.mean((Yl[m] > .2) & (Yl[m] < .8)):.1%}")
conf = m & ((Yl <= .3) | (Yl >= .7))
agree = ((Yt >= .5) == (Yl >= .7))[conf & ((Yl <= .3) | (Yl >= .7))]
print(f"  teacher vs label-v2 on label-v2-confident cells: same side of 0.5 {agree.mean():.1%}; AUC of teacher vs those labels (12t mean) " +
      f"{np.nanmean([pd.Series(Yt[conf[:, j], j]).rank().pipe(lambda r: (r[Yl[conf[:, j], j] >= .7].sum() - (Yl[conf[:, j], j] >= .7).sum() * ((Yl[conf[:, j], j] >= .7).sum() + 1) / 2) / max(1, (Yl[conf[:, j], j] >= .7).sum() * (Yl[conf[:, j], j] < .7).sum())) for j in range(12)]):.4f}")
print("  per finding teacher entropy / label-v2 entropy: " + " ".join(f"{T[j][:7]} {H(Yt[:, j]).mean():.2f}/{H(Yl[m[:, j], j]).mean():.2f}" for j in range(12)))
print("\n## 2 how well did the student fit its targets (train loss at epoch 25 vs the soft-target entropy floor)")
for name, d, floor in (("ra pure", P / "students_r/pubkd_kd_ra", H(Yt).mean()), ("rb mix73", P / "students_r/pubkd_kd_rb", None),
                       ("v4 label-v2", D / "run_r50_gl4_mlx/v4_l4copy_s2026", H(Yl[m]).mean())):
    h = json.loads((d / "history.json").read_text()); h = h if isinstance(h, list) else h.get("epochs", h)
    tl = h[-1]["train_loss"]
    print(f"  {name}: train BCE global {tl['global']:.3f} local {tl['local']:.3f}" + (f" | floor {floor:.3f} -> excess (KL) global {tl['global'] - floor:.3f}" if floor else ""))
print("\n## 3 fold-0 agreement with the teacher (Spearman per finding; teacher in-sample on fold 0)")
def val(p):
    d = json.loads(Path(p).read_text()); pr = d["predictions"]
    return {s: np.array(v) for s, v in (pr.items() if isinstance(pr, dict) else ((r["study_id"], r["probabilities"]) for r in pr))}
f0 = sorted(s for s in tg if fold.get(s) == "0" and s in lab)
mods = {"ra": val(P / "students_r/pubkd_kd_ra/val_swa_fused.json"), "rc": val(P / "students_r/pubkd_kd_rc/val_swa_fused.json"),
        "v4": val(D / "run_r50_gl4_mlx/v4_l4copy_s2026/val_swa_fused.json")}
f0 = [s for s in f0 if all(s in v for v in mods.values())]
Tt = np.array([tg[s] for s in f0]); sp = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())
rows = {k: [sp(np.array([v[s][j] for s in f0]), Tt[:, j]) for j in range(12)] for k, v in mods.items()}
rows["r384~r224"] = [sp(np.array([r384[s][j] for s in f0]), np.array([r224[s][j] for s in f0])) for j in range(12)]
rows["ra~rc"] = [sp(np.array([mods["ra"][s][j] for s in f0]), np.array([mods["rc"][s][j] for s in f0])) for j in range(12)]
print("  " + " ".join(f"{t[:7]:>7}" for t in T) + "   mean")
for k, r in rows.items(): print(f"  {k:10s}" + " ".join(f"{x:7.3f}" for x in r) + f"  {np.mean(r):.3f}")
print("\n## 4 where student and teacher disagree on fold 0 (|rank pct diff|, by label-v2 band)")
Y0 = np.array([lab[s][0] for s in f0]); W0 = np.array([lab[s][1] for s in f0]); ok0 = (W0 > 0) & np.isfinite(Y0)
pc = lambda X: np.column_stack([pd.Series(X[:, j]).rank(pct=True).to_numpy() for j in range(12)])
S = pc(np.array([mods["ra"][s] for s in f0])); Tp = pc(Tt); V = pc(np.array([mods["v4"][s] for s in f0]))
for nm, band in (("negative <=0.3", ok0 & (Y0 <= .3)), ("mild 0.3-0.7", ok0 & (Y0 > .3) & (Y0 < .7)), ("positive >=0.7", ok0 & (Y0 >= .7))):
    print(f"  {nm:15s} cells {band.sum():5d}: |ra - teacher| {np.abs(S - Tp)[band].mean():.3f}   |v4 - teacher| {np.abs(V - Tp)[band].mean():.3f}   |ra - v4| {np.abs(S - V)[band].mean():.3f}")
