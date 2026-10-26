"""PUB-KD label-set audit (CPU, read-only; development evidence, validation labels unchanged).

Q1  How do the three training label sets differ: Raptor blend targets (ra's teacher), full public blend targets (A's;
    B = 0.7 full + 0.3 label-v2) and label-v2? Per finding: Spearman between sets (all 4,407 studies and fold 0), AUC of
    each set against Steven-v4 confident cells on fold 0, mean target by report band (absent / mild / moderate+), and the
    Raptor-vs-full disagreement cells (opposite sides of 0.5) on Steven-confident fold-0 cells: whom does Steven back?
Q2  Is the LB predictor's ruler neutral between the teacher families? rapc = Steven cells Raptor agrees with; its mirror
    fullc = Steven cells the full blend agrees with; bothc = both agree; disc = the cells where they disagree.
    Each ruler scores the teachers themselves (bias = how much the ruler is built from one teacher) and the students
    A / B / ra at matched epochs (same-epoch pairs, fused files).
Usage: python3 -I scripts/audit_pubkd_labelsets.py  -> data/lb_predictor/labelsets_audit.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_predictor as L                                  # noqa: E402  (rulers built at import)

T, N, IDS, A, C = L.T, L.N, L.IDS, L.A, L.C
P = C / "pubkd"; S = P / "students_full"
OUT = L.OUT / "labelsets_audit.txt"
lines = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)


def labels_all(path):
    d = json.loads(Path(path).read_text()); assert d["targets_in_order"] == T
    return {r["study_id"]: np.array([.5 if v is None else v for v in r["probabilities"]], float) for r in d["labels"]}


SETS = {"rap": P / "pubkd_targets_raptor_pure.json", "full": P / "pubkd_targets_pure.json",
        "mix": P / "pubkd_targets_mix73.json", "lv2": A / "soft_labels_final_per_target_latD2.json"}
ALL = {k: labels_all(v) for k, v in SETS.items()}
common = sorted(set.intersection(*(set(v) for v in ALL.values())))
pr(f"studies: " + ", ".join(f"{k} {len(v)}" for k, v in ALL.items()) + f"; common {len(common)}; fold 0 {N}")
F0 = {k: np.array([ALL[k].get(s, np.full(12, np.nan)) for s in IDS]) for k in ALL}
for k in F0:
    assert not np.isnan(F0[k]).any(), f"{k} misses fold-0 studies"
AL = {k: np.array([ALL[k][s] for s in common]) for k in ALL}
sp = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())
t0, m0 = L.RL["steven"]; MOD = L.MODP
rep = pd.read_csv(L.R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                  dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
band = np.full((N, 12), "", object)
for i, s in enumerate(IDS):
    if s in rep.index:
        for j, t in enumerate(T):
            stt, sv = rep.at[s, f"{key(t)}__status"], rep.at[s, f"{key(t)}__severity"]
            band[i, j] = ("absent" if stt == "absent" else "mod+" if stt == "present" and sv in
                          ("moderate", "moderate_to_severe", "severe") else "mild" if stt == "present" else "silent")

# ------------------------------------------------------------------ Q1
pr("\n== Q1a Spearman between label sets per finding (all common studies | fold 0)")
pairs = [("rap", "full"), ("rap", "lv2"), ("full", "lv2"), ("mix", "lv2")]
pr(f"{'finding':17s} " + " ".join(f"{a}-{b:>4s}      " for a, b in pairs))
for j, t in enumerate(T):
    pr(f"{t:17s} " + " ".join(f"{sp(AL[a][:, j], AL[b][:, j]):.3f}|{sp(F0[a][:, j], F0[b][:, j]):.3f}  " for a, b in pairs))
pr(f"{'mean':17s} " + " ".join(f"{np.mean([sp(AL[a][:, j], AL[b][:, j]) for j in range(12)]):.3f}|"
                                f"{np.mean([sp(F0[a][:, j], F0[b][:, j]) for j in range(12)]):.3f}  " for a, b in pairs))

pr("\n== Q1b AUC of each label set vs Steven-v4 confident cells, fold 0 (rap / full are in-sample on fold 0)")
pr(f"{'finding':17s} {'pos':>4s} {'neg':>4s}  " + " ".join(f"{k:>6s}" for k in F0))
for j, t in enumerate(T):
    sel = m0[:, j]; npos, nneg = int((sel & t0[:, j]).sum()), int((sel & ~t0[:, j]).sum())
    pr(f"{t:17s} {npos:4d} {nneg:4d}  " + " ".join(f"{L.auc(F0[k][sel, j], t0[sel, j]):6.3f}" for k in F0))
f12 = L.TS["steven"][0]
pr(f"{'macro (12t set)':17s}            " + " ".join(f"{np.nanmean([L.auc(F0[k][m0[:, j], j], t0[m0[:, j], j]) for j in f12]):6.3f}" for k in F0))

pr("\n== Q1c mean target by report band, fold 0 (silent = finding not mentioned)")
for j, t in enumerate(T):
    row = []
    for b in ("absent", "silent", "mild", "mod+"):
        ix = band[:, j] == b
        row.append(f"{b}({ix.sum()}) " + "/".join(f"{F0[k][ix, j].mean():.2f}" if ix.any() else "-" for k in ("rap", "full", "lv2")))
    pr(f"{t:17s} " + "  ".join(row))
pr("(each cell: rap / full / lv2)")

pr("\n== Q1d Raptor vs full blend disagreement on Steven-confident fold-0 cells (side of 0.5)")
pr(f"{'finding':17s} {'conf':>5s} {'both ok':>7s} {'rap only':>8s} {'full only':>9s} {'both bad':>8s}  lv2 backs rap/full on disagreements")
tot = np.zeros(4, int); lvt = np.zeros(2, int)
for j, t in enumerate(T):
    sel = m0[:, j]; y = t0[sel, j]; r = (F0["rap"][sel, j] >= .5) == y; f = (F0["full"][sel, j] >= .5) == y
    c = np.array([(r & f).sum(), (r & ~f).sum(), (~r & f).sum(), (~r & ~f).sum()]); tot += c
    d = r != f; lvok = (F0["lv2"][sel, j] >= .5) == y
    lv = np.array([(d & r & lvok).sum(), (d & f & lvok).sum()]); lvt += lv
    pr(f"{t:17s} {sel.sum():5d} {c[0]:7d} {c[1]:8d} {c[2]:9d} {c[3]:8d}  {lv[0]}/{lv[1]}")
pr(f"{'all':17s} {tot.sum():5d} {tot[0]:7d} {tot[1]:8d} {tot[2]:9d} {tot[3]:8d}  {lvt[0]}/{lvt[1]}")
pr("rap only = cells rapc keeps but fullc drops (full blend wrong); full only = cells rapc drops (Raptor wrong)")

pr("\n== Q1e target sharpness (all common studies): share of cells < 0.05 or > 0.95 | mean |target - 0.5|")
for k in AL:
    pr(f"{k:5s} " + f"{((AL[k] < .05) | (AL[k] > .95)).mean():.3f} | {np.abs(AL[k] - .5).mean():.3f}")

# ------------------------------------------------------------------ Q2
RAPOK = (F0["rap"] >= .5) == t0; FULLOK = (F0["full"] >= .5) == t0
RUL = {"steven": m0, "clear": None, "rapc": m0 & RAPOK, "fullc": m0 & FULLOK, "bothc": m0 & RAPOK & FULLOK,
       "disc": m0 & (RAPOK != FULLOK), "bad": m0 & ~RAPOK & ~FULLOK}


def score(Sx, name):
    if name == "clear":
        return L.SM["steven_c"](Sx)
    m = RUL[name]; v = []
    for j in f12:
        sel = m[:, j]
        if (sel & t0[:, j]).sum() >= 5 and (sel & ~t0[:, j]).sum() >= 5:
            v.append(L.auc(Sx[sel, j], t0[sel, j]))
    return float(np.nanmean(v)) if v else np.nan


def pooled_disc(Sx):
    """disc cells pooled across findings on within-finding percentiles (few cells per finding)."""
    pc = np.column_stack([pd.Series(Sx[:, j]).rank(pct=True).to_numpy() for j in range(12)])
    m = RUL["disc"]; return L.auc(pc[m], t0[m]), int(m.sum())


names = list(RUL)
pr("\n== Q2a rulers scored on the label sets themselves (bias check; macro AUC over findings with >= 5 pos / neg)")
pr(f"{'set':6s} " + " ".join(f"{n:>7s}" for n in names) + "  disc-pooled")
for k in F0:
    a, n = pooled_disc(F0[k]); pr(f"{k:6s} " + " ".join(f"{score(F0[k], n_):7.4f}" for n_ in names) + f"  {a:.4f} ({n} cells)")

pr("\n== Q2b students at matched epochs (fused files), minus ra at the same epoch")
eps = sorted({int(p.stem[9:11]) for p in (S / "pubkd_kd_a").glob("val_epoch*_fused.json")}
             & {int(p.stem[9:11]) for p in (S / "pubkd_kd_b").glob("val_epoch*_fused.json")}
             & {int(p.stem[9:11]) for p in (S / "ra_epochs").glob("val_epoch*_fused.json")})
acc = {k: [] for k in ("a", "b")}
for e in eps:
    ra = L.load_pred(S / "ra_epochs" / f"val_epoch{e:02d}_fused.json")
    rs = {n_: score(ra, n_) for n_ in names}; rd = pooled_disc(ra)[0]
    pr(f"epoch {e}: ra " + " ".join(f"{n_} {rs[n_]:.4f}" for n_ in names) + f" disc-pooled {rd:.4f}")
    for k in ("a", "b"):
        Sx = L.load_pred(S / f"pubkd_kd_{k}" / f"val_epoch{e:02d}_fused.json")
        dd = {n_: score(Sx, n_) - rs[n_] for n_ in names}; dd["discp"] = pooled_disc(Sx)[0] - rd; acc[k].append(dd)
        pr(f"   {k.upper()} - ra: " + " ".join(f"{n_} {dd[n_]:+.4f}" for n_ in names) + f" disc-pooled {dd['discp']:+.4f}")
for k in acc:
    if acc[k]:
        pr(f"mean over epochs {eps}: {k.upper()} - ra: " + " ".join(f"{n_} {np.mean([d[n_] for d in acc[k]]):+.4f}"
                                                                  for n_ in names + ['discp']))
ras = L.load_pred(C / "pubkd/students_r/pubkd_kd_ra/val_swa_fused.json")
pr("ra SWA (LB 0.943): " + " ".join(f"{n_} {score(ras, n_):.4f}" for n_ in names) + f" disc-pooled {pooled_disc(ras)[0]:.4f}")
OUT.write_text("\n".join(lines) + "\n"); print(f"\nwrote {OUT}")
