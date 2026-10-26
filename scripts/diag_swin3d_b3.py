"""B3 diagnostics (CPU, fold 0, read-only): clear-band per target, pair with 1e-label-v2, rank correlation,
gap vs slice thickness / field strength."""
import sys
from pathlib import Path
src = Path(sys.argv[1] + "/repository/scripts/audit_phase1.py").read_text()
head = src.split("def per_target")[0]
sys.argv = [sys.argv[0], sys.argv[1]]
exec(head)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
C = D / "coatnet_d96_fold0"
M = {"1e": load(C / "run_r50_pertarget/val_swa.json"), "1e_s27": load(C / "run_r50_pertarget_seed/val_swa.json"),
     "1e_lv2": load(C / "kaggle_fold0_labelv2/val_swa.json"),
     "B2": load(C / "run_swin3d_b2/val_swa.json"), "B2_tta": load(C / "run_swin3d_b2/val_swa_mirror.json"),
     "B1_ep10": load(C / "run_swin3d_b/val_epoch10.json"), "B1": load(C / "run_swin3d_b/val_swa.json"),
     "cnx": load(C / "run_cnx_pertarget/val_swa.json")}
M["B2+lv2"] = (M["B2"] + M["1e_lv2"]) / 2; M["1e+lv2"] = (M["1e"] + M["1e_lv2"]) / 2
M["cnx+lv2"] = (M["cnx"] + M["1e_lv2"]) / 2; M["B1e10+B2"] = (M["B1_ep10"] + M["B2"]) / 2
idx = np.arange(len(ids))
def cb(S, k, sel=idx):
    s = sel[mask[sel, k] & (~truth[sel, k] | clear[sel, k])]
    return auc(S[s, k], truth[s, k]) if truth[s, k].sum() >= 5 else np.nan
def ac(S, k, sel=idx):
    s = sel[mask[sel, k]]
    return auc(S[s, k], truth[s, k]) if truth[s, k].sum() >= 5 else np.nan
print("clear-band AUC per target (moderate+ positives vs confident negatives)")
names = list(M)
print(f"{'target':17s}" + "".join(f"{n:>9s}" for n in names))
for k in CB:
    print(f"{T[k]:17s}" + "".join(f"{cb(M[n], k):9.4f}" for n in names))
print(f"{'clear6':17s}" + "".join(f"{scores(M[n], idx)[0]:9.4f}" for n in names))
print(f"{'10-target':17s}" + "".join(f"{scores(M[n], idx)[1]:9.4f}" for n in names))
print("\nall-cell AUC per target")
for k in range(12):
    print(f"{T[k]:17s}" + "".join(f"{ac(M[n], k):9.4f}" for n in names))
print("\nSpearman rank correlation with 1e_lv2 (confident cells), per target and mean")
for n in ("1e", "1e_s27", "cnx", "B1_ep10", "B1", "B2"):
    r = [spearmanr(M[n][mask[:, k], k], M["1e_lv2"][mask[:, k], k])[0] for k in range(12)]
    print(f"  {n:8s} mean {np.mean(r):.3f}  " + " ".join(f"{T[k][:6]} {r[k]:.2f}" for k in range(12)))
# strata
h = pd.read_csv(D / "dicom_demographics/study_header_tags.csv", dtype=str).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
th = pd.to_numeric(h.reindex(ids)["SliceThickness"], errors="coerce").to_numpy()
fs = pd.to_numeric(h.reindex(ids)["MagneticFieldStrength"], errors="coerce").to_numpy()
print(f"\nslice thickness (first series) on fold 0: n {np.isfinite(th).sum()}, quantiles {np.nanpercentile(th, [10, 25, 50, 75, 90])}")
strata = {"thick<=3.0": np.where(th <= 3.0)[0], "thick>3.0": np.where(th > 3.0)[0],
          "1.5T": np.where(fs < 2)[0], "3T": np.where(fs >= 2)[0]}
for nm, sel in strata.items():
    print(f"\n{nm}: {len(sel)} studies.  all-cell AUC  B2 / 1e_lv2 / B1_ep10 (n pos)")
    for t in ("ACL", "Lateral Meniscus", "Medial Meniscus", "PF OA", "Lateral OA", "Medial OA", "Effusion"):
        k = T.index(t); npos = int(truth[sel[mask[sel, k]], k].sum())
        a, b, c = ac(M["B2"], k, sel), ac(M["1e_lv2"], k, sel), ac(M["B1_ep10"], k, sel)
        print(f"  {t:17s} {a:.4f} {b:.4f} {c:.4f}  B2-lv2 {a - b:+.4f}  B1e10-lv2 {c - b:+.4f}  (n+ {npos})")
# bootstrap the thick-vs-thin interaction for ACL and Lateral Meniscus
rng = np.random.default_rng(0)
for t in ("ACL", "Lateral Meniscus", "PF OA"):
    k = T.index(t); d = []
    A, B_ = strata["thick<=3.0"], strata["thick>3.0"]
    for _ in range(1000):
        a = rng.choice(A, len(A)); b = rng.choice(B_, len(B_))
        g = lambda s: ac(M["B2"], k, s) - ac(M["1e_lv2"], k, s)
        d.append(g(b) - g(a))
    d = np.array(d); print(f"  {t}: (gap thick) - (gap thin) {np.nanmean(d):+.4f} [{np.nanpercentile(d, 2.5):+.4f}, {np.nanpercentile(d, 97.5):+.4f}]")
