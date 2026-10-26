"""B3-abl scoring (CPU, fold 0): pre-registered fine-4 / val12 rules, clear6, 10-target, per target, curves."""
import sys, json
from pathlib import Path
src = Path(sys.argv[1] + "/repository/scripts/audit_phase1.py").read_text()
head = src.split("def per_target")[0]
sys.argv = [sys.argv[0], sys.argv[1]]
exec(head)
import numpy as np
K = D / "kaggle_kernel_outputs/swin3d_b3_ablation_train_v3"
idx = np.arange(len(ids))
FINE = [T.index(t) for t in ("ACL", "Lateral Meniscus", "Medial Meniscus", "PF OA")]
def ac(S, k):
    s = idx[mask[idx, k]]; return auc(S[s, k], truth[s, k])
def fine4(S): return np.mean([ac(S, k) for k in FINE])
def val12(S): return np.mean([ac(S, k) for k in range(12)])
M = {"B2(modal)": load(C / "run_swin3d_b2/val_swa.json"),
     "A0": load(K / "A0_frozen/val_swa.json"), "A1": load(K / "A1_unfrozen_ckpt/val_swa.json"),
     "A0_tta": load(K / "A0_frozen/val_swa_mirror.json"), "A1_tta": load(K / "A1_unfrozen_ckpt/val_swa_mirror.json"),
     "1e_lv2": load(C / "kaggle_fold0_labelv2/val_swa.json")}
print(f"{'model':10s} fine4   val12   clear6  10-tgt")
R = {}
for n, S in M.items():
    c6, m10 = scores(S, idx)[:2]; R[n] = (fine4(S), val12(S), c6, m10)
    print(f"{n:10s} {R[n][0]:.4f}  {R[n][1]:.4f}  {c6:.4f}  {m10:.4f}")
d = np.array(R["A1"]) - np.array(R["A0"])
print(f"\nA1 - A0: fine4 {d[0]:+.4f} (rule >= +0.010)  val12 {d[1]:+.4f} (rule >= -0.002)  clear6 {d[2]:+.4f}  10-target {d[3]:+.4f}")
rng = np.random.default_rng(0); bs = []
for _ in range(2000):
    b = rng.choice(idx, len(idx))
    def f(S):
        return np.mean([auc(S[b[mask[b, k]], k], truth[b[mask[b, k]], k]) for k in FINE])
    bs.append(f(M["A1"]) - f(M["A0"]))
print(f"  fine4 delta bootstrap 95% [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}], {np.mean(np.array(bs) > 0):.0%} > 0")
proj = 0.9542 + d[2] + 0.005
print(f"  projected B3 clear6 = 0.9542 + {d[2]:+.4f} + 0.005 = {proj:.4f} (partner >= 0.965, single >= 0.9745)")
print("\nper target all-cell AUC")
print(f"{'target':17s}" + "".join(f"{n:>10s}" for n in ("B2(modal)", "A0", "A1", "A1-A0", "1e_lv2")))
for k in range(12):
    a = [ac(M[n], k) for n in ("B2(modal)", "A0", "A1")]
    print(f"{T[k]:17s}" + "".join(f"{x:10.4f}" for x in a) + f"{a[2]-a[1]:+10.4f}{ac(M['1e_lv2'], k):10.4f}")
print("\ncurves (epoch, train_loss, val12, val_bce)")
for arm in ("A0_frozen", "A1_unfrozen_ckpt"):
    h = json.loads((K / arm / "history.json").read_text())
    print(arm, [(r["epoch"], round(r["train_loss"], 4), None if r.get("val_macro_auc") is None else round(r["val_macro_auc"], 4),
                 None if r.get("val_bce") is None else round(r["val_bce"], 4)) for r in h])
