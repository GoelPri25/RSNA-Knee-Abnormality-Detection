"""clear6 screen (audit_clear_band_lb.py definitions) for v4 l4copy inference variants from the XAI kernel arrays."""
import sys, types
from pathlib import Path
import numpy as np
R = Path(sys.argv[1]); src = (R / "repository/scripts/audit_clear_band_lb.py").read_text()
head = src.split("run = lambda")[0]                     # labels, clear mask, CB, scores()
g = {"__name__": "x"}; sys.argv = [sys.argv[0], str(R)]; exec(head, g)
ids, T, scores, CB = g["ids"], g["T"], g["scores"], g["CB"]
X = R / "repository/data/coatnet_d96_fold0/l4copy_analysis"
lv = __import__("json").load(open(R / "repository/data/coatnet_d96_fold0/assets/soft_labels_final_per_target_latD2.json"))
assert lv["targets_in_order"] == T, (lv["targets_in_order"], T)
sig = lambda z: 1 / (1 + np.exp(-z)); CF = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]
def get(tag, v="kaggle_xai_v1"):
    z = np.load(X / v / f"xai_arrays_{tag}.npz"); zi = list(z["ids"]); ix = [zi.index(s) for s in ids]
    return {k: z[k][ix] for k in z.files if k != "ids" and z[k].ndim == 2 and z[k].shape[1] == 12}
def fused(z, loc, glob_only=()):
    f = z["glob"].copy(); f[:, CF] = 0.5 * (z["glob"][:, CF] + z[loc][:, CF])
    for t in glob_only: k = T.index(t); f[:, k] = z["glob"][:, k]
    return sig(f)
print("CB targets:", [T[k] for k in CB])
rows = {}
for tag in ("s2026", "s2027", "r1self"):
    z = get(tag)
    rows[f"{tag} global"] = sig(z["glob"]); rows[f"{tag} self (deployed)"] = fused(z, "loc_self_100")
    rows[f"{tag} self, PF+frac global"] = fused(z, "loc_self_100", ("PF OA", "Fracture"))
    rows[f"{tag} MS1 table"] = fused(z, "loc_ms1_100")
    rows[f"{tag} MS1, PF+frac global"] = fused(z, "loc_ms1_100", ("PF OA", "Fracture"))
rows["seedavg self"] = (rows["s2026 self (deployed)"] + rows["s2027 self (deployed)"]) / 2
rows["seedavg self, PF+frac"] = (rows["s2026 self, PF+frac global"] + rows["s2027 self, PF+frac global"]) / 2
for k, S in rows.items():
    c6, m10 = scores(S); per = [g["auc"](S[(sel := g["mask"][:, j] & (~g["truth"][:, j] | g["clear"][:, j])), j], g["truth"][sel, j]) for j in CB]
    print(f"{k:28s} clear6 {c6:.4f}  10-tgt {m10:.4f}  fit {-0.687 + 1.671 * c6:.3f}  | " + " ".join(f"{T[j][:6]} {v:.3f}" for j, v in zip(CB, per)))
C = R / "repository/data/coatnet_d96_fold0"; TM = R / "repository/data/teammate_pp4/fold0_predictions"
E = {"lv2": C / "kaggle_fold0_labelv2/val_swa.json", "1b": C / "run_continue/val_swa.json", "cnx": C / "run_cnx_pertarget/val_swa.json",
     "4net": TM / "four_network_fold0_val_swa.json"}
E = {k: g["load"](p) for k, p in E.items()}; E["E3"] = np.mean(list(E.values()), 0)
E["E3 + v4 self"] = np.mean([E[k] for k in ("lv2", "1b", "cnx", "4net")] + [rows["s2026 self (deployed)"]], 0)
for k, S in E.items():
    c6, m10 = scores(S); per = [g["auc"](S[(sel := g["mask"][:, j] & (~g["truth"][:, j] | g["clear"][:, j])), j], g["truth"][sel, j]) for j in CB]
    print(f"{k:28s} clear6 {c6:.4f}  10-tgt {m10:.4f}  fit {-0.687 + 1.671 * c6:.3f}  | " + " ".join(f"{T[j][:6]} {v:.3f}" for j, v in zip(CB, per)))
# mild band: positives that are present but not moderate+ (what the clear screen excludes)
def mild(S):
    out = []
    for j in range(12):
        sel = g["mask"][:, j] & (~g["truth"][:, j] | ~g["clear"][:, j])
        out.append(g["auc"](S[sel, j], g["truth"][sel, j]) if g["truth"][sel, j].sum() >= 10 else np.nan)
    return np.array(out)
print("mild-band (positives not moderate+) per target:")
for k, S in (("v4 s2026 self", rows["s2026 self (deployed)"]), ("E3", E["E3"]), ("lv2", E["lv2"])):
    print(f"{k:16s} " + " ".join(f"{T[j][:6]} {v:.3f}" for j, v in enumerate(mild(S))))
