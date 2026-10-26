"""Exp v3 vs Exp 1e, CPU part: score the branch ablations and summarise the spatial XAI written by
scripts/audit_v3_vs_1e_gpu.py (data/coatnet_d96_fold0/audit_v3_vs_1e/), and render the attention maps to PNG.

Usage (container):
  podman run --rm --network=none -v "$P":/proj -v "$P/repository/data/coatnet_d96_fold0/audit_v3_vs_1e":/out \
    localhost/rsna-label-ds python -I /proj/repository/scripts/audit_v3_vs_1e.py /proj /out
Fold-0 labels / clear band exactly as audit_d96_native.py. Development evidence only; no selection.
"""
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"; A = Path(sys.argv[2])
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; ids = [r["study_id"] for r in ref["labels"]]
y = np.array([[.5 if p is None else p for p in r["probabilities"]] for r in ref["labels"]])
w = np.array([r["weights"] for r in ref["labels"]])
mask = (w > 0) & ((y <= .3) | (y >= .7)); truth = y >= .7
TEN = [k for k, t in enumerate(T) if t not in ("Lateral OA", "Baker's")]
SLOTS = ["sag fluid", "sag other", "cor fluid", "cor other", "axial"]


def load(p):
    d = json.loads(Path(p).read_text()); m = {r["study_id"]: r["probabilities"] for r in d["predictions"]}
    tg = d.get("targets_in_order", T); a = np.array([m[s] for s in ids], float)
    return a[:, [list(tg).index(t) for t in T]]


def auc(s, t):
    r = pd.Series(s).rank().to_numpy(); n1 = t.sum(); n0 = len(t) - n1
    return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def per(S, idx=slice(None)):
    return np.array([auc(S[idx, k][mask[idx, k]], truth[idx, k][mask[idx, k]]) for k in range(12)])


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
        out.append(auc(S[sel, k], truth[sel, k]) if truth[sel, k].sum() >= 5 else np.nan)
    return np.nanmean(out)


CB = [k for k in range(12) if (mask[:, k] & truth[:, k] & (band[:, k] == "clear")).sum() >= 20]
rng = np.random.default_rng(0); n = len(ids); boots = [rng.integers(0, n, n) for _ in range(2000)]


def compare(X, Y, cols):
    d = np.array([per(X, b)[cols].mean() - per(Y, b)[cols].mean() for b in boots])
    return per(X)[cols].mean() - per(Y)[cols].mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


# ---------------------------------------------------------------- A. ablation
V = {k: load(A / f"val_{k}.json") for k in ("full", "base_only", "spatial_only", "uniform", "e1")}
V["e1 s2027"] = load(C / "run_r50_pertarget_seed/val_swa.json")
d1 = np.abs(V["full"] - load(C / "run_r50_spatial/val_swa.json")).max()
d2 = np.abs(V["e1"] - load(C / "run_r50_pertarget/val_swa.json")).max()
print(f"SANITY: max |audit full - v3 val_swa| {d1:.1e}; max |audit e1 - 1e val_swa| {d2:.1e} (bf16 run-to-run)")
print("\nA. ABLATION (fold 0; 10-target / 12-target / clear-band 6 / mild band 10-target)")
desc = {"full": "v3 as trained", "base_only": "v3 minus spatial term (GAP path only)",
        "spatial_only": "v3 minus GAP term (spatial pool only)", "uniform": "v3 with alpha := uniform (no selection)",
        "e1": "1e seed 2026", "e1 s2027": "1e seed 2027"}
for k, S in V.items():
    print(f"  {k:13s} {per(S)[TEN].mean():.4f}  {np.nanmean(per(S)):.4f}  {band_auc(S, 'clear', CB):.4f}  "
          f"{band_auc(S, 'mild', TEN):.4f}   {desc[k]}")
print("  paired bootstrap, 10-target:")
for a_, b_ in (("full", "uniform"), ("full", "base_only"), ("full", "e1"), ("base_only", "e1")):
    d, lo, hi, sh = compare(V[a_], V[b_], TEN)
    print(f"    {a_:10s} - {b_:10s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}] {sh:.0%} >0")
pt = pd.DataFrame({k: per(V[k]) for k in ("full", "uniform", "base_only", "spatial_only", "e1", "e1 s2027")}, index=T)
pt["sel = full-unif"] = pt["full"] - pt["uniform"]
pt["v3 - 1e mean"] = pt["full"] - (pt["e1"] + pt["e1 s2027"]) / 2
print("\nPer target:"); print(pt.round(4).to_string())

# ---------------------------------------------------------------- B. XAI
rows = pd.DataFrame([json.loads(l) for l in (A / "xai_rows.jsonl").read_text().splitlines()])
rows["pos"] = (rows["y"] >= .7) & (rows["w"] > 0); rows["neg"] = (rows["y"] <= .3) & (rows["w"] > 0)
print("\nB1. Window attention mass by series (positives, mean), v3 | 1e")
for t in T:
    g = rows[(rows.target == t) & rows.pos]
    m3, m1 = np.mean(g.slot_mass_v3.tolist(), 0), np.mean(g.slot_mass_e1.tolist(), 0)
    print(f"  {t:17s} " + "  ".join(f"{SLOTS[q]} {m3[q]:.2f}|{m1[q]:.2f}" for q in range(5)))

print("\nB2. Where v3 looks vs 1e's class-activation map (positives; top window per finding)")
print("  target            same_win  spearman(alpha,CAM) med  argmax dist mm med  entropy pos / neg   n_pos")
for t in T:
    g = rows[(rows.target == t) & rows.pos]; gn = rows[(rows.target == t) & rows.neg]
    sw = g[g.same_window]
    print(f"  {t:17s} {g.same_window.mean():7.0%}   {sw.spearman.median():+.2f}                  "
          f"{sw.argmax_mm.median():6.1f}            {g.entropy_top.mean():.3f} / {gn.entropy_top.mean():.3f}     {len(g)}")

print("\nB3. Does v3's focus help the studies it focuses on? (positives: rank-percentile gain v3 - 1e vs top-window entropy)")
P3, P1 = V["full"], V["e1"]
for k, t in enumerate(T):
    neg = mask[:, k] & ~truth[:, k]
    pos = np.where(mask[:, k] & truth[:, k])[0]
    if len(pos) < 10:
        continue
    pr = lambda S: np.array([(S[neg, k] < S[i, k]).mean() for i in pos])
    gain = pr(P3) - pr(P1)
    ent = rows[(rows.target == t)].set_index("study").loc[[ids[i] for i in pos], "entropy_top"].to_numpy()
    lo = ent <= np.median(ent)
    print(f"  {t:17s} positives {len(pos):3d}: mean gain {gain.mean():+.3f}; focused half {gain[lo].mean():+.3f}, "
          f"diffuse half {gain[~lo].mean():+.3f}; spearman(gain, -entropy) {pd.Series(gain).corr(pd.Series(-ent), method='spearman'):+.2f}")


# ---------------------------------------------------------------- C. maps (PNG, numpy-only encoder)
def png(path, rgb):
    h, wd, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", wd, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def overlay(img, heat):
    up = np.kron(heat, np.ones((img.shape[0] // heat.shape[0],) * 2))
    up = np.clip((up - up.min()) / (up.max() - up.min() + 1e-9), 0, 1)
    g = img.astype(float) / 255.0
    rgb = np.stack([g, g, g], -1) * (1 - 0.55 * up[..., None]) + 0.55 * up[..., None] * np.array([1.0, 0.2, 0.0])
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


M = np.load(A / "maps.npz")
keys = sorted({k.rsplit("|", 1)[0] for k in M.files})
out = A / "maps"; out.mkdir(exist_ok=True)
index = []
for t in T:
    studies = [k for k in keys if k.split("|")[0] == t]
    tiles = []
    for k in studies:
        a3 = overlay(M[f"{k}|img_v3"], M[f"{k}|alpha"]); a1 = overlay(M[f"{k}|img_e1"], np.clip(M[f"{k}|cam"], 0, None))
        sep = np.full((320, 6, 3), 255, np.uint8)
        tiles.append(np.concatenate([a3, sep, a1], 1))
        index.append(f"{t} | {k.split('|')[1][-12:]} | v3 p {float(M[f'{k}|p_v3']):.3f} ({SLOTS[int(M[f'{k}|slot_v3'])]}) "
                     f"| 1e p {float(M[f'{k}|p_e1']):.3f} ({SLOTS[int(M[f'{k}|slot_e1'])]})")
    if tiles:
        gap = np.full((6, tiles[0].shape[1], 3), 255, np.uint8)
        png(out / f"{t.replace(' ', '_').replace(chr(39), '')}.png",
            np.concatenate(sum([[x, gap] for x in tiles], [])[:-1], 0))
(out / "index.txt").write_text("Each row: left = v3 alpha (its top window), right = 1e positive CAM (its top window); "
                               "red = high. Studies: 10 fold-0 positives per finding, lowest SHA-256 of the id.\n"
                               + "\n".join(index) + "\n")
print(f"\nC. {len(keys)} maps rendered to {out}")
