"""Exp L4C-D on Modal (ledger row L4C-D): the Kaggle diagnostic notebook's per-study code
(kaggle/submission_kernels/v4_l4copy_diag_fold0/run.py, mounted as l4copy_diag_core) on the 825 fold-0 studies, volumes
read from the derived d96 cache (the same cache v4 was trained and scored on) instead of decoded from DICOMs.
Renders are saved as arrays (render_arrays.npz) and drawn locally by scripts/render_l4copy_diag.py.
Out (DIAG_OUT): val_global.json, val_<cond>_{fused,local}.json, tables.npz, render_arrays.npz, diag_log.json.
Env: D96_CACHE (shard_0..3 with manifest.jsonl), V4_ART (v4_submission.py, weights, cp_*.npz), DIAG_OUT, FOLDS (folds.csv),
DIAG_N (first N studies, CPU test).
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l4copy_diag_core as run  # noqa: E402

sub = run.sub
CACHE, OUT = Path(os.environ["D96_CACHE"]), Path(os.environ["DIAG_OUT"])


def manifest():
    out = {}
    for root in sorted(CACHE.glob("shard_*")):
        for line in (root / "manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            out[r["study_id"]] = (str(root / r["part"]), r["row_start"], np.array(r["mask"], bool))
    return out


def main():
    t0 = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(r["study_id"] for r in csv.DictReader(open(os.environ["FOLDS"])) if r["fold"] == "0")
    assert len(ids) == 825, len(ids)
    meta = manifest(); tab = run.ms1_table()
    if os.environ.get("DIAG_N"):                          # CPU test on the smoke cache: any studies it holds
        ids = sorted(s for s in meta if s in tab)[:int(os.environ["DIAG_N"])]
    rend = {}
    for f, lst in run.RENDER_IDS.items():
        for i, s in enumerate(lst):
            rend.setdefault(s, []).append((f, i))
    if os.environ.get("DIAG_N"):                          # CPU test: render the first studies instead
        rend = {ids[0]: [("Baker's", 0), ("PF OA", 0)]}
    assert all(s in meta and s in tab for s in ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sub.load_model(run.ART / sub.WEIGHTS["file"], device, sub.WEIGHTS["sha256"])
    mm = {}

    def read(s):
        f, row, mask = meta[s]
        if f not in mm:
            mm[f] = np.load(f, mmap_mode="r")
        return s, np.ascontiguousarray(mm[f][row:row + sub.MAXS]), mask

    out = {c: {h: {} for h in ("fused", "local")} for c in run.CONDS}
    glob_out, keep = {}, {c: {k: [] for k in "SRQ"} for c in ("self", "self_plane", "ms1win_selfpts")}
    tsec = {c: 0.0 for c in run.CONDS}; checks = {"equiv_ms1": [], "equiv_self": []}
    ren = {}
    with ThreadPoolExecutor(4) as pool, torch.no_grad():
        for n, (sid, vol, mask) in enumerate(pool.map(read, ids), 1):
            g, tabs, preds, dt = run.one_study(model, sid, vol, mask, tab[sid], device)
            glob_out[sid] = g
            for c in run.CONDS:
                tsec[c] += dt[c]
                out[c]["fused"][sid], out[c]["local"][sid] = preds[c]
            if n <= 3:
                for c in ("ms1", "self"):
                    ref = sub.predict_study(model, vol, mask, device, table=tabs["ms1"] if c == "ms1" else None)
                    checks[f"equiv_{c}"].append(float(np.abs(ref[0] - out[c]["fused"][sid]).max()))
            for c in keep:
                for k, v in zip("SRQ", tabs[c][:3]):
                    keep[c][k].append(v)
            for f, i in rend.get(sid, []):
                j = int(np.where(run.CF == sub.TARGETS.index(f))[0][0]); key = f"{f}|{i}"
                for nm in ("ms1", "self", "self_plane"):
                    S, R, Q, _ = tabs[nm]
                    ren[f"{key}|{nm}|img"] = vol[int(S[j, 0])]
                    ren[f"{key}|{nm}|srq"] = np.array([S[j, 0], R[j, 0], Q[j, 0], sub.BOXN[run.CF[j]]])
                    ren[f"{key}|{nm}|slices"] = S[j].copy()
            if n % 50 == 0 or n == len(ids):
                print(f"L4C-D {n}/{len(ids)} studies; {time.monotonic() - t0:.0f}s", flush=True)
    note = "v4 l4copy fold 0, Modal cache path"
    (OUT / "val_global.json").write_text(json.dumps({"targets_in_order": sub.TARGETS, "note": note + ", global",
        "predictions": [{"study_id": s, "probabilities": glob_out[s].tolist()} for s in ids]}))
    for c in run.CONDS:
        for h in ("fused", "local"):
            (OUT / f"val_{c}_{h}.json").write_text(json.dumps({"targets_in_order": sub.TARGETS, "note": f"{note}, {c}, {h}",
                "predictions": [{"study_id": s, "probabilities": out[c][h][s].tolist()} for s in ids]}))
    np.savez(OUT / "tables.npz", ids=np.array(ids), **{f"{c}_{k}": np.stack(v) for c, d in keep.items() for k, v in d.items()})
    np.savez_compressed(OUT / "render_arrays.npz", **ren)
    log = {"studies": len(ids), "checks": checks, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
           "crop_pass_s_per_study": {c: round(t / max(1, len(ids)), 4) for c, t in tsec.items()},
           "allowed": run.ALLOWED, "seconds": round(time.monotonic() - t0, 1)}
    (OUT / "diag_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("DIAG_DONE " + json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
