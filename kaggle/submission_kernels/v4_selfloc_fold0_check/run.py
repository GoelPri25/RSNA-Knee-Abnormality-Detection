"""SL-0v4 FOLD-0 CHECK (no submission; ledger row SL-0v4): the v4 l4copy submission code (v4_submission.py in the
weights dataset, byte-identical to kaggle/submission_kernels/v4_l4copy_submission/submission.py) end to end on the 825
fold-0 studies, volumes decoded from the train DICOMs by the submission's own builder. Per study, one global pass, then
  self  crops placed by the model's own attention + gradient x activation (the submission path);
  ms1   crops from MS1's table (slim copy; window index -> slice via the same 94 centres), the training-time placement.
Out (/kaggle/working): val_{self,ms1}_{fused,global,local}.json, selfloc_regions.npz, selfloc_log.json.
Scored locally by scripts/audit_selfloc_v4.py (decision: fused self - MS1 >= -0.002).
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ART = next(p for p in (Path("/kaggle/input/datasets/tumis001/rsna-v4-l4copy-fold0"),
                       Path("/kaggle/input/rsna-v4-l4copy-fold0")) if p.exists())
sys.path.insert(0, str(ART))                             # importable by name: Pool pickles sub.build_study
import v4_submission as sub  # noqa: E402
MM = 140.0 / sub.IMG


def ms1_table():
    tab = {}
    for p in sorted(ART.glob("cp_*.npz")):
        z = np.load(p)
        for s, cen, at in zip(z["ids"], z["centres"], z["a_top"]):
            tab[str(s)] = (cen[sub.CF].astype(np.int64), at[sub.CF].astype(np.float32))
    return tab


def main():
    t0 = time.monotonic()
    data, work = sub.competition_dir(), Path("/kaggle/working")
    ids_path = next(p for p in (Path("/kaggle/input/datasets/tumis001/rsna-d96-fold0-ids/fold0_ids.json"),
                                Path("/kaggle/input/rsna-d96-fold0-ids/fold0_ids.json")) if p.exists())
    ids = json.loads(ids_path.read_text()); assert len(ids) == 825, len(ids)
    tab = ms1_table(); assert all(s in tab for s in ids)
    sub.ensure_timm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sub.load_model(ART / sub.WEIGHTS["file"], device, sub.WEIGHTS["sha256"])
    by_study = {}
    for row in sub.read_csv(data / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    out = {nm: {} for nm in ("self", "ms1")}
    regions = {k: [] for k in ("ids", "slices", "rows", "cols", "a", "ms1_slices", "ms1_rows", "ms1_cols")}
    stats = {"empty_studies": 0, "decode_errors": 0, "fp32_retries": 0}
    t_self = t_ms1 = 0.0
    with Pool(4) as procs, torch.no_grad():
        for n, (sid, vol, mask, errors) in enumerate(
                procs.imap(sub.build_study, [(s, by_study.get(s, []), data / "train_series") for s in ids]), 1):
            stats["decode_errors"] += errors
            if not mask.any():
                stats["empty_studies"] += 1
                continue
            cs = sub.uniform_centres(mask)
            cen, at = tab[sid]
            t_ms = (cs[cen[..., 0]], cen[..., 1], cen[..., 2], at)
            for nm in ("self", "ms1"):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.monotonic()
                p = sub.predict_study(model, vol, mask, device, table=None if nm == "self" else t_ms)
                if not np.isfinite(p).all():
                    stats["fp32_retries"] += 1
                    p = sub.predict_study(model, vol, mask, device, amp=False, table=None if nm == "self" else t_ms)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.monotonic() - t1
                if nm == "self":
                    t_self += dt
                else:
                    t_ms1 += dt
                out[nm][sid] = p
            # the self table again for the agreement stats (cheap: global pass + locate)
            lo = (sub.IMG - sub.EVAL_RES) // 2
            xg = sub.windows(vol, cs, device)[..., lo:lo + sub.EVAL_RES, lo:lo + sub.EVAL_RES].float().div_(255.0)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                fm = model.backbone.forward_features(sub.normalise(xg))
            fm = fm.float()
            a = torch.softmax(model.att.float()(model.norm.float()(fm.mean((2, 3)))), dim=0)
            S, R, Q, A = sub.locate(model, fm, a, cs)
            for k, v in zip(("ids", "slices", "rows", "cols", "a", "ms1_slices", "ms1_rows", "ms1_cols"),
                            (sid, S, R, Q, A, *t_ms[:3])):
                regions[k].append(v)
            if n % 50 == 0 or n == len(ids):
                print(f"v4 check {n}/{len(ids)} studies; {time.monotonic() - t0:.0f}s", flush=True)
    done = [s for s in ids if s in out["self"]]
    for nm in ("self", "ms1"):
        for j, h in enumerate(("fused", "global", "local")):
            preds = [{"study_id": s, "probabilities": out[nm][s][j].tolist()} for s in done]
            (work / f"val_{nm}_{h}.json").write_text(json.dumps(
                {"targets_in_order": sub.TARGETS, "note": f"v4 l4copy fold 0, Kaggle path, {nm} crops, {h}",
                 "predictions": preds}))
    reg = {k: np.array(v) for k, v in regions.items()}
    np.savez(work / "selfloc_regions.npz", **reg)
    agree = {}
    for j, f in enumerate(sub.CF):
        near, dist = [], []
        for i in range(len(reg["ids"])):
            Sm, Rm, Qm = reg["ms1_slices"][i, j], reg["ms1_rows"][i, j], reg["ms1_cols"][i, j]
            Ss, Rs, Qs = reg["slices"][i, j], reg["rows"][i, j], reg["cols"][i, j]
            near.append(np.mean([np.abs(Sm - k).min() <= 1 for k in Ss]))
            hit = np.abs(Sm - Ss[0]) <= 1
            if hit.any():
                q = int(np.argmax(hit)); dist.append(MM * np.hypot(Rs[0] - Rm[q], Qs[0] - Qm[q]))
        agree[sub.TARGETS[f]] = {"windows_within_1_slice": round(float(np.mean(near)), 3),
                                 "top1_matched_share": round(len(dist) / len(reg["ids"]), 3),
                                 "top1_point_mm_median": round(float(np.median(dist)), 1) if dist else None}
    log = {"studies": len(done), "stats": stats, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
           "locate_s_per_study": None, "predict_self_s_per_study": round(t_self / max(1, len(done)), 4),
           "predict_ms1_s_per_study": round(t_ms1 / max(1, len(done)), 4), "agreement_vs_ms1": agree,
           "seconds": round(time.monotonic() - t0, 1)}
    (work / "selfloc_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("SL_DONE " + json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
