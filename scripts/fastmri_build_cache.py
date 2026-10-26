"""SURPASS-F: the fastMRI knee DICOM batches (NYU fastMRI, knee_DICOMs_batch{1,2}.tar.xz) as a d96 cache, same layout
and pixel pipeline as our competition cache (kaggle/experiment_kernels/coatnet_d96_cache_shard0/run.py, sha e76793ca:
header / pixels / crop_resize / build_study copied verbatim below, only the series table and file paths differ).

The competition builder took Anatomical_Plane / Fluid_Sensitive / Fat_Suppression from train_series.csv; fastMRI has no
such table, so each series row is derived from its DICOM headers:
  plane   dominant axis of the slice normal (IOP row x col): x Sagittal, y Coronal, z Axial (LPS)
  fluid   1 if "FS" in ScanOptions or the series description has a token fs / fat / stir / spair / spir / t2 (else 0)
  fat     1 if "FS" in ScanOptions or a token fs / fat / stir / spair / spir
  order   SeriesNumber (the competition used CSV order)
  skipped series with < 8 files or more than one orientation (localisers).
The archive is read as a stream (xz -> tar 'r|'): files are staged per study in a RAM directory, a study is built once
the stream moves to the next study (tar order is per directory), then its files are deleted.

study_id = "fm_" + sha256(<patient dir>/<study dir>)[:16]; the relative path is kept in the manifest ("source").
Shard = sha256(study_id) % NSHARD, written as shard_{OFFSET + k} (batch 2: offset 0, batch 1: offset 2, ...).
Outputs per shard: vol_NNN.npy (100 studies x 96 rows), manifest.jsonl, cache_summary.json; per run: audit_<tag>.json
(description -> plane / fluid counts, slot fill, empty slots, decode errors).
Usage: xz -T0 -dc batch.tar.xz | python -I fastmri_build_cache.py - <out root> <tag> <offset> <nshard> [workers] [limit]
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

if os.environ.get("FM_PYLIBS"):                           # pydicom / cv2 / openjpeg wheels (python -I ignores PYTHONPATH)
    sys.path.insert(0, os.environ["FM_PYLIBS"])

IMG, CROP_MM = 384, 140.0
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(k for _, _, k in SLOTS)                       # 96
STUDIES_PER_PART = 100
FLUID_TOK, FAT_TOK = {"fs", "fat", "stir", "spair", "spir", "t2"}, {"fs", "fat", "stir", "spair", "spir"}
SRC, OUTR, TAG, OFFSET, NSHARD = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
WORKERS = int(sys.argv[6]) if len(sys.argv) > 6 else 32
LIMIT = int(sys.argv[7]) if len(sys.argv) > 7 else 0
STAGE = Path(os.environ.get("FM_STAGE", "/dev/shm/fm_stage")) / TAG


# ---------------------------------------------------------------- verbatim from the competition builder (e76793ca)
def header(path):
    import pydicom
    try:
        h = pydicom.dcmread(path, stop_before_pixels=True)
        iop = getattr(h, "ImageOrientationPatient", None)
        ipp = getattr(h, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r, c = np.array(iop[:3], float), np.array(iop[3:], float)
            pos = float(np.dot(np.array(ipp, float), np.cross(r, c)))
        else:
            pos = float(getattr(h, "InstanceNumber", 0) or 0)
        ps = getattr(h, "PixelSpacing", None)
        geo = {"ipp": [float(v) for v in ipp] if ipp is not None else None,
               "iop": [float(v) for v in iop] if iop is not None else None,
               "ps": [float(v) for v in ps] if ps is not None else None}
        lat = {k: str(getattr(h, k, "") or "") for k in ("Laterality", "ImageLaterality", "BodyPartExamined")}
        return pos, path, (float(ps[0]) if ps is not None else 0.5), True, geo, lat
    except Exception:
        return 0.0, path, 0.5, False, None, None


def pixels(path):
    import pydicom
    from pydicom.pixels import apply_modality_lut
    d = pydicom.dcmread(path)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def crop_resize(a, ps):
    import cv2
    h, w = a.shape
    cpx = min(int(round(CROP_MM / max(ps, 1e-3))), h, w)
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (IMG, IMG), interpolation=cv2.INTER_AREA)


def build_study(job):
    study_id, rows, study_dir = job                      # only change: series files from study_dir / <series dir>
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx, used, slots_meta = 0, set(), []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for plane, fluid, k in SLOTS:
            cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
            pref = [r for r in cands if fluid in (0, 1) and int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
            row = (pref or cands or [None])[0]
            meta = {"plane": plane, "fluid_wanted": fluid, "k": k, "row_start": idx, "series": None}
            slots_meta.append(meta)
            if row is None:
                idx += k
                continue
            used.add(row["SeriesInstanceUID"])
            files = sorted((study_dir / row["SeriesInstanceUID"]).glob("*.dcm"))
            heads = list(pool.map(header, files))
            ok_ps = [h[2] for h in heads if h[3]]
            med_ps = float(np.median(ok_ps)) if ok_ps else 0.5
            heads.sort(key=lambda h: h[0])
            n = len(heads)
            meta.update(series=row["SeriesInstanceUID"], fluid=row.get("Fluid_Sensitive"),
                        fat=row.get("Fat_Suppression"), n_files=n,
                        laterality=next((h[5] for h in heads if h[5]), None))
            if n == 0:
                idx += k
                continue
            lo, hi = int(n * 0.02), int(n * 0.98) - 1
            hi = max(hi, lo)
            picks = [int(min(p, n - 1)) for p in (np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k)]
            meta["picks"] = picks
            meta["geometry"] = [heads[p][4] for p in picks]

            def load(p):
                try:
                    return pixels(heads[p][1]), heads[p][2]
                except Exception:
                    return None, med_ps
            got = list(pool.map(load, picks))
            valid = [a for a, _ in got if a is not None]
            meta["decode_errors"] = len(got) - len(valid)
            loq, hiq = (np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0]) if valid else (0.0, 1.0))
            for a, ps in got:
                if a is not None:
                    aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                    vol[idx] = (crop_resize(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8)
                idx += 1
    mask = (vol.reshape(MAXS, -1).max(1) > 0)
    return study_id, vol, {"study_id": study_id, "mask": mask.astype(int).tolist(), "slots": slots_meta}


# ---------------------------------------------------------------- fastMRI series table from headers
def series_rows(study_dir: Path):
    import pydicom
    rows, desc_log = [], []
    for sd in sorted(p for p in study_dir.iterdir() if p.is_dir()):
        files = sorted(sd.glob("*.dcm"))
        if not files:
            continue
        h = pydicom.dcmread(files[0], stop_before_pixels=True)
        desc = str(getattr(h, "SeriesDescription", "") or "")
        opts = getattr(h, "ScanOptions", "") or ""
        opts = [str(o).upper() for o in (opts if isinstance(opts, (list, tuple, pydicom.multival.MultiValue)) else [opts])]
        toks = set(re.split(r"[^a-z0-9]+", desc.lower()))
        fs = any("FS" in o for o in opts)
        if "non" in toks:  # "SAG PD FSE NON FAT SAT": negated fat-sat tokens don't count (job 5363573 missed this)
            toks -= FAT_TOK | {"sat"}
        fluid = int(fs or bool(toks & FLUID_TOK)); fat = int(fs or bool(toks & FAT_TOK))
        orients, plane = set(), None
        for f in (files[0], files[len(files) // 2], files[-1]):
            try:
                iop = pydicom.dcmread(f, stop_before_pixels=True).ImageOrientationPatient
                nrm = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                orients.add(int(np.argmax(np.abs(nrm))))
            except Exception:
                orients.add(-1)
        if len(orients) == 1 and -1 not in orients:
            plane = ("Sagittal", "Coronal", "Axial")[orients.pop()]
        skip = "few_files" if len(files) < 8 else ("orientation" if plane is None else None)
        desc_log.append((desc, plane, fluid, skip))
        if skip:
            continue
        num = getattr(h, "SeriesNumber", None)
        rows.append({"SeriesInstanceUID": sd.name, "Anatomical_Plane": plane, "Fluid_Sensitive": str(fluid),
                     "Fat_Suppression": str(fat), "description": desc,
                     "_order": (int(num) if num not in (None, "") else 10 ** 6, sd.name)})
    rows.sort(key=lambda r: r["_order"])
    return rows, desc_log


def work(job):
    study_id, rel, study_dir = job
    try:
        rows, desc_log = series_rows(study_dir)
        sid, vol, rec = build_study((study_id, rows, study_dir))
        rec["source"] = rel
        rec["series_table"] = [{k: r[k] for k in ("SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive",
                                                  "Fat_Suppression", "description")} for r in rows]
        return sid, vol, rec, desc_log, None
    except Exception as e:                                # noqa: BLE001 - a broken study is logged and skipped
        return study_id, None, None, [], f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(study_dir, ignore_errors=True)


def stream_studies(q, stop, inflight):
    """Reader thread: tar stream -> staged study dirs; puts the study once the stream leaves it. Staging is bounded by
    inflight (studies staged and not yet returned by a worker). Always ends the queue with None."""
    src = sys.stdin.buffer if SRC == "-" else open(SRC, "rb")
    cur, seen, n, skip_rel = None, set(), 0, None
    try:
        with tarfile.open(fileobj=src, mode="r|*") as tf:
            for m in tf:
                parts = Path(m.name).parts
                if not m.isfile() or len(parts) < 5 or not m.name.endswith(".dcm"):
                    continue
                rel = "/".join(parts[1:3])                   # <patient dir>/<study dir>
                if rel == skip_rel:
                    continue
                if rel != cur:
                    if cur is not None:
                        q.append(cur); inflight[0] += 1; n += 1
                        cur = None
                        if LIMIT and n >= LIMIT:
                            break
                    if rel in seen:                          # study split in the stream: keep the first part only
                        print(f"WARNING split study skipped after first part: {rel}", flush=True)
                        skip_rel = rel
                        continue
                    seen.add(rel); cur = rel
                    while inflight[0] - inflight[1] > 4 * WORKERS and not stop.is_set():
                        time.sleep(0.05)                     # bounded staging (RAM)
                dst = STAGE / rel / parts[3] / parts[-1]
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as fsrc, open(dst, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
    except (EOFError, tarfile.ReadError, OSError, ValueError) as e:   # truncated stream (smoke test)
        print(f"STREAM END {type(e).__name__}: {e}; open study dropped: {cur}", flush=True)
        if cur is not None:
            shutil.rmtree(STAGE / cur, ignore_errors=True)
        cur = None
    finally:
        if cur is not None:
            q.append(cur); inflight[0] += 1
        q.append(None)


def main():
    t0 = time.monotonic()
    STAGE.mkdir(parents=True, exist_ok=True)
    shards = {}
    for k in range(NSHARD):
        d = OUTR / f"shard_{OFFSET + k}"
        assert not (d / "manifest.jsonl").exists(), f"{d} exists"
        d.mkdir(parents=True, exist_ok=True)
        shards[k] = {"dir": d, "man": (d / "manifest.jsonl").open("w"), "buf": [], "part": 0,
                     "stats": collections.Counter()}

    def jobs():
        while True:
            while not q:
                time.sleep(0.02)
            rel = q.popleft()
            if rel is None:
                return
            yield "fm_" + hashlib.sha256(rel.encode()).hexdigest()[:16], rel, STAGE / rel

    def flush(s):
        if s["buf"]:
            np.save(s["dir"] / f"vol_{s['part']:03d}.npy", np.concatenate(s["buf"]))
            s["part"], s["buf"] = s["part"] + 1, []

    audit = {"desc": collections.Counter(), "failed": [], "fill5": 0, "studies": 0, "empty_slots": 0,
             "decode_errors": 0, "slices": 0, "slot_empty": collections.Counter()}
    procs = Pool(WORKERS)                                # fork the workers before the reader thread exists
    q, stop, inflight = collections.deque(), threading.Event(), [0, 0]
    th = threading.Thread(target=stream_studies, args=(q, stop, inflight), daemon=True); th.start()
    with procs:
        for sid, vol, rec, desc_log, err in procs.imap_unordered(work, jobs(), chunksize=1):
            inflight[1] += 1                                # [0] staged by the reader, [1] returned
            for d, p, f, skip in desc_log:
                audit["desc"][f"{d} | {p} | fluid {f}{' | skip ' + skip if skip else ''}"] += 1
            if err:
                audit["failed"].append([sid, err]); continue
            s = shards[int(hashlib.sha256(sid.encode()).hexdigest(), 16) % NSHARD]
            rec.update(part=f"vol_{s['part']:03d}.npy", row_start=len(s["buf"]) * MAXS, gold=False)
            s["man"].write(json.dumps(rec) + "\n")
            s["buf"].append(vol)
            es = [m["series"] is None for m in rec["slots"]]
            de = sum(m.get("decode_errors", 0) for m in rec["slots"])
            s["stats"].update(studies=1, empty_slots=sum(es), decode_errors=de, masked_slices=MAXS - sum(rec["mask"]))
            audit["studies"] += 1; audit["fill5"] += not any(es); audit["empty_slots"] += sum(es)
            audit["decode_errors"] += de; audit["slices"] += MAXS - sum(k for (_, _, k), e in zip(SLOTS, es) if e)
            for (pl, fl, _), e in zip(SLOTS, es):
                audit["slot_empty"][f"{pl}/{fl}"] += e
            if len(s["buf"]) == STUDIES_PER_PART:
                flush(s)
            if audit["studies"] % 200 == 0:
                print(f"{audit['studies']} studies; {time.monotonic() - t0:.0f}s; fill5 "
                      f"{audit['fill5'] / audit['studies']:.3f}; failed {len(audit['failed'])}", flush=True)
    stop.set(); th.join()
    for k, s in shards.items():
        flush(s); s["man"].close()
        summary = {"shard": [OFFSET + k, TAG], "img": IMG, "crop_mm": CROP_MM, "slots": SLOTS, **s["stats"],
                   "parts": s["part"], "seconds": round(time.monotonic() - t0, 1), "source": "fastMRI knee DICOMs",
                   "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (s["dir"] / "cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    n = max(audit["studies"], 1)
    out = {"tag": TAG, "studies": audit["studies"], "failed": audit["failed"],
           "fill5_frac": round(audit["fill5"] / n, 4), "empty_slots": audit["empty_slots"],
           "slot_empty": dict(audit["slot_empty"]), "decode_errors": audit["decode_errors"], "slices": audit["slices"],
           "decode_error_frac": round(audit["decode_errors"] / max(audit["slices"], 1), 5),
           "seconds": round(time.monotonic() - t0, 1),
           "desc_plane_fluid": dict(audit["desc"].most_common())}
    (OUTR / f"audit_{TAG}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "desc_plane_fluid"}, indent=1), flush=True)
    print("TOP DESCRIPTIONS", json.dumps(dict(audit["desc"].most_common(25)), indent=1), flush=True)
    shutil.rmtree(STAGE, ignore_errors=True)


if __name__ == "__main__":
    main()
