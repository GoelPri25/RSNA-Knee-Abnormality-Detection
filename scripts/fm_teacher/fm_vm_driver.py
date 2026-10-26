"""FB-F teacher driver (Colab VM, /content/fmv python): fastMRI knee exams -> public R6 + 2.5D reader predictions.

Streams both fastMRI archives (curl | xz -dc | tar 'r|', batch 2 then batch 1, as scripts/fastmri_build_cache.py), keeps
the exams whose bucket int(sha256(study_id)) % NBUCKET is in this VM's set, stages each one to disk, and every CHUNK
exams writes a competition-layout data dir (train.csv, train_series.csv, sample_submission.csv, test.csv,
train_series/<study>/<series>/*.dcm), points /content/kin/rsna-knee-abnormality-detection at it and runs
scripts/fm_teacher/teacher_fm.py (= the Kaggle PUB-KD teacher kernel, split train, one shard) on it. Outputs per chunk:
out/cNNN/{r6_train_shard0.csv, reader_train_shard0.csv, receipt_shard0.json, series.csv, teacher.log, ...}.

  study_id   "fm_" + sha256(<patient dir>/<study dir>)[:16], the d96 cache's id (fastmri_build_cache.py)
  series     the d96 builder's rules (0c558d7): plane = dominant axis of the slice normal, < 8 files or mixed
             orientation skipped, order = SeriesNumber; fat-sat = ScanOptions FS or a description token fs / fat / stir /
             spair / spir ("non" negates). Fluid_Sensitive = Fat_Suppression, as in every competition row.
  repeats    a study split in the stream keeps its first part (as the builder); a study id met again (4 exams are in
             both archives) is skipped.
  resume     out/done.txt (study ids with a finished chunk) and out/nodata.txt are skipped on a restart; a broken stream
             is reopened from the start of that archive (up to 5 times), skipping what is staged or done.
Status line in /content/fm/STATUS (the keep-alive cell and the host guard read it): RUN ... / ALLDONE ... / FAILED ...
Usage: fm_vm_driver.py --buckets 0-11 [--nbucket 24] [--chunk 600] [--limit N (smoke: first N kept exams)]
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

C = Path("/content/fm")
STAGE, CHUNKS, OUT = C / "stage", C / "chunks", C / "out"
COMP = Path("/content/kin/rsna-knee-abnormality-detection")
WORKDIR = Path("/kaggle/working")
TEACHER = C / "teacher_fm.py"
PY = "/content/fmv/bin/python"
ARCHIVES = ["knee_DICOMs_batch2.tar.xz", "knee_DICOMs_batch1.tar.xz"]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
FAT_TOK = {"fs", "fat", "stir", "spair", "spir"}

ap = argparse.ArgumentParser()
ap.add_argument("--buckets", required=True)
ap.add_argument("--nbucket", type=int, default=24)
ap.add_argument("--chunk", type=int, default=600)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-staged", type=int, default=1500)
ap.add_argument("--loader-workers", default="6")
A = ap.parse_args()
lo, _, hi = A.buckets.partition("-")
MINE = set(range(int(lo), int(hi or lo) + 1))


def sid_of(rel):
    return "fm_" + hashlib.sha256(rel.encode()).hexdigest()[:16]


def mine(sid):
    return int(hashlib.sha256(sid.encode()).hexdigest(), 16) % A.nbucket in MINE


def status(s):
    (C / "STATUS.tmp").write_text(s + "\n")
    os.replace(C / "STATUS.tmp", C / "STATUS")


def lines(p):
    return set(p.read_text().split()) if p.exists() else set()


OUT.mkdir(parents=True, exist_ok=True)
DONE, NODATA = lines(OUT / "done.txt"), lines(OUT / "nodata.txt")
shutil.rmtree(STAGE, ignore_errors=True)
shutil.rmtree(CHUNKS, ignore_errors=True)
STAGE.mkdir(parents=True)
CHUNKS.mkdir(parents=True)
ready = collections.deque()          # (sid, rel) staged and complete
stream_state = {"archive": None, "seen": 0, "kept": 0, "ended": False, "error": None}


def stream_all():
    urls = dict(line.split(None, 1) for line in (C / "urls.txt").read_text().splitlines() if line.strip())
    taken = set()                    # study ids staged in this run (or done before)
    try:
        for arc in ARCHIVES:
            for attempt in range(5):
                stream_state["archive"] = f"{arc} try {attempt}"
                env = dict(os.environ, FM_URL=urls[arc].strip())
                p = subprocess.Popen('curl -sSfL --retry 5 --retry-delay 10 "$FM_URL" | xz -T0 -dc', shell=True,
                                     env=env, stdout=subprocess.PIPE, bufsize=1 << 20, start_new_session=True)
                cur, seen_rel, skip_rel, cur_keep, ok = None, set(), None, False, False
                try:
                    with tarfile.open(fileobj=p.stdout, mode="r|*") as tf:
                        for m in tf:
                            parts = Path(m.name).parts
                            if not m.isfile() or len(parts) < 5 or not m.name.endswith(".dcm"):
                                continue
                            rel = "/".join(parts[1:3])
                            if rel == skip_rel:
                                continue
                            if rel != cur:
                                if cur is not None and cur_keep:
                                    ready.append((sid_of(cur), cur)); stream_state["kept"] += 1
                                cur, cur_keep = None, False
                                if rel in seen_rel:
                                    print("WARNING split study skipped after first part:", sid_of(rel), flush=True)
                                    skip_rel = rel
                                    continue
                                seen_rel.add(rel); cur = rel; stream_state["seen"] += 1
                                sid = sid_of(rel)
                                cur_keep = (mine(sid) and sid not in DONE and sid not in NODATA and sid not in taken
                                            and not (A.limit and stream_state["kept"] >= A.limit))
                                if cur_keep:
                                    taken.add(sid)
                                    shutil.rmtree(STAGE / sid, ignore_errors=True)
                                    while len(ready) >= A.max_staged:
                                        time.sleep(5)          # bounded disk: the teacher is the slow stage
                                if A.limit and stream_state["kept"] >= A.limit and not cur_keep:
                                    break
                            if cur_keep:
                                dst = STAGE / sid_of(rel) / parts[3] / parts[-1]
                                dst.parent.mkdir(parents=True, exist_ok=True)
                                with tf.extractfile(m) as fs, open(dst, "wb") as fd:
                                    shutil.copyfileobj(fs, fd)
                    if cur is not None and cur_keep:
                        ready.append((sid_of(cur), cur)); stream_state["kept"] += 1
                        cur = None
                    ok = True
                except (EOFError, tarfile.ReadError, OSError, ValueError) as e:
                    print(f"STREAM ERROR {arc} try {attempt}: {type(e).__name__}: {e}", flush=True)
                    if cur is not None and cur_keep:
                        shutil.rmtree(STAGE / sid_of(cur), ignore_errors=True)
                        taken.discard(sid_of(cur))
                finally:
                    try:
                        os.killpg(p.pid, 9)
                    except ProcessLookupError:
                        pass
                    p.wait()
                if ok or (A.limit and stream_state["kept"] >= A.limit):
                    break
                time.sleep(30)
            else:
                raise RuntimeError(f"{arc}: stream failed 5 times")
            if A.limit and stream_state["kept"] >= A.limit:
                break
    except Exception as e:                                    # noqa: BLE001 - reported through the status line
        stream_state["error"] = f"{type(e).__name__}: {e}"
    finally:
        stream_state["ended"] = True


def series_rows(study_dir):
    """the d96 builder's series table (fastmri_build_cache.py series_rows, 0c558d7) with Fluid_Sensitive = fat-sat."""
    rows = []
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
        if "non" in toks:
            toks -= FAT_TOK | {"sat"}
        fat = int(fs or bool(toks & FAT_TOK))
        orients = set()
        for f in (files[0], files[len(files) // 2], files[-1]):
            try:
                iop = pydicom.dcmread(f, stop_before_pixels=True).ImageOrientationPatient
                orients.add(int(np.argmax(np.abs(np.cross(np.array(iop[:3], float), np.array(iop[3:], float))))))
            except Exception:                                 # noqa: BLE001
                orients.add(-1)
        plane = ("Sagittal", "Coronal", "Axial")[orients.pop()] if len(orients) == 1 and -1 not in orients else None
        if len(files) < 8 or plane is None:
            continue
        num = getattr(h, "SeriesNumber", None)
        rows.append({"SeriesInstanceUID": sd.name, "Fluid_Sensitive": fat, "Fat_Suppression": fat,
                     "Anatomical_Plane": plane, "description": desc,
                     "_order": (int(num) if num not in (None, "") else 10 ** 6, sd.name)})
    rows.sort(key=lambda r: r["_order"])
    return rows


def run_chunk(k, items):
    t0 = time.time()
    cd, od = CHUNKS / f"c{k:03d}", OUT / f"c{k:03d}"
    shutil.rmtree(cd, ignore_errors=True); shutil.rmtree(od, ignore_errors=True)
    (cd / "train_series").mkdir(parents=True); od.mkdir(parents=True)
    se, ids, nodata = [], [], []
    for sid, rel in items:
        try:
            rows = series_rows(STAGE / sid)
        except Exception as e:                                # noqa: BLE001
            print("series table failed", sid, repr(e)[:200], flush=True); rows = []
        if not rows:
            nodata.append(sid); continue
        ids.append(sid)
        os.symlink(STAGE / sid, cd / "train_series" / sid)
        se += [{"StudyInstanceUID": sid, **{k2: r[k2] for k2 in ("SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression",
                                                                   "Anatomical_Plane", "description")}, "source": rel}
               for r in rows]
    se = pd.DataFrame(se)
    se.to_csv(od / "series.csv", index=False)
    se[["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]].to_csv(
        cd / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ids, **{L: 0 for L in LABELS}}).to_csv(cd / "train.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ids, **{L: 0.5 for L in LABELS}}).to_csv(cd / "sample_submission.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(cd / "test.csv", index=False)
    if COMP.is_symlink() or COMP.exists():
        COMP.unlink()
    os.symlink(cd, COMP)
    for p in (WORKDIR, Path("/tmp/probe/prediction")):
        shutil.rmtree(p, ignore_errors=True)
    for p in Path("/tmp").glob("pubkd_fake*"):
        shutil.rmtree(p, ignore_errors=True)
    for p in Path("/tmp").glob("_own_*.csv"):
        p.unlink()
    WORKDIR.mkdir(parents=True)
    rc = 0
    if ids:
        with open(od / "teacher.log", "w") as log:
            rc = subprocess.run([PY, "-u", str(TEACHER)], cwd=WORKDIR, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, FM_LOADER_WORKERS=str(A.loader_workers))).returncode
        for f in WORKDIR.iterdir():
            if f.is_file():
                shutil.copyfile(f, od / f.name)
    scored = set()
    if (od / "r6_train_shard0.csv").exists():
        scored = set(pd.read_csv(od / "r6_train_shard0.csv", dtype=str).StudyInstanceUID)
    rd_ok = (od / "reader_train_shard0.csv").exists()
    info = {"chunk": k, "studies": len(ids), "nodata": len(nodata), "r6_scored": len(scored), "reader": rd_ok,
            "rc": rc, "seconds": round(time.time() - t0, 1), "s_per_study": round((time.time() - t0) / max(1, len(ids)), 2)}
    (od / "chunk.json").write_text(json.dumps(info, indent=1))
    print("CHUNK", json.dumps(info), flush=True)
    if rc != 0 or (ids and (not scored or not rd_ok)):
        raise RuntimeError(f"chunk {k}: teacher rc {rc}, r6 {len(scored)} / {len(ids)}, reader {rd_ok}")
    with open(OUT / "done.txt", "a") as f:
        f.writelines(s + "\n" for s in ids)
    with open(OUT / "nodata.txt", "a") as f:
        f.writelines(s + "\n" for s in nodata)
    (od / "DONE").write_text("ok\n")
    for sid, _ in items:
        shutil.rmtree(STAGE / sid, ignore_errors=True)
    shutil.rmtree(cd, ignore_errors=True)
    return info


def main():
    k = max([int(p.name[1:]) for p in OUT.glob("c[0-9][0-9][0-9]") if (p / "DONE").exists()], default=-1) + 1
    th = threading.Thread(target=stream_all, daemon=True)
    th.start()
    t0, n_done = time.time(), len(DONE)
    status(f"RUN start buckets {A.buckets}/{A.nbucket} done_before {n_done}")
    while True:
        if len(ready) >= A.chunk or (stream_state["ended"] and ready):
            items = [ready.popleft() for _ in range(min(A.chunk, len(ready)))]
            status(f"RUN chunk c{k:03d} ({len(items)} studies) done {n_done} staged {len(ready)} stream {stream_state}")
            info = run_chunk(k, items)
            n_done += info["studies"]; k += 1
            rate = (n_done - len(DONE)) / max(1e-6, time.time() - t0) * 3600
            status(f"RUN chunk_done c{k - 1:03d} done {n_done} rate {rate:.0f}/h staged {len(ready)} stream {stream_state}")
            continue
        if stream_state["ended"] and not ready:
            break
        time.sleep(10)
    if stream_state["error"]:
        raise RuntimeError("stream: " + stream_state["error"])
    status(f"ALLDONE done {n_done} chunks {k} stream {json.dumps(stream_state)}")


try:
    main()
except Exception as e:                                        # noqa: BLE001
    status(f"FAILED {type(e).__name__}: {str(e)[:300]}")
    raise
