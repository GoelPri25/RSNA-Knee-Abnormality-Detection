"""Torch-free ConvNeXt-S TensorRT kernel, R7-D weights (FB-F R7-D: heliosli R7 init, FB-F mix73 targets over competition + fastMRI, SWA 6-8). The commit-run start of the TensorRT kernel (fast_cnxs_trtlean_cv) is
mostly the torch import (record trace v4: torch 183 MB + nvidia libs 286 MB of 685 MB read from a slow VM disk; import
4.6-28 s by VM), and torch only moved buffers and ran the volume build there. This kernel keeps the io / DICOM / study
loading code of fast_cnxs_trtlean_cv/submission.py verbatim and replaces main():
  volume  numpy + cv2 on the CPU in NPROC forked worker processes (file reads too, one GIL each): the same 2/98 percentiles (integer histogram), crop, cv2.resize
          INTER_AREA (the training builder's own call; the GPU kernel used it as a matrix product), uint8 truncation
  input   (u8 - mean*255)/(std*255) -> fp16 as one 256-entry table per channel (cv2.LUT, bit-identical to torch's
          float32 maths), written straight into pinned host buffers, NCHW
  gpu     CUDA runtime through ctypes (libcudart.so.12 from the image's nvidia wheel), TensorRT lean runtime +
          version-compatible engine (static 94 x 3 x 320 x 320 fp16 -> f [94, 768]), one stream per GPU
  head    numpy float32 (layer_norm, attention pool, per-finding linear, sigmoid) from cnxs_head.npz
"""
import os
import time

T0 = time.perf_counter()
import csv  # noqa: E402
import struct  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

MODE = "submit"
EVAL_N = 825
EVAL_IDS = "tumis001/rsna-fast-eval/eval_ids.txt"   # eval mode: gold-58 first, then the 825 fold-0 studies (IDs only)
IO_THREADS = 128
PREFETCH = "none"                  # torch libraries read ahead at start: none / torch (torch/lib) / all (+ nvidia libs)
WARM_NBCONVERT = True              # a nice-19 child imports nbconvert + renders an empty notebook (page cache for the HTML step)
NBWARM_EARLY = False               # start that child at t=0 (else after torch), so it is done before the HTML step starts
WARM_TRACE = False                 # at start: read the recorded page trace (torch / CUDA / cv2 pages) with WARM_THREADS threads
WARM_THREADS = 32
GAPMON = False                     # at exit: a detached monitor logs processes every 0.25 s to gapmon.log (what runs after the script)
RECORD = False                     # at the end: write the resident file pages of everything loaded (page trace for a later readahead)
HEAD_BYTES = 8192
LOOKAHEAD = 16
SLIM = "tumis001/rsna-fast-r7d-trt/r7d_head.npz"
NPROC = 4                          # volume worker processes (file reads + volume build, forked at start)
WLOOK = 4                          # studies loading at once per worker
INFLIGHT = 24                      # studies sent to the workers and not yet consumed
IO_AFTER_PACK = False              # study reads start after the TensorRT pack + head are read
WATCH_S = 30                       # eval: progress line every WATCH_S, thread dumps when stalled
EVAL_LIMIT_S = 1800                # eval: hard exit
NBUF = 6                           # pinned host input buffers (94 x 3 x 320 x 320 fp16, 58 MB each)
CV_THREADS = -1                    # cv2.setNumThreads (-1: cv2 default)
REF = None

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IMG, CROP_MM, EVAL_RES, EVAL_WINDOWS = 384, 140.0, 320, 94
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(k for _, _, k in SLOTS)


def find(rel):
    owner, slug, *rest = rel.split("/")
    for base in (f"/kaggle/input/datasets/{owner}/{slug}", f"/kaggle/input/{slug}"):
        p = os.path.join(base, *rest)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(rel)


def trt_pack():
    """TensorRT lean runtime + T4 engine: dataset rsna-fast-cnxs-trt (236 MB slim copy of the rsna-cnxs-pack output)."""
    import glob
    for p in ("/kaggle/input/datasets/tumis001/rsna-fast-cnxs-trt/trtpack", "/kaggle/input/rsna-fast-cnxs-trt/trtpack",
              "/kaggle/input/notebooks/tumis001/rsna-cnxs-pack/trtpack", "/kaggle/input/rsna-cnxs-pack/trtpack"):
        if os.path.exists(p + "/manifest.json"):
            return p
    for pat in ("/kaggle/input/*/trtpack", "/kaggle/input/notebooks/*/*/trtpack"):
        for p in glob.glob(pat):
            if os.path.exists(p + "/manifest.json"):
                return p
    raise FileNotFoundError("trtpack")


def competition_dir():
    for p in ("/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("competition data")


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- DICOM from raw bytes (explicit VR little endian)
LONG = {b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"UC", b"UN", b"UR", b"UT"}
NEED = {0x00020010: "ts", 0x00200032: "ipp", 0x00200037: "iop", 0x00200013: "inst", 0x00280002: "spp",
        0x00280004: "photo", 0x00280010: "rows", 0x00280011: "cols", 0x00280030: "ps", 0x00280100: "bits",
        0x00280103: "signed", 0x00281053: "slope"}
EVRLE = b"1.2.840.10008.1.2.1"


class Short(Exception):
    pass


class Unsupported(Exception):
    pass


def _skip_sq(b, p):
    """p = first byte inside an undefined-length sequence -> first byte after its delimiter."""
    while True:
        g, e, L = struct.unpack_from("<HHI", b, p)
        p += 8
        if g != 0xFFFE:
            raise Unsupported("item")
        if e == 0xE0DD:
            return p
        p = _elements(b, p, None, item=True) if L == 0xFFFFFFFF else p + L


def _elements(b, p, out, item=False, header_only=False):
    n = len(b)
    while True:
        if p + 8 > n:
            raise Short
        g, e = struct.unpack_from("<HH", b, p)
        if item and g == 0xFFFE and e == 0xE00D:
            return p + 8
        tag = (g << 16) | e
        if not item:
            if g != 2 and _str(out.get("ts", b"")) != EVRLE:
                raise Unsupported(out.get("ts"))
            if header_only and tag > 0x00281053:
                return p
        vr = b[p + 4:p + 6]
        if vr in LONG:
            if p + 12 > n:
                raise Short
            L = struct.unpack_from("<I", b, p + 8)[0]
            p += 12
        else:
            L = struct.unpack_from("<H", b, p + 6)[0]
            p += 8
        if tag == 0x7FE00010 and not item:
            out["pix"] = p
            return p
        if L == 0xFFFFFFFF:
            p = _skip_sq(b, p)
            continue
        if out is not None and not item and tag in NEED:
            if p + L > n:
                raise Short
            out[NEED[tag]] = bytes(b[p:p + L])
        p += L


def parse(b, header_only):
    if len(b) < 132 or b[128:132] != b"DICM":
        raise Unsupported("no preamble")
    out = {}
    try:
        _elements(b, 132, out, header_only=header_only)
    except struct.error:
        raise Short
    return out


def _str(v):
    return v.rstrip(b"\x00 ")


def _us(v):
    return struct.unpack_from("<H", v)[0]


def _ds(v):
    return [float(x) for x in _str(v).decode().split("\\")]


def header(path):
    """-> (slice position, path, row spacing, ok) = the E3 builder's header()."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)      # NFS: no readahead of the pixel data
            b = os.pread(fd, HEAD_BYTES, 0)
            while True:
                try:
                    h = parse(b, header_only=True)
                    break
                except Short:
                    more = os.pread(fd, len(b), len(b))
                    if not more:
                        raise
                    b += more
        finally:
            os.close(fd)
        iop, ipp = h.get("iop"), h.get("ipp")
        iop = _ds(iop) if iop else None
        if iop is not None and ipp is not None and len(iop) == 6:
            r, c = np.array(iop[:3]), np.array(iop[3:])
            pos = float(np.dot(np.array(_ds(ipp)), np.cross(r, c)))
        else:
            pos = float(int(_str(h["inst"]).decode())) if _str(h.get("inst", b"")) else 0.0
        ps = h.get("ps")
        return pos, path, (_ds(ps)[0] if ps and _str(ps) else 0.5), True
    except Unsupported:
        return header_pydicom(path)
    except Exception:
        return 0.0, path, 0.5, False


def header_pydicom(path):
    import pydicom
    try:
        h = pydicom.dcmread(path, stop_before_pixels=True)
        iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r, c = np.array(iop[:3], float), np.array(iop[3:], float)
            pos = float(np.dot(np.array(ipp, float), np.cross(r, c)))
        else:
            pos = float(getattr(h, "InstanceNumber", 0) or 0)
        ps = getattr(h, "PixelSpacing", None)
        return pos, path, (float(ps[0]) if ps is not None else 0.5), True
    except Exception:
        return 0.0, path, 0.5, False


def pixels(path):
    """-> (stored values, unsigned 16-bit read as int16?, inversions). Inversions = MONOCHROME1 + negative rescale
    slope; the 2/98 window removes any positive slope and intercept, so the modality LUT is not applied."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        b = os.pread(fd, os.fstat(fd).st_size, 0)
    finally:
        os.close(fd)
    try:
        h = parse(b, header_only=False)
        if "pix" not in h or _us(h["bits"]) != 16 or ("spp" in h and _us(h["spp"]) != 1):
            raise Unsupported("pixel format")
        rows, cols = _us(h["rows"]), _us(h["cols"])
        signed = "signed" in h and _us(h["signed"]) == 1
        a = np.frombuffer(b, "<i2", rows * cols, h["pix"]).reshape(rows, cols)
        slope = _str(h.get("slope", b""))
        inv = (_str(h.get("photo", b"")) == b"MONOCHROME1") + (bool(slope) and float(slope.decode()) < 0)
        return a, not signed, int(inv)
    except (Unsupported, Short, KeyError, ValueError, struct.error):
        import pydicom
        d = pydicom.dcmread(path)
        a = d.pixel_array.astype(np.int32)
        inv = (str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1") + (float(getattr(d, "RescaleSlope", 1) or 1) < 0)
        return a, False, int(inv)


# ---------------------------------------------------------------- study loading (threads)
def assign(rows):
    out, used, idx = [], set(), 0
    for plane, fluid, k in SLOTS:
        cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
        pref = [r for r in cands if fluid in (0, 1) and int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        row = (pref or cands or [None])[0]
        if row is not None:
            used.add(row["SeriesInstanceUID"])
        out.append((idx, k, row and row["SeriesInstanceUID"]))
        idx += k
    return out


def load_study(rows, series_dir, sid, files_pool):
    """-> [(slot start, [(slot rows, stacked arrays, unsigned, inversions, crop px)] per shape group)]
    Two network-disk rounds per study (all headers, then every distinct picked file once), not two per series."""
    series = []
    for start, k, uid in assign(rows):
        if uid is None:
            continue
        d = os.path.join(series_dir, sid, uid)
        files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".dcm")) if os.path.isdir(d) else []
        if files:
            series.append((start, k, files))
    heads_all = list(files_pool.map(header, [f for _, _, fs in series for f in fs]))
    plan, at = [], 0
    for start, k, files in series:
        heads = heads_all[at:at + len(files)]
        at += len(files)
        ok_ps = [h[2] for h in heads if h[3]]
        med_ps = float(np.median(ok_ps)) if ok_ps else 0.5
        heads.sort(key=lambda h: h[0])
        n = len(heads)
        lo, hi = int(n * 0.02), int(n * 0.98) - 1
        hi = max(hi, lo)
        picks = [int(min(p, n - 1)) for p in (np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k)]
        plan.append((start, heads, med_ps, picks))

    def load(path):
        try:
            return pixels(path)
        except Exception:
            return None
    paths = list(dict.fromkeys(heads[p][1] for _, heads, _, picks in plan for p in picks))   # each file read once
    got = dict(zip(paths, files_pool.map(load, paths)))
    out = []
    for start, heads, med_ps, picks in plan:
        groups = {}
        for j, p in enumerate(picks):
            g = got[heads[p][1]]
            if g is not None:
                a, u, inv = g
                ps = heads[p][2] if heads[p][2] > 0 else med_ps
                cpx = min(int(round(CROP_MM / max(ps, 1e-3))), *a.shape)
                groups.setdefault((a.shape, a.dtype.str, u, inv, cpx), []).append((j, a))
        out.append((start, [(np.array([j for j, _ in v]), np.stack([a for _, a in v]), u, inv, cpx)
                            for (_, _, u, inv, cpx), v in groups.items()]))
    return out


# ---------------------------------------------------------------- main
_last = [T0, 0.0, 0, 0]


def stage(name, out):
    """Append 'name wall/cpu/diskMB/majflt' since the previous stage."""
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF)
    try:
        rb = int([l for l in open("/proc/self/io") if l.startswith("read_bytes")][0].split()[1])
    except Exception:
        rb = 0
    now, cpu = time.perf_counter(), r.ru_utime + r.ru_stime
    out.append(f"{name} {now - _last[0]:.2f}s/cpu{cpu - _last[1]:.1f}/{(rb - _last[2]) / 1e6:.0f}MB/pf{r.ru_majflt - _last[3]}")
    _last[:] = [now, cpu, rb, r.ru_majflt]


def _readahead(paths):
    for f in paths:
        try:
            fd = os.open(f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            os.close(fd)
        except OSError:
            pass


def _libs(kind):
    import glob
    import importlib.util
    out = []
    t = importlib.util.find_spec("torch")
    if t and t.submodule_search_locations:
        out += sorted(glob.glob(os.path.join(t.submodule_search_locations[0], "lib", "*.so*")), key=os.path.getsize, reverse=True)
    if kind == "all":
        n = importlib.util.find_spec("nvidia")
        for d in (n.submodule_search_locations if n and n.submodule_search_locations else []):
            out += glob.glob(os.path.join(d, "*", "lib", "*.so*"))
    return out


def _warm_trace():
    """Read the file ranges a previous run had in page cache after torch import + GPU init (PAGETRACE), in parallel:
    the import then finds them cached instead of faulting them in one by one."""
    import base64
    import json
    import zlib
    tr = json.loads(zlib.decompress(base64.b64decode(PAGETRACE)))
    jobs, step = [], 1 << 20

    def first(f):                                             # torch's import path first, then the rest
        b = os.path.basename(f)
        return (0 if b.startswith(("libtorch", "libc10")) else 1 if "/nvidia/" in f or b.startswith("libcu") else
                2 if "/torch/" in f else 3)
    for f, rs in sorted(tr.items(), key=lambda kv: first(kv[0])):
        for off, ln in rs:
            for o in range(off, off + ln, step):
                jobs.append((f, o, min(step, off + ln - o)))

    def rd(j):
        try:
            fd = os.open(j[0], os.O_RDONLY)
            try:
                os.pread(fd, j[2], j[1])
            finally:
                os.close(fd)
        except OSError:
            pass
    with ThreadPoolExecutor(WARM_THREADS) as ex:
        list(ex.map(rd, jobs))


def cudart():
    """libcudart.so.12 (nvidia-cuda-runtime wheel) through ctypes: device buffers, pinned host buffers, streams."""
    import ctypes
    import glob
    import sys
    cands = [p for d in sys.path if d for p in glob.glob(os.path.join(d, "nvidia", "cuda_runtime", "lib", "libcudart.so.1*"))]
    cands += glob.glob("/usr/local/cuda/lib64/libcudart.so.1*") + ["libcudart.so.12"]
    for p in cands:
        try:
            lib = ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue
    else:
        raise OSError("libcudart")
    vp, sz = ctypes.c_void_p, ctypes.c_size_t
    sig = {"cudaSetDevice": [ctypes.c_int], "cudaGetDeviceCount": [ctypes.POINTER(ctypes.c_int)],
           "cudaMalloc": [ctypes.POINTER(vp), sz], "cudaHostAlloc": [ctypes.POINTER(vp), sz, ctypes.c_uint],
           "cudaMemcpyAsync": [vp, vp, sz, ctypes.c_int, vp], "cudaStreamCreate": [ctypes.POINTER(vp)],
           "cudaStreamSynchronize": [vp], "cudaFree": [vp]}
    for name, args in sig.items():
        getattr(lib, name).argtypes = args
        getattr(lib, name).restype = ctypes.c_int

    def ck(r, what):
        if r != 0:
            raise RuntimeError(f"{what}: cuda error {r}")
    return lib, ck, p


O = (IMG - EVAL_RES) // 2


def volume(slots, cv2):
    vol = np.zeros((MAXS, EVAL_RES, EVAL_RES), np.uint8)
    for start, groups in slots:
        if not groups:
            continue
        ts = []
        for js, arr, unsigned, inv, cpx in groups:
            t = arr.astype(np.int32)
            if unsigned:
                t &= 0xFFFF
            for _ in range(inv):
                t = t.max((1, 2), keepdims=True) - t
            ts.append((js, t, cpx))
        lo = min(int(t.min()) for _, t, _ in ts)
        size = max(int(t.max()) for _, t, _ in ts) - lo + 1
        hist = np.zeros(size, np.int64)
        for _, t, _ in ts:
            hist += np.bincount((t - lo).ravel(), minlength=size)
        cum = np.cumsum(hist)
        nn = sum(t.size for _, t, _ in ts)
        qv = []
        for q in (2.0, 98.0):                                    # np.percentile, linear interpolation
            hq = (nn - 1) * q / 100.0
            i = int(np.floor(hq))
            v = np.searchsorted(cum, [i, min(i + 1, nn - 1)], side="right").astype(np.float64)
            qv.append(np.float32(float(lo) + v[0] + (hq - i) * (v[1] - v[0])))
        loq, hiq = qv
        den = np.float32(hiq - loq) + np.float32(1e-6)
        for js, t, cpx in ts:
            h, w = t.shape[1:]
            y0, x0 = (h - cpx) // 2, (w - cpx) // 2
            c = t[:, y0:y0 + cpx, x0:x0 + cpx].astype(np.float32)
            c -= loq
            c /= den
            np.clip(c, 0, 1, out=c)
            for k, j in enumerate(js):
                r = cv2.resize(c[k], (IMG, IMG), interpolation=cv2.INTER_AREA)[O:O + EVAL_RES, O:O + EVAL_RES]
                r *= 255
                vol[start + j] = r                               # float -> uint8 truncation (as torch .to(uint8))
    return vol


def _vworker(tq, rq, nio):
    """Worker process (forked before any thread or CUDA): file reads in nio threads, WLOOK studies loading at once, volume
    built here, own GIL. Sends (sid, uint8 volume [MAXS, 320, 320], io s, build s), then None at the end."""
    import collections
    import queue
    import cv2
    cv2.setNumThreads(1)
    if MODE == "eval":
        import faulthandler
        faulthandler.dump_traceback_later(WATCH_S * 6, repeat=True)
    fpool, spool = ThreadPoolExecutor(nio), ThreadPoolExecutor(WLOOK)

    def job(rows, series_dir, sid):
        t = time.perf_counter()
        r = load_study(rows, series_dir, sid, fpool)
        return r, time.perf_counter() - t
    pend, done = collections.deque(), False
    try:
        while True:
            while not done and len(pend) < WLOOK:
                try:
                    task = tq.get() if not pend else tq.get_nowait()
                except queue.Empty:
                    break
                if task is None:
                    done = True
                    break
                sid, rows, series_dir = task
                pend.append((sid, spool.submit(job, rows, series_dir, sid)))
            if not pend:
                break
            sid, fut = pend.popleft()
            slots, t_io = fut.result()
            t = time.perf_counter()
            vol = volume(slots, cv2)
            rq.put((sid, vol, t_io, time.perf_counter() - t))
    except BaseException as e:
        rq.put(("__error__", repr(e), 0.0, 0.0))
        raise
    rq.put(None)


def main():
    import ctypes
    import multiprocessing as mp
    import threading
    ctx = mp.get_context("fork")                           # workers first: forked before any thread, CUDA or cv2 import
    tq, rq = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=_vworker, args=(tq, rq, max(4, IO_THREADS // NPROC)), daemon=True) for _ in range(NPROC)]
    for p_ in procs:
        p_.start()
    if WARM_NBCONVERT and NBWARM_EARLY:
        _nbwarm()
    if WARM_TRACE and PAGETRACE:
        threading.Thread(target=_warm_trace, daemon=True).start()
    wbuf, TPK = {}, {}

    def fetch():                                             # head weights (0.8 MB) + TensorRT pack on gcsfuse: sequential
        with open(find(SLIM), "rb") as f:                     # reads at t=0
            wbuf["b"] = f.read()
        wbuf["t"] = time.perf_counter() - T0

    def fetch_trt_libs():                                    # runtime package copied to local disk, imported from there
        import shutil
        shutil.copytree(trt_pack() + "/lean", "/tmp/trt_lean", dirs_exist_ok=True)

    def fetch_trt_engine():
        with open(find("tumis001/rsna-fast-r7d-trt/r7d_vc.engine"), "rb") as fh:
            TPK["eng"] = fh.read()
    tts = [threading.Thread(target=f, daemon=True) for f in (fetch, fetch_trt_libs, fetch_trt_engine)]
    for t_ in tts:
        t_.start()
    data = competition_dir()
    if MODE == "eval":
        ids = [s for s in open(find(EVAL_IDS)).read().split() if s][:EVAL_N]
        series_csv, series_dir = os.path.join(data, "train_series.csv"), os.path.join(data, "train_series")
        fallback = {s: [0.5] * 12 for s in ids}
    else:
        sample = read_csv(os.path.join(data, "sample_submission.csv"))
        assert list(sample[0]) == ["StudyInstanceUID", *TARGETS]
        ids = [r["StudyInstanceUID"] for r in sample]
        fallback = {r["StudyInstanceUID"]: [float(r[t]) for t in TARGETS] for r in sample}
        series_csv, series_dir = os.path.join(data, "test_series.csv"), os.path.join(data, "test_series")
    by, want = {}, set(ids)
    for r in read_csv(series_csv):
        if r["StudyInstanceUID"] in want:
            by.setdefault(r["StudyInstanceUID"], []).append(r)
    timed, stages = {}, []
    import queue as Q
    inflight = threading.Semaphore(INFLIGHT)

    def feeder():
        if IO_AFTER_PACK:                                    # the GPUs wait for the TensorRT pack anyway: give its gcsfuse
            for t_ in tts:                                   # read the bandwidth, then start the study reads
                t_.join()
        for s in ids:
            inflight.acquire()
            tq.put((s, by.get(s, []), series_dir))
        for _ in range(NPROC):
            tq.put(None)
    threading.Thread(target=feeder, daemon=True).start()     # file reads start before anything heavy loads
    stage("csv+io start", stages)

    import cv2
    if CV_THREADS >= 0:
        cv2.setNumThreads(CV_THREADS)
    stage("cv2", stages)
    lib, ck, cudart_path = cudart()
    n = ctypes.c_int(0)
    ck(lib.cudaGetDeviceCount(ctypes.byref(n)), "device count")
    devs = list(range(n.value))
    stage("cudart", stages)
    import sys
    for t_ in tts:
        t_.join()
    sys.path.insert(0, "/tmp/trt_lean")
    import tensorrt_lean as trt
    ENG = TPK.pop("eng")
    import io
    W = {k: v.astype(np.float32) for k, v in np.load(io.BytesIO(wbuf.pop("b"))).items()}
    stage("tensorrt+head", stages)
    if WARM_NBCONVERT and not NBWARM_EARLY:
        _nbwarm()

    XSHAPE, FSHAPE = (EVAL_WINDOWS, 3, EVAL_RES, EVAL_RES), (EVAL_WINDOWS, 768)
    XB, FB = int(np.prod(XSHAPE)) * 2, int(np.prod(FSHAPE)) * 2

    def host(nbytes, shape):
        p = ctypes.c_void_p()
        ck(lib.cudaHostAlloc(ctypes.byref(p), nbytes, 1), "cudaHostAlloc")   # portable pinned
        return np.ctypeslib.as_array((ctypes.c_uint16 * (nbytes // 2)).from_address(p.value)).view(np.float16).reshape(shape)
    free_bufs = Q.Queue()
    for _ in range(NBUF):
        free_bufs.put(host(XB, XSHAPE))
    stage("pinned", stages)

    mean = np.array([0.485, 0.456, 0.406], np.float32) * np.float32(255)    # as torch: float32 tensor * 255
    std = np.array([0.229, 0.224, 0.225], np.float32) * np.float32(255)
    LUT = [((np.arange(256, dtype=np.float32) - mean[c]) / std[c]).astype(np.float16).view(np.int16).reshape(1, 256)
           for c in range(3)]
    _u = np.random.default_rng(0).integers(0, 256, (EVAL_RES, EVAL_RES), dtype=np.uint8)   # cv2.LUT writes in place
    for c in range(3):
        _o = np.zeros((EVAL_RES, EVAL_RES), np.int16)
        cv2.LUT(_u, LUT[c], dst=_o)
        assert np.array_equal(_o, LUT[c][0][_u]), "cv2.LUT dst"

    def windows(vol, xb):
        mask = vol.reshape(MAXS, -1).max(1) > 0
        if not mask.any():
            return False
        valid = np.where(mask)[0]
        if len(valid) < 3:
            valid = np.arange(3)
        lo, hi = int(valid.min()), int(valid.max())
        cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, MAXS - 2))])
        cs = np.clip(cs[np.linspace(0, len(cs) - 1, EVAL_WINDOWS).round().astype(int)], 1, MAXS - 2)
        xi = xb.view(np.int16)
        for w, c0 in enumerate(cs):
            for ch in range(3):
                cv2.LUT(vol[c0 + ch - 1], LUT[ch], dst=xi[w, ch])
        return True

    built = Q.Queue(maxsize=max(2, NBUF - 2))
    t_load, t_vol, t_gpu = [0.0], [0.0], [0.0]
    lk = threading.Lock()

    def collector():
        try:
            left = NPROC
            while left:
                item = rq.get()
                if item is None:
                    left -= 1
                    continue
                sid, vol, t_io, t_b = item
                if sid == "__error__":
                    raise RuntimeError(f"volume worker: {vol}")
                inflight.release()
                timed[sid] = t_io
                t_load[0] += t_io
                t_vol[0] += t_b
                xb = free_bufs.get()
                if windows(vol, xb):
                    built.put((sid, xb))
                else:
                    free_bufs.put(xb)
            for _ in devs:
                built.put(None)
        except BaseException as e:
            for _ in devs:
                built.put(e)
            raise

    def head(f):
        h = f.astype(np.float32)
        mu = h.mean(1, keepdims=True)
        var = ((h - mu) ** 2).mean(1, keepdims=True)
        h = (h - mu) / np.sqrt(var + np.float32(1e-5)) * W["norm.weight"] + W["norm.bias"]
        a = np.tanh(h @ W["att.0.weight"].T + W["att.0.bias"]) @ W["att.3.weight"].T + W["att.3.bias"]
        a = np.exp(a - a.max(0, keepdims=True))
        a /= a.sum(0, keepdims=True)
        z = ((a.T @ h) * W["clsW"]).sum(-1) + W["clsb"]
        return 1.0 / (1.0 + np.exp(-z.astype(np.float64)))

    preds, ready = {}, {}

    def gpu_worker(d):
        ck(lib.cudaSetDevice(d), "cudaSetDevice")             # primary context current in this thread, then TensorRT
        xd, fd, st = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        ck(lib.cudaMalloc(ctypes.byref(xd), XB), "cudaMalloc x")
        ck(lib.cudaMalloc(ctypes.byref(fd), FB), "cudaMalloc f")
        ck(lib.cudaStreamCreate(ctypes.byref(st)), "cudaStreamCreate")
        eng = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(ENG)
        tctx = eng.create_execution_context()
        fh = host(FB, FSHAPE)
        tctx.set_tensor_address("x", xd.value)
        tctx.set_tensor_address("f", fd.value)
        fp = fh.ctypes.data

        def forward(xb):
            ck(lib.cudaMemcpyAsync(xd, xb.ctypes.data, XB, 1, st), "h2d")
            if not tctx.execute_async_v3(st.value):
                raise RuntimeError("execute_async_v3")
            ck(lib.cudaMemcpyAsync(fp, fd, FB, 2, st), "d2h")
            ck(lib.cudaStreamSynchronize(st), "sync")
            return fh
        if MODE == "eval":                                    # separate warm-up only for clean ms/study timing
            xb = free_bufs.get()
            xb[:] = 0
            head(forward(xb))
            free_bufs.put(xb)
        ready[str(d)] = f"{time.perf_counter() - T0:.2f}"
        while True:
            item = built.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            sid, xb = item
            t = time.perf_counter()
            f = forward(xb)
            free_bufs.put(xb)
            p = head(f)
            if not np.isfinite(p).all():
                raise ValueError(f"non-finite prediction {sid}")
            preds[sid] = p.tolist()
            with lk:
                t_gpu[0] += time.perf_counter() - t
    errs = []

    def run(fn, *a):
        try:
            fn(*a)
        except BaseException as e:                            # surface worker failures in the main thread
            errs.append(e)
            raise
    t_ready = time.perf_counter()
    if MODE == "eval":
        import faulthandler
        import sys

        last = [-1]
        def watch():                                          # progress, worker liveness, stall dump, hard limit
            while True:
                time.sleep(WATCH_S)
                el = time.perf_counter() - T0
                print(f"[watch {el:.0f}s] preds {len(preds)} vols {len(timed)} built {built.qsize()} free {free_bufs.qsize()} "
                      f"workers alive {[p_.is_alive() for p_ in procs]} exitcodes {[p_.exitcode for p_ in procs]}", flush=True)
                if len(preds) == last[0] and len(timed) < len(ids):    # stall only: no new prediction in WATCH_S
                    faulthandler.dump_traceback(all_threads=True)
                last[0] = len(preds)
                if el > EVAL_LIMIT_S:
                    import json
                    done = [s for s in ids if s in preds]
                    json.dump({"targets_in_order": TARGETS, "partial": True,
                               "predictions": [{"study_id": s, "probabilities": preds[s]} for s in done]},
                              open("/kaggle/working/eval_predictions.json", "w"))
                    print(f"[watch] EVAL_LIMIT_S reached, {len(done)} partial predictions written, exiting", flush=True)
                    sys.stdout.flush()
                    os._exit(3)
        threading.Thread(target=watch, daemon=True).start()
    workers = [threading.Thread(target=run, args=(gpu_worker, d)) for d in devs]
    workers += [threading.Thread(target=run, args=(collector,), daemon=True)]
    for w in workers:
        w.start()
    for w in workers[:len(devs)]:
        w.join()
    if errs:
        raise errs[0]
    stage("studies", stages)
    t_end = time.perf_counter()
    final = {s: preds.get(s, fallback[s]) for s in ids}
    with open("/kaggle/working/submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["StudyInstanceUID", *TARGETS])
        for s in ids:
            w.writerow([s, *[f"{v:.6f}" for v in final[s]]])
    msg = (f"{len(preds)}/{len(ids)} studies; total {time.perf_counter() - T0:.2f}s; gpus {len(devs)} ready at "
           + " ".join(ready.values()) + f" (head fetched {wbuf['t']:.2f}s) | " + " ".join(stages))
    if MODE == "eval":
        import json
        json.dump({"targets_in_order": TARGETS, "predictions": [{"study_id": s, "probabilities": final[s]} for s in ids]},
                  open("/kaggle/working/eval_predictions.json", "w"))
        k = len(ids)
        msg += (f" | studies {t_end - t_ready:.2f}s = {(t_end - t_ready) / k * 1000:.0f} ms/study (summed: io per study in the worker "
                f"processes {t_load[0] / k * 1000:.0f}, volume build {t_vol[0] / k * 1000:.0f}, gpu copy+forward+head "
                f"{t_gpu[0] / k * 1000:.0f}) "
                f"| cpus {os.cpu_count()} | cudart {cudart_path}")
        maps = {}
        for l in open("/proc/self/maps"):
            f = l.split()[-1]
            if ".so" in f and os.path.exists(f):
                maps[f] = os.path.getsize(f)
        top = sorted(maps.items(), key=lambda kv: -kv[1])[:25]
        msg += " | mapped libs MB: " + ", ".join(f"{os.path.basename(f)} {v / 1e6:.0f}" for f, v in top) + f" (total {sum(maps.values()) / 1e6:.0f} MB in {len(maps)} files)"
        if REF:
            ref = {r["study_id"]: r["probabilities"] for r in json.load(open(find(REF)))["predictions"]}
            com = [s for s in ids if s in ref and s in preds]
            dd = np.abs(np.array([preds[s] for s in com]) - np.array([ref[s] for s in com]))
            msg += f" | vs training pipeline max|d| {dd.max():.4f} mean|d| {dd.mean():.5f} on {len(com)}"
    print(msg, flush=True)
    if RECORD:
        record_trace("/kaggle/working/pagetrace.json")


NBWARM = """
try:
    import nbformat, nbconvert.nbconvertapp
    from nbconvert import HTMLExporter
    HTMLExporter().from_notebook_node(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1")]))
except Exception:
    pass
"""


def _nbwarm():
    import subprocess
    import sys
    subprocess.Popen([sys.executable, "-c", NBWARM], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     preexec_fn=lambda: os.nice(19))


def record_trace(out_path):
    """{file: [[offset, length], ...]} of page-cache-resident ranges (gaps < 128 KB merged) for every mapped file and
    every imported module file (+ its .pyc)."""
    import ctypes
    import json
    import mmap
    import sys
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    pg, files = mmap.PAGESIZE, set()
    for line in open("/proc/self/maps"):
        f = line.split()[-1]
        if f.startswith("/") and os.path.isfile(f) and not f.startswith("/kaggle"):
            files.add(f)
    for m in list(sys.modules.values()):
        for f in (getattr(m, "__file__", None), getattr(m, "__cached__", None)):
            if isinstance(f, str) and os.path.isfile(f) and not f.startswith("/kaggle"):
                files.add(f)
    trace, total = {}, 0
    for f in sorted(files):
        try:
            size = os.path.getsize(f)
            if size == 0:
                continue
            with open(f, "rb") as fh:
                mm = mmap.mmap(fh.fileno(), size, access=mmap.ACCESS_COPY)
            buf = ctypes.c_char.from_buffer(mm)
            n = (size + pg - 1) // pg
            vec = (ctypes.c_ubyte * n)()
            ok = libc.mincore(ctypes.addressof(buf), size, vec) == 0
            del buf
            mm.close()
            if not ok:
                continue
            res = np.frombuffer(vec, np.uint8) & 1
            idx = np.flatnonzero(res)
            if not len(idx):
                continue
            br = np.flatnonzero(np.diff(idx) > 32) + 1
            rs = [[int(a[0]) * pg, int(a[-1] - a[0] + 1) * pg] for a in np.split(idx, br)]
            trace[f] = rs
            total += sum(l for _, l in rs)
        except Exception:
            continue
    json.dump(trace, open(out_path, "w"))
    print(f"page trace: {len(trace)} files, {total / 1e6:.0f} MB resident", flush=True)


PAGETRACE = ""

if __name__ == "__main__":
    main()
    if MODE == "submit":                                      # submission.csv is closed: skip interpreter / CUDA teardown
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
