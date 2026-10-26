"""Torch-free ConvNeXt-S (E4-CNXS) TensorRT kernel. The commit-run start of the TensorRT kernel (fast_cnxs_trtlean_cv) is
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
Usage: python3 -I make.py [eval N] [--record]   (writes cnxs_notorch[_eval|_record]/submission.py)
"""
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
SRC = (H.parent / "fast_cnxs_trtlean_cv" / "submission.py").read_text()
EVAL = int(sys.argv[sys.argv.index("eval") + 1]) if "eval" in sys.argv else 0
RECORD = "--record" in sys.argv
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "r7d"   # another R7-D-arch model: own names
R7D = "--r7d" in sys.argv          # R7-D (FB-F, heliosli R7 ConvNeXt-S init): own engine + head from rsna-fast-r7d-trt

top = SRC[:SRC.index("def main():")]
tail = SRC[SRC.index('NBWARM = """'):]
tail = tail[:tail.index("PAGETRACE = ")] + 'PAGETRACE = ""\n\n' + tail[tail.index('if __name__ == "__main__":'):]
doc_end = top.index('"""', 3) + 3
top = '"""' + __doc__.split("Usage:")[0].strip() + '\n"""' + top[doc_end:]


def sub(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b)


top = sub(top, 'SLIM = "tumis001/rsna-fast-cnxs-trt/cnxs_head.pt"', 'SLIM = "tumis001/rsna-fast-cnxs-np/cnxs_head.npz"')
top = sub(top, "WARM_TRACE = True ", "WARM_TRACE = False")
top = sub(top, "HALF = True\nWIN_CHUNK = 48\n", "NPROC = 4                          # volume worker processes (file reads + volume build, forked at start)\n"
          "WLOOK = 4                          # studies loading at once per worker\n"
          "INFLIGHT = 24                      # studies sent to the workers and not yet consumed\n"
          "IO_AFTER_PACK = False              # study reads start after the TensorRT pack + head are read\n"
          "WATCH_S = 30                       # eval: progress line every WATCH_S, thread dumps when stalled\n"
          "EVAL_LIMIT_S = 1800                # eval: hard exit\n"
          "NBUF = 6                           # pinned host input buffers (94 x 3 x 320 x 320 fp16, 58 MB each)\n"
          "CV_THREADS = -1                    # cv2.setNumThreads (-1: cv2 default)\n")
if EVAL:
    top = sub(top, 'MODE = "submit"\n', 'MODE = "eval"\n')
    top = sub(top, "EVAL_N = 825\n", f"EVAL_N = {EVAL}\n")
    top = sub(top, "REF = None\n", 'REF = "tumis001/rsna-fast-cnxs-np/cnxs_val_swa_global.json"\n')
if RECORD:
    top = sub(top, "RECORD = False", "RECORD = True ")

MAIN = r'''def cudart():
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
        with open(trt_pack() + "/cnxs_vc.engine", "rb") as fh:
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


'''
if R7D:
    MAIN = MAIN.replace('open(trt_pack() + "/cnxs_vc.engine", "rb")', 'open(find("tumis001/rsna-fast-r7d-trt/r7d_vc.engine"), "rb")')
    top = sub(top, 'SLIM = "tumis001/rsna-fast-cnxs-np/cnxs_head.npz"', 'SLIM = "tumis001/rsna-fast-r7d-trt/r7d_head.npz"')
    if EVAL:
        top = sub(top, 'REF = "tumis001/rsna-fast-cnxs-np/cnxs_val_swa_global.json"', 'REF = "tumis001/rsna-fast-r7d-trt/r7d_val_swa_global.json"')
    top = top.replace("Torch-free ConvNeXt-S (E4-CNXS) TensorRT kernel.", "Torch-free ConvNeXt-S TensorRT kernel, R7-D weights (FB-F R7-D: heliosli R7 init, FB-F mix73 targets over competition + fastMRI, SWA 6-8).", 1)
    assert "r7d_vc.engine" in MAIN
assert "import torch" not in top + tail, "torch in the reused sections"
out = H.parent / (("r7d" if R7D else "cnxs") + "_notorch" + ("_record" if RECORD else "_eval" if EVAL else ""))
code = top + MAIN + tail
if TAG != "r7d":
    assert R7D; code = code.replace("r7d", TAG); out = H.parent / out.name.replace("r7d", TAG)
out.mkdir(exist_ok=True)
(out / "submission.py").write_text(code)
import json
slug = (f"rsna-fast-{TAG}-notorch" if R7D else "rsna-fast-cnxs-notorch") + ("-record" if RECORD else "-eval" if EVAL else "")
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"tumis001/{slug}", "title": slug, "code_file": "submission.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "dataset_sources": ["tumis001/rsna-fast-cnxs-trt", f"tumis001/rsna-fast-{TAG}-trt" if R7D else "tumis001/rsna-fast-cnxs-np"] + (["tumis001/rsna-fast-eval"] if EVAL else []),
    "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
    "machine_shape": "NvidiaTeslaT4"}, indent=2))
print("wrote", out / "submission.py")
