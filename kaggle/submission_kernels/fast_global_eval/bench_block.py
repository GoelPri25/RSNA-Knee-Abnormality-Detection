# ---------------------------------------------------------------- eval-only benchmarks (appended by make_eval.sh)
def bench():
    """Where startup CPU goes now that the page cache is warm: -X importtime for torch and nbconvert, bytecode caches."""
    import importlib.util
    import json
    import subprocess
    import sys
    out = {"python": sys.version.split()[0], "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
           "dont_write_bytecode": sys.flags.dont_write_bytecode, "pycache_prefix": sys.pycache_prefix}

    def importtime(code):
        walls = []
        for _ in range(2):
            t = time.perf_counter()
            subprocess.run([sys.executable, "-c", code], capture_output=True)
            walls.append(round(time.perf_counter() - t, 2))
        r = subprocess.run([sys.executable, "-X", "importtime", "-c", code], capture_output=True, text=True)
        rows = []
        for l in r.stderr.splitlines():
            if l.startswith("import time:") and "|" in l:
                p = l[len("import time:"):].split("|")
                try:
                    rows.append((int(p[1]), int(p[0]), p[2].strip()))
                except ValueError:
                    pass
        return {"wall_s": walls, "modules": len(rows), "self_ms_sum": sum(r[1] for r in rows) // 1000,
                "top_self_ms": [(m, s // 1000) for c, s, m in sorted(rows, key=lambda r: -r[1])[:25]],
                "top_cum_ms": [(m, c // 1000) for c, s, m in sorted(rows, key=lambda r: -r[0])[:25]]}
    out["import torch"] = importtime("import torch, torch.nn.functional")
    out["import nbconvert"] = importtime("import nbformat, nbconvert.nbconvertapp; from nbconvert import HTMLExporter")
    t = time.perf_counter()
    subprocess.run(["jupyter-nbconvert", "--version"], capture_output=True)
    out["jupyter-nbconvert --version s"] = round(time.perf_counter() - t, 2)

    def pyc(mod):
        d = os.path.dirname(importlib.util.find_spec(mod).origin)
        n_py = n_pyc = 0
        for root, _, files in os.walk(d):
            tag = os.path.join(root, "__pycache__")
            have = set(os.listdir(tag)) if os.path.isdir(tag) else set()
            for f in files:
                if f.endswith(".py"):
                    n_py += 1
                    n_pyc += any(x.startswith(f[:-3] + ".") and x.endswith(".pyc") for x in have)
        return {"dir": d, "py": n_py, "with_pyc": n_pyc, "writable": os.access(d, os.W_OK)}
    for m in ("torch", "nbconvert", "jinja2", "pygments", "traitlets", "mistune", "bleach"):
        try:
            out[f"pyc {m}"] = pyc(m)
        except Exception as e:
            out[f"pyc {m}"] = repr(e)
    print("BENCH " + json.dumps(out), flush=True)


bench()
