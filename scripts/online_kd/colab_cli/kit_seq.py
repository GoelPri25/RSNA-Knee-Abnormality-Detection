# Sequential, rate-limit-aware fetch of the missing kit files (Kaggle returned 429 to parallel per-file downloads).
import os, subprocess, time
os.environ["KAGGLE_API_TOKEN"] = open("/content/kaggle.env").read().strip().split("=", 1)[1]
DS, ROOT = "tumis001/rsna-pubkd-teammate-kit", "/content/kit"
files = [l.strip() for l in open("/content/okd/files_rsna-pubkd-teammate-kit.txt") if l.strip()]
todo = [f for f in files if not os.path.exists(f"{ROOT}/{f}")]
log = open("/content/okd/kit_seq.log", "a"); print(f"{len(todo)} of {len(files)} missing", file=log, flush=True)
for f in todo:
    d = f"{ROOT}/{os.path.dirname(f)}"; os.makedirs(d, exist_ok=True)
    for t in range(8):
        r = subprocess.run(["kaggle", "datasets", "download", DS, "-f", f, "-p", d, "-q"], capture_output=True, text=True)
        z = f"{d}/{os.path.basename(f)}.zip"
        if os.path.exists(z):
            subprocess.run(["unzip", "-oq", z, "-d", d]); os.remove(z)
        if os.path.exists(f"{ROOT}/{f}"):
            break
        wait = 60 if "429" in (r.stdout + r.stderr) else 10
        print(f"retry {t} {f}: {(r.stdout + r.stderr)[-120:].strip()} -> sleep {wait}", file=log, flush=True)
        time.sleep(wait)
    print(("OK " if os.path.exists(f"{ROOT}/{f}") else "MISSING ") + f, file=log, flush=True)
    time.sleep(1.5)
left = [f for f in files if not os.path.exists(f"{ROOT}/{f}")]
print(f"KIT_DONE missing {len(left)}", left[:5], file=log, flush=True)
