"""fastMRI d96 cache on Modal (2026-10-22, for lane C): r7's build (fmb.tar = fastmri_build_cache.py sha 451f3ed8 +
MeluXina pylibs + the signed URL list; data/_session_tmp/colab/r7vm/fm_cache_run.sh logic) in one CPU container,
written straight to the volume at fm/shard_{0..7}. Batch 2 -> shards 0-1 (5 workers), batch 1 -> shards 2-7 (6 workers),
both streams at once, as on r7. The tarball (with the URL list) is staged at fmbuild/fmb.tar by `modal volume put` and
deleted from the volume after the build, whatever the outcome. Status lines: BUILD <tag> OK / FAILED; ALL END.
Usage: source ~/.config/rsna/modal-personal.env && modal volume put rsna-coatnet-d96-cache <fmb.tar> fmbuild/fmb.tar
       && modal run --detach scripts/modal_fmcache.py
"""
import shutil
import subprocess
import threading
from pathlib import Path

import modal

volume = modal.Volume.from_name("rsna-coatnet-d96-cache")
image = modal.Image.debian_slim(python_version="3.11").apt_install("xz-utils", "curl").pip_install("numpy==2.4.6")
app = modal.App("rsna-fmcache", image=image)
VOL = Path("/root/vol")


@app.function(cpu=16, memory=65536, timeout=4 * 3600, volumes={str(VOL): volume})
def build():
    volume.reload()
    tar, out, fmb = VOL / "fmbuild/fmb.tar", VOL / "fm", Path("/root/fmb")
    try:
        subprocess.run(["tar", "xf", str(tar), "-C", "/root"], check=True)
        urls = dict(l.split()[:2] for l in (fmb / "urls.txt").read_text().splitlines() if l.strip())
        assert set(urls) >= {"knee_DICOMs_batch1.tar.xz", "knee_DICOMs_batch2.tar.xz"}, "URL list incomplete"
        out.mkdir(exist_ok=True)
        res = {}

        def run(archive, tag, offset, nshard, workers):
            for t in range(3):
                shutil.rmtree(f"/root/fmstage/{tag}", ignore_errors=True)
                for k in range(offset, offset + nshard):
                    shutil.rmtree(out / f"shard_{k}", ignore_errors=True); (out / f"shard_{k}").mkdir(parents=True)
                cmd = (f"set -o pipefail; curl -sSfL --retry 5 --retry-delay 10 \"$URL\" | xz -T6 -dc | "
                       f"FM_STAGE=/root/fmstage FM_PYLIBS={fmb}/libs python -I {fmb}/fastmri_build_cache.py - {out} {tag} "
                       f"{offset} {nshard} {workers} > /root/{tag}.log 2>&1")
                rc = subprocess.run(["bash", "-c", cmd], env={"PATH": "/usr/local/bin:/usr/bin:/bin", "URL": urls[archive]}).returncode
                print(f"BUILD {tag} try {t + 1} rc {rc}; log tail: "
                      + " | ".join(Path(f"/root/{tag}.log").read_text().splitlines()[-2:])[:400], flush=True)
                if rc == 0:
                    res[tag] = "OK"; return
            res[tag] = "FAILED"

        th = [threading.Thread(target=run, args=a) for a in (("knee_DICOMs_batch2.tar.xz", "b2", 0, 2, 5),
                                                              ("knee_DICOMs_batch1.tar.xz", "b1", 2, 6, 6))]
        [x.start() for x in th]; [x.join() for x in th]
        for tag, s in res.items():
            print(f"BUILD {tag} {s}", flush=True)
            shutil.copy(f"/root/{tag}.log", out / f"build_{tag}.log")
    finally:
        tar.unlink(missing_ok=True); shutil.rmtree(fmb, ignore_errors=True)
        volume.commit()
    print("ALL END", sorted((p.name, len(list(p.glob("vol_*.npy")))) for p in out.glob("shard_*")), flush=True)
    assert res == {"b2": "OK", "b1": "OK"}, res


@app.local_entrypoint()
def main():
    build.remote()
