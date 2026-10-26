"""CPU tests for scripts/l4copy_r2_runner.py (real timm, real v4 l4copy weights, no network). Run in localhost/rsna-mlx-train:v1
with the same mounts as test_l4copy_r1_cpu.py (+ /v4 = data/kaggle_datasets/rsna-v4-l4copy-fold0).
F1 arm crop, GL_SMOKE end-to-end: every weight outside layer4_c / norm_l / clsV / clsvb and every GLOBAL BN statistic is
   bitwise the parent's after training + SWA + refresh (so the global logits and the self placement are the parent's);
   the trained crop weights moved.
F2 arm all, GL_SMOKE end-to-end (table rebuilt per epoch, two-pass refresh).
"""
import os
import subprocess
import sys

ENV = {"D96_PHASE": "r50_gl", "GL_OUT": "/tmp/out", "GL_CROPS": "/tmp/cp", "D96_CACHE": "/c", "D96_ASSETS": "/a",
       "OMP_NUM_THREADS": "8", "V4_ARM": "l4copy", "GL_SMOKE": "1", "R2_INIT": "/v4/weights_swa21_25.pt", "R2_EPOCHS": "2"}


def run(arm):
    import numpy as np
    import torch
    sys.path.insert(0, "/w")
    from test_l4copy_r1_cpu import setup
    setup()
    import l4copy_r2_runner as R2
    R2.main()
    parent = torch.load(os.environ["R2_INIT"], map_location="cpu")
    new = torch.load(next(R2.OUT.glob("weights_swa*.pt")), map_location="cpu")
    same, moved = [], []
    for k, v in parent.items():
        (same if torch.equal(v, new[k]) else moved).append(k)
    if arm == "crop":
        bad = [k for k in moved if not (k.startswith(R2.CROP_TRAIN) or k.endswith(("crop_mean", "crop_var", "crop_nbt")))]
        assert not bad, bad[:10]
        assert any(k.startswith("layer4_c.") and k.endswith("weight") for k in moved)
        assert all(k in same for k in parent if k.endswith(("running_mean", "running_var")))
        print(f"F1 ok: {len(moved)} tensors moved, all crop-side; global weights + global BN statistics bitwise the parent's")
    else:
        assert any(k.startswith("backbone.layer1") for k in moved) and any(k.startswith("clsW") for k in moved)
        print(f"F2 ok: {len(moved)} tensors moved incl. shared backbone and global head")
    print(f"ARM_OK {arm}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        fails = 0
        for arm in ("crop", "all"):
            e = {**os.environ, **ENV, "R2_ARM": arm, "GL_OUT": f"/tmp/out_{arm}"}
            fails += subprocess.run([sys.executable, "-u", __file__, arm], env=e).returncode != 0
        print("ALL_OK" if not fails else f"FAILED {fails}")
        sys.exit(fails)
