"""Patch the teammate's r50_gl4_runner.py (kit version, 61,690 bytes) for online distillation (V4_ONLINE=1).

Every edit is an exact, asserted string replacement; with V4_ONLINE unset the runner behaves exactly as before.
  1. config: V4_ONLINE lane flag, recorded in G4 (so a resume with a different setup is refused) and in CFG["version"]
  2. train_step_loss: after the batch's offline targets are moved to the GPU, replace them by the online targets
  3. main: the training dataset returns study indices (GLStudiesI) and the teacher is built after the crop table
  4. per-epoch log: the size of the online shift (mean |d|, rms, clipped share)
Usage: python apply_online_patch.py /work/scripts/r50_gl4_runner.py
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text()
assert "V4_ONLINE" not in src, "already patched"

EDITS = [
    ("class DualBN4(v2.DualBN):",
     '''# ONLINE-KD (2026-10-21): the Raptor teacher scores the student's own augmented windows every step (online_kd.py).
ONLINE = os.environ.get("V4_ONLINE") == "1"
OKD = None
if ONLINE:
    import online_kd
    assert ARM == "l4copy" and IMP == "none" and BB == "r50" and TARGETS_FILE and not EXT and not NOISE, "ONLINE: E4 recipe only"
    G4["online"] = online_kd.config()
    CFG["version"] = f"{CFG['version']}_online"


class DualBN4(v2.DualBN):'''),
    ("    fid, wc, y, w = v2.to_dev(dev, batch[9], batch[10], batch[2], batch[3])\n",
     "    fid, wc, y, w = v2.to_dev(dev, batch[9], batch[10], batch[2], batch[3])\n"
     "    if ONLINE and not ext:\n"
     "        y = OKD.target(xg, b, batch[11])\n"),
    ("    else:\n        train_ds = v2.GLStudies(train_ids, meta, tlabels, CFG[\"train_windows\"], True, tab)\n",
     "    elif ONLINE:\n"
     "        global OKD\n"
     "        train_ds = GLStudiesI(train_ids, meta, tlabels, CFG[\"train_windows\"], True, tab)\n"
     "        OKD = online_kd.OnlineTeacher(train_ids, labels, tlabels, r.TARGETS, dev)\n"
     "        OKD.check_clean(meta)\n"
     "    else:\n        train_ds = v2.GLStudies(train_ids, meta, tlabels, CFG[\"train_windows\"], True, tab)\n"),
    ("        run = np.zeros(3); n = 0; gn = np.zeros(len(clip)); nst = 0; run_x = np.zeros(4)\n",
     "        run = np.zeros(3); n = 0; gn = np.zeros(len(clip)); nst = 0; run_x = np.zeros(4)\n"
     "        if ONLINE:\n"
     "            OKD.reset_stats()\n"),
    ('                "lr": [g["lr"] for g in opt.param_groups]}\n',
     '                "lr": [g["lr"] for g in opt.param_groups]}\n'
     "        if ONLINE:\n"
     "            rec_.update(OKD.epoch_stats())\n"),
]
for old, new in EDITS:
    assert src.count(old) == 1, ("anchor not found exactly once", old[:70], src.count(old))
    src = src.replace(old, new)
compile(src, str(p), "exec")
p.write_text(src)
print(f"patched {p} for V4_ONLINE ({len(EDITS)} edits)")
