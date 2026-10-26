"""Second patch (after apply_online_patch.py): logit matching, V4_LOGIT=<weight> (unset or 0 = unchanged).
Offline (no V4_ONLINE): teacher logit = logit(pure E4 teacher target); online: the shifted teacher logit of the view.
Loss per head: BCE(z, y_mixed) + V4_LOGIT * weighted mean (z - z_teacher)^2 (global: 12 findings; crop: the 10 cropped).
Usage: python apply_logit_patch.py /work/scripts/r50_gl4_runner.py
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]); src = p.read_text()
assert "V4_ONLINE" in src, "apply_online_patch.py first"
assert "V4_LOGIT" not in src, "already patched"
EDITS = [
    ('    CFG["version"] = f"{CFG[\'version\']}_online"\n',
     '    CFG["version"] = f"{CFG[\'version\']}_online"\n'
     'LOGIT = float(os.environ.get("V4_LOGIT", "0") or 0)\n'
     'LKD = None\n'
     'if LOGIT > 0:\n'
     '    import online_kd\n'
     '    assert ARM == "l4copy" and IMP == "none" and BB == "r50" and TARGETS_FILE and not EXT and not NOISE, "LOGIT: E4 recipe only"\n'
     '    G4["logit"] = online_kd.logit_cfg()\n'
     '    CFG["version"] = f"{CFG[\'version\']}_lg{LOGIT:g}"\n'),
    ("    if sil is None:\n        return torch.stack([v2.bce(g, y, w), v2.bce(l, y, w * cmask)])\n",
     "    if sil is None:\n"
     "        if LOGIT > 0:\n"
     "            zt = OKD.last_logit if ONLINE else LKD.logits(batch[11])\n"
     "            wl = w * cmask\n"
     "            mg = (((g.float() - zt) ** 2) * w).sum() / w.sum().clamp_min(1)\n"
     "            ml = (((l.float() - zt) ** 2) * wl).sum() / wl.sum().clamp_min(1)\n"
     "            return torch.stack([v2.bce(g, y, w) + LOGIT * mg, v2.bce(l, y, wl) + LOGIT * ml])\n"
     "        return torch.stack([v2.bce(g, y, w), v2.bce(l, y, w * cmask)])\n"),
    ("    elif ONLINE:\n",
     "    elif LOGIT > 0 and not ONLINE:\n"
     "        global LKD\n"
     "        train_ds = GLStudiesI(train_ids, meta, tlabels, CFG[\"train_windows\"], True, tab)\n"
     "        LKD = online_kd.LogitTargets(train_ids, r.TARGETS, dev)\n"
     "    elif ONLINE:\n"),
]
for old, new in EDITS:
    assert src.count(old) == 1, ("anchor not found exactly once", old[:70], src.count(old))
    src = src.replace(old, new)
compile(src, str(p), "exec")
p.write_text(src)
print(f"patched {p} for V4_LOGIT ({len(EDITS)} edits)")
