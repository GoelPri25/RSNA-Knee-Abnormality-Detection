"""Third patch (after apply_online_patch.py; not with apply_logit_patch.py): pairwise ranking loss, V4_RANK=<weight>.
Per head: BCE(z, y) + V4_RANK * RankBank loss on pairs ordered by the training target y (E4 target or any V4_TARGETS file).
Usage: python apply_rank_patch.py /work/scripts/r50_gl4_runner.py
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]); src = p.read_text()
assert "V4_ONLINE" in src and "V4_LOGIT" not in src and "V4_RANK" not in src, "needs the online patch, without the logit patch"
EDITS = [
    ('    CFG["version"] = f"{CFG[\'version\']}_online"\n',
     '    CFG["version"] = f"{CFG[\'version\']}_online"\n'
     'RANK = float(os.environ.get("V4_RANK", "0") or 0)\n'
     'RBG = RBL = None\n'
     'if RANK > 0:\n'
     '    import online_kd\n'
     '    assert ARM == "l4copy" and IMP == "none" and BB == "r50" and TARGETS_FILE and not EXT and not NOISE, "RANK: E4 recipe only"\n'
     '    G4["rank"] = online_kd.rank_cfg()\n'
     '    CFG["version"] = f"{CFG[\'version\']}_rk{RANK:g}"\n'),
    ("    if sil is None:\n        return torch.stack([v2.bce(g, y, w), v2.bce(l, y, w * cmask)])\n",
     "    if sil is None:\n"
     "        if RANK > 0 and not ext:\n"
     "            global RBG, RBL\n"
     "            if RBG is None:\n"
     "                RBG, RBL = online_kd.RankBank(dev), online_kd.RankBank(dev)\n"
     "            return torch.stack([v2.bce(g, y, w) + RANK * RBG.loss(g, y),\n"
     "                                v2.bce(l, y, w * cmask) + RANK * RBL.loss(l, y, cmask)])\n"
     "        return torch.stack([v2.bce(g, y, w), v2.bce(l, y, w * cmask)])\n"),
]
for old, new in EDITS:
    assert src.count(old) == 1, ("anchor not found exactly once", old[:70], src.count(old))
    src = src.replace(old, new)
compile(src, str(p), "exec")
p.write_text(src)
print(f"patched {p} for V4_RANK ({len(EDITS)} edits)")
