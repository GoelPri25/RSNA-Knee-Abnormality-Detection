"""Adds ConvNeXt-S ImageNet-22k (V4_BB=cnxs) to r50_gl4_runner.py's BB-CMP table: convnext_small.fb_in22k_ft_in1k (the
init of heliosli's R6 / R7 / S6 and nartaa's S6-all-data, public 0.950 alone). Same path as the existing ConvNeXt-T entry:
crop branch = own copy of stages.3 + head.norm, LayerNorm backbone (no BatchNorm), ImageNet normalisation, E4's optimiser
settings (S6 used the same lr 1e-4 / head 1e-3 / wd 0.05). Everything else = the E4 recipe.
Usage: python apply_cnxs_patch.py /work/scripts/r50_gl4_runner.py
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]); src = p.read_text()
assert '"cnxs"' not in src, "already patched"
old = 'BACKBONES["rap2"] = {'
assert src.count(old) == 1
src = src.replace(old, '# CNXS (2026-10-22, Priyanjali): E4 recipe on ConvNeXt-S 22k (S6 / R6 init), crop branch = own stages.3 copy\n'
                       'BACKBONES["cnxs"] = {"arch": "convnext_small.fb_in22k_ft_in1k", "last": ("stages.3", "head.norm"), "bns": 0}\n' + old)
compile(src, str(p), "exec"); p.write_text(src); print(f"patched {p} for V4_BB=cnxs")
