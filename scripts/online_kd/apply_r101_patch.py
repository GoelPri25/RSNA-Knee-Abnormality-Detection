"""Adds a ResNet-101 backbone (V4_BB=r101) to r50_gl4_runner.py's BB-CMP table: torchvision ImageNet-1k weights
(resnet101.tv_in1k, the same family as E4's resnet50.tv_in1k), crop branch = its own copy of layer4 (the BB-CMP path,
equivalent to l4copy), BatchNorm count 104 (stem + 33 blocks x 3 + 4 downsample). Everything else = the E4 recipe.
Usage: python apply_r101_patch.py /work/scripts/r50_gl4_runner.py
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]); src = p.read_text()
assert '"r101"' not in src, "already patched"
old = 'BACKBONES["rap2"] = {'
assert src.count(old) == 1
src = src.replace(old, '# R101 (2026-10-22, Priyanjali): E4 recipe on ResNet-101 (tv_in1k), crop branch = own layer4 copy (BB-CMP path)\n'
                       'BACKBONES["r101"] = {"arch": "resnet101.tv_in1k", "last": ("layer4",), "bns": 104}\n' + old)
compile(src, str(p), "exec"); p.write_text(src); print(f"patched {p} for V4_BB=r101")
