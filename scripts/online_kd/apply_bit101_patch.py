"""Adds Google's BiT ResNet-101 (ImageNet-21k, then ImageNet-1k; V4_BB=bit101) to r50_gl4_runner.py's BB-CMP table:
resnetv2_101x1_bit.goog_in21k_ft_in1k, a pre-activation ResNet-V2 with GroupNorm + weight standardisation (no BatchNorm,
so no crop statistics and no BN refresh effect), inputs normalised with mean / std 0.5 (the checkpoint's), crop branch =
its own copy of stages.3 + the final norm (as ConvNeXt's stages.3 + head.norm). Everything else = the E4 recipe.
Usage: python apply_bit101_patch.py /work/scripts/r50_gl4_runner.py
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]); src = p.read_text()
assert '"bit101"' not in src, "already patched"
old = 'BACKBONES["rap2"] = {'
assert src.count(old) == 1
src = src.replace(old, '# BIT101 (2026-10-22, Priyanjali): E4 recipe on BiT ResNet-101 (ImageNet-21k), crop branch = own stages.3 + norm\n'
                       'BACKBONES["bit101"] = {"arch": "resnetv2_101x1_bit.goog_in21k_ft_in1k", "last": ("stages.3", "norm"), "bns": 0,\n'
                       '                       "norm": [0.5, 0.5]}\n' + old)
compile(src, str(p), "exec"); p.write_text(src); print(f"patched {p} for V4_BB=bit101")
