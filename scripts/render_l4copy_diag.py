"""Exp L4C-D renders: draw render_arrays.npz (scripts/l4copy_diag_gpu.py) as one PNG per (finding, study): top row the
top-1 crop window of MS1 (green) / self (red) / self_plane (blue) with its box, bottom row the crop itself (2x).
Usage: python3 -I scripts/render_l4copy_diag.py <diag dir>   (writes <diag dir>/renders/*.png)"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

D = Path(sys.argv[1]); z = np.load(D / "render_arrays.npz"); out = D / "renders"; out.mkdir(exist_ok=True)
SLOT = ["SagF", "SagNF", "CorF", "CorNF", "Ax"]; B = np.cumsum([0, 26, 22, 18, 12, 18])
COL = {"ms1": (0, 200, 0), "self": (230, 40, 40), "self_plane": (60, 120, 255)}
keys = sorted({k.rsplit("|", 2)[0] for k in z.files})
for key in keys:
    f, i = key.split("|"); sheet = Image.new("RGB", (3 * 384, 2 * 384)); d = ImageDraw.Draw(sheet)
    for c, nm in enumerate(("ms1", "self", "self_plane")):
        img = z[f"{key}|{nm}|img"]; s, r, q, box = (int(x) for x in z[f"{key}|{nm}|srq"]); hb = box // 2
        r, q = min(max(r, hb), 384 - hb), min(max(q, hb), 384 - hb)
        sheet.paste(Image.fromarray(img).convert("RGB"), (c * 384, 0))
        d.rectangle([c * 384 + q - hb, r - hb, c * 384 + q + hb, r + hb], outline=COL[nm], width=2)
        sl = [SLOT[np.searchsorted(B, x, side="right") - 1] for x in z[f"{key}|{nm}|slices"]]
        d.text((c * 384 + 6, 4), f"{nm}: slice {s} {sl[0]}", fill=COL[nm]); d.text((c * 384 + 6, 370), "4 windows: " + ",".join(sl), fill=COL[nm])
        crop = Image.fromarray(img[r - hb:r + hb, q - hb:q + hb]).resize((384, 384)).convert("RGB")
        sheet.paste(crop, (c * 384, 384))
    d.text((6, 388), f"{f} #{i}", fill=(255, 255, 0))
    sheet.save(out / f"{f.replace(' ', '').replace(chr(39), '')}_{int(i):02d}.png")
print(len(keys), "renders ->", out)
