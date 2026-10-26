"""MS1 crop table -> slim copy for v4 (ids, centres, a_top only; drops MS1's logits and features, 1.3 GB -> ~2 MB).
Asserts coverage of every labelled study (4,349). Usage: python slim_ms1_crops.py <crop_probe dir> <out dir> <labels json>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

src, dst, lab = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
seen = []
for p in sorted(src.glob("cp_*.npz")):
    z = np.load(p, allow_pickle=False)
    np.savez(dst / p.name, ids=z["ids"], centres=z["centres"], a_top=z["a_top"])
    seen += [str(s) for s in z["ids"]]
labels = json.loads(lab.read_text())
ids = {r["study_id"] for r in labels["labels"]}
assert len(seen) == len(set(seen)), "duplicate study in the crop table"
assert ids <= set(seen), f"{len(ids - set(seen))} labelled studies missing"
man = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dst.glob("cp_*.npz"))}
(dst / "manifest.json").write_text(json.dumps({"studies": len(seen), "labelled": len(ids), "files": man}, indent=1) + "\n")
print(f"slim {len(man)} files, {len(seen)} studies, covers {len(ids)} labelled")
