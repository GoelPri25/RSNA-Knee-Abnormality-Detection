"""Lets r50_gl4_runner.py validate on another fold (V4_VAL_FOLD=<k>; unset or 0 = the runner exactly as before).

Fold k != 0: train on every other fold (folds 0-5 minus k), validate on fold k. The Steven-v4 validation labels exist
for fold 0 only, so on fold k both rulers are label-v2 (the val_* readouts duplicate lv2_*). The fold is recorded in G4,
so a resume with a different fold is refused.
For the teammate kit's runner copy (tumis001/rsna-pubkd-teammate-kit, used on Modal 2026-10-22). The runner on the
branch has its own V4_VAL_FOLD (K-fold, 2026-10-22) and is refused here ("already patched").
Usage: python apply_valfold_patch.py /work/scripts/r50_gl4_runner.py
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]); src = p.read_text()
assert "V4_VAL_FOLD" not in src, "already patched"
old = ('    train_ids = sorted(s for s in labels if fold[s] != "0"); val_ids = sorted(s for s in labels if fold[s] == "0")\n'
       '    assert len(train_ids) == 3524 and len(val_ids) == 825 and set(val_ids) == set(ref)\n')
new = ('    VF = os.environ.get("V4_VAL_FOLD", "0")                   # VAL-FOLD (2026-10-22): validation fold, default 0\n'
       '    train_ids = sorted(s for s in labels if fold[s] != VF); val_ids = sorted(s for s in labels if fold[s] == VF)\n'
       '    if VF == "0":\n'
       '        assert len(train_ids) == 3524 and len(val_ids) == 825 and set(val_ids) == set(ref)\n'
       '    else:                                                    # no Steven-v4 labels outside fold 0: label-v2 twice\n'
       '        assert len(train_ids) + len(val_ids) == 4349 and not set(val_ids) & set(ref), VF\n'
       '        ref = {s: labels[s] for s in val_ids}; G4["val_fold"] = VF\n'
       '        print(f"VAL_FOLD {VF}: train {len(train_ids)} (fold 0 included), validation {len(val_ids)}", flush=True)\n')
assert src.count(old) == 1
src = src.replace(old, new)
compile(src, str(p), "exec"); p.write_text(src); print(f"patched {p} for V4_VAL_FOLD")
