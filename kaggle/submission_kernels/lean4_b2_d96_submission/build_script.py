"""Build E-ref E2 (L4 + Model B2): lean4_d96_submission/submission.py with the B2 Swin3D submission script embedded and
run first as a subprocess; B2's probabilities join the four L4 members in the equal mean (studies B2 cannot read fall
back to the L4 members). Run from this directory: python3 build_script.py"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
l4 = (HERE.parent / "lean4_d96_submission/submission.py").read_text()
b2 = (HERE.parent / "d96n_swin3d_b2_submission/submission.py").read_text()

# B2 as a member: own output folder, predictions JSON for readable studies only
for a, b in [
    ('OUT = Path("/kaggle/working/submission.csv")',
     'MEMBER = Path("/kaggle/working/b2_member")\nMEMBER.mkdir(exist_ok=True)\nOUT = MEMBER / "submission.csv"'),
    ('            if not (slot_of >= 0).any():\n                stats["empty_studies"] += 1\n',
     '            if not (slot_of >= 0).any():\n                stats["empty_studies"] += 1\n                empty.add(s)\n'),
    ('    preds, stats = {}, {"studies": len(ids), "empty_studies": 0, "decode_errors": 0}',
     '    preds, stats, empty = {}, {"studies": len(ids), "empty_studies": 0, "decode_errors": 0}, set()'),
    ('    Path("/kaggle/working/inference_summary.json").write_text(json.dumps(summary, indent=2) + "\\n")',
     '    (MEMBER / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\\n")\n'
     '    (MEMBER / "predictions.json").write_text(json.dumps({s: p for s, p in preds.items() if s not in empty}) + "\\n")'),
]:
    assert b2.count(a) == 1, a[:60]
    b2 = b2.replace(a, b)
B2_SRC = repr(b2)

for a, b in [
    ('"""Lean four-member', '"""E-ref E2: L4 + Model B2 Swin3D (5 members, equal probability mean). Built by build_script.py from\n'
                           'lean4_d96_submission/submission.py and d96n_swin3d_b2_submission/submission.py (B2 runs first in a subprocess).\n\n'
                           'Original L4 docstring:\nLean four-member'),
    ('def main():\n    started = time.monotonic()\n    work = Path("/kaggle/working")\n',
     'B2_SRC = ' + B2_SRC + '\n\n\ndef run_b2(work):\n'
     '    """Model B2 member: its own script (parity check on 8 fold-0 studies included) in a subprocess, GPU freed after."""\n'
     '    import subprocess, sys\n'
     '    (work / "b2_member").mkdir(exist_ok=True)\n'
     '    src = work / "b2_member" / "b2_submission.py"\n'
     '    src.write_text(B2_SRC)\n'
     '    subprocess.run([sys.executable, "-u", str(src)], check=True, cwd=str(work / "b2_member"))\n'
     '    return json.loads((work / "b2_member" / "predictions.json").read_text())\n\n\n'
     'def main():\n    started = time.monotonic()\n    work = Path("/kaggle/working")\n    b2_pred = run_b2(work)\n'),
    ('    members = {**quad_ind, **d96_ind}\n    assert len(members) == 4, list(members)',
     '    members = {**quad_ind, **d96_ind, "swin3d_b2": b2_pred}\n    assert len(members) == 5, list(members)'),
    ('"equal arithmetic mean of 4 members\' probabilities"', '"equal arithmetic mean of 5 members\' probabilities (L4 + B2)"'),
]:
    assert l4.count(a) == 1, a[:60]
    l4 = l4.replace(a, b)
(HERE / "submission.py").write_text(l4)
print("wrote", HERE / "submission.py")
