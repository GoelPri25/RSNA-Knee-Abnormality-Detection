"""Builds the single-file Kaggle kernels: inlines gl4_core.py at `# <<GL4_CORE>>` and fills `FID = {}  # <<FID>>` with
FID_N fold-0 studies' saved global-head probabilities per model (from the local handoff packages), then writes
kernels/<slug>/{submission.py, kernel-metadata.json}.
Usage: python build_kernels.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[2] / "reports"
REFS = {"cnxs": REPORTS / "kd_e4cnxs_handoff/fold0_predictions/val_swa_global.json",
        "bit101f1": REPORTS / "kd_e4bit101f1_handoff/fold1_predictions/val_swa_global.json",
        "e4on": REPORTS / "online_kd_handoff/fold0_predictions/val_swa_global.json"}
KERNELS = {"rank_blend_submission.py": "priyanjaliml/rsna-gl4-rank-blend-submission",
           "mirror_tta_submission.py": "priyanjaliml/rsna-gl4-mirror-tta-submission"}
N = 8


RAP2_KERNEL = HERE.parents[2].parent / "skanee-d96-handoff/kaggle/submission_kernels/pubkd_rap2d_global_submission/submission.py"


def ref_preds(n):
    if n == "rap2":                                          # the teammate kernel's own 8 fold-0 references
        line = next(l for l in RAP2_KERNEL.read_text().splitlines() if l.startswith("FID_REF = "))
        return json.loads(line[len("FID_REF = "):])
    return {r["study_id"]: r["probabilities"] for r in json.loads(REFS[n].read_text())["predictions"]}


def fid_refs(names):
    """Per model its own validation fold's first N studies (models validated on different folds share none)."""
    preds = {n: ref_preds(n) for n in names}
    return {n: {s: [round(v, 6) for v in preds[n][s]] for s in sorted(preds[n])[:N]} for n in names}


core = (HERE / "gl4_core.py").read_text()
for src, slug in KERNELS.items():
    text = (HERE / src).read_text()
    assert text.count("# <<GL4_CORE>>") == 1 and text.count("FID = {}") == 1
    names = re.findall(r'^    \{"name": "(\w+)"', text, re.M) or re.findall(r'^MODEL = \{"name": "(\w+)"', text, re.M)
    datasets = sorted(set(re.findall(r'^\s+(?:\{"name".*\n\s+)?"dataset": "([\w-]+/[\w-]+)"', text, re.M)) | {"tumis001/rsna-coatnet-d96-exp1b-fold0"})
    text = text.replace("# <<GL4_CORE>>", core.rstrip())
    text = text.replace("FID = {}", f"FID = {json.dumps(fid_refs(names))}")
    compile(text, src, "exec")
    out = HERE / "kernels" / slug.split("/")[1]
    out.mkdir(parents=True, exist_ok=True)
    (out / "submission.py").write_text(text)
    meta = {"id": slug, "title": slug.split("/")[1], "code_file": "submission.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": datasets, "kernel_sources": [],
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{out}: models {names}, datasets {datasets}, {len(text.splitlines())} lines")
