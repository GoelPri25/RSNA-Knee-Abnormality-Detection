"""Render the native-capped cache kernels from run_template.py: 6 shard kernels + 1 smoke kernel (shard 0, 20 studies).

  python3 kaggle/experiment_kernels/d96_native_cache/render.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
template = (HERE / "run_template.py").read_text()
jobs = [(f"shard{i}", i, "None", f"tumis001/rsna-d96-native-cache-shard-{i}", f"RSNA d96 Native Cache Shard {i}")
        for i in range(6)]
jobs.append(("smoke", 0, "20", "tumis001/rsna-d96-native-cache-smoke", "RSNA d96 Native Cache Smoke"))
for name, shard, limit, kid, title in jobs:
    d = HERE / name
    d.mkdir(exist_ok=True)
    (d / "run.py").write_text(template.replace("__SHARD__", str(shard)).replace("__LIMIT__", limit))
    meta = {"id": kid, "title": title, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["tumis001/rsna-knee-hpo-v31-d2-report-labels"], "kernel_sources": [],
            "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("rendered", d)
