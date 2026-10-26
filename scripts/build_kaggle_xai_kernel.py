"""Build the Kaggle kernel for the v4 l4copy XAI + ablations on fold 0 (kaggle/submission_kernels/v4_l4copy_xai_fold0/).
Kaggle has no d96 cache, so the kernel decodes the 825 fold-0 studies from the train DICOMs with the submission's own
builder (sub.build_study, the LB path) and runs scripts/l4copy_xai.py's per-study analyses for every model.
The kernel is one script file: the L4C-D core (kaggle/submission_kernels/v4_l4copy_diag_fold0/run.py) and l4copy_xai.py
are embedded verbatim (base64) and written to /kaggle/working at start, so the code that runs is this repo's code.
Datasets: tumis001/rsna-v4-l4copy-fold0 (submission code, LB weights, MS1 table), tumis001/rsna-d96-fold0-ids,
tumis001/rsna-coatnet-d96-exp1b-fold0 (timm wheel), tumis001/rsna-v4-xai-extra (s2027 + MS1-free weights, case list).
Usage: python3 -I scripts/build_kaggle_xai_kernel.py <repo root> [KEY=VALUE env for the kernel ...]"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]); OUT = ROOT / "kaggle/submission_kernels/v4_l4copy_xai_fold0"; OUT.mkdir(parents=True, exist_ok=True)
ENV = dict(a.split("=", 1) for a in sys.argv[2:])          # fixed env for this kernel version, e.g. XAI_PEAK=1 XAI_LITE=1
emb = {name: base64.b64encode(p.read_bytes()).decode() for name, p in (
    ("l4copy_diag_core.py", ROOT / "kaggle/submission_kernels/v4_l4copy_diag_fold0/run.py"),
    ("l4copy_xai.py", ROOT / "scripts/l4copy_xai.py"))}
code = '''"""v4 l4copy XAI + ablations, fold 0 (no submission). Built by scripts/build_kaggle_xai_kernel.py; the embedded files are
the repo's kaggle/submission_kernels/v4_l4copy_diag_fold0/run.py and scripts/l4copy_xai.py."""
import base64
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

EMB = %s
os.environ.update(%s)
W = Path("/kaggle/working"); CODE = W / "code"; CODE.mkdir(exist_ok=True)
for name, b in EMB.items():
    (CODE / name).write_bytes(base64.b64decode(b))
sys.path.insert(0, str(CODE))


def find(slug):
    return next(p for p in (Path("/kaggle/input/datasets/tumis001") / slug, Path("/kaggle/input") / slug) if p.exists())


EXTRA = find("rsna-v4-xai-extra")
os.environ["XAI_CASES"] = str(EXTRA / "xai_cases.json")
import l4copy_diag_core as run  # noqa: E402
os.environ.setdefault("XAI_MODELS", ",".join([f"s2026={run.ART / run.sub.WEIGHTS['file']}",
                                              f"s2027={EXTRA / 'l4copy_s2027_swa21_25.pt'}",
                                              f"r1self={EXTRA / 'r1_self_s2026_swa21_25.pt'}"]))
import l4copy_xai as xai  # noqa: E402
import torch  # noqa: E402

sub = run.sub


def main():
    t0 = time.monotonic()
    sub.ensure_timm()
    data = sub.competition_dir()
    ids = json.loads((find("rsna-d96-fold0-ids") / "fold0_ids.json").read_text()); assert len(ids) == 825, len(ids)
    if os.environ.get("XAI_N"):
        ids = ids[:int(os.environ["XAI_N"])]
    tabm = run.ms1_table(); assert all(s in tabm for s in ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = xai.load_models(device)
    case_of = xai.case_index(ids)
    by_study = {}
    for row in sub.read_csv(data / "train_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    errors = [0]

    def items(it):
        for sid, vol, mask, err in it:
            errors[0] += err
            yield sid, vol, mask

    with Pool(4) as procs:
        it = procs.imap(sub.build_study, [(s, by_study.get(s, []), data / "train_series") for s in ids])
        R, panels, checks, done = xai.loop(items(it), len(ids), models, tabm, case_of, device, t0)
    xai.save(W, done, models, R, panels, checks, {"source": "kaggle DICOM (sub.build_study)", "decode_errors": errors[0],
             "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
             "seconds": round(time.monotonic() - t0, 1)})


if __name__ == "__main__":
    main()
''' % (json.dumps(emb, indent=0), json.dumps(ENV))
(OUT / "run.py").write_text(code)
meta = {"id": "tumis001/rsna-v4-l4copy-xai-fold0", "title": "RSNA v4 l4copy xai fold0", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["tumis001/rsna-coatnet-d96-exp1b-fold0", "tumis001/rsna-v4-l4copy-fold0",
                            "tumis001/rsna-d96-fold0-ids", "tumis001/rsna-v4-xai-extra"],
        "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("built", OUT / "run.py", len(code), "bytes")
