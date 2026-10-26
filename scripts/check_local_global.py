"""CPU mechanism checks; no downloaded weights, GPU or gold labels."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from local_global_data import NativeStudyCache, WindowBatch, MAX_WINDOWS, prepare_batch
from local_global_model import SliceModel

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source_reuse/cnn_topk_support_source_v1/skanee"
sys.path.insert(0, str(SOURCE))
from build_native_triplet_cache import prepare_slot


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def check_cache(out):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    folder = out / "synthetic_dicoms"; folder.mkdir(parents=True, exist_ok=True)
    for i in range(40):
        meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        path = folder / f"{i:03d}.dcm"
        d = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        d.SOPClassUID = meta.MediaStorageSOPClassUID; d.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        d.Rows = d.Columns = 32; d.SamplesPerPixel = 1; d.PhotometricInterpretation = "MONOCHROME2"
        d.BitsAllocated = d.BitsStored = 16; d.HighBit = 15; d.PixelRepresentation = 0
        d.PixelSpacing = [0.5, 0.5]; d.InstanceNumber = i + 1
        d.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; d.ImagePositionPatient = [0, 0, 4 * i]
        d.ScanOptions = ["FS"]
        d.PixelData = (np.arange(1024).reshape(32, 32) + i * 100).astype("uint16").tobytes()
        d.save_as(path, enforce_file_format=True)
    cfg = json.loads((ROOT / "configs/hpo_v31_d2_resnet50_reference.json").read_text())["cache"]
    cfg = {**cfg, "image_size": 32}
    job = dict(study_id="synthetic", series_id="series", slot=0, plane="Axial", fluid_sensitive="1", folder=str(folder))
    r, images, masks = prepare_slot(job, cfg)
    baseline = load_module("original_builder", SOURCE / "scripts/build_slice_cache.py")
    original_r, original, original_masks = baseline.SlotDataset([job], cfg)[0]
    mapping = np.asarray(r["triplet_offsets"])
    assert r["center_files"] == original_r["files"]
    assert np.array_equal(images[mapping[:, 1]], original)
    assert np.array_equal(masks[mapping[:, 1]], original_masks)
    native = np.asarray(r["native_indices"])[mapping]
    centres = np.asarray(r["center_indices"])
    assert np.array_equal(native, centres[:, None] + [-1, 0, 1])
    assert r["fat_suppression"] == "1" and r["plane"] == "Axial"
    assert np.allclose(r["neighbor_gaps_mm"], 4)
    assert len(images) < 3 * len(centres)
    np.save(out / "slices_000.npy", images); np.save(out / "support_000.npy", masks)
    r.update(row_start=0, part="slices_000.npy", support_part="support_000.npy", directory=str(out))
    cache = NativeStudyCache({"synthetic": [r]})
    labels = {"synthetic": (np.ones(12, dtype=np.float32) * .6, np.ones(12, dtype=np.float32))}
    packed, _, target, weight, _, valid = prepare_batch(cache, labels, ["synthetic"], 5, np.random.default_rng(2026),
        torch.device("cpu"), dict(crop_area_min=1., rotation_degrees=0., translate_fraction=0.,
        intensity_jitter=0., brightness_shift=0., erase_probability=0., erase_fraction=.25))
    assert packed.images.shape == (5, 6, 32, 32) and valid.sum() == 5
    assert packed.valid.sum() == 5 and target.shape == weight.shape == (1, 12)
    return {"center_matches_baseline": True, "native_adjacency_checked": True,
            "window_count": len(centres), "unique_slices": len(images), "duplicated_triplet_slices": 3 * len(centres)}


def check_model(backbone="resnet50"):
    torch.manual_seed(2026); torch.set_num_threads(2)
    model = SliceModel(backbone, False, .4, 12)
    n, studies = 6, 2
    meta = torch.zeros(n, 12); meta[:, :4] = torch.tensor([2, 1, 1, 0])
    meta[:, 4:] = torch.rand(n, 8)
    indices = torch.tensor([0, 1, 2, 48, 49, 50])
    valid = torch.zeros(studies, MAX_WINDOWS, dtype=torch.bool); valid[:, :3] = True
    images = torch.randn(n, 6, 32, 32).contiguous(memory_format=torch.channels_last)
    batch = WindowBatch(images, meta, indices, valid)
    owner = torch.tensor([0, 0, 0, 1, 1, 1])
    model.train(); prediction = model(batch, owner, studies)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, torch.rand(2, 12))
    loss.backward()
    required = ["queries", "projection.weight", "context.self_attn.in_proj_weight", "pool.in_proj_weight",
                "correction.weight", "backbone.conv1.weight", "head.1.weight", "position.weight"]
    params = dict(model.named_parameters())
    gradients = {name: float(params[name].grad.abs().sum()) for name in required}
    assert all(np.isfinite(value) and value > 0 for value in gradients.values()), gradients
    model.eval()
    with torch.no_grad():
        original = model(batch, owner, studies)
        permutation = torch.tensor([2, 0, 1, 5, 3, 4])
        shuffled = WindowBatch(images[permutation], meta[permutation], indices[permutation], valid)
        permuted = model(shuffled, owner[permutation], studies)
        assert torch.allclose(original, permuted, atol=2e-5, rtol=2e-5)
        changed = images.clone(); changed[3:] *= 3
        other = model(WindowBatch(changed, meta, indices, valid), owner, studies)
        assert torch.allclose(original[0], other[0], atol=2e-5, rtol=2e-5)
        # Evaluate the first study alone: padding and neighboring studies cannot
        # enter its attention or top-k pooling.
        alone = model(WindowBatch(images[:3], meta[:3], indices[:3], valid[:1]), owner[:3], 1)
        assert torch.allclose(original[:1], alone, atol=2e-5, rtol=2e-5)
    return {"finite_gradient_paths": gradients, "permutation_invariant": True,
            "eval_batch_isolation": True, "parameters": sum(p.numel() for p in model.parameters())}


def check_recovery(out, backbone="resnet50", patience=8, prepare=None):
    """Interrupt after a committed epoch, then verify optimizer-state recovery."""
    from local_global_runner import prepare_local_global
    source = ROOT / "data/kaggle_reference/hpo_xla_actual/skanee"
    raw = (source / "scripts/train_cnn_baseline.py").read_text()
    trainer_path = out / "trainer_smoke.py"; trainer_path.write_text((prepare or prepare_local_global)(raw))
    from src.reports.schema import TARGETS
    cache_dir = out / "toy_cache"; cache_dir.mkdir(exist_ok=True)
    record, _, _ = prepare_slot(dict(study_id="toy", series_id="series", slot=0, plane="Axial", fluid_sensitive="1",
                                     folder=str(out / "synthetic_dicoms")),
        {**json.loads((ROOT / "configs/hpo_v31_d2_resnet50_reference.json").read_text())["cache"], "image_size": 32})
    record.update(row_start=0, part="slices_000.npy", support_part="support_000.npy")
    import shutil
    for name in ["slices_000.npy", "support_000.npy"]: shutil.copy2(out / name, cache_dir / name)
    records = [{**record, "study_id": f"toy_{i:02d}"} for i in range(14)]
    (cache_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    labels = {"targets_in_order": list(TARGETS), "labels": [dict(study_id=r["study_id"],
        probabilities=[.1 if i % 2 == 0 else .9] * 12, weights=[1.] * 12) for i, r in enumerate(records)]}
    labels_path = out / "toy_labels.json"; labels_path.write_text(json.dumps(labels))
    folds = out / "toy_folds.csv"; folds.write_text("study_id,fold\n" + "".join(f"toy_{i:02d},{1 if i<2 else 0}\n" for i in range(14)))
    config = json.loads((ROOT / "configs/cnn_local_global_v31_d2.json").read_text())
    config.update(epochs=2, batch_studies=2, mixed_precision=False)
    config["model"]["backbone"] = backbone
    config["early_stopping_patience"] = patience
    config["cache"]["image_size"] = 32; config["model"]["pretrained"] = False
    cp = out / "toy_config.json"; cp.write_text(json.dumps(config))
    run_dir = out / "recovery_smoke"; run_dir.mkdir(exist_ok=True)
    # Preserve a prior successful check; remove only this check's known files.
    for path in run_dir.iterdir():
        if path.is_file(): path.unlink()
    argv = ["trainer", "--cache", str(cache_dir), "--soft-labels", str(labels_path), "--folds", str(folds),
            "--config", str(cp), "--output-dir", str(run_dir), "--train-slices-per-study", "24"]
    old_argv = sys.argv
    class SimulatedPreemption(Exception): pass
    try:
        first = load_module("local_global_smoke_first", trainer_path)
        first.STEVEN_AUDIT_LABELS = first.read_soft_labels(labels_path)
        def interrupt():
            if (run_dir / "recovery.pt").is_file():
                raise SimulatedPreemption("after durable epoch boundary")
        first.DURABLE_COMMIT = interrupt; sys.argv = argv
        try: first.main()
        except SimulatedPreemption: pass
        state = torch.load(run_dir / "recovery.pt", weights_only=False)
        assert state["epoch"] == 0 and state["optimizer"]["state"] and state["global_step"] == 1
        resumed = load_module("local_global_smoke_resumed", trainer_path)
        resumed.STEVEN_AUDIT_LABELS = resumed.read_soft_labels(labels_path)
        resumed.DURABLE_COMMIT = lambda: None; sys.argv = argv; resumed.main()
        state = torch.load(run_dir / "recovery.pt", weights_only=False)
        assert state["epoch"] == 1 and state["global_step"] == 2
        assert [r["epoch"] for r in state["history"]] == [0, 1]
        assert (run_dir / "checkpoint_best.pt").is_file()
        return {"resumed_epoch": 1, "optimizer_steps": 2, "history_not_duplicated": True}
    finally: sys.argv = old_argv


if __name__ == "__main__":
    out = ROOT / "data/local_global/verification"; out.mkdir(parents=True, exist_ok=True)
    report = {"cache": check_cache(out), "model": check_model(), "recovery": check_recovery(out), "device": "cpu", "gold_read": False}
    (out / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
