"""Frozen Pillar-0 spatial-support audit using temporary Modal compute/storage.

Example (run with the personal Modal token environment):
    modal run scripts/pillar0_support_feature_audit_modal.py \
      --checkpoint-dir /path/to/pillar0_breast_mri \
      --input-archive /path/to/pillar0_transfer.npz

The checkpoint and the one-study input archive are uploaded to an anonymous
ephemeral Modal Volume. The volume and Modal App are removed on exit. The GPU
function returns aggregate activation metrics and a paired token-gating
counterfactual.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal


EXPECTED_REVISION = "be5050ba482607f90e607e044f03a2ebc38f98c3"
EXPECTED_WEIGHTS_SHA256 = (
    "d658fa8aad744c66d1abf43c2bf6bf594e75e834186dce9aadd1b7a9e4381bf0"
)

app = modal.App("rsna-pillar0-support-feature-audit")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==4.53.2",
        "timm==1.0.15",
        "einops==0.8.1",
        "safetensors==0.6.2",
        "PyYAML==6.0.2",
        "numpy==2.2.6",
    )
)


@app.function(
    image=image,
    gpu="T4",
    memory=32768,
    timeout=900,
)
def run_feature_audit(data_root: str = "/mnt/pillar0") -> dict:
    import gc
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import torch
    from safetensors.torch import load_file

    root = Path(data_root)
    model_dir = root / "bundle"
    archive_path = root / "pillar0_transfer.npz"
    provenance = json.loads((model_dir / "MODEL_PROVENANCE.json").read_text())
    weights_path = model_dir / "model.safetensors"

    digest = hashlib.sha256()
    with weights_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    weights_hash = digest.hexdigest()
    if provenance["source_revision"] != EXPECTED_REVISION:
        raise RuntimeError(f"Unexpected checkpoint revision: {provenance}")
    if weights_hash != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(f"Unexpected checkpoint hash: {weights_hash}")

    if not torch.cuda.is_available():
        raise RuntimeError("Modal T4 did not expose CUDA")
    device = torch.device("cuda:0")
    print(
        f"Checkpoint verified: {EXPECTED_REVISION}; GPU: {torch.cuda.get_device_name(0)}",
        flush=True,
    )

    package_parent = Path("/tmp")
    package_dir = package_parent / "pillar0_local"
    package_dir.mkdir(exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    for source in model_dir.glob("*.py"):
        (package_dir / source.name).write_bytes(source.read_bytes())
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    from pillar0_local.config_clip_mmatlas import CLIPMultimodalAtlasConfig
    from pillar0_local.modeling_clip_mmatlas import CLIPMultimodalAtlas

    config = CLIPMultimodalAtlasConfig.from_pretrained(
        model_dir, local_files_only=True
    )
    with torch.device("cpu"):
        pillar = CLIPMultimodalAtlas(config)
    state = load_file(str(weights_path), device="cpu")
    load_result = torch.nn.Module.load_state_dict(pillar, state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {load_result}")
    del state
    gc.collect()
    pillar.eval().requires_grad_(False).to(device)

    archive = np.load(archive_path, allow_pickle=False)
    volumes = archive["volumes"]
    supports = archive["supports"]
    planes = archive["planes"].astype(str).tolist()
    if volumes.shape != (3, 384, 384, 192):
        raise RuntimeError(f"Unexpected input shape: {volumes.shape}")
    if supports.shape != (3, 64, 32, 16):
        raise RuntimeError(f"Unexpected support grid shape: {supports.shape}")
    if set(planes) != {"Axial", "Coronal", "Sagittal"}:
        raise RuntimeError(f"Unexpected planes: {planes}")

    activation_holder = {}
    gate_state = {"enabled": False, "mask": None}

    def inspect_and_optionally_gate_patch_features(_module, _inputs, output):
        activation_holder["grid"] = output.detach()
        if gate_state["enabled"]:
            return output * gate_state["mask"].to(
                device=output.device, dtype=output.dtype
            )
        return None

    patch_embed = pillar.model.visual.patch_embeds["breast_mr"]
    hook = patch_embed.register_forward_hook(
        inspect_and_optionally_gate_patch_features
    )

    rows = []
    embeddings = []
    for index, plane in enumerate(planes):
        print(f"Frozen inference {index + 1}/3: {plane}", flush=True)
        gray = torch.from_numpy(np.ascontiguousarray(volumes[index]))[
            None, None
        ]
        image_tensor = (
            gray.expand(1, 3, *gray.shape[-3:])
            .contiguous()
            .to(device=device, dtype=torch.float16)
        )
        activation_holder.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            embedding = pillar.extract_vision_feats(
                {"breast_mr": image_tensor}
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        feature_grid = activation_holder.pop("grid")
        if feature_grid.ndim != 5 or feature_grid.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected patch feature shape: {tuple(feature_grid.shape)}"
            )
        spatial_shape = tuple(feature_grid.shape[-3:])
        if spatial_shape != tuple(supports[index].shape):
            raise RuntimeError(
                f"Feature/support grid mismatch: {spatial_shape} vs "
                f"{supports[index].shape}"
            )
        if not torch.isfinite(feature_grid).all() or not torch.isfinite(
            embedding
        ).all():
            raise RuntimeError(f"Non-finite activation or embedding: {plane}")

        energy = feature_grid.float().square().mean(dim=1)[0].cpu().numpy()
        support = supports[index].astype(np.float32)
        groups = {
            "supported_core": support >= 0.95,
            "unsupported_padding": support <= 0.05,
            "mixed_boundary": (support > 0.05) & (support < 0.95),
        }
        total_energy = float(energy.sum()) + 1e-12
        spatial_metrics = {
            "grid_shape": list(spatial_shape),
            "total_cells": int(energy.size),
            "total_mean_squared_activation": float(energy.mean()),
        }
        for name, mask in groups.items():
            count = int(mask.sum())
            selected = energy[mask]
            spatial_metrics[name] = {
                "cells": count,
                "fraction_of_cells": float(count / energy.size),
                "mean_squared_activation": (
                    float(selected.mean()) if count else None
                ),
                "fraction_of_total_feature_energy": (
                    float(selected.sum() / total_energy) if count else None
                ),
            }
        pad_energy = spatial_metrics["unsupported_padding"][
            "mean_squared_activation"
        ]
        core_energy = spatial_metrics["supported_core"][
            "mean_squared_activation"
        ]
        spatial_metrics["padding_to_core_mean_energy_ratio"] = (
            float(pad_energy / core_energy)
            if pad_energy is not None and core_energy
            else None
        )

        vector = embedding.float().cpu().numpy().reshape(-1)
        if vector.shape != (1152,):
            raise RuntimeError(f"Unexpected embedding size: {vector.shape}")
        embeddings.append(vector)
        support_mask = torch.from_numpy(
            (support > 0.05).astype(np.float32)
        )[None, None].to(device)
        gate_state["mask"] = support_mask
        gate_state["enabled"] = True
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            gated_embedding = pillar.extract_vision_feats(
                {"breast_mr": image_tensor}
            )
        gate_state["enabled"] = False
        gated_vector = gated_embedding.float().cpu().numpy().reshape(-1)
        if gated_vector.shape != (1152,) or not np.isfinite(gated_vector).all():
            raise RuntimeError(f"Invalid support-gated embedding: {plane}")
        base_norm = float(np.linalg.norm(vector)) + 1e-12
        gated_norm = float(np.linalg.norm(gated_vector))
        gated_cosine = float(
            np.dot(vector, gated_vector) / (base_norm * (gated_norm + 1e-12))
        )
        rows.append(
            {
                "plane": plane,
                "embedding_dim": int(vector.size),
                "embedding_norm": float(np.linalg.norm(vector)),
                "inference_seconds": float(elapsed),
                "canvas_support_fraction": float(support.mean()),
                "gated_out_token_fraction": float((support <= 0.05).mean()),
                "support_gated_embedding_cosine": gated_cosine,
                "support_gated_embedding_relative_l2_change": float(
                    np.linalg.norm(gated_vector - vector) / base_norm
                ),
                "support_gated_embedding_norm": gated_norm,
                "patch_grid": spatial_metrics,
            }
        )
        print(
            f"  grid={spatial_shape}; support={support.mean():.3f}; "
            f"pad/core energy ratio="
            f"{spatial_metrics['padding_to_core_mean_energy_ratio']}; "
            f"pad energy share="
            f"{spatial_metrics['unsupported_padding']['fraction_of_total_feature_energy']}",
            flush=True,
        )
        del gray, image_tensor, embedding, gated_embedding, feature_grid, support_mask
        gc.collect()
        torch.cuda.empty_cache()
    hook.remove()

    embedding_matrix = np.stack(embeddings)
    normalized = embedding_matrix / np.maximum(
        np.linalg.norm(embedding_matrix, axis=1, keepdims=True), 1e-12
    )
    cosine = normalized @ normalized.T
    pairwise = {}
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            pairwise[f"{planes[i]}__{planes[j]}"] = float(cosine[i, j])

    return {
        "audit": "frozen_pillar0_support_token_gating_counterfactual",
        "source_revision": provenance["source_revision"],
        "weights_sha256": weights_hash,
        "training_performed": False,
        "learning_rate_used": None,
        "followup_reader_learning_rate": 0.0001,
        "reference_reader_learning_rate": 0.0003,
        "labels_or_reports_read": False,
        "source_artifacts": "three identifier-free report-only series from Kaggle",
        "temporary_modal_volume": True,
        "persistent_modal_storage": False,
        "gpu": torch.cuda.get_device_name(0),
        "study_count": 1,
        "series_count": len(planes),
        "measured_layer": "breast_mr patch embedding convolution output before multiscale Atlas stages; paired counterfactual gates unsupported patch tokens",
        "counterfactual": "zero patch-token vectors where support <= 0.05 before Atlas; preserve partially supported boundary tokens",
        "support_groups": "core >= 0.95; padding <= 0.05; mixed boundary otherwise",
        "per_plane": rows,
        "pairwise_embedding_cosine": pairwise,
        "mean_inference_seconds": float(np.mean([r["inference_seconds"] for r in rows])),
        "interpretation_limit": "This one-study frozen counterfactual tests whether unsupported patch tokens influence the final representation. It does not establish label learning, generalization, or validation AUC improvement.",
    }


@app.local_entrypoint()
def main(checkpoint_dir: str, input_archive: str) -> None:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    input_archive = Path(input_archive).resolve()
    if not (checkpoint_dir / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint_dir / "model.safetensors")
    if not input_archive.is_file():
        raise FileNotFoundError(input_archive)

    with modal.Volume.ephemeral() as scratch:
        with scratch.batch_upload() as batch:
            batch.put_directory(str(checkpoint_dir), "/bundle")
            batch.put_file(str(input_archive), "/pillar0_transfer.npz")

        remote = run_feature_audit.with_options(
            volumes={"/mnt/pillar0": scratch.with_mount_options(read_only=True)}
        )
        result = remote.remote()
        print(json.dumps(result, indent=2), flush=True)
