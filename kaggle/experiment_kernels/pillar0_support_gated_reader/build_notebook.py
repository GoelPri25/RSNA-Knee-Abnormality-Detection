"""Build the self-contained Kaggle notebook for the Pillar-0 support test."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


OUT = Path(__file__).with_name("pillar0_support_gated_reader.ipynb")


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
        # Pillar-0 spatial-token reader preparation

        **The training representation will be Pillar-0's spatial tokens.** The model
        creates multiscale, contextual tokens for each 3D series, but its standard
        image output max-pools them into one 1,152-D vector. Our reader should use the
        pre-pooling tokens together with their spatial scale and support information.
        The encoder remains frozen for the first reader comparison.

        The eight-series compression probe has passed. This run extracts the
        quantized spatial-token pyramid for every eligible report-only series.
        Chunks are saved locally and published periodically to a private Kaggle
        dataset, so a runtime disconnect can resume from completed series. This is
        feature extraction, not reader training; the reader will be trained from
        the completed cache in a separate step.

        Bulk extraction of the pooled 1,152-D features is disabled. The next model
        must be built around the pre-pooling spatial tokens measured by this probe.
        The competition images stay on Kaggle.
        """
    ),
    markdown(
        r"""
        ## 1. Run controls and CPU-safe preflight

        This cell checks metadata and run limits before inference. Full mode excludes
        every officially labeled study, limits a Kaggle session to 9,000 new series,
        and resumes from the private cache. The 58-study gold set remains untouched.
        """
    ),
    code(
        r'''
        import gc
        import hashlib
        import json
        import os
        import shutil
        import subprocess
        import sys
        import time
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import pydicom
        import scipy
        import torch
        import torch.nn.functional as F
        from scipy.ndimage import affine_transform

        SEED = 2026
        RUN_MODE = "full_spatial_cache" # Production mode; alternatives: compression_probe, token_probe, pilot.
        PILOT_SERIES = 12
        COMPRESSION_PROBE_SERIES = 8
        MAX_NEW_SERIES_PER_RUN = 9000   # Leave margin under Kaggle's per-run execution cap.
        SPATIAL_CHUNK_SIZE = 64
        CHUNK_SIZE = 128              # A local chunk is written after this many series.
        PUBLISH_EVERY_CHUNKS = 8      # Publish the spatial cache after 512 new series.
        PUBLISH_PRIVATE_CHECKPOINTS = RUN_MODE == "full_spatial_cache"
        CHECKPOINT_DATASET_ID = "tumis001/rsna-pillar0-spatial-token-cache-v1"
        SUPPORT_THRESHOLD = 0.05
        TARGET_SPACING_XYZ_MM = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        TARGET_SHAPE_ZYX = (192, 384, 384)
        # The checkpoint config is H,W,D = 384,384,192. Conv3D consumes
        # D,H,W, so the canonical Z,Y,X volume must be passed without a
        # permutation. Its 6x12x12 patch stride then yields a cubic 32^3 grid.
        PATCH_GRID = (32, 32, 32)
        MODEL_INPUT_KEY = "breast_mr"
        EXPECTED_REVISION = "be5050ba482607f90e607e044f03a2ebc38f98c3"
        EXPECTED_WEIGHTS_SHA256 = "d658fa8aad744c66d1abf43c2bf6bf594e75e834186dce9aadd1b7a9e4381bf0"
        EXPECTED_EMBEDDING_DIM = 1152

        if RUN_MODE not in {"full_spatial_cache", "compression_probe", "token_probe", "pilot"}:
            raise ValueError("Unsupported Pillar-0 extraction mode")

        DATA_CANDIDATES = [
            Path("/kaggle/input/rsna-knee-abnormality-detection"),
            Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        ]
        DATA = next((path for path in DATA_CANDIDATES if path.is_dir()), None)
        if DATA is None:
            raise FileNotFoundError(
                "Attach the rsna-knee-abnormality-detection competition data."
            )

        TRAIN_SERIES_ROOT = DATA / "train_series"
        if not TRAIN_SERIES_ROOT.is_dir():
            raise FileNotFoundError(f"Expected series folders under {TRAIN_SERIES_ROOT}")

        train_series = pd.read_csv(DATA / "train_series.csv", dtype=str)
        required_columns = {
            "StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane",
            "Fluid_Sensitive", "Fat_Suppression",
        }
        missing = required_columns - set(train_series.columns)
        if missing:
            raise ValueError(f"train_series.csv is missing columns: {sorted(missing)}")
        if train_series["SeriesInstanceUID"].duplicated().any():
            raise ValueError("train_series.csv contains duplicate SeriesInstanceUIDs")
        train_series = train_series.sort_values(
            ["StudyInstanceUID", "SeriesInstanceUID"], kind="stable"
        ).reset_index(drop=True)

        if RUN_MODE == "full_spatial_cache":
            official_train = pd.read_csv(DATA / "train.csv", low_memory=False)
            target_columns = [
                column for column in official_train.columns
                if column not in {"StudyInstanceUID", "Report"}
            ]
            if len(target_columns) != 12:
                raise RuntimeError(f"Expected 12 official target columns; found {target_columns}")
            report_only_ids = set(
                official_train.loc[
                    official_train[target_columns].isna().all(axis=1), "StudyInstanceUID"
                ].astype(str)
            )
            if not 4000 <= len(report_only_ids) <= 4350:
                raise RuntimeError(f"Unexpected report-only study count: {len(report_only_ids)}")
            train_series = train_series[
                train_series["StudyInstanceUID"].astype(str).isin(report_only_ids)
            ].reset_index(drop=True)
            if train_series.empty:
                raise RuntimeError("No report-only series remain after the gold-lock filter")
            print(
                f"Gold lock: excluded {len(official_train) - len(report_only_ids)} studies; "
                f"processing {len(report_only_ids)} report-only studies and "
                f"{len(train_series)} series.", flush=True
            )

        CHECKPOINT_ROOT = Path("/kaggle/working/pillar0_support_gated_reader")
        CHUNK_ROOT = CHECKPOINT_ROOT / "chunks"
        UPLOAD_ROOT = Path("/kaggle/working/pillar0_private_checkpoint_upload")
        SPATIAL_CHUNK_ROOT = (
            UPLOAD_ROOT
            if RUN_MODE == "full_spatial_cache"
            else CHECKPOINT_ROOT / "spatial_token_chunks"
        )
        CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
        SPATIAL_CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
        print("Data:", DATA)
        print("Training series rows:", len(train_series))
        print("pydicom:", pydicom.__version__, "SciPy:", scipy.__version__)
        print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        print("Run mode:", RUN_MODE, "| support threshold:", SUPPORT_THRESHOLD)
        '''
    ),
    markdown(
        r"""
        ## 2. Verify the offline Pillar-0 bundle

        Kaggle mounts the private `pillar0-breastmri-offline-bundle` notebook source
        under `/kaggle/input/pillar0-breastmri-offline-bundle`. This avoids downloading
        model weights from Hugging Face during the run.
        """
    ),
    code(
        r'''
        def sha256_file(path, block_size=1024 * 1024):
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(block_size), b""):
                    digest.update(block)
            return digest.hexdigest()

        bundle_roots = [
            Path("/kaggle/input/pillar0-breastmri-offline-bundle/pillar0_breast_mri"),
            *Path("/kaggle/input").glob("*/pillar0_breast_mri"),
        ]
        bundle_candidates = [
            path for path in dict.fromkeys(bundle_roots)
            if (path / "model.safetensors").is_file()
            and (path / "config.json").is_file()
            and (path / "MODEL_PROVENANCE.json").is_file()
            and (path / "inventory.csv").is_file()
        ]
        if len(bundle_candidates) != 1:
            raise RuntimeError(
                "Attach exactly one verified Pillar-0 offline bundle; found: "
                f"{bundle_candidates}"
            )
        MODEL_DIR = bundle_candidates[0]
        provenance = json.loads((MODEL_DIR / "MODEL_PROVENANCE.json").read_text())
        weights_hash = sha256_file(MODEL_DIR / "model.safetensors")
        if provenance.get("source_revision") != EXPECTED_REVISION:
            raise RuntimeError(f"Unexpected checkpoint revision: {provenance}")
        if weights_hash != EXPECTED_WEIGHTS_SHA256:
            raise RuntimeError(f"Pillar-0 weights hash mismatch: {weights_hash}")
        if not (MODEL_DIR / "LICENSE").exists():
            raise FileNotFoundError("Checkpoint bundle is missing its LICENSE file")

        run_config = {
            "experiment": "pillar0_multiscale_spatial_token_cache",
            "run_mode": RUN_MODE,
            "schema_version": 1,
            "checkpoint_revision": EXPECTED_REVISION,
            "weights_sha256": weights_hash,
            "support_threshold": SUPPORT_THRESHOLD,
            "operation": "capture baseline contextual tokens and preserve support for downstream reader masking",
            "encoder_token_gating_applied": False,
            "support_map_saved": True,
            "target_spacing_xyz_mm": TARGET_SPACING_XYZ_MM.tolist(),
            "target_shape_zyx": list(TARGET_SHAPE_ZYX),
            "patch_grid": list(PATCH_GRID),
            "embedding_dim": EXPECTED_EMBEDDING_DIM,
            "spatial_cache": {
                "fine_scale": "32^3 average-pooled to 8^3",
                "middle_scale": "native 8^3",
                "coarse_scale": "native 2^3",
                "channels_per_scale": 384,
                "quantization": "symmetric per-series per-scale per-channel int8",
                "chunk_series": SPATIAL_CHUNK_SIZE,
            },
            "seed": SEED,
        }
        config_path = CHECKPOINT_ROOT / "run_config.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text())
            for key in ("schema_version", "checkpoint_revision", "weights_sha256",
                        "support_threshold", "operation", "encoder_token_gating_applied",
                        "support_map_saved", "target_spacing_xyz_mm", "target_shape_zyx",
                        "patch_grid", "spatial_cache"):
                if previous.get(key) != run_config.get(key):
                    raise RuntimeError(f"Resume config mismatch for {key}: {previous.get(key)}")
        config_path.write_text(json.dumps(run_config, indent=2))

        print("Verified bundle:", MODEL_DIR)
        print("Checkpoint revision:", EXPECTED_REVISION)
        print("Weights SHA256:", weights_hash)
        print("Offline bundle verification passed.")
        '''
    ),
    markdown(
        r"""
        ## 3. Checkpoint publishing

        Full mode resumes from the private cache attached as a Kaggle input. Kaggle
        API credentials are checked before extraction so interval checkpoint uploads
        are durable. Probe modes do not require credentials.
        """
    ),
    code(
        r'''
        def configure_kaggle_api():
            from kaggle_secrets import UserSecretsClient

            secrets = UserSecretsClient()

            def get_secret_with_retry(name, attempts=3):
                last_error = None
                for attempt in range(attempts):
                    try:
                        value = secrets.get_secret(name)
                        if value:
                            return value
                    except Exception as error:
                        last_error = error
                        if attempt + 1 < attempts:
                            delay = 2 ** (attempt + 1)
                            print(
                                f"Kaggle Secrets temporarily unavailable for {name}; "
                                f"retrying in {delay}s ({attempt + 1}/{attempts - 1}).",
                                flush=True,
                            )
                            time.sleep(delay)
                if last_error is not None:
                    raise RuntimeError(
                        f"Could not read Kaggle Secret {name} after {attempts} attempts."
                    ) from last_error
                return None

            token = None
            try:
                token = get_secret_with_retry("KAGGLE_API_TOKEN")
            except RuntimeError:
                pass
            if token:
                os.environ["KAGGLE_API_TOKEN"] = token
            else:
                try:
                    os.environ["KAGGLE_USERNAME"] = get_secret_with_retry("KAGGLE_USERNAME")
                    os.environ["KAGGLE_KEY"] = get_secret_with_retry("KAGGLE_KEY")
                except Exception as error:
                    raise RuntimeError(
                        "Add Kaggle Secrets KAGGLE_API_TOKEN, or KAGGLE_USERNAME and "
                        "KAGGLE_KEY, before enabling durable full-run checkpoints."
                    ) from error
            auth_check = kaggle_cli(["datasets", "list", "--mine", "--csv"], check=False)
            if auth_check.returncode:
                raise RuntimeError(
                    "Kaggle API authentication check failed before processing. "
                    "Verify the Kaggle Secrets and notebook internet setting."
                )

        def kaggle_cli(args, check=True):
            result = subprocess.run(
                ["kaggle", *args], text=True, capture_output=True, check=False
            )
            if check and result.returncode:
                # Never print environment variables or credential material.
                raise RuntimeError(
                    f"Kaggle CLI failed ({result.returncode}): "
                    f"{result.stderr[-1200:]}"
                )
            return result

        def download_previous_checkpoint_dataset():
            if not PUBLISH_PRIVATE_CHECKPOINTS:
                return False
            # Mount the prior private cache as an input dataset. This avoids using
            # Kaggle's API download path for the multi-gigabyte checkpoint archive.
            checkpoint_input_candidates = [
                Path("/kaggle/input/rsna-pillar0-spatial-token-cache-v1"),
            ]
            checkpoint_input = next(
                (path for path in checkpoint_input_candidates
                 if (path / "run_config.json").is_file()),
                None,
            )
            if checkpoint_input is not None:
                UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
                for path in checkpoint_input.iterdir():
                    if path.is_file():
                        target = UPLOAD_ROOT / path.name
                        if not target.exists() or target.stat().st_size != path.stat().st_size:
                            shutil.copy2(path, target)
                print("Mounted prior private checkpoint as Kaggle input; local resume is ready.", flush=True)
            else:
                configure_kaggle_api()
                result = kaggle_cli(
                    ["datasets", "download", "-d", CHECKPOINT_DATASET_ID,
                     "-p", str(UPLOAD_ROOT), "--unzip"],
                    check=False,
                )
                if result.returncode:
                    print("No previous private feature dataset found; first checkpoint will create it.")
                    return False
                for path in UPLOAD_ROOT.glob("*.zip"):
                    path.unlink(missing_ok=True)
            remote_config = UPLOAD_ROOT / "run_config.json"
            if remote_config.exists():
                remote = json.loads(remote_config.read_text())
                if (remote.get("checkpoint_revision") != EXPECTED_REVISION
                        or remote.get("weights_sha256") != EXPECTED_WEIGHTS_SHA256
                        or remote.get("support_threshold") != SUPPORT_THRESHOLD):
                    raise RuntimeError("Private Kaggle checkpoint belongs to a different run config")
            # Uploading new interval checkpoints still requires an authenticated API.
            configure_kaggle_api()
            return True

        REMOTE_DATASET_EXISTS = download_previous_checkpoint_dataset()
        if PUBLISH_PRIVATE_CHECKPOINTS:
            print("Interval publishing enabled for private dataset:", CHECKPOINT_DATASET_ID)
        else:
            print("Interval publishing disabled for this probe/pilot run.")
        '''
    ),
    markdown(
        r"""
        ## 4. Reconstruct a series and preserve valid support

        DICOM padding is represented as `NaN`; a genuine pixel value of zero remains
        valid. Signal and support are resampled separately on the physical LPS grid,
        intensity percentiles use supported voxels only, and the support map is carried
        through centre padding to the encoder's patch grid.
        """
    ),
    code(
        r'''
        from itertools import product

        class GeometryError(ValueError):
            pass

        def reconstruct_series(study_uid, series_uid):
            folder = TRAIN_SERIES_ROOT / str(study_uid) / str(series_uid)
            paths = sorted(folder.glob("*.dcm"))
            if len(paths) < 2:
                raise GeometryError(f"Expected >=2 slices in {folder}; found {len(paths)}")
            headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in paths]
            required = ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
                        "Rows", "Columns", "SOPInstanceUID")
            for header in headers:
                if any(not hasattr(header, field) for field in required):
                    raise GeometryError(f"Missing DICOM geometry in {folder}")
                if (str(header.StudyInstanceUID) != str(study_uid)
                        or str(header.SeriesInstanceUID) != str(series_uid)):
                    raise GeometryError(f"DICOM UID does not match folder: {folder}")
                if getattr(header, "Modality", "") != "MR" or int(getattr(header, "NumberOfFrames", 1)) != 1:
                    raise GeometryError("Only classic single-frame MR is supported")
            iop = np.asarray(headers[0].ImageOrientationPatient, dtype=np.float64)
            spacing = np.asarray(headers[0].PixelSpacing, dtype=np.float64)
            if not np.isfinite(iop).all() or not np.isfinite(spacing).all() or (spacing <= 0).any():
                raise GeometryError("Nonfinite orientation or invalid pixel spacing")
            col_dir, row_dir = iop[:3], iop[3:]
            if not np.allclose([np.linalg.norm(col_dir), np.linalg.norm(row_dir),
                                np.dot(col_dir, row_dir)], [1, 1, 0], atol=1e-4):
                raise GeometryError("Orientation cosines are not orthonormal")
            if any(not np.allclose(h.ImageOrientationPatient, iop, atol=1e-4, rtol=0)
                   or not np.allclose(h.PixelSpacing, spacing, atol=1e-4, rtol=0)
                   or (int(h.Rows), int(h.Columns)) != (int(headers[0].Rows), int(headers[0].Columns))
                   for h in headers):
                raise GeometryError("Orientation, spacing, or matrix varies in this series")
            if len({str(h.SOPInstanceUID) for h in headers}) != len(headers):
                raise GeometryError("Duplicate SOPInstanceUID")
            for field in ("EchoTime", "TemporalPositionIdentifier", "AcquisitionNumber"):
                if len({str(getattr(h, field, "")) for h in headers}) > 1:
                    raise GeometryError(f"Mixed acquisition: {field}")
            positions = np.asarray([h.ImagePositionPatient for h in headers], dtype=np.float64)
            normal = np.cross(col_dir, row_dir)
            order = np.argsort(positions @ normal, kind="stable")
            gaps = np.diff((positions @ normal)[order])
            if len(gaps) == 0 or (gaps < 1e-3).any():
                raise GeometryError("Repeated or indistinguishable physical slice positions")
            dz = float(np.median(gaps))
            if (not np.allclose(gaps, dz, atol=0.05, rtol=0.01)
                    or not np.allclose(np.diff(positions[order], axis=0), normal * dz,
                                       atol=0.05, rtol=0.01)):
                raise GeometryError("Irregular spacing or sheared volume")
            affine = np.eye(4, dtype=np.float64)
            affine[:3, 0] = normal * dz
            affine[:3, 1] = row_dir * spacing[0]
            affine[:3, 2] = col_dir * spacing[1]
            affine[:3, 3] = positions[order][0]
            slices = []
            for index in order:
                dcm = pydicom.dcmread(paths[index])
                if int(getattr(dcm, "SamplesPerPixel", 1)) != 1:
                    raise GeometryError("Expected grayscale MRI")
                pixels = dcm.pixel_array.astype(np.float32)
                if hasattr(dcm, "PixelPaddingValue"):
                    low = min(float(dcm.PixelPaddingValue),
                              float(getattr(dcm, "PixelPaddingRangeLimit", dcm.PixelPaddingValue)))
                    high = max(float(dcm.PixelPaddingValue),
                               float(getattr(dcm, "PixelPaddingRangeLimit", dcm.PixelPaddingValue)))
                    pixels[(pixels >= low) & (pixels <= high)] = np.nan
                pixels = pixels * float(getattr(dcm, "RescaleSlope", 1.0))
                pixels += float(getattr(dcm, "RescaleIntercept", 0.0))
                if not np.isfinite(pixels).any():
                    raise GeometryError(f"Slice has no valid pixels: {paths[index].name}")
                slices.append(pixels)
            volume = np.stack(slices).astype(np.float32)
            return volume, {
                "affine_array_to_lps": affine,
                "spacing_drc_mm": [dz, float(spacing[0]), float(spacing[1])],
                "native_shape_drc": list(volume.shape),
            }

        def canonical_resample_lps(volume, affine, target_spacing_xyz=TARGET_SPACING_XYZ_MM):
            shape = np.asarray(volume.shape, dtype=int)
            validity = np.isfinite(volume).astype(np.float32)
            signal = np.where(validity > 0, np.nan_to_num(volume), 0.0).astype(np.float32)
            corners = np.asarray(list(product(*[(0, int(size - 1)) for size in shape])), dtype=np.float64)
            corners_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
            lps = (affine @ corners_h.T).T[:, :3]
            lps_min, lps_max = lps.min(axis=0), lps.max(axis=0)
            extent = lps_max - lps_min
            shape_xyz = np.ceil(extent / target_spacing_xyz).astype(int) + 1
            out_shape = tuple(shape_xyz[::-1])
            canonical_affine = np.eye(4, dtype=np.float64)
            canonical_affine[:3, 0] = [0, 0, target_spacing_xyz[2]]
            canonical_affine[:3, 1] = [0, target_spacing_xyz[1], 0]
            canonical_affine[:3, 2] = [target_spacing_xyz[0], 0, 0]
            canonical_affine[:3, 3] = lps_min
            mapping = np.linalg.inv(affine) @ canonical_affine
            transform = dict(matrix=mapping[:3, :3], offset=mapping[:3, 3],
                             output_shape=out_shape, order=1, mode="constant", cval=0.0,
                             prefilter=False)
            numerator = affine_transform(signal, **transform).astype(np.float32)
            support = np.clip(affine_transform(validity, **transform).astype(np.float32), 0, 1)
            canonical = np.full(out_shape, np.nan, dtype=np.float32)
            good = support > 1e-6
            canonical[good] = numerator[good] / support[good]
            return canonical, support

        def supported_window(volume, support, lower=20, upper=99, threshold=0.5):
            tensor = torch.from_numpy(volume)
            support_tensor = torch.from_numpy(support)
            core = torch.isfinite(tensor) & (support_tensor >= threshold)
            values = tensor[core]
            if values.numel() == 0:
                raise GeometryError("No supported voxels for intensity window")
            # torch.quantile has a backend size limit. Large scans can exceed it,
            # so estimate the percentiles from a deterministic, evenly spaced
            # sample of at most one million supported voxels.
            max_quantile_samples = 1_000_000
            if values.numel() > max_quantile_samples:
                sample_indices = torch.linspace(
                    0, values.numel() - 1, steps=max_quantile_samples,
                    dtype=torch.float64,
                ).round().to(torch.long)
                values = values.index_select(0, sample_indices)
            lo = torch.quantile(values, lower / 100)
            hi = torch.quantile(values, upper / 100)
            if not torch.isfinite(lo) or not torch.isfinite(hi) or hi <= lo:
                raise GeometryError("Degenerate supported intensity window")
            safe = torch.where(torch.isfinite(tensor), tensor, lo)
            normalized = ((safe - lo) / (hi - lo)).clamp(0, 1)
            normalized = torch.where(support_tensor > 1e-6, normalized, 0.0)
            return normalized, float(lo), float(hi)

        def center_pad_or_crop(tensor, target_shape):
            output = torch.zeros(tuple(target_shape), dtype=tensor.dtype)
            source_slices, destination_slices = [], []
            for current, target in zip(tensor.shape, target_shape):
                if current <= target:
                    start = (target - current) // 2
                    source_slices.append(slice(0, current))
                    destination_slices.append(slice(start, start + current))
                else:
                    start = (current - target) // 2
                    source_slices.append(slice(start, start + target))
                    destination_slices.append(slice(0, target))
            output[tuple(destination_slices)] = tensor[tuple(source_slices)]
            return output

        def prepare_model_input(study_uid, series_uid):
            native, geometry = reconstruct_series(study_uid, series_uid)
            canonical, support = canonical_resample_lps(native, geometry["affine_array_to_lps"])
            windowed, lo, hi = supported_window(canonical, support)
            padded = center_pad_or_crop(windowed, TARGET_SHAPE_ZYX)
            support_padded = center_pad_or_crop(torch.from_numpy(support), TARGET_SHAPE_ZYX).clamp(0, 1)
            # canonical volume is Z,Y,X and torch Conv3D expects D,H,W.
            # Keep that order so actual patch tokens align with the checkpoint's
            # configured H,W,D image_size and Atlas grid (32 x 32 x 32).
            model_dhw = padded.contiguous()
            support_dhw = support_padded.contiguous()
            support_grid = F.adaptive_avg_pool3d(
                support_dhw[None, None].float(), PATCH_GRID
            ).to(torch.float16)
            model_input = model_dhw[None, None].expand(1, 3, *model_dhw.shape).contiguous()
            if tuple(model_input.shape) != (1, 3, 192, 384, 384):
                raise RuntimeError(f"Unexpected model input shape: {tuple(model_input.shape)}")
            return model_input, support_grid, geometry, {
                "intensity_low": lo, "intensity_high": hi,
                "native_finite_fraction": float(np.isfinite(native).mean()),
                "canvas_support_fraction": float(support_grid.float().mean()),
            }

        print("Support-aware DICOM and resampling functions defined.")
        '''
    ),
    markdown(
        r"""
        ## 5. Load the frozen encoder and check its outputs

        Inference requires a Kaggle GPU. The checkpoint is built on CPU and loaded
        strictly from the attached offline bundle, then moved to the selected GPU.
        """
    ),
    code(
        r'''
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a Kaggle GPU, restart the notebook, and rerun from the top.")
        from safetensors.torch import load_file

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        PACKAGE_PARENT = Path("/kaggle/working")
        PACKAGE_DIR = PACKAGE_PARENT / "pillar0_local"
        PACKAGE_DIR.mkdir(exist_ok=True)
        (PACKAGE_DIR / "__init__.py").write_text("", encoding="utf-8")
        for source_file in MODEL_DIR.glob("*.py"):
            shutil.copy2(source_file, PACKAGE_DIR / source_file.name)
        if str(PACKAGE_PARENT) not in sys.path:
            sys.path.insert(0, str(PACKAGE_PARENT))
        from pillar0_local.config_clip_mmatlas import CLIPMultimodalAtlasConfig
        from pillar0_local.modeling_clip_mmatlas import CLIPMultimodalAtlas

        config = CLIPMultimodalAtlasConfig.from_pretrained(MODEL_DIR, local_files_only=True)
        with torch.device("cpu"):
            pillar = CLIPMultimodalAtlas(config)
        state_dict = load_file(str(MODEL_DIR / "model.safetensors"), device="cpu")
        load_result = torch.nn.Module.load_state_dict(pillar, state_dict, strict=True)
        del state_dict
        gc.collect()
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(f"Checkpoint state mismatch: {load_result}")
        device = torch.device("cuda:0")
        pillar.eval().requires_grad_(False).to(device)
        if any(parameter.requires_grad for parameter in pillar.parameters()):
            raise RuntimeError("Pillar-0 must remain frozen")

        patch_module = pillar.model.visual.patch_embeds[MODEL_INPUT_KEY]
        gate_state = {"enabled": False, "mask": None}
        def gate_patch_tokens(_module, _inputs, output):
            if not gate_state["enabled"]:
                return None
            if output.ndim != 5 or tuple(output.shape[-3:]) != PATCH_GRID:
                raise RuntimeError(f"Unexpected patch grid: {tuple(output.shape)}")
            return output * gate_state["mask"].to(device=output.device, dtype=output.dtype)

        hook_handle = patch_module.register_forward_hook(gate_patch_tokens)
        visual = pillar.model.visual
        modal_cfg = visual.model_config["modalities"][MODEL_INPUT_KEY]
        spatial_layout = visual.prepare_multiscale_layout(
            img_size=modal_cfg["image_size"],
            merge_ratio=modal_cfg["merge_ratio"],
            local2global=modal_cfg["local2global"],
            patch_size=modal_cfg["patch_size"],
        )
        token_probe_state = {"enabled": False, "tokens": []}

        def capture_prepool_tokens(_module, inputs):
            if token_probe_state["enabled"]:
                pooled_input = inputs[0]
                if pooled_input.ndim != 3:
                    raise RuntimeError(f"Unexpected prepool tensor: {tuple(pooled_input.shape)}")
                token_probe_state["tokens"].append(
                    pooled_input.detach().transpose(1, 2).to(device="cpu", dtype=torch.float16)
                )

        token_hook_handle = visual.maxpool.register_forward_pre_hook(capture_prepool_tokens)
        print("Frozen model ready on", torch.cuda.get_device_name(0))
        print("Parameters:", sum(parameter.numel() for parameter in pillar.parameters()))
        print("Trainable parameters:", sum(parameter.numel() for parameter in pillar.parameters() if parameter.requires_grad))
        print("Patch grid:", PATCH_GRID, "| output dimension:", EXPECTED_EMBEDDING_DIM)
        print("Atlas spatial scales:", len(spatial_layout))
        '''
    ),
    markdown(
        r"""
        ## 6. Pilot or resumable feature extraction

        Full mode captures three spatial scales before global pooling, pools the fine
        32³ grid to 8³, quantizes the three scales per channel, and saves batches of
        64 series atomically. Every eight batches are published to the private Kaggle
        cache. Probe modes retain their prior pooled-feature behavior.
        """
    ),
    code(
        r'''
        def existing_processed_ids():
            result = set()
            for path in sorted(CHUNK_ROOT.glob("features_*.npz")):
                with np.load(path, allow_pickle=False) as archive:
                    result.update(zip(archive["study_uid"].astype(str),
                                      archive["series_uid"].astype(str)))
            if RUN_MODE == "compression_probe":
                for path in sorted(SPATIAL_CHUNK_ROOT.glob("spatial_*.npz")):
                    with np.load(path, allow_pickle=False) as archive:
                        result.add((str(archive["study_uid"].item()),
                                    str(archive["series_uid"].item())))
            if RUN_MODE == "full_spatial_cache":
                for path in sorted(SPATIAL_CHUNK_ROOT.glob("spatial_batch_*.npz")):
                    with np.load(path, allow_pickle=False) as archive:
                        result.update(zip(archive["study_uid"].astype(str),
                                          archive["series_uid"].astype(str)))
            return result

        def upload_private_checkpoint(remote_exists):
            UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
            if RUN_MODE != "full_spatial_cache":
                for old in UPLOAD_ROOT.glob("features_*.npz"):
                    old.unlink()
                for chunk in CHUNK_ROOT.glob("features_*.npz"):
                    shutil.copy2(chunk, UPLOAD_ROOT / chunk.name)
            (UPLOAD_ROOT / "run_config.json").write_text(json.dumps(run_config, indent=2))
            metadata = {
                "title": "RSNA Pillar-0 Spatial Token Cache",
                "id": CHECKPOINT_DATASET_ID,
                "subtitle": "Private resumable report-only spatial token cache",
                "description": (
                    "Private derived multiscale Pillar-0 spatial tokens for report-only RSNA "
                    "studies. Contains no DICOM pixels and no labels. Source data remains on "
                    "Kaggle. Chunk identifiers, geometry, checkpoint provenance, and quantization "
                    "settings are recorded in run_config.json."
                ),
                "licenses": [{"name": "other"}],
            }
            (UPLOAD_ROOT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
            command = ["datasets", "version", "-p", str(UPLOAD_ROOT),
                       "-m", f"Pillar-0 feature checkpoint {int(time.time())}",
                       "--dir-mode", "skip", "--keep-tabular"]
            if not remote_exists:
                command = ["datasets", "create", "-p", str(UPLOAD_ROOT),
                           "--dir-mode", "skip", "--keep-tabular"]
            result = kaggle_cli(command, check=False)
            if result.returncode and not remote_exists:
                # If a prior run created it just before disconnecting, update it.
                retry = kaggle_cli(
                    ["datasets", "version", "-p", str(UPLOAD_ROOT),
                     "-m", f"Pillar-0 feature checkpoint {int(time.time())}",
                     "--dir-mode", "skip", "--keep-tabular"],
                    check=False,
                )
                if retry.returncode:
                    raise RuntimeError(
                        "Private Kaggle checkpoint upload failed: "
                        + retry.stderr[-1200:]
                    )
            elif result.returncode:
                raise RuntimeError(
                    "Private Kaggle checkpoint version failed: "
                    + result.stderr[-1200:]
                )
            print("Private Kaggle checkpoint published:", CHECKPOINT_DATASET_ID, flush=True)
            return True

        processed = existing_processed_ids()
        train_series["_key"] = list(zip(train_series.StudyInstanceUID, train_series.SeriesInstanceUID))
        pending = train_series[~train_series["_key"].isin(processed)].drop(columns="_key")
        if RUN_MODE == "pilot":
            sample_indices = np.linspace(
                0, len(pending) - 1, min(PILOT_SERIES, len(pending))
            ).round().astype(int)
            pending = pending.iloc[sample_indices]
        elif RUN_MODE == "token_probe":
            pending = pending.iloc[[len(pending) // 2]]
        elif RUN_MODE == "compression_probe":
            sample_indices = np.linspace(
                0, len(pending) - 1, min(COMPRESSION_PROBE_SERIES, len(pending))
            ).round().astype(int)
            pending = pending.iloc[sample_indices]
        elif RUN_MODE == "full_spatial_cache":
            pending = pending.iloc[:MAX_NEW_SERIES_PER_RUN]
        if pending.empty:
            print("All eligible series are already present in the resumable cache.", flush=True)
        print(f"Already complete: {len(processed):,} | this run: {len(pending):,} | total: {len(train_series):,}", flush=True)

        rows_buffer = []
        baseline_buffer = []
        gated_buffer = []
        run_start = time.perf_counter()
        series_times = []
        chunk_number = 1 + max(
            [int(path.stem.split("_")[-1]) for path in CHUNK_ROOT.glob("features_*.npz")]
            or [0]
        )
        chunk_count = 0
        newly_processed = 0
        compressed_token_records = []
        spatial_cache_buffer = []
        spatial_chunk_number = 1 + max(
            [int(path.stem.split("_")[-1]) for path in SPATIAL_CHUNK_ROOT.glob("spatial_batch_*.npz")]
            or [0]
        )
        spatial_chunks_published_since_start = 0

        def save_spatial_batch():
            global spatial_chunk_number, spatial_chunks_published_since_start
            if not spatial_cache_buffer:
                return
            records = spatial_cache_buffer
            final_path = SPATIAL_CHUNK_ROOT / f"spatial_batch_{spatial_chunk_number:05d}.npz"
            temp_path = final_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temp_path,
                study_uid=np.asarray([r["study_uid"] for r in records], dtype=str),
                series_uid=np.asarray([r["series_uid"] for r in records], dtype=str),
                plane=np.asarray([r["plane"] for r in records], dtype=str),
                fluid_sensitive=np.asarray([r["fluid_sensitive"] for r in records], dtype=str),
                fat_suppression=np.asarray([r["fat_suppression"] for r in records], dtype=str),
                native_slice_count=np.asarray([r["native_slice_count"] for r in records], dtype=np.int32),
                native_finite_fraction=np.asarray([r["native_finite_fraction"] for r in records], dtype=np.float32),
                canvas_support_fraction=np.asarray([r["canvas_support_fraction"] for r in records], dtype=np.float32),
                fine8=np.stack([r["fine8"] for r in records]).astype(np.int8),
                native8=np.stack([r["native8"] for r in records]).astype(np.int8),
                native2=np.stack([r["native2"] for r in records]).astype(np.int8),
                channel_scales=np.stack([r["channel_scales"] for r in records]).astype(np.float32),
                support8=np.stack([r["support8"] for r in records]).astype(np.uint8),
                grid_shapes=np.asarray([[8, 8, 8], [8, 8, 8], [2, 2, 2]], dtype=np.int16),
                checkpoint_revision=np.asarray(EXPECTED_REVISION),
                weights_sha256=np.asarray(EXPECTED_WEIGHTS_SHA256),
            )
            temp_path.replace(final_path)
            with np.load(final_path, allow_pickle=False) as saved:
                expected = {
                    "fine8": (len(records), 512, 384),
                    "native8": (len(records), 512, 384),
                    "native2": (len(records), 8, 384),
                    "channel_scales": (len(records), 3, 384),
                    "support8": (len(records), 512),
                }
                for key, shape in expected.items():
                    if saved[key].shape != shape:
                        raise RuntimeError(f"Saved {key} shape {saved[key].shape}; expected {shape}")
            print(f"Saved durable spatial batch {final_path.name}: {len(records)} series, "
                  f"{final_path.stat().st_size / 1024:.1f} KiB", flush=True)
            spatial_cache_buffer.clear()
            spatial_chunk_number += 1
            spatial_chunks_published_since_start += 1
            if spatial_chunks_published_since_start % PUBLISH_EVERY_CHUNKS == 0:
                global REMOTE_DATASET_EXISTS
                REMOTE_DATASET_EXISTS = upload_private_checkpoint(REMOTE_DATASET_EXISTS)

        def save_chunk():
            global chunk_number, chunk_count
            if not rows_buffer:
                return
            frame = pd.DataFrame(rows_buffer)
            base = np.stack(baseline_buffer).astype(np.float16)
            gated = np.stack(gated_buffer).astype(np.float16)
            final_path = CHUNK_ROOT / f"features_{chunk_number:05d}.npz"
            temp_path = final_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temp_path,
                study_uid=frame["study_uid"].to_numpy(dtype=str),
                series_uid=frame["series_uid"].to_numpy(dtype=str),
                plane=frame["plane"].to_numpy(dtype=str),
                fluid_sensitive=frame["fluid_sensitive"].to_numpy(dtype=str),
                fat_suppression=frame["fat_suppression"].to_numpy(dtype=str),
                native_slice_count=frame["native_slice_count"].to_numpy(dtype=np.int32),
                native_finite_fraction=frame["native_finite_fraction"].to_numpy(dtype=np.float32),
                canvas_support_fraction=frame["canvas_support_fraction"].to_numpy(dtype=np.float32),
                baseline=base,
                support_gated=gated,
                support_threshold=np.asarray(SUPPORT_THRESHOLD, dtype=np.float32),
                checkpoint_revision=np.asarray(EXPECTED_REVISION),
            )
            temp_path.replace(final_path)
            print(f"Saved local checkpoint: {final_path.name} ({len(frame)} series)", flush=True)
            rows_buffer.clear()
            baseline_buffer.clear()
            gated_buffer.clear()
            chunk_number += 1
            chunk_count += 1
            if PUBLISH_PRIVATE_CHECKPOINTS and chunk_count % PUBLISH_EVERY_CHUNKS == 0:
                global REMOTE_DATASET_EXISTS
                REMOTE_DATASET_EXISTS = upload_private_checkpoint(REMOTE_DATASET_EXISTS)

        for position, row in enumerate(pending.itertuples(index=False), start=1):
            series_start = time.perf_counter()
            print(f"Series {position}/{len(pending)} | study={row.StudyInstanceUID} | series={row.SeriesInstanceUID}", flush=True)
            model_input, support_grid, geometry, prep = prepare_model_input(
                row.StudyInstanceUID, row.SeriesInstanceUID
            )
            gpu_input = model_input.to(device=device, dtype=torch.float16, non_blocking=True)
            gate_state["mask"] = (support_grid.to(device) > SUPPORT_THRESHOLD).to(torch.float16)
            gate_state["enabled"] = False
            token_probe_state["tokens"].clear()
            token_probe_state["enabled"] = (
                RUN_MODE in {"compression_probe", "full_spatial_cache"}
                or (RUN_MODE == "token_probe" and position == 1)
            )
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                baseline_embedding = pillar.extract_vision_feats({MODEL_INPUT_KEY: gpu_input})
            token_probe_state["enabled"] = False
            if RUN_MODE in {"token_probe", "compression_probe", "full_spatial_cache"} and (
                RUN_MODE in {"compression_probe", "full_spatial_cache"} or position == 1
            ):
                captured = token_probe_state["tokens"]
                if len(captured) != len(spatial_layout):
                    raise RuntimeError(
                        f"Captured {len(captured)} scales, expected {len(spatial_layout)}"
                    )
                pooled_parts = []
                scale_summary = []
                for scale_index, (tokens, layout) in enumerate(zip(captured, spatial_layout)):
                    batch_size, token_count, channel_count = tokens.shape
                    if batch_size != 1 or token_count != int(layout["seq_length"]):
                        raise RuntimeError(
                            f"Scale {scale_index} tensor/layout mismatch: "
                            f"{tuple(tokens.shape)} vs {layout}"
                        )
                    pooled_parts.append(tokens.float().amax(dim=1))
                    scale_summary.append({
                        "scale": scale_index,
                        "tokens": int(token_count),
                        "channels": int(channel_count),
                        "grid_size": list(layout["grid_size"]),
                        "window_dims": list(layout["window_dims"]),
                        "sequence_length": int(layout["seq_length"]),
                        "fp16_bytes_per_series": int(token_count * channel_count * 2),
                    })
                reproduced = F.normalize(torch.cat(pooled_parts, dim=-1), dim=-1)
                pool_error = float(
                    (reproduced - baseline_embedding.detach().float().cpu()).abs().max().item()
                )
                if pool_error > 2e-3:
                    raise RuntimeError(
                        f"Captured tokens do not reproduce Pillar-0 output; max error={pool_error}"
                    )
                if RUN_MODE in {"compression_probe", "full_spatial_cache"}:
                    if len(captured) != 3 or tuple(PATCH_GRID) != (32, 32, 32):
                        raise RuntimeError("Compression probe expects the verified 32³/8³/2³ pyramid")

                    # The first contextual scale is a 32³ grid. Average pooling
                    # each 4³ neighborhood creates 8³ tokens while retaining
                    # its coarse location. The next scales are already 8³ and 2³.
                    fine = captured[0].float().reshape(1, *PATCH_GRID, 384)
                    fine = fine.permute(0, 4, 1, 2, 3).contiguous()
                    fine8 = F.avg_pool3d(fine, kernel_size=4, stride=4)[0]
                    fine8 = fine8.permute(1, 2, 3, 0).reshape(512, 384).cpu()
                    native8 = captured[1][0].float().cpu()
                    native2 = captured[2][0].float().cpu()
                    if tuple(native8.shape) != (512, 384) or tuple(native2.shape) != (8, 384):
                        raise RuntimeError(
                            f"Unexpected coarser token shapes: {tuple(native8.shape)}, {tuple(native2.shape)}"
                        )

                    # Per-series, per-scale, per-channel symmetric int8. Store
                    # the scale factors so the reader can recover FP32 values.
                    quantized_scales = []
                    channel_scales = []
                    quant_metrics = []
                    for values in (fine8, native8, native2):
                        if not torch.isfinite(values).all():
                            raise RuntimeError("Non-finite contextual token before compression")
                        channel_scale = values.abs().amax(dim=0) / 127.0
                        channel_scale = torch.where(
                            channel_scale > 0, channel_scale, torch.ones_like(channel_scale)
                        )
                        quantized = torch.round(values / channel_scale).clamp(-127, 127).to(torch.int8)
                        restored = quantized.float() * channel_scale
                        cosine = float(F.cosine_similarity(
                            values.reshape(1, -1), restored.reshape(1, -1), dim=1
                        ).item())
                        relative_rmse = float(
                            ((values - restored).square().mean().sqrt()
                             / values.square().mean().sqrt().clamp_min(1e-8)).item()
                        )
                        quantized_scales.append(quantized.numpy())
                        channel_scales.append(channel_scale.numpy().astype(np.float32))
                        quant_metrics.append({"cosine": cosine, "relative_rmse": relative_rmse})

                    support8 = F.avg_pool3d(
                        support_grid.float(), kernel_size=4, stride=4
                    )[0, 0].reshape(512)
                    support8_u8 = torch.round(support8.clamp(0, 1) * 255).to(torch.uint8).numpy()
                    pooled_mean_error = float(
                        (fine8.mean(dim=0) - captured[0].float().mean(dim=1)[0].cpu()).abs().max().item()
                    )
                    if pooled_mean_error > 2e-5:
                        raise RuntimeError(
                            f"Spatial pooling changed global mean unexpectedly: {pooled_mean_error}"
                        )
                    token_record = {
                        "study_uid": str(row.StudyInstanceUID),
                        "series_uid": str(row.SeriesInstanceUID),
                        "fine8": quantized_scales[0],
                        "native8": quantized_scales[1],
                        "native2": quantized_scales[2],
                        "channel_scales": np.stack(channel_scales),
                        "support8": support8_u8,
                        "quant_metrics": quant_metrics,
                        "pooled_mean_max_abs_error": pooled_mean_error,
                    }
                    compressed_token_records.append({
                        "study_uid": token_record["study_uid"],
                        "series_uid": token_record["series_uid"],
                        "quant_metrics": quant_metrics,
                        "pooled_mean_max_abs_error": pooled_mean_error,
                    })
                    token_name = hashlib.sha256(
                        token_record["series_uid"].encode("utf-8")
                    ).hexdigest()[:20]
                    if RUN_MODE == "compression_probe":
                        token_path = SPATIAL_CHUNK_ROOT / f"spatial_{token_name}.npz"
                        token_temp = token_path.with_suffix(".tmp.npz")
                        np.savez_compressed(
                            token_temp,
                            study_uid=np.asarray(token_record["study_uid"]),
                            series_uid=np.asarray(token_record["series_uid"]),
                            fine8=token_record["fine8"],
                            native8=token_record["native8"],
                            native2=token_record["native2"],
                            channel_scales=token_record["channel_scales"],
                            support8=token_record["support8"],
                            grid_shapes=np.asarray([[8, 8, 8], [8, 8, 8], [2, 2, 2]], dtype=np.int16),
                            checkpoint_revision=np.asarray(EXPECTED_REVISION),
                        )
                        token_temp.replace(token_path)
                        with np.load(token_path, allow_pickle=False) as saved:
                            expected_shapes = {
                                "fine8": (512, 384), "native8": (512, 384),
                                "native2": (8, 384), "channel_scales": (3, 384),
                                "support8": (512,),
                            }
                            for name, expected_shape in expected_shapes.items():
                                if saved[name].shape != expected_shape:
                                    raise RuntimeError(
                                        f"Saved {name} shape {saved[name].shape}, expected {expected_shape}"
                                    )
                            if not np.isfinite(saved["channel_scales"]).all():
                                raise RuntimeError("Non-finite compression scales after disk round trip")
                        if (token_record["study_uid"], token_record["series_uid"]) not in existing_processed_ids():
                            raise RuntimeError("Spatial token chunk was not detected for resume")
                        print(
                            f"Saved and reloaded spatial chunk: {token_path.name} | "
                            f"{token_path.stat().st_size:,} bytes | int8 scales passed resume check",
                            flush=True,
                        )
                    else:
                        spatial_cache_buffer.append({
                            **token_record,
                            "plane": str(row.Anatomical_Plane),
                            "fluid_sensitive": str(row.Fluid_Sensitive),
                            "fat_suppression": str(row.Fat_Suppression),
                            "native_slice_count": int(geometry["native_shape_drc"][0]),
                            "native_finite_fraction": prep["native_finite_fraction"],
                            "canvas_support_fraction": prep["canvas_support_fraction"],
                        })
                        if len(spatial_cache_buffer) >= SPATIAL_CHUNK_SIZE:
                            save_spatial_batch()
                token_probe = {
                    "series_uid": str(row.SeriesInstanceUID),
                    "scales": scale_summary,
                    "total_tokens": int(sum(item["tokens"] for item in scale_summary)),
                    "fp16_bytes_per_series_all_scales": int(
                        sum(item["fp16_bytes_per_series"] for item in scale_summary)
                    ),
                    "max_abs_error_reconstructed_global_embedding": pool_error,
                    "token_storage_saved": RUN_MODE == "compression_probe",
                }
                if RUN_MODE != "full_spatial_cache":
                    probe_filename = (
                        "spatial_compression_probe.json"
                        if RUN_MODE == "compression_probe" else "spatial_token_probe.json"
                    )
                    (CHECKPOINT_ROOT / probe_filename).write_text(json.dumps(token_probe, indent=2))
                    print("Spatial token probe:", json.dumps(token_probe, sort_keys=True), flush=True)
                del captured, pooled_parts, reproduced
                token_probe_state["tokens"].clear()
            if RUN_MODE != "full_spatial_cache":
                gate_state["enabled"] = True
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    gated_embedding = pillar.extract_vision_feats({MODEL_INPUT_KEY: gpu_input})
                gate_state["enabled"] = False
                base = baseline_embedding.float().reshape(-1).cpu().numpy()
                gated = gated_embedding.float().reshape(-1).cpu().numpy()
                if base.shape != (EXPECTED_EMBEDDING_DIM,) or gated.shape != base.shape:
                    raise RuntimeError(f"Bad embedding shapes: {base.shape}, {gated.shape}")
                if not np.isfinite(base).all() or not np.isfinite(gated).all():
                    raise RuntimeError("Non-finite feature output; stopping before checkpoint")
                if abs(np.linalg.norm(base) - 1.0) > 0.02 or abs(np.linalg.norm(gated) - 1.0) > 0.02:
                    raise RuntimeError("Unexpected Pillar-0 embedding norm")
            elapsed = time.perf_counter() - series_start
            series_times.append(elapsed)
            if RUN_MODE != "full_spatial_cache":
                rows_buffer.append({
                "study_uid": str(row.StudyInstanceUID),
                "series_uid": str(row.SeriesInstanceUID),
                "plane": str(row.Anatomical_Plane),
                "fluid_sensitive": str(row.Fluid_Sensitive),
                "fat_suppression": str(row.Fat_Suppression),
                "native_slice_count": int(geometry["native_shape_drc"][0]),
                "native_finite_fraction": prep["native_finite_fraction"],
                "canvas_support_fraction": prep["canvas_support_fraction"],
                })
                baseline_buffer.append(base)
                gated_buffer.append(gated)
            newly_processed += 1
            if newly_processed % 4 == 0 or position == len(pending):
                print(
                    f"Progress {newly_processed}/{len(pending)} | "
                    f"series/s {newly_processed / max(time.perf_counter()-run_start, 1e-6):.3f} | "
                    f"last {elapsed:.1f}s | gated-out {(support_grid <= SUPPORT_THRESHOLD).float().mean().item():.1%}",
                    flush=True,
                )
            del model_input, support_grid, gpu_input, baseline_embedding
            if RUN_MODE != "full_spatial_cache":
                del gated_embedding, base, gated
            gc.collect()
            torch.cuda.empty_cache()
            if len(rows_buffer) >= CHUNK_SIZE:
                save_chunk()

        save_chunk()
        save_spatial_batch()
        if RUN_MODE == "compression_probe":
            if len(compressed_token_records) != newly_processed:
                raise RuntimeError(
                    f"Saved {len(compressed_token_records)} spatial records for {newly_processed} new series"
                )
            scale_names = ("fine32_to_fine8", "native8", "native2")
            quantization_summary = {}
            for scale_index, name in enumerate(scale_names):
                cosines = [record["quant_metrics"][scale_index]["cosine"]
                           for record in compressed_token_records]
                relative_rmses = [record["quant_metrics"][scale_index]["relative_rmse"]
                                  for record in compressed_token_records]
                quantization_summary[name] = {
                    "mean_cosine": float(np.mean(cosines)),
                    "minimum_cosine": float(np.min(cosines)),
                    "mean_relative_rmse": float(np.mean(relative_rmses)),
                    "maximum_relative_rmse": float(np.max(relative_rmses)),
                }
            token_files = sorted(SPATIAL_CHUNK_ROOT.glob("spatial_*.npz"))
            probe_summary = {
                "experiment": "pillar0_spatial_token_compression_probe",
                "run_mode": RUN_MODE,
                "checkpoint_revision": EXPECTED_REVISION,
                "patch_grid": list(PATCH_GRID),
                "series_processed_this_run": newly_processed,
                "compressed_spatial_chunks": len(token_files),
                "mean_spatial_chunk_bytes": float(np.mean([path.stat().st_size for path in token_files])),
                "candidate_raw_int8_bytes_per_series": 512 * 384 * 2 + 8 * 384 + 3 * 384 * 4 + 512,
                "full_pyramid_fp16_bytes_per_series": 25565184,
                "quantization": quantization_summary,
                "all_spatial_chunks_roundtrip_verified": True,
                "resume_detection_verified": True,
                "labels_read": False,
                "training_performed": False,
            }
            summary_path = CHECKPOINT_ROOT / "spatial_compression_probe_summary.json"
            summary_path.write_text(json.dumps(probe_summary, indent=2))
            print("Spatial compression probe summary:", json.dumps(probe_summary, sort_keys=True), flush=True)
        if PUBLISH_PRIVATE_CHECKPOINTS:
            # Publish the final partial chunk as well as all complete chunks.
            REMOTE_DATASET_EXISTS = upload_private_checkpoint(REMOTE_DATASET_EXISTS)

        run_seconds = time.perf_counter() - run_start
        if not PUBLISH_PRIVATE_CHECKPOINTS:
            print("Probe/pilot artifacts are under /kaggle/working; save a notebook version to retain them.")
        print("New series:", newly_processed)
        print("Mean end-to-end seconds/series:", round(float(np.mean(series_times)) if series_times else 0.0, 2))
        print("Elapsed minutes:", round(run_seconds / 60, 2))
        print("Processed so far (including prior chunks):", len(existing_processed_ids()))
        print("Feature chunks:", len(list(CHUNK_ROOT.glob("features_*.npz"))))
        '''
    ),
    markdown(
        r"""
        ## 7. Verify the saved cache

        This CPU-only check validates all saved chunks and reports the measured pace.
        It does not read label columns or unlock the 58-study evaluation set.
        """
    ),
    code(
        r'''
        if RUN_MODE == "full_spatial_cache":
            chunk_paths = sorted(SPATIAL_CHUNK_ROOT.glob("spatial_batch_*.npz"))
            if not chunk_paths:
                raise RuntimeError("No spatial-token batches were saved")
            seen = set()
            row_count = 0
            for path in chunk_paths:
                with np.load(path, allow_pickle=False) as archive:
                    keys = list(zip(archive["study_uid"].astype(str), archive["series_uid"].astype(str)))
                    expected = {
                        "fine8": (len(keys), 512, 384),
                        "native8": (len(keys), 512, 384),
                        "native2": (len(keys), 8, 384),
                        "channel_scales": (len(keys), 3, 384),
                        "support8": (len(keys), 512),
                    }
                    for name, shape in expected.items():
                        if archive[name].shape != shape:
                            raise RuntimeError(f"Invalid {name} shape in {path.name}: {archive[name].shape}")
                    if not np.isfinite(archive["channel_scales"]).all():
                        raise RuntimeError(f"Invalid quantization scales in {path.name}")
                    if len(keys) != len(set(keys)) or seen.intersection(keys):
                        raise RuntimeError(f"Duplicate series key in {path.name}")
                    seen.update(keys)
                    row_count += len(keys)
            if row_count != len(seen):
                raise RuntimeError("Spatial cache row accounting failed")
            expected_count = len(existing_processed_ids())
            if row_count != expected_count:
                raise RuntimeError(f"Cache verification mismatch: {row_count} rows, {expected_count} indexed")
            print("Verified spatial-token series:", row_count)
            print("Per-series scales:", "fine 8³ (512×384), middle 8³ (512×384), coarse 2³ (8×384)")
            print("Quantization:", "per-series/per-scale/per-channel int8 with FP32 scales")
            print("Gold studies extracted:", 0)
            print("Private cache:", CHECKPOINT_DATASET_ID)
        else:
            chunk_paths = sorted(CHUNK_ROOT.glob("features_*.npz"))
            if not chunk_paths:
                raise RuntimeError("No feature chunks were saved")
            seen = set()
            row_count = 0
            norm_rows = []
            for path in chunk_paths:
                with np.load(path, allow_pickle=False) as archive:
                    base = archive["baseline"]
                    gated = archive["support_gated"]
                    if base.ndim != 2 or base.shape[1] != EXPECTED_EMBEDDING_DIM or gated.shape != base.shape:
                        raise RuntimeError(f"Invalid feature arrays in {path.name}")
                    if not np.isfinite(base).all() or not np.isfinite(gated).all():
                        raise RuntimeError(f"Non-finite features in {path.name}")
                    keys = list(zip(archive["study_uid"].astype(str), archive["series_uid"].astype(str)))
                    if len(keys) != len(set(keys)) or seen.intersection(keys):
                        raise RuntimeError(f"Duplicate series key in {path.name}")
                    seen.update(keys)
                    row_count += len(keys)
                    norm_rows.extend(np.linalg.norm(base.astype(np.float32), axis=1).tolist())
                    norm_rows.extend(np.linalg.norm(gated.astype(np.float32), axis=1).tolist())
            if row_count != len(seen):
                raise RuntimeError("Feature row accounting failed")
            print("Verified feature rows:", row_count)
            print("Unique study-series keys:", len(seen))
            print("Feature shape per view:", (EXPECTED_EMBEDDING_DIM,))
            print("Norm range:", round(min(norm_rows), 4), "to", round(max(norm_rows), 4))
        print("Local output:", CHECKPOINT_ROOT)
        if PUBLISH_PRIVATE_CHECKPOINTS:
            print("Private Kaggle dataset:", CHECKPOINT_DATASET_ID)
        '''
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"pillar0-cell-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUT}")
