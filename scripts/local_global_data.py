"""Native-neighbor cache and selected-window batches shared by train and inference."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCHEMA = "native_triplets_v1"
MAX_WINDOWS = 48
MEAN, STD = 0.449, 0.226
# Runner chooses these only after the same-GPU timing/memory preflight.
EXECUTION = {"augment_on_gpu": False, "channels_last": False}


def category(value, choices):
    value = str(value).strip().lower()
    return choices.index(value) if value in choices else len(choices)


def window_metadata(record):
    """Coordinates are relative to their own series, never aligned across planes."""
    n = record["count"]
    plane = category(record.get("plane"), ["sagittal", "coronal", "axial"])
    fluid = category(record.get("fluid_sensitive"), ["0", "1"])
    fat = category(record.get("fat_suppression"), ["0", "1"])
    geometry = record["order_source"] == "physical"
    position = np.asarray(record["positions"], dtype=np.float32)
    gaps = np.asarray(record["neighbor_gaps_mm"], dtype=np.float32)
    if position.shape != (n,) or gaps.shape != (n, 2):
        raise ValueError("Metadata and centre counts differ")
    # Four categorical fields and eight numerical fields. Unknown geometry has
    # zero gap encodings plus an explicit availability indicator.
    phases = np.pi * position
    numeric = np.column_stack((position * 2 - 1, np.sin(phases), np.cos(phases),
                               np.sin(2 * phases), np.cos(2 * phases),
                               np.log1p(gaps[:, 0]), np.log1p(gaps[:, 1]),
                               np.full(n, float(geometry)))).astype(np.float32)
    discrete = np.tile([plane, fluid, fat, record["slot"]], (n, 1)).astype(np.float32)
    result = np.column_stack((discrete, numeric))
    if not np.isfinite(result).all() or (gaps < 0).any():
        raise ValueError("Invalid position/gap metadata")
    return result


class NativeStudyCache:
    """Select centres before materializing triplets; mmap unique slices once."""
    def __init__(self, manifest):
        self.manifest = manifest
        self._parts = {}
        self.records = {}
        for sid, rows in manifest.items():
            rows = sorted(rows, key=lambda r: r["slot"])
            for r in rows:
                if r.get("schema") != SCHEMA:
                    raise ValueError("Native acquired-neighbor cache required; old cache is incompatible")
                mapping = np.asarray(r["triplet_offsets"], dtype=np.int64)
                if mapping.shape != (r["count"], 3) or mapping.min() < 0 or mapping.max() >= r["stored_count"]:
                    raise ValueError("Invalid triplet row mapping")
            self.records[sid] = (rows, np.cumsum([0] + [r["count"] for r in rows]),
                                 np.concatenate([window_metadata(r) for r in rows]))

    def part(self, directory, name):
        key = (directory, name)
        if key not in self._parts:
            self._parts[key] = np.load(Path(directory) / name, mmap_mode="r")
        return self._parts[key]

    def selected(self, sid, keep):
        rows, boundaries, metadata = self.records[sid]
        images, supports = [], []
        for i, r in enumerate(rows):
            local = keep[(keep >= boundaries[i]) & (keep < boundaries[i + 1])] - boundaries[i]
            if not len(local):
                continue
            offsets = np.asarray(r["triplet_offsets"], dtype=np.int64)[local] + r["row_start"]
            images.append(self.part(r["directory"], r["part"])[offsets])
            supports.append(self.part(r["directory"], r["support_part"])[offsets])
        # Caller supplies sorted indices; row ordering remains slot-major.
        return np.concatenate(images), np.concatenate(supports), metadata[keep]


@dataclass
class WindowBatch:
    images: torch.Tensor
    metadata: torch.Tensor
    padded_indices: torch.Tensor
    valid: torch.Tensor


def augment_images(images, owners, studies, rng, settings):
    """Baseline random draws; transform intensity and support together.

    No CUDA-to-CPU queries or full-size erasing masks. Mask geometry uses nearest
    interpolation; intensity uses bilinear. Erasing applies only to intensities.
    """
    device = images.device
    area = rng.uniform(settings["crop_area_min"], 1.0, studies)
    extent = np.sqrt(area)
    angles = np.radians(rng.uniform(-settings["rotation_degrees"], settings["rotation_degrees"], studies))
    margin = (1 - extent) + settings.get("translate_fraction", 0.0)
    ox = rng.uniform(-1, 1, studies) * margin
    oy = rng.uniform(-1, 1, studies) * margin
    theta = np.zeros((studies, 2, 3), dtype=np.float32)
    theta[:, 0, 0] = theta[:, 1, 1] = np.cos(angles) * extent
    theta[:, 0, 1], theta[:, 1, 0] = -np.sin(angles) * extent, np.sin(angles) * extent
    theta[:, 0, 2], theta[:, 1, 2] = ox, oy
    grid = F.affine_grid(torch.as_tensor(theta, device=device)[owners], images.shape, align_corners=False)
    intensity = F.grid_sample(images[:, :3], grid, mode="bilinear", align_corners=False)
    support = F.grid_sample(images[:, 3:], grid, mode="nearest", align_corners=False)
    gain = torch.as_tensor(rng.uniform(1 - settings["intensity_jitter"], 1 + settings["intensity_jitter"], studies), dtype=torch.float32, device=device)
    shift = torch.as_tensor(rng.uniform(-settings["brightness_shift"], settings["brightness_shift"], studies), dtype=torch.float32, device=device)
    intensity.mul_(gain[owners, None, None, None]).add_(shift[owners, None, None, None])
    erase = rng.random(len(images)) < settings["erase_probability"]
    side = int(round(images.shape[-1] * settings["erase_fraction"]))
    for i in np.flatnonzero(erase):
        top, left = (int(rng.integers(0, images.shape[-1] - side)) for _ in range(2))
        intensity[i, :, top:top + side, left:left + side] = 0
    return torch.cat((intensity, support), dim=1)


def prepare_batch(cache, labels, study_ids, slice_cap, rng, device, augmentation=None, sources=None):
    pixels, metadata, owners, padded, targets, weights = [], [], [], [], [], []
    valid = np.zeros((len(study_ids), MAX_WINDOWS), dtype=bool)
    for i, sid in enumerate(study_ids):
        n = len(cache.records[sid][2])
        if not 0 < n <= MAX_WINDOWS:
            raise ValueError(f"Invalid centre count for {sid}: {n}")
        keep = np.arange(n)
        if slice_cap is not None and n > slice_cap:
            keep = (np.sort(rng.choice(n, slice_cap, replace=False)) if rng is not None
                    else np.linspace(0, n - 1, slice_cap).round().astype(int))
        images, support, meta = cache.selected(sid, keep)
        pixels.append(np.concatenate((images, support), axis=1))
        metadata.append(meta)
        owners.extend([i] * len(keep))
        padded.extend(i * MAX_WINDOWS + np.arange(len(keep)))
        valid[i, :len(keep)] = True
        targets.append(labels[sid][0]); weights.append(labels[sid][1])
    # Keep uint8 until transfer: 4x smaller host buffer and PCIe transfer than
    # moving float32. No extra prefetched batch or permanent device cache.
    host = torch.from_numpy(np.concatenate(pixels))
    if device.type == "cuda":
        host = host.pin_memory()
    processing_device = device if EXECUTION["augment_on_gpu"] else torch.device("cpu")
    images = host.to(processing_device, non_blocking=True).float()
    images[:, :3].div_(255).sub_(MEAN).div_(STD)
    images[:, 3:].div_(255)
    owner = torch.tensor(owners, dtype=torch.long, device=processing_device)
    if rng is not None:
        images = augment_images(images, owner, len(study_ids), rng, augmentation)
    images = images.to(device, non_blocking=True)
    if EXECUTION["channels_last"]:
        images = images.contiguous(memory_format=torch.channels_last)
    owner = owner.to(device)
    batch = WindowBatch(images, torch.as_tensor(np.concatenate(metadata), device=device),
                        torch.tensor(padded, device=device), torch.as_tensor(valid, device=device))
    return (batch, owner, torch.as_tensor(np.stack(targets), device=device),
            torch.as_tensor(np.stack(weights), device=device),
            torch.as_tensor(np.stack([sources[s] for s in study_ids]), device=device) if sources else None,
            torch.ones(len(owners), device=device))
