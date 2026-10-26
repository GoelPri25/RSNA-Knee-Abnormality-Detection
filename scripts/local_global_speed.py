"""Short paired timing preflight; retain only faster choices within baseline VRAM."""
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from local_global_data import EXECUTION, prepare_batch


def choose_execution(model, cache, labels, train_ids, config, device, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    path = output / "execution.json"
    if device.type != "cuda":
        EXECUTION.update(augment_on_gpu=False, channels_last=False)
        model.to(memory_format=torch.contiguous_format)
        return
    if path.exists():
        report = json.loads(path.read_text())
        if report["gpu"] != torch.cuda.get_device_name(device):
            raise ValueError("Recovery GPU differs; do not silently change execution settings")
        EXECUTION.update(report["selected"])
        model.to(memory_format=torch.channels_last if EXECUTION["channels_last"] else torch.contiguous_format)
        print("EXECUTION_RECOVERED", json.dumps(report["selected"]), flush=True)
        return
    cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all()
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ids = train_ids[:int(config["batch_studies"])]
    results = []
    try:
        for gpu_aug, channels_last in [(False, False), (False, True), (True, False), (True, True)]:
            choice = dict(augment_on_gpu=gpu_aug, channels_last=channels_last)
            EXECUTION.update(choice)
            model.load_state_dict(initial)
            model.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
            torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
            rng = np.random.default_rng(config["seed"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=config["mixed_precision"])
            model.train(); durations = []
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            batch = owner = target = weight = pred = cells = loss = None
            try:
                for step in range(5):
                    torch.cuda.synchronize(); start = time.perf_counter()
                    batch, owner, target, weight, _, _ = prepare_batch(cache, labels, ids, 24, rng, device, config["augmentation"])
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", enabled=config["mixed_precision"]):
                        pred = model(batch, owner, len(ids))
                        cells = torch.nn.functional.binary_cross_entropy_with_logits(pred.float(), target, reduction="none")
                        loss = (cells * weight).sum() / weight.sum()
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                    scaler.step(optimizer); scaler.update()
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
                    if step >= 2: durations.append(elapsed)
                    del batch, pred, cells, loss
                result = {**choice, "median_seconds": statistics.median(durations),
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "ok": True}
            except torch.cuda.OutOfMemoryError:
                result = {**choice, "ok": False, "reason": "oom"}
            finally:
                # An OOM may leave a partial autograd graph alive. Release it
                # before the next variant so one failure cannot poison others.
                batch = owner = target = weight = pred = cells = loss = None
                optimizer.zero_grad(set_to_none=True); del optimizer, scaler
                gc.collect(); torch.cuda.empty_cache()
            results.append(result); print("SPEED_PREFLIGHT", json.dumps(result), flush=True)
        base = results[0]
        if not base["ok"]: raise RuntimeError("Batch-8 reference exceeds GPU memory; stop rather than silently change experiment")
        # Allocated and reserved high-water marks must both stay within the
        # reference footprint. No larger GPU, batch, prefetched queue or cache.
        eligible = [r for r in results if r["ok"] and
                    r["peak_allocated_bytes"] <= base["peak_allocated_bytes"] and
                    r["peak_reserved_bytes"] <= base["peak_reserved_bytes"]]
        winner = min(eligible, key=lambda r: r["median_seconds"])
        selected = {k: winner[k] for k in ("augment_on_gpu", "channels_last")}
        report = {"gpu": torch.cuda.get_device_name(device), "batch": len(ids), "windows_per_study": 24,
                  "candidates": results, "selected": selected,
                  "measured_step_speedup": base["median_seconds"] / winner["median_seconds"],
                  "note": "Short warm-cache step benchmark; not a claimed epoch speedup."}
        temporary = path.with_suffix(".json.tmp"); temporary.write_text(json.dumps(report, indent=2) + "\n"); temporary.replace(path)
        EXECUTION.update(selected)
        print("EXECUTION_SELECTED", json.dumps(report), flush=True)
    finally:
        model.load_state_dict(initial)
        model.to(memory_format=torch.channels_last if EXECUTION["channels_last"] else torch.contiguous_format)
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
        del initial
        gc.collect(); torch.cuda.empty_cache()
