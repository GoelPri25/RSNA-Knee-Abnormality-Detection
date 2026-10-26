"""Ledger row T-enc: whole-report embeddings from several multilingual text encoders (Kaggle T4, fp16).

Reads the competition train.csv inside Kaggle; report text never leaves the kernel (internet is used only to download
encoder weights from Hugging Face). Gold rows (host labels present) are skipped. Output in /kaggle/working:
emb_<name>.npy (float32, L2-normalised, rows in study_ids.json order), study_ids.json, meta.json. No text is written.
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

ROOTS = (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
         Path("/kaggle/input/rsna-knee-abnormality-detection"))
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OUT = Path("/kaggle/working")
# name -> (hub id, max tokens, prefix)
ENCODERS = {
    "e5_base": ("intfloat/multilingual-e5-base", 512, "passage: "),   # parity check vs the local T-pre cache
    "labse": ("sentence-transformers/LaBSE", 512, ""),
    "e5_large": ("intfloat/multilingual-e5-large", 512, "passage: "),
    "bge_m3": ("BAAI/bge-m3", 2048, ""),
    "qwen3_emb_06b": ("Qwen/Qwen3-Embedding-0.6B", 2048, ""),  # documents take no prompt; fp32 (bf16-trained)
}
FP32 = {"qwen3_emb_06b"}
BATCH = {"qwen3_emb_06b": 2}  # fp32 at 2,048 tokens OOMs a T4 at batch 16 (v2)
ONLY = {"qwen3_emb_06b"}  # v3: v2 already produced the other four

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
    from sentence_transformers import SentenceTransformer

csv.field_size_limit(10 ** 8)
root = next((p for p in ROOTS if (p / "train.csv").is_file()), None)
if root is None:
    raise FileNotFoundError("Competition train.csv is not attached")
rows = list(csv.DictReader(open(root / "train.csv", newline="", encoding="utf-8")))
rows = [r for r in rows if not any((r.get(c) or "").strip() for c in TARGETS) and r["Report"].strip()]
ids = [r["StudyInstanceUID"] for r in rows]
texts = [r["Report"].strip() for r in rows]
print(f"{len(rows)} report-only studies (gold excluded); GPU {torch.cuda.get_device_name(0)}", flush=True)
(OUT / "study_ids.json").write_text(json.dumps(ids) + "\n")

meta = {}
for name, (hub, max_len, prefix) in ENCODERS.items():
    if ONLY and name not in ONLY:
        continue
    t = time.time()
    try:
        model = SentenceTransformer(hub, device="cuda", trust_remote_code=False)
        model.max_seq_length = max_len
        if name not in FP32:
            model.half()
        emb = model.encode([prefix + x for x in texts], batch_size=BATCH.get(name, 16), normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        assert emb.shape[0] == len(ids) and np.isfinite(emb).all(), name
        np.save(OUT / f"emb_{name}.npy", emb)
        n_trunc = sum(len(model.tokenizer(prefix + x)["input_ids"]) > max_len for x in texts)
        meta[name] = {"hub": hub, "dim": int(emb.shape[1]), "max_tokens": max_len, "prefix": prefix,
                      "truncated_reports": int(n_trunc), "seconds": round(time.time() - t, 1),
                      "modules": [type(m).__name__ for m in model]}
        del model
        torch.cuda.empty_cache()
    except Exception as e:  # keep the other encoders' outputs
        meta[name] = {"hub": hub, "error": f"{type(e).__name__}: {e}"[:500]}
    print(name, meta[name], flush=True)
(OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
print("done", flush=True)
