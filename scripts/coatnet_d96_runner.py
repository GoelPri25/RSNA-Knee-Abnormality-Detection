"""Experiment 1: the public 0.949 CoAtNet recipe trained on our fold-0 split and HPO-v3.1-D2 labels (H100).

Reproduced from the public inference code (nartaa / dreaddevelopment "Raptor"):
  * 96-slice d96 corpus (built by kaggle/experiment_kernels/coatnet_d96_cache), 384 px, 140 mm field;
  * 3-slice windows (c-1, c, c+1) over the study's valid slice range, slot boundaries ignored as in theirs;
  * coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k -> LayerNorm -> per-finding attention (Linear 256, tanh) ->
    per-finding linear classifier (their RaptorClassifier);
  * train at 384 full field; evaluate on the centre 320 crop with 94 windows; anatomical mirror TTA.
Their training script is not public. Our choices (fixed before the run): 16 stratified training windows,
AdamW (backbone 1e-4, head 1e-3, wd 0.05), 1 warm-up epoch then cosine, 15 epochs, effective batch 8 studies,
bf16, per-study augmentation (anatomical mirror p=0.5, rotation +-10 deg, scale +-10%, shift +-6%,
gamma/gain jitter), soft-label BCE with the label file's cell weights.

Pre-registered primary model: weight average of epochs 11-15 (fixed, not chosen on validation) with BatchNorm
statistics refreshed on 256 training studies, scored on the 825 fold-0 studies against Steven-v4
(confident cells: <= 0.3 / >= 0.7), 320 crop, 94 windows, no TTA. Accept if it beats d2_224 (0.8963) by
>= +0.005 with >= 90% of 2,000 paired bootstraps better (scored locally by scripts/audit_coatnet_d96.py).
Best-epoch and TTA variants are reported as secondary. Gold studies are only predicted, never trained on;
they are scored locally.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from d96_positions import positions

ASSETS = Path(os.environ.get("D96_ASSETS", "/root/assets"))
CACHE = Path(os.environ.get("D96_CACHE", "/tmp/d96"))
OUT = Path(os.environ.get("D96_OUT", "/root/vol/experiments/coatnet_d96_fold0/run"))
SHARDS = 4
CFG = {
    "version": "coatnet_d96_exp1_v1",
    "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "train_res": 384, "eval_res": 320, "img": 384, "maxs": 96, "slot_sizes": [26, 22, 18, 12, 18],
    "train_windows": 16, "eval_windows": 94,
    "epochs": 15, "studies_per_step": 2, "accum": 4, "lr": 1e-4, "head_lr": 1e-3, "wd": 0.05,
    "warmup_epochs": 1, "grad_clip": 1.0, "swa_epochs": [11, 12, 13, 14, 15], "bn_refresh_studies": 256,
    "aug": {"mirror_p": 0.5, "rot_deg": 10.0, "scale": 0.10, "shift": 0.06, "gamma": 0.25, "gain": 0.10},
    "validation_fold": "0", "seed": 2026, "workers": 8,
    "sha256": {"labels": "38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d",
               "folds": "2dd1044adcc54ca4d9bcb9c9fd3b910a6784bd2b99e792bbeaf84a42d417f06a",
               "validation": "10fdaaab54e67300b91415fb3a5d4f7567d21cceb71ae973df7f3fd7e545a7e5"},
}
# Exp 1b (pre-registered 2026-10-10, after Exp 1 failed with AUC still rising when the cosine reached zero):
# restart from Exp 1 epoch-15 weights and optimizer state, 8 more epochs (16-23), 1 warm-up epoch to half
# the original peak (5e-5 / 5e-4), cosine to 0, SWA of 20-23, same BN refresh, same scorer and acceptance rule.
if os.environ.get("D96_PHASE") == "continue":
    CFG.update({"version": "coatnet_d96_exp1b_continue_v1", "init_from": "exp1 recovery.pt (epoch 15)",
                "first_epoch": 16, "epochs": 8, "lr": 5e-5, "head_lr": 5e-4, "swa_epochs": [20, 21, 22, 23]})
# Exp 1c (pre-registered 2026-10-10): the same recipe with a ResNet-50 backbone, one 25-epoch cosine holding a
# 1e-6 floor (1% of peak), SWA of 21-25. Isolates recipe (vs d2_224) from backbone (vs Exp 1b).
if os.environ.get("D96_PHASE") == "r50":
    CFG.update({"version": "d96_r50_exp1c_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25]})
# Exp 1e (pre-registered 2026-10-10): Exp 1c exactly, but trained on the teammate's per-target label set
# (priyanjaliml/rsna-knee-final-per-target-labels; Lateral OA and Baker's are Steven-v4, so fold 0 is scored on the
# other 10 targets for the decision).
LABELS_FILE = "soft_labels_hpo_v3_1_d2_structured.json"
if os.environ.get("D96_PHASE") == "r50_pertarget":
    CFG.update({"version": "d96_r50_exp1e_pertarget_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25]})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp 2a (pre-registered 2026-10-10): Exp 1c exactly (D2 labels, same schedule, seed and SWA), plus context between
# windows: each window's LayerNorm feature is projected 2048 -> 512, given a learned slot embedding (5 slots) and an
# anatomical position (scripts/d96_positions.py: sagittal medial->lateral, coronal anterior->posterior, axial
# superior->inferior; 4 sin/cos frequencies + a known flag), passed through a 2-layer pre-norm transformer (8 heads,
# FFN 1024, dropout 0.1), projected back to 2048 and added through a per-channel gate initialised at 0 (so the
# untrained model is Exp 1c's head exactly), then the same per-finding attention pooling.
# Accept if the SWA 21-25 model beats Exp 1c (0.9074, 12-target) by >= +0.003 with >= 90% of 2,000 paired
# bootstraps better. Secondary: 6 x 16-window passes averaged (the training set size), mirror TTA, gold.
if os.environ.get("D96_PHASE") == "r50_ctx":
    CFG.update({"version": "d96_r50_exp2a_ctx_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25],
                "ctx": {"dim": 512, "layers": 2, "heads": 8, "ffn": 1024, "dropout": 0.1, "freqs": 4,
                        "multi_passes": 6}})
# Exp 3a (pre-registered 2026-10-11): Exp 1c exactly, but D2's borderline cells (0.3 < p < 0.7; severity-aware
# "mild / small / minimal") become 0.85 for the 7 targets where the ruler calls most of them positive. Training
# folds only by construction (fold-0 labels are never trained on); a fixed rule on the text labels, no model.
BORDER = {"targets": ["ACL", "MCL", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Baker's"], "lo": 0.3, "hi": 0.7,
          "to": 0.85}
if os.environ.get("D96_PHASE") == "r50_border":
    CFG.update({"version": "d96_r50_exp3a_border_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25], "border": BORDER})
# Exp 3b (pre-registered 2026-10-12): Exp 1c exactly, but trained on D2 re-valued by D2's own rule states
# (scripts/build_d2_state_labels.py): "mild" (BORD) -> 0.85 for Effusion / MCL / Medial OA; trace / below threshold
# (BELOW) -> 0.15 for Effusion / MCL / ACL / Contusion. 3a's LB loss is attributed to BELOW cells (gold mostly negative).
if os.environ.get("D96_PHASE") == "r50_state":
    CFG.update({"version": "d96_r50_exp3b_state_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25]})
    CFG["sha256"] = {**CFG["sha256"], "labels": "5d5e8b3eb39ac80fdaf57f07d8bdc4af59ab40245499481021794dfb2fc00b29"}
    LABELS_FILE = "soft_labels_hpo_v3_1_d2_state_rule.json"
# Exp 2c (pre-registered 2026-10-12): Exp 1c exactly (D2 labels, schedule, seed, SWA), but per-plane branches
# (RaptorPlane): plane-specific attention pooling and classifier for sagittal / coronal / axial, main logit = sum,
# plus a 0.3-weighted auxiliary BCE on each plane's own logit. Motivation: 1c puts 57-79% of every finding's
# attention on sagittal fluid and 1-5% on axial (data/cam_centring_audit/audit_cam_centring.txt).
if os.environ.get("D96_PHASE") == "r50_plane":
    CFG.update({"version": "d96_r50_exp2c_plane_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25], "plane": {"aux": 0.3}})
# Exp 2d (pre-registered 2026-10-12): Exp 2c's per-plane head trained on Exp 1e's labels (the teammate's per-target
# set, best single on LB 0.936). Compared with 1e on the 10 targets not trained on the ruler.
if os.environ.get("D96_PHASE") == "r50_plane_pertarget":
    CFG.update({"version": "d96_r50_exp2d_plane_pertarget_v1", "arch": "resnet50.tv_in1k", "epochs": 25,
                "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25], "plane": {"aux": 0.3}})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp 1e-s2 (pre-registered 2026-10-12): Exp 1e exactly (labels, schedule, SWA, everything), only the seed changes
# (2026 -> 2027: init of the head, sampling, augmentation, dropout, BN-refresh subset). Measures the seed noise floor
# on fold 0 and, submitted, the run-to-run spread on the public LB.
if os.environ.get("D96_PHASE") == "r50_pertarget_seed":
    CFG.update({"version": "d96_r50_exp1e_pertarget_seed2027_v1", "arch": "resnet50.tv_in1k", "epochs": 25,
                "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25], "seed": 2027})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp 6a (pre-registered 2026-10-12): Exp 1e exactly (labels, seed 2026, cosine) for epochs 1-17, then 8 one-epoch
# cycles (18-25): each restarts at the learning rate the cosine had reached at the start of epoch 18 (0.2575 of
# peak, so no jump) and anneals to the 1% floor by the epoch's end; SWA = average of the 8 cycle ends, BN refreshed.
# Izmailov et al. SWA with a cyclical tail: checkpoints spread over the basin approximate an ensemble in one model.
if os.environ.get("D96_PHASE") == "r50_swa_cyclic":
    CFG.update({"version": "d96_r50_exp6a_swa_cyclic_v1", "arch": "resnet50.tv_in1k", "epochs": 25,
                "lr_floor": 0.01, "swa_epochs": [18, 19, 20, 21, 22, 23, 24, 25], "swa_cycle_start": 18})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp 6b (pre-registered 2026-10-12): Exp 1e exactly (labels, seed 2026, schedule, SWA 21-25), but windows are
# allocated by series. 2-abl: dropping sagittal / coronal non-fluid moves macro AUC by <= 0.003 while ~5.7 of the 16
# stratified training windows land there; coronal fluid (MCL, fracture, contusion) gets ~3. Training: 7 sagittal
# fluid (slot 0), 6 coronal fluid (slot 2), 3 axial (slot 4), each stratified within its slot's valid range; a slot
# missing from a study passes its windows to the present slots in that order. Evaluation: every interior slice of the
# three slots (non-fluid dropped the same way); BN refresh: the 16-window allocation at fixed mid-bin offsets.
if os.environ.get("D96_PHASE") == "r50_series_alloc":
    CFG.update({"version": "d96_r50_exp6b_series_alloc_v1", "arch": "resnet50.tv_in1k", "epochs": 25,
                "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25], "alloc": [[0, 7], [2, 6], [4, 3]]})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp 4a/4b (native-capped input): Exp 1c / 1e exactly, but on the native-capped cache (d96n,
# kaggle/experiment_kernels/d96_native_cache): sagittal fluid, sagittal non-fluid, coronal fluid, axial (fluid
# preferred); every native slice in 2-98%, series capped at 40; variable-length studies; eval = one window per slice.
# 4a = D2 labels (vs 1c), 4b = per-target labels (vs 1e, two seeds). Pre-registration in docs/experiment_ledger.md.
NATIVE = {"layout": "native", "shards": 6, "eval_windows": None, "maxs": None, "slot_sizes": None}
if os.environ.get("D96_PHASE") == "r50_native":
    CFG.update({"version": "d96_r50_exp4a_native_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25], **NATIVE})
if os.environ.get("D96_PHASE") == "r50_native_pertarget":
    CFG.update({"version": "d96_r50_exp4b_native_pertarget_v1", "arch": "resnet50.tv_in1k", "epochs": 25,
                "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25], **NATIVE})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp 5a (pre-registered 2026-10-12): Exp 1e exactly (d96 cache, per-target labels, schedule, seed, SWA), but the r50
# backbone starts from the continued self-supervised checkpoint (scripts/ssl_barlow_d96.py: Barlow Twins ImageNet r50,
# then Barlow Twins on training-fold d96 windows, final epoch) instead of tv_in1k. Pre-registration in the ledger.
if os.environ.get("D96_PHASE") == "r50_ssl_pertarget":
    CFG.update({"version": "d96_r50_exp5a_ssl_pertarget_v1", "arch": "resnet50", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25], "backbone_init": os.environ["D96_BACKBONE"]})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp v3 (pre-registered 2026-10-12; docs/exp_v3_spatial_finding_attention.md revision 3): Exp 1e exactly (labels,
# seed 2026, schedule, SWA 21-25), but the r50 runs at output stride 16 (dilated layer 4, same tv_in1k tensors) and
# each finding also pools the layer-4 cells of every window with its own softmax over valid cells (score MLP on a
# per-cell LayerNorm, last layer 0); the pooled vector gets its own LayerNorm and classifier U (init 0), added to the
# unchanged v2 path (GAP -> LN -> per-finding window attention). At step 0 v3 equals C0. C0 (r50_os16_gap) = the same
# class with the spatial path off: run only if v3 passes conditions 1-2 (attribution, not a gate).
# Exp v3-OS32 (pre-registered 2026-10-13, factorial attribution after v3 tied 1e): v3 exactly but at output stride 32
# (1e's backbone; 12 x 12 training / 10 x 10 eval cells): spotlights without the finer grid. C0 = finer grid without them.
if os.environ.get("D96_PHASE") in ("r50_spatial", "r50_os16_gap", "r50_spatial_os32"):
    v3 = os.environ["D96_PHASE"] != "r50_os16_gap"
    os32 = os.environ["D96_PHASE"] == "r50_spatial_os32"
    CFG.update({"version": ("d96_r50_v3_spatial_os32_v1" if os32 else "d96_r50_v3_spatial_v1") if v3
                else "d96_r50_c0_os16_gap_v1", "arch": "resnet50.tv_in1k",
                "epochs": 25, "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25],
                "spatial": {"output_stride": 32 if os32 else 16, "on": v3, "hidden": 256}})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp v3.1 (pre-registered 2026-10-13, after v3-OS32 kept MCL without v3's losses): v3-OS32 exactly (labels, seed,
# schedule, SWA 21-25, undilated r50) with three changes to the spotlight path: (1) the window attention also reads each
# finding's spotlight vectors (V, init 0: starts as the GAP-only choice); (2) 4 spotlights per finding (U small random
# so the heads can diverge); (3) spotlights over a stride-16 grid of [layer 3, layer 4 upsampled 2x] (fine cells
# without dilating layer 4).
# v2 of the setup (2026-10-13, after the first launch was stopped at epoch 3 as a setup error: U ~ N(0, 0.02) put noise
# into every logit and V re-chose windows from still-uniform spotlights): U = 0 as in v3; the heads are separated by a
# small random last score layer (std 0.01) instead; the V term is off (V stays 0) until epoch v_start.
if os.environ.get("D96_PHASE") == "r50_spatial_v31":
    CFG.update({"version": "d96_r50_v31_spatial_v2", "arch": "resnet50.tv_in1k",
                "epochs": 25, "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25],
                "spatial": {"output_stride": 32, "cell_stride": 16, "on": True, "hidden": 256, "heads": 4,
                            "v31": True, "score_init_std": 0.01, "v_start": 4}})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp B+D (pre-registered 2026-10-13): Exp 1e exactly (per-target labels, seed 2026, schedule, SWA 21-25, 16 / 94
# windows), plus (B) a second complete head (own LayerNorm, window attention, classifier) trained on the D2 labels on
# the same bags, loss = mean of the two heads' weighted BCE, prediction = mean of the two heads' probabilities; and
# (D) attention-guided bags: epochs 1-3 stratified random as 1e; before epochs 4, 7, ..., 25 the current model
# (eval mode, no gradients, 320 centre crop, every second of the 94 eval windows) scores every training study, and
# each finding's top window under the two heads' mean attention is kept; a training bag is then the deduplicated top
# windows (<= 12) + stratified random windows up to 16. Attention picks windows only; labels are untouched.
if os.environ.get("D96_PHASE") == "r50_dual_guided":
    CFG.update({"version": "d96_r50_dual_guided_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25],
                "dual": {"labels2": "soft_labels_hpo_v3_1_d2_structured.json"},
                "guide": {"start": 4, "every": 3, "res": 320, "stride": 2}})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134",
                     "labels2": "38367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp P1-r949 (pre-registered 2026-10-13, Phase 1 step 1): Exp 1e exactly (per-target labels, seed 2026, 25-epoch
# cosine, SWA 21-25, effective batch 8, 16 / 94 windows) with the public 0.949 recipe's training settings (forum
# 746792 / 3537858): (1) per-target pos_weight = clip((1 - p) / p, 1, 10), p = training-fold weighted label mean
# clipped to [0.03, 0.7]; (2) learning rates backbone 4e-5 / head 1.95e-3; (3) intensity jitter only (gain +-10%):
# no mirror, rotation, scale, shift or gamma.
if os.environ.get("D96_PHASE") == "r50_r949":
    CFG.update({"version": "d96_r50_p1_r949_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25], "lr": 4e-5, "head_lr": 1.95e-3,
                "aug": {"mirror_p": 0.0, "rot_deg": 0.0, "scale": 0.0, "shift": 0.0, "gamma": 0.0, "gain": 0.10},
                "pos_weight": {"p_clip": [0.03, 0.7], "w_clip": [1.0, 10.0]}})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp P1-r949b (pre-registered 2026-10-13, launched while P1-r949 was finishing; P1-r949 plateaued from epoch 13 with
# train loss still falling): P1-r949 exactly (pos_weight, LR 4e-5 / 1.95e-3) but with 1e's augmentation restored
# (mirror 0.5, rotation 10 deg, scale 0.10, shift 0.06, gamma 0.25, gain 0.10). r949 vs r949b isolates augmentation.
if os.environ.get("D96_PHASE") == "r50_r949_aug":
    CFG.update({"version": "d96_r50_p1_r949b_aug_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25], "lr": 4e-5, "head_lr": 1.95e-3,
                "pos_weight": {"p_clip": [0.03, 0.7], "w_clip": [1.0, 10.0]}})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp P1-cnx (pre-registered 2026-10-13, Phase 1 step 2): Exp 1e exactly (per-target labels, seed 2026, LR 1e-4 /
# 1e-3, wd 0.05, 25-epoch cosine, SWA 21-25, batch 8, 16 / 94 windows, 1e augmentation, plain soft BCE), only the
# backbone changes: ConvNeXt-T, ImageNet-22k pretrained, fine-tuned on 1k at 384 (timm default drop path 0). No
# BatchNorm, so the post-SWA BN refresh is a no-op.
if os.environ.get("D96_PHASE") == "cnx_pertarget":
    CFG.update({"version": "d96_cnxt_p1_pertarget_v1", "arch": "convnext_tiny.fb_in22k_ft_in1k_384", "epochs": 25,
                "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25]})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp P1-aug2 (pre-registered 2026-10-13, Phase 1 step 3, run alongside P1-cnx): Exp 1e exactly (r50 tv_in1k, labels,
# seed, LR, schedule, SWA 21-25, plain soft BCE, mirror / shift / gamma / gain) with stronger geometry and slice-spacing
# jitter in training only: rotation +-20 deg (1e 10), scale +-15% (1e 10%); each training window independently uses
# slice gap 1 (c-1, c, c+1; p 0.5) or 2 (c-2, c, c+2). Evaluation and BN refresh unchanged (gap 1).
if os.environ.get("D96_PHASE") == "r50_aug2":
    CFG.update({"version": "d96_r50_p1_aug2_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25],
                "aug": {**CFG["aug"], "rot_deg": 20.0, "scale": 0.15}, "slice_gap": {"gaps": [1, 2], "p": [0.5, 0.5]}})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp P1-cnx2 (pre-registered 2026-10-14, Phase 1): P1-cnx (ConvNeXt-T on the exact 1e recipe) overfitted after
# epoch 16 (train loss 0.31 vs 1e 0.38, val12 flat). Two ConvNeXt fine-tuning standards against that, nothing else:
# layer-wise LR decay 0.8 over 8 depth groups (stem, stage 0, stage 1, stage 2 blocks 0-2 / 3-5 / 6-8, stage 3,
# head norm; top group at the 1e backbone LR 1e-4, stem 0.8^7 x 1e-4 = 2.1e-5) and stochastic depth (drop path) 0.1.
if os.environ.get("D96_PHASE") == "cnx_tune":
    CFG.update({"version": "d96_cnxt_p1_tune_v1", "arch": "convnext_tiny.fb_in22k_ft_in1k_384", "epochs": 25,
                "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25], "layer_decay": 0.8, "drop_path": 0.1})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp CK-A / CK-B (pre-registered 2026-10-16, run in parallel): P1-cnx exactly (1e recipe, same labels, seed, LR,
# schedule, SWA 21-25, drop path 0), only the pretrained checkpoint changes (same ConvNeXt-T architecture, ImageNet
# mean / std): CK-A = timm ImageNet-12k pretrained, fine-tuned on 1k at 384 (top-1 85.1 vs 84.1); CK-B = DINOv3
# self-supervised on LVD-1689M, distilled from ViT-7B, no ImageNet fine-tune (feature backbone).
if os.environ.get("D96_PHASE") in ("cnx_in12k", "cnx_dinov3"):
    CFG.update({"version": {"cnx_in12k": "d96_cnxt_ck_in12k_v1", "cnx_dinov3": "d96_cnxt_ck_dinov3_v1"}[os.environ["D96_PHASE"]],
                "arch": {"cnx_in12k": "convnext_tiny.in12k_ft_in1k_384",
                         "cnx_dinov3": "convnext_tiny.dinov3_lvd1689m"}[os.environ["D96_PHASE"]],
                "epochs": 25, "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25]})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp P1-sev (pre-registered 2026-10-14, Phase 1): Exp 1e exactly plus auxiliary cumulative severity heads. For each
# finding, two extra logits (>=moderate, >=severe) read the same per-finding attention-pooled feature as the main
# logit; masked BCE on report-derived targets (severity_aux_targets.json, training folds only), aux weight 0.3. The
# main head, its labels, the validation labels and inference are unchanged; predictions are the main head only.
if os.environ.get("D96_PHASE") == "r50_sev":
    CFG.update({"version": "d96_r50_p1_sev_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25], "sev": {"aux": 0.3, "file": "severity_aux_targets.json"}})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134",
                     "sev": "2631ce41178e21eb597f5139452a97dc7bb5bc6d0e9849b45db4abce5bb48e77"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp P1-tags (pre-registered 2026-10-16, text arm A): Exp 1e exactly plus 6 auxiliary logits for report lexicon tags
# (T-pre3's kept tags). Each tag logit reads the mean of its related findings' attention-pooled features; masked BCE on
# tag_aux_targets.json (training folds, en / es / de / fr / nl reports only), aux weight 0.3. The main head, its labels,
# the validation labels and inference are unchanged; predictions are the main head only.
if os.environ.get("D96_PHASE") == "r50_tags":
    CFG.update({"version": "d96_r50_p1_tags_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25], "tags": {"aux": 0.3, "file": "tag_aux_targets.json", "anchors": [
                    [2, 3],      # bucket handle -> Medial, Lateral Meniscus
                    [6],         # chondromalacia -> PF OA
                    [4, 5, 6],   # osteophyte -> Medial, Lateral, PF OA
                    [4, 5, 6],   # loose body -> Medial, Lateral, PF OA
                    [2, 4],      # meniscal extrusion -> Medial Meniscus, Medial OA
                    [2, 3]]}})   # parameniscal cyst / ganglion -> Medial, Lateral Meniscus
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134",
                     "tags": "c1cf8b316049efb2073be7910e379b080f6042af8d21e9b6b00f7cf4b567827e"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp P1-shift (pre-registered 2026-10-14, Phase 1): Exp 1e exactly plus a residual slice shift (TSM, Lin et al. 2019)
# in every r50 bottleneck: before conv1, channels [0, C/8) come from the window d slices earlier and [C/8, C/4) from
# the window d slices later in the same series slot (zeros if absent); the identity path is untouched and no
# parameters are added, so feature maps mix across neighbouring slices at every depth. Training windows are
# contiguous runs at stride d inside each slot ([4, 4, 3, 2, 3] = 16, the slot shares of 1e's stratified draw)
# instead of 16 stratified centres; evaluation keeps 1e's 94 centres and looks neighbours up by slice distance d.
# The sagittal mirror reverses slice order, so it also swaps earlier / later neighbours.
if os.environ.get("D96_PHASE") == "r50_shift":
    CFG.update({"version": "d96_r50_p1_shift_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25], "shift": {"d": 3, "fold": 8, "runs": [4, 4, 3, 2, 3]}})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
# Exp 7a / 7b (pre-registered 2026-10-14): Exp 1e exactly (recipe, seed 2026, schedule, SWA 21-25), trained on label-v3
# (scripts/build_rulecap_labels.py): the teammate's label-v2 (LB 0.937) with rule-state caps, downward only, from the
# HPO-v3.1 rule extractor (regex, no LLM): below-threshold (BELOW) cells -> min(label, 0.35), hedged (HEDGE) cells ->
# min(label, 0.40), on all 12 targets (7a, r50_rulecap) or with Effusion left as label-v2 (7b, r50_rulecap_noeff).
_rulecap = {"r50_rulecap": ("85f0ff10dd1b7c928e70d77be1ccdd6aff33fbed9995e7008b1a454c07b7921b",
                            "soft_labels_labelv3_rulecap.json"),
            "r50_rulecap_noeff": ("263dd548c6fb432be153b644378ae53fcf53568abcd7502eba7fce90c5984c29",
                                  "soft_labels_labelv3_rulecap_noeff.json")}.get(os.environ.get("D96_PHASE"))
if _rulecap:
    CFG.update({"version": f"d96_{os.environ['D96_PHASE']}_v1", "arch": "resnet50.tv_in1k", "epochs": 25,
                "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25]})
    CFG["sha256"] = {**CFG["sha256"], "labels": _rulecap[0]}
    LABELS_FILE = _rulecap[1]
# Phase 3 teachers (pre-registered in the ledger before launch): Exp 1e (r50_teacher{k}) or P1-cnx (cnx_teacher{k})
# exactly - labels, seed, LR, schedule, augmentation, SWA 21-25 - trained on the training studies outside teacher
# split k (scripts/build_teacher_splits.py: 4 splits of folds 1-5, report-duplicate groups kept together), so split k
# and fold 0 are out of fold. The SWA weights predict split k (oof_swa.json), fold 0 and gold. Fold 0 is scored at
# epochs 5 / 10 / 15 / 20 / 25 only (monitoring; teachers are not selected on it); no mirror TTA or best-epoch outputs.
# P3 v2 (pre-registered 2026-10-16, ledger P3v2-T): cnxa_teacher{k} = CK-A exactly (ConvNeXt-T in12k checkpoint, 1e
# recipe) on label-v2 (teammate, sha dd0ab9ab...) with the same splits and outputs as the teachers above.
_teacher = next((p for p in ("r50", "cnx", "cnxa") for k in range(4) if os.environ.get("D96_PHASE") == f"{p}_teacher{k}"), None)
if _teacher:
    CFG.update({"version": f"d96_{_teacher}_teacher_v1", "epochs": 25, "lr_floor": 0.01,
                "arch": {"r50": "resnet50.tv_in1k", "cnx": "convnext_tiny.fb_in22k_ft_in1k_384",
                         "cnxa": "convnext_tiny.in12k_ft_in1k_384"}[_teacher],
                "swa_epochs": [21, 22, 23, 24, 25], "eval_epochs": [5, 10, 15, 20, 25],
                "teacher": {"split": int(os.environ["D96_PHASE"][-1]), "file": "teacher_splits.csv"}})
    CFG["sha256"] = {**CFG["sha256"], "labels": "d40803d0b6347a7beb346f55476b948ce7ed7a868c91d560b3d81c95758a3134",
                     "teacher": "7cb597a737bc0799ff652a8c4f6f2672c17d9a6448590a1034004fb201669340"}
    LABELS_FILE = "soft_labels_final_per_target_any_model.json"
    if _teacher == "cnxa":
        CFG["sha256"]["labels"] = "dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d"
        LABELS_FILE = "soft_labels_final_per_target_latD2.json"
# P3 v2 students (pre-registered 2026-10-16, ledger P3v2-S): Exp 1e exactly (r50 tv_in1k, seed 2026, LRs, schedule,
# SWA 21-25, augmentation) on label-v2 mixed with the cross-fitted CK-A teachers (scripts/build_student_labels_v2.py):
# r50_student_state = alpha by report rule state (SIL 0.7; BELOW / BORD / HEDGE / LIKELY 0.5; POS / NEG 0.2);
# r50_student_a050 = alpha 0.5 everywhere (control). Label-file hashes pinned after the files are built, before launch.
_student = {"r50_student_state": ("soft_labels_student_v2_state.json", "7459684bb823c692a4002580fdc9a888e71a0289c3a48a5708d830837c2e8e68"),
            "r50_student_a050": ("soft_labels_student_v2_a050.json", "0c767ac9f783b4ac19cffbe5f76319e13e3ee00cf38b800cf32f5e18249530ca")}.get(os.environ.get("D96_PHASE"))
if _student:
    assert _student[1], "student label hash not pinned yet"
    CFG.update({"version": f"d96_{os.environ['D96_PHASE']}_v1", "arch": "resnet50.tv_in1k", "epochs": 25,
                "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25]})
    CFG["sha256"] = {**CFG["sha256"], "labels": _student[1]}
    LABELS_FILE = _student[0]
# Exp CW1 (pre-registered 2026-10-17): Exp 1e exactly on label-v2 (r50 tv_in1k, seed 2026, LRs, schedule, SWA 21-25,
# augmentation), with the head replaced by a cross-window reader with no gate (the public 0.956 notebook's ConvNeXt
# reader; Exp 2a's zero gate never opened): each window's LayerNorm feature -> Dropout 0.2 -> Linear 2048 -> 384 -> LN,
# + slot embedding + Exp 2a's anatomical position, -> 2-layer pre-norm transformer (8 heads, FFN 768, dropout 0.1) ->
# LN; the per-finding attention pooling and classifier read these 384-d tokens (no path around the transformer).
if os.environ.get("D96_PHASE") == "r50_tok":
    CFG.update({"version": "d96_r50_cw1_tok_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25],
                "tok": {"dim": 384, "layers": 2, "heads": 8, "ffn": 768, "dropout": 0.1, "proj_dropout": 0.2,
                        "freqs": 4, "multi_passes": 6}})
    CFG["sha256"] = {**CFG["sha256"], "labels": "dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d"}
    LABELS_FILE = "soft_labels_final_per_target_latD2.json"
# Exp r50-GL (pre-registered 2026-10-18): the label-v2 r50 recipe (Exp 1e on label-v2: r50 tv_in1k, seed 2026, LRs,
# schedule, SWA 21-25, augmentation) as the base of scripts/r50_gl_runner.py (global windows + per-finding crops).
if os.environ.get("D96_PHASE") == "r50_gl":
    CFG.update({"version": "d96_r50_gl_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25], "eval_epochs": [5, 10, 15, 20, 25]})
    CFG["sha256"] = {**CFG["sha256"], "labels": "dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d"}
    LABELS_FILE = "soft_labels_final_per_target_latD2.json"
# Exp MS1 (pre-registered 2026-10-17): MedSigLIP-448's vision tower (google/medsiglip-448 rev 9cea28a1, HAI-DEF terms;
# SoViT-400M/14, 27 blocks, MAP-pooled 1152-d) fine-tuned in the label-v2 recipe: label-v2, seed 2026, 25-epoch cosine
# with a 1% floor, SWA 21-25, wd 0.05, head LR 1e-3, 16 / 94 windows, 1e augmentation, plain soft BCE. Backbone-specific
# changes only: native 448 px input (training: the 1e affine resamples the 384 px / 140 mm crop to 448; evaluation: the
# whole 384 px crop resized to 448, the same mm per pixel as training, instead of the CNNs' 320 centre crop), SigLIP
# normalisation (mean 0.5, std 0.5), ViT fine-tuning standards (top backbone LR 5e-5, layer-wise decay 0.9 over 29 depth
# groups, drop path 0.1), fold 0 scored at epochs 5/10/15/20/25 and after SWA. Gradient checkpointing off (user
# 2026-10-17: ~25% faster, ~35-45 GB activations est.); on an out-of-memory error at step 1, relaunch with it on. Weights converted
# to timm by `modal_coatnet_d96.py --medsiglip-weights-run` (safetensors, strict load, parity vs transformers).
if os.environ.get("D96_PHASE") == "sig_lv2":
    CFG.update({"version": "d96_medsiglip448_lv2_v1", "arch": "vit_so400m_patch14_siglip_384", "train_res": 448,
                "eval_res": 448, "eval_resize": True, "norm": [0.5, 0.5], "epochs": 25, "lr": 5e-5, "layer_decay": 0.9,
                "drop_path": 0.1, "grad_ckpt": False, "lr_floor": 0.01, "swa_epochs": [21, 22, 23, 24, 25],
                "eval_epochs": [5, 10, 15, 20, 25], "backbone_init": "/root/vol/pretrained/medsiglip448_timm.pt"})
    CFG["sha256"] = {**CFG["sha256"], "labels": "dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d",
                     "backbone": "276657d2256b4f258d77f4e7aabdd81c8f1b56988cc71edb83b6598ee8dcee43"}   # conversion 2026-10-17
    LABELS_FILE = "soft_labels_final_per_target_latD2.json"
# Exp v4 baseline (pre-registered 2026-10-18, ledger r50-GL4): label-v2 r50 retrained by us (the 0.9425 bar is the
# teammate's run; its training config was never recorded): the r50_gl CFG without eval_epochs, so fold 0 is scored every
# epoch (v4's early-stop rule reads epochs 10-12). Seeds via D96_SEED (seed noise of the label-v2 recipe).
if os.environ.get("D96_PHASE") == "r50_lv2":
    CFG.update({"version": "d96_r50_lv2_v1", "arch": "resnet50.tv_in1k", "epochs": 25, "lr_floor": 0.01,
                "swa_epochs": [21, 22, 23, 24, 25]})
    CFG["sha256"] = {**CFG["sha256"], "labels": "dd0ab9ab503c873b13b7a94c50630e6a8839c600f91b6e7eb461a8d914ed991d"}
    LABELS_FILE = "soft_labels_final_per_target_latD2.json"
if os.environ.get("D96_SEED"):                         # seed override for any phase; the version records it (recovery)
    CFG.update({"seed": int(os.environ["D96_SEED"]), "version": f"{CFG['version']}_seed{int(os.environ['D96_SEED'])}"})
NATIVE_LAYOUT = CFG.get("layout") == "native"
INIT = Path(os.environ.get("D96_INIT", "/root/vol/experiments/coatnet_d96_fold0/run"))
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
MEAN = torch.tensor([CFG["norm"][0]] * 3 if "norm" in CFG else [0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([CFG["norm"][1]] * 3 if "norm" in CFG else [0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SLOT_OF = None if NATIVE_LAYOUT else np.repeat(np.arange(5), CFG["slot_sizes"])   # slice index -> slot (d96)
SAGITTAL_SLOTS = (0, 1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def teacher_ids(train_ids, val_ids):
    """Phase 3: (teacher training ids, out-of-fold ids) for CFG['teacher']['split']; disjoint, fold 0 excluded."""
    tp = ASSETS / CFG["teacher"]["file"]
    assert sha(tp) == CFG["sha256"]["teacher"], f"{tp} hash differs"
    split = {r["study_id"]: int(r["teacher_split"]) for r in csv.DictReader(tp.open())}
    assert set(split) == set(train_ids), "teacher splits must cover exactly the training folds"
    k = CFG["teacher"]["split"]
    oof = sorted(s for s in train_ids if split[s] == k)
    fit = sorted(s for s in train_ids if split[s] != k)
    assert oof and fit and not set(oof) & set(fit) and not set(val_ids) & (set(oof) | set(fit))
    return fit, oof


def load_labels(path: Path, key_sha: str):
    assert sha(path) == CFG["sha256"][key_sha], f"{path} hash differs"
    d = json.loads(path.read_text())
    assert d["targets_in_order"] == TARGETS
    return {r["study_id"]: (np.array([0.5 if p is None else p for p in r["probabilities"]], np.float32),
                            np.array(r["weights"], np.float32)) for r in d["labels"]}


def border_labels(labels):
    """Exp 3a: D2 borderline cells of the BORDER targets -> BORDER['to']; weights unchanged."""
    b = CFG["border"]; cols = [TARGETS.index(t) for t in b["targets"]]
    out, n = {}, np.zeros(12, int)
    for s, (y, w) in labels.items():
        y = y.copy()
        for k in cols:
            if b["lo"] < y[k] < b["hi"]:
                y[k] = b["to"]; n[k] += 1
        out[s] = (y, w)
    print("BORDER cells moved per target (all studies; only training folds are trained on): "
          + json.dumps({TARGETS[k]: int(n[k]) for k in cols}), flush=True)
    return out


# ---------------------------------------------------------------- data
def read_manifest():
    studies = {}
    for i in range(CFG.get("shards", SHARDS)):
        root = CACHE / f"shard_{i}"
        for line in (root / "manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            assert r["study_id"] not in studies
            if NATIVE_LAYOUT:                                    # variable length; per-row slot; no 3D positions
                n = r["n"]
                assert len(r["mask"]) == n == len(r["slot_of"]) and n >= 3, r["study_id"]
                studies[r["study_id"]] = {"file": str(root / r["part"]), "row": r["row_start"], "n": n,
                                          "mask": np.array(r["mask"], bool), "gold": r["gold"],
                                          "slot": np.array(r["slot_of"], np.int64),
                                          "pos": np.full(n, 0.5, np.float32), "known": np.zeros(n, np.float32)}
                continue
            pos, known = positions(r)
            studies[r["study_id"]] = {"file": str(root / r["part"]), "row": r["row_start"],
                                      "mask": np.array(r["mask"], bool), "gold": r["gold"], "pos": pos, "known": known}
    return studies


def centres(mask: np.ndarray, k: int, train: bool, offset: float | None = None) -> np.ndarray:
    """Their _eval_centers over the valid range (lo, hi); training draws one centre per 1/k of that range.
    offset (Exp 2a secondary readout): the stratified draw with a fixed place in each 1/k bin instead of random."""
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, len(mask) - 2))])
    if not train:
        if k is None:                                            # native layout eval: one window per slice
            return cs
        return cs[np.linspace(0, len(cs) - 1, k).round().astype(int)]
    edges = np.linspace(0, len(cs), k + 1)
    u = np.random.rand(k) if offset is None else np.full(k, offset)
    idx = np.minimum((edges[:-1] + u * (edges[1:] - edges[:-1])).astype(int), len(cs) - 1)
    return cs[idx]


def alloc_centres(mask: np.ndarray, k: int, train: bool) -> np.ndarray:
    """Exp 6b: windows per series slot (CFG["alloc"] = [[slot, n], ...]), each kept inside its slot's valid range.
    Training: n stratified random centres per slot; eval (k = eval_windows): every interior centre of the slots;
    otherwise (BN refresh, k = train_windows): the allocation at fixed mid-bin offsets."""
    bounds = np.concatenate([[0], np.cumsum(CFG["slot_sizes"])])
    cs = {}
    for s, _ in CFG["alloc"]:
        valid = np.where(mask[bounds[s]:bounds[s + 1]])[0] + bounds[s]
        if len(valid) >= 3:
            cs[s] = np.arange(valid.min() + 1, valid.max())
    if not cs:
        return centres(mask, k, train)
    if not train and k == CFG["eval_windows"]:
        return np.concatenate([cs[s] for s, _ in CFG["alloc"] if s in cs])
    n = {s: c for s, c in CFG["alloc"] if s in cs}
    spare = k - sum(n.values())
    order = [s for s, _ in CFG["alloc"] if s in cs]
    for j in range(spare):                                       # missing slots' windows, round-robin in order
        n[order[j % len(order)]] += 1
    out = []
    for s in order:
        c, m = cs[s], n[s]
        edges = np.linspace(0, len(c), m + 1)
        u = np.random.rand(m) if train else np.full(m, 0.5)
        out.append(c[np.minimum((edges[:-1] + u * (edges[1:] - edges[:-1])).astype(int), len(c) - 1)])
    return np.concatenate(out)


def run_centres(mask: np.ndarray, k: int, train: bool) -> np.ndarray:
    """Exp P1-shift: contiguous runs at stride d inside each slot (CFG["shift"]["runs"] windows per slot), centres
    kept within the slot's valid interior. A slot too short for its run takes what fits; the shortfall goes to
    the slots with room, in slot order; anything still missing is drawn as 1e's stratified centres. Training: random
    run starts; otherwise (BN refresh) the run is centred in the slot."""
    d = CFG["shift"]["d"]
    bounds = np.concatenate([[0], np.cumsum(CFG["slot_sizes"])])
    span = {}
    for s in range(5):
        valid = np.where(mask[bounds[s]:bounds[s + 1]])[0] + bounds[s]
        if len(valid) >= 3:
            span[s] = (valid.min() + 1, valid.max() - 1)
    cap = {s: 1 + (hi - lo) // d for s, (lo, hi) in span.items()}
    n = {s: min(CFG["shift"]["runs"][s], cap[s]) for s in span}
    short = k - sum(n.values())
    while short > 0 and any(n[s] < cap[s] for s in n):
        for s in sorted(n):
            if short > 0 and n[s] < cap[s]:
                n[s] += 1; short -= 1
    out = []
    for s in sorted(n):
        lo, hi = span[s]
        room = hi - lo - (n[s] - 1) * d
        start = lo + (np.random.randint(room + 1) if train else room // 2)
        out.append(start + d * np.arange(n[s]))
    if short > 0:
        out.append(centres(mask, short, train))
    return np.concatenate(out) if out else centres(mask, k, train)


def neighbours(c: np.ndarray, slot_of: np.ndarray) -> np.ndarray:
    """Exp P1-shift: [k, 2] local index of the window d slices earlier / later in the same slot, -1 if none."""
    d = CFG["shift"]["d"]
    first = {}
    for i, ci in enumerate(c):
        first.setdefault(int(ci), i)
    nb = np.full((len(c), 2), -1, np.int64)
    for i, ci in enumerate(c):
        for j, cj in enumerate((ci - d, ci + d)):
            q = first.get(int(cj), -1)
            if q >= 0 and slot_of[cj] == slot_of[ci]:
                nb[i, j] = q
    return nb


class Studies(torch.utils.data.Dataset):
    def __init__(self, ids, meta, labels, k, train, offset=None):
        self.ids, self.meta, self.labels, self.k, self.train = list(ids), meta, labels, k, train
        self.offset = offset
        self.guide = {}                                          # Exp B+D: study -> top centre per finding [12]
        self._mm = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        s = self.ids[i]
        m = self.meta[s]
        if m["file"] not in self._mm:
            self._mm[m["file"]] = np.load(m["file"], mmap_mode="r")
        length = m["n"] if NATIVE_LAYOUT else CFG["maxs"]
        slot_of = m["slot"] if NATIVE_LAYOUT else SLOT_OF          # native slots: 0-1 sagittal, 2 coronal, 3 axial
        vol = self._mm[m["file"]][m["row"]:m["row"] + length]
        if "alloc" in CFG:
            c = np.clip(alloc_centres(m["mask"], self.k, self.train), 1, length - 2)
        elif "shift" in CFG and self.k != CFG["eval_windows"]:         # Exp P1-shift: runs (training, BN refresh)
            c = np.clip(run_centres(m["mask"], self.k, self.train), 1, length - 2)
        elif self.train and s in self.guide:                     # Exp B+D: top windows + stratified random
            top = np.unique(self.guide[s])
            c = np.clip(np.concatenate([top, centres(m["mask"], self.k - len(top), True)]), 1, length - 2)
        else:
            c = np.clip(centres(m["mask"], self.k, self.train or self.offset is not None, self.offset), 1, length - 2)
        if self.train and "slice_gap" in CFG:                    # Exp P1-aug2: per-window slice gap 1 or 2
            sg = CFG["slice_gap"]
            g = np.random.choice(sg["gaps"], size=len(c), p=sg["p"])
            c = np.clip(c, g, length - 1 - g)
        else:
            g = 1
        self._c = c                                              # V4-IMP MC-PAIR: the drawn centres (read by GLStudies6)
        win = np.ascontiguousarray(vol[np.stack([c - g, c, c + g], 1)])           # [k, 3, H, W] uint8
        sag = np.isin(slot_of[c], SAGITTAL_SLOTS)
        y, w = self.labels.get(s, (np.zeros(12, np.float32), np.zeros(12, np.float32)))
        ctx = np.stack([m["pos"][c], m["known"][c], slot_of[c].astype(np.float32)], 1)   # [k, 3]
        if "shift" in CFG:                                       # Exp P1-shift: + earlier / later neighbour [k, 5]
            ctx = np.concatenate([ctx, neighbours(c, slot_of).astype(np.float32)], 1)
        return torch.from_numpy(win), torch.from_numpy(sag), torch.from_numpy(y), torch.from_numpy(w), torch.from_numpy(ctx)


FLIP = {"windows": None}                 # Exp P1-shift: per-window mirror flags of the last prepare() / mirror TTA


def mirror(x, sag):
    """Their anatomical L/R mirror: sagittal windows reverse the 3 slice channels, others flip width."""
    return torch.where(sag.view(-1, 1, 1, 1), x.flip(1), x.flip(-1))


def prepare(x, sag, train, res, device, valid=None):
    """x uint8 [B, K, 3, H, W] -> normalised float [B*K, 3, res, res].
    valid (Exp v3): a module whose .valid is set to the [B*K, res/OS, res/OS] bool mask of feature cells that are
    mostly image after the training affine (a ones-image warped by the same grid, cell-averaged, > 0.5)."""
    b, k = x.shape[:2]
    x = x.to(device, non_blocking=True).flatten(0, 1)
    sag = sag.to(device).flatten()
    a = CFG["aug"]
    FLIP["windows"] = None
    if train:
        flip = (torch.rand(b, device=device) < a["mirror_p"]).repeat_interleave(k)
        x = torch.where(flip.view(-1, 1, 1, 1), mirror(x, sag), x)
        FLIP["windows"] = flip                                  # Exp P1-shift: mirrored sagittal swaps neighbours
    x = x.float().div_(255.0)
    if train:
        ang = (torch.rand(b, device=device) * 2 - 1) * math.radians(a["rot_deg"])
        sc = 1 + (torch.rand(b, device=device) * 2 - 1) * a["scale"]
        tx, ty = [(torch.rand(b, device=device) * 2 - 1) * a["shift"] * 2 for _ in range(2)]
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
        grid = F.affine_grid(theta.repeat_interleave(k, 0), (b * k, 3, res, res), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if valid is not None:
            os_ = CFG["spatial"].get("cell_stride", CFG["spatial"]["output_stride"])
            ones = F.grid_sample(torch.ones_like(x[:, :1]), grid, mode="bilinear", padding_mode="zeros",
                                 align_corners=False)
            valid.valid = F.avg_pool2d(ones, os_, os_)[:, 0] > 0.5
        gamma = torch.exp((torch.rand(b, device=device) * 2 - 1) * a["gamma"]).repeat_interleave(k).view(-1, 1, 1, 1)
        gain = (1 + (torch.rand(b, device=device) * 2 - 1) * a["gain"]).repeat_interleave(k).view(-1, 1, 1, 1)
        x = (x.clamp_min(1e-4).pow(gamma) * gain).clamp_(0, 1)
    elif CFG.get("eval_resize"):                                               # Exp MS1: whole crop -> ViT input
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    elif res != x.shape[-1]:
        lo = (x.shape[-1] - res) // 2                                          # their centre crop, no resize
        x = x[..., lo:lo + res, lo:lo + res]
    x = (x - MEAN.to(device)) / STD.to(device)
    return x.contiguous(memory_format=torch.channels_last)


# ---------------------------------------------------------------- model (their RaptorClassifier)
class Raptor(nn.Module):
    def __init__(self, res, pretrained):
        super().__init__()
        import timm
        size = {"img_size": res} if CFG["arch"].startswith("coatnet") else {}   # CNNs take any input size
        if CFG["arch"].startswith("vit"):                                       # Exp MS1: timm 1.0.29 names the
            size["img_size"] = CFG["train_res"]                                 # SoViT-400M/14 "_384"; built at 448
            if "siglip" in CFG["arch"]:
                size["act_layer"] = "gelu_tanh"                                 # SigLIP / HF gelu_pytorch_tanh
            else:                                                               # BB-CMP DINOv2 378 / 322, DINOv3 384 / 320
                size["dynamic_img_size"] = True
        if "spatial" in CFG:                                                    # Exp v3 / C0: dilated layer 4
            size["output_stride"] = CFG["spatial"]["output_stride"]
        init = CFG.get("backbone_init") if pretrained else None                 # Exp 5a: self-supervised start
        if "drop_path" in CFG:                                                  # Exp P1-cnx2: stochastic depth
            size["drop_path_rate"] = CFG["drop_path"]
        self.backbone = timm.create_model(CFG["arch"], pretrained=pretrained and not init, num_classes=0, **size)
        if init:
            if CFG["sha256"].get("backbone") is not None:                       # Exp MS1: pinned converted weights
                assert sha(Path(init)) == CFG["sha256"]["backbone"], "backbone weights differ from the pinned sha256"
            self.backbone.load_state_dict(torch.load(init, map_location="cpu", weights_only=True), strict=True)
        if CFG.get("grad_ckpt"):                                                # Exp MS1: ViT-400M activations
            self.backbone.set_grad_checkpointing(True)
        d = self.backbone.num_features
        self.norm = nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, d))
        self.clsb = nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.nbr, self.shift_off = None, False                                  # shift_off: preflight / timing only
        if "shift" in CFG:                                                      # Exp P1-shift: residual slice shift
            for layer in (self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4):
                for blk in layer:
                    blk.conv1.register_forward_pre_hook(self.slice_shift)

    def slice_shift(self, mod, args):
        """Exp P1-shift pre-hook on each bottleneck conv1: channels [0, f) from the earlier neighbour window,
        [f, 2f) from the later one (row N = zeros when absent), the rest unchanged. One clone (keeps channels_last)
        and a write of the shifted quarter; timed faster than full-tensor rebuilds and than row-slice selection."""
        if self.nbr is None:
            return None
        x = args[0]
        f = x.shape[1] // CFG["shift"]["fold"]
        xz = torch.cat([x[:, :2 * f], x.new_zeros(1, 2 * f, *x.shape[2:])])
        y = x.clone()
        y[:, :2 * f] = torch.cat([xz[self.nbr[:, 0], :f], xz[self.nbr[:, 1], f:]], 1)
        return (y,)

    def shift_index(self, ctx, b, n):
        """[n, 2] global row of each window's earlier / later neighbour (n = none); mirrored sagittal windows swap."""
        k = n // b
        loc = ctx[..., 3:5].round().long()                                      # [b, k, 2]
        off = torch.arange(b, device=loc.device).view(b, 1, 1) * k
        g = torch.where(loc >= 0, loc + off, torch.full_like(loc, n)).view(n, 2)
        flip = FLIP["windows"]
        if flip is not None:
            swap = flip.to(g.device) & (ctx[..., 2] <= 1).reshape(n)            # slots 0-1 = sagittal
            g = torch.where(swap[:, None], g.flip(1), g)
        return g

    def forward(self, x, b, ctx=None):
        if "shift" in CFG and not self.shift_off:
            assert ctx is not None and ctx.shape[-1] == 5, "Exp P1-shift needs neighbour columns in ctx"
            self.nbr = self.shift_index(ctx.to(x.device), b, x.shape[0])
        f = self.backbone(x).view(b, x.shape[0] // b, -1)
        self.nbr = None
        h = self.context(self.norm(f), ctx)
        a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)
        return (torch.einsum("bkn,bkf->bnf", a, h) * self.clsW).sum(-1) + self.clsb

    def context(self, h, ctx):
        return h


class RaptorCtx(Raptor):
    """Exp 2a: Raptor plus slot/position tokens and a transformer across windows, gated residual starting at 0."""

    def __init__(self, res, pretrained):
        super().__init__(res, pretrained)
        c, d = CFG["ctx"], self.backbone.num_features
        self.inp = nn.Linear(d, c["dim"])
        self.slot = nn.Embedding(5, c["dim"])
        self.pos = nn.Linear(2 * c["freqs"] + 1, c["dim"])
        layer = nn.TransformerEncoderLayer(c["dim"], c["heads"], c["ffn"], c["dropout"], activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, c["layers"], enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(c["dim"]), nn.Linear(c["dim"], d))
        self.gate = nn.Parameter(torch.zeros(d))
        nn.init.trunc_normal_(self.slot.weight, std=0.02)

    def context(self, h, ctx):
        pos, known, slot = ctx[..., 0], ctx[..., 1], ctx[..., 2].long()
        f = math.pi * torch.arange(1, CFG["ctx"]["freqs"] + 1, device=h.device, dtype=torch.float32)
        enc = torch.cat([torch.sin(pos[..., None] * f), torch.cos(pos[..., None] * f)], -1) * known[..., None]
        t = self.inp(h) + self.slot(slot) + self.pos(torch.cat([enc, known[..., None]], -1)).to(h.dtype)
        return h + self.gate.to(h.dtype) * self.out(self.enc(t))


class RaptorTok(Raptor):
    """Exp CW1: per-finding attention pooling and classifier on transformer tokens (no gate, no bypass)."""

    def __init__(self, res, pretrained):
        super().__init__(res, pretrained)
        c, d = CFG["tok"], self.backbone.num_features
        self.inp = nn.Sequential(nn.Dropout(c["proj_dropout"]), nn.Linear(d, c["dim"]), nn.LayerNorm(c["dim"]))
        self.slot = nn.Embedding(5, c["dim"])
        self.pos = nn.Linear(2 * c["freqs"] + 1, c["dim"])
        layer = nn.TransformerEncoderLayer(c["dim"], c["heads"], c["ffn"], c["dropout"], activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, c["layers"], norm=nn.LayerNorm(c["dim"]), enable_nested_tensor=False)
        self.att = nn.Sequential(nn.Linear(c["dim"], 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, c["dim"]))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        nn.init.trunc_normal_(self.slot.weight, std=0.02)

    def context(self, h, ctx):
        pos, known, slot = ctx[..., 0], ctx[..., 1], ctx[..., 2].long()
        f = math.pi * torch.arange(1, CFG["tok"]["freqs"] + 1, device=h.device, dtype=torch.float32)
        enc = torch.cat([torch.sin(pos[..., None] * f), torch.cos(pos[..., None] * f)], -1) * known[..., None]
        t = self.inp(h) + self.slot(slot) + self.pos(torch.cat([enc, known[..., None]], -1)).to(h.dtype)
        return self.enc(t)


PLANE_OF_SLOT = torch.tensor([0, 0, 1, 1, 2])                      # sagittal, coronal, axial


class RaptorPlane(Raptor):
    """Exp 2c: per-plane branches on the shared backbone. For each plane (sagittal / coronal / axial) and finding, a
    plane-specific attention pool over that plane's windows only and a plane-specific classifier; the main logit is
    the sum of the plane logits (= one linear head on the concatenated plane features). Each plane logit also gets
    an auxiliary BCE (CoPAS-style branch loss), so no plane can be left unused. A plane with no windows gives 0."""

    def __init__(self, res, pretrained):
        super().__init__(res, pretrained)
        d = self.backbone.num_features
        del self.att, self.clsW
        self.patt = nn.ModuleList([nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
                                   for _ in range(3)])
        self.pW = nn.Parameter(torch.zeros(3, 12, d)); self.auxW = nn.Parameter(torch.zeros(3, 12, d))
        self.auxb = nn.Parameter(torch.zeros(3, 12))
        nn.init.trunc_normal_(self.pW, std=0.02); nn.init.trunc_normal_(self.auxW, std=0.02)
        self.aux = None

    def forward(self, x, b, ctx=None):
        f = self.backbone(x).view(b, x.shape[0] // b, -1)
        h = self.norm(f)
        plane = PLANE_OF_SLOT.to(h.device)[ctx[..., 2].long()]                # [b, k]
        main, aux, present = 0, [], []
        for p in range(3):
            m = (plane == p)
            s = self.patt[p](h).float().masked_fill(~m[..., None], -1e4)
            a = (torch.softmax(s, dim=1) * m[..., None]).to(h.dtype)            # [b, k, 12]; all-zero if plane absent
            z = torch.einsum("bkn,bkf->bnf", a, h)                              # [b, 12, d]
            main = main + (z * self.pW[p]).sum(-1)
            aux.append((z * self.auxW[p]).sum(-1) + self.auxb[p]); present.append(m.any(1))
        self.aux = (torch.stack(aux, 1), torch.stack(present, 1))              # [b, 3, 12], [b, 3]
        return main + self.clsb


class RaptorSpatial(Raptor):
    """Exp v3 / C0: v2's path on the GAP of the stride-16 map, plus (v3 only) a per-finding pool over each window's
    cells: score MLP(LN_s(z)) with its last layer 0 -> softmax over valid cells -> LN_p -> classifier U (init 0), summed
    into each finding's window-attended logit. The new modules are always built, inside a forked CPU RNG, so v3 and C0
    consume the same RNG stream and C0 carries them unused. self.valid = training padding mask (None: all valid).
    self.stats accumulates the pre-registered diagnostics while self.collect is a key."""

    def __init__(self, res, pretrained):
        super().__init__(res, pretrained)
        d, hid = self.backbone.num_features, CFG["spatial"]["hidden"]
        with torch.random.fork_rng(devices=[]):
            self.norm_s = nn.LayerNorm(d)
            self.score = nn.Sequential(nn.Linear(d, hid), nn.Tanh(), nn.Dropout(0.2), nn.Linear(hid, 12))
            self.norm_p = nn.LayerNorm(d)
        nn.init.zeros_(self.score[-1].weight); nn.init.zeros_(self.score[-1].bias)
        self.clsU = nn.Parameter(torch.zeros(12, d))
        self.on, self.valid, self.collect, self.stats = CFG["spatial"]["on"], None, None, {}

    def forward(self, x, b, ctx=None):
        z = self.backbone.forward_features(x)                                   # [N, d, H, W]
        k = x.shape[0] // b
        h = self.norm(z.mean((2, 3))).view(b, k, -1)                           # v2 path: unmasked GAP
        a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)
        base = (torch.einsum("bkn,bkf->bnf", a, h) * self.clsW).sum(-1)
        if not self.on:
            return base + self.clsb
        z = z.flatten(2).transpose(1, 2)                                        # [N, cells, d]
        s = self.score(self.norm_s(z)).float()                                  # [N, cells, 12]
        valid = None if self.valid is None else self.valid.flatten(1)
        if valid is not None:
            assert valid.shape == s.shape[:2], (valid.shape, s.shape)
            s = s.masked_fill(~valid[..., None], -1e4)
        al = torch.softmax(s, dim=1)
        p = self.norm_p(torch.einsum("nct,ncf->ntf", al.to(z.dtype), z)).view(b, k, 12, -1)
        spatial = (torch.einsum("bkt,bktf->btf", a, p.to(a.dtype)) * self.clsU).sum(-1)
        out = base + spatial + self.clsb
        if self.collect is not None:
            with torch.no_grad():
                n = valid.sum(1).float().clamp_min(2) if valid is not None else torch.full((s.shape[0],), s.shape[1], device=s.device)
                ent = -(al * torch.log(al.clamp_min(1e-12))).sum(1) / torch.log(n)[:, None]   # [N, 12], 1 = uniform
                st = self.stats.setdefault(self.collect, {"ent": torch.zeros(12, dtype=torch.float64), "n": 0,
                                                           "sp": torch.zeros(12, dtype=torch.float64),
                                                           "lg": torch.zeros(12, dtype=torch.float64),
                                                           "b": 0, "valid": 0.0})
                st["ent"] += ent.double().sum(0).cpu(); st["n"] += ent.shape[0]
                st["sp"] += spatial.float().abs().double().sum(0).cpu()
                st["lg"] += out.float().abs().double().sum(0).cpu(); st["b"] += b
                st["valid"] += float(n.sum() / s.shape[1])
        return out

    def summary(self, key):
        """Per-finding mean normalised alpha entropy and mean |U.p| / mean |logit|; mean valid-cell fraction."""
        st = self.stats.get(key)
        if not st:
            return None
        ent, share = (st["ent"] / st["n"]).tolist(), (st["sp"] / st["lg"].clamp_min(1e-12)).tolist()
        return {"entropy": dict(zip(TARGETS, [round(v, 4) for v in ent])),
                "u_share": dict(zip(TARGETS, [round(v, 4) for v in share])),
                "valid_fraction": round(st["valid"] / st["n"], 4),
                "u_share_all": round(float(st["sp"].sum() / st["lg"].sum().clamp_min(1e-12)), 4),
                "branch_used": bool(sum(e < 0.95 for e in ent) >= 4
                                    and float(st["sp"].sum() / st["lg"].sum().clamp_min(1e-12)) >= 0.10)}


class RaptorSpatial31(Raptor):
    """Exp v3.1: the v2 path on the GAP of layer 4 (stride 32, undilated), plus per finding H spotlights over the
    stride-16 cells c = [layer 3, layer 4 upsampled 2x] (score MLP(LN_s(c)), last layer 0 -> softmax over valid cells
    per finding and head -> LN_p). Window attention logit = att(h) + sum_h V_nh . p_knh (V init 0); logit = W . sum_k a h
    + sum_k a sum_h U_nh . p_knh + b; U = 0, last score layer ~ N(0, 0.01) so the heads can diverge; V term off before
    CFG v_start. Diagnostics as v3 plus cross-finding
    and within-finding (head) spotlight overlap and how often the spotlights move the top window."""

    def __init__(self, res, pretrained):
        super().__init__(res, pretrained)
        sp = CFG["spatial"]
        d, hid, self.heads = self.backbone.num_features, sp["hidden"], sp["heads"]
        dc = d + self.backbone.layer3[-1].conv3.out_channels                     # 2048 + 1024
        self.norm_s = nn.LayerNorm(dc)
        self.score = nn.Sequential(nn.Linear(dc, hid), nn.Tanh(), nn.Dropout(0.2), nn.Linear(hid, 12 * self.heads))
        self.norm_p = nn.LayerNorm(dc)
        nn.init.normal_(self.score[-1].weight, std=sp["score_init_std"]); nn.init.zeros_(self.score[-1].bias)
        self.clsU = nn.Parameter(torch.zeros(12, self.heads, dc))
        self.clsV = nn.Parameter(torch.zeros(12, self.heads, dc))
        self.on, self.valid, self.collect, self.stats = True, None, None, {}
        self.v_on = True                                                          # training sets False before v_start

    def features(self, x):
        bb = self.backbone
        x = bb.maxpool(bb.act1(bb.bn1(bb.conv1(x))))
        z3 = bb.layer3(bb.layer2(bb.layer1(x)))
        return z3, bb.layer4(z3)

    def forward(self, x, b, ctx=None):
        z3, z4 = self.features(x)
        k, H = x.shape[0] // b, self.heads
        h = self.norm(z4.mean((2, 3))).view(b, k, -1)
        c = torch.cat([z3, F.interpolate(z4, size=z3.shape[-2:], mode="nearest")], 1).flatten(2).transpose(1, 2)
        s = self.score(self.norm_s(c)).float().view(c.shape[0], c.shape[1], 12, H)          # [N, cells, 12, H]
        valid = None if self.valid is None else self.valid.flatten(1)
        if valid is not None:
            assert valid.shape == s.shape[:2], (valid.shape, s.shape)
            s = s.masked_fill(~valid[..., None, None], -1e4)
        al = torch.softmax(s, dim=1)
        p = self.norm_p(torch.einsum("ncth,ncf->nthf", al.to(c.dtype), c)).view(b, k, 12, H, -1)
        g = self.att(h).float()
        a = torch.softmax(g + torch.einsum("bkthf,thf->bkt", p.float(), self.clsV.float()) if self.v_on else g, dim=1)
        base = (torch.einsum("bkn,bkf->bnf", a.to(h.dtype), h) * self.clsW).sum(-1)
        spatial = (torch.einsum("bkt,bkthf->bthf", a.to(p.dtype), p) * self.clsU).sum((-1, -2))
        out = base + spatial + self.clsb
        if self.collect is not None:
            with torch.no_grad():
                n = valid.sum(1).float().clamp_min(2) if valid is not None else torch.full((s.shape[0],), s.shape[1], device=s.device)
                ent = (-(al * torch.log(al.clamp_min(1e-12))).sum(1) / torch.log(n)[:, None, None]).mean(-1)  # [N, 12]
                m = al.mean(-1); m = m / m.norm(dim=1, keepdim=True)                              # [N, cells, 12]
                cf = torch.einsum("nct,ncu->ntu", m, m)
                ov_f = (cf.sum((1, 2)) - 12) / (12 * 11)                                          # [N]
                hn = al / al.norm(dim=1, keepdim=True)
                ch = torch.einsum("ncth,nctg->nthg", hn, hn)
                ov_h = (ch.sum((-1, -2)) - H) / (H * (H - 1))                                     # [N, 12]
                shift = (a.argmax(1) != g.argmax(1)).float()                                      # [b, 12]
                st = self.stats.setdefault(self.collect, {"ent": torch.zeros(12, dtype=torch.float64), "n": 0,
                                                           "sp": torch.zeros(12, dtype=torch.float64),
                                                           "lg": torch.zeros(12, dtype=torch.float64),
                                                           "b": 0, "valid": 0.0, "ov_f": 0.0,
                                                           "ov_h": torch.zeros(12, dtype=torch.float64),
                                                           "shift": torch.zeros(12, dtype=torch.float64)})
                st["ent"] += ent.double().sum(0).cpu(); st["n"] += ent.shape[0]
                st["sp"] += spatial.float().abs().double().sum(0).cpu()
                st["lg"] += out.float().abs().double().sum(0).cpu(); st["b"] += b
                st["valid"] += float(n.sum() / s.shape[1])
                st["ov_f"] += float(ov_f.sum()); st["ov_h"] += ov_h.double().sum(0).cpu()
                st["shift"] += shift.double().sum(0).cpu()
        return out

    def summary(self, key):
        out = RaptorSpatial.summary(self, key)
        if out:
            st = self.stats[key]
            out["overlap_findings"] = round(st["ov_f"] / st["n"], 4)
            out["overlap_heads"] = dict(zip(TARGETS, [round(v, 4) for v in (st["ov_h"] / st["n"]).tolist()]))
            out["window_shift"] = dict(zip(TARGETS, [round(v, 4) for v in (st["shift"] / st["b"]).tolist()]))
        return out


class RaptorDual(Raptor):
    """Exp B+D: a shared backbone with two complete Raptor heads (head 1: per-target labels, head 2: D2 labels).
    mode "train" -> [b, 24] logits (head 1, head 2); "avg" -> logit of the mean of the two probabilities (what predict()
    sigmoids); "h1" / "h2" -> one head. self.att_mean keeps the heads' mean window attention [b, k, 12] (detached)."""

    def __init__(self, res, pretrained):
        super().__init__(res, pretrained)
        d = self.backbone.num_features
        self.norm2 = nn.LayerNorm(d)
        self.att2 = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12))
        self.clsW2 = nn.Parameter(torch.zeros(12, d))
        self.clsb2 = nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.clsW2, std=0.02)
        self.mode, self.att_mean = "avg", None

    def forward(self, x, b, ctx=None):
        f = self.backbone(x).view(b, x.shape[0] // b, -1)
        out, att = [], []
        for norm, net, W, bias in ((self.norm, self.att, self.clsW, self.clsb),
                                   (self.norm2, self.att2, self.clsW2, self.clsb2)):
            h = norm(f)
            a = torch.softmax(net(h).float(), dim=1)
            att.append(a.detach())
            out.append((torch.einsum("bkn,bkf->bnf", a.to(h.dtype), h) * W).sum(-1) + bias)
        self.att_mean = 0.5 * (att[0] + att[1])
        if self.mode == "train":
            return torch.cat(out, 1)
        if self.mode in ("h1", "h2"):
            return out[int(self.mode[1]) - 1]
        p = 0.5 * (out[0].float().sigmoid() + out[1].float().sigmoid())
        return torch.logit(p.clamp(1e-7, 1 - 1e-7))


class RaptorSev(Raptor):
    """Exp P1-sev: the 1e head plus, per finding, two auxiliary logits (>=moderate, >=severe) on the same per-finding
    attention-pooled feature. forward() returns the main logits only; self.aux keeps [b, 2, 12] for the loss."""

    def __init__(self, res, pretrained):
        super().__init__(res, pretrained)
        d = self.backbone.num_features
        self.sevW = nn.Parameter(torch.zeros(2, 12, d)); self.sevb = nn.Parameter(torch.zeros(2, 12))
        nn.init.trunc_normal_(self.sevW, std=0.02)
        self.aux = None

    def forward(self, x, b, ctx=None):
        f = self.backbone(x).view(b, x.shape[0] // b, -1)
        h = self.norm(f)
        a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)
        z = torch.einsum("bkn,bkf->bnf", a, h)                                  # [b, 12, d]
        self.aux = (z[:, None] * self.sevW).sum(-1) + self.sevb                 # [b, 2, 12]
        return (z * self.clsW).sum(-1) + self.clsb


class RaptorTags(Raptor):
    """Exp P1-tags: the 1e head plus one auxiliary logit per report tag on the mean of its related findings'
    attention-pooled features. forward() returns the main logits only; self.aux keeps [b, n_tags] for the loss."""

    def __init__(self, res, pretrained):
        super().__init__(res, pretrained)
        d, anchors = self.backbone.num_features, CFG["tags"]["anchors"]
        M = torch.zeros(len(anchors), 12)
        for k, a in enumerate(anchors):
            M[k, a] = 1.0 / len(a)
        self.register_buffer("tagM", M, persistent=False)
        self.tagW = nn.Parameter(torch.zeros(len(anchors), d)); self.tagb = nn.Parameter(torch.zeros(len(anchors)))
        nn.init.trunc_normal_(self.tagW, std=0.02)
        self.aux = None

    def forward(self, x, b, ctx=None):
        f = self.backbone(x).view(b, x.shape[0] // b, -1)
        h = self.norm(f)
        a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)
        z = torch.einsum("bkn,bkf->bnf", a, h)                                  # [b, 12, d]
        za = torch.einsum("tn,bnf->btf", self.tagM.to(z.dtype), z)              # [b, n_tags, d]
        self.aux = (za * self.tagW).sum(-1) + self.tagb                         # [b, n_tags]
        return (z * self.clsW).sum(-1) + self.clsb


def convnext_layer_id(name):
    """Exp P1-cnx2: depth group of a timm ConvNeXt-T parameter (0 = stem ... 7 = head norm)."""
    if name.startswith("stem"):
        return 0
    if name.startswith("stages."):
        st = int(name.split(".")[1])
        if st == 2 and ".blocks." in name:
            return 3 + int(name.split(".")[3]) // 3
        return {0: 1, 1: 2, 2: 3, 3: 6}[st]
    return 7


def vit_layer_id(name):
    """Exp MS1: depth group of a timm ViT parameter (0 = patch / position embedding, 1..depth = blocks, depth + 1 =
    final norm and MAP pooling head)."""
    if name.startswith(("patch_embed", "pos_embed", "cls_token", "reg_token")):
        return 0
    if name.startswith("blocks."):
        return 1 + int(name.split(".")[1])
    return 10 ** 6                                                              # top group, capped in layer_id()


def layer_id(model):
    """Depth-group function for CFG['layer_decay']: ConvNeXt (Exp P1-cnx2, 8 groups) or ViT (Exp MS1, depth + 2)."""
    if CFG["arch"].startswith("vit"):
        top = 1 + len(model.backbone.blocks)
        return lambda n: min(vit_layer_id(n), top)
    return convnext_layer_id


def build(res, pretrained):
    return (RaptorCtx if "ctx" in CFG else RaptorTok if "tok" in CFG else RaptorPlane if "plane" in CFG else RaptorDual if "dual" in CFG
            else RaptorSev if "sev" in CFG else RaptorTags if "tags" in CFG
            else RaptorSpatial31 if CFG.get("spatial", {}).get("v31")
            else RaptorSpatial if "spatial" in CFG else Raptor)(res, pretrained)


# ---------------------------------------------------------------- scoring
def auc(score, truth):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); s = np.asarray(score)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = truth.sum(); n0 = len(truth) - n1
    return float((ranks[truth].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def macro_auc(pred, ids, ref):
    y = np.stack([ref[s][0] for s in ids]); w = np.stack([ref[s][1] for s in ids])
    m = (w > 0) & ((y <= 0.3) | (y >= 0.7)); t = y >= 0.7
    per = [auc(pred[m[:, k], k], t[m[:, k], k]) for k in range(12)]
    return float(np.nanmean(per)), dict(zip(TARGETS, [round(v, 4) for v in per]))


def val_bce(pred, ids, ref):
    """Validation loss: per-study weighted soft BCE against the validation labels (as the training loss)."""
    y = np.stack([ref[s][0] for s in ids]); w = np.stack([ref[s][1] for s in ids])
    p = np.clip(pred, 1e-6, 1 - 1e-6)
    b = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(((b * w).sum(1) / np.maximum(w.sum(1), 1)).mean())


@torch.no_grad()
def predict(model, ids, meta, device, tta=False, k=None, offset=None, res=None):
    model.eval()
    ds = Studies(ids, meta, {}, k if k is not None else CFG["eval_windows"], False, offset)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=CFG["workers"], pin_memory=True)
    out, res = [], res or CFG["eval_res"]
    for x, sag, _, _, ctx in dl:
        ctx = ctx.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(prepare(x, sag, False, res, device), 1, ctx).float().sigmoid()
            if tta:
                xm = mirror(x.flatten(0, 1), sag.flatten()).unsqueeze(0)
                xm = prepare(xm, sag, False, res, device)
                FLIP["windows"] = torch.ones(xm.shape[0], dtype=torch.bool, device=device)
                p = 0.5 * (p + model(xm, 1, ctx).float().sigmoid())
                FLIP["windows"] = None
        out.append(p[0].cpu().numpy())
    return np.stack(out)


def predict_multi(model, ids, meta, device):
    """Exp 2a / CW1 secondary: average of n stratified 16-window passes (fixed offsets (j + 0.5) / n in each bin)."""
    n = (CFG.get("ctx") or CFG["tok"])["multi_passes"]
    return np.mean([predict(model, ids, meta, device, k=CFG["train_windows"], offset=(j + 0.5) / n)
                    for j in range(n)], 0)


@torch.no_grad()
def predict_planes(model, ids, meta, device):
    """Exp 2c secondary: sigmoid of each plane's auxiliary logit [n, 3, 12] (absent plane -> sigmoid(bias))."""
    model.eval()
    dl = torch.utils.data.DataLoader(Studies(ids, meta, {}, CFG["eval_windows"], False), batch_size=1,
                                     num_workers=CFG["workers"], pin_memory=True)
    out = []
    for x, sag, _, _, ctx in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(prepare(x, sag, False, CFG["eval_res"], device), 1, ctx.to(device))
        out.append(model.aux[0].float().sigmoid()[0].cpu().numpy())
    return np.stack(out)


@torch.no_grad()
def guide_refresh(model, ids, meta, device, prev):
    """Exp B+D (D): each finding's top window per training study under the two heads' mean attention (eval mode,
    320 centre crop, every second eval window). Returns ({study: top centres [12]}, logged stats)."""
    g, n = CFG["guide"], CFG["maxs"]
    k = CFG["eval_windows"] // g["stride"]
    model.eval(); model.mode = "avg"
    dl = torch.utils.data.DataLoader(Studies(ids, meta, {}, k, False), batch_size=1, num_workers=CFG["workers"],
                                     pin_memory=True)
    guide, ent, hit, near, moved, uniq = {}, [], [], [], [], []
    for s, (x, sag, _, _, ctx) in zip(ids, dl):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(prepare(x, sag, False, g["res"], device), 1, ctx.to(device))
        a = model.att_mean[0].float()                                           # [k, 12]
        c = np.clip(centres(meta[s]["mask"], k, False), 1, n - 2)
        top = c[a.argmax(0).cpu().numpy()]
        guide[s] = top
        ent.append((-(a * a.clamp_min(1e-12).log()).sum(0) / math.log(k)).cpu().numpy())
        rnd = np.clip(centres(meta[s]["mask"], CFG["train_windows"], True), 1, n - 2)
        dist = np.abs(top[:, None] - rnd[None]).min(1)
        hit.append(dist == 0); near.append(dist <= 1); uniq.append(len(np.unique(top)))
        if s in prev:
            moved.append(np.abs(top - prev[s]) > 1)
    model.mode = "train"; model.train()
    r4 = lambda v: dict(zip(TARGETS, [round(float(x), 4) for x in v]))
    stats = {"unique_tops_mean": round(float(np.mean(uniq)), 2),
             "top_in_random16": r4(np.mean(hit, 0)), "top_within1_random16": r4(np.mean(near, 0)),
             "attention_entropy_norm": r4(np.mean(ent, 0)),
             "top_moved_gt1_vs_prev": r4(np.mean(moved, 0)) if moved else None}
    return guide, stats


def save_predictions(path, ids, pred, note):
    path.write_text(json.dumps({"model": CFG["version"], "note": note, "targets_in_order": TARGETS,
                                "predictions": [{"study_id": s, "probabilities": [float(v) for v in p]}
                                                for s, p in zip(ids, pred)]}) + "\n")


# ---------------------------------------------------------------- main
def main():
    started = time.monotonic()
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels(ASSETS / LABELS_FILE, "labels")
    if "border" in CFG:
        labels = border_labels(labels)
    if "dual" in CFG:                                   # Exp B+D: [24] targets = per-target labels, then D2 labels
        lab2 = load_labels(ASSETS / CFG["dual"]["labels2"], "labels2")
        assert set(lab2) == set(labels)
        labels = {s: (np.concatenate([y, lab2[s][0]]), np.concatenate([w, lab2[s][1]])) for s, (y, w) in labels.items()}
    if "sev" in CFG:                                    # Exp P1-sev: [36] targets = main 12, >=moderate 12, >=severe 12
        sp = ASSETS / CFG["sev"]["file"]
        assert sha(sp) == CFG["sha256"]["sev"], f"{sp} hash differs"
        sd = {r["study_id"]: (np.array([0.0 if v is None else v for v in r["probabilities"]], np.float32),
                              np.array(r["weights"], np.float32)) for r in json.loads(sp.read_text())["labels"]}
        z24 = (np.zeros(24, np.float32), np.zeros(24, np.float32))
        labels = {s: (np.concatenate([y, sd.get(s, z24)[0]]), np.concatenate([w, sd.get(s, z24)[1]]))
                  for s, (y, w) in labels.items()}
        print(f"SEV aux targets for {len(sd)} studies; observed cells {int(sum(v[1].sum() for v in sd.values()))}",
              flush=True)
    if "tags" in CFG:                                   # Exp P1-tags: [12 + n_tags] targets = main 12, then tags
        tp = ASSETS / CFG["tags"]["file"]
        assert sha(tp) == CFG["sha256"]["tags"], f"{tp} hash differs"
        td = {r["study_id"]: (np.array([0.0 if v is None else v for v in r["probabilities"]], np.float32),
                              np.array(r["weights"], np.float32)) for r in json.loads(tp.read_text())["labels"]}
        nt = len(CFG["tags"]["anchors"])
        assert all(len(v[0]) == nt for v in td.values())
        zt = (np.zeros(nt, np.float32), np.zeros(nt, np.float32))
        labels = {s: (np.concatenate([y, td.get(s, zt)[0]]), np.concatenate([w, td.get(s, zt)[1]]))
                  for s, (y, w) in labels.items()}
        print(f"TAG aux targets for {len(td)} studies; observed cells {int(sum(v[1].sum() for v in td.values()))}; "
              f"positives {[int(sum(v[0][k] * v[1][k] for v in td.values())) for k in range(nt)]}", flush=True)
    ref =load_labels(ASSETS / "validation_labels.json", "validation")
    assert sha(ASSETS / "folds.csv") == CFG["sha256"]["folds"]
    fold = {r["study_id"]: r["fold"] for r in csv.DictReader((ASSETS / "folds.csv").open())}
    meta = read_manifest()
    gold_ids = sorted(s for s, m in meta.items() if m["gold"])
    train_ids = sorted(s for s in labels if fold[s] != CFG["validation_fold"])
    val_ids = sorted(s for s in labels if fold[s] == CFG["validation_fold"])
    assert len(train_ids) == 3524 and len(val_ids) == 825 and len(gold_ids) == 58
    assert set(val_ids) == set(ref) and not (set(gold_ids) & set(labels))
    assert all(s in meta for s in train_ids + val_ids)
    oof_ids = []
    if "teacher" in CFG:                                # Phase 3: train outside split k, predict split k out of fold
        train_ids, oof_ids = teacher_ids(train_ids, val_ids)
        print(f"TEACHER split {CFG['teacher']['split']}: fit {len(train_ids)}, out-of-fold {len(oof_ids)}", flush=True)
    print(f"train {len(train_ids)}, validation {len(val_ids)}, gold (predict only) {len(gold_ids)}", flush=True)
    pos_weight = None
    if "pos_weight" in CFG:                             # Exp P1-r949: prevalence from the training folds only
        pc, wc = CFG["pos_weight"]["p_clip"], CFG["pos_weight"]["w_clip"]
        ys = np.stack([labels[s][0] for s in train_ids]); ws = np.stack([labels[s][1] for s in train_ids])
        p = np.clip((ys * ws).sum(0) / np.maximum(ws.sum(0), 1e-9), *pc)
        pos_weight = torch.tensor(np.clip((1 - p) / p, *wc), dtype=torch.float32, device=device)
        print("POS_WEIGHT " + json.dumps({t: [round(float(a), 4), round(float(b), 3)]
                                          for t, a, b in zip(TARGETS, p, pos_weight.tolist())}), flush=True)

    model = build(CFG["train_res"], pretrained=True).to(device).to(memory_format=torch.channels_last)
    eval_model = build(CFG["eval_res"], pretrained=False).to(device).to(memory_format=torch.channels_last)
    eval_model.load_state_dict(model.state_dict(), strict=True)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    if "layer_decay" in CFG:                            # Exp P1-cnx2 / MS1: one group per depth, LR x decay^(top - depth)
        groups, lid = {}, layer_id(model)
        for n, p_ in model.backbone.named_parameters():
            groups.setdefault(lid(n), []).append(p_)
        top = max(groups)                               # ConvNeXt 7 (unchanged), ViT depth + 1
        bb = [{"params": ps, "lr": CFG["lr"] * CFG["layer_decay"] ** (top - i)} for i, ps in sorted(groups.items())]
        print("LAYER_DECAY " + json.dumps({i: [len(ps), CFG["lr"] * CFG["layer_decay"] ** (top - i)]
                                           for i, ps in sorted(groups.items())}), flush=True)
    else:
        bb = [{"params": model.backbone.parameters(), "lr": CFG["lr"]}]
    opt = torch.optim.AdamW(bb + [{"params": head, "lr": CFG["head_lr"]}], weight_decay=CFG["wd"])
    first = CFG.get("first_epoch", 1)
    if "init_from" in CFG and not (OUT / "recovery.pt").is_file():
        r = torch.load(INIT / "recovery.pt", map_location="cpu", weights_only=False)
        assert r["epoch"] == first - 1 and r["cfg"]["version"] == "coatnet_d96_exp1_v1", (r["epoch"], r["cfg"]["version"])
        model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"])
        for g, lr in zip(opt.param_groups, (CFG["lr"], CFG["head_lr"])):
            g["lr"] = g["initial_lr"] = lr            # new cycle; Adam moments carried over
        print(f"INIT from {INIT / 'recovery.pt'} (epoch {r['epoch']}, val {r['history'][-1]['val_macro_auc']:.4f})",
              flush=True)
        del r
    steps_per_epoch = math.ceil(len(train_ids) / (CFG["studies_per_step"] * CFG["accum"]))
    total, warm = steps_per_epoch * CFG["epochs"], steps_per_epoch * CFG["warmup_epochs"]
    floor = CFG.get("lr_floor", 0.0)
    def cosine(s):
        return (s + 1) / warm if s < warm else floor + (1 - floor) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))

    def lr_factor(s):
        """Exp 6a: from the cycle start, one cosine cycle per epoch from the scheduled level down to the floor."""
        if "swa_cycle_start" not in CFG:
            return cosine(s)
        c0 = steps_per_epoch * (CFG["swa_cycle_start"] - first)
        if s < c0:
            return cosine(s)
        frac = ((s - c0) % steps_per_epoch) / steps_per_epoch
        return floor + (cosine(c0) - floor) * 0.5 * (1 + math.cos(math.pi * frac))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    start_epoch, history = first, []
    rec = OUT / "recovery.pt"
    if rec.is_file():
        r = torch.load(rec, map_location="cpu", weights_only=False)
        assert r["cfg"] == CFG, "recovery config differs"
        model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"]); sched.load_state_dict(r["sched"])
        start_epoch, history = r["epoch"] + 1, r["history"]
        print(f"RESUMED after epoch {r['epoch']}", flush=True)

    train_ds = Studies(train_ids, meta, labels, CFG["train_windows"], True)
    if rec.is_file() and "guide" in CFG:
        train_ds.guide = r.get("guide", {})
    for epoch in range(start_epoch, first + CFG["epochs"]):
        guide_stats = None
        if "guide" in CFG and epoch >= CFG["guide"]["start"] and (epoch - CFG["guide"]["start"]) % CFG["guide"]["every"] == 0:
            tg = time.monotonic()
            train_ds.guide, guide_stats = guide_refresh(model, train_ids, meta, device, train_ds.guide)
            guide_stats["seconds"] = round(time.monotonic() - tg, 1)
            print("GUIDE " + json.dumps({"epoch": epoch, **guide_stats}), flush=True)
        if "dual" in CFG:
            model.mode = "train"
        model.train(); t0 = time.monotonic()
        dl = torch.utils.data.DataLoader(train_ds, batch_size=CFG["studies_per_step"], shuffle=True,
                                         num_workers=CFG["workers"], pin_memory=True, drop_last=False,
                                         persistent_workers=False, worker_init_fn=lambda w: np.random.seed(
                                             (CFG["seed"] * 1000 + epoch * 100 + w) % 2**32))
        running, n = 0.0, 0
        tag_run, tag_n = 0.0, 0                            # Exp P1-tags: mean masked tag-aux BCE per epoch
        opt.zero_grad(set_to_none=True)
        spatial = "spatial" in CFG and CFG["spatial"]["on"]
        if "v_start" in CFG.get("spatial", {}):
            model.v_on = eval_model.v_on = epoch >= CFG["spatial"]["v_start"]
        gnorm, gsteps = {}, 0
        if spatial:
            model.stats, model.collect = {}, "train"
        for i, (x, sag, y, w, ctx) in enumerate(dl):
            b = x.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(prepare(x, sag, True, CFG["train_res"], device, model if spatial else None), b,
                               ctx.to(device)).float()
            y, w = y.to(device), w.to(device)
            if "dual" in CFG:                                                   # mean of the two heads' losses
                loss = 0.5 * sum((F.binary_cross_entropy_with_logits(logits[:, j:j + 12], y[:, j:j + 12], reduction="none")
                                  * w[:, j:j + 12]).sum() / w[:, j:j + 12].sum().clamp_min(1) for j in (0, 12))
            elif "sev" in CFG:                                                  # main 12 + masked severity aux
                loss = (F.binary_cross_entropy_with_logits(logits, y[:, :12], reduction="none") * w[:, :12]).sum() \
                    / w[:, :12].sum().clamp_min(1)
                ya, wa = y[:, 12:].view(b, 2, 12), w[:, 12:].view(b, 2, 12)
                la = F.binary_cross_entropy_with_logits(model.aux.float(), ya, reduction="none")
                loss = loss + CFG["sev"]["aux"] * (la * wa).sum() / wa.sum().clamp_min(1)
            elif "tags" in CFG:                                                 # main 12 + masked tag aux
                loss = (F.binary_cross_entropy_with_logits(logits, y[:, :12], reduction="none") * w[:, :12]).sum() \
                    / w[:, :12].sum().clamp_min(1)
                ya, wa = y[:, 12:], w[:, 12:]
                la = (F.binary_cross_entropy_with_logits(model.aux.float(), ya, reduction="none") * wa).sum() \
                    / wa.sum().clamp_min(1)
                loss = loss + CFG["tags"]["aux"] * la
                tag_run, tag_n = tag_run + float(la), tag_n + 1
            else:
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
                        * w).sum() / w.sum().clamp_min(1)
            if "plane" in CFG:
                al, pr = model.aux
                wa = w[:, None] * pr[..., None].float()                             # [b, 3, 12]
                la = F.binary_cross_entropy_with_logits(al.float(), y[:, None].expand_as(al), reduction="none")
                loss = loss + CFG["plane"]["aux"] * (la * wa).sum() / wa.sum().clamp_min(1)
            (loss / CFG["accum"]).backward()
            if (i + 1) % CFG["accum"] == 0 or i + 1 == len(dl):
                if spatial:                                                     # Exp v3 diagnostic: branch gradients
                    for nm, t in (("score_in", model.score[0].weight), ("score_out", model.score[-1].weight),
                                  ("U", model.clsU), ("W", model.clsW)) + ((("V", model.clsV),)
                                                                           if hasattr(model, "clsV") else ()):
                        gnorm[nm] = gnorm.get(nm, 0.0) + (t.grad.norm().item() if t.grad is not None else 0.0)
                    gsteps += 1
                    if epoch == first and gsteps == 3:     # U moves at 1, score_out at 2, score_in at 3
                        print("GRAD_STEPS1_3 " + json.dumps({k: v for k, v in gnorm.items()}), flush=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            running += loss.item() * b; n += b
            if (i + 1) % 200 == 0:
                print(f"  epoch {epoch} step {i + 1}/{len(dl)} loss {running / n:.4f} "
                      f"{(time.monotonic() - t0) / n:.3f}s/study", flush=True)
        train_s = time.monotonic() - t0
        eval_model.load_state_dict(model.state_dict(), strict=True)
        if spatial:
            model.collect, model.valid = None, None
            eval_model.stats, eval_model.collect = {}, "eval"
        scored = epoch in CFG.get("eval_epochs", [epoch])   # Phase 3 teachers score fold 0 at a few epochs only
        pred = predict(eval_model, val_ids, meta, device) if scored else None
        if spatial:
            eval_model.collect = None
        score, per = macro_auc(pred, val_ids, ref) if scored else (None, None)
        vloss = val_bce(pred, val_ids, ref) if scored else None
        record = {"epoch": epoch, "train_loss": running / n, "val_macro_auc": score, "val_bce": vloss, "per_target": per,
                  "train_s": round(train_s, 1), "total_s": round(time.monotonic() - t0, 1),
                  "lr": [g["lr"] for g in opt.param_groups]}
        if guide_stats:
            record["guide"] = guide_stats
        if "tags" in CFG:
            record["tag_aux_loss"] = tag_run / max(1, tag_n)
            print(f"TAG_AUX epoch {epoch} loss {record['tag_aux_loss']:.4f}", flush=True)
        if spatial:
            record["spatial"] = {"train": model.summary("train"), "eval": eval_model.summary("eval"),
                                 "grad_norm_mean": {k: v / max(1, gsteps) for k, v in gnorm.items()}}
            print("SPATIAL " + json.dumps(record["spatial"]), flush=True)
        history.append(record)
        print(f"EPOCH {epoch}: loss {record['train_loss']:.4f} val macro AUC {'-' if score is None else f'{score:.4f}'} "
              f"val BCE {'-' if vloss is None else f'{vloss:.4f}'} "
              f"({record['total_s']:.0f}s) {json.dumps(per)}", flush=True)
        if scored:
            save_predictions(OUT / f"val_epoch{epoch:02d}.json", val_ids, pred, f"epoch {epoch}, 320 crop, no TTA")
        if epoch in CFG["swa_epochs"] or (scored and "teacher" not in CFG and score >= max(
                h["val_macro_auc"] for h in history if h["val_macro_auc"] is not None)):
            torch.save(model.state_dict(), OUT / f"weights_epoch{epoch:02d}.pt")
        torch.save({"cfg": CFG, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "history": history, "guide": train_ds.guide}, OUT / "recovery.tmp")
        os.replace(OUT / "recovery.tmp", rec)
        (OUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    # ---- pre-registered primary: weight average of the SWA epochs (11-15; 20-23 for Exp 1b), BN refreshed
    sd = [torch.load(OUT / f"weights_epoch{e:02d}.pt", map_location="cpu") for e in CFG["swa_epochs"]]
    avg = {k: (sum(s[k].float() for s in sd) / len(sd)).to(sd[-1][k].dtype) if sd[-1][k].is_floating_point()
           else sd[-1][k] for k in sd[-1]}
    model.load_state_dict(avg)
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    model.train()
    refresh = random.Random(CFG["seed"]).sample(train_ids, CFG["bn_refresh_studies"])
    plain = Studies(refresh, meta, labels, CFG["train_windows"], False)
    with torch.no_grad():
        for x, sag, _, _, ctx in (torch.utils.data.DataLoader(plain, batch_size=2, num_workers=CFG["workers"])
                                  if bns else ()):      # no BatchNorm (ConvNeXt, ViT): the refresh is a no-op
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(prepare(x, sag, False, CFG["train_res"], device), x.shape[0], ctx.to(device))
    torch.save(model.state_dict(), OUT / f"weights_swa{CFG['swa_epochs'][0]}_{CFG['swa_epochs'][-1]}.pt")
    eval_model.load_state_dict(model.state_dict(), strict=True)
    results, spatial_swa = {}, None
    if "spatial" in CFG and CFG["spatial"]["on"]:
        eval_model.stats, eval_model.collect = {}, "swa"
    passes = (("swa", False),) if "teacher" in CFG else (("swa", False), ("swa_mirror", True))
    for name, tta in passes:
        for split, ids in (("val", val_ids), ("gold", gold_ids)) + ((("oof", oof_ids),) if oof_ids else ()):
            p = predict(eval_model, ids, meta, device, tta=tta)
            if name == "swa" and split == "val" and getattr(eval_model, "collect", None) == "swa":
                spatial_swa, eval_model.collect = eval_model.summary("swa"), None
            save_predictions(OUT / f"{split}_{name}.json", ids, p, f"{name}, 320 crop")
            if split == "val":
                results[name] = macro_auc(p, val_ids, ref)
    if "plane" in CFG:                                                  # secondary: each plane branch alone
        for split, ids in (("val", val_ids), ("gold", gold_ids)):
            p = predict_planes(eval_model, ids, meta, device)
            for j, nm in enumerate(("sagittal", "coronal", "axial")):
                save_predictions(OUT / f"{split}_swa_aux_{nm}.json", ids, p[:, j], f"swa, {nm} auxiliary head only")
                if split == "val":
                    results[f"aux_{nm}"] = macro_auc(p[:, j], val_ids, ref)
    if "spatial" in CFG:                                                # Exp v3 secondary: full 384 input
        for split, ids in (("val", val_ids), ("gold", gold_ids)):
            p = predict(eval_model, ids, meta, device, res=CFG["train_res"])
            save_predictions(OUT / f"{split}_swa_384.json", ids, p, "swa, full 384 input (secondary)")
            if split == "val":
                results["swa_384"] = macro_auc(p, val_ids, ref)
    if "dual" in CFG:                                                   # Exp B+D secondary: each head alone
        hp = {}
        for hd in ("h1", "h2"):
            eval_model.mode = hd
            for split, ids in (("val", val_ids), ("gold", gold_ids)):
                p = predict(eval_model, ids, meta, device)
                save_predictions(OUT / f"{split}_swa_{hd}.json", ids, p, f"swa, head {hd[1]} alone, 320 crop")
                if split == "val":
                    results[f"swa_{hd}"], hp[hd] = macro_auc(p, val_ids, ref), p
        eval_model.mode = "avg"
        rk = lambda v: np.argsort(np.argsort(v, 0), 0)
        results["head_spearman"] = dict(zip(TARGETS, [round(float(np.corrcoef(rk(hp["h1"])[:, j], rk(hp["h2"])[:, j])[0, 1]), 4)
                                                      for j in range(12)]))
    if "ctx" in CFG or "tok" in CFG:
        for split, ids in (("val", val_ids), ("gold", gold_ids)):
            p = predict_multi(eval_model, ids, meta, device)
            save_predictions(OUT / f"{split}_swa_multi16.json", ids, p,
                             f"swa, {(CFG.get('ctx') or CFG['tok'])['multi_passes']} x {CFG['train_windows']}-window passes, 320 crop")
            if split == "val":
                results["swa_multi16"] = macro_auc(p, val_ids, ref)
    best = max((h for h in history if h["val_macro_auc"] is not None), key=lambda h: h["val_macro_auc"])
    if "teacher" not in CFG:                            # teachers keep only the pre-registered SWA weights
        eval_model.load_state_dict(torch.load(OUT / f"weights_epoch{best['epoch']:02d}.pt", map_location="cpu"))
        for split, ids in (("val", val_ids), ("gold", gold_ids)):
            save_predictions(OUT / f"{split}_best_epoch.json", ids, predict(eval_model, ids, meta, device),
                             f"best epoch {best['epoch']} (selected on validation), 320 crop")
    summary = {"cfg": CFG, "history": history, "primary_swa11_15": results["swa"],
               "swa_mirror_tta": results.get("swa_mirror"), "swa_multi16": results.get("swa_multi16"),
               "teacher": {**CFG["teacher"], "fit": len(train_ids), "oof": len(oof_ids)} if "teacher" in CFG else None,
               "aux_planes": {k: v for k, v in results.items() if k.startswith("aux_")} or None,
               "gate_abs_mean": float(model.gate.abs().mean()) if hasattr(model, "gate") else None,
               "swa_384": results.get("swa_384"), "spatial_swa_val": spatial_swa,
               "dual_heads": {k: results[k] for k in ("swa_h1", "swa_h2", "head_spearman")} if "dual" in CFG else None,
               "best_epoch": {"epoch": best["epoch"], "val_macro_auc": best["val_macro_auc"]},
               "seconds": round(time.monotonic() - started, 1),
               "note": "Gold predicted only; score locally. Primary = swa (pre-registered)."}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps({k: summary[k] for k in ("primary_swa11_15", "swa_mirror_tta", "swa_multi16", "best_epoch",
                                                                 "seconds")}), flush=True)


if __name__ == "__main__":
    main()
