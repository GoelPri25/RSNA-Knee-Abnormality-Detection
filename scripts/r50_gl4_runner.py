"""Exp r50-GL v4: one ResNet-50 that reads the whole study (global windows) and a zoom of each finding (local crops),
answered as ONE fixed blend of its two heads; no reader, no learned gate, no fused loss.

Why (r50-GL3-XAI, 2026-10-18): the only global + local number above the bar is a FIXED 50/50 mean of a global and a
crop-trained local (label-v2 r50 + r50-GL v2 local 0.9468, v2's own g + l 0.9453); every learned combiner (v1/v2
reader, v3 gate) was fit in-sample, where the local is near-perfect (v3 local training AUC 0.9998), and lost.
v4 = r50-GL v2 (scripts/r50_gl_runner.py: label-v2 recipe, MS1 crop table, boxes, 1 of 4 windows by MS1 attention
+ -16..16 px jitter in training, all 4 MS1-weighted in evaluation, one augmentation draw shared by both views, dual
BatchNorm statistics) without the reader:
    loss  = BCE(global) + 0.5 x BCE_cropped(local)
    fused = sigmoid((g + l) / 2) on the 10 cropped findings, g on effusion / synovitis   (PRIMARY readout)
Arms (V4_ARM), all one network, one forward, one checkpoint:
    shared   v2's sharing: every backbone weight sees both losses (local x 0.5); BN affine shared, statistics per view.
    adapter  the global path is the label-v2 r50 recipe exactly: the crop pass uses the shared weights DETACHED
             (stem-layer2 under no_grad, BN affine detached there), plus its own capacity in layer3/4: LoRA rank 32
             (alpha 32) on every 1x1 conv (conv1, conv3, downsample: 20 convs, ~1.08 M) and its own BN affine; local
             parameters in their own optimiser group (LR 5e-4) and their own gradient-clip group (round-2 N1, N2).
             SWA averages the dense LoRA delta B A (N3), not the factors.
    l4copy   stem-layer3 shared as v2; the crops run their own copy of layer4 (initialised from ImageNet, LR 1e-4).
    l3copy   (round 1, L4C-R1) stem-layer2 shared; the crops run their own copies of layer3 and layer4.
Logged every epoch: fold-0 AUC on label-v2 confident cells (and Steven validation labels) and val BCE for fused / global
/ local, fused - global, and the local head's training-vs-fold-0 AUC gap on a fixed 300-study training sample.
After SWA (21-25): both BN statistic sets refreshed (globals: 16 eval-centre windows at 384; crops: all 4 crops per
finding, the evaluation distribution; v2 used the top-1 crop).

Env: D96_CACHE, D96_PHASE=r50_gl, D96_SEED, GL_CROPS (slim MS1 table), GL_OUT, V4_ARM, GL_SMOKE=1.
"""
import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("GL_VERSION", "v2")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coatnet_d96_runner as r
import r50_gl_runner as v2

OUT, SMOKE, CFG = v2.OUT, v2.SMOKE, r.CFG
IMG, MAXS = v2.IMG, v2.MAXS
ARM = os.environ.get("V4_ARM", "shared"); assert ARM in ("shared", "adapter", "l4copy", "l3copy"), ARM
CF, CMASK, K = v2.CF, v2.CMASK, v2.K
G4 = {"version": f"r50_gl4_{ARM}", "arm": ARM, "crops": "r50_gl v2 (MS1 table, boxes, 1-of-4 + jitter / all 4)",
      "global_w": 1.0, "aux_local": 0.5, "readout": "sigmoid((g + l) / 2) on the 10 cropped findings, g elsewhere",
      "bn_refresh_crops": "all 4 per finding", "train_gap_studies": 300, "smoke_steps": 32}
if ARM == "adapter":
    G4.update({"lora_rank": 32, "lora_alpha": 32, "lora_on": "layer3/4 1x1 convs", "local_lr": 5e-4,
               "crop_affine": "own in layer3/4, shared-detached in stem-layer2", "clip": "separate"})
if ARM == "l4copy":
    G4.update({"crop_layer4": "own copy, backbone LR"})
if ARM == "l3copy":
    G4.update({"crop_layer3_4": "own copies, backbone LR"})
HEADS = ("fused", "global", "local")
# V4-IMP (docs/v4_improve_spec.md v2): one change each on the l4copy recipe; V4_IMP unset = v4 exactly (G4 unchanged).
#   silw    report-silent assumed-negative cells (rule state SIL, label-v2 y < 0.5) BCE weight x 0.25
#   hill    the same cells: BCE's negative term -> Hill loss (1 - y)(1.5 - p) p^2 (Zhang et al. 2021); full weight
#   mcpair  each window / crop + the co-registered complementary-contrast triplet (6 channels), seq-type attention bias
#   silw / hill: loss denominator = cell count (the weights before the change), so the step size is v4's.
#   fuse    each window / crop + its co-registered complementary-contrast window encoded by the shared backbone
#           (a normal 3-channel window of the partner series), fused at the feature level: h + Fuse([h, hp, h - hp]),
#           Fuse's output layer zero-init (starts as v4 exactly), head LR; no input-layer or attention change.
IMP = os.environ.get("V4_IMP", "none"); assert IMP in ("none", "silw", "hill", "mcpair", "fuse"), IMP
# BB-CMP (2026-10-19): the v4 l4copy recipe on other timm backbones (V4_BB; unset = r50 = v4 exactly). The crop branch
# runs its own copy of the backbone's last stage (r50: layer4), swapped in for the crop pass. Forced deviations only:
# the checkpoint's normalisation; CoAtNet's fixed input size (globals evaluated at 384, no 320 centre crop); DINOv2's
# patch 14 (378 train / crops, 322 eval) and ViT fine-tuning standards (as MS1: top LR 5e-5, layer decay 0.9, drop path
# 0.1). BatchNorm (incl. timm BatchNormAct2d) gets v4's crop statistics; LayerNorm backbones need none.
BB = os.environ.get("V4_BB", "r50")
BACKBONES = {
    "r50": {"arch": "resnet50.tv_in1k", "last": ("layer4",), "bns": 53},
    "cnx": {"arch": "convnext_tiny.in12k_ft_in1k_384", "last": ("stages.3", "head.norm"), "bns": 0},
    "coat0": {"arch": "coatnet_0_rw_224.sw_in1k", "last": ("stages.3", "norm"), "bns": 16, "norm": [0.5, 0.5],
              "eval_res": 384},
    "effv2s": {"arch": "tf_efficientnetv2_s.in21k_ft_in1k", "last": ("blocks.5", "conv_head", "bn2"), "bns": 110,
               "norm": [0.5, 0.5]},
    "dino2s": {"arch": "vit_small_patch14_reg4_dinov2.lvd142m", "last": ("blocks.9", "blocks.10", "blocks.11", "norm"),
               "bns": 0, "train_res": 378, "eval_res": 322, "crop_res": 378, "lr": 5e-5, "layer_decay": 0.9,
               "drop_path": 0.1},
    "dino3s": {"arch": "vit_small_patch16_dinov3.lvd1689m", "last": ("blocks.9", "blocks.10", "blocks.11", "norm"),
               "bns": 0, "train_res": 384, "eval_res": 320, "crop_res": 384, "lr": 5e-5, "layer_decay": 0.9,
               "drop_path": 0.1},
}
# PUB-KD (2026-10-19): the v4 l4copy recipe on nartaa's public 0.949 CoAtNet (backbone + global head initialised from
# its checkpoint, crop branch = a copy of its last stage, crop head = a copy of its head); fine-tune schedule (6 epochs,
# SWA 4-6, LR 5e-5 / 5e-4), globals trained and evaluated at 384, gradient checkpointing, 1 study per step x 8.
# CNXS (2026-10-22, Priyanjali): E4 recipe on ConvNeXt-S 22k (S6 / R6 init), crop branch = own stages.3 copy
BACKBONES["cnxs"] = {"arch": "convnext_small.fb_in22k_ft_in1k", "last": ("stages.3", "head.norm"), "bns": 0}
BACKBONES["rap2"] = {"arch": "coatnet_rmlp_2_rw_384", "last": ("stages.3", "norm"), "bns": 25, "eval_res": 384,
                     "lr": 5e-5, "head_lr": 5e-4, "epochs": 6, "swa_epochs": [4, 5, 6], "studies_per_step": 1, "accum": 8,
                     "grad_ckpt": True, "init": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt",
                     "init_sha256": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"}
# R7-D (2026-10-22): the l4copy recipe on heliosli's public R7 ConvNeXt-S (rsna-knee-compact-r7cnxfsss fold 0, trained
# with fastMRI+ boxes as an aux task): backbone only (its heads differ from ours), converted to timm convnext_small
# (enc.* renamed, timm's head.norm identity-init; scratchpad r7_convert.py); ImageNet normalisation as R7. Fine-tune
# schedule as rap2 with a full head LR (our heads start random): 8 epochs, SWA 6-8, LR 5e-5 / 1e-3.
BACKBONES["r7cnx"] = {"arch": "convnext_small", "last": ("stages.3", "head.norm"), "bns": 0, "lr": 5e-5,
                      "head_lr": 1e-3, "epochs": 8, "swa_epochs": [6, 7, 8], "init": "r7cnx_f0_timm.pt",
                      "init_sha256": "51103633d73f63fdce2d379fcaa1502b7808ef3f31e56ef3147643db6be7a8af"}
PUB_INIT = ("rap2", "r7cnx")
# PUB-KD: training targets from a teacher file (V4_TARGETS, same JSON layout as the label files, sha in V4_TARGETS_SHA);
# fold-0 scoring stays on label-v2 / the validation labels exactly as v4.
TARGETS_FILE, TARGETS_SHA = os.environ.get("V4_TARGETS"), os.environ.get("V4_TARGETS_SHA")
assert BB in BACKBONES, BB
if BB != "r50":
    assert ARM == "l4copy" and IMP == "none", "BB-CMP: the l4copy recipe only"
    _b = BACKBONES[BB]
    CFG.update({k: _b[k] for k in ("arch", "norm", "train_res", "eval_res", "lr", "layer_decay", "drop_path", "head_lr",
                                   "epochs", "swa_epochs", "studies_per_step", "accum", "grad_ckpt") if k in _b})
    CFG["version"] = f"{CFG['version']}_{BB}"
    r.MEAN = torch.tensor([CFG["norm"][0]] * 3 if "norm" in CFG else [0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    r.STD = torch.tensor([CFG["norm"][1]] * 3 if "norm" in CFG else [0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    v2.GL["crop_res"] = _b.get("crop_res", v2.GL["crop_res"])
    G4.update({"version": f"r50_gl4_{ARM}_{BB}", "bb": BB, "crop_copy": list(_b["last"]),
               "deviations": {k: _b[k] for k in ("norm", "train_res", "eval_res", "crop_res", "lr", "layer_decay",
                                                  "drop_path", "head_lr", "epochs", "swa_epochs", "studies_per_step",
                                                  "accum", "grad_ckpt", "init", "init_sha256") if k in _b}})
if TARGETS_FILE:
    G4.update({"targets_file": Path(TARGETS_FILE).name, "targets_sha256": TARGETS_SHA})
    CFG["version"] = f"{CFG['version']}_kd_{Path(TARGETS_FILE).stem}"
    CFG["sha256"] = {**CFG["sha256"], "kd_targets": TARGETS_SHA}
# SURPASS-F (2026-10-20): extra unlabelled studies (fastMRI knee DICOMs, own d96 cache, V4_EXT_CACHE with V4_EXT_SHARDS
# shards) trained on their teacher targets (V4_TARGETS must cover them). They have no MS1 crop table: they are batched
# apart from the competition studies (each batch one source) and train the global head only (no crop pass, no crop BN
# statistics); BN refresh and the train-gap sample stay on competition studies. V4_EXT_HOLDOUT: fastMRI ids never trained,
# predicted (global) after SWA. V4_STAGE: joint (competition + extra), pre (extra only), fine (competition only, from
# V4_INIT = a pre-stage SWA checkpoint; the crop layer 4 restarts as a copy of the pre-trained layer 4).
# V4_NOISE=1: Noisy Student's student noise: stochastic depth 0.2, dropout 0.3 on the window / crop features before the
# heads, stronger augmentation. V4_EPOCHS: schedule length (SWA = the last 5 epochs, rap2 the last 3).
# V4_EVAL_ONLY=<weights>: load, predict the holdout (global), exit (the ra baseline on the fastMRI holdout).
EXT, EXT_SHARDS = os.environ.get("V4_EXT_CACHE"), int(os.environ.get("V4_EXT_SHARDS", "0"))
EXT_HOLDOUT, EVAL_ONLY = os.environ.get("V4_EXT_HOLDOUT"), os.environ.get("V4_EVAL_ONLY")
STAGE, INIT = os.environ.get("V4_STAGE", "joint"), os.environ.get("V4_INIT")
NOISE = os.environ.get("V4_NOISE") == "1"
# R7-D-C (2026-10-22): V4_INIT_MODE=continue loads a finished student's SWA weights strictly (heads and crop copy kept,
# no layer4_c reset; the optimizer state was not kept) and trains V4_EPOCHS more with peak LRs x V4_LR_SCALE.
INIT_MODE, LR_SCALE = os.environ.get("V4_INIT_MODE", "fine"), float(os.environ.get("V4_LR_SCALE", "1"))
assert INIT_MODE in ("fine", "continue"), INIT_MODE
assert STAGE in ("joint", "pre", "fine") and (STAGE != "fine" or INIT) and (STAGE == "joint" or BB == "r50"), STAGE
if EXT or NOISE or INIT or os.environ.get("V4_EPOCHS"):
    assert ARM == "l4copy" and IMP == "none" and TARGETS_FILE, "SURPASS-F: the KD l4copy recipe only"
    sf = {"stage": STAGE}
    if EXT:
        assert EXT_SHARDS > 0
        sf.update({"ext_cache": Path(EXT).name, "ext_shards": EXT_SHARDS, "ext_batches": "one source per batch",
                   "ext_loss": "global only", "holdout": Path(EXT_HOLDOUT).name if EXT_HOLDOUT else None})
    if os.environ.get("V4_EPOCHS"):
        E = int(os.environ["V4_EPOCHS"]); n_swa = 3 if BB in PUB_INIT else 5
        CFG.update({"epochs": E, "swa_epochs": list(range(E - n_swa + 1, E + 1))})
        if "eval_epochs" in CFG:
            CFG["eval_epochs"] = [e for e in range(5, E + 1, 5)] or [E]
        sf["epochs"] = E
    if NOISE:
        CFG["drop_path"] = 0.2
        CFG["aug"] = {"mirror_p": 0.5, "rot_deg": 15.0, "scale": 0.15, "shift": 0.08, "gamma": 0.35, "gain": 0.15}
        sf.update({"noise": {"drop_path": 0.2, "feature_dropout": 0.3, "aug": CFG["aug"]}})
    if INIT:
        sf.update({"init": Path(INIT).name, "init_sha256": r.sha(Path(INIT)) if Path(INIT).is_file() else None})
    if INIT_MODE == "continue":                              # R7-D-C: warm restart from a finished student's SWA weights
        assert INIT and STAGE == "joint", "continue: needs V4_INIT, joint stage"
        CFG.update({"lr": CFG["lr"] * LR_SCALE, "head_lr": CFG["head_lr"] * LR_SCALE})
        sf.update({"init_mode": "continue (strict, all heads + crop copy, fresh Adam)", "lr_scale": LR_SCALE})
    G4["surpass_f"] = sf
    CFG["version"] = f"{CFG['version']}_sf_{STAGE}" + ("_ext" if EXT else "") + ("_noise" if NOISE else "")
# KD-VIEW / KD-CV (ledger rows): V4_VIEW=consistent trains the global path on the view its targets came from: 48
# stratified random windows (V4_TRAIN_WINDOWS), the 320 eval centre crop (train_res = eval_res, same affine path, no
# geometry), intensity augmentation only (gamma 0.25, gain 0.10); V4_VIEW_MIRROR = mirror probability (default 0).
# BN refresh follows (train_windows eval centres at train_res = the 320 crop). Crop branch: same draws, no geometry.
# V4_VIEW=windows (KD-W48): E4's view (384 whole field, full augmentation) with only the window count changed.
VIEW = os.environ.get("V4_VIEW")
# E4-ALL (ledger row): V4_ALLDATA=1 trains on every labelled study (folds 0-5, 4,349; gold never); fold 0 is still
# scored every epoch but is in-sample (monitoring only, LB judges). Steps per epoch, BN refresh and gap sample follow.
ALLDATA = os.environ.get("V4_ALLDATA") == "1"
if ALLDATA:
    assert IMP == "none" and not EXT and not EVAL_ONLY and TARGETS_FILE, "all-data: plain KD lanes only"
    G4["all_data"] = "train = folds 0-5 (4,349 labelled studies); fold-0 scores in-sample"
    CFG["version"] = f"{CFG['version']}_all"
# K-fold (2026-10-22): V4_VAL_FOLD=k holds out fold k instead of fold 0 (train = the other five folds). Steven-v4 validation
# labels exist for fold 0 only, so for k != 0 the val_* scores use label-v2 (all cells) and only lv2_* is comparable.
VAL_FOLD = os.environ.get("V4_VAL_FOLD", "0")
if VAL_FOLD != "0":
    assert VAL_FOLD in "12345" and len(VAL_FOLD) == 1 and not ALLDATA and not EVAL_ONLY and TARGETS_FILE, "k-fold: KD lanes only"
    G4["val_fold"] = f"fold {VAL_FOLD} held out (train = the other five folds); val_* = label-v2, no Steven labels outside fold 0"
    CFG["version"] = f"{CFG['version']}_vf{VAL_FOLD}"
if VIEW == "windows":
    assert ARM == "l4copy" and IMP == "none" and BB == "r50" and TARGETS_FILE and not NOISE, VIEW
    CFG["train_windows"] = int(os.environ.get("V4_TRAIN_WINDOWS", "48"))
    G4["kd_view"] = {k: CFG[k] for k in ("train_windows", "train_res", "aug")}
    CFG["version"] = f"{CFG['version']}_w{CFG['train_windows']}"
elif VIEW:
    assert VIEW == "consistent" and ARM == "l4copy" and IMP == "none" and BB == "r50" and TARGETS_FILE and not NOISE, VIEW
    _mp = float(os.environ.get("V4_VIEW_MIRROR", "0"))
    CFG.update({"train_windows": int(os.environ.get("V4_TRAIN_WINDOWS", "48")), "train_res": CFG["eval_res"],
                "train_crop": True,
                "aug": {"mirror_p": _mp, "rot_deg": 0.0, "scale": 0.0, "shift": 0.0, "gamma": 0.25, "gain": 0.10}})
    G4["kd_view"] = {k: CFG[k] for k in ("train_windows", "train_res", "train_crop", "aug")}
    CFG["version"] = f"{CFG['version']}_view{CFG['train_windows']}" + ("m" if _mp else "")
if IMP == "fuse":
    G4.update({"imp": IMP, "pair": {"max_mm": 1.5, "max_deg": 5.0, "max_inplane_mm": 2.0, "fluid_differs": True},
               "partner": "centre row paired -> partner triplet P[c]-1..P[c]+1 clipped to the partner series",
               "fuse": "h + MLP([h, hp, h - hp]) (3d -> 512 -> d, GELU, output zero-init), global and crop paths "
                       "separate, before LayerNorm, head LR; partner encoded in the same backbone call",
               "partner_dropout": 0.2, "fuse_hidden": 512})
if IMP in ("silw", "hill"):
    G4.update({"imp": IMP, "silent_cells": "rule state SIL and label-v2 y < 0.5, training studies",
               "rule_states_sha256": "ce4473c71f3ea100db021805750c6bb9038bc7b0e56a143c81516edf2d29e77b",
               "loss_denominator": "cell count"})
    G4.update({"silent_weight": 0.25} if IMP == "silw" else
              {"hill_lambda": 1.5, "silent_loss": "y softplus(-z) + (1 - y)(1.5 - p) p^2"})
if IMP == "mcpair":
    G4.update({"imp": IMP, "pair": {"max_mm": 1.5, "max_deg": 5.0, "max_inplane_mm": 2.0, "fluid_differs": True},
               "partner": "per channel: row r -> P[r], zero (after normalisation) where unpaired", "partner_dropout": 0.2,
               "conv1": "6 ch, partner half zero-init", "seq_bias": "[own fluid x centre paired, 12] on attention logits"})
# RANK (ledger row, 2026-10-22; the teammate's "compare one study with other studies" idea): + V4_RANK x a pairwise
# RankNet loss on the global logits. Per finding, each study in the batch is compared with a FIFO bank of the last
# V4_RANK_BANK studies' detached global logits and targets; pairs whose targets differ by >= V4_RANK_MARGIN (both
# weights > 0) add softplus(-(s_i - s_j)), i the higher target; mean over a finding's pairs, then over findings with pairs.
RANK_W = float(os.environ.get("V4_RANK", "0"))
RANK_MARGIN, RANK_BANK = float(os.environ.get("V4_RANK_MARGIN", "0.15")), int(os.environ.get("V4_RANK_BANK", "512"))
if RANK_W:
    assert IMP == "none" and TARGETS_FILE and RANK_BANK > 0 and 0 < RANK_MARGIN < 1, "RANK: the KD recipes only"
    G4["rank"] = {"weight": RANK_W, "margin": RANK_MARGIN, "bank": RANK_BANK, "logits": "global",
                  "loss": "RankNet softplus(-(s_hi - s_lo)) vs a detached FIFO bank, mean per finding then over findings"}
    CFG["version"] = f"{CFG['version']}_rank{RANK_W:g}"
# RUN-C (2026-10-22): two extra training signals on the R7-soup student, both off by default; the inference model is
# unchanged (extra parameters are training-only or foldable).
# V4_HYDRA=<member targets json> (+ V4_HYDRA_SHA; build_hydra_targets.py: r384, r224, r6, reader): one extra linear head
# per teacher member on the main global head's attention-pooled features, soft BCE on that member's probabilities,
# V4_HYDRA_W per head (default 0.125: 0.5 in total over 4 members). Heads start as copies of the main head (Hydra phase 2,
# Tran et al. 2020). Linear heads on the same pooled features: the mean logit of main + members equals one head with
# averaged weights (foldable for the kernel); every scored point is read both ways (main head; *_hydra = mean).
# V4_FKD=<Raptor-384 checkpoint>: feature distillation on the final per-window embedding: a linear projector maps each
# training window's student feature (after norm) to the frozen Raptor-384's pooled feature of the same augmented window
# (its 320 centre crop, no grad); loss V4_FKD_W x (1 - cosine), ramped linearly from 0 over epoch 1, all batches
# (competition + fastMRI). Projector dropped after training.
HYDRA, HYDRA_W = os.environ.get("V4_HYDRA"), float(os.environ.get("V4_HYDRA_W", "0.125"))
FKD, FKD_W = os.environ.get("V4_FKD"), float(os.environ.get("V4_FKD_W", "0.5"))
# C-A (2026-10-22): V4_AKD_W > 0 (needs V4_FKD): attention distillation: per finding, KL(Raptor-384's softmax over the
# study's windows || the student's), Raptor's own norm + attention head on the features it already computes for the FKD
# loss; weight V4_AKD_W, same epoch-1 ramp. Teaches which windows matter per finding.
AKD_W = float(os.environ.get("V4_AKD_W", "0"))
HY, HT, TEACHER, T_HEAD, FKD_D, LAST_AUX = None, None, None, None, 0, {}
AUX_NAMES = ("hydra", "fkd", "akd")
if HYDRA or FKD:
    assert BB == "r7cnx" and TARGETS_FILE and IMP == "none" and not RANK_W, "RUN-C: the R7 KD recipe only"
if HYDRA:
    assert r.sha(Path(HYDRA)) == os.environ["V4_HYDRA_SHA"], f"{HYDRA} hash differs"
    HY = json.loads(Path(HYDRA).read_text()); assert HY["targets_in_order"] == r.TARGETS
    G4["hydra"] = {"file": Path(HYDRA).name, "sha256": os.environ["V4_HYDRA_SHA"], "members": HY["members"],
                   "w_per_head": HYDRA_W, "init": "copies of the main global head"}
    CFG["version"] = f"{CFG['version']}_hydra"
if FKD:
    G4["fkd"] = {"teacher": Path(FKD).name, "w": FKD_W, "ramp": "linear over epoch 1",
                 "loss": "1 - cos(linear(student window feature), Raptor-384 pooled feature of the 320 centre crop)"}
    CFG["version"] = f"{CFG['version']}_fkd"
if AKD_W:
    assert FKD, "AKD: the teacher of V4_FKD"
    G4["akd"] = {"w": AKD_W, "ramp": "linear over epoch 1",
                 "loss": "KL(teacher || student) of the per-finding softmax over windows, mean over studies and findings"}
    CFG["version"] = f"{CFG['version']}_akd{AKD_W:g}"


def load_fkd_teacher(dev):
    """RUN-C: Raptor-384 backbone (pooled features, frozen), as online_kd.load_teacher("384") without its heads."""
    import timm
    global TEACHER, FKD_D
    assert r.sha(Path(FKD)) == BACKBONES["rap2"]["init_sha256"], f"{FKD} hash differs"
    ck = torch.load(FKD, map_location="cpu", weights_only=True); assert list(ck["lab"]) == r.TARGETS
    TEACHER = timm.create_model("coatnet_rmlp_2_rw_384", pretrained=False, num_classes=0, img_size=320)
    TEACHER.load_state_dict({k[9:]: v for k, v in ck["model"].items() if k.startswith("backbone.")}, strict=True)
    TEACHER = TEACHER.eval().requires_grad_(False).to(dev).to(memory_format=torch.channels_last)
    FKD_D = TEACHER.num_features
    if AKD_W:                                                # C-A: Raptor's norm + attention head (nartaa's head)
        global T_HEAD
        T_HEAD = nn.ModuleDict({"norm": nn.LayerNorm(FKD_D), "att": nn.Sequential(nn.Linear(FKD_D, 256), nn.Tanh(),
                                                                                  nn.Dropout(0.0), nn.Linear(256, 12))})
        T_HEAD.load_state_dict({k: v for k, v in ck["model"].items() if k.startswith(("norm.", "att."))}, strict=True)
        T_HEAD = T_HEAD.eval().requires_grad_(False).to(dev)


class DualBN4(v2.DualBN):
    """v2's DualBN (crop statistics, self.crop flag) with the crop branch's affine chosen by mode:
    shared (v2: the local loss trains the shared affine), detached (shared values, no gradient), own (crop_weight/bias)."""

    def __init__(self, bn, mode):
        super().__init__(bn)
        self.mode = mode
        if mode == "own":
            self.crop_weight = nn.Parameter(bn.weight.detach().clone())
            self.crop_bias = nn.Parameter(bn.bias.detach().clone())
        # BB-CMP: timm BatchNormAct2d (CoAtNet MBConv, EfficientNet) = BN, then its drop and activation (no parameters)
        self.post = nn.Sequential(bn.drop, bn.act) if hasattr(bn, "act") else None

    def forward(self, x):
        if not self.crop:
            y = nn.BatchNorm2d.forward(self, x)
        else:
            f = 0.0
            if self.training:
                self.crop_nbt.add_(1)
                f = 1.0 / float(self.crop_nbt) if self.momentum is None else self.momentum
            if self.mode == "own":
                w, b = self.crop_weight, self.crop_bias
            elif self.mode == "detached":
                w, b = self.weight.detach(), self.bias.detach()
            else:
                w, b = self.weight, self.bias
            y = F.batch_norm(x, self.crop_mean, self.crop_var, w, b, self.training, f, self.eps)
        return y if self.post is None else self.post(y)


class LoRAConv(nn.Conv2d):
    """Conv2d whose crop pass (self.crop) uses the shared weight detached plus, for rank > 0, a low-rank delta
    B A x alpha / rank (A kaiming, B zero: the crop path starts as the shared conv). Dense mode (after SWA) uses the
    averaged dense delta in lora_delta instead of the factors. The global pass is the plain conv."""

    def __init__(self, conv, rank, alpha):
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                         conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
        self.load_state_dict(conv.state_dict())
        self.crop, self.rank, self.dense = False, rank, False
        if rank:
            assert conv.kernel_size == (1, 1) and conv.groups == 1
            self.scale = alpha / rank
            self.lora_A = nn.Parameter(torch.empty(rank, conv.in_channels))
            self.lora_B = nn.Parameter(torch.zeros(conv.out_channels, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.register_buffer("lora_delta", torch.zeros_like(self.weight))
            self.register_buffer("lora_dense", torch.zeros((), dtype=torch.bool))

    def delta(self):
        return (self.lora_B @ self.lora_A).view_as(self.weight) * self.scale

    def forward(self, x):
        if not self.crop:
            return super().forward(x)
        w = self.weight.detach()
        if self.rank:
            w = w + (self.lora_delta if self.dense else self.delta())
        return self._conv_forward(x, w, None if self.bias is None else self.bias.detach())


def wrap_bns(module, mode):
    out = []
    for name, ch in module.named_children():
        if isinstance(ch, nn.BatchNorm2d):
            d = DualBN4(ch, mode(name)); setattr(module, name, d); out.append(d)
        else:
            out += wrap_bns(ch, mode)
    return out


def _get(mod, path):
    for p in path.split("."):
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return mod


def _set(mod, path, new):
    *a, last = path.split(".")
    parent = _get(mod, ".".join(a)) if a else mod
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


class Fuse(nn.Module):
    """fuse: delta = W2 GELU(W1 [h, hp, h - hp]); W2, b2 zero (the model starts as v4). W1 drawn from its own generator."""

    def __init__(self, d, hid, seed):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(3 * d, hid), nn.Linear(hid, d)
        g = torch.Generator().manual_seed(CFG["seed"] * 10 + seed)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5), generator=g)
        nn.init.uniform_(self.fc1.bias, -1 / math.sqrt(3 * d), 1 / math.sqrt(3 * d), generator=g)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, h, hp):
        return self.fc2(F.gelu(self.fc1(torch.cat([h, hp, h - hp], -1))))


class RaptorGL4(r.Raptor):
    def __init__(self, res, pretrained):
        super().__init__(res, pretrained and BB not in PUB_INIT)   # backbone, then 1e's head (same init order as 1e / v2)
        bb, d = self.backbone, self.backbone.num_features
        if BB == "rap2" and pretrained:                      # PUB-KD: nartaa's 0.949 weights, backbone + global head
            ip = Path(os.environ.get("V4_RAPTOR_DIR", "/work/pubckpt")) / BACKBONES["rap2"]["init"]
            assert r.sha(ip) == BACKBONES["rap2"]["init_sha256"], f"{ip} hash differs"
            ck = torch.load(ip, map_location="cpu", weights_only=True)
            assert ck["lab"] == r.TARGETS and ck["arch"].startswith("coatnet_rmlp_2_rw_384")
            self.load_state_dict(ck["model"], strict=True)
        if BB == "r7cnx" and pretrained:                     # R7-D: heliosli's R7 encoder (fold 0), backbone only
            # SOUP-R7 (2026-10-22): V4_R7_INIT / V4_R7_INIT_SHA swap in another timm encoder file (R7 f0-f3 soup)
            ip = Path(os.environ.get("V4_RAPTOR_DIR", "/work/pubckpt")) / os.environ.get("V4_R7_INIT", BACKBONES["r7cnx"]["init"])
            assert r.sha(ip) == os.environ.get("V4_R7_INIT_SHA", BACKBONES["r7cnx"]["init_sha256"]), f"{ip} hash differs"
            bb.load_state_dict(torch.load(ip, map_location="cpu", weights_only=True), strict=True)
        self.convs = []
        if ARM == "adapter":
            bb.bn1 = DualBN4(bb.bn1, "detached"); self.bns = [bb.bn1]
            for li in (1, 2):
                self.bns += wrap_bns(getattr(bb, f"layer{li}"), lambda n: "detached")
            for li in (3, 4):
                layer = getattr(bb, f"layer{li}")
                self.bns += wrap_bns(layer, lambda n: "own")
                for blk in layer:
                    for nm in ("conv1", "conv2", "conv3"):
                        c = getattr(blk, nm); one = c.kernel_size == (1, 1)
                        w = LoRAConv(c, G4["lora_rank"] if one else 0, G4["lora_alpha"]); setattr(blk, nm, w)
                        self.convs.append(w)
                    if blk.downsample is not None:
                        w = LoRAConv(blk.downsample[0], G4["lora_rank"], G4["lora_alpha"]); blk.downsample[0] = w
                        self.convs.append(w)
            assert sum(c.rank > 0 for c in self.convs) == 20 and len(self.convs) == 29, len(self.convs)
        else:
            self.bns = wrap_bns(bb, lambda n: "shared")
        assert len(self.bns) == BACKBONES[BB]["bns"], len(self.bns)   # ResNet-50: stem + 16 blocks x 3 + 4 downsample
        self.cbns = []
        if BB != "r50":                                      # BB-CMP: crop-only copy of the last stage, swapped in
            self.crop_parts = nn.ModuleDict({p.replace(".", "_"): copy.deepcopy(_get(bb, p))
                                             for p in BACKBONES[BB]["last"]})
            self.cbns = [m for m in self.crop_parts.modules() if isinstance(m, DualBN4)]
            for m in self.cbns:
                m.crop = True
        elif ARM in ("l4copy", "l3copy"):                    # crop-only layer4 (l3copy: + layer3): ImageNet init,
            self.layer4_c = copy.deepcopy(bb.layer4)         # their own BN (crop stats)
            if ARM == "l3copy":
                self.layer3_c = copy.deepcopy(bb.layer3)
            self.cbns = [m for n in ("layer3_c", "layer4_c") if hasattr(self, n)
                         for m in getattr(self, n).modules() if isinstance(m, DualBN4)]
            for m in self.cbns:
                m.crop = True
        self.norm_l = nn.LayerNorm(d)
        self.clsV = nn.Parameter(torch.zeros(12, d)); self.clsvb = nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.clsV, std=0.02)
        if BB == "rap2" and pretrained:                      # PUB-KD: crop head starts as the public global head
            self.norm_l.load_state_dict(self.norm.state_dict())
            with torch.no_grad():
                self.clsV.copy_(self.clsW); self.clsvb.copy_(self.clsb)
        if IMP == "mcpair":                                  # no RNG draws: v4's init otherwise unchanged
            w = bb.conv1.weight.data
            bb.conv1.weight = nn.Parameter(torch.cat([w, torch.zeros_like(w)], 1)); bb.conv1.in_channels = 6
            self.seq_bias = nn.Parameter(torch.zeros(4, 12))
        if IMP == "fuse":                                    # own generator: v4's global RNG stream unchanged
            st = torch.get_rng_state()
            self.fuse_g, self.fuse_c = Fuse(d, G4["fuse_hidden"], 1), Fuse(d, G4["fuse_hidden"], 2)
            torch.set_rng_state(st)
            self.fuse_stats = [0.0] * 4                      # running sums: |delta| / |h| global, n, crop, n
        self.hydra_infer, self._hg, self._pooled = False, None, None
        if HYDRA:                                            # RUN-C: member heads = copies of the main head (no RNG)
            m = len(HY["members"])
            self.hW = nn.Parameter(self.clsW.detach().clone()[None].repeat(m, 1, 1))
            self.hb = nn.Parameter(self.clsb.detach().clone()[None].repeat(m, 1))
        if FKD:                                              # own generator: the init RNG stream unchanged
            st = torch.get_rng_state(); self.fkd_proj = nn.Linear(d, FKD_D); torch.set_rng_state(st)

    def _flags(self, crop):
        for m in self.bns + self.convs:
            m.crop = crop

    def set_dense(self):
        for c in self.convs:
            if c.rank:
                c.dense = bool(c.lora_dense)

    def stem(self, x):
        bb = self.backbone
        return bb.maxpool(bb.act1(bb.bn1(bb.conv1(x))))

    def encode(self, x, crop):
        bb = self.backbone
        if not crop:
            return bb(x)
        self._flags(True)
        if BB != "r50":                                      # BB-CMP: the crop copies stand in for the last stage
            orig = {p: _get(bb, p) for p in BACKBONES[BB]["last"]}
            try:
                for p in orig:
                    _set(bb, p, self.crop_parts[p.replace(".", "_")])
                return bb(x)
            finally:
                for p, mod in orig.items():
                    _set(bb, p, mod)
                self._flags(False)
        try:
            if ARM == "adapter":
                with torch.no_grad():
                    h = bb.layer2(bb.layer1(self.stem(x)))
                h = bb.layer4(bb.layer3(h))
            elif ARM == "l4copy":
                h = self.layer4_c(bb.layer3(bb.layer2(bb.layer1(self.stem(x)))))
            elif ARM == "l3copy":
                h = self.layer4_c(self.layer3_c(bb.layer2(bb.layer1(self.stem(x)))))
            else:
                h = bb.forward_features(x)
            return bb.forward_head(h)
        finally:
            self._flags(False)

    def forward(self, x, b, xc, fidc, wc, wt=None):
        """-> (g, l) logits [b, 12]; l = 0 on the findings without crops. wt (mcpair): window type [b * k]."""
        kc = xc.shape[0] // b
        if IMP == "fuse":                                    # own + paired partners in one backbone call per path
            ng, nc = x.shape[0], xc.shape[0]
            fg, fc = self.encode(torch.cat([x, wt["xp"]]), False), self.encode(torch.cat([xc, wt["xcp"]]), True)
            hg = self.norm(self._fuse(fg[:ng], fg[ng:], wt["ig"], self.fuse_g, 0).view(b, ng // b, -1))
            hc = self.norm_l(self._fuse(fc[:nc], fc[nc:], wt["ic"], self.fuse_c, 2).view(b, kc, -1))
            wt = None
        else:
            hg = self.norm(self.encode(x, False).view(b, x.shape[0] // b, -1))
            hc = self.norm_l(self.encode(xc, True).view(b, kc, -1))
        if NOISE and self.training:                          # SURPASS-F noise: feature dropout before both heads
            q = G4["surpass_f"]["noise"]["feature_dropout"]
            hg, hc = F.dropout(hg, q), F.dropout(hc, q)
        g = self.head_g(hg, b, wt)
        lc = ((hc * self.clsV[fidc]).sum(-1) + self.clsvb[fidc]).float()                # [b, kc], own finding
        l = torch.zeros(b, 12, device=x.device).scatter_add(1, fidc, lc * wc)
        return g, l

    def head_g(self, hg, b, wt=None):
        al = self.att(hg).float()
        if self.training:                                    # C-A: the student's attention logits [b, k, 12]
            self._al = al
        if wt is not None:
            al = al + self.seq_bias[wt.view(b, -1)]
        a = torch.softmax(al, dim=1).to(hg.dtype)                                          # 1e's head
        pooled = torch.einsum("bkn,bkf->bnf", a, hg)
        if self.training:                                    # RUN-C: read by the Hydra / FKD losses
            self._hg, self._pooled = hg, pooled
        g = ((pooled * self.clsW).sum(-1) + self.clsb).float()
        if HYDRA and self.hydra_infer:                       # mean logit of main + member heads (= folded weights)
            g = (g + (torch.einsum("bnf,mnf->bmn", pooled, self.hW) + self.hb).float().sum(1)) / (1 + self.hW.shape[0])
        return g

    def forward_global(self, x, b):
        """SURPASS-F extra studies: the global path only (no crop pass) -> g logits [b, 12]."""
        hg = self.norm(self.encode(x, False).view(b, x.shape[0] // b, -1))
        if NOISE and self.training:
            hg = F.dropout(hg, G4["surpass_f"]["noise"]["feature_dropout"])
        return self.head_g(hg, b)

    def _fuse(self, h, hp, idx, mod, slot):
        """fuse: h[idx] += mod(h[idx], hp); logs sum |delta| / |h| and count (slot 0 global, 2 crop)."""
        if len(idx) == 0:
            return h
        delta = mod(h[idx], hp).to(h.dtype)
        with torch.no_grad():
            self.fuse_stats[slot] += float((delta.float().norm(dim=1) / h[idx].float().norm(dim=1).clamp_min(1e-6)).sum())
            self.fuse_stats[slot + 1] += len(idx)
        return h.index_add(0, idx, delta)

    def param_sets(self):
        """-> (shared backbone, global head, local backbone-side, local head) parameter lists, disjoint and complete."""
        sets = {"bb": [], "gh": [], "lb": [], "lh": []}
        for n, p in self.named_parameters():
            if n.startswith(("layer4_c.", "layer3_c.", "crop_parts.")):
                sets["lb"].append(p)
            elif n.startswith("backbone."):
                sets["lb" if ("lora_" in n or ".crop_" in n) else "bb"].append(p)
            elif n.split(".")[0] in ("norm_l", "clsV", "clsvb"):
                sets["lh"].append(p)
            else:
                sets["gh"].append(p)
        assert sum(len(v) for v in sets.values()) == len(list(self.parameters()))
        return sets


PAIRS = {}                                               # mcpair: study -> (partner row [96], window type base [96])


def pair_table():
    """mcpair: per cache row of SagF / SagNF / CorF / CorNF the complementary-contrast row at the same physical place:
    the nearest row of the other series along this series' slice normal (DICOM IPP, mm), kept if |d| <= 1.5 mm, the two
    normals within 5 deg, the in-plane corner offset <= 2 mm and the two series' fluid flags differ; else -1 (axial: -1).
    Also the row's own fluid flag (window type = 2 x fluid + paired)."""
    cfg, out = G4["pair"], {}
    cos = math.cos(math.radians(cfg["max_deg"]))
    unit = lambda v: v / np.linalg.norm(v)
    for i in range(CFG.get("shards", r.SHARDS)):
        for line in (r.CACHE / f"shard_{i}" / "manifest.jsonl").read_text().splitlines():
            m = json.loads(line); sl = m["slots"]
            P = np.full(MAXS, -1, np.int64); fl = np.zeros(MAXS, np.int64)
            lo, hi = np.arange(MAXS), np.arange(MAXS)                    # fuse: each row's series bounds
            for s in sl:
                fl[s["row_start"]:s["row_start"] + s["k"]] = s.get("fluid") == "1"
                lo[s["row_start"]:s["row_start"] + s["k"]] = s["row_start"]
                hi[s["row_start"]:s["row_start"] + s["k"]] = s["row_start"] + s["k"] - 1
            for a, b in ((0, 1), (2, 3)):
                A, B = sl[a], sl[b]
                if not A.get("geometry") or not B.get("geometry") or A.get("fluid") == B.get("fluid"):
                    continue
                for X, Y in ((A, B), (B, A)):
                    assert len(X["geometry"]) <= X["k"]
                    iX, iY = np.array(X["geometry"][0]["iop"]), np.array(Y["geometry"][0]["iop"])
                    n, nY = unit(np.cross(iX[:3], iX[3:])), unit(np.cross(iY[:3], iY[3:]))
                    if abs(n @ nY) < cos:
                        break
                    px = np.array([g["ipp"] for g in X["geometry"]]); py = np.array([g["ipp"] for g in Y["geometry"]])
                    d = (px @ n)[:, None] - (py @ n)[None]
                    j = np.abs(d).argmin(1)                                   # ties (duplicate picks): first row
                    off = py[j] - px; off = off - (off @ n)[:, None] * n
                    ok = (np.abs(d[np.arange(len(j)), j]) <= cfg["max_mm"]) & (np.linalg.norm(off, axis=1) <= cfg["max_inplane_mm"])
                    P[X["row_start"] + np.where(ok)[0]] = Y["row_start"] + j[ok]
            out[m["study_id"]] = (P, fl, lo, hi)
    return out


class GLStudiesI(v2.GLStudies):
    """silw / hill: + the study's index in the dataset (the silent-cell table row)."""

    def __getitem__(self, i):
        return super().__getitem__(i) + (torch.tensor(i),)


class GLStudies6(v2.GLStudies):
    """mcpair: x [k, 6, H, W], xc [kc, 6, H, W] = own triplet + partner triplet (row by row through PAIRS; zero where
    unpaired), + partner flags okg [k, 3], okc [kc, 3] and window type [k]. Training: partner dropout per window / crop."""

    def __getitem__(self, i):
        x, sag, y, w, ctx, xc, sagc, ctxc, cen, fid, wt = super().__getitem__(i)
        m = self.meta[self.ids[i]]; P, fl = PAIRS[self.ids[i]][:2]
        vol = self._mm[m["file"]][m["row"]:m["row"] + MAXS]
        rg, rc = (np.stack([c - 1, c, c + 1], 1) for c in (self._c, self._S))
        okg, okc = P[rg] >= 0, P[rc] >= 0
        if self.train:
            okg &= (np.random.rand(len(rg)) >= G4["partner_dropout"])[:, None]
            okc &= (np.random.rand(len(rc)) >= G4["partner_dropout"])[:, None]
        pg = vol[np.where(okg, P[rg], 0)] * okg[..., None, None]
        pc = vol[np.where(okc, P[rc], 0)] * okc[..., None, None]
        typ = 2 * fl[self._c] + okg[:, 1]
        return (torch.cat([x, torch.from_numpy(pg)], 1), sag, y, w, ctx, torch.cat([xc, torch.from_numpy(pc)], 1), sagc,
                ctxc, cen, fid, wt, torch.from_numpy(okg), torch.from_numpy(okc), torch.from_numpy(typ))


class GLStudiesF(v2.GLStudies):
    """fuse: + per window / crop the partner triplet [P[c] - 1, P[c], P[c] + 1] clipped to the partner series (a normal
    window of that series; zeros where the centre row is unpaired or, in training, dropped) and its flag."""

    def __getitem__(self, i):
        out = super().__getitem__(i)
        m = self.meta[self.ids[i]]; P, _, lo, hi = PAIRS[self.ids[i]]
        vol = self._mm[m["file"]][m["row"]:m["row"] + MAXS]

        def partner(c):
            ok = P[c] >= 0
            if self.train:
                ok &= np.random.rand(len(c)) >= G4["partner_dropout"]
            pc = np.where(ok, P[c], 0)
            rows = np.clip(pc[:, None] + np.arange(-1, 2), lo[pc][:, None], hi[pc][:, None])
            return torch.from_numpy(np.ascontiguousarray(vol[rows]) * ok[:, None, None, None].astype(np.uint8)), torch.from_numpy(ok)

        return out + partner(self._c) + partner(self._S)


def mirror6(x, sag):
    """r.mirror per triplet: sagittal reverses the slice order within own and partner triplets, others flip width."""
    return torch.where(sag.view(-1, 1, 1, 1), x.view(x.shape[0], 2, 3, *x.shape[-2:]).flip(2).flatten(1, 2), x.flip(-1))


def prepare_gl6(x, sag, xc, sagc, cen, fid, okg, okc, train, dev, gres=None):
    """v2.prepare_gl on 6 channels (same draws, same order); unpaired partner channels exactly 0 after normalisation."""
    b, k, kc = x.shape[0], x.shape[1], xc.shape[1]
    x = x.to(dev, non_blocking=True).flatten(0, 1); xc = xc.to(dev, non_blocking=True).flatten(0, 1)
    sag = sag.to(dev).flatten(); sagc = sagc.to(dev).flatten()
    okg = okg.to(dev).flatten(0, 1).float(); okc = okc.to(dev).flatten(0, 1).float()
    cen = cen.to(dev).view(-1, 2)
    s = torch.as_tensor(v2.BOXN, dtype=torch.float32, device=dev)[fid.to(dev).flatten()].view(-1, 1, 1) / IMG
    px = torch.stack([(cen[:, 1] + 0.5) / IMG * 2 - 1, (cen[:, 0] + 0.5) / IMG * 2 - 1], 1)
    a, cres = CFG["aug"], v2.GL["crop_res"]
    if train:
        flip = torch.rand(b, device=dev) < a["mirror_p"]
        fg, fc = flip.repeat_interleave(k), flip.repeat_interleave(kc)
        x = torch.where(fg.view(-1, 1, 1, 1), mirror6(x, sag), x)
        xc = torch.where(fc.view(-1, 1, 1, 1), mirror6(xc, sagc), xc)
        okg = torch.where((fg & sag).view(-1, 1), okg.flip(1), okg)          # the flags follow the slice reversal
        okc = torch.where((fc & sagc).view(-1, 1), okc.flip(1), okc)
        px = torch.stack([torch.where(fc & ~sagc, -px[:, 0], px[:, 0]), px[:, 1]], 1)
    x = x.float().div_(255.0); xc = xc.float().div_(255.0)
    if train:
        res = CFG["train_res"]
        ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(a["rot_deg"])
        sc = 1 + (torch.rand(b, device=dev) * 2 - 1) * a["scale"]
        tx, ty = [(torch.rand(b, device=dev) * 2 - 1) * a["shift"] * 2 for _ in range(2)]
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        A = torch.stack([torch.stack([cos, -sin], 1), torch.stack([sin, cos], 1)], 1)
        th_g = torch.cat([A, torch.stack([tx, ty], 1)[..., None]], 2).repeat_interleave(k, 0)
        th_c = torch.cat([s * A.repeat_interleave(kc, 0), px[..., None]], 2)
        x = F.grid_sample(x, F.affine_grid(th_g, (b * k, 6, res, res), align_corners=False), mode="bilinear",
                          padding_mode="zeros", align_corners=False)
    else:
        res = gres or CFG["eval_res"]
        if res != x.shape[-1]:
            lo = (x.shape[-1] - res) // 2
            x = x[..., lo:lo + res, lo:lo + res]
        th_c = torch.cat([s * torch.eye(2, device=dev).expand(b * kc, 2, 2), px[..., None]], 2)
    xc = F.grid_sample(xc, F.affine_grid(th_c, (b * kc, 6, cres, cres), align_corners=False), mode="bilinear",
                       padding_mode="zeros", align_corners=False)
    if train:
        gamma = torch.exp((torch.rand(b, device=dev) * 2 - 1) * a["gamma"])
        gain = 1 + (torch.rand(b, device=dev) * 2 - 1) * a["gain"]
        x = (x.clamp_min(1e-4).pow(gamma.repeat_interleave(k).view(-1, 1, 1, 1))
             * gain.repeat_interleave(k).view(-1, 1, 1, 1)).clamp_(0, 1)
        xc = (xc.clamp_min(1e-4).pow(gamma.repeat_interleave(kc).view(-1, 1, 1, 1))
              * gain.repeat_interleave(kc).view(-1, 1, 1, 1)).clamp_(0, 1)
    mean, std = r.MEAN.to(dev).repeat(1, 2, 1, 1), r.STD.to(dev).repeat(1, 2, 1, 1)
    x, xc = (x - mean) / std, (xc - mean) / std
    x = torch.cat([x[:, :3], x[:, 3:] * okg[..., None, None]], 1).contiguous(memory_format=torch.channels_last)
    xc = torch.cat([xc[:, :3], xc[:, 3:] * okc[..., None, None]], 1).contiguous(memory_format=torch.channels_last)
    return x, xc, px.view(b, kc, 2)


def make_ds(ids, meta, labels, k, train, tab, top1=False):
    cls = {"mcpair": GLStudies6, "fuse": GLStudiesF}.get(IMP, v2.GLStudies)
    return cls(ids, meta, labels, k, train, tab, None, top1)


def prep(batch, train, dev, gres=None, nopartner=False):
    """-> global images, crop images, window type (mcpair) / partner dict (fuse) / None."""
    x, sag, _, _, _, xc, sagc, _, cen, fid, _ = batch[:11]
    if IMP == "fuse":                                        # partners ride along as extra windows: same draws as v4
        xp, okg, xcp, okc = batch[11:15]
        b, k, kc = x.shape[0], x.shape[1], xc.shape[1]
        cat = lambda a, p: torch.cat([a, p], 1)
        xg, xcr, _ = v2.prepare_gl(cat(x, xp), cat(sag, sag), cat(xc, xcp), cat(sagc, sagc), cat(cen, cen),
                                   cat(fid, fid), train, dev, gres=gres)
        cl = torch.channels_last
        xg, xcr = xg.view(b, 2 * k, *xg.shape[1:]), xcr.view(b, 2 * kc, *xcr.shape[1:])
        ig = torch.zeros(0, dtype=torch.long, device=dev) if nopartner else okg.to(dev).flatten().nonzero()[:, 0]
        ic = torch.zeros(0, dtype=torch.long, device=dev) if nopartner else okc.to(dev).flatten().nonzero()[:, 0]
        part = {"xp": xg[:, k:].flatten(0, 1)[ig].contiguous(memory_format=cl), "ig": ig,
                "xcp": xcr[:, kc:].flatten(0, 1)[ic].contiguous(memory_format=cl), "ic": ic}
        return (xg[:, :k].flatten(0, 1).contiguous(memory_format=cl), xcr[:, :kc].flatten(0, 1).contiguous(memory_format=cl),
                part)
    if IMP == "mcpair":
        okg, okc, typ = batch[11:14]
        xg, xcr, _ = prepare_gl6(x, sag, xc, sagc, cen, fid, okg, okc, train, dev, gres)
        return xg, xcr, typ.to(dev).flatten()
    xg, xcr, _ = v2.prepare_gl(x, sag, xc, sagc, cen, fid, train, dev, gres=gres)
    return xg, xcr, None


def silent_table(ids, labels):
    """silw / hill: [n, 12] bool, rule state SIL and label-v2 y < 0.5 (training studies, dataset order)."""
    path = r.ASSETS / "rule_states.csv"
    assert r.sha(path) == G4["rule_states_sha256"], "rule_states.csv hash differs"
    st = {x["StudyInstanceUID"]: [x[t] for t in r.TARGETS] for x in csv.DictReader(path.open())}
    return np.array([[st[s][k] == "SIL" and labels[s][0][k] < 0.5 for k in range(12)] for s in ids])


def silent_loss(z, y, w, m):
    """BCE with the silent assumed-negative cells m changed (silw: x 0.25; hill: Hill negative term); / cell count."""
    e = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    if IMP == "silw":
        e = e * torch.where(m, G4["silent_weight"], 1.0)
    else:
        p = torch.sigmoid(z)
        e = torch.where(m, y * F.softplus(-z) + (1 - y) * (G4["hill_lambda"] - p) * p * p, e)
    return (e * w).sum() / w.sum().clamp_min(1)


def ext_manifest():
    """SURPASS-F: the extra cache's studies in read_manifest's layout (absolute volume paths), + their source path.
    Ids present more than once (FB-F's Colab cache keeps the 4 exams in both archives) are dropped, as on MeluXina."""
    out, dup = {}, set()
    for i in range(EXT_SHARDS):
        root = Path(EXT) / f"shard_{i}"
        for line in (root / "manifest.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if rec["study_id"] in out:
                dup.add(rec["study_id"])
            pos, known = r.positions(rec)
            out[rec["study_id"]] = {"file": str(root / rec["part"]), "row": rec["row_start"], "gold": False,
                                    "mask": np.array(rec["mask"], bool), "pos": pos, "known": known,
                                    "source": rec["source"]}
    assert len(dup) <= 4, sorted(dup)
    print(f"EXT duplicate ids dropped: {sorted(dup)}", flush=True)
    return {k: v for k, v in out.items() if k not in dup}


def ext_crop_table(ids, meta):
    """Placeholder crop windows for the extra studies (no MS1 table): never forwarded (their batches run the global
    path only); the dataset still reads them, so they point at the valid centre slice, image centre, uniform weight."""
    T, tab = v2.GL["top"], {}
    for s in ids:
        c = int(np.clip(r.centres(meta[s]["mask"], 1, False)[0], 1, MAXS - 2))
        tab[s] = (np.full((12, T), c, np.int64), np.full((12, T), IMG // 2, np.int64), np.full((12, T), IMG // 2, np.int64),
                  np.ones((12, T), np.float32))
    return tab


def ext_setup(meta, tab, tlabels):
    """SURPASS-F: extra studies into meta / tab (placeholder crops) -> (trained ids = those with targets, holdout ids)."""
    em = ext_manifest(); assert not set(em) & set(meta); meta.update(em)
    tab.update(ext_crop_table(sorted(em), meta))
    hold_ids = sorted(json.loads(Path(EXT_HOLDOUT).read_text())) if EXT_HOLDOUT else []
    assert set(hold_ids) <= set(em) and not set(hold_ids) & set(tlabels), "holdout must be unseen by the targets"
    ext_ids = sorted(s for s in em if s in tlabels)
    print(f"EXT {len(em)} studies in {EXT_SHARDS} shards: {len(ext_ids)} with targets (trained), holdout "
          f"{len(hold_ids)}, neither {len(em) - len(ext_ids) - len(hold_ids)}", flush=True)
    return ext_ids, hold_ids


@torch.no_grad()
def init_from(model, path):
    """SURPASS-F fine: the pre stage's SWA weights (strict); the crop layer 4 restarts as a copy of the pre-trained
    layer 4, its crop statistics = layer 4's global ones (as v4's crop copy starts from ImageNet's)."""
    res_ = model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
    assert not res_.unexpected_keys and all(k in ("hW", "hb") or k.startswith("fkd_proj.") for k in res_.missing_keys), res_
    if HYDRA and "hW" in res_.missing_keys:                  # RUN-C from a finished student: member heads = copies of
        model.hW.copy_(model.clsW[None].expand_as(model.hW)); model.hb.copy_(model.clsb[None].expand_as(model.hb))   # its trained head
    if INIT_MODE == "continue":
        print(f"INIT {Path(path).name} (strict but RUN-C's new keys: {sorted(res_.missing_keys)}, continue)", flush=True)
        return
    model.layer4_c.load_state_dict(model.backbone.layer4.state_dict())
    for c, g in zip(model.layer4_c.modules(), model.backbone.layer4.modules()):
        if isinstance(c, DualBN4):
            c.crop_mean.copy_(g.running_mean); c.crop_var.copy_(g.running_var); c.crop_nbt.copy_(g.num_batches_tracked)
    print(f"INIT {Path(path).name} (strict); layer4_c <- layer4", flush=True)


class SourceBatches:
    """Batches of one source each (competition / extra), the order shuffled per epoch; is_ext[i] for batch i."""

    def __init__(self, n_comp, n_ext, bs, seed):
        self.n_comp, self.n_ext, self.bs, self.seed, self.epoch = n_comp, n_ext, bs, seed, 0
        self.batches, self.is_ext = [], []

    def set_epoch(self, epoch):
        rng = random.Random(self.seed * 1000 + epoch)
        comp, ext = list(range(self.n_comp)), list(range(self.n_comp, self.n_comp + self.n_ext))
        rng.shuffle(comp); rng.shuffle(ext)
        bl = [(0, comp[i:i + self.bs]) for i in range(0, len(comp), self.bs)]
        bl += [(1, ext[i:i + self.bs]) for i in range(0, len(ext), self.bs)]
        rng.shuffle(bl)
        self.is_ext, self.batches = [e for e, _ in bl], [b for _, b in bl]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def readout(g, l):
    """fused = mean of the two logits on the cropped findings, global elsewhere (sigmoid applied by the caller)."""
    m = torch.as_tensor(CMASK, device=g.device)
    return torch.where(m, 0.5 * (g + l), g)


def lora_dense_average(sds):
    """SWA for the adapter arm (N3): per LoRA conv the mean over epochs of the dense delta B_e A_e x alpha / rank."""
    out = {}
    for k in sds[-1]:
        if k.endswith(".lora_A"):
            base = k[: -len(".lora_A")]
            out[base + ".lora_delta"] = (sum((s[base + ".lora_B"].float() @ s[k].float()) for s in sds) / len(sds)
                                         * G4["lora_alpha"] / G4["lora_rank"]).view_as(sds[-1][base + ".lora_delta"])
            out[base + ".lora_dense"] = torch.ones((), dtype=torch.bool)
    return out


@torch.no_grad()
def predict(model, ids, meta, tab, dev, top1=False, nopartner=False):
    """-> sigmoid [3, n, 12] (fused, global, local); local's effusion / synovitis columns hold the global."""
    model.eval()
    dl = torch.utils.data.DataLoader(make_ds(ids, meta, {}, K, False, tab, top1), batch_size=1,
                                     num_workers=CFG["workers"], pin_memory=True)
    out = np.zeros((3, len(ids), 12), np.float32)
    for i, batch in enumerate(dl):
        xg, xcr, typ = prep(batch, False, dev, nopartner=nopartner)
        fid, wc = v2.to_dev(dev, batch[9], batch[10])
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            g, l = model(xg, 1, xcr, fid, wc, typ)
        out[:, i] = torch.stack([readout(g, l), g, l]).float().sigmoid()[:, 0].cpu().numpy()
    out[2][:, ~CMASK] = out[1][:, ~CMASK]
    return out


@torch.no_grad()
def predict_global(model, ids, meta, tab, dev):
    """SURPASS-F holdout: sigmoid of the global head [n, 12] (eval windows, no crop pass)."""
    model.eval()
    dl = torch.utils.data.DataLoader(make_ds(ids, meta, {}, K, False, tab), batch_size=1,
                                     num_workers=CFG["workers"], pin_memory=True)
    out = np.zeros((len(ids), 12), np.float32)
    tta = os.environ.get("V4_TTA") == "mirror"               # TTA-M (2026-10-22): + the training mirror (r.mirror), mean prob
    for i, batch in enumerate(dl):
        xg, _, _ = prep(batch, False, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            p = model.forward_global(xg, 1).float().sigmoid()[0]
            if tta:                                          # mirror the uint8 windows before normalisation, as in training
                x, sag = batch[0], batch[1]
                mb = [torch.where(sag[..., None, None, None], x.flip(2), x.flip(-1))] + list(batch[1:])
                xm, _, _ = prep(mb, False, dev)
                p = (p + model.forward_global(xm, 1).float().sigmoid()[0]) / 2
            out[i] = p.cpu().numpy()
    return out


def score(p, ids, ref, labels):
    rec = {}
    for j, nm in enumerate(HEADS):
        rec[f"lv2_{nm}"], per = r.macro_auc(p[j], ids, labels)
        rec[f"lv2_{nm}10"] = float(np.nanmean([per[r.TARGETS[k]] for k in CF]))
        rec[f"val_{nm}"] = r.macro_auc(p[j], ids, ref)[0]
        rec[f"lv2_bce_{nm}"] = r.val_bce(p[j], ids, labels); rec[f"val_bce_{nm}"] = r.val_bce(p[j], ids, ref)
        rec[f"lv2_per_{nm}"] = per
    rec["fused_minus_global"] = rec["lv2_fused"] - rec["lv2_global"]
    return rec


@torch.no_grad()
def bn_refresh(model, ids, meta, labels, tab, dev):
    """Global statistics as 1e (16 eval-centre windows at 384, batch 2); crop statistics from all 4 crops per finding."""
    for m in model.bns + model.cbns:
        m.reset_running_stats(); m.momentum = None
    model.train()
    ds = make_ds(ids, meta, labels, CFG["train_windows"], False, tab)
    for batch in torch.utils.data.DataLoader(ds, batch_size=2, num_workers=CFG["workers"]):
        xg, xcr, part = prep(batch, False, dev, gres=CFG["train_res"])
        if IMP == "fuse":                                    # partners share the BN batch, as in training
            xg, xcr = torch.cat([xg, part["xp"]]), torch.cat([xcr, part["xcp"]])
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            model.encode(xg, False); model.encode(xcr, True)


def build(dev, pretrained=True):
    model = RaptorGL4(CFG["train_res"], pretrained=pretrained).to(dev).to(memory_format=torch.channels_last)
    ps = model.param_sets()
    groups = [{"params": ps["bb"] + (ps["lb"] if ARM in ("l4copy", "l3copy") else []), "lr": CFG["lr"]},
              {"params": ps["gh"] + ps["lh"], "lr": CFG["head_lr"]}]
    if "layer_decay" in CFG:                                 # BB-CMP DINOv2: MS1's depth groups; crop copies = their source
        top, by = 1 + len(model.backbone.blocks), {}
        for n, p_ in model.named_parameters():
            if n.startswith("backbone."):
                i = min(r.vit_layer_id(n[len("backbone."):]), top)
            elif n.startswith("crop_parts."):
                q = n.split(".")[1]
                i = 1 + int(q.split("_")[1]) if q.startswith("blocks_") else top
            else:
                continue
            by.setdefault(i, []).append(p_)
        assert sum(len(v) for v in by.values()) == len(ps["bb"]) + len(ps["lb"])
        groups = [{"params": v, "lr": CFG["lr"] * CFG["layer_decay"] ** (top - i)} for i, v in sorted(by.items())] + groups[1:]
        print("LAYER_DECAY " + json.dumps({i: [len(v), CFG["lr"] * CFG["layer_decay"] ** (top - i)] for i, v in sorted(by.items())}), flush=True)
    if ARM == "adapter":
        groups.append({"params": ps["lb"], "lr": G4["local_lr"]})
    opt = torch.optim.AdamW(groups, weight_decay=CFG["wd"])
    clip = ([ps["bb"] + ps["gh"], ps["lb"] + ps["lh"]] if ARM == "adapter" else [list(model.parameters())])
    return model, opt, clip


class RankBank:
    """RANK: FIFO bank of detached global logits, targets and weights [RANK_BANK, 12]; loss() compares the batch with it."""

    def __init__(self, dev):
        self.s, self.y, self.w = (torch.zeros(RANK_BANK, 12, device=dev) for _ in range(3))
        self.pos, self.n, self.pairs = 0, 0, 0.0

    def loss(self, g, y, w):
        g = g.float(); k = self.n
        out, self.pairs = g.sum() * 0, 0.0
        if k:
            d = y[:, None, :] - self.y[None, :k]                                   # [b, k, 12]
            ok = (d.abs() >= RANK_MARGIN) & (w[:, None, :] > 0) & (self.w[None, :k] > 0)
            e = F.softplus(-torch.sign(d) * (g[:, None, :] - self.s[None, :k])) * ok
            np_ = ok.sum((0, 1)); has = np_ > 0
            if has.any():
                out = (e.sum((0, 1))[has] / np_[has]).mean()
            self.pairs = float(np_.float().mean())
        for j in range(g.shape[0]):                                                # push after the loss
            self.s[self.pos], self.y[self.pos], self.w[self.pos] = g[j].detach(), y[j], w[j]
            self.pos = (self.pos + 1) % RANK_BANK; self.n = min(self.n + 1, RANK_BANK)
        return out


def aux_losses(model, xg, batch, dev, ramp):
    """RUN-C: {hydra, fkd, akd} losses for this step (each scaled by its weight; fkd / akd x ramp) into LAST_AUX."""
    LAST_AUX.clear()
    if HYDRA:
        mt = HT[batch[11].to(dev)]                                                  # [b, M, 12]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            hl = torch.einsum("bnf,mnf->bmn", model._pooled, model.hW) + model.hb
        LAST_AUX["hydra"] = (HYDRA_W * F.binary_cross_entropy_with_logits(hl.float(), mt, reduction="none").mean((0, 2)).sum())
    if FKD:
        assert xg.shape[-1] == 384; lo = (384 - 320) // 2
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            v = xg[..., lo:lo + 320, lo:lo + 320].contiguous(memory_format=torch.channels_last)
            ft = torch.cat([TEACHER(c) for c in v.split(32)]).float()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            fs = model.fkd_proj(model._hg.reshape(ft.shape[0], -1))
        LAST_AUX["fkd"] = FKD_W * ramp * (1 - F.cosine_similarity(fs.float(), ft, dim=-1)).mean()
        if AKD_W:                                            # C-A: per finding, softmax over the study's k windows
            b, k = model._al.shape[:2]
            with torch.no_grad():
                lt = torch.log_softmax(T_HEAD["att"](T_HEAD["norm"](ft)).float().view(b, k, 12), dim=1)
            ls = torch.log_softmax(model._al, dim=1)
            LAST_AUX["akd"] = AKD_W * ramp * (lt.exp() * (lt - ls)).sum(1).mean()


def train_step_loss(model, batch, dev, cmask, sil=None, ext=False, bank=None, ramp=1.0):
    """sil (silw / hill): the silent-cell table [n_train, 12] on dev; batch[11] = the studies' dataset indices.
    ext (SURPASS-F extra studies): the global path and loss only; the local loss is 0 (no crop pass).
    RUN-C: the Hydra / FKD terms go to LAST_AUX (added by the caller)."""
    b = batch[0].shape[0]
    xg, xcr, typ = prep(batch, True, dev)
    fid, wc, y, w = v2.to_dev(dev, batch[9], batch[10], batch[2], batch[3])
    if ext:
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            g = model.forward_global(xg, b)
        if HYDRA or FKD:
            aux_losses(model, xg, batch, dev, ramp)
        return torch.stack([v2.bce(g, y, w), torch.zeros((), device=dev)]
                           + ([bank.loss(g, y, w)] if bank is not None else []))
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        g, l = model(xg, b, xcr, fid, wc, typ)
    if HYDRA or FKD:
        aux_losses(model, xg, batch, dev, ramp)
    if sil is None:
        return torch.stack([v2.bce(g, y, w), v2.bce(l, y, w * cmask)] + ([bank.loss(g, y, w)] if bank is not None else []))
    m = sil[batch[11].to(dev)]
    return torch.stack([silent_loss(g, y, w, m), silent_loss(l, y, w * cmask, m)])


def main():
    started = time.monotonic()
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
    ref = r.load_labels(r.ASSETS / "validation_labels.json", "validation")
    assert r.sha(r.ASSETS / "folds.csv") == CFG["sha256"]["folds"]
    fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
    meta = r.read_manifest()
    train_ids = sorted(s for s in labels if fold[s] != VAL_FOLD); val_ids = sorted(s for s in labels if fold[s] == VAL_FOLD)
    if VAL_FOLD == "0":
        assert len(train_ids) == 3524 and len(val_ids) == 825 and set(val_ids) == set(ref)
    else:
        assert len(train_ids) + len(val_ids) == 4349 and val_ids and not set(val_ids) & set(ref)
        ref = {s: labels[s] for s in val_ids}
    if ALLDATA:
        train_ids = sorted(train_ids + val_ids); assert len(train_ids) == 4349
    tlabels = labels                                         # PUB-KD: training targets (scoring stays on label-v2)
    if TARGETS_FILE:
        tlabels = r.load_labels(Path(TARGETS_FILE), "kd_targets")
        assert set(tlabels) >= set(train_ids), "teacher targets must cover every training study"
        d = np.mean([np.abs(tlabels[s][0] - labels[s][0]).mean() for s in train_ids])
        print(f"KD_TARGETS {Path(TARGETS_FILE).name}: mean |target - label-v2| over training cells {d:.4f}", flush=True)
    tab = v2.crop_table(meta)
    assert set(tab) >= set(train_ids) | set(val_ids), "crop table must cover every labelled study"
    comp_ids, ext_ids, hold_ids = train_ids, [], []
    if EXT:
        ext_ids, hold_ids = ext_setup(meta, tab, tlabels)
    if EVAL_ONLY:                                            # SURPASS-F: a finished student on the holdout, nothing else
        model, _, _ = build(dev, pretrained=False)
        res_ = model.load_state_dict(torch.load(EVAL_ONLY, map_location="cpu", weights_only=True), strict=False)
        assert all(k.startswith("fkd_proj.") for k in res_.unexpected_keys) and all(k.startswith(("crop_parts.", "norm_l.", "clsV", "clsvb", "layer4_c.", "layer3_c."))
                                                for k in res_.missing_keys), res_          # global-only files: crop branch absent
        print(f"EVAL_ONLY weights {Path(EVAL_ONLY).name}: {len(res_.missing_keys)} crop-branch keys absent", flush=True)
        model.hydra_infer = HYDRA is not None and os.environ.get("V4_HYDRA_INFER") == "1"   # RUN-C: main + member heads
        # KF-CNXS (2026-10-22): V4_EVAL_SET=all = every fastMRI exam with targets + the holdout (teacher v2's fastMRI member)
        allset = os.environ.get("V4_EVAL_SET") == "all"
        if os.environ.get("V4_EVAL_SET") == "val":           # KF-SOUP (2026-10-22): the validation fold, global head
            r.save_predictions(OUT / "val_global.json", vids := val_ids[:16] if SMOKE else val_ids,
                               predict_global(model, vids, meta, tab, dev), f"{Path(EVAL_ONLY).name}, global head, fold {VAL_FOLD}")
            print(f"EVAL_ONLY {len(vids)} validation studies (fold {VAL_FOLD}) {time.monotonic() - started:.0f}s", flush=True)
            return
        ids = sorted(ext_ids + hold_ids) if allset else hold_ids
        ids = ids[:16] if SMOKE else ids
        r.save_predictions(OUT / ("fm_all_global.json" if allset else "fm_holdout_global.json"), ids,
                           predict_global(model, ids, meta, tab, dev),
                           f"{Path(EVAL_ONLY).name}, global head, fastMRI {'all' if allset else 'holdout'}")
        print(f"EVAL_ONLY {len(ids)} holdout studies {time.monotonic() - started:.0f}s", flush=True)
        return
    train_ids = {"joint": comp_ids + ext_ids, "pre": ext_ids, "fine": comp_ids}[STAGE]
    assert train_ids, STAGE
    gap_ids = sorted(random.Random(CFG["seed"] + 1).sample(comp_ids, 16 if SMOKE else G4["train_gap_studies"]))
    print(f"{G4['version']} seed {CFG['seed']} ({CFG['version']}): train {len(train_ids)}, validation {len(val_ids)}, "
          f"gap sample {len(gap_ids)}; GPU {torch.cuda.get_device_name()} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    if FKD:
        load_fkd_teacher(dev); print(f"FKD teacher {Path(FKD).name} feature dim {FKD_D}", flush=True)
    model, opt, clip = build(dev)
    if INIT:
        init_from(model, INIT)
    ps = model.param_sets()
    print("PARAMS " + json.dumps({k: sum(p.numel() for p in v) for k, v in ps.items()}), flush=True)
    spe = math.ceil(len(train_ids) / (CFG["studies_per_step"] * CFG["accum"]))
    total, warm, floor = spe * CFG["epochs"], spe * CFG["warmup_epochs"], CFG["lr_floor"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else floor + (1 - floor) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))

    vids = val_ids[:16] if SMOKE else val_ids
    start, history, rec = 1, [], OUT / "recovery.pt"
    if rec.is_file():
        st = torch.load(rec, map_location="cpu", weights_only=False)
        assert st["g4"] == G4 and st["cfg"] == CFG, "recovery config differs"
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, history = st["epoch"] + 1, st["history"]
        print(f"RESUMED after epoch {st['epoch']}", flush=True)
    sil = None
    if IMP == "mcpair":
        PAIRS.update(pair_table()); assert set(PAIRS) >= set(train_ids) | set(val_ids)
        cov = np.mean([(PAIRS[s][0][meta[s]["mask"]] >= 0).mean() for s in train_ids])
        print(f"MCPAIR rows paired (valid rows, training studies) {cov:.3f}", flush=True)
        train_ds = GLStudies6(train_ids, meta, labels, CFG["train_windows"], True, tab)
    elif IMP == "fuse":
        PAIRS.update(pair_table()); assert set(PAIRS) >= set(train_ids) | set(val_ids)
        cov = np.mean([(PAIRS[s][0][meta[s]["mask"]] >= 0).mean() for s in train_ids])
        print(f"FUSE rows paired (valid rows, training studies) {cov:.3f}", flush=True)
        train_ds = GLStudiesF(train_ids, meta, labels, CFG["train_windows"], True, tab)
    elif IMP in ("silw", "hill"):
        assert all((labels[s][1] == 1).all() for s in train_ids), "silent lanes assume unit training weights"
        st = silent_table(train_ids, labels)
        print(f"SILENT assumed-negative cells {st.mean():.4f} of training cells; per target "
              + json.dumps({t: round(float(v), 3) for t, v in zip(r.TARGETS, st.mean(0))}), flush=True)
        sil = torch.from_numpy(st).to(dev)
        train_ds = GLStudiesI(train_ids, meta, labels, CFG["train_windows"], True, tab)
    else:
        train_ds = (GLStudiesI if HYDRA else v2.GLStudies)(train_ids, meta, tlabels, CFG["train_windows"], True, tab)
    if HYDRA:                                                # RUN-C: member targets in dataset order
        global HT
        HT = torch.tensor(np.array([HY["studies"][s] for s in train_ds.ids]), dtype=torch.float32, device=dev)
        print(f"HYDRA members {HY['members']} targets {tuple(HT.shape)} per-member mean "
              f"{HT.mean((0, 2)).round(decimals=3).tolist()}", flush=True)
    sampler = None
    if EXT:                                                  # dataset order: competition first, then extra
        cs = set(comp_ids); n_c = sum(s in cs for s in train_ids); assert train_ids[n_c:] == [s for s in train_ids if s.startswith("fm_")]
        sampler = SourceBatches(n_c, len(train_ids) - n_c, CFG["studies_per_step"], CFG["seed"])
    cmask = torch.from_numpy(CMASK.astype(np.float32)).to(dev)
    bank = RankBank(dev) if RANK_W else None
    last = 1 if SMOKE else CFG["epochs"]
    for epoch in range(start, last + 1):
        model.train(); t0 = time.monotonic()
        wif = lambda w: np.random.seed((CFG["seed"] * 1000 + epoch * 100 + w) % 2**32)
        if sampler is None:
            dl = torch.utils.data.DataLoader(train_ds, batch_size=CFG["studies_per_step"], shuffle=True,
                                             num_workers=CFG["workers"], pin_memory=True, worker_init_fn=wif)
        else:
            sampler.set_epoch(epoch)
            dl = torch.utils.data.DataLoader(train_ds, batch_sampler=sampler, num_workers=CFG["workers"],
                                             pin_memory=True, worker_init_fn=wif)
        run = np.zeros(3); n = 0; gn = np.zeros(len(clip)); nst = 0; run_x = np.zeros(4); run_r = np.zeros(2)
        run_a = np.zeros(3)
        if IMP == "fuse":
            model.fuse_stats = [0.0] * 4
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            b = batch[0].shape[0]
            ext = sampler is not None and sampler.is_ext[i]
            ramp = min(1.0, (i + 1) / len(dl)) if epoch == 1 else 1.0
            parts = train_step_loss(model, batch, dev, cmask, sil, ext, bank, ramp)
            if sampler is not None:                          # global loss per source: competition / extra
                run_x[2 * ext:2 * ext + 2] += [float(parts[0]) * b, b]
            loss = G4["global_w"] * parts[0] + G4["aux_local"] * parts[1]
            if bank is not None:                             # RANK: rank loss and mean pairs per finding (bank side)
                loss = loss + RANK_W * parts[2]; run_r += [float(parts[2]) * b, bank.pairs * b]; parts = parts[:2]
            if LAST_AUX:                                     # RUN-C: hydra, fkd, akd (weighted; fkd / akd ramped in epoch 1)
                loss = loss + sum(LAST_AUX.values())
                for j, nm in enumerate(AUX_NAMES):
                    run_a[j] += float(LAST_AUX[nm]) * b if nm in LAST_AUX else 0.0
            (loss / CFG["accum"]).backward()
            if (i + 1) % CFG["accum"] == 0 or i + 1 == len(dl):
                gn += [float(torch.nn.utils.clip_grad_norm_(c, CFG["grad_clip"])) for c in clip]; nst += 1
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += np.append(parts.detach().cpu().numpy(), loss.item()) * b; n += b
            if (i + 1) % 400 == 0 or (SMOKE and i + 1 in (8, G4["smoke_steps"])):
                print(f"  epoch {epoch} step {i + 1}/{len(dl)} loss global/local {run[0] / n:.4f}/{run[1] / n:.4f} "
                      f"{(time.monotonic() - t0) / n:.3f}s/study mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB"
                      + (f" rank {run_r[0] / n:.4f} pairs/finding {run_r[1] / n:.0f}" if bank is not None else "")
                      + (f" hydra/fkd/akd {run_a[0] / n:.4f}/{run_a[1] / n:.4f}/{run_a[2] / n:.4f}" if HYDRA or FKD else ""),
                      flush=True)
            if SMOKE and i + 1 == G4["smoke_steps"]:
                break
        rec_ = {"epoch": epoch, "train_loss": dict(zip(("global", "local", "total"), (run / n).round(5).tolist())),
                "grad_norm": (gn / max(1, nst)).round(4).tolist(), "train_s": round(time.monotonic() - t0, 1),
                "lr": [g["lr"] for g in opt.param_groups]}
        if bank is not None:
            rec_["train_rank"] = [round(run_r[0] / n, 5), round(run_r[1] / n, 1)]
        if HYDRA or FKD:
            rec_["train_hydra_fkd"] = (run_a / n).round(5).tolist()
        if sampler is not None:
            rec_["train_loss_global_comp_ext"] = [round(run_x[0] / max(1, run_x[1]), 5), round(run_x[2] / max(1, run_x[3]), 5)]
        if IMP == "fuse":                                    # is the partner path used? (MC-PAIR's never grew)
            fs = model.fuse_stats
            rec_["fuse_ratio"] = [round(fs[0] / max(1, fs[1]), 5), round(fs[2] / max(1, fs[3]), 5)]
            rec_["fuse_paired"] = [int(fs[1]), int(fs[3])]
            rec_["fuse_w2"] = [round(float(model.fuse_g.fc2.weight.norm()), 4), round(float(model.fuse_c.fc2.weight.norm()), 4)]
        te = time.monotonic(); p = predict(model, vids, meta, tab, dev)
        rec_["eval_s_per_study"] = round((time.monotonic() - te) / len(vids), 3)
        if HYDRA:                                            # RUN-C: the mean of main + member heads, global
            model.hydra_infer = True; ph = predict_global(model, vids, meta, tab, dev); model.hydra_infer = False
            rec_["lv2_global_hydra"] = r.macro_auc(ph, vids, labels)[0]
            r.save_predictions(OUT / f"val_epoch{epoch:02d}_global_hydra.json", vids, ph, f"RUN-C epoch {epoch}, global, main + member heads")
        pt = predict(model, gap_ids, meta, tab, dev)
        rec_["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        if not SMOKE:
            rec_.update(score(p, val_ids, ref, labels))
            tr = {}
            for j, nm in enumerate(HEADS):
                a, per = r.macro_auc(pt[j], gap_ids, labels)
                tr[f"train_lv2_{nm}"] = a
                tr[f"train_lv2_{nm}10"] = float(np.nanmean([per[r.TARGETS[k]] for k in CF]))
                tr[f"train_bce_{nm}"] = r.val_bce(pt[j], gap_ids, labels)
            rec_.update(tr)
            rec_["gap_local10"] = tr["train_lv2_local10"] - rec_["lv2_local10"]
            rec_["gap_global"] = tr["train_lv2_global"] - rec_["lv2_global"]
            for j, nm in enumerate(HEADS):
                r.save_predictions(OUT / f"val_epoch{epoch:02d}_{nm}.json", val_ids, p[j], f"r50-GL4 {ARM} epoch {epoch}, {nm}")
        history.append(rec_)
        print(f"EPOCH {epoch} " + json.dumps({k: v for k, v in rec_.items() if not k.startswith("lv2_per")}), flush=True)
        if epoch in CFG["swa_epochs"] or SMOKE:
            torch.save(model.state_dict(), OUT / f"weights_epoch{epoch:02d}.pt")
        if not SMOKE:
            torch.save({"g4": G4, "cfg": CFG, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch, "history": history}, OUT / "recovery.tmp")
            os.replace(OUT / "recovery.tmp", rec)
        (OUT / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    # ---- primary: weight average of the SWA epochs (adapter: dense LoRA delta), both BN statistic sets refreshed
    swa = [1] if SMOKE else CFG["swa_epochs"]
    sds = [torch.load(OUT / f"weights_epoch{e:02d}.pt", map_location="cpu") for e in swa]
    avg = {k: (sum(s[k].float() for s in sds) / len(sds)).to(sds[-1][k].dtype) if sds[-1][k].is_floating_point()
           else sds[-1][k] for k in sds[-1]}
    if ARM == "adapter":
        avg.update(lora_dense_average(sds))
    model.load_state_dict(avg, strict=True); model.set_dense()
    tr = time.monotonic()
    refresh = random.Random(CFG["seed"]).sample(comp_ids, 8 if SMOKE else CFG["bn_refresh_studies"])
    bn_refresh(model, refresh, meta, labels, tab, dev)
    print(f"BN_REFRESH {len(refresh)} studies {time.monotonic() - tr:.0f}s; bn[1] global/crop mean "
          + (f"{model.bns[1].running_mean[:4].tolist()} / {model.bns[1].crop_mean[:4].tolist()}" if len(model.bns) > 1
             else "(no BatchNorm)"), flush=True)
    tag = f"swa{swa[0]}_{swa[-1]}"
    torch.save(model.state_dict(), OUT / f"weights_{tag}.pt")
    p = predict(model, vids, meta, tab, dev)
    res = score(p, vids, ref, labels)
    for j, nm in enumerate(HEADS):
        r.save_predictions(OUT / f"val_swa_{nm}.json", vids, p[j], f"r50-GL4 {ARM} SWA {swa}, {nm}")
    pt = predict(model, vids, meta, tab, dev, top1=True)                         # descriptive: top-1 crop per finding
    r.save_predictions(OUT / "val_swa_fused_top1.json", vids, pt[0], f"r50-GL4 {ARM} SWA, fused, top-1 crop")
    res["lv2_fused_top1"] = r.macro_auc(pt[0], vids, labels)[0]
    if IMP == "fuse":                                        # descriptive: the same weights without partners
        po = predict(model, vids, meta, tab, dev, nopartner=True)
        r.save_predictions(OUT / "val_swa_fused_nopartner.json", vids, po[0], f"r50-GL4 {ARM} fuse SWA, fused, no partner")
        res["lv2_fused_nopartner"] = r.macro_auc(po[0], vids, labels)[0]
    if hold_ids:                                             # SURPASS-F: student on the unseen fastMRI holdout
        ids = hold_ids[:16] if SMOKE else hold_ids
        r.save_predictions(OUT / "fm_holdout_global.json", ids, predict_global(model, ids, meta, tab, dev),
                           f"r50-GL4 {ARM} SWA {swa}, global head, fastMRI holdout")
    if HYDRA:                                                # RUN-C: the same SWA weights, main + member heads averaged
        model.hydra_infer = True
        ph = predict_global(model, vids, meta, tab, dev); res["lv2_global_hydra"] = r.macro_auc(ph, vids, labels)[0]
        r.save_predictions(OUT / "val_swa_global_hydra.json", vids, ph, f"RUN-C SWA {swa}, global, main + member heads")
        if hold_ids:
            r.save_predictions(OUT / "fm_holdout_global_hydra.json", ids, predict_global(model, ids, meta, tab, dev),
                               f"RUN-C SWA {swa}, global, main + member heads, fastMRI holdout")
        model.hydra_infer = False
    summary = {"g4": G4, "cfg": CFG, "history": history, "swa": res, "seconds": round(time.monotonic() - started, 1)}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RUN_SUMMARY " + json.dumps({k: v for k, v in res.items() if not k.startswith("lv2_per")}), flush=True)
    if SMOKE:
        print(f"SMOKE_DONE {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
