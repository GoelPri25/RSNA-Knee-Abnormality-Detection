# HPO-v1 +281 CoAtNet-0 comparison

This is one fold-0 backbone comparison against the public-score 0.918 HPO-v1
ResNet-34 recipe. It runs on one Modal T4 while the separate ResNet-50 run uses
Kaggle GPUs.

## Controlled comparison

The study-level pipeline is kept fixed: HPO-v1 labels, the 281 additional
training-only studies, the original 825 fold-0 report-validation studies scored
against Steven v4, the 58 locked gold studies, support-aware 2.5D inputs, 24
sampled windows, target-wise top-10% pooling, augmentation, loss, optimizer,
ReduceLROnPlateau-on-BCE, and AUC checkpoint/early-stopping rules. The only
planned model change is the image backbone and its matching per-window head.
CoAtNet-0 does not fit the ResNet recipe's 8-study physical microbatch on a
16-GiB T4, so this run uses 2 studies per microbatch with four-step gradient
accumulation. That preserves the effective 8-study batch and optimizer-step
count; the smaller physical batch does change the backbone's per-microbatch
normalization behavior, so this is a hardware accommodation to record when
interpreting the comparison.

CoAtNet cannot load the ResNet-34 checkpoint: its stage weights and output
dimensions differ. This run starts from the ImageNet-1K-pretrained
`coatnet_0_rw_224.sw_in1k` weights, copies the pretrained RGB stem weights into
the three image channels, initializes the three acquired-support channels to
zero, and trains a new window head. After the unchanged cache preprocessing
and augmentation, its intensity channels are converted to CoAtNet's pretrained
ImageNet normalization (mean/std 0.5/0.5); the support channels remain in
`[0, 1]`. Study pooling remains the proven Top-K reader.

```text
three adjacent slices + three support masks
                  │
       ImageNet CoAtNet-0 backbone
        convolution → attention stages
                  │
         new 12-target window head
                  │
     unchanged target-wise Top-K pooling
                  │
         twelve study logits
```

## Data and durability

The training function reads the already-derived Kaggle slice-cache shards from
the existing Modal volume `rsna-cnn-topk-support-cache-v1`. It downloads only
the HPO-v1 and Steven-v4 label files and uses the source package attached to the
known Kaggle dataset. Raw DICOMs and official gold labels are not copied to
Modal. The durable run artifacts go to the private Kaggle dataset
`tumis001/rsna-hpo-blend-add281-coatnet0-fold0`; checkpoints and histories are
published every two epochs, and the trainer writes its latest checkpoint each
epoch.

The Modal entrypoint uses one T4 and a 12-hour maximum timeout. It runs detached
with the personal account profile, checks for an active duplicate app, and
verifies the existing cache volume before allocating the GPU:

```bash
python scripts/run_modal_personal_coatnet.py --check-only
python scripts/run_modal_personal_coatnet.py
```

The first command is a no-GPU preflight. The second starts the experiment.

## Decision criterion

Compare best report-fold macro AUC and epoch, validation BCE, per-target AUC,
runtime, and the public score from a valid submission against the fixed 0.918
ResNet-34 reference. Report-fold metrics use weak labels and are diagnostics;
the public score remains the external comparison. A one-fold change is a
screening result, not cross-fold proof.
