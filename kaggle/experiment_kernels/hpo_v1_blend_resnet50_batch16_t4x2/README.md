# HPO-v1 ResNet-50 batch-16 run using both Kaggle T4 GPUs

This is a from-scratch ResNet-50 run with global batch 16, the HPO-v1 blend labels and the 281
additional report-only studies kept in training. The original 825-study fold-0
validation set remains fixed and is scored against Steven v4. The 58 gold
studies stay excluded.

Before training, a disposable worst-case batch of 16 studies × 24 slice windows
checks memory. If its mixed-precision forward/backward pass does not fit, the
kernel stops before training.

The trainer uses `torch.nn.DataParallel` **only around the per-slice ResNet
backbone**. It splits flattened slice windows over the two T4 GPUs, then gathers
their features on GPU 0 for the existing target-specific head and top-k
per-study pooling. This keeps the global study batch at 16 and preserves the
existing loss, optimizer, scheduler, sampling, validation, and checkpoint
selection logic. Checkpoint keys are normalized when saved so the resulting
weights remain compatible with the single-GPU submission model.

The run has a 60-epoch maximum, AUC-based checkpoint selection and early
stopping, and a validation-BCE `ReduceLROnPlateau` scheduler. The runner uses
fixed Kaggle input paths only; it does not recursively search datasets. Model
weights, learning curves, and history are saved to `/kaggle/working` each epoch
and are retained as the kernel output when the run completes.

The attached source dataset contains the original 4,068-study fold file. The
runner verifies its checksum and the HPO-label checksum, then reconstructs the
expanded split by assigning the 281 missing report-only IDs to fold 5. This is
the same training/validation membership as the canonical expanded fold file:
3,524 train, 825 original validation, and zero added/validation report-group
overlap.

The Kaggle CLI metadata machine shape `NvidiaTeslaT4` requests the Kaggle
two-T4 configuration. Push with a 12-hour kernel timeout:

```bash
kaggle kernels push -p kaggle/experiment_kernels/hpo_v1_blend_resnet50_batch16_t4x2 --timeout 43200
```
