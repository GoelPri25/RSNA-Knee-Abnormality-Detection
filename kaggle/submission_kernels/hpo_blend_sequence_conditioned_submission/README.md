# HPO-v1 sequence-conditioned checkpoint submission

This inference-only Kaggle code submission uses the best checkpoint from the
from-scratch sequence-conditioned experiment. It preserves the proven support-
aware ResNet-34 preprocessing and adds the checkpoint's learned 7-way plane ×
fluid-sensitive logit correction before the same per-target top-10% pooling.

The notebook uses explicit Kaggle input paths and the attached competition
metadata. It does not recursively search `/kaggle/input`. Before inference it
checks the checkpoint SHA-256, epoch, architecture configuration, metadata
columns, target order, and final CSV schema/probabilities. The checkpoint is
`checkpoint_best.pt` from
`tumis001/rsna-hpo-blend-sequence-from-scratch-fold0` (epoch 17; report-fold
macro AUC 0.8893).

To publish and run the notebook:

```bash
kaggle kernels push -p kaggle/submission_kernels/hpo_blend_sequence_conditioned_submission
```

After the kernel completes successfully, submit that kernel version to the
code competition with `kaggle competitions submit -k
tumis001/rsna-hpo-blend-sequence-conditioned-submission -v VERSION -m
"Sequence-conditioned epoch-17 checkpoint"`.
