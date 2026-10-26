# HPO v2 provenance ResNet-50, fixed fold 0

This Kaggle notebook runs the label-provenance experiment after the Modal
preemption/restart issue. It changes only the report-only training probabilities
from HPO v1 to the audited HPO v2 provenance candidate. It keeps the HPO-v1
ResNet-50 configuration, 24 ordered slice windows, global study batch 8,
3,524 training studies (including the 281 training-only additions), the same
825 Steven-v4 report-validation studies, ImageNet initialization, AUC-based
checkpoint selection/early stopping, and validation-BCE ReduceLROnPlateau.
The 58 gold studies remain excluded.

The runner uses explicit Kaggle input paths only; it does not recursively search
the input tree. It verifies the label, configuration, and fold checksums before
allocating the model, and checks that the validation fold is unchanged. If
Kaggle provides two T4s, DataParallel is used only for the per-slice backbone;
the global batch remains 8. Checkpoints and histories are written to
`/kaggle/working/hpo_v2_provenance_resnet50_fold0/run` after each epoch and are
available as notebook outputs when the run completes.

Regenerate the notebook after editing `run.py`:

```bash
python build_notebook.py
```

Push and start exactly one Kaggle run:

```bash
kaggle kernels push -p kaggle/experiment_kernels/hpo_v2_provenance_resnet50_fold0
```

The validation score can compare training utility against HPO v1, but it does
not establish medical label truth.
