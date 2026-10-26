# Matched report-label A/B on the MedSigLIP baseline

Private CPU-only Kaggle run. It reuses the existing three weak-study MedSigLIP feature shards and 58-study feature cache; it does not re-encode images or request a GPU. The two arms differ only in report-derived training labels:

- A: our Confidence-V2 probability and weight mapping, generated from the saved structured report extraction.
- B: Flight's public hybrid-v4 report scores, used as soft targets with unit weight where a score is available.

Both arms exclude every gold study from training. The saved development/evaluation split is used to create only the 29 development holdout examples. The other 29 gold labels are never loaded by this notebook, predicted, scored, or summarized. The original hierarchical MIL architecture, seed, fixed 30-epoch schedule, and training configuration are shared. There is no model selection on gold and no early stopping; the goal is a matched teacher-label comparison.

The notebook saves checkpoints, epoch logs, history, and current learning curves after every epoch under `/kaggle/working/label_ab/`, then writes development-only macro AUC and per-target metrics. The CPU run may take longer than the old T4 baseline; it uses no GPU quota.

Regenerate and submit with:

```bash
python build_notebook.py
kaggle kernels push -p .
```
