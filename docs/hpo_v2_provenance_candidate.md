# HPO v2 provenance candidate

This experiment tests one narrow label-construction change on top of the HPO v1 blend. It uses each source's explicit `m_*` evidence mask to avoid treating a probability inferred from silence as if it were an independent report reading.

## Construction rule

For each of 4,349 report-only studies and each of the 12 findings:

| Steven addressed | Gemini addressed | Candidate target |
| --- | --- | --- |
| yes | yes | Keep the existing HPO v1 blended probability |
| yes | no | Use Steven v4's probability alone when it is a usable probability |
| no | yes | Use Gemini-low's probability alone |
| no | no | Keep the existing HPO v1 probability |

The HPO v1 values remain the reference for both-source agreement and silence. This candidate does not convert silence to a negative, discard Gemini's silent-cell prior, sharpen probabilities, or use an image model to overwrite labels. Source masks define evidence availability, but we also reject an internally contradictory Steven cell: Steven v4 uses exactly `0.25` as its not-mentioned sentinel. One report-only Synovitis cell has `m_Synovitis=1` while Steven's value is `0.25`; because that is not a usable probability, the candidate retains HPO v1 for that cell. The other mask/sentinel mismatch is both-addressed and already retains HPO v1.

Every target retains training weight `1.0`, so the label probabilities are the only intended change. After excluding the contradictory sentinel cell, the rule changes 1,993 cells across 1,937 studies; all are usable Steven-only cells. The other evidence states are unchanged.

## Build and audit

From the repository root:

```bash
python scripts/prepare_hpo_v2_provenance_candidate.py
```

The generated trainable labels and their audit files are under `data/label_datasets/hpo_v2_provenance_candidate/`. The JSON uses the same `study_id`, ordered `probabilities`, and `weights` structure as the HPO v1 labels. The manifest records source hashes; the CSV files enumerate evidence-state counts and every changed target cell.

The label dataset is private on Kaggle as `tumis001/rsna-knee-hpo-v2-provenance-candidate`.

## Experiment comparison

The comparison preserves the public-score 0.924 HPO v1 +281 ResNet-50 recipe: ResNet-50 with ImageNet initialization, 24 slice windows per study, effective batch size 8, 140 mm cache, fixed fold-0 validation IDs (825 studies), seed, optimizer, 60-epoch cap, AUC checkpointing/early stopping (patience 8, min delta 0.0005), and BCE `ReduceLROnPlateau` (factor 0.5, patience 2). Only report-label construction changes. The 281 formerly unassigned report-only studies remain training-only; validation membership does not change.

The Kaggle retry attaches all three verified 140 mm cache shards as private datasets, rather than depending on the flaky notebook-output mounts. The dataset files are the same cached arrays, support masks, and manifests. The training runner uses exact input paths and writes checkpoint/history files to `/kaggle/working` after each epoch.

Do not read the 58 gold labels while constructing this candidate or training the compared checkpoints. First verify identical study IDs, split membership, target order, and weights. Compare report-fold macro AUC and BCE with the existing HPO v1 ResNet-50 run (best report-fold AUC 0.88245 at epoch 15), then submit the candidate only if the validation comparison justifies it.

This can test whether the provenance-aware fusion is more useful for prediction. It cannot establish that a changed label is medically correct; that still requires independent image review or another independent source of truth. Do not use the candidate model's own predictions as label truth.
