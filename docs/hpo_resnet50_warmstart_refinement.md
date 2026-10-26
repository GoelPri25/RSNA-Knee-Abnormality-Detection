# ResNet-50 warm-start refinement

This short follow-up starts from the saved best ResNet-50 HPO-blend checkpoint
(epoch 17; report-fold macro AUC 0.890904; validation BCE 0.516915). It keeps the
same HPO v1 labels, 3,524/825 train-validation split, 24 sampled windows,
support-aware 2.5D ResNet-50 reader, BCE objective, and augmentation. Only the
optimization stage changes: initialize fresh optimizer/scheduler state, start
at learning rate `3.75e-5`, and train for at most 12 more epochs with the
existing ReduceLROnPlateau-on-validation-BCE and early-stop-on-AUC rules.

This is a **warm-start refinement**, not an exact resume: the original checkpoint
contains model weights, config, and epoch, but no optimizer or scheduler state.
The original epoch-17 checkpoint is installed as the initial best checkpoint,
so the refinement can only replace it if report-fold macro AUC improves. The
58 gold studies remain locked and unused.

Each epoch writes the latest model, optimizer, scheduler, scaler, step, and RNG
states and publishes a private Kaggle artifact version. It reuses the existing
Modal cache volume for derived slice shards; no raw DICOMs are moved. Launch
only through the personal-account wrapper, which checks for a duplicate app and
starts Modal detached:

```bash
python scripts/run_modal_personal_resnet50_warmstart.py
```

The app requests one Modal T4 with the existing 4-hour timeout. The expected
training portion is roughly 2–2.5 hours if all 12 epochs run; data shards are
already in the persistent Modal volume.
