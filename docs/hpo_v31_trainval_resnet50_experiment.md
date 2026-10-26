# v3.1-D2 validation-control experiment

Notebook: https://www.kaggle.com/code/tumis001/rsna-hpo-v31-trainval-resnet50-fold0

Controlled change from the0.925 run: training labels remain HPO-v3.1-D2; validation labels now use the same v3.1-D2 file, subset to the original825 fold0 IDs. BCE controls plateau scheduling; v3.1 confident-cell macro AUC controls best checkpoint and early stopping. Steven-v4 validation metrics are audit-only and do not affect any training decision. Both metric sets use the same score array, with no duplicate inference. The v3.1 AUC must not be compared numerically to Steven-v4 AUC as if labels/masks were unchanged.

ImageNet ResNet50 from initialization, oneT4,batch8,24training windows/all48 evaluation, original140mm/224px support cache,3524train with281 train-only additions,825fixedvalidation,58goldexcluded. Same60epoch configuration, augmentation, seed2026, AdamW and controls as the0.925 run. Allthree shards attached as datasets; fixed paths, no recursive search.

Labelsha25638367d62eb9e7c198b69d88e2b9ad419e00fea9d1211b98feae45cd8a386f92d. Source trainer pinned to b8f8302319afbcf9970f0cf5b806d8fb78e72ef6e169b276274ddf7cd5e0ba85.

History records v3.1 metrics plus Steven AUC/BCE. Each epoch saves validation_audit_epoch_XX.json with Steven per-target metrics and the shared predictions. Latest and best checkpoints saved each epoch in Kaggle working outputs. Existing10.5h epoch-boundary time guard retained; no extra GPU preflight or separate notebook. EfficientNet on Modal remains independent and unchanged.

Verification: runner/notebook/actual source instrumentation compile. Audit hook reproduces epoch15 fixedSteven AUC0.896327 and BCE0.493979 from existing predictions within rounding tolerance. Same split and label checksum assertions retained.

## Startup failure investigation

Version1 returned ERROR, empty execution logs and no failure message. Successful prior v3.1 version2 and failed version1 were both pulled from Kaggle, not inferred from repository files. Input dataset lists, notebook4.5 structure, kernelspec, GPU/internet settings and machine shape match. Code diff consists of validation-label path, audit-only Steven metrics, descriptive config/output names; training backbone/optimizer/sampling/split unchanged. Complete setup dry-run against local copies verified allpaths/cache files/label-source-config hashes/gold lock/825validation/3524training/import and exacttrainerpatch compilation. Dry-run did not allocate GPU or train.

Successful server metadata records container sha2562757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f; failed metadata image field blank. Version2 retry explicitly supplies that recorded image with unchanged inputs/code. This is an execution-setting alignment, not an established diagnosis: blank image may be a consequence of failure before startup. No evidence identifies the precise infrastructure failure.

## Fresh notebook launch

Fresh notebook: https://www.kaggle.com/code/tumis001/rsna-hpo-v31-trainval-resnet50-fresh (version1). Skeleton and metadata copied from successful server-pulled notebook, then only verified trainval changes applied. Dataset/kernel/competition attachments, GPU/TPU/internet flags, language, notebook type, machine shape and container image asserted identical. Startup definitions and execution are separate cells with explicit flushed markers. CLI has no individual interactive-cell execution and exposes no fork lineage parameter; this is a new exact-source copy, not an in-app fork. Prior failed notebook was ERROR, so no duplicate live Kaggle job was created.

Fresh version1 also returned ERROR with logs=[] and failure_message="". Startup marker was not exposed. No further Kaggle retries launched; no training is confirmed. Precise startup cause remains unavailable through the official API.
