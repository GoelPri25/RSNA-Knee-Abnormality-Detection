# Gold AUC audit checkpoint and dataset handoff

Prepared 2026-10-10 for Kaggle teammate `priyanjaliml`. This covers the exact eight image models scored in `data/share/gold_auc_audit.html`, not the later 336px/EfficientNet/quad ensemble evaluations.

Private reproduction bundle: [tumis001/rsna-gold-auc-audit-exact-checkpoints](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-exact-checkpoints). Each checkpoint was SHA256-checked against the original gold inference metadata before upload. `checkpoint_manifest.json` records full hashes, epochs, labels and original sources; `SHA256SUMS` covers the bundle files.

| Model | Epoch (zero-based) | Gold macro AUC | Exact checkpoint | Original artifact source | Training / validation labels |
| --- | --- | --- | --- | --- | --- |
| ResNet50 · v3.1-D2 / Steven validation | 15 | 0.8854 | [r50_hpo_v31_d2_fold0.pt](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-exact-checkpoints?select=r50_hpo_v31_d2_fold0.pt) | [Dataset](https://www.kaggle.com/datasets/tumis001/rsna-hpo-v31-d2-resnet50-best) | HPO-v3.1-D2 / Steven-v4 |
| ResNet50 · v3.1-D2 / HPO validation | 18 | 0.8909 | [r50_hpo_v31_trainval.pt](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-exact-checkpoints?select=r50_hpo_v31_trainval.pt) | [Dataset](https://www.kaggle.com/datasets/tumis001/rsna-hpo-v31-trainval-resnet50-best) | HPO-v3.1-D2 / HPO-v3.1-D2 |
| ResNet50 · HPO-v2 | 15 | 0.9003 | [r50_hpo_v2_fold0.pt](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-exact-checkpoints?select=r50_hpo_v2_fold0.pt) | [Dataset](https://www.kaggle.com/datasets/tumis001/rsna-hpo-v2-provenance-resnet50-fold0-best) | HPO-v2 / Steven-v4 |
| ResNet50 · HPO-v1 | 17 | 0.8983 | [r50_hpo_blend_fold0.pt](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-exact-checkpoints?select=r50_hpo_blend_fold0.pt) | [Dataset](https://www.kaggle.com/datasets/tumis001/rsna-hpo-blend-add281-resnet50-fold0) | HPO-v1 / Steven-v4 |
| ResNet34 · Steven BCE plateau60 | 18 | 0.8683 | [r34_plateau60_fold0.pt](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-exact-checkpoints?select=r34_plateau60_fold0.pt) | [Notebook output](https://www.kaggle.com/code/tumis001/rsna-cnn-top-k-support-plateau-60/output) | Steven-v4 / Steven-v4 |
| ResNet34 · original top-k v1 | 12 | 0.8719 | [r34_topk_v1_fold0.pt](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-exact-checkpoints?select=r34_topk_v1_fold0.pt) | [Notebook output](https://www.kaggle.com/code/tumis001/rsna-cnn-top-k-support-v1/output) | Steven-v4 / Steven-v4 |
| ResNet34 · Steven pairwise AUC | 15 | 0.8698 | [r34_pairwise_fold0.pt](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-exact-checkpoints?select=r34_pairwise_fold0.pt) | [Notebook output](https://www.kaggle.com/code/tumis001/rsna-cnn-top-k-pairwise-auc-fold-0/output) | Steven-v4 / Steven-v4 |
| ResNet34 · HPO-v1 AUC controls | 22 | 0.8987 | [r34_hpo_v1_auc_control_fold0.pt](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-exact-checkpoints?select=r34_hpo_v1_auc_control_fold0.pt) | [Dataset](https://www.kaggle.com/datasets/tumis001/rsna-hpo-blend-add281-auc-control-fold0) | HPO-v1 / Steven-v4 |

**The gold ensemble is a prediction combination, not another checkpoint.** Its members are `r50_hpo_v31_d2_fold0`, `r50_hpo_blend_fold0` and `r50_hpo_v31_trainval`, using per-target rank averaging (gold macro AUC 0.9023). Neither the 336px checkpoint nor EfficientNet was in this audit. The report-plus-image diagnostic also has no separate checkpoint and requires reports, which are unavailable at test time.

## Label datasets

| Label set | Dataset | File |
| --- | --- | --- |
| HPO-v3.1-D2 | [Private dataset](https://www.kaggle.com/datasets/tumis001/rsna-knee-hpo-v31-d2-report-labels) | `soft_labels_hpo_v3_1_d2_structured.json` |
| HPO-v2 | [Private dataset](https://www.kaggle.com/datasets/tumis001/rsna-knee-hpo-v2-provenance-candidate) | `soft_labels_hpo_v2_provenance_candidate.json` |
| HPO-v1 | [Private dataset](https://www.kaggle.com/datasets/tumis001/rsna-knee-hpo-v1-blended-report-labels) | `soft_labels_hpo_v1_blend.json` |
| Steven-v4 | [Public source](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) | `llm_labels_v4_blend.csv`; runner converts report-only training and validation labels |
| HPO-v1 upstream asset | [Public source](https://www.kaggle.com/datasets/nartaa/rsna-knee-hpo-assets) | `labels_v1_blend.parquet` |

## Image caches and source code

The eight models use the original 140mm/224px support-aware cache, not Fit160 or native-triplet/highres caches. Current dataset rehosts are copies of the original three notebook-output shards. Split/config pins are stored in each checkpoint; do not assume all older ResNet34 runs include the 281 training-only additions.

| Resource | Kaggle link |
| --- | --- |
| Training cache shard 0 | [tumis001/rsna-cnn-topk-support-cache-shard-0-rehost](https://www.kaggle.com/datasets/tumis001/rsna-cnn-topk-support-cache-shard-0-rehost) |
| Training cache shard 1 | [tumis001/rsna-cnn-topk-support-cache-shard-1-rehost](https://www.kaggle.com/datasets/tumis001/rsna-cnn-topk-support-cache-shard-1-rehost) |
| Training cache shard 2 | [tumis001/rsna-cnn-topk-support-cache-shard-2-rehost](https://www.kaggle.com/datasets/tumis001/rsna-cnn-topk-support-cache-shard-2-rehost) |
| Exact 58-study gold evaluation cache | [Gold cache manifest](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-exact-checkpoints?select=gold_slice_cache_224%2Fslice_cache%2Fmanifest.jsonl); Kaggle automatically expands the uploaded ZIP into `gold_slice_cache_224/slice_cache/`; verify against `gold_cache_sha256.json` |
| Original CNN source | [tumis001/rsna-cnn-topk-support-source-v1](https://www.kaggle.com/datasets/tumis001/rsna-cnn-topk-support-source-v1) |
| HPO/XLA source | [tumis001/rsna-cnn-hpo-blend-xla-source-v1](https://www.kaggle.com/datasets/tumis001/rsna-cnn-hpo-blend-xla-source-v1) |
| HPO AUC-control source | [tumis001/rsna-cnn-hpo-auc-control-xla-source-v1](https://www.kaggle.com/datasets/tumis001/rsna-cnn-hpo-auc-control-xla-source-v1) |

## Report-reader columns

These are extraction rules and stored label tables, not task-trained image checkpoints. The HTML scores the rule reader, the three LLM-derived readers below, and their blends. HPO-v1 itself has no gold rows; the gold report blend excludes that parent. `gold_ceiling_meta.json` records the input hashes.

- pilkwang: [rsna-knee-llm-labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels), `report_labels_v2.csv`.
- laymond: [rsna-knee-abnormality-qwen3-8b-weak-labels](https://www.kaggle.com/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels), `weak_labels_v2_raw.csv`.
- The column named `teammate` reads the local archived `colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv`. The hash-verified table is now copied to the private [report-reader input dataset](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-report-readers?select=report_structured_labels.csv), shared with `priyanjaliml`. The audit column name does not establish which account originally generated the archive; do not substitute a different file on the basis of its name.
- Rule implementation: project files `label_rules/v3_1/rule_extractor_v31.py` and `build_hpo_v3_1.py`; source snapshots and saved gold rule states are in the [private report-reader dataset](https://www.kaggle.com/datasets/tumis001/rsna-gold-auc-audit-report-readers), and aggregate reader results are in the checkpoint reproduction bundle.

## Access and verification

Private resources are shared only with `priyanjaliml` as READER where newly added. Her pre-existing WRITER access to HPO-v1 and HPO-v3.1-D2 is preserved. Original source code already had READER access. All datasets remain private; existing collaborators and editable metadata were checked after mutation.

Read/write receipts: `data/share/gold_audit_kaggle_sharing/sharing_receipt_existing.json` and `sharing_receipt_extra.json`. Published-file/manifest verification: `published_bundle_verification.json`. Bundle data and receipts remain gitignored. Original notebook outputs may have separate notebook permissions; all eight exact weights are therefore also available in the shared dataset.

Gold has been inspected repeatedly. These AUCs are exploratory development evidence on 58 studies, not estimates of public leaderboard AUC. No training, inference or submission was launched for this handoff.
