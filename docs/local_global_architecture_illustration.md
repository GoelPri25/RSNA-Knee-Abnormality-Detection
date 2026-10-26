# Local and global reader, with real MRI inputs

`scripts/local_global_model.py` is the model used by the running experiment. It contains one ImageNet-initialized, trainable ResNet50 and two jointly trained prediction paths. There is no teacher, image decoder or generative model.

Each 224×224 window has three native acquired intensity slices and three support masks. ResNet50 yields one 2,048-dimensional vector per window. Training randomly selects 24 of up to 48 cache windows, sorted back into series order; evaluation uses all available windows. Windows are not individual Transformer pixel patches.

The local head produces 12 logits per window, then averages the strongest 10% separately for each finding. The context path projects image vectors to 256 dimensions and adds plane, fluid sensitivity, fat suppression, slot and within-series position/gap metadata. One pre-normalized Transformer encoder layer has four attention heads, a 256→512→256 ReLU feed-forward network, dropout 0.1 and residual connections. Padded windows are masked. Attention never mixes different studies, and positional metadata does not physically register the different planes.

Twelve learned finding queries retrieve from the contextual window vectors using a separate four-head cross-attention pooling layer. LayerNorm and a shared 256→1 linear layer produce a correction for each target. The final logit is local logit + 0.1 × correction; sigmoid gives its probability. Context weights start small but nonzero. Weighted soft-label BCE is computed from the combined logits, and AdamW updates the CNN and both paths. Validation AUC selects checkpoints and governs stopping; validation BCE governs the LR scheduler. There is no direct AUC training term in this experiment.

Measured parameter counts: ResNet trunk 23,517,440; local head 24,588; Transformer encoder layer 527,104; complete context path 1,325,057; total 24,867,085. All are trainable. The context path is about 5.3% of the total model parameters.

Training labels are unchanged HPO-v3.1-D2; validation labels are the fixed Steven-v4 fold-0 reference. The separate Synovitis weighting candidate is not used here.

## Real-image provenance and reproduction

The examples are from one report-training knee, with all four series represented. Native filenames and indices from the new cache manifest are matched against the locally available original 140mm/224px centre cache. The chosen triplets have all three genuine acquired neighbours available locally; we do not pretend consecutive sparse cache rows are acquired neighbours. Source pixel hashes, filenames, indices and cache row mappings are saved in `data/local_global/architecture_illustration/sample_provenance.json`. No labels, diagnoses or learned attention maps are assigned to these illustrations.

The first attempt to retrieve images used a CPU-only Modal helper, failed on an import, and was stopped. The finished illustration uses only local data and the existing offline CPU container. Neither training deployment was altered for image rendering.

With the original cache shards, new shard-0 manifest audit and experiment assets available locally:

```bash
podman run --rm --network none -v "$PWD:/workspace:rw" localhost/rsna-local-global-checks:latest python scripts/render_local_global_architecture.py
```

The renderer writes PNG/SVG overview and Transformer detail figures, parameter counts, source provenance and the interactive fragment under `data/local_global/architecture_illustration/`. Its editable template is `docs/figures/illustrations/local_global_reader.template.html`. Derived competition images stay in the gitignored data directory.
