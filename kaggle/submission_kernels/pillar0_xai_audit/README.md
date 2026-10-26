# Pillar-0 reader XAI audit

Inference-only report-fold audit for the saved `checkpoint_best.pt`. The private
Kaggle kernel attaches the token cache, reader artifacts, Steven v4 report
labels, and report-fold assignments. It reads only fold 0 report-validation
labels; it never attaches the competition dataset or loads official-label gold.

The audit compares the baseline with removal of each visual feature scale,
removal of all visual content, unknown protocol metadata, hard masking of
zero-support tokens, and a within-series feature-to-coordinate shuffle. It
also records per-target attention routing and target-specific Grad×Input maps
for a deterministic label-aware subset. Attention is reported as routing
behavior, not treated as a stand-alone explanation.

Outputs are aggregate CSV/JSON/PNG artifacts under `/kaggle/working/pillar0_xai`;
the validation-only temporary int8 memmaps live under `/kaggle/temp`. No study
or series identifiers are exported.

Run with:

```bash
kaggle kernels push -p kaggle/submission_kernels/pillar0_xai_audit
```
