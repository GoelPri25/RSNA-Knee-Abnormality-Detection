# Pillar-0 spatial-token cache

`pillar0_support_gated_reader.ipynb` generates a resumable, report-only cache of the
frozen Pillar-0 encoder's three-scale spatial tokens and corresponding valid-support
maps. It preserves the token grid before the encoder's global max-pooling; it does not
train a reader or use labels. DICOM pixels remain on Kaggle. The cache contains only
quantized derived features and series metadata.

The Kaggle kernel attaches the private checkpoint dataset as an input. On each run it
copies saved batches into its working directory, skips series already present, and
processes at most 9,000 new series. It publishes every 512 new series to the same
private dataset, so a later run resumes from the last published checkpoint. Kaggle
Secrets are used only for publishing new dataset versions, with retries for temporary
Secrets-service outages.

The intended complete cache contains 24,035 eligible series from 4,349 report-only
studies; all 58 official-label studies remain excluded. Current cache progress should
be checked against the saved `spatial_batch_*.npz` files before training begins. The
reader training notebook is a separate follow-up and must validate every batch's
schema, support map, quantization scales, and series-key uniqueness before fitting.

The earlier 12-series timing pilot and spatial representation audits are documented
in `PILOT_RESULTS.md` and the adjacent Pillar-0 audit reports.
