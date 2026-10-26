# Pillar-0 support-path probe

This private, CPU-only Kaggle notebook replays the historical Pillar-0 transfer preprocessing next to a controlled support-aware variant on three report-only MRI series from one study. It reads official-label nullness only to select report-only data; it does not read label values, report text, load the frozen encoder, or train. The output archive contains paired one-channel FP16 inputs and a support map, with plane names only. It stays in Kaggle and is not uploaded to Modal by this notebook.

Run `python build_notebook.py` to regenerate the notebook and kernel metadata.
