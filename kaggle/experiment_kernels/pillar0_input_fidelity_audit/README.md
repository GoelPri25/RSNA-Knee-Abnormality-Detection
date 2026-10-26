# Input-fidelity audit

This notebook is a CPU-only, no-training audit of the RSNA MRI input path. It
samples up to 36 report-only studies, balancing explicit present/absent
report-derived labels across the twelve findings. It measures DICOM padding and
valid support, true-zero frequency, native physical field of view, slice gaps,
square-padding fraction, and a rough intensity-foreground extent proxy by
plane and report state.

The official target values are not used; their nullness is used only to exclude
the 58 gold studies from the sample. The report extraction kernel is attached
as an input, but report text and study/series identifiers are not written to
the output. Aggregate CSV/JSON files and plots are written to
`/kaggle/working/input_fidelity_audit`.

The first cell intentionally does not install dependencies. If the Kaggle image
lacks `pydicom`, stop and attach a locally prepared wheel as an input rather
than relying on notebook internet. No GPU, Hugging Face model, Modal volume, or
training quota is needed.

Build the notebook after editing its source:

```bash
python build_notebook.py
```

To publish a private kernel version from this directory:

```bash
kaggle kernels push -p .
```
