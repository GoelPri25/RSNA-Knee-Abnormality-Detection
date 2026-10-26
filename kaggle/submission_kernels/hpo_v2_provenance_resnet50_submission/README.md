# HPO-v2 provenance ResNet-50 submission

Inference-only submission using the verified epoch-15 checkpoint from the
HPO-v2 provenance-label experiment. It loads the `SliceModel` from the HPO
training source and the preprocessing modules from the top-k source dataset;
the shared dataset/schema modules have matching SHA-256 hashes. The selected
source files and checkpoint are hash-pinned. Input paths are explicit; there
is no recursive search of `/kaggle/input`.

Before inference the script verifies the checkpoint hash and epoch, run-summary
selection, target schema, architecture, output row order, and probability
bounds. It writes a schema-correct fallback file before decoding test images,
then replaces it with predictions after successful inference.
Inference uses PyTorch AMP on the attached T4. It does not use TensorRT.

The notebook is published and submitted with Kaggle CLI:

```bash
kaggle kernels push -p kaggle/submission_kernels/hpo_v2_provenance_resnet50_submission
kaggle competitions submit rsna-knee-abnormality-detection \
  -f submission.csv \
  -k tumis001/rsna-hpo-v2-provenance-resnet50-submission \
  -v VERSION \
  -m "HPO-v2 provenance ResNet50 epoch 15"
```
