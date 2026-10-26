# RadImageNet-initialized ResNet-50 experiment

This run isolates **initialization** within the existing HPO-v1 +281 ResNet-50
Modal setup. It uses the same architecture, labels, fixed report-validation
studies, training-only additions, 24-window sampling, batch size 8, loss,
optimizer, scheduler, and AUC early-stopping rule as the ImageNet-initialized
single-T4 ResNet-50 run. The only intended change is the ResNet-50 trunk weights.

The 4,349 HPO-v1 report-labeled studies supply training labels. The 281 studies
without an original fold assignment are added to training only; the original
825 fold-0 studies remain the report-validation set and are evaluated against
Steven v4. Gold studies remain excluded. The pretrained classification head is
not reused: the 12-output knee head and optimizer state are new. In the
six-channel input stem, the RadImageNet RGB weights populate the three image
channels and the three acquired-support-mask channels start at zero.

The checkpoint was manually downloaded from the PyTorch weights link in the
[official RadImageNet repository](https://github.com/BMEII-AI/RadImageNet),
which points to the authors' [Google Drive archive](https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing).
The archive is kept in the ignored project data folder at
`rsna/data/RadImageNet_pytorch.zip`; the extracted ResNet-50 state dict is
`rsna/data/pretrained_weights/radimagenet_resnet50.pt`.

The extracted checkpoint SHA-256 is
`08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734`.
The official checkpoint stores feature layers using sequential module indices.
The isolated trainer maps those indices to torchvision's named ResNet-50
modules, checks every parameter and buffer, and adapts only the first
convolution from three to six input channels. A CPU preflight verifies the
checkpoint hash, strict trunk load, zero-initialized support channels, and a
synthetic `[1, 6, 224, 224] -> [1, 12]` forward pass before the T4 function is
started.

The run uses one Modal T4 with a 12-hour function timeout. It resumes derived
slice shards from the existing Modal cache volume, does not transfer raw
DICOMs, and publishes checkpoints and histories to a new private Kaggle
artifact dataset every two epochs. The run can be launched detached with:

```bash
python scripts/run_modal_personal_radimagenet.py
```

Compare the result to the **ImageNet-initialized ResNet-50 run with the same
batch size and data recipe**, not directly to the ResNet-34 0.918 submission.
The public 0.918 score remains a useful broader reference, but this experiment
changes the backbone family relative to that submission.
