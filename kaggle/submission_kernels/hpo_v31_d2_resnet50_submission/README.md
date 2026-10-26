# HPO v3.1 D2 submission

Submitted version 1: https://www.kaggle.com/code/tumis001/rsna-hpo-v31-d2-resnet50-submission

Successful 0.924 ResNet50 inference code reused with epoch-15 checkpoint SHA256 4091948e1ce9000e8f4ab7dbd7d859de9370a31d3fe09bd09bf99eaf45d8e48c. Fixed input paths; no recursive searching. PyTorch GPU inference, same as reference. Visible test: 3 studies, 144 windows, zero decode failures or empty studies, internal elapsed 14.2 seconds.

Version 1 summary carries a stale HPO-v1 source description from the reference script; actual pinned checkpoint/config is HPO-v3.1-D2. The local source description is corrected; no inference behavior changes and no additional notebook version launched.
