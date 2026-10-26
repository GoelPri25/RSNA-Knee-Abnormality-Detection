336px submission of epoch 17, SHA256 51aa5e5b907bfce0f0ce770dd70f546b0189ff3d0b8687358459d63958fbe9d7.
Reuses the successful 224px ResNet50 inference code with checkpoint pins and config-derived 336px preprocessing. Fixed input paths; no recursive search. PyTorch AMP/channels-last, unchanged global per-study top-k pooling (48 windows fit within the 64-window chunk).
