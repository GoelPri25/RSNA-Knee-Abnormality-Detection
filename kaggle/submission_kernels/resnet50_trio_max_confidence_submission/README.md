Same as resnet50_trio_probability_submission (same pinned checkpoints, preprocessing and inference), except the combination: per study and target, the member probability furthest from 0.5 is used instead of the equal mean. Fold-0 Steven-v4 macro AUC 0.8982 vs 0.9077 for the mean (audit_fold0_submission_candidates.py); submitted as a public-LB comparison at the user's request.

## Public scoring result

2026-10-08: submission 56964142 (kernel version 1) scored public macro AUC **0.929**, vs 0.935 for the equal probability mean of the same three members (submission 56961865). Matches the fold-0 ordering (most-confident 0.898 vs mean 0.908). Keep the mean.
