# Run R6 teacher shard 2 on your Kaggle account

## What this is

We are distilling the public 0.958 blend into one model. One member of that blend is R6, a public ConvNeXt
ensemble (rostigor) plus the small public 2.5D reader (goodpjw2008). It has to score every **training** study,
straight from the DICOMs, which only works on Kaggle. That is about 15 T4 GPU-hours, split into 3 shards of
about 1,470 studies each. Kaggle runs only 2 GPU sessions at a time per account. Shards 0 and 1 are already
running on Sumit's account, so shard 2 would otherwise wait until about 03:00 CEST. On your account it can run
now, and all three finish together.

Nothing is trained and nothing is submitted. The kernel only writes prediction CSVs.

**Files:** `teacher.py` (the script, do not edit it) and `kernel-metadata.json` (for the CLI route).
`teacher.py` sha256: `2683650086c8b2d53466505318ff3d626a1de129afdaf8c4d71ee656c3512e67`

## Before you start

- You need to be on the team for `rsna-knee-abnormality-detection`, so the competition data attaches.
- You need at least about 6 h of weekly GPU quota left (shown in your Kaggle profile, under Settings / Quota).
- You need a free GPU session: no more than 1 other GPU notebook running on your account.

## Option A: Kaggle CLI (fastest)

1. Put both files in an empty folder, for example `pubkd_s2/`.
2. In `kernel-metadata.json`, replace `TEAMMATE_USERNAME` with your Kaggle username. Leave everything else
   unchanged: private, GPU on, internet off, T4, the 3 datasets and the competition.
3. Optionally check the file: `sha256sum teacher.py` should print the hash above.
4. Run `kaggle kernels push -p pubkd_s2`
5. Check it started: `kaggle kernels status <your-username>/pubkd-r6-teacher-s2`. It should say `RUNNING`, or
   `QUEUED` for a few minutes first.

If the push says "Maximum batch GPU session count of 2 reached", stop one of your other GPU sessions and push
again.

## Option B: Kaggle website

1. Go to **Code → New Notebook**.
2. **File → Import Notebook** is not needed. Delete the starter cell, add one code cell, and paste the
   **entire** contents of `teacher.py` into it. It is long (about 190 KB). Make sure the end of the file is
   pasted too: the last lines write `receipt_shard2.json`.
3. In the right-hand panel, **Input → Add Input**:
   - Competition: `rsna-knee-abnormality-detection`
   - Dataset: `rostigor/rsna-knee-r6-fe055406`
   - Dataset: `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`
   - Dataset: `goodpjw2008/rsna-knee-2-5d-convnext-reader`
4. **Settings**:
   - Accelerator: **GPU T4 x2**. It must be the two-T4 option: the script runs one worker per GPU.
   - Internet: **Off**.
   - Language: Python. Keep the default environment.
5. Click **Save Version**, choose **Save & Run All (Commit)**, then Save. Do not use Quick Save; it does not run
   anything.
6. You can close the browser. The run continues on Kaggle.

## What a healthy run looks like

In the version's log (Option A: `kaggle kernels output <your-username>/pubkd-r6-teacher-s2 -p out/` after
it ends; Option B: the version page → Logs):

- `TRAIN_STUDIES 4407 SHARD 2 of 3 studies 1469` near the top. **If the numbers differ, stop the run and tell
  Sumit.**
- Two lines saying `Tesla T4`.
- `ASSETS_READY ...` with `flip_tta True`.
- `WORKER_START 0 ...` and `WORKER_START 1 ...`. Then for about 5 minutes only `PROGRESS ... (starting)`
  appears; that is a one-time compile check, so this is normal.
- Then progress lines with a growing `done` count, at about 20 s per study per worker. The `fallback` count
  should stay at 0 or close to it.
- At the end, the reader part runs, then `receipt_shard2.json` is written.

**Expected time:** about 4.5–5 h, plus queue time. The script has its own time guards and stops itself well
inside Kaggle's 12 h limit.

## When it finishes

The version status shows **Complete**, not Error. The output (Data tab / `/kaggle/working`) must contain:

- `r6_train_shard2.csv`: about 1,469 rows, the study id, 12 probability columns and `z_` columns
- `reader_train_shard2.csv`: about 1,469 rows, the study id and 12 probability columns
- `receipt_shard2.json`

There will be other files too (worker logs, `partial_*`, stacker CSVs). Send them all; they help with checking.

**Getting it to Sumit (pick one):**

1. Easiest: open the notebook → **Share → add `tumis001` as a collaborator (viewer is enough)**. Sumit then
   downloads the output himself.
2. Or download the output (zip) and send it to Sumit directly. It is only predictions on training studies, but
   keep it inside the team.

## If something goes wrong

- **Status Error near the start:** usually a missing input. Check that all 3 datasets and the competition are
  attached, then run again.
- **"CUDA not available" or only 1 GPU in the log:** the accelerator was not set to GPU T4 x2. Fix it and run
  again.
- **It runs out of time or stops with partial results:** send whatever output exists. The `partial_r6_worker*`
  files let us rerun only the missing studies.
- **The study count is not 1,469 (shard 2 of 3):** the file was edited or cut off. Paste it again in full.
- Anything else: send Sumit the log and the error lines.

Please don't change the code, the shard number or the GPU setting. All three shards must be scored by
exactly the same pipeline.
