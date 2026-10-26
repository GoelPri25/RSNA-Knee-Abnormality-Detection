# Development environment and data audit

Notes kept from the original project README (container workflow, dataset audit tools, Pillar-0 checkpoint access).

## Container-only local workflow

Do not install dependencies on the host.

```sh
docker compose build
docker compose run --rm app
```

Tests use synthetic DICOMs with deliberately misleading filenames and instance numbers.
To audit mounted data, mount it read-only and write results separately:

```sh
docker compose run --rm -v /path/to/sample:/input:ro app \
  knee-audit /input --config configs/audit.json --output outputs/audit

docker compose run --rm -v /path/to/sample:/input:ro app \
  knee-volume /input/path/to/one/series --output outputs/volume
```

The volume command writes `volume.npy`, `geometry.json`, and `views.png`. Input must be
one classic single-frame MR series. Unsupported geometry fails explicitly.

## Kaggle audit without downloading images locally

The ready-to-run notebook is [kaggle/notebooks/01_dataset_audit.ipynb](kaggle/notebooks/01_dataset_audit.ipynb).
It audits the mounted data and reconstructs up to three sampled series for visual review.

Attach the competition dataset to a Kaggle notebook. Add this working repository as a
private notebook input, or use your GitHub copy after these changes have been published.
The local working changes are not automatically pushed to GitHub.

From the project directory in the Kaggle runtime:

```sh
pip install .
sh scripts/kaggle_audit.sh /kaggle/input/ACTUAL_COMPETITION_DIRECTORY
```

Use the actual mounted directory; the script does not assume the hierarchy. It writes
`/kaggle/working/knee-audit/AUDIT.md` and `audit.json`. The default inspects CSV contents
and 32 sampled series headers, with no pixel decoding. Review extension counts for
unrecognized DICOM filenames. After reviewing columns, set `labels_csv`, `study_column`
and `target_columns` in `configs/audit.json` to audit the official gold labels explicitly.
No target names or report-derived labels are guessed.

Keep audit artifacts containing identifiers, data, reports, checkpoints and cached
embeddings out of Git. See the competition's [data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
and [evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation).

The [official MRI checkpoint](https://huggingface.co/YalaLab/Pillar0-BreastMRI) is gated.
Its access conditions must be accepted by the account used for extraction. Checkpoint
integration follows the real-data audit and visual geometry verification.
