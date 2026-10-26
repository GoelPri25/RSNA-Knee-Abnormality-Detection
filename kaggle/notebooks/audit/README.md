# Private Kaggle data gate

`audit.ipynb` bundles this branch's Python source so no GitHub publication or code
dataset upload is required. Regenerate it after source edits:

```sh
python scripts/prepare_kaggle_notebook.py
```

Set the username in `kernel-metadata.json` to the authenticated Kaggle account, then
run `kaggle kernels push -p notebooks/kaggle` in the authenticated CLI container.
This creates and executes a **private CPU notebook**, attaches the competition and
keeps real images, CSVs, identifiers and generated volumes on Kaggle. Internet is
needed to install pydicom. GPU is deliberately deferred until geometry is reviewed.

Only code is uploaded. No credentials are embedded. Authentication was verified for `tumis001` using the access-token CLI.
The notebook is configured as `tumis001/rsna-knee-pillar0-data-audit`.
Use `scripts/kaggle_cli.py` inside a container with the credential directory mounted
read-only at `/credentials`; it loads the token without embedding it in notebook code.
