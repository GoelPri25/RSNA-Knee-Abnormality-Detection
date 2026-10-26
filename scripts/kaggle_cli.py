"""Container entry point: credentials mounted read-only, never copied into the project."""
import os
from pathlib import Path
import runpy

os.environ['KAGGLE_API_TOKEN'] = Path('/credentials/access_token').read_text().strip()
os.environ['KAGGLE_CONFIG_DIR'] = '/tmp/kaggle-config'
runpy.run_module('kaggle', run_name='__main__')
