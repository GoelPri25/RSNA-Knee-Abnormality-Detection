"""Spawn a deployed function of rsna-e4-lanes detached (returns at once; the call keeps running on Modal).
Usage: MODAL_PROFILE=<profile> python spawn.py prepare | MODAL_PROFILE=<profile> python spawn.py train <lane> <bb> [val_fold] [seed]
"""
import os
import sys

import modal

fn = modal.Function.from_name(os.environ.get("E4_APP", "rsna-e4-lanes"), sys.argv[1])
call = fn.spawn(*sys.argv[2:])
print(f"spawned {sys.argv[1]} {sys.argv[2:]} call {call.object_id}")
