#!/bin/bash
# Runs ON the VM: the GPU check of the online code, with the lane's env.
cd /work/scripts
env $(grep -vE '^\s*(#|$)' /work/env/pubkd_kd_e4on.env | xargs) /content/venv/bin/python -u /content/okd/vm_check.py
