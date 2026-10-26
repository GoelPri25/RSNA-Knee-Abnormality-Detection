"""Compact status of a v4 job (login node, python 3.6): job state, job log without dmon rows, one line per epoch per run."""
import glob
import json
import os
import re
import subprocess
import sys

J = sys.argv[1]
st = subprocess.run(["squeue", "-j", J, "-h", "-o", "%T"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    universal_newlines=True).stdout.strip()
print("STATE " + (st or "ENDED"))
H = os.path.expanduser("~/rsna")
LOG = (glob.glob(os.path.join(H, "logs", "rsna-*-%s.out" % J)) or [""])[0]
for line in open(LOG) if os.path.exists(LOG) else []:
    if not line.startswith(" ") and not line.startswith("#"):
        print("JOB " + line.rstrip())
for f in sorted(sum((glob.glob(os.path.join(H, "out", p, "train.log")) for p in (sys.argv[2:] or ["v4_*", "lv2_*"])), [])):
    d = os.path.basename(os.path.dirname(f))
    for line in open(f, errors="replace"):
        line = line.rstrip()
        if re.match(r"EPOCH \d+ \{", line):                       # v4 runner
            r = json.loads(line[line.index("{"):])
            print("{}: ep {} fused {:.4f} global {:.4f} local {:.4f} f-g {:+.4f} gap {:+.3f} bce {:.4f}".format(
                d, r["epoch"], r["lv2_fused"], r["lv2_global"], r["lv2_local"], r["lv2_fused"] - r["lv2_global"],
                r["gap_local10"], r["lv2_bce_fused"]))
        elif line.startswith("EPOCH "):                                      # label-v2 r50 runner
            m = re.match(r"EPOCH (\d+): loss ([\d.]+) val macro AUC ([\d.]+|-) val BCE ([\d.]+|-)", line)
            print("{}: ep {} loss {} valAUC(steven) {} valBCE {}".format(d, *m.groups()) if m else d + ": " + line[:150])
        elif line.startswith("RUN_SUMMARY"):
            r = json.loads(line[len("RUN_SUMMARY "):])
            print(d + ": SWA " + " ".join("{} {:.4f}".format(k, v) for k, v in r.items() if isinstance(v, float)))
        elif re.search(r"Traceback|Error|error:|OOM|Killed", line):
            print(d + ": " + line[:200])
