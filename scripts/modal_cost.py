"""Modal cost per app (run) from the workspace billing report, split into GPU / CPU / memory.

Usage: python3 scripts/modal_cost.py [--for today|yesterday|this week|...]   (default: today)
Billing data covers completed hourly intervals only, so a running app's cost lags by up to an hour.
"""
import json
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")
from run_modal_personal_convnext_v31 import personal_env  # noqa: E402

rng = sys.argv[sys.argv.index("--for") + 1] if "--for" in sys.argv else "today"
apps = {a["App ID"]: a for a in json.loads(subprocess.run(["modal", "app", "list", "--json"], env=personal_env(),
                                                          capture_output=True, text=True, check=True).stdout)}
rows = json.loads(subprocess.run(["modal", "billing", "report", "--for", rng, "-r", "h", "--show-resources", "--json"],
                                 env=personal_env(), capture_output=True, text=True, check=True).stdout)
cost = defaultdict(lambda: defaultdict(float)); first = {}
for r in rows:
    oid = r.get("object_id") or r.get("Object ID")
    res = r.get("resource") or r.get("Resource")
    cost[oid][res] += float(r.get("cost") or r.get("Cost"))
    t = r.get("interval_start") or r.get("Interval Start")
    first[oid] = min(first.get(oid, t), t)
print(f"{'app':28s} {'state':10s} {'first hour':26s} {'GPU':>7s} {'CPU':>7s} {'mem':>7s} {'total':>7s}")
grand = 0.0
for oid in sorted(cost, key=lambda o: first[o]):
    c = cost[oid]; gpu = sum(v for k, v in c.items() if k not in ("CPU", "Memory", "Network Egress"))
    tot = sum(c.values()); grand += tot
    print(f"{oid:28s} {apps.get(oid, {}).get('State', '?')[:10]:10s} {str(first[oid])[:26]:26s} "
          f"{gpu:7.2f} {c['CPU']:7.2f} {c['Memory']:7.2f} {tot:7.2f}")
print(f"{'total':66s} {grand:7.2f}  (USD, {rng})")
