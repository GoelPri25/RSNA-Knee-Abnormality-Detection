"""head_predictions.json ({study: {"global": [12], "local": [12]}}) -> eval_predictions-style JSON for one head.
Usage: heads_to_eval.py head_predictions.json eval_predictions.json HEAD OUT.json (targets order from the eval file)."""
import json
import sys

heads, ev, head, out = sys.argv[1:5]
h = json.load(open(heads))
T = json.load(open(ev))["targets_in_order"]
json.dump({"targets_in_order": T, "predictions": [{"study_id": s, "probabilities": v[head]} for s, v in h.items()]},
          open(out, "w"))
print(out, len(h))
