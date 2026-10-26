"""Label-v3 (CPU, no LLM, no model): label-v2 with rule-state caps on every target.

Base: the teammate's label-v2 (soft_labels_final_per_target_latD2.json, LB 0.937 as Exp 1e retrained). Rule states:
the HPO-v3.1 rule extractor's final stage (rules/s5_fx_site/rule_states.csv of the D2 release), regex only.
For every study and target, downward only:
  BELOW (present but below the host threshold: small / trace / grade I)  -> label = min(label, 0.35)
  HEDGE (possible / suspected / cannot exclude)                         -> label = min(label, 0.40)
All other states (POS, LIKELY, BORD, NEG, SIL) and all weights are left exactly as they are.
Cap values: pooled gold-58 positive rate by v3.1 rule state (BELOW 18/50 = 0.36, HEDGE 7/19 = 0.37; gold studies are
not in the training file), rounded and fixed before any run. Variants:
  rulecap        caps on all 12 targets
  rulecap_noeff  same, Effusion left as label-v2 (isolates the largest and most contested target)
Applied to all 4,349 rows; the fold-0 reference (validation_labels.json), the rulers and gold are separate files and
are not touched. Writes repository/data/label_releases/labelv3_<variant>/soft_labels_labelv3_<variant>.json + summary.txt.

Usage (container):
  podman run --rm --network=none -v "$P":/proj localhost/rsna-label-ds python -I /proj/repository/scripts/build_rulecap_labels.py /proj
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1]) / "repository/data"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
BASE = R / "teammate_labelv2/soft_labels_final_per_target_latD2.json"
RULES = R / "label_datasets/hpo_v3_1_d2/rules/s5_fx_site/rule_states.csv"
FOLDS = R / "coatnet_d96_fold0/assets/folds.csv"
assert sha(BASE).startswith("dd0ab9ab") and sha(RULES).startswith("ce4473c7") and sha(FOLDS).startswith("2dd1044a")
CAP = {"BELOW": 0.35, "HEDGE": 0.40}
base = json.loads(BASE.read_text()); T = base["targets_in_order"]
state = {r["StudyInstanceUID"]: r for r in csv.DictReader(RULES.open())}
fold = {r["study_id"]: r["fold"] for r in csv.DictReader(FOLDS.open())}
assert {r["study_id"] for r in base["labels"]} <= set(state) and len(base["labels"]) == 4349

for variant, skip in (("rulecap", ()), ("rulecap_noeff", ("Effusion",))):
    rows, lines = [], []
    moved = {t: {s: [0, 0, 0.0] for s in CAP} for t in T}               # [training cells, fold-0 cells, sum of drops]
    for r in base["labels"]:
        sid, probs = r["study_id"], list(r["probabilities"])
        for k, t in enumerate(T):
            c = CAP.get(state[sid][t])
            if t in skip or c is None or probs[k] is None or probs[k] <= c:
                continue
            m = moved[t][state[sid][t]]; m[fold[sid] == "0"] += 1; m[2] += probs[k] - c
            probs[k] = c
        rows.append({**r, "probabilities": probs})
    out = R / f"label_releases/labelv3_{variant}"; out.mkdir(parents=True, exist_ok=True)
    name = f"soft_labels_labelv3_{variant}.json"
    doc = {**{k: v for k, v in base.items() if k != "labels"}, "version": f"labelv3_{variant}",
           "note": base.get("note", "") + f" | Label-v3 ({variant}): rule-state caps BELOW <= 0.35, HEDGE <= 0.40, downward "
                   f"only, from HPO-v3.1 rules s5_fx_site (sha ce4473c7...)" + ("; Effusion unchanged" if skip else ""),
           "labels": rows}
    (out / name).write_text(json.dumps(doc) + "\n")
    lines.append(f"{name}: sha256 {sha(out / name)}")
    lines.append(f"base label-v2 sha {sha(BASE)}; rules sha {sha(RULES)}; caps {CAP}; skipped {list(skip)}")
    tot = [0, 0]
    for t in T:
        parts = []
        for s in CAP:
            a, b, d = moved[t][s]; tot[0] += a; tot[1] += b
            parts.append(f"{s} train {a:4d} fold0 {b:3d} mean drop {d / max(a + b, 1):.2f}")
        lines.append(f"  {t:17s} " + " | ".join(parts))
    lines.append(f"total cells moved: training {tot[0]}, fold-0 rows {tot[1]}")
    k_ok = all(np.isfinite(p) and 0 <= p <= 1 for r in rows for p in r["probabilities"] if p is not None)
    lines.append(f"rows {len(rows)}; all probabilities in [0, 1]: {k_ok}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines) + "\n")
