"""Residual label mismatches after Exp 7a (label-v3 rule caps) - CPU, no LLM is run, no label is changed.

Training studies (folds 1-5) only. Label = label-v3 (7a, sha 85f0ff10...); rule state = HPO-v3.1 rules s5_fx_site.
Independent evidence per cell, from label tables we already hold (data/public_labels, CC0):
  pilkwang verdict (YES = the reader found a positive mention in the report);
  votes = number of independent labeller families (steven_v4, pilkwang, jev_final2, yunus, dread, nartaa_v1_blend;
          steven_v2 / lixin / jev_hybrid are Steven-family duplicates and are not counted) with value >= 0.5.
Groups:
  A  SIL, label >= 0.5     rules found no mention, label positive: regex miss (keep) vs unsupported inference (cap?)
  B  NEG, label >= 0.3     rules found an explicit negative
  C  POS, label < 0.5      rules found a clear positive; flag history / post-op / old wording in the evidence clause
  D  sibling               label >= 0.5 for one side (medial / lateral meniscus or OA) while the rules call that side
                           NEG / SIL and the other side POS / LIKELY / BORD (possible compartment mix-up)
Output: data/label_audits/labelv3_residuals/summary.txt + cells.csv (one row per flagged cell; no report text).

Usage (container):
  podman run --rm --network=none -v "$P":/proj localhost/rsna-label-ds python -I /proj/repository/scripts/audit_labelv3_residuals.py /proj
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]) / "repository/data"; B = R / "public_labels"; A = R / "coatnet_d96_fold0/assets"
OUT = R / "label_audits/labelv3_residuals"; OUT.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
LAB = A / "soft_labels_labelv3_rulecap.json"; RULES = R / "label_datasets/hpo_v3_1_d2/rules/s5_fx_site"
assert sha(LAB).startswith("85f0ff10") and sha(RULES / "rule_states.csv").startswith("ce4473c7")
d = json.loads(LAB.read_text()); T = d["targets_in_order"]
lab = pd.DataFrame({r["study_id"]: r["probabilities"] for r in d["labels"]}, index=T).T.astype(float)
fold = {r["study_id"]: r["fold"] for r in csv.DictReader((A / "folds.csv").open())}
tr = sorted(s for s in lab.index if fold[s] != "0"); assert len(tr) == 3524
st = pd.read_csv(RULES / "rule_states.csv").set_index("StudyInstanceUID")[T]
ev = pd.read_csv(RULES / "rule_evidence.csv", keep_default_na=False).set_index("StudyInstanceUID")[T]
FAM = {"steven_v4": "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
       "pilkwang": "pilkwang_rsna-knee-llm-labels/report_labels_v2.csv",
       "jev_final2": "riadmohamed42_jev-knee-labels/FINAL2_labels_scores.csv",
       "yunus": "yunusgmsoy_rsna-knee-llm-report-labels/yunus_llm_labels.csv",
       "dread": "dreaddevelopment_rsna-knee-labels/labels_llm_soft.csv",
       "nartaa_v1_blend": "nartaa_rsna-knee-hpo-assets/labels_v1_blend.parquet"}
src = {}
for n, f in FAM.items():
    x = pd.read_parquet(B / f) if f.endswith(".parquet") else pd.read_csv(B / f)
    src[n] = x.set_index("StudyInstanceUID")
pk = src["pilkwang"]
OLD = re.compile(r"s/p|status post|post-?op|postoperative|reconstruct|graft|previous|prior|history|old|chronic|healed|"
                 r"remote|known|z\.n\.|status na|eski|antecedente|voorgeschiedenis|vorbekannt")


def votes(s, t):
    return sum(1 for x in src.values() if s in x.index and pd.notna(x.at[s, t]) and float(x.at[s, t]) >= 0.5)


rows = []
for t in T:
    for s in tr:
        y, q = lab.at[s, t], st.at[s, t]
        g = ("A" if q == "SIL" and y >= 0.5 else "B" if q == "NEG" and y >= 0.3 else "C" if q == "POS" and y < 0.5 else None)
        if g:
            rows.append({"study_id": s, "target": t, "group": g, "label": y, "state": q, "votes": votes(s, t),
                         "pilkwang": pk.at[s, t + "__verdict"] if s in pk.index else "",
                         "old_wording": bool(OLD.search(str(ev.at[s, t]).lower())) if g == "C" else ""})
for a, b in (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA")):
    for t, o in ((a, b), (b, a)):
        for s in tr:
            if lab.at[s, t] >= 0.5 and st.at[s, t] in ("NEG", "SIL") and st.at[s, o] in ("POS", "LIKELY", "BORD"):
                rows.append({"study_id": s, "target": t, "group": "D", "label": lab.at[s, t], "state": st.at[s, t],
                             "votes": votes(s, t), "pilkwang": pk.at[s, t + "__verdict"] if s in pk.index else "",
                             "old_wording": ""})
c = pd.DataFrame(rows); c.to_csv(OUT / "cells.csv", index=False)

L = [f"labels {LAB.name} sha {sha(LAB)[:12]}; rules s5_fx_site; training studies {len(tr)}; independent families "
     f"{list(FAM)} (votes = families with value >= 0.5, max {len(FAM)})", ""]
for g, name in (("A", "SIL, label >= 0.5"), ("B", "NEG, label >= 0.3"), ("C", "POS, label < 0.5"),
                ("D", "sibling mix-up candidates")):
    x = c[c.group == g]
    L.append(f"Group {g} ({name}): {len(x)} cells")
    for t, y in x.groupby("target"):
        v = y.votes.to_numpy()
        extra = f"  old/history wording {int((y.old_wording == True).sum())}" if g == "C" else ""
        L.append(f"  {t:17s} n {len(y):4d}  pilkwang YES {int((y.pilkwang == 'YES').sum()):4d}  votes 0-1 {int((v <= 1).sum()):4d}"
                 f"  2-3 {int(((v >= 2) & (v <= 3)).sum()):4d}  4-6 {int((v >= 4).sum()):4d}  mean label {y.label.mean():.2f}" + extra)
    L.append("")
(OUT / "summary.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
