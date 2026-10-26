"""Are Exp 1e's fold-0 misses mostly mild / borderline findings (label + subtlety) or clear findings (seeing)?

Positives = Steven-v4 confident positives (the fold-0 ruler); a positive's score = its rank among confident negatives
(the share of negatives scored lower; the mean over positives is the AUC). Misses = bottom third, hits = top third.
1e logits from the attention-side audit kernel (SWA 21-25, full 384 input; same studies, labels unused there).
Report-side descriptions of each positive, all from report text only (no image model):
  structured reader (teammate LLM extraction): status present / uncertain / absent / not_mentioned, severity
    none / mild / mild_to_moderate / moderate / moderate_to_severe / severe / unspecified;
  rule extractor state (label_rules/output/rule_states.csv): POS / BORD / BELOW / HEDGE / NEG / SIL;
  D2 soft label (HPO-v3.1-D2 structured, severity-aware).
Readout: severity mix of misses vs hits, and the AUC of each severity group against all confident negatives.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_d96_miss_severity.py /proj
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"
z = np.load(D / "attention_side_audit/kaggle_output/attention_side_fold0.npz")
ids = list(z["ids"]); ok = z["ok"]; logits = z["logits"]
ref = json.loads((D / "hpo_v31_d2_kaggle_outputs/completed/hpo_v31_d2_resnet50_fold0/steven_v4_fold0_reference.json").read_text())
T = ref["targets_in_order"]; m = {r["study_id"]: r for r in ref["labels"]}
y = np.array([[.5 if p is None else p for p in m[s]["probabilities"]] for s in ids])
w = np.array([m[s]["weights"] for s in ids])
POS, NEG = (w > 0) & (y >= .7) & ok[:, None], (w > 0) & (y <= .3) & ok[:, None]
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
rules = pd.read_csv(R / "label_rules/output/rule_states.csv", dtype=str).fillna("SIL").set_index("StudyInstanceUID")
d2 = {r["study_id"]: r["probabilities"] for r in
      json.loads((D / "coatnet_d96_fold0/assets/soft_labels_hpo_v3_1_d2_structured.json").read_text())["labels"]}
key = lambda t: t.lower().replace("'", "").replace(" ", "_")


def sev_group(s, sev):
    if s in ("absent", "not_mentioned", ""):
        return "reader: absent/silent"
    if s == "uncertain":
        return "reader: uncertain"
    return {"mild": "mild", "mild_to_moderate": "mild-moderate", "unspecified": "unspecified",
            "moderate": "moderate+", "moderate_to_severe": "moderate+", "severe": "moderate+"}.get(sev, "unspecified")


GROUPS = ["mild", "mild-moderate", "unspecified", "moderate+", "reader: uncertain", "reader: absent/silent"]
print("Positives = Steven-v4 confident positives on fold 0; miss / hit = bottom / top third by rank among negatives.")
print("Columns: share of the group's positives in each report-severity class; AUC = that class's positives vs all "
      "confident negatives.\n")
summary = []
for t, name in enumerate(T):
    neg = np.sort(logits[NEG[:, t], t])
    rows = []
    for n in np.where(POS[:, t])[0]:
        s = ids[n]
        stt, sev = (st.at[s, f"{key(name)}__status"], st.at[s, f"{key(name)}__severity"]) if s in st.index else ("", "")
        rows.append({"rank": np.searchsorted(neg, logits[n, t]) / len(neg), "grp": sev_group(stt, sev),
                     "rule": rules.at[s, name] if s in rules.index else "SIL", "d2": d2.get(s, [np.nan] * 12)[t]})
    df = pd.DataFrame(rows)
    q1, q2 = df["rank"].quantile([1 / 3, 2 / 3])
    df["part"] = np.where(df["rank"] <= q1, "miss", np.where(df["rank"] >= q2, "hit", "mid"))
    print(f"{name}: {len(df)} positives, {len(neg)} negatives, AUC {df['rank'].mean():.3f}")
    print(f"   {'':6s}" + "".join(f"{g[:13]:>15s}" for g in GROUPS) + f"{'D2 label':>10s}{'rule BORD/BELOW/HEDGE':>23s}")
    for part in ("hit", "miss"):
        g = df[df.part == part]
        share = g["grp"].value_counts(normalize=True)
        soft = g["rule"].isin(["BORD", "BELOW", "HEDGE"]).mean()
        print(f"   {part:6s}" + "".join(f"{share.get(k, 0):15.2f}" for k in GROUPS) + f"{g['d2'].mean():10.3f}{soft:23.2f}")
    print(f"   {'AUC':6s}" + "".join(f"{df[df.grp == k]['rank'].mean():15.3f}" if (df.grp == k).sum() >= 5
                                     else f"{'n<5':>15s}" for k in GROUPS))
    print(f"   {'n':6s}" + "".join(f"{(df.grp == k).sum():15d}" for k in GROUPS) + "\n")
    clear = df.grp.eq("moderate+")
    mildish = df.grp.isin(["mild", "mild-moderate", "reader: uncertain", "reader: absent/silent"])
    summary.append((name, len(df), (df.part == "miss").sum(), mildish[df.part == "miss"].mean(),
                    clear[df.part == "miss"].mean(), df[mildish]["rank"].mean(), df[clear]["rank"].mean() if clear.sum() >= 5 else np.nan))
print("Summary: share of misses that are mild / uncertain / reader-absent vs moderate+; AUC of those groups")
print(f"{'target':18s}{'pos':>5s}{'miss':>6s}{'miss mild-ish':>15s}{'miss mod+':>11s}{'AUC mild-ish':>14s}{'AUC mod+':>10s}")
for r in summary:
    print(f"{r[0]:18s}{r[1]:5d}{r[2]:6d}{r[3]:15.2f}{r[4]:11.2f}{r[5]:14.3f}{r[6]:10.3f}")
