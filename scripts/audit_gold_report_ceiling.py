"""How well do report readers agree with the 58 host gold labels? (evaluation only)

Runs the HPO rule reader (each v3.1 correction step) and the three LLM-derived readers that cover the gold IDs on the
gold studies' own reports, and scores each against the gold labels per target. Gold is used for scoring only:
no clause text is printed or saved, and nothing here feeds a label build.

Container:
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_gold_report_ceiling.py /proj /out
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(R / "label_rules/v3_1"))
import rule_extractor_v31 as rx31  # noqa: E402
import build_hpo_v3_1 as b  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
T = rx31.T
NEGP = 0.04  # silent reader -> negative (report silence ~ absent)
STEPS = [("s0_v30", ()), ("s1_scope", ("scope",)), ("s2_synovitis", ("scope", "synovitis")),
         ("s3_oa_extent", ("scope", "synovitis", "oa_extent")), ("s4_baker", ("scope", "synovitis", "oa_extent", "baker")),
         ("s5_v31", rx31.ALL_FIXES)]

train_csv = R / "reports_raw/train.csv"
tr = pd.read_csv(train_csv)
g = tr[tr[T].notna().all(1)].set_index("StudyInstanceUID")
assert len(g) == 58
Y = g[T].astype(int)
ids = g.index

rules = {}
for name, fx in STEPS:
    rx31.FIXES = set(fx)
    rules[name] = pd.DataFrame({t: {s: rx31.extract(r)[t][0] for s, r in g.Report.items()} for t in T}).loc[ids]
rules_path = OUT / "gold_rule_states_v31.csv"  # states only (no clauses); for this audit, never a label input
rules["s5_v31"].to_csv(rules_path, index_label="StudyInstanceUID")
Rd, reader_hashes = b.readers(R, rules_path, ids)


def score(states):
    return states.map(lambda s: b.PROB.get(s, NEGP) if isinstance(s, str) else NEGP).astype(float)


def auc(s, y):
    r = pd.Series(s).rank().to_numpy(); y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


cols, blends = {}, {}
for name in rules:
    cols[f"rules_{name}"] = {t: auc(score(rules[name][t]), Y[t]) for t in T}
for r in ("teammate", "pilkwang", "laymond"):
    cols[r] = {t: auc(score(Rd[r][t]), Y[t]) for t in T}
# v3-style reader blend without HPO-v1 (HPO-v1 has no gold rows): mean of voting readers, all-silent -> NEG
for tag, rk in (("blend_v30rules", "s0_v30"), ("blend_v31rules", "s5_v31")):
    RR = dict(Rd); RR["rules"] = rules[rk].replace({"SIL": np.nan})
    votes = sum(RR[k][T].apply(lambda c: c.map(b.PROB)).fillna(0) for k in RR)
    n = sum(RR[k][T].notna().astype(int) for k in RR)
    blends[tag] = (votes / n.replace(0, np.nan)).fillna(NEGP)
    cols[tag] = {t: auc(blends[tag][t], Y[t]) for t in T}
A = pd.DataFrame(cols)
A.loc["MACRO"] = A.mean()
A.insert(0, "gold_positives", list(Y.sum()) + [np.nan])
A.round(4).to_csv(OUT / "gold_auc_by_reader.csv", index_label="target")

rng = np.random.default_rng(0)
B31, B30 = blends["blend_v31rules"], blends["blend_v30rules"]
boot = []
for t in T + ["MACRO"]:
    d = []
    for _ in range(2000):
        i = rng.integers(0, 58, 58)
        if t == "MACRO":
            a = np.nanmean([auc(B31[u].iloc[i], Y[u].iloc[i]) for u in T]); c = np.nanmean([auc(B30[u].iloc[i], Y[u].iloc[i]) for u in T])
        else:
            a, c = auc(B31[t].iloc[i], Y[t].iloc[i]), auc(B30[t].iloc[i], Y[t].iloc[i])
        d.append(a - c)
    d = np.array(d); d = d[np.isfinite(d)]
    boot.append({"target": t, "delta_v31_minus_v30": round(float(A.at[t, "blend_v31rules"] - A.at[t, "blend_v30rules"]), 4),
                 "ci_low": round(float(np.percentile(d, 2.5)), 4), "ci_high": round(float(np.percentile(d, 97.5)), 4),
                 "valid_draws": int(len(d))})
pd.DataFrame(boot).to_csv(OUT / "gold_blend_v31_vs_v30_bootstrap.csv", index=False)

ct = []
for t in ("ACL", "MCL", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Baker's"):
    x = pd.crosstab(Y[t].map({1: "gold_pos", 0: "gold_neg"}), rules["s5_v31"][t])
    for lab, row in x.iterrows():
        ct.append({"target": t, "gold": lab, **row.to_dict()})
pd.DataFrame(ct).fillna(0).to_csv(OUT / "gold_rule_state_by_gold_label.csv", index=False)

meta = {"gold_studies": 58, "gold_positives": Y.sum().to_dict(),
        "inputs_sha256": {"train_csv": hashlib.sha256(train_csv.read_bytes()).hexdigest(),
                          "rule_extractor_v31": hashlib.sha256((R / "label_rules/v3_1/rule_extractor_v31.py").read_bytes()).hexdigest(),
                          **{f"reader_{k}": v for k, v in reader_hashes.items() if k != "rules"}},
        "bootstrap": {"draws": 2000, "seed": 0}, "silent_reader_probability": NEGP}
(OUT / "gold_ceiling_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
pd.set_option("display.width", 250)
print(A.round(3).to_string())
print(pd.DataFrame(boot).to_string(index=False))
