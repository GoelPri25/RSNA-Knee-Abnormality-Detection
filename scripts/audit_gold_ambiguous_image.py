"""Gold-58 mechanism check (development evidence only, not for selection): within cells whose report is
ambiguous (mild / mild-to-moderate / unspecified severity / uncertain, per the teammate structured reader), do the
d96 image models separate the competition labels? Pooled over targets using within-target prediction percentiles;
1,000 cell-level bootstraps, 90% interval (cells of one study are not independent, so intervals are optimistic). Usage: python -I scripts/audit_gold_ambiguous_image.py  (expects /proj)."""
import csv, json, numpy as np, pandas as pd
R = "/proj"; D = R + "/repository/data/coatnet_d96_fold0"
T = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
st = pd.read_csv(R + "/colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv", dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
def cls(s, t):
    if s not in st.index: return None
    a, v = st.at[s, f"{key(t)}__status"], st.at[s, f"{key(t)}__severity"]
    if a in ("absent", "not_mentioned"): return "absent"
    if a == "uncertain": return "amb"
    return "clear" if v in ("moderate", "moderate_to_severe", "severe") else "amb"   # mild, mild-mod, unspec, uncertain = report-ambiguous
gold = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(R + "/reports_raw/train.csv", encoding="utf-8")) if all(r[t] in ("0","1") for t in T)}
def auc(p, y):
    p, y = np.asarray(p), np.asarray(y, bool)
    if y.all() or (~y).all(): return float("nan")
    r = pd.Series(p).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1*(n1+1)/2) / (n1*n0)
rng = np.random.default_rng(0)
for run in ["run_r50_pertarget", "run_r50_pertarget_seed", "run_r50", "run_r50_native_pertarget"]:
    pr = {r["study_id"]: r["probabilities"] for r in json.load(open(f"{D}/{run}/gold_swa.json"))["predictions"]}
    ids = sorted(gold)
    rows = []
    for k, t in enumerate(T):
        p = np.array([pr[s][k] for s in ids]); pct = pd.Series(p).rank(pct=True).values   # within-target percentile, for pooling
        for s, q in zip(ids, pct):
            rows.append((t, cls(s, t), q, int(gold[s][t])))
    df = pd.DataFrame(rows, columns=["t", "c", "q", "y"])
    out = []
    for c in ["clear", "amb", "absent"]:
        x = df[df.c == c]
        b = [auc(*(lambda i: (x.q.values[i], x.y.values[i]))(rng.integers(0, len(x), len(x)))) for _ in range(1000)]
        out.append(f"{c}: n {len(x)} pos {x.y.sum()} AUC {auc(x.q, x.y):.3f} [{np.nanpercentile(b,5):.3f}, {np.nanpercentile(b,95):.3f}]")
    print(f"{run:26s} " + " | ".join(out))
