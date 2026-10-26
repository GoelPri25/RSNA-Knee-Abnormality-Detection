"""AM-0 breakdown (post-hoc, no gate; user 2026-10-17: "how many cases of unsure per target is it getting hurt ... where it
could improve. give me full distribution"). Fold 0, Exp 1e maps, judge = label-v2 (confident: weight > 0, y <= 0.3 / >= 0.7).

Old score = 1e logit. New scores = total positive CAM mass (evidence for the finding, counter-evidence dropped) and
rank-mean(logit, mass). Per finding x report state (unsure states only), within the same (finding, state):
  pairs fixed  = judge-positive / judge-negative pairs ranked wrong by the logit and right by the new score; broken = reverse
  cases helped / hurt = judged cells whose own pairs gain more fixed than broken (helped) or the reverse (hurt)
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_attention_map_breakdown.py <project root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"
OUT = D / "attention_side_audit/attention_map_breakdown_labelv2.txt"; lines = []
say = lambda s="": (print(s), lines.append(s))
z = np.load(D / "attention_side_audit/kaggle_output/attention_side_fold0.npz")
ids = list(z["ids"]); ok = z["ok"]; logit = z["logits"].astype(float)
mass = z["camx"].astype(np.float64).sum((1, 3))                                   # [S, 12] total positive CAM mass
ref = json.loads((D / "teammate_labelv2/soft_labels_final_per_target_latD2.json").read_text())
T = ref["targets_in_order"]; m = {r["study_id"]: r for r in ref["labels"]}
y = np.array([[.5 if p is None else p for p in m[s]["probabilities"]] for s in ids])
w = np.array([m[s]["weights"] for s in ids])
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")


def state(s, t):
    if s not in st.index:
        return "NA"
    a, v = st.at[s, f"{key(t)}__status"], st.at[s, f"{key(t)}__severity"]
    if a == "absent":
        return "NEG"
    if a in ("not_mentioned", ""):
        return "SIL"
    if a == "uncertain":
        return "UNC"
    if v in ("moderate", "moderate_to_severe", "severe"):
        return "MOD+"
    return "MILD" if v in ("mild", "mild_to_moderate") else "UNSP"


S = np.array([[state(s, t) for t in T] for s in ids])
UNSURE = ("SIL", "MILD", "UNC", "UNSP")
jp = (w > 0) & (y >= .7); jn = (w > 0) & (y <= .3)


def rank01(V):
    return np.column_stack([pd.Series(V[:, k]).rank(pct=True).to_numpy() for k in range(12)])


NEW = {"mass": mass, "rank-mean": (rank01(logit) + rank01(mass)) / 2}


def cmp(old, new, p, n):
    """pair sign matrices (+1 right, 0 tie, -1 wrong) -> fixed, broken, per-case net, AUCs"""
    so = np.sign(old[p][:, None] - old[n][None, :]); sn = np.sign(new[p][:, None] - new[n][None, :])
    fix = (sn > so); brk = (sn < so)
    net_p = fix.sum(1) - brk.sum(1); net_n = fix.sum(0) - brk.sum(0)
    auc = lambda s_: ((s_ > 0).sum() + 0.5 * (s_ == 0).sum()) / s_.size
    return fix.sum(), brk.sum(), net_p, net_n, auc(so), auc(sn)


for name, V in NEW.items():
    say(f"===== new score: {name}  (old = 1e logit; judge = label-v2) =====")
    say(f"{'finding':17s} {'state':5s} {'unsure':>6s} {'j+':>4s} {'j-':>5s} {'pairs':>6s} {'fixed':>6s} {'broken':>6s} "
        f"{'net':>6s}  AUC old->new     pos help/hurt  neg help/hurt")
    rows = []
    for k, t in enumerate(T):
        for s_ in UNSURE:
            u = ok & (S[:, k] == s_)
            p = np.where(u & jp[:, k])[0]; n = np.where(u & jn[:, k])[0]
            r = dict(finding=t, state=s_, unsure=int(u.sum()), jpos=len(p), jneg=len(n), pairs=len(p) * len(n))
            if len(p) and len(n):
                f, b, npn, nnn, ao, an = cmp(logit[:, k], V[:, k], p, n)
                r.update(fixed=int(f), broken=int(b), net=int(f - b), auc_old=ao, auc_new=an,
                         pos_help=int((npn > 0).sum()), pos_hurt=int((npn < 0).sum()),
                         neg_help=int((nnn > 0).sum()), neg_hurt=int((nnn < 0).sum()))
            rows.append(r)
            if r["pairs"]:
                say(f"{t:17s} {s_:5s} {r['unsure']:6d} {r['jpos']:4d} {r['jneg']:5d} {r['pairs']:6d} {r['fixed']:6d} "
                    f"{r['broken']:6d} {r['net']:+6d}  {r['auc_old']:.3f} -> {r['auc_new']:.3f}  "
                    f"{r['pos_help']:4d}/{r['pos_hurt']:<4d}     {r['neg_help']:4d}/{r['neg_hurt']:<4d}")
            else:
                say(f"{t:17s} {s_:5s} {r['unsure']:6d} {r['jpos']:4d} {r['jneg']:5d}      0   (no judged pairs)")
    df = pd.DataFrame(rows).fillna(0)
    for by in ("finding", "state"):
        g = df.groupby(by, sort=False)[["unsure", "jpos", "jneg", "pairs", "fixed", "broken", "net",
                                        "pos_help", "pos_hurt", "neg_help", "neg_hurt"]].sum()
        g["AUC old"] = 0.0; g["AUC new"] = 0.0
        for key_, sub in df.groupby(by, sort=False):
            sub = sub[sub.pairs > 0]
            if len(sub):
                g.loc[key_, "AUC old"] = (sub.auc_old * sub.pairs).sum() / sub.pairs.sum()
                g.loc[key_, "AUC new"] = (sub.auc_new * sub.pairs).sum() / sub.pairs.sum()
        g["cases help"] = g.pos_help + g.neg_help; g["cases hurt"] = g.pos_hurt + g.neg_hurt
        say(f"\n-- totals by {by} --")
        say(g[["unsure", "jpos", "jneg", "pairs", "fixed", "broken", "net", "AUC old", "AUC new", "cases help",
               "cases hurt"]].round(3).to_string())
    tot = df.sum(numeric_only=True)
    say(f"\nALL unsure: {int(tot.unsure)} cells, judged {int(tot.jpos)} pos / {int(tot.jneg)} neg, {int(tot.pairs)} pairs: "
        f"fixed {int(tot.fixed)}, broken {int(tot.broken)}, net {int(tot.net):+d}; cases helped "
        f"{int(tot.pos_help + tot.neg_help)}, hurt {int(tot.pos_hurt + tot.neg_hurt)}\n")
OUT.write_text("\n".join(lines) + "\n")
