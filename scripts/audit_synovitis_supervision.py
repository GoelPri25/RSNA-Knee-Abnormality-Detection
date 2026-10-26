"""Where does Synovitis supervision come from in HPO-v1, HPO-v3.0 and the exact v3.1-D2 file used in training?

Restricted to the fixed 3,524 training studies. No gold labels, no model scores, no GPU; nothing is written to any label
file. Report excerpts are printed from the local competition train.csv only.

Labels compared (sha256 prefixes asserted):
  HPO-v1   repository/data/label_datasets/hpo_v1_blend/soft_labels_hpo_v1_blend.json            ffdffcb2
  v3.0     label_rules/hpo_v3/soft_labels_hpo_v3_structured.json                               843014ed
  D2       repository/data/label_datasets/hpo_v3_1_d2/kaggle/soft_labels_hpo_v3_1_d2_structured.json  38367d62
           (the file whose sha is recorded in the 0.925 run's run_summary.json)

Mechanisms traced:
  1. HPO-v1 parents: Steven-v4 (labels_v0raw.parquet; 0.25 = not-mentioned marker) and Gemini-low
     (labels_v1_gemlow.parquet). HPO-v1 = mean of both where Steven-v4 addressed the finding, else Gemini-low.
  2. v3 reader blend for cells with at least one voting reader or a parent "mentioned" flag:
     L = (2*HPO_v1 + sum reader probabilities) / (2 + n_voting). The HPO-v1 share is 2/(2+n).
  3. Co-occurrence for evidence-free cells (no reader vote and no parent mention): L = 0.5*HPO_v1 + 0.5*score(other 11).
  4. Severity / scope / synovitis rule corrections between v3.0 and D2 (builder steps c0..c6), attributed step by step
     by re-running the unchanged builder with the D2 rule states; the final step must reproduce the D2 file exactly.

Report categories (from report text and the v3.1-D2 rule extractor states, never from labels):
  SYN_POS    rule Synovitis state POS / LIKELY / BORD / HEDGE (explicit or hedged synovitis)
  SYN_NEG    rule Synovitis state NEG / BELOW (explicit negative)
  SYN_OTHER  a synov- term appears (any language) but the rule reader is silent (cyst, plica, fluid, missed)
  EFF_ONLY   no synov- term; rule Effusion state is a mention other than NEG (effusion only)
  EFF_NEG    no synov- term; rule Effusion state NEG
  SILENT     no synov- term; Effusion silent

Container:
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/audit_synovitis_supervision.py /proj /out
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / "label_rules/v3_1"))
import build_hpo_v3_1 as b  # noqa: E402  (the unchanged v3.1 builder: readers, blend, split, PROB)

T = b.T
SY, EF = "Synovitis", "Effusion"
E = R / "repository/data/external/nartaa_hpo_assets"
D2DIR = R / "repository/data/label_datasets/hpo_v3_1_d2"
FILES = {"hpo_v1": ("repository/data/label_datasets/hpo_v1_blend/soft_labels_hpo_v1_blend.json", "ffdffcb2"),
         "v3_0": ("label_rules/hpo_v3/soft_labels_hpo_v3_structured.json", "843014ed"),
         "d2": ("repository/data/label_datasets/hpo_v3_1_d2/kaggle/soft_labels_hpo_v3_1_d2_structured.json", "38367d62")}
SYN_TERM = re.compile(r"s[iy]nov|синов|συνοβ|υμεν", re.I)  # υμενίτιδα = synovitis (Greek)
POSLIKE = {"POS", "LIKELY", "BORD", "HEDGE"}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    d = json.loads(p.read_text())
    return pd.DataFrame([r["probabilities"] for r in d["labels"]], index=[r["study_id"] for r in d["labels"]],
                        columns=d["targets_in_order"])


L = {}
for k, (p, pre) in FILES.items():
    assert sha(R / p).startswith(pre), k
    L[k] = load(R / p)
ids = L["hpo_v1"].index
L = {k: v.loc[ids] for k, v in L.items()}
H = L["hpo_v1"]
train, split_prov = b.split(R, ids)
tid = ids[train]

# ---- HPO-v1 parents
stev = pd.read_parquet(E / "labels_v0raw.parquet").set_index("StudyInstanceUID").reindex(ids)
gem = pd.read_parquet(E / "labels_v1_gemlow.parquet").set_index("StudyInstanceUID").reindex(ids)
recon = pd.DataFrame({t: np.where(np.isclose(stev[t], 0.25), gem[t], (stev[t] + gem[t]) / 2) for t in T}, index=ids)
hpo_recon_max_err = float((recon - H).abs().to_numpy().max())

# ---- rebuild every builder step with the D2 rule states (exact attribution)
v0 = pd.read_parquet(E / "train_labels_v0.parquet").set_index("StudyInstanceUID").reindex(ids)
v1 = pd.read_parquet(E / "train_labels_v1.parquet").set_index("StudyInstanceUID").reindex(ids)
parent_voice = pd.DataFrame({t: (v0[f"m_{k}"].fillna(0) == 1) | (v1[f"m_{k}"].fillna(0) == 1) for t, k in zip(T, b.K)})
steps = {}
for name, rules_step, cap, fit in b.CONFIGS:
    Rd, _ = b.readers(R, D2DIR / "rules" / rules_step / "rule_states.csv", ids)
    Lx, free, params, capped, n = b.blend(Rd, H, parent_voice, cap, fit, train)
    steps[name] = {"L": Lx, "free": free, "params": params, "n": n, "R": Rd}
assert np.array_equal(steps["c0_v30"]["L"].to_numpy(), L["v3_0"].to_numpy()), "c0 != v3.0"
assert np.array_equal(steps["c6_train_fit"]["L"].to_numpy(), L["d2"].to_numpy()), "c6 with D2 rules != D2 file"
fin = steps["c6_train_fit"]; c0 = steps["c0_v30"]
Rd, n, free = fin["R"], fin["n"], fin["free"]

# ---- report categories (text + D2 rule states; no labels)
rep = pd.read_csv(R / "reports_raw/train.csv").set_index("StudyInstanceUID").loc[ids, "Report"].fillna("")
rs = pd.read_csv(D2DIR / "rules/s5_fx_site/rule_states.csv").set_index("StudyInstanceUID").reindex(ids)
rs0 = pd.read_csv(D2DIR / "rules/s0_v30/rule_states.csv").set_index("StudyInstanceUID").reindex(ids)
ev = pd.read_csv(D2DIR / "rules/s5_fx_site/rule_evidence.csv").set_index("StudyInstanceUID").reindex(ids).fillna("")
term = rep.str.contains(SYN_TERM)
cat = pd.Series(np.select(
    [rs[SY].isin(POSLIKE), rs[SY].isin({"NEG", "BELOW"}), term,
     ~term & rs[EF].notna() & ~rs[EF].isin({"SIL", "NEG"}), ~term & rs[EF].eq("NEG")],
    ["SYN_POS", "SYN_NEG", "SYN_OTHER", "EFF_ONLY", "EFF_NEG"], "SILENT"), index=ids)
CATS = ["SYN_POS", "SYN_NEG", "SYN_OTHER", "EFF_ONLY", "EFF_NEG", "SILENT"]


def spear(a, c):
    return float(a.rank().corr(c.rank())) if len(a) > 2 and a.nunique() > 1 and c.nunique() > 1 else np.nan


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


SRC = {"steven_v4": stev, "gemini_low": gem, "hpo_v1": H, "v3_0": L["v3_0"], "d2": L["d2"]}

# ---- table 1: per category, Synovitis level and Effusion coupling for each source (training only)
rows = []
for c in CATS:
    m = (cat == c) & train
    row = {"category": c, "train_studies": int(m.sum())}
    for k, X in SRC.items():
        row[f"{k}_syn_mean"] = round(float(X.loc[m, SY].mean()), 3)
        row[f"{k}_syn_sd"] = round(float(X.loc[m, SY].std()), 3)
        row[f"{k}_rho_syn_eff"] = round(spear(X.loc[m, SY], X.loc[m, EF]), 3)
    rows.append(row)
m = pd.Series(train, index=ids)
row = {"category": "ALL_TRAIN", "train_studies": int(m.sum())}
for k, X in SRC.items():
    row[f"{k}_syn_mean"] = round(float(X.loc[m, SY].mean()), 3); row[f"{k}_syn_sd"] = round(float(X.loc[m, SY].std()), 3)
    row[f"{k}_rho_syn_eff"] = round(spear(X.loc[m, SY], X.loc[m, EF]), 3)
rows.append(row)
T1 = pd.DataFrame(rows); T1.to_csv(OUT / "t1_category_level_and_effusion_coupling.csv", index=False)

# ---- table 2: how much of each label's Synovitis ranking comes from studies with no synovitis text
# "share of top-decile Synovitis labels whose report has no synov- term" and AUC of the label for
# explicit-positive vs no-synov-text studies (how far no-text studies are pushed up by inferred values)
notext = cat.isin(["EFF_ONLY", "EFF_NEG", "SILENT"])
rows = []
for k, X in SRC.items():
    s = X.loc[tid, SY]; q = s >= s.quantile(0.9)
    top_notext = float(notext.loc[tid][q].mean())
    pos = (cat.loc[tid] == "SYN_POS")
    rows.append({"source": k,
                 "top_decile_share_no_synov_text": round(top_notext, 3),
                 "share_ge_0_5_no_synov_text": round(float(notext.loc[tid][s >= 0.5].mean()), 3) if (s >= 0.5).any() else None,
                 "n_ge_0_5": int((s >= 0.5).sum()),
                 "auc_synpos_vs_effonly": round(auc(np.r_[s[pos], s[cat.loc[tid] == "EFF_ONLY"]],
                                                    np.r_[np.ones(pos.sum()), np.zeros((cat.loc[tid] == "EFF_ONLY").sum())]), 3),
                 "auc_effonly_vs_silent_or_effneg": round(auc(
                     np.r_[s[cat.loc[tid] == "EFF_ONLY"], s[cat.loc[tid].isin(["SILENT", "EFF_NEG"])]],
                     np.r_[np.ones((cat.loc[tid] == "EFF_ONLY").sum()), np.zeros(cat.loc[tid].isin(["SILENT", "EFF_NEG"]).sum())]), 3),
                 "rho_syn_eff_within_no_synov_text": round(spear(X.loc[tid][notext.loc[tid]][SY], X.loc[tid][notext.loc[tid]][EF]), 3),
                 "rho_syn_eff_within_synpos": round(spear(X.loc[tid][pos][SY], X.loc[tid][pos][EF]), 3)})
T2 = pd.DataFrame(rows); T2.to_csv(OUT / "t2_ranking_sources.csv", index=False)

# ---- table 3: Steven-v4 / Gemini-low Synovitis on EFF_ONLY reports by rule Effusion severity state
m = train & (cat == "EFF_ONLY")
T3 = pd.DataFrame({"eff_state": rs.loc[m, EF], "steven_v4": stev.loc[m, SY], "gemini_low": gem.loc[m, SY], "hpo_v1": H.loc[m, SY],
                   "v3_0": L["v3_0"].loc[m, SY], "d2": L["d2"].loc[m, SY],
                   "steven_marker_0_25": np.isclose(stev.loc[m, SY], 0.25)}).groupby("eff_state").agg(
    n=("d2", "size"), steven_v4=("steven_v4", "mean"), steven_marker_share=("steven_marker_0_25", "mean"),
    gemini_low=("gemini_low", "mean"), hpo_v1=("hpo_v1", "mean"), v3_0=("v3_0", "mean"), d2=("d2", "mean")).round(3)
T3 = T3.reindex([s for s in ["POS", "LIKELY", "BORD", "HEDGE", "BELOW"] if s in T3.index])
T3.to_csv(OUT / "t3_effusion_only_by_effusion_state.csv")

# ---- table 4: readers' Synovitis votes by category (training). Shows which readers vote on effusion-only reports
rows = []
for r, X in Rd.items():
    for c in CATS:
        mm = train & (cat == c)
        vc = X.loc[mm, SY].fillna("SIL").value_counts()
        rows.append({"reader": r, "category": c, "n": int(mm.sum()), **{s: int(vc.get(s, 0)) for s in
                     ["POS", "LIKELY", "BORD", "HEDGE", "BELOW", "NEG", "SIL"]}})
T4 = pd.DataFrame(rows); T4.to_csv(OUT / "t4_reader_synovitis_votes_by_category.csv", index=False)

# ---- table 5: mechanism decomposition of the D2 and v3.0 Synovitis label per cell
def decompose(step):
    Lx, fr, nn, RR = step["L"], step["free"], step["n"], step["R"]
    w = 2 / (2 + nn[SY])
    votes = sum(RR[r][SY].map(b.PROB).fillna(0) for r in RR)
    hpo_part = np.where(fr[SY], 0.5 * H[SY], w * H[SY])
    reader_part = np.where(fr[SY], 0.0, votes / (2 + nn[SY]))
    cooc_part = np.where(fr[SY], Lx[SY] - 0.5 * H[SY], 0.0)
    return pd.DataFrame({"hpo_v1_part": hpo_part, "reader_part": reader_part, "cooc_part": cooc_part,
                         "n_voting": nn[SY], "evidence_free": fr[SY], "label": Lx[SY]}, index=ids)


rows = []
for vname, step in (("v3_0", c0), ("d2", fin)):
    Dd = decompose(step)
    assert np.allclose(Dd[["hpo_v1_part", "reader_part", "cooc_part"]].sum(1), Dd.label, atol=2e-5)
    for c in CATS:
        mm = train & (cat == c)
        x = Dd[mm]
        rows.append({"label": vname, "category": c, "n": int(mm.sum()), "evidence_free_cells": int(x.evidence_free.sum()),
                     "mean_n_voting": round(float(x.n_voting.mean()), 2),
                     "mean_label": round(float(x.label.mean()), 3), "mean_hpo_v1_part": round(float(x.hpo_v1_part.mean()), 3),
                     "mean_reader_part": round(float(x.reader_part.mean()), 3), "mean_cooc_part": round(float(x.cooc_part.mean()), 3),
                     "share_of_label_from_hpo_v1": round(float(x.hpo_v1_part.sum() / x.label.sum()), 3),
                     "rho_hpo_v1_part_vs_label": round(spear(x.hpo_v1_part, x.label), 3)})
T5 = pd.DataFrame(rows); T5.to_csv(OUT / "t5_d2_v30_decomposition_by_category.csv", index=False)

# ---- table 6: v3.0 -> D2 Synovitis changes attributed to builder steps (training cells)
rows = []
names = [c[0] for c in b.CONFIGS]
for a_, b_ in zip(names[:-1], names[1:]):
    d = (steps[b_]["L"][SY] - steps[a_]["L"][SY])[train]
    ch = d.abs() > 1e-6
    rows.append({"step": b_, "cells_changed": int(ch.sum()), "changed_gt_0_05": int((d.abs() > 0.05).sum()),
                 "mean_shift_changed": round(float(d[ch].mean()), 4) if ch.any() else 0.0,
                 "mean_abs_shift_changed": round(float(d[ch].abs().mean()), 4) if ch.any() else 0.0,
                 "max_abs_shift": round(float(d.abs().max()), 4),
                 "changed_by_category": cat[train][ch].value_counts().to_dict()})
d = (L["d2"][SY] - L["v3_0"][SY])[train]
rows.append({"step": "TOTAL v3.0->D2", "cells_changed": int((d.abs() > 1e-6).sum()), "changed_gt_0_05": int((d.abs() > 0.05).sum()),
             "mean_shift_changed": round(float(d[d.abs() > 1e-6].mean()), 4), "mean_abs_shift_changed": round(float(d[d.abs() > 1e-6].abs().mean()), 4),
             "max_abs_shift": round(float(d.abs().max()), 4), "changed_by_category": cat[train][d.abs() > 1e-6].value_counts().to_dict()})
T6 = pd.DataFrame(rows); T6.to_csv(OUT / "t6_v30_to_d2_synovitis_by_step.csv", index=False)

# rule-state transitions for Synovitis on training studies (v3.0 rules -> D2 rules)
T6b = pd.crosstab(rs0.loc[train, SY].fillna("SIL"), rs.loc[train, SY].fillna("SIL"), rownames=["v3_0_rule"], colnames=["d2_rule"])
T6b.to_csv(OUT / "t6b_synovitis_rule_transitions_train.csv")

# ---- table 7: HPO-v1 -> v3.0 Synovitis change by mechanism and category
Dd0 = decompose(c0)
d = (L["v3_0"][SY] - H[SY])
rows = []
for c in CATS:
    mm = train & (cat == c)
    for kind, km in (("evidence_free(cooc)", Dd0.evidence_free), ("reader_blend", ~Dd0.evidence_free & (Dd0.n_voting > 0)),
                     ("hpo_only(parent mention, no vote)", ~Dd0.evidence_free & (Dd0.n_voting == 0))):
        x = d[mm & km]
        rows.append({"category": c, "mechanism": kind, "cells": int(len(x)), "mean_shift": round(float(x.mean()), 4) if len(x) else None,
                     "mean_abs_shift": round(float(x.abs().mean()), 4) if len(x) else None,
                     "up_gt_0_1": int((x > 0.1).sum()), "down_gt_0_1": int((x < -0.1).sum())})
T7 = pd.DataFrame(rows); T7.to_csv(OUT / "t7_hpo_v1_to_v30_synovitis_by_mechanism.csv", index=False)

# ---- co-occurrence model for Synovitis (D2, fitted on training IDs)
p = fin["params"][SY]
coef = pd.Series(p["coef"], index=p["features_in_order"]).sort_values(key=np.abs, ascending=False)
p0 = c0["params"][SY]
coef0 = pd.Series(p0["coef"], index=p0["features_in_order"]).reindex(coef.index)
cooc = {"d2_coef_standardised": coef.round(3).to_dict(), "v3_0_coef_standardised": coef0.round(3).to_dict(),
        "d2_evidence_free_synovitis_cells_train": int((free[SY] & train).sum()),
        "v3_0_evidence_free_synovitis_cells_train": int((c0["free"][SY] & train).sum())}
# D2 vs my v3.1 build: which cells differ (co-occurrence coupling through other targets)
mine_p = R / "label_rules/hpo_v3_1/build/soft_labels_hpo_v3_1_structured.json"
mine = load(mine_p).loc[ids]
dd = (L["d2"] - mine).abs() > 1e-9
cooc["d2_vs_v3_1_build_cells_differ"] = {t: int(dd[t].sum()) for t in T}
cooc["d2_vs_v3_1_build_differing_cells_all_evidence_free_except_synovitis"] = {
    t: bool((~dd[t] | free[t]).all()) for t in T}
cooc["d2_vs_v3_1_build_synovitis_non_free_differences"] = int((dd[SY] & ~free[SY]).sum())

# ---- examples (report excerpts) per mechanism, with counterexamples
def excerpt(sid, pat):
    t = rep[sid]
    mt = re.search(pat, t, re.I)
    if not mt:
        return t[:160].replace("\n", " ")
    a = max(0, mt.start() - 90)
    return ("…" if a else "") + t[a:mt.end() + 90].replace("\n", " ") + "…"


EFF_PAT = r"effusion|erguss|derrame|épanchement|versamento|efüzyon|выпот|hydrops|fluid"
rng = np.random.default_rng(0)
ex = []


def add(group, mask, pat, k=4, sort=None):
    pool = ids[mask & train]
    if sort is not None:
        pool = sort(pool)[:k]
    elif len(pool) > k:
        pool = pool[np.sort(rng.choice(len(pool), k, replace=False))]
    for sid in pool:
        ex.append({"group": group, "study_id": sid, "category": cat[sid], "rule_syn_v3_0": rs0.at[sid, SY], "rule_syn_d2": rs.at[sid, SY],
                   "rule_eff_d2": rs.at[sid, EF], **{f"{r}_syn": Rd[r].at[sid, SY] for r in Rd},
                   "steven_v4": round(float(stev.at[sid, SY]), 3), "gemini_low": round(float(gem.at[sid, SY]), 3),
                   "hpo_v1": round(float(H.at[sid, SY]), 3), "v3_0": L["v3_0"].at[sid, SY], "d2": L["d2"].at[sid, SY],
                   "d2_effusion": L["d2"].at[sid, EF], "evidence_free_d2": bool(free.at[sid, SY]),
                   "excerpt": excerpt(sid, r"s[iy]nov|синов|συνοβ" if term[sid] else EFF_PAT)})


eff_only = cat.eq("EFF_ONLY")
add("A effusion-only, high HPO-v1 Synovitis (copying)", eff_only & (H[SY] >= 0.3), EFF_PAT)
add("B effusion-only, low HPO-v1 Synovitis (counterexample)", eff_only & (H[SY] < 0.12) & rs[EF].isin(["POS", "LIKELY"]), EFF_PAT)
add("C explicit synovitis, D2 POS (independent supervision)", cat.eq("SYN_POS") & rs[SY].eq("POS"), SYN_TERM.pattern)
add("D explicit synovitis but low D2 label (counterexample)", cat.eq("SYN_POS"), SYN_TERM.pattern,
    sort=lambda pool: pool[np.argsort(L["d2"].loc[pool, SY].to_numpy())])
add("E explicit synovitis negative", cat.eq("SYN_NEG"), SYN_TERM.pattern)
add("F synov- term but rule silent", cat.eq("SYN_OTHER"), SYN_TERM.pattern)
add("G v3.0 BORD -> D2 POS (step c2 / D2)", rs0[SY].eq("BORD") & rs[SY].eq("POS"), SYN_TERM.pattern)
add("H effusion-only, reader votes Synovitis POS", eff_only & pd.concat([Rd[r][SY].eq("POS") for r in Rd if r != "rules"], axis=1).any(axis=1), EFF_PAT)
add("I silent report, high Steven-v4 Synovitis", cat.eq("SILENT") & (stev[SY] >= 0.3), EFF_PAT)
EX = pd.DataFrame(ex); EX.to_csv(OUT / "t8_examples.csv", index=False)

# ---- table 9: Steven-v4 confidence mask (<= 0.3 negative, >= 0.7 positive, else excluded) by category, training rows.
# The fold-0 validation ruler is Steven-v4 with this mask; this shows on training rows which categories it would score.
s = stev.loc[tid, SY]
T9 = pd.crosstab(cat.loc[tid], pd.Series(np.select([s <= 0.3, s >= 0.7], ["negative", "positive"], "excluded"), index=tid),
                 rownames=["category"], colnames=["steven_v4_mask"]).reindex(CATS)
T9.to_csv(OUT / "t9_steven_v4_mask_by_category_train.csv")

# ---- table 10: words over-represented in reports where laymond votes Synovitis POS without any synov- term
laypos = Rd["laymond"][SY].eq("POS") & ~term & train
base_ = ~term & train & ~laypos
cnt_a, cnt_b = {}, {}
for mask_, cnt in ((laypos, cnt_a), (base_, cnt_b)):
    for txt in rep[mask_]:
        for w in set(re.findall(r"[a-zà-ÿğışöüç]{5,}", txt.lower())):
            cnt[w] = cnt.get(w, 0) + 1
na, nb = int(laypos.sum()), int(base_.sum())
T10 = pd.DataFrame([{"token": w, "reports_laymond_pos": c, "share_laymond_pos": c / na, "share_other_no_term": cnt_b.get(w, 0) / nb,
                     "enrichment": (c / na) / ((cnt_b.get(w, 0) + 1) / nb)} for w, c in cnt_a.items() if c >= 25]
                   ).sort_values("enrichment", ascending=False).head(25).round(3)
T10.to_csv(OUT / "t10_laymond_pos_without_synov_term_tokens.csv", index=False)

# ---- table 11: preview (counts only, no label file written) of the minimal Synovitis-only candidate:
# no synovitis text (categories EFF_ONLY, EFF_NEG, SILENT) -> constant = training mean of D2 over those cells.
cand = L["d2"][SY].copy()
const = float(L["d2"].loc[tid][notext.loc[tid]][SY].mean())
cand[notext] = round(const, 5)
dlt = (cand - L["d2"][SY])[train]
T11 = {"constant_training_mean_d2_no_synov_text": round(const, 4),
       "cells_changed_train": int((dlt.abs() > 1e-9).sum()), "cells_changed_all_4349": int(notext.sum()),
       "mean_abs_shift_changed_train": round(float(dlt[dlt.abs() > 1e-9].abs().mean()), 4),
       "cells_down_gt_0_2_train": int((dlt < -0.2).sum()), "cells_up_gt_0_1_train": int((dlt > 0.1).sum()),
       "train_label_mean_before_after": [round(float(L["d2"].loc[tid, SY].mean()), 4), round(float(cand[train].mean()), 4)],
       "rho_syn_eff_train_before_after": [round(spear(L["d2"].loc[tid, SY], L["d2"].loc[tid, EF]), 3),
                                          round(spear(cand[train], L["d2"].loc[tid, EF]), 3)],
       "explicit_positive_cells_unchanged": int((cat.loc[tid] == "SYN_POS").sum()),
       "other_11_targets": "unchanged by construction (only the Synovitis column would differ)"}

# ---- preview 2 (counts only): treat LLM-reader positive-direction Synovitis votes as silent when the report has no
# synov- term and the rule reader is silent; re-run the unchanged builder and keep only its Synovitis column.
notext2 = ~term & Rd["rules"][SY].isna()
R2 = {k: v.copy() for k, v in Rd.items()}
dropped = {}
for r in R2:
    if r == "rules":
        continue
    mk = notext2 & R2[r][SY].isin({"POS", "LIKELY", "BORD", "HEDGE", "BELOW"})
    dropped[r] = int((mk & train).sum()); R2[r].loc[mk, SY] = np.nan
L2, free2, _, _, _ = b.blend(R2, H, parent_voice, False, "train", train)
d2c = (L2[SY] - L["d2"][SY])[train]; ch2 = d2c.abs() > 1e-9; nt2 = notext2[train]
T11["preview_2_drop_ungrounded_reader_votes"] = {
    "votes_dropped_train": dropped, "cells_changed_train": int(ch2.sum()),
    "mean_shift_changed": round(float(d2c[ch2].mean()), 4), "cells_down_gt_0_1": int((d2c < -0.1).sum()),
    "no_text_cells_ge_0_5_before_after": [int((L["d2"].loc[tid, SY][nt2] >= 0.5).sum()), int((L2.loc[tid, SY][nt2] >= 0.5).sum())],
    "rho_syn_eff_no_text_before_after": [round(spear(L["d2"].loc[tid, SY][nt2], L["d2"].loc[tid, EF][nt2]), 3),
                                         round(spear(L2.loc[tid, SY][nt2], L["d2"].loc[tid, EF][nt2]), 3)],
    "rho_syn_eff_all_train_before_after": [round(spear(L["d2"].loc[tid, SY], L["d2"].loc[tid, EF]), 3),
                                           round(spear(L2.loc[tid, SY], L["d2"].loc[tid, EF]), 3)]}

meta = {"split": split_prov, "train_studies": int(train.sum()), "files_sha256": {k: sha(R / p) for k, (p, _) in FILES.items()},
        "hpo_v1_reconstruction_from_parents_max_abs_error": hpo_recon_max_err,
        "rebuild_checks": {"c0_equals_v3_0": True, "c6_with_d2_rules_equals_d2_file": True},
        "category_counts_train": cat[train].value_counts().to_dict(), "cooccurrence": cooc,
        "candidate_preview_no_label_file_written": T11,
        "inputs_sha256": {"train_csv": sha(R / "reports_raw/train.csv"),
                          "d2_rules_s0": sha(D2DIR / "rules/s0_v30/rule_states.csv"),
                          "d2_rules_s5": sha(D2DIR / "rules/s5_fx_site/rule_states.csv"),
                          "steven_v4_parquet": sha(E / "labels_v0raw.parquet"), "gemini_low_parquet": sha(E / "labels_v1_gemlow.parquet"),
                          "builder": sha(R / "label_rules/v3_1/build_hpo_v3_1.py")}}
(OUT / "synovitis_supervision_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
for name, tab in (("T1", T1), ("T2", T2), ("T3", T3), ("T4", T4), ("T5", T5), ("T6", T6), ("T6b", T6b), ("T7", T7),
                  ("T9", T9), ("T10", T10)):
    print(f"==== {name}"); print(tab.to_string())
print(json.dumps({k: meta[k] for k in ("hpo_v1_reconstruction_from_parents_max_abs_error", "category_counts_train", "cooccurrence",
                                       "candidate_preview_no_label_file_written")},
                 indent=1, default=str))
