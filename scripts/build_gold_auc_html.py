"""Build a self-contained HTML report of the gold AUC analysis (no network, no external assets, no images of data).

Reads only the saved audit outputs (and train.csv for the 58 gold labels, for the Synovitis/Effusion cross-tab) and
writes one HTML file with inline SVG charts. Nothing is recomputed except that cross-tab and two within-subset AUCs,
which reproduce audit_gold_synovitis_effusion.py.

Container:
  podman run --rm --network=none -v "$P":/proj:ro,Z -v "$OUT":/out:rw,Z localhost/rsna-label-ds \
    python -I /proj/repository/scripts/build_gold_auc_html.py /proj /out
"""
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
A = R / "repository/data/image_detail_audit_2026-10-07"
G = A / "gold_image_model"
auc_t = pd.read_csv(G / "gold_auc_image_vs_report.csv", index_col=0)
boot = pd.read_csv(G / "gold_model_vs_report_bootstrap.csv", index_col=0)
rec = pd.read_csv(G / "gold_model_recovery_of_report_errors.csv", index_col=0)
meta = json.loads((G / "gold_image_model_meta.json").read_text())
readers = pd.read_csv(A / "gold_auc_by_reader.csv", index_col=0)
P = json.loads((G / "gold_image_model_predictions.json").read_text())
T = P["targets_in_order"]; ids = P["gold_study_ids"]
tr = pd.read_csv(R / "reports_raw/train.csv").set_index("StudyInstanceUID")
Y = tr.loc[ids, T].astype(int)
PRIMARY = "r50_hpo_v31_d2_fold0"
MODELS = [k for k in auc_t.columns if k.startswith(("r50_", "r34_"))]
NICE = {"r50_hpo_v31_d2_fold0": "ResNet50 · v3.1-D2 (0.925 public)", "r50_hpo_v31_trainval": "ResNet50 · v3.1 train+val",
        "r50_hpo_v2_fold0": "ResNet50 · HPO-v2", "r50_hpo_blend_fold0": "ResNet50 · HPO-v1",
        "r34_plateau60_fold0": "ResNet34 · plateau60", "r34_topk_v1_fold0": "ResNet34 · top-k v1",
        "r34_pairwise_fold0": "ResNet34 · pairwise AUC", "r34_hpo_v1_auc_control_fold0": "ResNet34 · HPO-v1 AUC control"}


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


Mp = {k: pd.DataFrame(v["gold_probabilities"], index=ids, columns=T) for k, v in P["models"].items()}
syn = {"crosstab": pd.crosstab(Y.Synovitis, Y.Effusion), "rows": []}
for k in (PRIMARY, "r50_hpo_blend_fold0"):
    e = Y.Effusion == 1
    syn["rows"].append((NICE[k], auc(Mp[k].Synovitis, Y.Synovitis), auc(Mp[k].Effusion, Y.Synovitis),
                        auc(Mp[k].Synovitis[e], Y.Synovitis[e])))
esc = html.escape

# ---------------------------------------------------------------- SVG helpers
W, LW, RP = 760, 150, 24  # width, label column, right padding


def xmap(v, lo, hi):
    return LW + (v - lo) / (hi - lo) * (W - LW - RP)


def axis(lo, hi, ticks, y0, y1, fmt="{:.2f}", zero=None):
    s = []
    for t in ticks:
        x = xmap(t, lo, hi)
        s.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="grid"/>'
                 f'<text x="{x:.1f}" y="{y1 + 16}" class="tick" text-anchor="middle">{fmt.format(t)}</text>')
    if zero is not None:
        x = xmap(zero, lo, hi); s.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="zero"/>')
    return "".join(s)


def dot_chart(rows, lo=0.6, hi=1.0):
    """rows: target names; draws report, primary, ensemble markers plus the range of the other 7 models."""
    rh, top = 30, 10
    h = top + rh * len(rows) + 30
    s = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Per-target AUC on gold">', axis(lo, hi, np.arange(lo, hi + 1e-9, 0.1), top, top + rh * len(rows))]
    for i, t in enumerate(rows):
        y = top + rh * i + rh / 2
        vals = [auc_t.at[t, m] for m in MODELS]
        s.append(f'<text x="{LW - 10}" y="{y + 4}" class="lab" text-anchor="end">{esc(t)}</text>')
        s.append(f'<line x1="{xmap(min(vals), lo, hi):.1f}" y1="{y}" x2="{xmap(max(vals), lo, hi):.1f}" y2="{y}" class="range"><title>All 8 models: {min(vals):.3f}–{max(vals):.3f}</title></line>')
        r, p, e = auc_t.at[t, "report_blend"], auc_t.at[t, PRIMARY], auc_t.at[t, "ens_r50x3_rank"]
        s.append(f'<circle cx="{xmap(r, lo, hi):.1f}" cy="{y}" r="6" class="m-rep"><title>Report blend {r:.3f}</title></circle>')
        s.append(f'<rect x="{xmap(e, lo, hi) - 5:.1f}" y="{y - 5}" width="10" height="10" class="m-ens"><title>3×ResNet50 ensemble {e:.3f}</title></rect>')
        s.append(f'<path d="M{xmap(p, lo, hi):.1f},{y - 7} l7,7 l-7,7 l-7,-7z" class="m-pri"><title>v3.1-D2 ResNet50 {p:.3f}</title></path>')
    s.append("</svg>")
    return "".join(s)


def forest(df, col, lo_c, hi_c, lo=-0.3, hi=0.3):
    rows = list(df.index)
    rh, top = 28, 10
    h = top + rh * len(rows) + 30
    s = [f'<svg viewBox="0 0 {W} {h}" role="img">', axis(lo, hi, np.arange(lo, hi + 1e-9, 0.1), top, top + rh * len(rows), "{:+.1f}", zero=0)]
    for i, t in enumerate(rows):
        y = top + rh * i + rh / 2
        v, a, b = df.at[t, col], df.at[t, lo_c], df.at[t, hi_c]
        sig = "sig" if (a > 0 or b < 0) else "ns"
        bold = ' font-weight="700"' if t == "MACRO" else ""
        s.append(f'<text x="{LW - 10}" y="{y + 4}" class="lab" text-anchor="end"{bold}>{esc("Macro" if t == "MACRO" else t)}</text>')
        s.append(f'<line x1="{xmap(max(a, lo), lo, hi):.1f}" y1="{y}" x2="{xmap(min(b, hi), lo, hi):.1f}" y2="{y}" class="ci {sig}"/>')
        s.append(f'<circle cx="{xmap(v, lo, hi):.1f}" cy="{y}" r="5.5" class="pt {sig}"><title>{t}: {v:+.3f} [{a:+.3f}, {b:+.3f}]</title></circle>')
        s.append(f'<text x="{W - 4}" y="{y + 4}" class="val" text-anchor="end">{v:+.3f}</text>')
    s.append("</svg>")
    return "".join(s)


def recovery_chart():
    rows = [t for t in T if rec.at[t, "report_overcalled_neg"] > 0]
    lo, hi, rh, top = 0.2, 1.0, 30, 10
    h = top + rh * len(rows) + 30
    s = [f'<svg viewBox="0 0 {W} {h}" role="img">', axis(lo, hi, np.arange(lo, hi + 1e-9, 0.2), top, top + rh * len(rows), zero=0.5)]
    for i, t in enumerate(rows):
        y = top + rh * i + rh / 2
        r, m = rec.at[t, "report_all_pos_vs_overcalled_neg_auc"], rec.at[t, "model_auc_pos_vs_overcalled_neg"]
        s.append(f'<text x="{LW - 10}" y="{y + 4}" class="lab" text-anchor="end">{esc(t)} <tspan class="muted">({int(rec.at[t, "report_overcalled_neg"])})</tspan></text>')
        s.append(f'<line x1="{xmap(r, lo, hi):.1f}" y1="{y}" x2="{xmap(m, lo, hi):.1f}" y2="{y}" class="link"/>')
        s.append(f'<circle cx="{xmap(r, lo, hi):.1f}" cy="{y}" r="6" class="m-rep"><title>Reports {r:.3f}</title></circle>')
        s.append(f'<path d="M{xmap(m, lo, hi):.1f},{y - 7} l7,7 l-7,7 l-7,-7z" class="m-pri"><title>Model {m:.3f}</title></path>')
    s.append("</svg>")
    return "".join(s)


def heat(v, lo=0.6, hi=1.0):
    if pd.isna(v):
        return ""
    a = min(max((v - lo) / (hi - lo), 0), 1)
    return f' style="background:color-mix(in oklab, var(--heat) {a * 55:.0f}%, transparent)"'


# ---------------------------------------------------------------- tables
cols = [("report_blend", "Report blend"), ("rules_v31", "Rules v3.1"), (PRIMARY, "r50 v3.1-D2"),
        ("r50_hpo_v31_trainval", "r50 v3.1 train+val"), ("r50_hpo_v2_fold0", "r50 HPO-v2"), ("r50_hpo_blend_fold0", "r50 HPO-v1"),
        ("r34_plateau60_fold0", "r34 plateau60"), ("r34_topk_v1_fold0", "r34 top-k v1"), ("r34_pairwise_fold0", "r34 pairwise"),
        ("r34_hpo_v1_auc_control_fold0", "r34 HPO-v1 AUC ctrl"), ("ens_r50x3_rank", "3×r50 ens"),
        ("primary_plus_report_rank", "D2 + report*"), ("primary_fold0_vs_stevenv4", "D2 fold-0 vs Steven-v4")]
GROUP = {"report_blend": "g-rep", "rules_v31": "g-rep", "ens_r50x3_rank": "g-ens", "primary_plus_report_rank": "g-ens",
         "primary_fold0_vs_stevenv4": "g-ref"}
th = "".join(f'<th class="{GROUP.get(c, "g-mod")}">{esc(n)}</th>' for c, n in cols)
body = []
for t in T + ["MACRO"]:
    gp = "" if t == "MACRO" else int(auc_t.at[t, "gold_pos"])
    best = max(MODELS, key=lambda m: auc_t.at[t, m])
    cells = "".join(f'<td{heat(auc_t.at[t, c])}{" class=best" if c == best else ""}>{auc_t.at[t, c]:.4f}</td>' for c, _ in cols)
    body.append(f'<tr{" class=macro" if t == "MACRO" else ""}><th scope="row">{"Macro" if t == "MACRO" else esc(t)}</th><td>{gp}</td>{cells}</tr>')
main_table = f'<table><thead><tr><th>Target</th><th>Gold +</th>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'

rd_cols = [("rules_s0_v30", "Rules v3.0"), ("rules_s5_v31", "Rules v3.1"), ("teammate", "teammate"), ("pilkwang", "pilkwang"),
           ("laymond", "laymond"), ("blend_v30rules", "Blend (v3.0 rules)"), ("blend_v31rules", "Blend (v3.1 rules)")]
rbody = []
for t in T + ["MACRO"]:
    cells = "".join(f"<td{heat(readers.at[t, c])}>{readers.at[t, c]:.3f}</td>" for c, _ in rd_cols)
    rbody.append(f'<tr{" class=macro" if t == "MACRO" else ""}><th scope="row">{"Macro" if t == "MACRO" else esc(t)}</th>{cells}</tr>')
reader_table = f'<table><thead><tr><th>Target</th>{"".join(f"<th>{esc(n)}</th>" for _, n in rd_cols)}</tr></thead><tbody>{"".join(rbody)}</tbody></table>'

fid = "".join(f'<tr><th scope="row">{esc(NICE[k])}</th><td>{v["epoch"]}</td><td><code>{v["sha256"][:8]}</code></td>'
              + (f'<td>{meta["fidelity_32_fold0"][k]["mean_spearman_over_targets"]:.4f}</td><td>{meta["fidelity_32_fold0"][k]["max_abs_diff"]:.5f}</td>'
                 if meta["fidelity_32_fold0"].get(k) else '<td colspan="2" class="muted">no saved fold-0 predictions</td>')
              + f'<td>{auc_t.at["MACRO", k]:.3f}</td></tr>' for k, v in meta["checkpoints"].items())
fid_table = f'<table><thead><tr><th>Checkpoint</th><th>Epoch</th><th>sha256</th><th>Fidelity ρ</th><th>Max |Δp|</th><th>Gold macro AUC</th></tr></thead><tbody>{fid}</tbody></table>'

ct = syn["crosstab"]
syn_ct = (f'<table class="small"><thead><tr><th></th><th>Effusion −</th><th>Effusion +</th></tr></thead><tbody>'
          f'<tr><th scope="row">Synovitis −</th><td>{ct.at[0, 0]}</td><td>{ct.at[0, 1]}</td></tr>'
          f'<tr><th scope="row">Synovitis +</th><td>{ct.at[1, 0]}</td><td><b>{ct.at[1, 1]}</b></td></tr></tbody></table>')
syn_rows = "".join(f"<tr><th scope='row'>{esc(n)}</th><td>{a:.3f}</td><td>{b:.3f}</td><td><b>{c:.3f}</b></td></tr>" for n, a, b, c in syn["rows"])
syn_tab = (f'<table class="small"><thead><tr><th>Model</th><th>Synovitis head → gold Synovitis</th><th>Effusion head → gold Synovitis</th>'
           f'<th>Synovitis head within gold-Effusion+ (n={int(Y.Effusion.sum())})</th></tr></thead><tbody>{syn_rows}</tbody></table>')

order = T
bt = boot.loc[T + ["MACRO"]]
pool = meta["pooled_recovery"]
macro = auc_t.loc["MACRO"]

css = """
:root{--bg:#f7f6f2;--panel:#ffffff;--ink:#1d2329;--muted:#5d6670;--line:#d9dcd8;--grid:#e7e9e5;
--rep:#c4682b;--pri:#2a6fb0;--ens:#3f8f5a;--range:#9aa3ab;--sig:#2a6fb0;--ns:#9aa3ab;--heat:#3f8f5a;--accent:#2a6fb0;--warn:#a5521c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15181b;--panel:#1d2125;--ink:#e6e8e6;--muted:#9ba3aa;
--line:#2f353a;--grid:#272c31;--rep:#e08a4f;--pri:#6aa8e3;--ens:#6cc08a;--range:#5d666e;--sig:#6aa8e3;--ns:#5d666e;--heat:#6cc08a;--accent:#6aa8e3;--warn:#e08a4f}}
:root[data-theme="dark"]{--bg:#15181b;--panel:#1d2125;--ink:#e6e8e6;--muted:#9ba3aa;--line:#2f353a;--grid:#272c31;--rep:#e08a4f;
--pri:#6aa8e3;--ens:#6cc08a;--range:#5d666e;--sig:#6aa8e3;--ns:#5d666e;--heat:#6cc08a;--accent:#6aa8e3;--warn:#e08a4f}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.9rem;line-height:1.2;margin:0 0 6px}h2{font-size:1.3rem;margin:44px 0 10px;padding-top:6px;border-top:1px solid var(--line)}
h3{font-size:1.02rem;margin:22px 0 6px}p{margin:8px 0}.muted{color:var(--muted);fill:var(--muted)}
.sub{color:var(--muted);margin:0 0 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.card .v{font-size:1.6rem;font-weight:650;margin-top:2px}
.card .n{font-size:.85rem;color:var(--muted)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.take{border-left:4px solid var(--accent)}.caveat{border-left:4px solid var(--warn)}
.panel ul{margin:6px 0;padding-left:20px}.panel li{margin:3px 0}
svg{width:100%;height:auto;display:block}svg text{font:12.5px system-ui,sans-serif;fill:var(--ink)}
.tick{fill:var(--muted)!important;font-size:11.5px!important}.val{fill:var(--muted)!important;font-variant-numeric:tabular-nums}
.grid{stroke:var(--grid);stroke-width:1}.zero{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:3 3}
.range{stroke:var(--range);stroke-width:5;stroke-linecap:round;opacity:.55}
.m-rep{fill:var(--rep)}.m-pri{fill:var(--pri)}.m-ens{fill:var(--ens)}.link{stroke:var(--range);stroke-width:2}
.ci{stroke-width:3;stroke-linecap:round}.ci.sig{stroke:var(--sig)}.ci.ns{stroke:var(--ns)}.pt.sig{fill:var(--sig)}.pt.ns{fill:var(--panel);stroke:var(--ns);stroke-width:2}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.88rem;color:var(--muted);margin:4px 0 8px}
.legend i{display:inline-block;width:11px;height:11px;margin-right:6px;vertical-align:-1px}
.lg-rep{background:var(--rep);border-radius:50%}.lg-pri{background:var(--pri);transform:rotate(45deg) scale(.85)}.lg-ens{background:var(--ens)}
.lg-range{background:var(--range);height:5px!important;width:18px!important;border-radius:3px;vertical-align:2px!important;opacity:.6}
.lg-sig{background:var(--sig);border-radius:50%}.lg-ns{border:2px solid var(--ns);border-radius:50%}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:8px 0}
table{border-collapse:collapse;font-size:.86rem;font-variant-numeric:tabular-nums;min-width:100%}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
thead th{color:var(--muted);font-weight:600;vertical-align:bottom;white-space:normal;min-width:64px}
tbody th{text-align:left;font-weight:500}tr.macro th,tr.macro td{font-weight:700;border-top:2px solid var(--line)}
table.small{min-width:0}code{font-size:.85em}
td.best{outline:2px solid var(--pri);outline-offset:-3px;font-weight:650}
thead th.g-rep{color:var(--rep)}thead th.g-ens{color:var(--ens)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}.two>*{min-width:0}@media (max-width:700px){.two{grid-template-columns:1fr}}
footer{margin-top:48px;color:var(--muted);font-size:.85rem}
"""

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold AUC Audit</title><style>{css}</style></head><body><main>
<h1>Gold AUC audit: image models vs report readers</h1>
<p class="sub">RSNA knee abnormality detection · 58 host-labelled gold studies · analysis of 2026-10-07 · evaluation only</p>

<div class="cards">
<div class="card"><div class="k">v3.1-D2 ResNet50 on gold</div><div class="v">{macro[PRIMARY]:.3f}</div><div class="n">macro AUC (0.925 public LB)</div></div>
<div class="card"><div class="k">Report-reader blend</div><div class="v">{macro['report_blend']:.3f}</div><div class="n">same 58 studies, same targets</div></div>
<div class="card"><div class="k">3×ResNet50 rank ensemble</div><div class="v">{macro['ens_r50x3_rank']:.3f}</div><div class="n">best macro on gold</div></div>
<div class="card"><div class="k">Model + report evidence*</div><div class="v">{bt.at['MACRO','combined_minus_model']:+.3f}</div><div class="n">[{bt.at['MACRO','cmm_ci_low']:+.3f}, {bt.at['MACRO','cmm_ci_high']:+.3f}] over model alone</div></div>
</div>

<div class="panel take"><b>Main reading</b><ul>
<li><b>Reports and images make complementary errors.</b> Combining them improves gold macro AUC, so the image model has not recovered all the information the reports express. This does not show that the labels are generally sufficient, or that one cause, such as resolution, explains the gap.</li>
<li><b>MCL, Lateral OA, PF OA:</b> the report readers outperform the model. These findings involve thin structures, so higher in-plane resolution is a <i>plausible hypothesis</i>, which the running 336 px experiment tests.</li>
<li><b>Effusion, Contusion, Baker's:</b> the model outperforms its report labels and down-ranks report over-calls. It generalises beyond some report errors, but that does not mean those errors no longer constrain it.</li>
<li><b>Synovitis:</b> the strongest candidate for label work. The v3.1-D2 Synovitis head behaves almost like Effusion on gold. A fluid shortcut, correlated disease, limited image detail and noisy supervision could each produce that.</li>
</ul></div>

<h2>1 · The full gold AUC table</h2>
<p>AUC on the 58 gold studies for each target (rows) and scorer (columns). Values are shown at the 4 decimals saved in <code>gold_auc_image_vs_report.csv</code>. Cells are shaded from 0.6 (no shading) up to 1.0 (darkest), and the best image model in each row is outlined.</p>
<ul class="muted" style="font-size:.9rem;margin-top:0">
<li><b>Report blend:</b> v3.1 rules plus 3 LLM readers, run on each gold study's own report.</li>
<li><b>Rules v3.1:</b> the rules reader alone.</li>
<li><b>r50 / r34:</b> ResNet50 / ResNet34 checkpoints, none trained on gold.</li>
<li><b>3×r50 ens:</b> rank-average of r50 v3.1-D2, r50 HPO-v1 and r50 v3.1 train+val.</li>
<li><b>D2 + report*:</b> rank-average of r50 v3.1-D2 and the report blend.</li>
<li><b>Last column:</b> the same v3.1-D2 model on its 825 fold-0 validation studies, scored against Steven-v4 report labels (the training-time reference), for comparison.</li>
</ul>
<div class="tw">{main_table}</div>
<p class="muted">*Report text is not available at test time: the public <code>test.csv</code> holds <code>StudyInstanceUID</code> only. This column only measures whether the two sources carry complementary information.</p>

<h2>2 · Per-target AUC at a glance</h2>
<p>The same numbers as a chart, one row per target. The grey bar spans all 8 checkpoints; the markers pick out the report blend, the v3.1-D2 ResNet50 that scored 0.925 public, and the rank ensemble of three ResNet50s. Hover a marker for its value.</p>
<div class="legend"><span><i class="lg-rep"></i>Report blend (v3.1 rules + 3 LLM readers)</span><span><i class="lg-pri"></i>v3.1-D2 ResNet50</span><span><i class="lg-ens"></i>3×ResNet50 ensemble</span><span><i class="lg-range"></i>range of 8 models</span></div>
<div class="panel">{dot_chart(order)}</div>

<h2>3 · Is the model better or worse than the reports?</h2>
<p>Primary model AUC minus report-blend AUC, with 95% intervals from a 2,000-draw paired bootstrap over studies (seed 0). A filled marker means the interval excludes zero.</p>
<div class="legend"><span><i class="lg-sig"></i>interval excludes 0</span><span><i class="lg-ns"></i>interval includes 0</span></div>
<div class="panel">{forest(bt, "model_minus_report", "mmr_ci_low", "mmr_ci_high")}</div>
<p>The model is clearly ahead on <b>Contusion</b> and <b>Baker's</b>, and nearly so on <b>Effusion</b>. It trails on <b>MCL</b>, <b>Lateral OA</b>, <b>PF OA</b> and <b>Synovitis</b>, but every one of those intervals includes zero.</p>

<h2>4 · Does report evidence add to the model?</h2>
<p>AUC of the rank-average of (primary model, report blend) minus the model alone. A positive value means the reports hold information the image model has not learned.</p>
<div class="panel">{forest(bt, "combined_minus_model", "cmm_ci_low", "cmm_ci_high", lo=-0.15, hi=0.2)}</div>
<p>The gain is significant for <b>MCL</b> (+{bt.at['MCL','combined_minus_model']:.3f}), <b>Synovitis</b> (+{bt.at['Synovitis','combined_minus_model']:.3f}, lower bound 0.000) and macro. For Effusion, Contusion and Baker's, adding the reports does not help.</p>

<h2>5 · Do the models repair report errors?</h2>
<div class="two">
<div><h3>Over-called negatives</h3><p>Gold negatives the reports score ≥ 0.5 (count in brackets), ranked against all gold positives. Higher is better.</p>
<div class="legend"><span><i class="lg-rep"></i>reports</span><span><i class="lg-pri"></i>v3.1-D2 ResNet50</span></div></div>
<div><h3>Pooled across targets</h3><div class="cards" style="grid-template-columns:1fr 1fr">
<div class="card"><div class="k">Over-calls repaired</div><div class="v">{pool['model_auc_pos_vs_overcalled_neg_pooled']:.2f}</div><div class="n">model AUC, positives vs over-called negatives</div></div>
<div class="card"><div class="k">Missed positives found</div><div class="v">{pool['model_auc_missed_pos_vs_neg_pooled']:.2f}</div><div class="n">model AUC, report-missed positives vs negatives</div></div></div>
<p class="muted">Missed positives are few per target (Synovitis 10, PF OA 4, Fracture 4, otherwise ≤ 3), so per-target values are not charted.</p></div>
</div>
<div class="panel">{recovery_chart()}</div>
<p>The model separates report over-calls far better than the reports do for Effusion, Contusion, Baker's, ACL and Medial OA. It is weaker on MCL and PF OA. It finds report-missed positives only modestly (pooled 0.66).</p>

<h2>6 · Synovitis behaves like Effusion</h2>
<div class="two">
<div class="panel"><h3 style="margin-top:0">Gold labels, 58 studies</h3><div class="tw">{syn_ct}</div>
<p class="muted">22 of 27 gold synovitis cases also have effusion. Gold prevalence: Synovitis 47%, Effusion 60%.</p></div>
<div class="panel"><h3 style="margin-top:0">What each head ranks</h3><div class="tw">{syn_tab}</div>
<p class="muted">The v3.1-D2 Synovitis head ranks gold synovitis no better than its own Effusion head does, and barely separates synovitis inside effusion-positive knees.</p></div>
</div>
<div class="panel caveat"><b>What this does and doesn't show.</b> This behaviour is consistent with a fluid shortcut. It is equally consistent with correlated disease, limited image detail or noisy report supervision, so it does not prove a label-definition mismatch.
The follow-up label trace (<code>docs/synovitis_supervision_audit.md</code>) found that Effusion-to-Synovitis copying is strongest in HPO-v1's own labels, not introduced by the v3.1 changes. On fold 0, the 0.951 Synovitis score against Steven-v4 can't detect such copying, because that reference's confidence mask leaves out effusion-only studies.</div>

<h2>7 · Report readers alone vs gold</h2>
<p>Each reader run on the gold studies' own reports. A silent reader scores 0.04. HPO-v1 has no gold rows, so it isn't included. Moving from v3.0 to v3.1 rules changes the blend's macro score by {(readers.at['MACRO','blend_v31rules'] - readers.at['MACRO','blend_v30rules']):+.4f} (95% interval −0.0058 to +0.0002).</p>
<div class="tw">{reader_table}</div>

<h2>8 · Method and fidelity</h2>
<div class="panel"><ul>
<li><b>Gold studies.</b> The 58 studies with host labels in <code>train.csv</code>, two MSK radiologists plus an adjudicator. Their slice cache uses the training cache settings exactly (140 mm / 224 px, 4 slots × 12 slices).</li>
<li><b>Checkpoints.</b> 8 checkpoints, none trained on gold: no gold study is in any training cache or fold. Gold-tuned runs were excluded.</li>
<li><b>Inference.</b> Run on a local GPU inside a container with no network, using the training script's own model and batch code unchanged, with every cached window per study.</li>
<li><b>Fidelity check.</b> Before scoring gold, 32 fold-0 studies were re-scored and compared with each run's saved predictions.</li>
<li><b>Bootstrap.</b> 2,000 paired draws over studies, seed 0. Macro is the mean of the 12 per-target AUCs.</li>
</ul><div class="tw">{fid_table}</div></div>

<div class="panel caveat"><b>Caveats</b><ul>
<li><b>Small sample.</b> 58 studies with 9–35 positives per target, so intervals are wide. Read the direction, not the size.</li>
<li><b>Gold is now development evidence.</b> These studies have been inspected per target repeatedly; they are not an independent final evaluation. Any label change motivated here should be justified from report evidence and the host's definitions, and judged on public LB.</li>
<li><b>Gold vs leaderboard.</b> Gold macro ({macro[PRIMARY]:.3f}) is below the same model's public LB (0.925). Gold is a different, enriched sample, not an estimate of LB.</li>
<li><b>Single seeds.</b> Each checkpoint is one seed, so model-to-model differences include run variance.</li>
</ul></div>

<footer>Sources: <code>repository/data/image_detail_audit_2026-10-07/</code> (<code>gold_image_model/</code> and <code>gold_auc_by_reader.csv</code>), produced by
<code>scripts/gold_image_model_predictions.py</code>, <code>scripts/audit_gold_image_model.py</code>, <code>scripts/audit_gold_report_ceiling.py</code> and <code>scripts/audit_gold_synovitis_effusion.py</code>.
Page built by <code>scripts/build_gold_auc_html.py</code>. Predictions sha256 <code>{meta['predictions_sha256'][:12]}…</code>.</footer>
</main></body></html>"""
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "gold_auc_audit.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "gold_auc_audit.html", len(page), "bytes")
