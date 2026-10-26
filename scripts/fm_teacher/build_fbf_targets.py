"""FB-F targets: E4's target recipe (full public blend ORDER, Raptor VALUE scale, 0.7 / 0.3 with label-v2 where label-v2
has weight > 0) over one pool = competition train studies + fastMRI exams, so both sources share one target scale.

Members per study: Raptor 384 / 224 (competition: data/coatnet_d96_fold0/pubkd; fastMRI: the MeluXina Job 1b teacher
jsons), R6 and the 2.5D reader (competition: Kaggle teacher shards r6/*_all.csv; fastMRI: the Colab FB-F chunks).
Pool = competition studies + fastMRI exams with all four members, minus the holdout (fm_holdout.json, never trained),
minus fm_exclude.json (312 exams whose Raptor values came from the mis-slotted old cache), minus the 4 archive duplicates.
G2 flags (fbf_port_gate.py, "member:finding" list): for a flagged (member, finding) the fastMRI exams take Raptor's
logit-mix probability in place of that member (Raptor only for that finding, as pre-registered).
Steps: assemble <out>/pool (member files in build_pubkd_targets.py's layout) -> build_pubkd_targets.py (full blend) and
(raptor) -> build_pubkd_swap_targets.py -> <out>/pool/pubkd_targets_fullorder_rapvalues_mix73.json (= E4's recipe).
QC: competition-only pool reproduces E4's targets (Spearman per finding vs the Kaggle-pool file, logged); per-source
target distribution. Holdout teacher values (logit blend, no ranks) -> <out>/fbf_holdout_teacher.json.
Usage: python3 -I build_fbf_targets.py <project root> <out dir> <chunk root> [<chunk root> ...] [--flag member:finding ...]
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
W_RAPTOR = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35}
W_READER = {"Baker's": 0.035, "Contusion": 0.03, "Medial OA": 0.025, "ACL": 0.025, "Medial Meniscus": 0.02, "MCL": 0.02,
            "Lateral Meniscus": 0.008, "Fracture": 0.008, "Lateral OA": 0.008, "PF OA": 0.008, "Effusion": 0.004,
            "Synovitis": 0.003}
A224 = 0.35
args = sys.argv[1:]
flags = [a.split(":", 1) for a in args[args.index("--flag") + 1:]] if "--flag" in args else []
args = args[:args.index("--flag")] if "--flag" in args else args
R, O, ROOTS = Path(args[0]), Path(args[1]), [Path(p) for p in args[2:]]
REPO = R / "repository"
CP = REPO / "data/coatnet_d96_fold0/pubkd"
FR = REPO / "data/meluxina_mirror/project_sumit/fastmri_teacher"
META = REPO / "data/meluxina_mirror/fastmri_d96_meta"
HOLD = REPO / "data/meluxina_mirror/project_sumit/fastmri_targets/fm_holdout.json"
SCRIPTS = REPO / "scripts"
L = []


def pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); L.append(s)


def raptor_json(d, model):
    out = {}
    for f in sorted(d.glob(f"raptor{model}_*.json")):
        j = json.loads(f.read_text()); assert j["targets"] == T
        for s, v in j["pred"].items():
            assert s not in out
            out[s] = v
    return out


def csvs(files):
    df = pd.concat([pd.read_csv(f, dtype={"StudyInstanceUID": str}) for f in files]).drop_duplicates("StudyInstanceUID")
    return df.set_index("StudyInstanceUID")[T].astype(float)


logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))

# ---- members
done = [c for r in ROOTS for c in sorted(r.glob("c[0-9][0-9][0-9]")) if (c / "DONE").exists()]
f6, frd = csvs([c / "r6_train_shard0.csv" for c in done]), csvs([c / "reader_train_shard0.csv" for c in done])
c6, crd = csvs([CP / "r6/r6_train_all.csv"]), csvs([CP / "r6/reader_train_all.csv"])
cr384, cr224 = raptor_json(CP, "384"), raptor_json(CP, "224")
fr384, fr224 = raptor_json(FR, "384"), raptor_json(FR, "224")
hold = set(json.loads(HOLD.read_text()))
excl = set(json.loads((META / "fm_exclude.json").read_text())["study_ids"])
dups = set(json.loads((META / "fm_dup_archive_ids.json").read_text())["study_ids"])
fm_all = sorted(set(f6.index) & set(frd.index) & set(fr384) & set(fr224))
fm_pool = [s for s in fm_all if s not in hold | excl | dups]
fm_hold = [s for s in fm_all if s in hold]
comp = sorted(cr384)
assert set(comp) <= set(c6.index) & set(crd.index) & set(cr224), "competition member missing"
assert not set(comp) & set(fm_all)
pr(f"chunks {len(done)}; fastMRI scored by R6+reader {len(set(f6.index) & set(frd.index))}, with Raptor {len(fm_all)}; "
   f"pool {len(fm_pool)} (holdout {len(fm_hold)}, excluded misslot {len(set(fm_all) & excl)}, archive dups {len(set(fm_all) & dups)}); "
   f"competition {len(comp)}")
pr(f"G2 flags: {flags or 'none'}")

# G2: flagged (member, finding) -> Raptor's logit-mix probability on fastMRI exams
for m, c in flags:
    j = T.index(c); df = f6 if m == "r6" else frd
    for s in set(df.index) & set(fr384):
        p384 = 0.5 * (fr384[s]["plain"][j] + fr384[s]["mirror"][j]); p224 = 0.5 * (fr224[s]["plain"][j] + fr224[s]["mirror"][j])
        df.loc[s, c] = sig((1 - A224) * logit(p384) + A224 * logit(p224))

# ---- assemble the pool dir in build_pubkd_targets.py's layout
P = O / "pool"; shutil.rmtree(P, ignore_errors=True); (P / "r6").mkdir(parents=True)
for model, cj, fj in (("384", cr384, fr384), ("224", cr224, fr224)):
    pred = {**{s: cj[s] for s in comp}, **{s: fj[s] for s in fm_pool}}
    (P / f"raptor{model}_0.json").write_text(json.dumps({"targets": T, "pred": pred}))
pd.concat([c6.loc[comp], f6.loc[fm_pool]]).rename_axis("StudyInstanceUID").to_csv(P / "r6/r6_train_all.csv")
pd.concat([crd.loc[comp], frd.loc[fm_pool]]).rename_axis("StudyInstanceUID").to_csv(P / "r6/reader_train_all.csv")
lv = REPO / "data/coatnet_d96_fold0/assets/soft_labels_final_per_target_latD2.json"
for cmd in ([str(SCRIPTS / "build_pubkd_targets.py"), str(R), str(P)],
            [str(SCRIPTS / "build_pubkd_targets.py"), str(R), str(P), "raptor"],
            [str(SCRIPTS / "build_pubkd_swap_targets.py"), str(P), str(lv)]):
    r = subprocess.run([sys.executable, "-I", *cmd], capture_output=True, text=True)
    L.append(r.stdout); print(r.stdout[-3000:], flush=True)
    assert r.returncode == 0, r.stderr[-3000:]

# ---- QC: competition rows vs E4's targets (Kaggle-only pool), per-source distribution
out = P / "pubkd_targets_fullorder_rapvalues_mix73.json"
new = {r["study_id"]: np.array(r["probabilities"]) for r in json.loads(out.read_text())["labels"]}
e4 = {r["study_id"]: np.array(r["probabilities"]) for r in json.loads((CP / out.name).read_text())["labels"]}
ids = sorted(set(new) & set(e4))
A, B = np.array([new[s] for s in ids]), np.array([e4[s] for s in ids])
rho = [pd.Series(A[:, j]).rank().corr(pd.Series(B[:, j]).rank()) for j in range(12)]
pr(f"QC competition rows vs E4 targets ({len(ids)} studies): Spearman min {min(rho):.4f} mean {np.mean(rho):.4f}; "
   f"mean |diff| {np.abs(A - B).mean():.4f}")
for name, ss in (("competition", comp), ("fastMRI", fm_pool)):
    X = np.array([new[s] for s in ss])
    pr(f"QC {name:11s} n {len(ss)} mean {X.mean():.3f} extreme cells (<0.05 or >0.95) {((X < .05) | (X > .95)).mean():.3f}")

# ---- holdout teacher values (logit blend under the blend's effective weights; no ranks)
hv = {}
for s in fm_hold:
    p = []
    for j, c in enumerate(T):
        w, v = W_RAPTOR.get(c, 0.5), W_READER[c]
        r = (1 - A224) * logit(0.5 * (fr384[s]["plain"][j] + fr384[s]["mirror"][j])) + A224 * logit(0.5 * (fr224[s]["plain"][j] + fr224[s]["mirror"][j]))
        p.append(float(sig((1 - v) * (w * r + (1 - w) * logit(f6.loc[s, c])) + v * logit(frd.loc[s, c]))))
    hv[s] = p
(O / "fbf_holdout_teacher.json").write_text(json.dumps({"targets": T, "pred": hv}))
pr(f"holdout teacher values: {len(hv)} exams")
pr(f"wrote {out} sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
(O / "build_fbf_targets.txt").write_text("\n".join(L) + "\n")
