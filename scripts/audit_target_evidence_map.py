"""LOC-1 (ledger, pre-registered 2026-10-17): per-target evidence map for crop design. CPU, read-only, descriptive.

(A) Report sites: in report-present studies (teammate structured reader), the sentences that mention the target; share of
    sentences with each site / descriptor term (local regex counts only). (B) Own signal vs co-finding: u = the other
    target whose E3 output best ranks this target (label-v2 confident cells); within u-positive studies, AUC of the
    target's own output vs u's output (E3, label-v2 r50).
Usage: OMP_NUM_THREADS=1 python3 -I scripts/audit_target_evidence_map.py <project root>
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1]); D = R / "repository/data"; C = D / "coatnet_d96_fold0"
src = (R / "repository/scripts/audit_phase1.py").read_text()
sys.argv = [sys.argv[0], str(R)]
exec(src.split("def per_target")[0])                       # ids, T, load, auc
OUT = D / "attention_side_audit/target_evidence_map.txt"; lines = []
say = lambda s="": (print(s), lines.append(s))
OA = r"(chondr|cartilag|osteophyt|joint space|arthro|degenerat|osteoarth)"
SENT = {"ACL": r"\bacl\b|anterior cruciate", "MCL": r"\bmcl\b|medial collateral|tibial collateral",
        "Medial Meniscus": r"medial menisc", "Lateral Meniscus": r"lateral menisc",
        "Medial OA": rf"(medial (compartment|femoral|tibial|joint|femorotibial|tibiofemoral)).*{OA}|{OA}.*medial (compartment|femoral|tibial|femorotibial|tibiofemoral)",
        "Lateral OA": rf"(lateral (compartment|femoral|tibial|joint|femorotibial|tibiofemoral)).*{OA}|{OA}.*lateral (compartment|femoral|tibial|femorotibial|tibiofemoral)",
        "PF OA": rf"(patell|trochle).*{OA}|{OA}.*(patell|trochle)|chondromalacia",
        "Effusion": r"effusion", "Synovitis": r"synovitis|synovial (thicken|prolif|hypertroph|debris|frond)",
        "Baker's": r"baker|popliteal cyst", "Contusion": r"contusion|bone bruise|marrow (edema|oedema)|bone (marrow )?(edema|oedema)",
        "Fracture": r"fracture"}
LIG = {"proximal / femoral": r"proximal|femoral (attachment|insertion|origin|side)", "mid-substance": r"mid[- ]?(substance|portion|fibers|body)|midsubstance",
       "distal / tibial": r"distal|tibial (attachment|insertion|side)", "partial": r"partial|low[- ]grade|high[- ]grade", "complete / full": r"complete|full[- ]thickness|rupture|disrupt",
       "sprain / periligamentous edema": r"sprain|periligament|edema|oedema", "mucoid / degenerative": r"mucoid|degenerat", "graft / reconstruction": r"graft|reconstruct",
       "thickening (chronic)": r"thicken"}
MEN = {"posterior horn": r"posterior horn", "body": r"\bbody\b", "anterior horn": r"anterior horn", "root": r"\broot", "free edge": r"free edge",
       "bucket / flap / displaced": r"bucket|flap|displac|parrot", "radial": r"radial", "horizontal": r"horizontal", "oblique": r"oblique",
       "complex": r"complex", "vertical / longitudinal": r"vertical|longitudinal", "extrusion": r"extru", "degenerative / intrasubstance": r"degenerat|intrasubstance|myxoid",
       "undersurface / articular surface": r"undersurface|inferior (articular )?surface|articular surface"}
OAS = {"femoral condyle": r"femoral condyl", "tibial plateau": r"tibial plateau", "joint space narrowing": r"joint space", "osteophyte": r"osteophyt",
       "cartilage loss / thinning": r"thinn|loss|defect|denud", "fissuring": r"fissur", "full thickness": r"full[- ]thickness", "subchondral edema": r"subchondral (bone )?(marrow )?(edema|oedema)",
       "subchondral cyst": r"subchondral cyst|cystic", "weight-bearing / central": r"weight[- ]?bearing|central", "posterior": r"posterior", "meniscal extrusion": r"extru"}
PF = {"lateral facet": r"lateral (patellar )?facet", "medial facet": r"medial (patellar )?facet", "median ridge / apex": r"ridge|apex", "trochlea": r"trochle",
      "osteophyte": r"osteophyt", "chondromalacia": r"chondromalacia", "full thickness": r"full[- ]thickness", "subchondral": r"subchondral", "fissuring": r"fissur",
      "lateral tilt / subluxation": r"tilt|sublux|maltrack"}
EFF = {"small": r"\bsmall|minimal|trace", "moderate": r"moderate", "large": r"large|marked|massive", "suprapatellar": r"suprapatell",
       "debris / hemarthrosis": r"debris|hemarthros|haemarthros|lipohem|fat[- ]fluid", "synovitis in same sentence": r"synovi"}
SYN = {"effusion in same sentence": r"effusion", "suprapatellar": r"suprapatell", "posterior / popliteal": r"posterior|poplit|gastroc|semimembran",
       "Hoffa / infrapatellar": r"hoffa|infrapatell|fat pad", "debris / fronds": r"debris|frond|prolif|villous|nodular", "thickening": r"thicken"}
BAK = {"medial / gastrocnemius-semimembranosus": r"medial|gastroc|semimembran", "ruptured / leaking": r"ruptur|leak|dissect", "septated / debris / loose body": r"septat|debris|loose|bod",
       "small": r"\bsmall|minimal|tiny", "moderate": r"moderate", "large": r"large", "size in cm / mm": r"\d+(\.\d+)?\s?(cm|mm)"}
CON = {"lateral femoral condyle": r"lateral femoral condyl", "medial femoral condyle": r"medial femoral condyl", "lateral tibial plateau": r"lateral tibial plateau",
       "medial tibial plateau": r"medial tibial plateau", "posterior / posterolateral plateau": r"poster\w* (lateral )?tibial|posterolateral|posteromedial",
       "patella": r"patell", "fibula": r"fibul", "kissing / pivot-shift": r"kissing|pivot", "subchondral": r"subchondral", "degenerative / reactive": r"degenerat|reactive|insufficien|stress"}
FRA = {"tibial plateau": r"tibial plateau", "femoral condyle": r"femoral condyl", "patella": r"patell", "fibula": r"fibul", "Segond / avulsion": r"segond|avulsion",
       "subchondral insufficiency / stress": r"insufficien|stress|subchondral", "osteochondral": r"osteochondral", "impaction / depression": r"impact|depress",
       "nondisplaced / occult / hairline": r"non[- ]?displac|undisplac|occult|hairline|trabecular|microfract", "old / healed": r"old|heal|remote|chronic"}
SITES = {"ACL": LIG, "MCL": LIG, "Medial Meniscus": MEN, "Lateral Meniscus": MEN, "Medial OA": OAS, "Lateral OA": OAS, "PF OA": PF,
         "Effusion": EFF, "Synovitis": SYN, "Baker's": BAK, "Contusion": CON, "Fracture": FRA}

rep = {r["StudyInstanceUID"]: r["Report"] for r in csv.DictReader((R / "reports_raw/train.csv").open(encoding="utf-8"))}
st = pd.read_csv(R / "colab_bundle/raw/report_labels/report_label_extraction/report_structured_labels.csv",
                 dtype=str).fillna("").set_index("StudyInstanceUID")
key = lambda t: t.lower().replace("'", "").replace(" ", "_")
split = lambda txt: [s.strip().lower() for s in re.split(r"(?<=[.;])\s+|\n", txt) if s.strip()]
say("(A) REPORT SITES: share of the target's sentences (report-present studies) containing each term\n")
for t in T:
    pres = [s for s in st.index if st.at[s, f"{key(t)}__status"] == "present" and s in rep]
    sents = [x for s in pres for x in split(rep[s]) if re.search(SENT[t], x)]
    if not sents:
        say(f"{t}: no sentences"); continue
    cnt = {k: np.mean([bool(re.search(p, x)) for x in sents]) for k, p in SITES[t].items()}
    say(f"{t}: {len(pres)} present studies, {len(sents)} sentences")
    say("   " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])))

lv = json.loads((D / "teammate_labelv2/soft_labels_final_per_target_latD2.json").read_text()); mm = {r["study_id"]: r for r in lv["labels"]}
yv = np.array([[.5 if p is None else p for p in mm[s]["probabilities"]] for s in ids]); wv = np.array([mm[s]["weights"] for s in ids])
mv = (wv > 0) & ((yv <= .3) | (yv >= .7)); tv = yv >= .7
TM = D / "teammate_pp4/fold0_predictions"
L2 = load(C / "kaggle_fold0_labelv2/val_swa.json")
E3 = np.mean([L2, load(TM / "four_network_fold0_val_swa.json"), load(C / "run_continue/val_swa.json"), load(C / "run_cnx_pertarget/val_swa.json")], 0)
say("\n(B) OWN SIGNAL vs CO-FINDING (fold 0, label-v2 confident cells)")
say(f"{'target':17s} {'own AUC':>7s}  best other output (AUC)        within that finding's positives: n+/n-   own E3 / r50   other E3 / r50")
for k, t in enumerate(T):
    sel = mv[:, k]; own = auc(E3[sel, k], tv[sel, k])
    oth = sorted(((auc(E3[sel, u], tv[sel, k]), u) for u in range(12) if u != k), reverse=True)[0]
    u = oth[1]; g = sel & tv[:, u] & mv[:, u]; n1, n0 = (g & tv[:, k]).sum(), (g & ~tv[:, k]).sum()
    if n1 >= 5 and n0 >= 5:
        r = (f"{n1:3d}/{n0:<3d}     {auc(E3[g, k], tv[g, k]):.3f} / {auc(L2[g, k], tv[g, k]):.3f}   "
             f"{auc(E3[g, u], tv[g, k]):.3f} / {auc(L2[g, u], tv[g, k]):.3f}")
    else:
        r = f"{n1:3d}/{n0:<3d}     (too few)"
    say(f"{t:17s} {own:7.3f}  {T[u]:17s} ({oth[0]:.3f})       {r}")
OUT.write_text("\n".join(lines) + "\n")
