"""Exp P1-tags asset (CPU, 2026-10-16): extra-finding tag targets for the auxiliary heads, training folds only.

The 6 lexicon tags kept by T-pre3 (bucket handle, chondromalacia, osteophyte, loose body, meniscal extrusion,
parameniscal cyst / ganglion), with T-pre3's lexicon and negation rule exactly (accents stripped, lower case; a match
is negated if a negation word occurs in the preceding 40 characters before the last . ; : or newline):
  1 = at least one affirmed match in the report, 0 = no affirmed match (no match, or negated matches only)
  weight 0 on all 6 = report language not in en / es / de / fr / nl (languages.json from T-pre), or no report
Only studies in folds.csv with fold != 0 are written; fold 0 (validation) and gold studies get no row, so the runner
gives them weight 0. The output holds the binary tags only, no report text. Positives must equal T-pre3's counts.

Usage (container):
  podman run --rm --network=none -v "$P":/proj localhost/rsna-label-ds python -I \
      /proj/repository/scripts/build_tag_aux_targets.py /proj
"""
import csv
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

R = Path(sys.argv[1]); A = R / "repository/data/coatnet_d96_fold0/assets"
PRE = R / "repository/data/text_alignment_prechecks"
TAG_LANGS = ("en", "es", "de", "fr", "nl")
TAGS = {
    "bucket_handle": r"bucket[- ]?handle|asa de cubo|asa de balde|korbhenkel|anse de seau|emmerhengsel",
    "chondromalacia": r"chondromala[cz]|condromala[cz]",
    "osteophyte": r"osteophyt|osteofit|osteofyt",
    "loose_body": r"loose bod|cuerpos? libres?|freie[nrs]? (gelenk)?korper|gelenkmaus|corps (etrangers?|libres?)"
                  r"|(losse|vrije) (corpus|corpora|lichaam|lichamen)|corp(us|ora) liber",
    "meniscal_extrusion": r"extrus|extrudi",
    "parameniscal_cyst_ganglion": r"paramenisc|paramenisk|ganglion|meniscal cyst|quiste meniscal",
}
NEG = re.compile(r"\b(no|not|without|negative for|free of|absence of|sin|ausencia de|ni|kein|keine|keinen|keiner"
                 r"|ohne|nicht|pas de|sans|aucun|aucune|geen|zonder|niet)\b")


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def affirmed(text, pat):
    for m in re.finditer(pat, text):
        pre = re.split(r"[.;:\n]", text[max(0, m.start() - 40):m.start()])[-1]
        if not NEG.search(pre):
            return 1.0
    return 0.0


fold = {r["study_id"]: r["fold"] for r in csv.DictReader((A / "folds.csv").open())}
train_ids = sorted(s for s, f in fold.items() if f != "0")
reports = {r["StudyInstanceUID"]: r["Report"] for r in csv.DictReader(open(R / "reports_raw/train.csv", encoding="utf-8"))}
LANG = json.loads((PRE / "languages.json").read_text())
ref = json.loads((PRE / "site_stripped/summary.json").read_text())

out, pos, n_obs = [], dict.fromkeys(TAGS, 0), 0
for s in train_ids:
    rep = reports.get(s, "")
    if LANG.get(s) in TAG_LANGS and rep.strip():
        t = norm(rep)
        vals = [affirmed(t, p) for p in TAGS.values()]
        n_obs += 1
        for k, v in zip(TAGS, vals):
            pos[k] += int(v)
        out.append({"study_id": s, "probabilities": vals, "weights": [1.0] * len(TAGS)})
    else:
        out.append({"study_id": s, "probabilities": [None] * len(TAGS), "weights": [0.0] * len(TAGS)})

print(f"training studies {len(out)}, tag-observed {n_obs}, masked {len(out) - n_obs}")
for k in TAGS:
    want = ref["A_tags"][k]["positives"]
    print(f"  {k:28s} positives {pos[k]:4d}  (T-pre3 {want})")
    assert pos[k] == want, f"{k}: {pos[k]} != T-pre3 {want}"
doc = {"targets_in_order": list(TAGS), "source": "report lexicon tags (T-pre3), en/es/de/fr/nl, training folds only",
       "labels": out}
p = A / "tag_aux_targets.json"
p.write_text(json.dumps(doc) + "\n")
print("sha256", hashlib.sha256(p.read_bytes()).hexdigest())
