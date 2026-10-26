"""Duplicate / near-duplicate studies across folds (CPU, 2026-10-13): gate for the cross-fitted teacher stage.

If the same exam (or the same patient's other knee / repeat exam) sits in two folds, an out-of-fold teacher has
effectively seen it and its "OOF" labels leak (PANDA 1st-place warning). Patient IDs are stripped, so duplicates are
detected from the report text: exact matches after whitespace / case normalisation, and near-duplicates by word
5-gram Jaccard >= 0.8 (MinHash-free exact pairwise over candidate pairs sharing a rare 5-gram). For each group:
fold spread and whether the 12 competition labels agree. DICOM header tags (scan parameters) are compared within
groups as a weak same-scanner signal. Nothing leaves the machine.

Usage (container):
  podman run --rm --network=none -v "$P":/proj:ro localhost/rsna-label-ds python -I /proj/repository/scripts/audit_duplicate_studies.py /proj
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 9)
R = Path(sys.argv[1])
rows = list(csv.DictReader(open(R / "reports_raw/train.csv", encoding="utf-8-sig")))
fold = {r["study_id"]: r["fold"] for r in csv.DictReader(open(R / "repository/data/coatnet_d96_fold0/assets/folds.csv"))}
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
hdr = {r["StudyInstanceUID"]: r for r in csv.DictReader(open(R / "repository/data/dicom_demographics/study_header_tags.csv"))}
ids = [r["StudyInstanceUID"] for r in rows]
rep = {r["StudyInstanceUID"]: r["Report"] for r in rows}
lab = {r["StudyInstanceUID"]: tuple(r[t] for t in T) for r in rows}
print(f"{len(rows)} studies with reports; {sum(s in fold for s in ids)} in folds.csv; fold sizes {dict(Counter(fold.values()))}")

norm = lambda t: re.sub(r"\s+", " ", t.lower()).strip()
words = {s: re.findall(r"\w+", norm(rep[s])) for s in ids}
empty = [s for s in ids if len(words[s]) < 5]
print(f"reports with < 5 words: {len(empty)}")


def summarise(groups, name):
    groups = [g for g in groups if len(g) > 1]
    n_st = sum(len(g) for g in groups)
    cross = [g for g in groups if len({fold.get(s) for s in g}) > 1]
    cross_val = [g for g in cross if "0" in {fold.get(s) for s in g}]
    same_lab = sum(len({lab[s] for s in g}) == 1 for g in groups)
    print(f"\n== {name}: {len(groups)} groups covering {n_st} studies (sizes {dict(Counter(len(g) for g in groups))})")
    print(f"   groups spanning > 1 fold: {len(cross)} ({sum(len(g) for g in cross)} studies); touching validation fold 0: "
          f"{len(cross_val)}")
    print(f"   groups whose 12 labels are identical: {same_lab}/{len(groups)}")
    diff = Counter()
    for g in groups:
        for k, t in enumerate(T):
            if len({lab[s][k] for s in g}) > 1:
                diff[t] += 1
    print("   groups where a finding's label differs: " + ", ".join(f"{t} {n}" for t, n in diff.most_common()))
    scan = [g for g in groups if all(s in hdr for s in g)]
    same_scan = sum(len({(hdr[s]["ManufacturerModelName"], hdr[s]["MagneticFieldStrength"]) for s in g}) == 1 for g in scan)
    print(f"   same scanner model + field strength within group: {same_scan}/{len(scan)}")
    return groups


exact = defaultdict(list)
for s in ids:
    exact[norm(rep[s])].append(s)
g_exact = summarise(list(exact.values()), "exact duplicate reports (normalised)")
for g in sorted(g_exact, key=len, reverse=True)[:3]:
    print(f"   e.g. size {len(g)}: '{norm(rep[g[0]])[:140]}...' folds {[fold.get(s) for s in g]}")

# near duplicates: 5-gram Jaccard over pairs sharing any 5-gram that occurs in <= 10 reports
grams = {s: {" ".join(w[i:i + 5]) for i in range(len(w) - 4)} for s, w in words.items() if len(w) >= 5}
df = Counter(g for gs in grams.values() for g in gs)
cand = defaultdict(set)
post = defaultdict(list)
for s, gs in grams.items():
    for g in gs:
        if df[g] <= 10:
            post[g].append(s)
for g, ss in post.items():
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            cand[ss[i]].add(ss[j])
parent = {s: s for s in ids}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x


pairs = 0
for a, bs in cand.items():
    for b in bs:
        j = len(grams[a] & grams[b]) / len(grams[a] | grams[b])
        if j >= 0.8:
            pairs += 1; parent[find(a)] = find(b)
near = defaultdict(list)
for s in ids:
    near[find(s)].append(s)
print(f"\nnear-duplicate pairs (Jaccard >= 0.8): {pairs}")
g_near = summarise(list(near.values()), "near-duplicate report groups (5-gram Jaccard >= 0.8, transitive)")
ex = [g for g in g_near if len({norm(rep[s]) for s in g}) > 1][:2]
for g in ex:
    a, b = norm(rep[g[0]]), norm(rep[g[1]])
    print(f"   e.g. folds {[fold.get(s) for s in g]}, labels identical {lab[g[0]] == lab[g[1]]}:\n     A '{a[:160]}'\n     B '{b[:160]}'")
