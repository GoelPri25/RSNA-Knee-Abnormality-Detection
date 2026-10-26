"""Phase 3 teacher splits (CPU): 4 cross-fitting splits over the 3,524 training studies (folds 1-5); fold 0 untouched.

Base assignment: fold 1 -> split 0, fold 2 -> 1, fold 3 -> 2, fold 4 -> 3; fold 5 (281 studies) goes to the smallest
split, one duplicate group at a time, in a fixed hashed order. Then every report-duplicate group among the training
studies is put into one split (the split holding most of it; ties -> lowest split), so no teacher has trained on a
template-duplicate of a study it labels (ledger 2-dup: keep cross-fold near-duplicate groups in one teacher split).
Groups = exact duplicates after whitespace / case normalisation UNION word 5-gram Jaccard >= 0.8 near-duplicates
(same rules as scripts/audit_duplicate_studies.py; the union matters because exact template groups share only common
5-grams and are not found by the near-duplicate candidate step). Fold-0 members of a group stay in fold 0.

Output: data/teacher_splits/teacher_splits.csv (study_id, teacher_split) + summary.txt. Labels are not read.

Usage (container):
  podman run --rm --network=none -v "$P":/proj localhost/rsna-label-ds python -I /proj/repository/scripts/build_teacher_splits.py /proj
"""
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 9)
R = Path(sys.argv[1])
OUT = R / "repository/data/teacher_splits"
FOLDS = R / "repository/data/coatnet_d96_fold0/assets/folds.csv"
assert hashlib.sha256(FOLDS.read_bytes()).hexdigest().startswith("2dd1044a")
fold = {r["study_id"]: r["fold"] for r in csv.DictReader(FOLDS.open())}
train = sorted(s for s, f in fold.items() if f != "0")
assert len(train) == 3524 and sum(f == "0" for f in fold.values()) == 825
rep = {r["StudyInstanceUID"]: r["Report"] for r in csv.DictReader(open(R / "reports_raw/train.csv", encoding="utf-8-sig"))}

# ---- duplicate groups (exact UNION near), as scripts/audit_duplicate_studies.py
norm = lambda t: re.sub(r"\s+", " ", t.lower()).strip()
ids = sorted(rep)
parent = {s: s for s in ids}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x


exact = defaultdict(list)
for s in ids:
    exact[norm(rep[s])].append(s)
for g in exact.values():
    for s in g[1:]:
        parent[find(s)] = find(g[0])
words = {s: re.findall(r"\w+", norm(rep[s])) for s in ids}
grams = {s: {" ".join(w[i:i + 5]) for i in range(len(w) - 4)} for s, w in words.items() if len(w) >= 5}
df = Counter(g for gs in grams.values() for g in gs)
post = defaultdict(list)
for s, gs in grams.items():
    for g in gs:
        if df[g] <= 10:
            post[g].append(s)
cand = defaultdict(set)
for ss in post.values():
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            cand[ss[i]].add(ss[j])
for a, bs in cand.items():
    for b in bs:
        if len(grams[a] & grams[b]) / len(grams[a] | grams[b]) >= 0.8:
            parent[find(a)] = find(b)
groups = defaultdict(list)
for s in train:                                     # training members only; fold-0 members never move
    groups[find(s)].append(s)
multi = [g for g in groups.values() if len(g) > 1]

# ---- base assignment, fold 5 balanced by group
split = {s: int(fold[s]) - 1 for s in train if fold[s] in "1234"}
size = Counter(split.values())
f5 = defaultdict(list)
for s in train:
    if fold[s] == "5":
        f5[find(s)].append(s)
for key in sorted(f5, key=lambda k: hashlib.sha256(k.encode()).hexdigest()):
    k = min(range(4), key=lambda j: (size[j], j))
    for s in f5[key]:
        split[s] = k
    size[k] += len(f5[key])

# ---- one split per duplicate group
moved, spanning = 0, 0
for g in multi:
    c = Counter(split[s] for s in g)
    if len(c) > 1:
        spanning += 1
        k = min(c, key=lambda j: (-c[j], j))
        for s in g:
            moved += split[s] != k; split[s] = k
assert set(split) == set(train) and all(len({split[s] for s in g}) == 1 for g in multi)

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "teacher_splits.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["study_id", "teacher_split"])
    for s in train:
        w.writerow([s, split[s]])
sz = Counter(split.values())
lines = [f"training studies {len(train)} (folds 1-5); fold 0 (825) untouched",
         f"duplicate groups among training studies: {len(multi)} covering {sum(len(g) for g in multi)} studies; "
         f"spanning > 1 split before the fix: {spanning}; studies moved: {moved}",
         "split sizes: " + ", ".join(f"{k}: {sz[k]}" for k in range(4)),
         "fold -> split counts: " + "; ".join(f"fold {f_}: {dict(sorted(Counter(split[s] for s in train if fold[s] == f_).items()))}"
                                             for f_ in "12345"),
         "sha256 teacher_splits.csv: " + hashlib.sha256((OUT / "teacher_splits.csv").read_bytes()).hexdigest()]
(OUT / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
