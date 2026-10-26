"""SURPASS-F: list fastMRI studies whose d96 slots got the wrong contrast (read-only over manifest.jsonl).
Job 5363573's builder read "SAG PD FSE NON FAT SAT" as fat-sat (fluid), so such a non-FS series could land in a fluid
slot (0 / 2) and the real FS series in the non-fluid slot (1 / 3). Flag a study if a slot-0/2 pick's description is a
negated fat-sat, or a slot-1/3 pick's description is clearly fat-sat (and not negated).
Reason per study: "negation" (a series description in the study is a negated fat-sat: the classifier bug) or
"fs_fallback" (no negation; the competition builder's fallback put a second FS series, e.g. a repeat, in slot 1/3).
Usage: python3 -I fastmri_misslot.py <fastMRI d96 root> <out json> <tag>=<shard,shard,..> ...   e.g. b2=0,1 b1=2,3,4,5,6,7
"""
import collections
import json
import re
import sys
from pathlib import Path

NEG = re.compile(r"(?<![a-z])(non|no|w/o|wo|without)[\s_\-/]*(fat[\s_\-]*(sat|sup|supp|suppressed|suppression)?|fs)(?![a-z])")
POS = re.compile(r"fat[\s_\-]*sat|fatsat|ftst|(?<![a-z])(fs|stir|spair|spir)(?![a-z])|fat[\s_\-]*sup")
RULE = ("slot 0/2 (fluid) pick description matches negated fat-sat " + NEG.pattern +
        ", or slot 1/3 (non-fluid) pick description matches fat-sat " + POS.pattern + " without that negation")


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    ids, counts, why, by = [], {}, collections.Counter(), collections.defaultdict(list)
    for arg in sys.argv[3:]:
        tag, shards = arg.split("=")
        n = collections.Counter()
        for k in shards.split(","):
            for line in (root / f"shard_{k}" / "manifest.jsonl").read_text().splitlines():
                r = json.loads(line)
                desc = {s["SeriesInstanceUID"]: s.get("description", "") for s in r["series_table"]}
                hit = []
                for j, s in enumerate(r["slots"][:4]):
                    if s["series"] is None:
                        continue
                    d = desc.get(s["series"], "").lower()
                    neg = bool(NEG.search(d))
                    if (j in (0, 2) and neg) or (j in (1, 3) and not neg and POS.search(d)):
                        hit.append(f"slot{j}: {desc.get(s['series'], '')}")
                if hit:
                    reason = "negation" if any(NEG.search(d.lower()) for d in desc.values()) else "fs_fallback"
                    ids.append(r["study_id"]); by[reason].append(r["study_id"]); n[reason] += 1
                    why.update(hit)
        counts[tag] = dict(n, total=sum(n.values()))
    res = {"rule": RULE, "counts": counts, "n": len(ids), "study_ids": sorted(ids),
           "study_ids_by_reason": {k: sorted(v) for k, v in by.items()}, "picks_flagged": dict(why.most_common())}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("study_ids")}, indent=1))


if __name__ == "__main__":
    main()
