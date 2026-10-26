"""Kaggle submission kernel -> eval twin on the first N study IDs of tumis001/rsna-fast-eval/eval_ids.txt
(training studies; gold-58 come first). Only the study list, series folder and an extra predictions file change."""
import json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = open(f"{src}/submission.py").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep('''    ids = [r["StudyInstanceUID"] for r in sample]
    fallback = {r["StudyInstanceUID"]: [float(r[t]) for t in TARGETS] for r in sample}
''', f'''    # EVAL VARIANT: first {n} training studies of tumis001/rsna-fast-eval/eval_ids.txt (gold-58 first), else unchanged
    ids = [s for s in (dataset_dir("tumis001/rsna-fast-eval") / "eval_ids.txt").read_text().split() if s][:{n}]
    fallback = {{s: [0.5] * 12 for s in ids}}
''')
rep('read_csv(data / "test_series.csv")', 'read_csv(data / "train_series.csv")')
rep('data / "test_series") for s in ids]', 'data / "train_series") for s in ids]')
rep('''    (work / "head_predictions.json").write_text(json.dumps(heads) + "\\n")
''', '''    (work / "head_predictions.json").write_text(json.dumps(heads) + "\\n")
    (work / "eval_predictions.json").write_text(json.dumps({"targets_in_order": TARGETS, "predictions": [
        {"study_id": s, "probabilities": final[s]} for s in ids if s in preds]}) + "\\n")
''')
import os
os.makedirs(dst, exist_ok=True)
open(f"{dst}/submission.py", "w").write(s)
m = json.load(open(f"{src}/kernel-metadata.json"))
slug = os.path.basename(dst.rstrip("/")).replace("_", "-")
m["id"], m["title"] = f"tumis001/rsna-{slug}", f"rsna-{slug}"
m["dataset_sources"] = m["dataset_sources"] + ["tumis001/rsna-fast-eval"]
json.dump(m, open(f"{dst}/kernel-metadata.json", "w"), indent=2)
print(m["id"])
