"""Runs ON the VM with the lane env: loads both teachers, checks the target rule reproduces E4's offline targets with no
shift, checks the 384 teacher reproduces its stored clean-view predictions, and times one teacher call."""
import csv, time
import torch
import r50_gl4_runner as L
r = L.r
dev = torch.device("cuda")
labels = r.load_labels(r.ASSETS / r.LABELS_FILE, "labels")
tl = r.load_labels(L.Path(L.TARGETS_FILE), "kd_targets")
fold = {x["study_id"]: x["fold"] for x in csv.DictReader((r.ASSETS / "folds.csv").open())}
ids = sorted(s for s in labels if fold[s] != "0")
okd = L.online_kd.OnlineTeacher(ids, labels, tl, r.TARGETS, dev)
okd.check_clean(r.read_manifest())
xg = torch.randn(32, 3, 384, 384, device=dev).contiguous(memory_format=torch.channels_last)
for _ in range(2):
    okd.target(xg, 2, torch.tensor([0, 1]))
torch.cuda.synchronize(); t = time.time()
for _ in range(5):
    okd.target(xg, 2, torch.tensor([0, 1]))
torch.cuda.synchronize()
print(f"CHECK_OK teacher per micro-batch (2 studies x 16 windows): {(time.time() - t) / 5:.3f} s; "
      f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
