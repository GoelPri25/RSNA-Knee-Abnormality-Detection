"""Exp MG0 (ledger row MG0): MedGemma 1.5 4B as a knee-MRI reader, Stage 0 = what its pretraining gives for free.
Inference only: the model is never trained here and never sees a report. Labels are not touched.

Per study, one prompt holds all valid d96-cache slices (5 series, each introduced by a text header, every slice an
image followed by "SLICE n", as in Google's CT notebook), then 12 finding questions phrased with the host rubric.
The image prefix is prefilled once; each question branches from a copy of its KV cache.
Saved per study:
  zs     [12]        zero-shot score = logsumexp(logit Yes, yes) - logsumexp(logit No, no) at the answer position
  h_last [12, 2560]  final-layer hidden state at the answer position (fp16)   -> latent probe
  h_mid  [12, 2560]  middle-layer hidden state at the answer position (fp16)
  vis    [96, 2560]  MedGemma projector output mean-pooled per slice (fp16, zero rows where the slot is empty)
  r50    [96, 2048]  frozen ImageNet ResNet-50 (tv_in1k) pooled feature per slice, the same-protocol baseline
Probes (CPU, $0): scripts/audit_medgemma_stage0.py.

Run (personal Modal profile only):
  source ~/.config/rsna/modal-personal.env && modal run scripts/modal_medgemma_stage0.py --mode profile
  source ~/.config/rsna/modal-personal.env && modal run --detach scripts/modal_medgemma_stage0.py --mode extract --n-train N
Weights: google/medgemma-1.5-4b-it (Health AI Developer Foundations terms), fetched from Hugging Face with the
user's read token (HF_TOKEN in ~/.config/rsna/hf.env, never printed); safetensors + configs only, no remote code.
"""
import json
import os
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parents[1]
MODEL_ID = "google/medgemma-1.5-4b-it"
MODEL_REV = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"            # google/medgemma-1.5-4b-it, 2026-04-13
VOL = Path("/root/vol")
WEIGHTS = VOL / "hf/medgemma-1.5-4b-it"
OUT = VOL / "experiments/medgemma_stage0"
volume = modal.Volume.from_name("rsna-coatnet-d96-cache", create_if_missing=False)


def _hf_secret():
    if not modal.is_local():
        return modal.Secret.from_dict({})
    env = Path.home() / ".config/rsna/hf.env"
    lines = [l.strip() for l in env.read_text().splitlines() if l.strip()] if env.is_file() else []
    tok = next((l.split("=", 1)[1].strip().strip('"') for l in lines if l.startswith("HF_TOKEN=")),
               next((l for l in lines if l.startswith("hf_")), ""))   # "HF_TOKEN=hf_..." or a bare token line
    return modal.Secret.from_dict({"HF_TOKEN": tok})


image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "transformers==4.57.1", "accelerate==1.10.1", "huggingface_hub==0.35.3",
                      "timm==1.0.29", "numpy", "pillow")
         .run_commands("python -c \"import timm; timm.create_model('resnet50.tv_in1k', pretrained=True)\"")
         .add_local_dir(str(LOCAL / "data/coatnet_d96_fold0/assets"), "/root/assets"))
app = modal.App("rsna-medgemma-stage0", image=image)

SLOTS = [26, 22, 18, 12, 18]
PLANE = ["sagittal", "sagittal", "coronal", "coronal", "axial"]
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# Host rubric (public forum post by the hosts): on-the-fence = negative; thresholds below.
QUESTIONS = [
    "Is there an anterior cruciate ligament tear, with more than 50% of the fibers disrupted?",
    "Is there a high-grade medial collateral ligament injury?",
    "Is there a medial meniscus tear: abnormal signal that clearly reaches the articular surface on at least two "
    "images, or a morphologic abnormality of the medial meniscus?",
    "Is there a lateral meniscus tear: abnormal signal that clearly reaches the articular surface on at least two "
    "images, or a morphologic abnormality of the lateral meniscus?",
    "Is there osteoarthritis of the medial tibiofemoral compartment: cartilage loss of more than 50% of its thickness "
    "over at least 1 cm?",
    "Is there osteoarthritis of the lateral tibiofemoral compartment: cartilage loss of more than 50% of its thickness "
    "over at least 1 cm?",
    "Is there patellofemoral osteoarthritis: cartilage loss of more than 50% of its thickness over at least 1 cm in the "
    "patellofemoral compartment?",
    "Is there a moderate or large joint effusion?",
    "Is there synovitis?",
    "Is there a moderate or large Baker's (popliteal) cyst?",
    "Is there a bone contusion (bone marrow edema without a fracture line)?",
    "Is there a fracture?",
]
INSTRUCTION = ("You are an expert musculoskeletal radiologist. Below is one knee MRI examination: several series, each "
               "a stack of consecutive slices in order. Review every slice of every series carefully.")
SUFFIX = " Answer with yes or no."


def study_index():
    """study_id -> (npy path, first row, valid mask [96], slot headers); fold / gold from the runner's assets."""
    import csv
    meta = {}
    for i in range(4):
        root = VOL / f"d96/shard_{i}"
        for line in (root / "manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            heads = []
            for j, s in enumerate(r["slots"]):
                fat = {"1": "fat-suppressed", "0": "non-fat-suppressed"}.get(str(s.get("fat")), "")
                fl = {"1": "fluid-sensitive", "0": "non-fluid-sensitive"}.get(str(s.get("fluid")), "")
                heads.append(f"{PLANE[j]} series, {fl} {fat}".replace("  ", " ").strip().rstrip(","))
            meta[r["study_id"]] = {"file": str(root / r["part"]), "row": r["row_start"],
                                   "mask": r["mask"], "gold": r["gold"], "heads": heads}
    fold = {r["study_id"]: r["fold"] for r in csv.DictReader(open("/root/assets/folds.csv"))}
    return meta, fold


class Reader:
    def __init__(self):
        import timm
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        if not (WEIGHTS / "config.json").is_file():
            snapshot_download(MODEL_ID, revision=MODEL_REV, local_dir=str(WEIGHTS), token=os.environ["HF_TOKEN"] or None,
                              allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.jinja"])
            volume.commit()
        assert not list(WEIGHTS.glob("*.py")), "remote code present; refusing"
        self.torch = torch
        self.proc = transformers.AutoProcessor.from_pretrained(str(WEIGHTS))
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            str(WEIGHTS), dtype=torch.bfloat16, device_map="cuda").eval()
        self.vis = []
        self.model.model.multi_modal_projector.register_forward_hook(lambda m, a, o: self.vis.append(o.float().mean(1)))
        tok = self.proc.tokenizer
        def ids(words):                                   # single-token spellings only
            enc = [tok.encode(w, add_special_tokens=False) for w in words]
            return sorted({e[0] for e in enc if len(e) == 1})
        self.yes, self.no = ids(["Yes", "yes"]), ids(["No", "no"])
        assert self.yes and self.no, (self.yes, self.no)
        self.r50 = timm.create_model("resnet50.tv_in1k", pretrained=True, num_classes=0).cuda().eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        self.n_layers = self.model.config.text_config.num_hidden_layers

    def messages(self, imgs, heads, mask, q):
        content = [{"type": "text", "text": INSTRUCTION}]
        r = 0
        for j, k in enumerate(SLOTS):
            rows = [i for i in range(r, r + k) if mask[i]]
            if rows:
                content.append({"type": "text", "text": f"\n\nSeries {j + 1}: {heads[j]}, {len(rows)} slices."})
                for n, i in enumerate(rows, 1):
                    content.append({"type": "image", "image": imgs[i]})
                    content.append({"type": "text", "text": f"SLICE {n}"})
            r += k
        content.append({"type": "text", "text": "\n\n" + q + SUFFIX})
        return [{"role": "user", "content": content}]

    def encode(self, msgs):
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                             return_tensors="pt")

    def study(self, vol, mask, heads, parity=False):
        import copy
        import numpy as np
        import PIL.Image
        torch = self.torch
        imgs = [PIL.Image.fromarray(np.ascontiguousarray(v)).convert("RGB") for v in vol]
        full = [self.encode(self.messages(imgs, heads, mask, q)) for q in QUESTIONS[:2]]
        a, b = full[0]["input_ids"][0], full[1]["input_ids"][0]
        p = int((a[:min(len(a), len(b))] != b[:min(len(a), len(b))]).nonzero()[0])   # first differing token
        enc = full[0]
        itok = getattr(self.model.config, "image_token_id", None) or self.model.config.image_token_index
        n_img_tok = int((enc["input_ids"] == itok).sum())
        pre = {k: v[:, :p].cuda() if k != "pixel_values" else v.cuda(dtype=torch.bfloat16) for k, v in enc.items()}
        self.vis.clear()
        with torch.inference_mode():
            out = self.model(**pre, use_cache=True)
            cache = out.past_key_values
            assert len(self.vis) == 1, len(self.vis)          # projector ran once, on every valid slice
            v = self.vis[0]
            zs, hl, hm = [], [], []
            for q in QUESTIONS:
                e = self.encode(self.messages(imgs, heads, mask, q))["input_ids"][0]
                assert torch.equal(e[:p], enc["input_ids"][0, :p]), "question changed the shared prefix"
                suf = e[p:].view(1, -1).cuda()
                o = self.model(input_ids=suf, past_key_values=copy.deepcopy(cache), use_cache=True,
                               attention_mask=torch.ones(1, len(e), dtype=torch.long, device="cuda"),
                               output_hidden_states=True)
                lg = o.logits[0, -1].float()
                zs.append(float(torch.logsumexp(lg[self.yes], 0) - torch.logsumexp(lg[self.no], 0)))
                hl.append(o.hidden_states[-1][0, -1].half().cpu().numpy())
                hm.append(o.hidden_states[self.n_layers // 2][0, -1].half().cpu().numpy())
            par = None
            if parity:                                    # branch-from-cache vs one full forward, question 1
                f = {k: v.cuda() if k != "pixel_values" else v.cuda(dtype=torch.bfloat16) for k, v in enc.items()}
                lg = self.model(**f).logits[0, -1].float()
                par = abs(float(torch.logsumexp(lg[self.yes], 0) - torch.logsumexp(lg[self.no], 0)) - zs[0])
            vis = np.zeros((96, v.shape[-1]), np.float16)
            vis[np.where(mask)[0]] = v.half().cpu().numpy()[:int(np.sum(mask))]
            x = torch.from_numpy(np.ascontiguousarray(vol)).cuda().float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = self.r50((x - self.mean) / self.std).float()
            r50 = r.half().cpu().numpy() * np.asarray(mask, np.float16)[:, None]
        assert v.shape[0] == int(np.sum(mask)), (v.shape, int(np.sum(mask)))
        return (np.array(zs, np.float32), np.stack(hl), np.stack(hm), vis, r50,
                {"prefix_tokens": p, "image_tokens": n_img_tok, "parity": par})


def load_vol(m):
    import numpy as np
    return np.load(m["file"], mmap_mode="r")[m["row"]:m["row"] + 96]


@app.function(gpu="H100", cpu=8, memory=65536, timeout=3600, secrets=[_hf_secret()], volumes={"/root/vol": volume})
def profile(n: int = 8):
    """Load, parity-check the cache branching on 2 studies, time n fold-0 studies, report memory."""
    import numpy as np
    import torch
    meta, fold = study_index()
    rd = Reader()
    ids = sorted(s for s in meta if fold.get(s) == "0")[:n]
    times, info = [], []
    for i, s in enumerate(ids):
        m = meta[s]
        t = time.time()
        zs, hl, hm, vis, r50, inf = rd.study(load_vol(m), m["mask"], m["heads"], parity=i < 2)
        torch.cuda.synchronize(); times.append(time.time() - t)
        assert np.isfinite(zs).all() and np.isfinite(hl.astype(np.float32)).all()
        info.append(inf)
        print(f"{s[-12:]}: {times[-1]:.1f} s, slices {int(np.sum(m['mask']))}, {inf}, zs {np.round(zs, 2).tolist()}",
              flush=True)
    rep = {"s_per_study_median": float(np.median(times[1:])), "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
           "parity_max": max(i["parity"] for i in info[:2]), "yes_ids": rd.yes, "no_ids": rd.no,
           "n_layers": rd.n_layers}
    assert rep["parity_max"] < 0.1, rep
    print("PROFILE " + json.dumps(rep), flush=True)
    return rep


@app.function(gpu="H100", cpu=8, memory=65536, timeout=6 * 3600, secrets=[_hf_secret()],
              volumes={"/root/vol": volume})
def extract(ids: list, part: int):
    import numpy as np
    meta, _ = study_index()
    rd = Reader()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"part_{part:02d}.npz"
    done = dict(np.load(f, allow_pickle=False)) if f.is_file() else None
    have = set(done["ids"].tolist()) if done else set()
    buf = {k: [] for k in ("ids", "zs", "h_last", "h_mid", "vis", "r50")}
    if done:
        for k in buf:
            buf[k] = list(done[k])
    t0 = time.time()
    for n, s in enumerate([s for s in ids if s not in have], 1):
        m = meta[s]
        zs, hl, hm, vis, r50, _ = rd.study(load_vol(m), m["mask"], m["heads"])
        for k, v in zip(buf, (s, zs, hl, hm, vis, r50)):
            buf[k].append(v)
        if n % 50 == 0 or len(buf["ids"]) == len(ids):
            np.savez(f, **{k: np.array(v) if k == "ids" else np.stack(v) for k, v in buf.items()})
            volume.commit()
            print(f"part {part}: {len(buf['ids'])}/{len(ids)}, {(time.time() - t0) / n:.1f} s/study", flush=True)
    return len(buf["ids"])


@app.local_entrypoint()
def main(mode: str = "profile", n_train: int = 0, parts: int = 4):
    import csv
    import random
    if mode == "profile":
        print(profile.remote())
        return
    assert mode == "extract" and n_train > 0
    fold = {r["study_id"]: r["fold"] for r in csv.DictReader(open(LOCAL / "data/coatnet_d96_fold0/assets/folds.csv"))}
    labels = json.loads((LOCAL / "data/coatnet_d96_fold0/assets/soft_labels_final_per_target_latD2.json").read_text())
    lab = sorted(r["study_id"] for r in labels["labels"])
    val = sorted(s for s in lab if fold[s] == "0")
    train = sorted(s for s in lab if fold[s] != "0")
    random.Random(2026).shuffle(train)
    gold = sorted(json.loads(l)["study_id"] for i in range(4)
                  for l in (LOCAL / f"data/coatnet_d96_fold0/manifests/manifest_{i}.jsonl").read_text().splitlines()
                  if json.loads(l)["gold"])
    ids = val + gold + train[:n_train]
    print(f"fold 0 {len(val)}, gold {len(gold)}, train {min(n_train, len(train))} -> {len(ids)} studies, {parts} parts")
    chunks = [ids[i::parts] for i in range(parts)]
    print(list(extract.starmap([(c, i) for i, c in enumerate(chunks)])))
