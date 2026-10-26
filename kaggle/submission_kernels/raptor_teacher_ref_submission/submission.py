"""Teacher reference (SURPASS-R, docs/experiment_ledger.md): the PUB-KD-R teacher itself on the public LB.
Raptor half of the public 0.958 blend = nartaa CoAtNet-rmlp-2 384 (320 centre crop) and 224 (0.8 centre crop, resized),
94 windows over the valid range, anatomical mirror TTA (probabilities averaged), fp16 autocast; per finding
final = 0.65 pct(384) + 0.35 pct(224) over the test studies (goodpjw2008 v25/27 Part A). The model code is
scripts/pubkd_raptor_teacher.py (which produced the training targets) on the E3 kernel's d96 builder (= training cache).
Parity gate before the test run: 6 fold-0 training studies rebuilt from DICOMs must match the MeluXina teacher run
(max |dp| <= 0.03 per model and view).
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IMG, CROP_MM, K = 384, 140.0, 94
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(k for _, _, k in SLOTS)
SLOT_OF = np.repeat(np.arange(5), [k for _, _, k in SLOTS])
W384 = 0.65
CKPT = {"384": ("raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "coatnet_rmlp_2_rw_384",
                "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"),
        "224": ("raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt", "coatnet_rmlp_2_rw_224",
                "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6")}
PARITY = {"1.2.826.0.1.3680043.8.498.10021288557171866794624317186288618513": {"384": {"plain": [0.26993998885154724, 0.3571600019931793, 0.3111039996147156, 0.34344300627708435, 0.3824920058250427, 0.30602899193763733, 0.45643600821495056, 0.35242900252342224, 0.37164199352264404, 0.30931100249290466, 0.1986899971961975, 0.39365100860595703], "mirror": [0.2884010076522827, 0.36017298698425293, 0.3999300003051758, 0.3445799946784973, 0.46133700013160706, 0.344217985868454, 0.47981199622154236, 0.32482099533081055, 0.35502898693084717, 0.3790780007839203, 0.2037619948387146, 0.38952499628067017]}, "224": {"plain": [0.37253499031066895, 0.3900589942932129, 0.3931719958782196, 0.49657300114631653, 0.3937720060348511, 0.33073899149894714, 0.4289799928665161, 0.24527199566364288, 0.3239380121231079, 0.3373419940471649, 0.20380300283432007, 0.4272089898586273], "mirror": [0.3863779902458191, 0.4099400043487549, 0.44880199432373047, 0.49754899740219116, 0.49398699402809143, 0.3892959952354431, 0.48754000663757324, 0.2660120129585266, 0.3464989960193634, 0.3360869884490967, 0.2702380120754242, 0.43744200468063354]}}, "1.2.826.0.1.3680043.8.498.12104799278874963688461235330096856415": {"384": {"plain": [0.2699030041694641, 0.3517859876155853, 0.26788198947906494, 0.4544839859008789, 0.2786090075969696, 0.3606500029563904, 0.2878279983997345, 0.7385179996490479, 0.6624209880828857, 0.9868360161781311, 0.5172150135040283, 0.4287880063056946], "mirror": [0.267536997795105, 0.3207019865512848, 0.2784300148487091, 0.5108199715614319, 0.31951600313186646, 0.4326629936695099, 0.3276109993457794, 0.7217419743537903, 0.6687219738960266, 0.9816970229148865, 0.4547480046749115, 0.4413659870624542]}, "224": {"plain": [0.17758099734783173, 0.33656400442123413, 0.299468994140625, 0.41813600063323975, 0.3905690014362335, 0.2708140015602112, 0.5493019819259644, 0.8153449892997742, 0.8956660032272339, 0.96943199634552, 0.3903839886188507, 0.40788599848747253], "mirror": [0.20199799537658691, 0.3051620125770569, 0.2333119958639145, 0.4515939950942993, 0.33696499466896057, 0.27913299202919006, 0.43097200989723206, 0.8192430138587952, 0.899740993976593, 0.9664710164070129, 0.3901210129261017, 0.4230070114135742]}}, "1.2.826.0.1.3680043.8.498.20933151653103548000061448393771614270": {"384": {"plain": [0.34725499153137207, 0.6439329981803894, 0.10179399698972702, 0.2025880068540573, 0.2323179990053177, 0.40699100494384766, 0.5348870158195496, 0.8419070243835449, 0.5203109979629517, 0.21835699677467346, 0.7378060221672058, 0.6188700199127197], "mirror": [0.34986400604248047, 0.6501650214195251, 0.09967000037431717, 0.26951101422309875, 0.21997399628162384, 0.39339300990104675, 0.573153018951416, 0.857778012752533, 0.5475299954414368, 0.2588529884815216, 0.7436090111732483, 0.6329320073127747]}, "224": {"plain": [0.24998199939727783, 0.5028449892997742, 0.07957100123167038, 0.11003299802541733, 0.1987909972667694, 0.33089199662208557, 0.5836449861526489, 0.844897985458374, 0.5853070020675659, 0.17879599332809448, 0.7421460151672363, 0.5580559968948364], "mirror": [0.23260900378227234, 0.5115990042686462, 0.06985300034284592, 0.16215300559997559, 0.20193099975585938, 0.3400439918041229, 0.6022430062294006, 0.8547629714012146, 0.5867900252342224, 0.21699999272823334, 0.646431028842926, 0.5210859775543213]}}, "1.2.826.0.1.3680043.8.498.38717329577625255474350472405748992219": {"384": {"plain": [0.26488399505615234, 0.3588710129261017, 0.9450830221176147, 0.9359909892082214, 0.4840199947357178, 0.6275749802589417, 0.48527899384498596, 0.5020049810409546, 0.49609899520874023, 0.36861398816108704, 0.39663198590278625, 0.468766987323761], "mirror": [0.26156699657440186, 0.4304949939250946, 0.9285299777984619, 0.9434270262718201, 0.48517701029777527, 0.6240310072898865, 0.49223700165748596, 0.5407710075378418, 0.5420659780502319, 0.5153570175170898, 0.44833600521087646, 0.49459201097488403]}, "224": {"plain": [0.19690300524234772, 0.3506399989128113, 0.9040390253067017, 0.9674010276794434, 0.4592280089855194, 0.6674330234527588, 0.5107849836349487, 0.48998600244522095, 0.4803760051727295, 0.35200798511505127, 0.2728640139102936, 0.39112401008605957], "mirror": [0.20198500156402588, 0.41121798753738403, 0.8805620074272156, 0.9658880233764648, 0.4735330045223236, 0.7028759717941284, 0.563601016998291, 0.5106080174446106, 0.4849810004234314, 0.41457700729370117, 0.2775219976902008, 0.406127005815506]}}, "1.2.826.0.1.3680043.8.498.58298767414933858487578597918936525600": {"384": {"plain": [0.25188499689102173, 0.24659499526023865, 0.09533999860286713, 0.17369599640369415, 0.12973999977111816, 0.2036440074443817, 0.22821900248527527, 0.2045450061559677, 0.28980299830436707, 0.2531220018863678, 0.1256909966468811, 0.3565309941768646], "mirror": [0.26201000809669495, 0.252157986164093, 0.09368100017309189, 0.17520500719547272, 0.12486899644136429, 0.17934100329875946, 0.2158380001783371, 0.2142219990491867, 0.2991600036621094, 0.26099100708961487, 0.1235010027885437, 0.35357800126075745]}, "224": {"plain": [0.23129500448703766, 0.23471400141716003, 0.06919199973344803, 0.16584600508213043, 0.1264220029115677, 0.17516200244426727, 0.2489060014486313, 0.21329499781131744, 0.2832779884338379, 0.23865899443626404, 0.10751300305128098, 0.32659101486206055], "mirror": [0.22905400395393372, 0.24838000535964966, 0.08274900168180466, 0.1959100067615509, 0.11957799643278122, 0.16989700496196747, 0.23588000237941742, 0.20812100172042847, 0.2764270007610321, 0.2472669929265976, 0.09161999821662903, 0.3247930109500885]}}, "1.2.826.0.1.3680043.8.498.80284846390295834578273590480193235798": {"384": {"plain": [0.16945800185203552, 0.23289699852466583, 0.11683300137519836, 0.0777290016412735, 0.15614600479602814, 0.07749400287866592, 0.14769500494003296, 0.19315700232982635, 0.28648099303245544, 0.2347870022058487, 0.08624900132417679, 0.3514859974384308], "mirror": [0.16857600212097168, 0.21611300110816956, 0.12144800275564194, 0.08916100114583969, 0.14309599995613098, 0.07243499904870987, 0.1314370036125183, 0.19665199518203735, 0.28778600692749023, 0.2654699981212616, 0.082737997174263, 0.34381601214408875]}, "224": {"plain": [0.18655900657176971, 0.26387399435043335, 0.15029799938201904, 0.13146600127220154, 0.14472399652004242, 0.10953699797391891, 0.14435000717639923, 0.1779710054397583, 0.2655690014362335, 0.25110700726509094, 0.10988199710845947, 0.3810519874095917], "mirror": [0.16735999286174774, 0.2667889893054962, 0.16868899762630463, 0.11674299836158752, 0.12804099917411804, 0.09399200230836868, 0.1367350071668625, 0.16052499413490295, 0.25387799739837646, 0.24322199821472168, 0.09941200166940689, 0.3703790009021759]}}}

DATA = next(p for p in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                        Path("/kaggle/input/rsna-knee-abnormality-detection")) if p.exists())


def dataset_dir(ref):
    owner, slug = ref.split("/")
    return next(p for p in (Path("/kaggle/input/datasets") / owner / slug, Path("/kaggle/input") / slug) if p.exists())


ART = dataset_dir("tumis001/rsna-coatnet-d96-exp1b-fold0")                 # holds the timm 1.0.29 wheel
NARTAA = dataset_dir("nartaa/rsna-knee-publication-swa-weights-20261007")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------- d96 builder (same as the training cache)
def header(path):
    import pydicom
    try:
        h = pydicom.dcmread(path, stop_before_pixels=True)
        iop = getattr(h, "ImageOrientationPatient", None)
        ipp = getattr(h, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r, c = np.array(iop[:3], float), np.array(iop[3:], float)
            pos = float(np.dot(np.array(ipp, float), np.cross(r, c)))
        else:
            pos = float(getattr(h, "InstanceNumber", 0) or 0)
        ps = getattr(h, "PixelSpacing", None)
        return pos, path, (float(ps[0]) if ps is not None else 0.5), True
    except Exception:
        return 0.0, path, 0.5, False


def pixels(path):
    import pydicom
    from pydicom.pixels import apply_modality_lut
    d = pydicom.dcmread(path)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def crop_resize(a, ps):
    import cv2
    h, w = a.shape
    cpx = min(int(round(CROP_MM / max(ps, 1e-3))), h, w)
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (IMG, IMG), interpolation=cv2.INTER_AREA)


def build_study(job):  # noqa: C901
    """E1 kernel's builder plus the teammate's per-slot Fat_Suppression flag (volume identical)."""
    study_id, rows, SPLIT = job
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx, used, errors = 0, set(), 0
    fat = np.zeros(len(SLOTS), np.int64)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for j, (plane, fluid, k) in enumerate(SLOTS):
            cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
            pref = [r for r in cands if fluid in (0, 1) and int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
            row = (pref or cands or [None])[0]
            if row is None:
                idx += k
                continue
            used.add(row["SeriesInstanceUID"])
            fat[j] = 1 if str(row.get("Fat_Suppression")) == "1" else 0     # training manifest's rule
            files = sorted((DATA / SPLIT / study_id / row["SeriesInstanceUID"]).glob("*.dcm"))
            heads = list(pool.map(header, files))
            ok_ps = [h[2] for h in heads if h[3]]
            med_ps = float(np.median(ok_ps)) if ok_ps else 0.5
            heads.sort(key=lambda h: h[0])
            n = len(heads)
            if n == 0:
                idx += k
                continue
            lo, hi = int(n * 0.02), int(n * 0.98) - 1
            hi = max(hi, lo)
            picks = [int(min(p, n - 1)) for p in (np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k)]

            def load(p):
                try:
                    return pixels(heads[p][1]), heads[p][2]
                except Exception:
                    return None, med_ps
            got = list(pool.map(load, picks))
            valid = [a for a, _ in got if a is not None]
            errors += len(got) - len(valid)
            loq, hiq = (np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0]) if valid else (0.0, 1.0))
            for a, ps in got:
                if a is not None:
                    aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                    vol[idx] = (crop_resize(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8)
                idx += 1
    mask = vol.reshape(MAXS, -1).max(1) > 0
    return study_id, vol, mask, errors, fat


class RaptorClassifier(nn.Module):
    """Their head (scripts/pubkd_raptor_teacher.py): LayerNorm, per-finding attention, softmax over windows."""

    def __init__(self, backbone, d, n=12):
        super().__init__()
        self.backbone, self.norm = backbone, nn.LayerNorm(d)
        self.att = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.0), nn.Linear(256, n))
        self.clsW, self.clsb = nn.Parameter(torch.zeros(n, d)), nn.Parameter(torch.zeros(n))

    def forward(self, x):
        f = torch.cat([self.backbone(c) for c in x.split(16)])
        h = self.norm(f)
        a = torch.softmax(self.att(h).float(), 0).to(h.dtype)
        return (torch.einsum("kn,kf->nf", a, h) * self.clsW).sum(-1) + self.clsb


def eval_centers(mask):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, MAXS))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, MAXS - 2))]
    return np.array([max(1, min(cs[i], MAXS - 2)) for i in np.linspace(0, len(cs) - 1, K).round().astype(int)])


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def windows(vol, dev, mirror, model):
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    cs = eval_centers(mask)
    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)
    if mirror:
        sag = torch.from_numpy(np.isin(SLOT_OF[cs], (0, 1))).to(dev).view(-1, 1, 1, 1)
        tri = torch.where(sag, tri.flip(1), tri.flip(-1))
    x = tri.float() / 255.0
    if model == "384":
        lo = (IMG - 320) // 2
        x = x[..., lo:lo + 320, lo:lo + 320]
    else:
        c = int(round(IMG * 0.8)); o = (IMG - c) // 2
        x = F.interpolate(x[..., o:o + c, o:o + c], size=(224, 224), mode="bilinear", align_corners=False)
    return (x - MEAN.to(dev)) / STD.to(dev)


def ensure_timm():
    try:
        import timm
        if timm.__version__ == "1.0.29":
            return
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "-q",
                    str(ART / "timm-1.0.29-py3-none-any.whl")], check=True)


def load(model, dev):
    import timm
    fn, arch, sha = CKPT[model]
    hits = sorted(NARTAA.rglob(fn))
    if len(hits) != 1:
        raise FileNotFoundError(f"{fn}: {hits}")
    if hashlib.sha256(hits[0].read_bytes()).hexdigest() != sha:
        raise ValueError(f"checkpoint hash differs: {fn}")
    ck = torch.load(hits[0], map_location="cpu", weights_only=True)
    if list(ck["lab"]) != TARGETS:
        raise ValueError(f"target order differs: {ck['lab']}")
    bb = timm.create_model(arch, pretrained=False, num_classes=0, **({"img_size": 320} if model == "384" else {}))
    m = RaptorClassifier(bb, bb.num_features)
    m.load_state_dict(ck["model"], strict=True)
    return m.eval().to(dev)


def score(models, vol, dev):
    out = {}
    for name, m in models.items():
        rec = {}
        for view, mir in (("plain", False), ("mirror", True)):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                z = m(windows(vol, dev, mir, name)).float()
            if not torch.isfinite(z).all():
                z = m(windows(vol, dev, mir, name)).float()
            rec[view] = torch.sigmoid(z).cpu().numpy()
        out[name] = rec
    return out


def pct(v):
    v = np.asarray(v, float); o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r / len(v)


def main():
    started = time.monotonic()
    work = Path("/kaggle/working")
    sample = read_csv(DATA / "sample_submission.csv")
    if not sample or list(sample[0]) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("sample_submission.csv columns differ from the model target order")
    ids = [r["StudyInstanceUID"] for r in sample]
    fallback = {r["StudyInstanceUID"]: [float(r[t]) for t in TARGETS] for r in sample}

    def write(preds):
        with (work / "submission.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["StudyInstanceUID", *TARGETS])
            for s in ids:
                w.writerow([s, *[f"{v:.8f}" for v in preds[s]]])
    write(fallback)
    ensure_timm()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {k: load(k, dev) for k in ("384", "224")}
    stats = {"studies": len(ids), "empty_studies": 0, "decode_errors": 0}

    # parity gate on training studies
    tr = {}
    for row in read_csv(DATA / "train_series.csv"):
        if row["StudyInstanceUID"] in PARITY:
            tr.setdefault(row["StudyInstanceUID"], []).append(row)
    worst = 0.0
    with torch.inference_mode():
        for s in PARITY:
            _, vol, mask, err, _ = build_study((s, tr.get(s, []), "train_series"))
            got = score(models, vol, dev)
            for mname in ("384", "224"):
                for view in ("plain", "mirror"):
                    worst = max(worst, float(np.abs(got[mname][view] - np.array(PARITY[s][mname][view])).max()))
    print(f"PARITY {len(PARITY)} train studies: max |dp| {worst:.4f}", flush=True)
    stats["parity_max_abs_dp"] = round(worst, 5)
    if worst > 0.03:
        raise ValueError(f"parity failed: {worst}")

    by_study = {}
    for row in read_csv(DATA / "test_series.csv"):
        by_study.setdefault(row["StudyInstanceUID"], []).append(row)
    probs = {}
    with Pool(4) as procs, torch.inference_mode():
        for n, (sid, vol, mask, errors, _) in enumerate(
                procs.imap(build_study, [(s, by_study.get(s, []), "test_series") for s in ids]), 1):
            stats["decode_errors"] += errors
            if not mask.any():
                stats["empty_studies"] += 1
                continue
            got = score(models, vol, dev)
            probs[sid] = {m: (0.5 * (got[m]["plain"] + got[m]["mirror"])).tolist() for m in got}
            if n % 50 == 0 or n == len(ids):
                print(f"raptor {n}/{len(ids)} studies; {time.monotonic() - started:.0f}s", flush=True)

    have = [s for s in ids if s in probs]
    final = {s: fallback[s] for s in ids}
    if have:
        A = np.array([probs[s]["384"] for s in have]); B = np.array([probs[s]["224"] for s in have])
        R = np.column_stack([W384 * pct(A[:, j]) + (1 - W384) * pct(B[:, j]) for j in range(len(TARGETS))])
        for i, s in enumerate(have):
            final[s] = R[i].tolist()
    write(final)
    written = read_csv(work / "submission.csv")
    vals = np.array([[float(r[t]) for t in TARGETS] for r in written])
    if [r["StudyInstanceUID"] for r in written] != ids or not np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
        raise ValueError("Saved submission is invalid")
    (work / "member_probabilities.json").write_text(json.dumps(probs) + "\n")
    summary = {"model": "Raptor teacher: 0.65 pct(nartaa 384) + 0.35 pct(nartaa 224), mirror TTA (PUB-KD-R teacher)",
               "checkpoints_sha256": {k: v[2] for k, v in CKPT.items()}, "d96": stats,
               "seconds": round(time.monotonic() - started, 1)}
    (work / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
