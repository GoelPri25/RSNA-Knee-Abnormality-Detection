"""Build a Kaggle GPU script for one-time inference-only gold evaluation."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
MODULE = PROJECT / "src/rsna_knee/pillar0_reader.py"
RUNNER = HERE / "run.py"


RUNNER_MAIN = r'''
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from scipy.ndimage import affine_transform
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
FEATURE_DIM = 384
PATCH_GRID = (32, 32, 32)
TARGET_SHAPE_ZYX = (192, 384, 384)
TARGET_SPACING_XYZ_MM = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
MODEL_INPUT_KEY = "breast_mr"
EXPECTED_REVISION = "be5050ba482607f90e607e044f03a2ebc38f98c3"
EXPECTED_WEIGHTS_SHA256 = "d658fa8aad744c66d1abf43c2bf6bf594e75e834186dce9aadd1b7a9e4381bf0"
FOLDS_TO_SCORE = (0, 1, 2)
ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working/pillar0_regional_gold_eval")
WORK.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0")


def dataset_dir(slug: str) -> Path:
    direct = ROOT / "datasets" / slug
    if direct.exists():
        return direct
    found = [p for p in ROOT.rglob(slug) if p.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one Kaggle input for {slug}; found {found[:5]}")
    return found[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate_data() -> Path:
    candidates = [ROOT / "competitions/rsna-knee-abnormality-detection",
                  ROOT / "rsna-knee-abnormality-detection"]
    data = next((p for p in candidates if (p / "train.csv").is_file()), None)
    if data is None:
        raise FileNotFoundError("Attach RSNA Knee Abnormality Detection competition data")
    return data


def locate_bundle() -> Path:
    matches = []
    for candidate in ROOT.rglob("model.safetensors"):
        folder = candidate.parent
        provenance = folder / "MODEL_PROVENANCE.json"
        if not provenance.is_file():
            continue
        info = json.loads(provenance.read_text())
        if info.get("source_revision") == EXPECTED_REVISION:
            if sha256(candidate) != EXPECTED_WEIGHTS_SHA256:
                raise RuntimeError("Pillar-0 weights hash mismatch")
            matches.append(folder)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one verified Pillar-0 bundle; found {matches}")
    return matches[0]


def load_pillar(bundle: Path):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    package_parent = Path("/kaggle/working")
    package = package_parent / "pillar0_gold_eval_model"
    shutil.rmtree(package, ignore_errors=True)
    package.mkdir()
    (package / "__init__.py").write_text("")
    for source in bundle.glob("*.py"):
        shutil.copy2(source, package / source.name)
    sys.path.insert(0, str(package_parent))
    from pillar0_gold_eval_model.config_clip_mmatlas import CLIPMultimodalAtlasConfig
    from pillar0_gold_eval_model.modeling_clip_mmatlas import CLIPMultimodalAtlas
    config = CLIPMultimodalAtlasConfig.from_pretrained(bundle, local_files_only=True)
    with torch.device("cpu"):
        model = CLIPMultimodalAtlas(config)
    weights = load_file(str(bundle / "model.safetensors"), device="cpu")
    result = torch.nn.Module.load_state_dict(model, weights, strict=True)
    del weights
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Pillar-0 state mismatch: {result}")
    model.eval().requires_grad_(False).to(DEVICE)
    return model


class GeometryError(ValueError):
    pass


def reconstruct_series(root: Path, study: str, series: str):
    folder = root / str(study) / str(series)
    paths = sorted(folder.glob("*.dcm"))
    if len(paths) < 2:
        raise GeometryError(f"Expected >=2 slices; found {len(paths)}")
    headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in paths]
    required = ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
                "Rows", "Columns", "SOPInstanceUID")
    for header in headers:
        if any(not hasattr(header, field) for field in required):
            raise GeometryError("Missing DICOM geometry")
        if (str(header.StudyInstanceUID) != str(study)
                or str(header.SeriesInstanceUID) != str(series)):
            raise GeometryError("DICOM UID does not match folder")
        if getattr(header, "Modality", "") != "MR" or int(getattr(header, "NumberOfFrames", 1)) != 1:
            raise GeometryError("Only classic single-frame MR is supported")
    iop = np.asarray(headers[0].ImageOrientationPatient, dtype=np.float64)
    spacing = np.asarray(headers[0].PixelSpacing, dtype=np.float64)
    if not np.isfinite(iop).all() or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise GeometryError("Nonfinite orientation or invalid pixel spacing")
    col_dir, row_dir = iop[:3], iop[3:]
    if not np.allclose([np.linalg.norm(col_dir), np.linalg.norm(row_dir),
                        np.dot(col_dir, row_dir)], [1, 1, 0], atol=1e-4):
        raise GeometryError("Orientation cosines are not orthonormal")
    if any(not np.allclose(h.ImageOrientationPatient, iop, atol=1e-4, rtol=0)
           or not np.allclose(h.PixelSpacing, spacing, atol=1e-4, rtol=0)
           or (int(h.Rows), int(h.Columns)) != (int(headers[0].Rows), int(headers[0].Columns))
           for h in headers):
        raise GeometryError("Orientation, spacing, or matrix varies in the series")
    if len({str(h.SOPInstanceUID) for h in headers}) != len(headers):
        raise GeometryError("Duplicate SOPInstanceUID")
    for field in ("EchoTime", "TemporalPositionIdentifier", "AcquisitionNumber"):
        if len({str(getattr(h, field, "")) for h in headers}) > 1:
            raise GeometryError(f"Mixed acquisition: {field}")
    positions = np.asarray([h.ImagePositionPatient for h in headers], dtype=np.float64)
    normal = np.cross(col_dir, row_dir)
    order = np.argsort(positions @ normal, kind="stable")
    gaps = np.diff((positions @ normal)[order])
    if len(gaps) == 0 or (gaps < 1e-3).any():
        raise GeometryError("Repeated or indistinguishable physical slice positions")
    dz = float(np.median(gaps))
    if (not np.allclose(gaps, dz, atol=0.05, rtol=0.01)
            or not np.allclose(np.diff(positions[order], axis=0), normal * dz,
                               atol=0.05, rtol=0.01)):
        raise GeometryError("Irregular spacing or sheared volume")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = normal * dz
    affine[:3, 1] = row_dir * spacing[0]
    affine[:3, 2] = col_dir * spacing[1]
    affine[:3, 3] = positions[order][0]
    slices = []
    for index in order:
        dcm = pydicom.dcmread(paths[index])
        if int(getattr(dcm, "SamplesPerPixel", 1)) != 1:
            raise GeometryError("Expected grayscale MRI")
        pixels = dcm.pixel_array.astype(np.float32)
        if hasattr(dcm, "PixelPaddingValue"):
            low = min(float(dcm.PixelPaddingValue),
                      float(getattr(dcm, "PixelPaddingRangeLimit", dcm.PixelPaddingValue)))
            high = max(float(dcm.PixelPaddingValue),
                       float(getattr(dcm, "PixelPaddingRangeLimit", dcm.PixelPaddingValue)))
            pixels[(pixels >= low) & (pixels <= high)] = np.nan
        pixels = pixels * float(getattr(dcm, "RescaleSlope", 1.0))
        pixels += float(getattr(dcm, "RescaleIntercept", 0.0))
        if not np.isfinite(pixels).any():
            raise GeometryError("Slice has no valid pixels")
        slices.append(pixels)
    volume = np.stack(slices).astype(np.float32)
    return volume, affine, len(paths), float(np.isfinite(volume).mean())


def canonical_resample_lps(volume, affine):
    shape = np.asarray(volume.shape, dtype=int)
    validity = np.isfinite(volume).astype(np.float32)
    signal = np.where(validity > 0, np.nan_to_num(volume), 0.0).astype(np.float32)
    corners = np.asarray(list(product(*[(0, int(size - 1)) for size in shape])), dtype=np.float64)
    lps = (affine @ np.concatenate([corners, np.ones((8, 1))], axis=1).T).T[:, :3]
    low, high = lps.min(axis=0), lps.max(axis=0)
    output_xyz = np.ceil((high - low) / TARGET_SPACING_XYZ_MM).astype(int) + 1
    out_shape = tuple(output_xyz[::-1])
    canonical_affine = np.eye(4, dtype=np.float64)
    canonical_affine[:3, 0] = [0, 0, TARGET_SPACING_XYZ_MM[2]]
    canonical_affine[:3, 1] = [0, TARGET_SPACING_XYZ_MM[1], 0]
    canonical_affine[:3, 2] = [TARGET_SPACING_XYZ_MM[0], 0, 0]
    canonical_affine[:3, 3] = low
    mapping = np.linalg.inv(affine) @ canonical_affine
    kwargs = dict(matrix=mapping[:3, :3], offset=mapping[:3, 3], output_shape=out_shape,
                  order=1, mode="constant", cval=0.0, prefilter=False)
    numerator = affine_transform(signal, **kwargs).astype(np.float32)
    support = np.clip(affine_transform(validity, **kwargs).astype(np.float32), 0, 1)
    canonical = np.full(out_shape, np.nan, dtype=np.float32)
    good = support > 1e-6
    canonical[good] = numerator[good] / support[good]
    return canonical, support


def supported_window(volume, support, lower=20, upper=99, threshold=0.5):
    tensor, coverage = torch.from_numpy(volume), torch.from_numpy(support)
    values = tensor[torch.isfinite(tensor) & (coverage >= threshold)]
    if values.numel() == 0:
        raise GeometryError("No supported voxels for intensity window")
    if values.numel() > 1_000_000:
        indices = torch.linspace(0, values.numel()-1, 1_000_000, dtype=torch.float64).round().long()
        values = values.index_select(0, indices)
    lo, hi = torch.quantile(values, .20), torch.quantile(values, .99)
    if not torch.isfinite(lo) or not torch.isfinite(hi) or hi <= lo:
        raise GeometryError("Degenerate supported intensity window")
    safe = torch.where(torch.isfinite(tensor), tensor, lo)
    normalized = ((safe - lo) / (hi - lo)).clamp(0, 1)
    return torch.where(coverage > 1e-6, normalized, 0.0)


def center_pad_or_crop(tensor, target_shape):
    output = torch.zeros(tuple(target_shape), dtype=tensor.dtype)
    src, dst = [], []
    for current, target in zip(tensor.shape, target_shape):
        if current <= target:
            start = (target-current)//2
            src.append(slice(0,current)); dst.append(slice(start,start+current))
        else:
            start = (current-target)//2
            src.append(slice(start,start+target)); dst.append(slice(0,target))
    output[tuple(dst)] = tensor[tuple(src)]
    return output


def prepare_model_input(root, study, series):
    native, affine, slice_count, finite_fraction = reconstruct_series(root, study, series)
    canonical, support = canonical_resample_lps(native, affine)
    windowed = supported_window(canonical, support)
    image = center_pad_or_crop(windowed, TARGET_SHAPE_ZYX).contiguous()
    support = center_pad_or_crop(torch.from_numpy(support), TARGET_SHAPE_ZYX).clamp(0, 1)
    support_grid = F.adaptive_avg_pool3d(support[None,None].float(), PATCH_GRID).half()
    model_input = image[None,None].expand(1,3,*image.shape).contiguous()
    return model_input, support_grid, slice_count, finite_fraction


def plane_id(value):
    return {"axial":1,"coronal":2,"sagittal":3,"oblique":4}.get(str(value).strip().lower(),0)


def bool_id(value):
    value = str(value).strip().lower()
    if value in {"true","1","yes","y"}: return 2
    if value in {"false","0","no","n"}: return 1
    return 0


def build_gold_split(data):
    split_path = next(iter(ROOT.rglob("report_gold_split.csv")), None)
    if split_path is None:
        raise FileNotFoundError("Attach report-label-audit kernel source with report_gold_split.csv")
    split = pd.read_csv(split_path, dtype={"StudyInstanceUID":str})
    eval_ids = set(split.loc[split["split"].str.lower()=="evaluation", "StudyInstanceUID"].astype(str))
    if len(eval_ids) != 29:
        raise RuntimeError(f"Expected 29 locked gold evaluation studies; found {len(eval_ids)}")
    labels = pd.read_csv(data / "train.csv", low_memory=False, dtype={"StudyInstanceUID":str})
    if not set(TARGETS).issubset(labels.columns):
        raise RuntimeError("Official train.csv is missing gold target columns")
    labels = labels[labels.StudyInstanceUID.astype(str).isin(eval_ids)].set_index("StudyInstanceUID")
    if set(labels.index.astype(str)) != eval_ids:
        raise RuntimeError("Gold split and official labels do not match exactly")
    y = labels[TARGETS].to_numpy(dtype=np.float32)
    if y.shape != (29,12) or not np.isfinite(y).all() or not np.isin(y,[0,1]).all():
        raise RuntimeError(f"Gold evaluation labels are not a complete binary matrix: {y.shape}")
    # Keep study IDs in memory for alignment only; never print or export them.
    return sorted(eval_ids), {sid: labels.loc[sid, TARGETS].to_numpy(dtype=np.float32) for sid in eval_ids}


def capture_token_record(pillar, visual, layout, train_series_root, row):
    tokens_state = {"tokens": [], "enabled": True}
    def capture(_module, inputs):
        if tokens_state["enabled"]:
            tokens_state["tokens"].append(inputs[0].detach().transpose(1,2).cpu().half())
    hook = visual.maxpool.register_forward_pre_hook(capture)
    image, support_grid, slice_count, finite_fraction = prepare_model_input(
        train_series_root, str(row.StudyInstanceUID), str(row.SeriesInstanceUID))
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        _ = pillar.extract_vision_feats({MODEL_INPUT_KEY:image.to(DEVICE,dtype=torch.float16)})
    tokens_state["enabled"] = False
    hook.remove()
    captured = tokens_state["tokens"]
    if len(captured) != len(layout) or len(captured) != 3:
        raise RuntimeError(f"Expected 3 contextual scales, captured {len(captured)}")
    if tuple(PATCH_GRID) != (32,32,32):
        raise RuntimeError("Unexpected Pillar-0 patch grid")
    fine = captured[0].float().reshape(1,*PATCH_GRID,FEATURE_DIM).permute(0,4,1,2,3).contiguous()
    fine = F.avg_pool3d(fine,kernel_size=4,stride=4)[0].permute(1,2,3,0).reshape(512,FEATURE_DIM).cpu()
    native8, native2 = captured[1][0].float().cpu(), captured[2][0].float().cpu()
    if native8.shape != (512,FEATURE_DIM) or native2.shape != (8,FEATURE_DIM):
        raise RuntimeError("Unexpected contextual token scale shapes")
    quantized, scales = [], []
    for values in (fine,native8,native2):
        channel = values.abs().amax(dim=0)/127.0
        channel = torch.where(channel>0,channel,torch.ones_like(channel))
        q = torch.round(values/channel).clamp(-127,127).to(torch.int8)
        quantized.append(q.numpy())
        scales.append(channel.numpy().astype(np.float32))
    support8 = F.avg_pool3d(support_grid.float(),kernel_size=4,stride=4)[0,0].reshape(512)
    return {
        "fine8":quantized[0],"native8":quantized[1],"native2":quantized[2],
        "scales":np.stack(scales),"support8":torch.round(support8.clamp(0,1)*255).byte().numpy(),
        "plane":plane_id(row.Anatomical_Plane),"fluid":bool_id(row.Fluid_Sensitive),
        "fat":bool_id(row.Fat_Suppression),"slice_count":slice_count,
        "finite_fraction":finite_fraction,
    }


def partition_8_to_2(grid):
    trailing=grid.shape[3:]
    shaped=grid.reshape(2,4,2,4,2,4,*trailing)
    axes=(0,2,4,1,3,5,*range(6,6+len(trailing)))
    return shaped.transpose(axes).reshape(8,64,*trailing)


def pack_study(records, series_rows):
    coords8=grid_coordinates(8).numpy()
    coords2=grid_coordinates(2).numpy()
    tokens=[]; scales=[]; xyz=[]; protocol=[]; support=[]
    for record,row in zip(records,series_rows):
        fine=record["fine8"].astype(np.float32)*record["scales"][0]
        middle=record["native8"].astype(np.float32)*record["scales"][1]
        coarse=record["native2"].astype(np.float32)*record["scales"][2]
        fr=partition_8_to_2(fine.reshape(8,8,8,FEATURE_DIM))
        mr=partition_8_to_2(middle.reshape(8,8,8,FEATURE_DIM))
        cr=coarse.reshape(2,2,2,FEATURE_DIM).reshape(8,FEATURE_DIM)
        xr=partition_8_to_2(coords8.reshape(8,8,8,3))
        sr=partition_8_to_2(record["support8"].astype(np.float32).reshape(8,8,8)/255.0)
        metadata=np.asarray([record["plane"],record["fluid"],record["fat"]],dtype=np.int64)
        pr=np.repeat(np.repeat(metadata[None,None,:],8,axis=0),129,axis=1)
        support.append(np.concatenate([sr,sr,sr.mean(axis=1)[:,None]],axis=1))
        xyz.append(np.concatenate([xr,xr,coords2[:,None,:]],axis=1))
        tokens.append(np.concatenate([partition_8_to_2(fine.reshape(8,8,8,FEATURE_DIM)),
                                     partition_8_to_2(middle.reshape(8,8,8,FEATURE_DIM)),
                                     cr[:,None,:]],axis=1))
        scales.append(np.concatenate([np.zeros((8,64),np.int64),np.ones((8,64),np.int64),
                                      np.full((8,1),2,np.int64)],axis=1))
        protocol.append(pr)
    return {
        "tokens":torch.from_numpy(np.concatenate(tokens).astype(np.float16)),
        "scale_ids":torch.from_numpy(np.concatenate(scales)),
        "xyz":torch.from_numpy(np.concatenate(xyz).astype(np.float16)),
        "protocol_ids":torch.from_numpy(np.concatenate(protocol)),
        "support":torch.from_numpy(np.concatenate(support).astype(np.float16)),
    }


class GoldStudies(Dataset):
    def __init__(self, ids, by_study): self.ids,self.by_study=ids,by_study
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        sid=self.ids[i]
        return {"study_id":sid,**pack_study(self.by_study[sid][0],self.by_study[sid][1])}


def collate(items):
    b=len(items); max_regions=max(x["tokens"].shape[0] for x in items); t=129
    tokens=torch.zeros((b,max_regions,t,FEATURE_DIM),dtype=torch.float16)
    scales=torch.zeros((b,max_regions,t),dtype=torch.long)
    xyz=torch.zeros((b,max_regions,t,3),dtype=torch.float16)
    protocol=torch.zeros((b,max_regions,t,3),dtype=torch.long)
    support=torch.zeros((b,max_regions,t),dtype=torch.float16)
    padding=torch.ones((b,max_regions),dtype=torch.bool)
    for i,item in enumerate(items):
        n=item["tokens"].shape[0]
        for key,array in (("tokens",tokens),("scale_ids",scales),("xyz",xyz),
                          ("protocol_ids",protocol),("support",support)):
            array[i,:n]=item[key]
        padding[i,:n]=False
    return tokens,scales,xyz,protocol,support,padding


def checkpoint_path(artifacts: Path, fold: int) -> Path:
    found=list(artifacts.rglob(f"fold_{fold}_artifacts.zip"))
    if len(found)==1:
        target=WORK/f"fold_{fold}_artifacts"; target.mkdir(exist_ok=True)
        with zipfile.ZipFile(found[0]) as z: z.extractall(target)
        path=target/"checkpoint_best.pt"
        if path.is_file(): return path
    found=[p for p in artifacts.rglob("checkpoint_best.pt")
           if f"fold_{fold}" in str(p)]
    if len(found)!=1: raise FileNotFoundError(f"Expected one best checkpoint for fold {fold}; found {found}")
    return found[0]


def main():
    if not torch.cuda.is_available(): raise RuntimeError("Enable a Kaggle GPU (T4)")
    torch.set_num_threads(4)
    # All series use the same fixed 3D input shape; let cuDNN select an efficient
    # convolution plan once instead of flushing its allocator after every scan.
    torch.backends.cudnn.benchmark = True
    data=locate_data(); train_root=data/"train_series"
    eval_ids,labels=build_gold_split(data)
    series=pd.read_csv(data/"train_series.csv",dtype=str)
    series=series[series.StudyInstanceUID.astype(str).isin(set(eval_ids))].copy()
    series=series.sort_values(["StudyInstanceUID","SeriesInstanceUID"],kind="stable")
    by_study={sid:([],[]) for sid in eval_ids}
    for row in series.itertuples(index=False): by_study[str(row.StudyInstanceUID)][1].append(row)
    if any(not rows for _,rows in by_study.values()): raise RuntimeError("A gold study has no series")
    bundle=locate_bundle(); pillar=load_pillar(bundle); visual=pillar.model.visual
    cfg=visual.model_config["modalities"][MODEL_INPUT_KEY]
    layout=visual.prepare_multiscale_layout(img_size=cfg["image_size"],merge_ratio=cfg["merge_ratio"],
        local2global=cfg["local2global"],patch_size=cfg["patch_size"])
    # Generate only gold evaluation tokens in memory. Raw DICOM never leaves Kaggle.
    for n,row in enumerate(series.itertuples(index=False),1):
        token=capture_token_record(pillar,visual,layout,train_root,row)
        by_study[str(row.StudyInstanceUID)][0].append(token)
        if n%16==0 or n==len(series): print(f"Encoded gold series {n}/{len(series)}",flush=True)
        del token
        if n%32==0:
            gc.collect()
    def collate_dict(items):
        batch=collate([{k:v for k,v in item.items() if k!="study_id"} for item in items])
        return [item["study_id"] for item in items],batch
    loader=DataLoader(GoldStudies(eval_ids,by_study),batch_size=2,shuffle=False,
                      num_workers=0,collate_fn=collate_dict)
    # Match output row order explicitly rather than relying on a closure/global order.
    artifacts=dataset_dir("rsna-pillar0-regional-reader-cv-v1")
    results={}
    for fold in FOLDS_TO_SCORE:
        path=checkpoint_path(artifacts,fold)
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        model=Pillar0RegionalTargetQueryReader().to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True); model.eval()
        aligned_logits=[]; aligned_ids=[]
        with torch.inference_mode():
            for ids,batch in loader:
                values=[x.to(DEVICE,non_blocking=True) for x in batch]
                with torch.autocast(device_type="cuda",dtype=torch.float16): pred=model(*values)
                aligned_ids.extend(ids); aligned_logits.append(pred.float().cpu().numpy())
        logits=np.concatenate(aligned_logits)
        truth=np.stack([labels[sid] for sid in aligned_ids])
        per={}
        for j,name in enumerate(TARGETS):
            per[name]=float(roc_auc_score(truth[:,j],logits[:,j])) if len(np.unique(truth[:,j]))==2 else None
        vals=[v for v in per.values() if v is not None]
        results[str(fold)]={"checkpoint_selected_epoch":int(checkpoint["best_epoch"]),
            "report_validation_macro_auc":float(checkpoint["best_raw_auc"]),
            "gold_evaluation_studies":len(aligned_ids),"gold_macro_auc":float(np.mean(vals)),
            "gold_per_target_auc":per}
        print(f"Fold {fold} one-time gold evaluation: macro AUC={results[str(fold)]['gold_macro_auc']:.6f}",flush=True)
        del model,checkpoint; gc.collect(); torch.cuda.empty_cache()
    summary={"experiment":"pillar0_regional_reader_gold_evaluation",
        "training_performed":False,"gold_split":"locked 29-study evaluation split",
        "gold_used_for_model_selection":False,"raw_dicom_transferred":False,
        "encoder_revision":EXPECTED_REVISION,"gold_series_encoded":int(len(series)),
        "scores":results,
        "note":"Use this only as a final evaluation; do not select or tune a checkpoint on these labels."}
    (WORK/"gold_auc_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("Gold evaluation summary:",json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__": main()
'''


def main() -> None:
    reader_source = MODULE.read_text(encoding="utf-8")
    # The model module has a future import; runner imports are placed after it.
    RUNNER.write_text(reader_source.rstrip() + "\n\n" + RUNNER_MAIN.lstrip(), encoding="utf-8")
    metadata = {
        "id": "tumis001/rsna-pillar-0-regional-reader-gold-evaluation",
        "title": "RSNA Pillar-0 Regional Reader Gold Evaluation",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["tumis001/rsna-pillar0-regional-reader-cv-v1"],
        "kernel_sources": [
            "tumis001/pillar0-breastmri-offline-bundle",
            "tumis001/report-label-audit",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (HERE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
