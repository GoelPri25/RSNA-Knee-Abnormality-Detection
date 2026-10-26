"""Exp MP-0: map each finding's anatomy point into the sagittal, coronal and axial series of the same knee (geometry
only, numpy; no images, no labels).

Source point per finding: MS1's crop table (run_sig_lv2/crop_probe; the gradient x activation centroid on MS1's most
attended window, label-free per LOC-0-MS1), top-1 window. Cache pixel -> DICOM pixel -> patient (LPS) coordinate, as the
d96 builder made the cache: a centred square of cpx = min(round(140 / ps_row), rows, cols) native pixels resized to
384 (cv2 INTER_AREA: native = y0 + (r + 0.5) cpx / 384 - 0.5), P = IPP + IOP[:3] ps_col col + IOP[3:] ps_row row.
Into each plane (fluid slot preferred, else the other slot of that plane; the source slice itself for the source
plane): the d96 pick of that slot nearest to P along the slice normal, P projected into it, back to cache pixels.
Output per study, finding (12), plane (sagittal, coronal, axial): study row of the slice in the d96 volume (-1 = none),
cache row / col, distance of P from that slice (mm), the slot's pick spacing (mm), inside-crop flag, first / last row
of that slice's slot (the 3-slice window stays inside its series).
points = MS1's top-1 window per finding (Exp MP-0); points4 = each of MS1's 4 windows (r50-GL v3: Baker's regions in the
sagittal series).
Usage: python3 -I scripts/multiplane_points.py <project root>  -> run_sig_lv2/crop_probe/multiplane_points.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

IMG, CROP_MM, K, MAXS = 384, 140.0, 94, 96
PLANES = ("Sagittal", "Coronal", "Axial")
SLOTS = {"Sagittal": (0, 1), "Coronal": (2, 3), "Axial": (4,)}                    # fluid slot first


def eval_centres(mask, k=K):
    """coatnet_d96_runner.centres(mask, k, train=False), clipped to [1, MAXS - 2] as the runner"""
    valid = np.where(mask)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([max(1, min((lo + hi) // 2, len(mask) - 2))])
    return np.clip(cs[np.linspace(0, len(cs) - 1, k).round().astype(int)], 1, MAXS - 2)


class Slice:
    """one cached slice: cache pixel <-> patient coordinate"""

    def __init__(self, g, rows, cols):
        self.ipp, self.iop = np.array(g["ipp"], float), np.array(g["iop"], float)
        self.psr, self.psc = float(g["ps"][0]), float(g["ps"][1])
        self.u, self.v = self.iop[:3], self.iop[3:]                                  # +col, +row directions
        self.n = np.cross(self.u, self.v)
        self.cpx = min(int(round(CROP_MM / max(self.psr, 1e-3))), rows, cols)
        self.y0, self.x0 = (rows - self.cpx) // 2, (cols - self.cpx) // 2
        self.f = self.cpx / IMG                                                      # native px per cache px

    def to_patient(self, r, c):
        row = self.y0 + (r + 0.5) * self.f - 0.5; col = self.x0 + (c + 0.5) * self.f - 0.5
        return self.ipp + self.u * self.psc * col + self.v * self.psr * row

    def from_patient(self, P):
        d = P - self.ipp
        col, row = d @ self.u / self.psc, d @ self.v / self.psr
        return (row - self.y0 + 0.5) / self.f - 0.5, (col - self.x0 + 0.5) / self.f - 0.5, float(d @ self.n)


def study_slices(m, fov):
    """[(slot, plane, row in study volume, Slice)] for every pick with geometry"""
    out, start = [], 0
    for j, s in enumerate(m["slots"]):
        assert s["row_start"] == start, (m["study_id"], j, s["row_start"], start)
        if s.get("series") and s.get("geometry"):
            rows, cols = fov[s["series"]]
            for p, g in enumerate(s["geometry"]):
                if g and g.get("ipp") and g.get("iop") and g.get("ps"):
                    out.append((j, s["plane"], start + p, Slice(g, rows, cols)))
        start += s["k"]
    return out


def map_point(sl, slot_rows, src_row, r, c):
    """-> [3, 8]: per plane (row, cache r, cache c, |distance| mm, pick spacing mm, inside 0/1, slot first / last row);
    row -1 = none"""
    o = np.full((3, 8), np.nan); o[:, 0] = -1
    src = [x for x in sl if x[2] == src_row]
    if not src:
        return o
    sj, splane, _, S = src[0]
    P = S.to_patient(r, c)
    for q, plane in enumerate(PLANES):
        if plane == splane:
            o[q, :6] = [src_row, r, c, 0.0, np.nan, 1.0]
            cand = [x for x in sl if x[0] == sj]
        else:
            cand = []
            for j in SLOTS[plane]:
                cand = [x for x in sl if x[0] == j]
                if len(cand) >= 3:
                    break
            if len(cand) < 3:
                continue
            d = np.array([x[3].from_patient(P)[2] for x in cand])
            i = int(np.argmin(np.abs(d)))
            rr, cc, dist = cand[i][3].from_patient(P)
            o[q, :6] = [cand[i][2], rr, cc, abs(dist), np.nan, float(0 <= rr < IMG and 0 <= cc < IMG)]
        pos = np.sort([x[3].ipp @ cand[0][3].n for x in cand])
        o[q, 4] = float(np.median(np.diff(pos))) if len(pos) > 1 else np.nan
        o[q, 6:] = slot_rows[cand[0][0]]
    return o


def main():
    R = Path(sys.argv[1]); C = R / "repository/data/coatnet_d96_fold0"
    import pandas as pd
    fv = pd.read_csv(R / "repository/data/external/rsna-knee-crop-fov-check/series_fov.csv").drop_duplicates("series")
    fov = dict(zip(fv["series"], zip(fv["rows"].astype(int), fv["cols"].astype(int))))
    man = {}
    for f in sorted((C / "manifests").glob("manifest_*.jsonl")):
        for line in f.read_text().splitlines():
            m = json.loads(line); man[m["study_id"]] = m
    ids, pts, pts4, srcp = [], [], [], []
    for p in sorted((C / "run_sig_lv2/crop_probe").glob("cp_*.npz")):
        z = np.load(p)
        for s, cen, at in zip(z["ids"], z["centres"], z["a_top"]):
            s = str(s); m = man[s]; sl = study_slices(m, fov)
            win = eval_centres(np.array(m["mask"], bool))
            o = np.full((12, 3, 8), np.nan); o[..., 0] = -1; sp = np.full(12, -1)
            slot_rows = [(x["row_start"], x["row_start"] + x["k"] - 1) for x in m["slots"]]
            o4 = np.array([[map_point(sl, slot_rows, int(win[int(cen[k, j, 0])]), float(cen[k, j, 1]), float(cen[k, j, 2]))
                            for j in range(cen.shape[1])] for k in range(12)])                  # [12, TOP, 3, 8]
            for k in range(12):
                j = int(np.argmax(at[k]))
                row = int(win[int(cen[k, j, 0])])
                o[k] = o4[k, j]
                pl = [x[1] for x in sl if x[2] == row]
                sp[k] = PLANES.index(pl[0]) if pl else -1
            ids.append(s); pts.append(o); pts4.append(o4); srcp.append(sp)
    pts = np.array(pts, np.float32)
    out = C / "run_sig_lv2/crop_probe/multiplane_points.npz"
    np.savez(out, ids=np.array(ids), points=pts, points4=np.array(pts4, np.float32), src_plane=np.array(srcp, np.int8),
             planes=np.array(PLANES))
    T = json.loads((C / "assets/soft_labels_final_per_target_latD2.json").read_text())["targets_in_order"]
    print(f"{len(ids)} studies -> {out}")
    for k, t in enumerate(T):
        sp = np.array(srcp)[:, k]
        line = f"{t:17s} source plane sag/cor/ax {[(sp == q).mean().round(2) for q in range(3)]}"
        for q, pl in enumerate(PLANES):
            ok = (pts[:, k, q, 0] >= 0) & (pts[:, k, q, 5] == 1)
            line += f" | {pl[:3]} ok {ok.mean():.2f} dist {np.nanmedian(pts[ok, k, q, 3]):.1f}mm (spacing {np.nanmedian(pts[ok, k, q, 4]):.1f})"
        print(line)


if __name__ == "__main__":
    main()
