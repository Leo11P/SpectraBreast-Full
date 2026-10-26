# -*- coding: utf-8 -*-
"""Per-bersaglio error: where the reconstruction puts each bersaglio.

The bersagli are the blind round holes of the phantom STL (fixture.find_holes),
filled with a colour that stands out from the phantom (white on black, or the
reverse: ``polarity``). For each bersaglio:

1. expected position: the hole mouth moved to the reconstruction by the marker
   alignment (kabsch: it does not use the surface), so the bersaglio id comes
   from the STL and nothing has to be matched afterwards;
2. in every photo of the vision run where it is visible (facing the camera
   within ``max_view_angle_deg``, not hidden by the STL around it), the spot is
   looked for within ``search_radius_mm`` of the projected expected position
   and its centre is taken (intensity-weighted centroid);
3. a ray from the camera through that pixel hits the reconstructed mesh: the
   point where the reconstruction puts the bersaglio (what registration does
   with an HSI pixel);
4. the median over the views, after dropping the views farther than
   ``outlier_mm`` from it (e.g. a highlight taken for the spot), is the
   measured position; its spread over the views is reported.

The camera model is the run's (pinhole K + poses of the vision output), as
the reconstruction was made with it. Units: mm, output frame of vision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import geometry as G
from .fixture import Holes

#: Thumbnails per bersaglio in fiducials.png (the most frontal views).
SHEET_VIEWS = 6
SHEET_CELL = 112


@dataclass
class Fiducial:
    id: int
    status: str = "not visible in any view"
    measured: Optional[np.ndarray] = None          # (3,) output frame, mm
    spread_mm: Optional[float] = None
    views: List[dict] = field(default_factory=list)
    crops: List[dict] = field(default_factory=list)  # for fiducials.png


def view_images(vision_dir: Path, meta: dict, n_views: int) -> tuple[Optional[List[Path]], str]:
    """Photos of the views in camera_poses_output_frame.npy (same order):
    inputs_exif_canonical/, minus the views vision rejected."""
    d = Path(vision_dir) / "inputs_exif_canonical"
    files = sorted(p for p in d.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    if not files:
        return None, f"no photos in {d} (the vision run on this machine has none: run the validation where vision ran)"
    kept = (meta.get("view_rejection") or {}).get("kept_original_indices")
    if kept is not None and len(kept) == n_views and max(kept) < len(files):
        files = [files[int(i)] for i in kept]
    if len(files) != n_views:
        return None, f"{len(files)} photos in {d} but {n_views} camera poses: cannot pair them"
    return files, ""


def detect_spot(gray: np.ndarray, uv: Sequence[float], r_search: float, r_dot: float,
                cos_view: float, polarity: str, min_contrast: float) -> Dict[str, Any]:
    """The spot nearest to ``uv`` within ``r_search`` px: a connected region
    above half its peak contrast, of plausible size (disc of radius ``r_dot``
    px seen at ``cos_view``), not touching the window border."""
    import cv2

    H, W = gray.shape
    h = int(math.ceil(r_search + 2.0 * r_dot + 3.0))
    u0, v0 = int(round(uv[0])), int(round(uv[1]))
    if u0 - h < 0 or v0 - h < 0 or u0 + h >= W or v0 + h >= H:
        return {"status": "near the image border"}
    win = gray[v0 - h:v0 + h + 1, u0 - h:u0 + h + 1].astype(np.float64)
    if polarity == "dark":
        win = 255.0 - win
    rows, cols = np.mgrid[0:2 * h + 1, 0:2 * h + 1]
    du, dv = cols + (u0 - h) - uv[0], rows + (v0 - h) - uv[1]
    disc = du * du + dv * dv <= r_search * r_search
    out: Dict[str, Any] = {"window": (u0 - h, v0 - h, 2 * h + 1)}
    bg = float(np.median(win))
    sigma = 1.4826 * float(np.median(np.abs(win - bg))) + 0.5
    peak = float(win[disc].max())
    out["contrast"] = peak - bg
    if peak - bg < max(min_contrast, 5.0 * sigma):
        out["status"] = "no spot"
        return out
    thr = bg + 0.5 * (peak - bg)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((win > thr).astype(np.uint8), connectivity=8)
    a_exp = max(math.pi * r_dot * r_dot * max(cos_view, 0.2), 1.0)
    best, why = None, "no spot"
    for k in range(1, n):
        comp = lab == k
        if not (comp & disc).any():
            continue
        x, y, w, hh, area = stats[k]
        if x == 0 or y == 0 or x + w == win.shape[1] or y + hh == win.shape[0]:
            why = "bright region larger than a bersaglio"
            continue
        if not 0.25 * a_exp <= area <= 4.0 * a_exp + 4.0:
            why = f"spot of {area} px, expected about {a_exp:.0f}"
            continue
        grow = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        wgt = np.clip(win - bg, 0.0, None) * grow
        cu = float((wgt * (cols + (u0 - h))).sum() / wgt.sum())
        cv = float((wgt * (rows + (v0 - h))).sum() / wgt.sum())
        dist = math.hypot(cu - uv[0], cv - uv[1])
        if dist > r_search:
            why = "spot outside the search radius"
            continue
        if best is None or dist < best[0]:
            best = (dist, cu, cv, int(area))
    if best is None:
        out["status"] = why
        return out
    out.update(status="found", uv=[best[1], best[2]], area_px=best[3], offset_px=best[0])
    return out


def measure(holes: Holes, *, T_pred: np.ndarray, cams: np.ndarray, K: np.ndarray,
            image_paths: Sequence[Path], mesh_v: np.ndarray, mesh_f: np.ndarray,
            occluders_cad: np.ndarray, image_wh: Sequence[int], opt, visibility_tol: float,
            keep_crops: bool = True) -> List[Fiducial]:
    """Measured position of every bersaglio (see the module docstring).
    ``T_pred``: CAD -> output frame used for the expected positions; ``cams``:
    [V, 4, 4] camera-to-world in the output frame, mm; ``mesh_v`` (mm) /
    ``mesh_f``: the reconstructed mesh."""
    import cv2

    fids = [Fiducial(id=i + 1) for i in range(len(holes))]
    if len(holes) == 0:
        return fids
    T_inv = G.inv(T_pred)
    cam_cad = np.einsum("ij,vjk->vik", T_inv, cams)
    min_cos = math.cos(math.radians(opt.max_view_angle_deg))
    seen = G.visible_per_view(holes.mouth, holes.axis, occluders_cad, cam_cad, K, image_wh,
                              tol=visibility_tol, min_cos=min_cos)
    P = G.apply(T_pred, holes.mouth)                       # expected, output frame
    N = G.rotate(T_pred, holes.axis)
    tri_all = mesh_v[mesh_f]
    cen = tri_all.mean(axis=1)
    local = [tri_all[np.linalg.norm(cen - p, axis=1) < opt.search_radius_mm + 5.0] for p in P]

    hits: List[List[tuple]] = [[] for _ in fids]
    for v in np.flatnonzero(seen.any(axis=1)):
        img = cv2.imread(str(image_paths[v]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        R, C, Kv = cams[v][:3, :3], cams[v][:3, 3], K[v]
        f_px = 0.5 * (Kv[0, 0] + Kv[1, 1])
        Kinv = np.linalg.inv(Kv)
        for i in np.flatnonzero(seen[v]):
            Xc = R.T @ (P[i] - C)
            uv = (Kv[0, 0] * Xc[0] / Xc[2] + Kv[0, 2], Kv[1, 1] * Xc[1] / Xc[2] + Kv[1, 2])
            to_cam = C - P[i]
            cos_view = float(N[i] @ to_cam / np.linalg.norm(to_cam))
            r_search = opt.search_radius_mm * f_px / Xc[2]
            r_dot = 0.5 * float(holes.diameter[i]) * f_px / Xc[2]
            det = detect_spot(img, uv, r_search, r_dot, cos_view, opt.polarity, opt.min_contrast)
            rec = {"view": int(v), "image": Path(image_paths[v]).name,
                   "angle_deg": round(math.degrees(math.acos(min(1.0, cos_view))), 1),
                   "expected_px": [round(uv[0], 2), round(uv[1], 2)], "status": det["status"]}
            if det["status"] == "found":
                rec["found_px"] = [round(x, 2) for x in det["uv"]]
                ray = R @ (Kinv @ np.array([det["uv"][0], det["uv"][1], 1.0]))
                ray /= np.linalg.norm(ray)
                t = G.ray_first_hit(C, ray, local[i]) if len(local[i]) else None
                if t is None:
                    rec["status"] = "ray misses the reconstructed mesh"
                else:
                    hits[i].append((int(v), C + t * ray))
            fids[i].views.append(rec)
            if keep_crops and "window" in det:
                x0, y0, s = det["window"]
                fids[i].crops.append({"img": img[y0:y0 + s, x0:x0 + s].copy(), "x0": x0, "y0": y0,
                                      "r": r_dot, "rec": rec, "angle": rec["angle_deg"]})

    for i, fd in enumerate(fids):
        if not fd.views:
            continue
        pts = np.array([h[1] for h in hits[i]]) if hits[i] else np.zeros((0, 3))
        if len(pts) == 0:
            fd.status = f"not found in {len(fd.views)} views"
            continue
        med = np.median(pts, axis=0)
        inl = np.linalg.norm(pts - med, axis=1) <= opt.outlier_mm
        if inl.sum() >= 1:
            med = np.median(pts[inl], axis=0)
            inl = np.linalg.norm(pts - med, axis=1) <= opt.outlier_mm
        for (v, _), ok in zip(hits[i], inl):
            for rec in fd.views:
                if rec["view"] == v and rec["status"] == "found":
                    rec["status"] = "used" if ok else f"outlier (> {opt.outlier_mm:g} mm from the median)"
        if inl.sum() < opt.min_views:
            fd.status = f"{int(inl.sum())} consistent views of {len(pts)} found (min {opt.min_views})"
            continue
        fd.measured = med
        fd.spread_mm = float(np.sqrt(np.mean(np.sum((pts[inl] - med) ** 2, axis=1))))
        fd.status = "ok"
    return fids


def errors(fids: List[Fiducial], holes: Holes, T: np.ndarray) -> tuple[List[Optional[dict]], dict]:
    """Measured - expected (expected = hole mouth moved by ``T``): norm, along
    the outward normal (signed, + = outside the STL, as accuracy) and in the
    surface plane. Returns (per bersaglio, stats over the bersagli found)."""
    per: List[Optional[dict]] = []
    nrm, nor, inp = [], [], []
    P, N = G.apply(T, holes.mouth), G.rotate(T, holes.axis)
    for i, fd in enumerate(fids):
        if fd.measured is None:
            per.append(None)
            continue
        e = fd.measured - P[i]
        a = float(e @ N[i])
        b = float(np.linalg.norm(e - a * N[i]))
        per.append({"norm_mm": float(np.linalg.norm(e)), "normal_mm": a, "in_plane_mm": b,
                    "xyz_mm": [float(x) for x in e]})
        nrm.append(np.linalg.norm(e))
        nor.append(a)
        inp.append(b)
    if not nrm:
        return per, {"n_found": 0, "n": len(fids)}
    nrm, nor, inp = np.array(nrm), np.array(nor), np.array(inp)
    stats = {"n_found": int(len(nrm)), "n": len(fids),
             "norm_rmse_mm": float(np.sqrt(np.mean(nrm ** 2))), "norm_median_mm": float(np.median(nrm)),
             "norm_max_mm": float(nrm.max()),
             "normal_rmse_mm": float(np.sqrt(np.mean(nor ** 2))), "normal_mean_signed_mm": float(nor.mean()),
             "in_plane_rmse_mm": float(np.sqrt(np.mean(inp ** 2)))}
    return per, stats


def _cell_xy(p, crop: dict, k: float) -> tuple:
    """Image pixel -> position in a thumbnail scaled by k (pixel centres at +0.5)."""
    return (int(round((p[0] - crop["x0"] + 0.5) * k)), int(round((p[1] - crop["y0"] + 0.5) * k)))


def write_sheet(path: Path, fids: List[Fiducial]) -> Optional[str]:
    """fiducials.png: per bersaglio (rows) the most frontal views (columns),
    cyan cross = expected position, circle = spot found (green used, red dropped)."""
    import cv2

    rows = [fd for fd in fids if fd.crops]
    if not rows:
        return None
    S, lw = SHEET_CELL, 150
    sheet = np.full((len(rows) * (S + 4) + 4, lw + SHEET_VIEWS * (S + 4) + 4, 3), 40, np.uint8)
    for r, fd in enumerate(rows):
        y = 4 + r * (S + 4)
        txt = fd.status if fd.status == "ok" else fd.status[:22]
        cv2.putText(sheet, f"#{fd.id}", (6, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(sheet, txt, (6, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (120, 230, 120) if fd.status == "ok" else (90, 90, 255), 1)
        if fd.spread_mm is not None:
            cv2.putText(sheet, f"spread {fd.spread_mm:.2f} mm", (6, y + 68), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (220, 220, 220), 1)
        for c, cr in enumerate(sorted(fd.crops, key=lambda z: z["angle"])[:SHEET_VIEWS]):
            k = S / cr["img"].shape[0]
            cell = cv2.cvtColor(cv2.resize(cr["img"], (S, S), interpolation=cv2.INTER_NEAREST),
                                cv2.COLOR_GRAY2BGR)
            rec = cr["rec"]
            r = int(round(cr["r"] * k)) + 4                 # just outside the spot
            ex, ey = _cell_xy(rec["expected_px"], cr, k)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # cross with a gap: the spot stays visible
                cv2.line(cell, (ex + dx * r, ey + dy * r), (ex + dx * (r + 9), ey + dy * (r + 9)),
                         (255, 255, 0), 2)
            if "found_px" in rec:
                cv2.circle(cell, _cell_xy(rec["found_px"], cr, k), r,
                           (0, 220, 0) if rec["status"] == "used" else (0, 0, 255), 2)
            cv2.putText(cell, f"v{rec['view']} {rec['angle_deg']:.0f}d", (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 255, 255), 1)
            x = lw + 4 + c * (S + 4)
            sheet[y:y + S, x:x + S] = cell
    cv2.imwrite(str(path), sheet)
    return Path(path).name


def to_json(fids: List[Fiducial]) -> List[dict]:
    return [{"id": fd.id, "status": fd.status,
             "measured_output_frame_mm": None if fd.measured is None else [round(float(x), 4) for x in fd.measured],
             "spread_mm": None if fd.spread_mm is None else round(fd.spread_mm, 4),
             "n_views_tried": len(fd.views),
             "n_views_used": sum(1 for r in fd.views if r["status"] == "used"),
             "views": fd.views} for fd in fids]
