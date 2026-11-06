# -*- coding: utf-8 -*-
"""Phase 2 — the HSI -> 3D registration, on one sample (Protocollo, Fase 2).

Separated from the reconstruction (phase 1): the reference is the marker
corners as VISION reconstructed them (aruco_markers_3d.json, output frame),
not the CAD, so the reconstruction error does not enter these numbers.

1. Same inputs and steps as the registration stage: the HSI image (or the
   LiveView PNG in roi mode), the vision surface (registration.source: mesh |
   cloud), the orthographic top-view render at resolution_reg (homography) and
   at resolution (3D lookup), ArUco on the HSI image, homography HSI -> render.
2. Leave-one-marker-out (hold-out, the protocol's TRE): for every marker seen
   in both, the homography is fitted on the corners of the OTHER markers and
   evaluated on the 4 corners of the one left out:
     2D   reprojection error in HSI pixels (render corners mapped back by H^-1
          vs the detected ones) and in mm (HSI corners mapped by H vs the
          render corners: the render is metric, mm = px x resolution_reg)
     3D   HSI corner -> H -> render -> (X, Y, Z) on the surface vs the JSON
          corner: norm, in plane (xy) and in depth (z)
   FRE: the same errors with the homography fitted on ALL markers (what the
   registration stage reports), for comparison.
3. Outputs in RESULTS/<date>/<sample>/validation/registration/: metrics.json,
   overlay.png (detected corners vs the hold-out predictions), and a copy of
   the registration stage's Excel reports (<sample>_registration_errors.xlsx,
   <sample>_sweep_resolution_summary.xlsx) from RESULTS/<date>/<sample>/registration/.

Limits: every marker lies on the marker plane (z = 0), so the z error is the
surface height at the markers vs the reconstructed corners; the error above
the plane (parallax) needs targets at known heights (not measured here).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std()), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def _case_info(vcfg, sample: str) -> dict:
    m = re.match(vcfg.sample_pattern, sample)
    g = m.groupdict() if m else {}
    return {"phantom": vcfg.cases.get(sample) or g.get("phantom"),
            "pose": g.get("pose"), "rep": g.get("rep")}


def resolved_paths(cfg, vision_dir: Path):
    """registration.paths filled as the registration stage would, on this vision run."""
    from ..orchestrator import resolve_unified_paths

    return resolve_unified_paths(cfg, vision_dir=vision_dir).registration


def is_roi(rcfg) -> bool:
    return rcfg.mode == "roi" or (rcfg.mode == "sweep" and rcfg.sweep.roi_mode)


def missing_inputs(cfg, vision_dir: Path) -> List[str]:
    """Inputs of the registration validation that are not on disk."""
    r = resolved_paths(cfg, vision_dir)
    need = [r.paths.mesh, r.paths.aruco_json] + ([r.paths.liveview_png] if is_roi(r)
                                                 else [r.paths.hsi_hdr])
    return [str(p) for p in need if p is None or not Path(p).is_file()]


def _render(surface, res: float, margin: float):
    try:
        from ..registration.render_gpu import render_orthographic_topview_gpu as fn
    except Exception:                       # no torch / cupy: CPU ray casting
        from ..registration.pipeline import render_orthographic_topview as fn
    return fn(surface, resolution_mm_per_px=res, margin_mm=margin)


def _errors(H, H_pc, ids, data_img, data_reg, aruco_3d, xyz_pc, res_reg, lookup) -> dict:
    """Per-corner errors of the markers ``ids`` for one homography (HSI -> render_reg)."""
    import cv2
    from ..registration.pipeline import lookup_3d

    img = np.vstack([data_img[m] for m in ids]).astype(np.float32)
    ren = np.vstack([data_reg[m] for m in ids]).astype(np.float32)
    gt = np.vstack([aruco_3d[m] for m in ids]).astype(np.float64)
    fwd = cv2.perspectiveTransform(img.reshape(-1, 1, 2), H).reshape(-1, 2)
    back = cv2.perspectiveTransform(ren.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    pc = cv2.perspectiveTransform(img.reshape(-1, 1, 2), H_pc).reshape(-1, 2)
    xyz = np.asarray(lookup_3d(pc, xyz_pc, mode=lookup, verbose=False), dtype=np.float64)
    d = xyz - gt
    return {"marker": np.repeat(ids, 4), "corner": np.tile(np.arange(4), len(ids)),
            "px": np.linalg.norm(back - img, axis=1),
            "mm2d": np.linalg.norm(fwd - ren, axis=1) * res_reg,
            "d3": d, "pred_img": back}


REPORTS = ("{s}_registration_errors.xlsx", "{s}_sweep_resolution_summary.xlsx")


def copy_registration_reports(reg_dir, sample: str, out_dir: Path) -> dict:
    """Copies the Excel reports of the registration stage (error report, sweep
    summary) next to the validation outputs; the originals stay in registration/."""
    import shutil

    files = {}
    if reg_dir is None:
        return files
    for name in (n.format(s=sample) for n in REPORTS):
        src = Path(reg_dir) / name
        if src.is_file():
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_dir / name)
            files[name.split(f"{sample}_", 1)[1].rsplit(".", 1)[0]] = name
    if not files:
        print(f"[validation] no registration Excel report in {reg_dir} to copy")
    return files


def _summary(e: dict) -> dict:
    d = e["d3"]
    return {"reproj_px": _stats(e["px"]), "reproj_mm": _stats(e["mm2d"]),
            "err3d_mm": _stats(np.linalg.norm(d, axis=1)),
            "err_xy_mm": _stats(np.linalg.norm(d[:, :2], axis=1)),
            "err_z_abs_mm": _stats(np.abs(d[:, 2])),
            "err_z_signed_mean_mm": float(np.nanmean(d[:, 2])) if np.isfinite(d[:, 2]).any() else None,
            "n_no_surface": int((~np.isfinite(d[:, 0])).sum())}


def _rows(e: dict, which: str) -> List[dict]:
    return [{"set": which, "marker": int(m), "corner": int(c), "reproj_px": float(p),
             "reproj_mm": float(q), "dx_mm": float(v[0]), "dy_mm": float(v[1]), "dz_mm": float(v[2]),
             "err3d_mm": float(np.linalg.norm(v)), "err_xy_mm": float(np.linalg.norm(v[:2]))}
            for m, c, p, q, v in zip(e["marker"], e["corner"], e["px"], e["mm2d"], e["d3"])]


def _overlay(path: Path, img, data_img, pred: Dict[int, np.ndarray]) -> Optional[str]:
    import cv2

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for mid, c in data_img.items():
        cv2.polylines(vis, [np.round(c).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
        cv2.putText(vis, str(mid), tuple(int(v) for v in c.mean(0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1)
    for c in pred.values():
        for p in c:
            cv2.drawMarker(vis, (int(round(p[0])), int(round(p[1]))), (0, 0, 255),
                           cv2.MARKER_CROSS, 6, 1)
    cv2.imwrite(str(path), vis)
    return path.name


def validate_registration(cfg, vision_dir: Path, out_dir: Path) -> Dict[str, Any]:
    from ..registration import pipeline as P
    from ..registration.runner import _ARUCO_DICT_MAP

    t0 = time.time()
    vcfg = cfg.validation
    opt = vcfg.registration
    sample = cfg.sample_name
    rcfg = resolved_paths(cfg, vision_dir)
    det = rcfg.detection
    missing = missing_inputs(cfg, vision_dir)
    if missing:
        raise SystemExit(f"[validation] registration of {sample}: missing {missing}")
    roi = is_roi(rcfg)
    res_reg = float(rcfg.render.resolution_reg_mm_per_px)
    res_pc = float(rcfg.render.resolution_mm_per_px)
    print(f"[validation] registration — sample {sample}, surface {rcfg.paths.mesh} "
          f"(source {rcfg.source}), {'LiveView PNG' if roi else 'HSI'} "
          f"{rcfg.paths.liveview_png if roi else rcfg.paths.hsi_hdr}")

    # ------------------------------------------------- image + ArUco (2D) --
    if roi:
        from ..registration.roi_align import load_liveview_png

        img = load_liveview_png(str(rcfg.paths.liveview_png))
    else:
        cube, meta = P.load_envi(str(rcfg.paths.hsi_hdr))
        img = P.extract_2d_from_hsi(cube, meta, method=det.hsi_extraction_method,
                                    equalize_method=det.equalize_method)
        del cube
    dict_cv = _ARUCO_DICT_MAP.get((det.aruco_dict or "4x4_50").upper())
    data_img, _ = P.detect_aruco(img, aruco_dict_type=dict_cv, use_subpix=det.use_subpix,
                                 subpix_winsize=det.subpix_winsize, tune_detector=det.tune_detector)

    # -------------------------------------------------- surface + renders --
    aruco_3d, _ = P.load_aruco_3d_json(str(rcfg.paths.aruco_json), scale_m_to_mm=True)
    surface = P.load_mesh(str(rcfg.paths.mesh), scale_m_to_mm=True)
    _, _, xyz_reg, origin_reg, res_reg = _render(surface, res_reg, rcfg.render.margin_mm)
    if abs(res_pc - res_reg) > 1e-6:
        _, _, xyz_pc, origin_pc, res_pc = _render(surface, res_pc, rcfg.render.margin_mm)
    else:
        xyz_pc, origin_pc, res_pc = xyz_reg, origin_reg, res_reg
    del xyz_reg
    if np.abs(np.subtract(origin_reg, origin_pc)).max() > 1e-6:
        raise SystemExit("[validation] the two renders have different origins")
    data_reg = P.project_aruco3d_to_render(aruco_3d, origin_reg, res_reg)
    S = np.diag([res_reg / res_pc, res_reg / res_pc, 1.0])

    ids = sorted(set(data_img) & set(data_reg) & set(aruco_3d))
    print(f"[validation] markers: HSI {sorted(data_img)}, JSON {sorted(aruco_3d)}, common {ids}")
    if len(ids) < opt.min_fit_markers + 1:
        raise SystemExit(f"[validation] {len(ids)} markers in both the HSI and the JSON: the "
                         f"hold-out needs at least min_fit_markers + 1 = {opt.min_fit_markers + 1}")

    def fit(use):
        src = np.vstack([data_img[m] for m in use]).astype(np.float32)
        dst = np.vstack([data_reg[m] for m in use]).astype(np.float32)
        H, mask = P.compute_homography(src, dst, tag=f" fit {use}")
        return H, int(mask.ravel().sum()), len(src)

    # ------------------------------------------------- hold-out + all-fit --
    loo, per_marker, pred = [], {}, {}
    for m in ids:
        H, inl, n = fit([i for i in ids if i != m])
        e = _errors(H, S @ H, [m], data_img, data_reg, aruco_3d, xyz_pc, res_reg, opt.lookup)
        loo.append(e)
        pred[m] = e["pred_img"]
        per_marker[str(m)] = {"fit_inliers": f"{inl}/{n}", **{
            k: v.get("mean") if isinstance(v, dict) else v for k, v in _summary(e).items()}}
    tre = {k: np.concatenate([e[k] for e in loo]) for k in ("marker", "corner", "px", "mm2d", "d3")}
    H, inl, n = fit(ids)
    fre = _errors(H, S @ H, ids, data_img, data_reg, aruco_3d, xyz_pc, res_reg, opt.lookup)

    s_tre, s_fre = _summary(tre), _summary(fre)
    thr = opt.thresholds
    checks = {
        f"reproj_px_mean < {thr.reproj_px:g}": bool(s_tre["reproj_px"].get("mean", np.inf) < thr.reproj_px),
        f"tre_mm_mean < {thr.tre_mm:g}": bool(s_tre["err3d_mm"].get("mean", np.inf) < thr.tre_mm),
    }

    # ------------------------------------------------------------ outputs --
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"metrics": "metrics.json"}
    if opt.save_overlay:
        files["overlay"] = _overlay(out_dir / "overlay.png", img, data_img, pred)
    files.update(copy_registration_reports(rcfg.paths.output_dir, sample, out_dir))
    result = {
        "sample": sample, **_case_info(vcfg, sample),
        "date": Path(out_dir).parents[2].name if len(Path(out_dir).parents) > 2 else None,
        "vision_dir": str(vision_dir), "surface": str(rcfg.paths.mesh), "source": rcfg.source,
        "image": str(rcfg.paths.liveview_png if roi else rcfg.paths.hsi_hdr),
        "mode": "roi (LiveView PNG)" if roi else "single (HSI)",
        "reference": "marker corners reconstructed by vision (aruco_markers_3d.json): phase 1 excluded",
        "units": "mm, px", "resolution_reg_mm_per_px": res_reg, "resolution_mm_per_px": res_pc,
        "lookup": opt.lookup,
        "markers": {"hsi": sorted(int(i) for i in data_img), "json": sorted(int(i) for i in aruco_3d),
                    "used": [int(i) for i in ids]},
        "tre": s_tre, "fre": {**s_fre, "fit_inliers": f"{inl}/{n}"},
        "per_marker_tre": per_marker, "checks": checks,
        "corners": _rows(tre, "tre") + _rows(fre, "fre"),
        "note": "TRE = leave-one-marker-out (homography fitted on the other markers); "
                "FRE = fitted on all of them. Markers lie at z = 0: the error above the "
                "marker plane is not measured here.",
        "files": files, "elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    t = s_tre
    print(f"[validation] TRE (leave-one-marker-out, {len(ids)} markers): 3D mean "
          f"{t['err3d_mm'].get('mean', float('nan')):.3f} mm (xy {t['err_xy_mm'].get('mean', float('nan')):.3f}, "
          f"|z| {t['err_z_abs_mm'].get('mean', float('nan')):.3f}), reprojection "
          f"{t['reproj_px'].get('mean', float('nan')):.2f} px / {t['reproj_mm'].get('mean', float('nan')):.3f} mm; "
          f"FRE 3D {s_fre['err3d_mm'].get('mean', float('nan')):.3f} mm")
    print(f"[validation] checks: " + ", ".join(f"{k} {'OK' if v else 'NO'}" for k, v in checks.items()))
    print(f"[validation] outputs in {out_dir} ({result['elapsed_s']} s)")
    return result


SUMMARY_KEYS = [("tre", "err3d_mm"), ("tre", "err_xy_mm"), ("tre", "err_z_abs_mm"),
                ("tre", "reproj_px"), ("tre", "reproj_mm"), ("fre", "err3d_mm")]


def summarize(cfg) -> Path:
    """Every validation/registration/metrics.json under results_root ->
    RESULTS/_validation/registration_summary.xlsx (+ .json)."""
    rows = []
    for p in sorted(Path(cfg.results_root).glob("*/*/validation/registration/metrics.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        row = {"date": p.parents[3].name, "sample": r["sample"], "phantom": r.get("phantom"),
               "pose": r.get("pose"), "rep": r.get("rep"), "source": r.get("source"),
               "n_markers": len(r["markers"]["used"])}
        for s, k in SUMMARY_KEYS:
            row[f"{s}.{k}.mean"] = r[s][k].get("mean")
            row[f"{s}.{k}.max"] = r[s][k].get("max")
        row.update({f"check: {k}": v for k, v in r["checks"].items()})
        rows.append(row)
    if not rows:
        raise SystemExit(f"[validation] no registration validation under {cfg.results_root}"
                         "/*/*/validation/registration/: run `spectra validate -m registration` first")
    out_dir = Path(cfg.results_root) / "_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "registration_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    xlsx = out_dir / "registration_summary.xlsx"
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Cases"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    ws.append(keys)
    for r in rows:
        ws.append([r.get(k) for k in keys])
    wb.save(xlsx)
    print(f"[validation] {len(rows)} samples -> {xlsx}")
    return xlsx
