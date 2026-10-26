# -*- coding: utf-8 -*-
"""Phase 1 — the reconstruction against the phantom STL, on one sample.

1. Ground truth: marker corners from the table + the phantom STL (CAD frame, mm).
2. Alignment of the STL to the reconstruction (output frame of vision, mm):
     frame   same rule as vision on the CAD markers (origin / X from the IDs in
             the run's metadata) -> coordinates compared directly
     kabsch  rigid fit of the CAD corners on the reconstructed ones (all markers)
     icp     kabsch refined by point-to-plane ICP on the phantom region (no scale)
   Kabsch residual and leave-one-marker-out errors check the markers.
3. Metrics per alignment, in the phantom region (XY box of the STL above the
   marker plane + roi_margin_mm, or validation.phantoms.<name>.roi_mm):
     accuracy      reconstruction -> STL, signed (+ = outside the STL)
     completeness  STL -> reconstruction, on the STL seen by the cameras
     hausdorff_p95 max of the two 95th percentiles
     height        signed error vs height above the marker plane (+ linear fit)
4. Acquisition geometry: per view, distance from the phantom, elevation above
   the marker plane, azimuth and whether the phantom is in the image, plus a
   summary (variables of the protocol for the robustness analysis).
5. Fiducials: the bersagli, blind round holes of the phantom found in the STL
   (centre of the hole mouth + axis), looked for in the photos of the vision
   run and hit on the reconstructed mesh (fiducials.py); error per bersaglio
   and per alignment: norm, along the normal, in the surface plane.
6. Outputs in RESULTS/<date>/<sample>/validation/vision/: metrics.json,
   error_map.ply (+ .npy: phantom region of the mesh coloured by |error|, mm),
   fiducials.png (photo crops around each bersaglio) and validation.rrd (STL +
   reconstruction + fiducials for Rerun).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from . import fiducials as F
from . import geometry as G
from .fixture import Holes, cad_frame, find_holes, load_mesh, read_fixture


def _case_info(vcfg, sample: str) -> dict:
    """phantom / pose / rep of a sample (validation.cases, else sample_pattern)."""
    info = {"phantom": vcfg.cases.get(sample), "pose": None, "rep": None}
    m = re.match(vcfg.sample_pattern, sample)
    if m:
        g = m.groupdict()
        info["pose"], info["rep"] = g.get("pose"), g.get("rep")
        info["phantom"] = info["phantom"] or g.get("phantom")
    if not info["phantom"]:
        raise SystemExit(f"[validation] which phantom is sample {sample!r}? Add it to "
                         "validation.cases or name the sample as validation.sample_pattern "
                         "(e.g. T1_P1_R1).")
    if info["phantom"] not in vcfg.phantoms:
        raise SystemExit(f"[validation] phantom {info['phantom']!r} (sample {sample!r}) is not "
                         f"in validation.phantoms {list(vcfg.phantoms)}")
    return info


def _read_vision_run(vision_dir: Path) -> dict:
    need = ("reconstruction_metadata.json", "aruco_markers_3d.json",
            "camera_poses_output_frame.npy", "intrinsics_original_frame.npy")
    missing = [n for n in need if not (vision_dir / n).is_file()]
    if missing:
        raise SystemExit(f"[validation] {vision_dir}: missing {missing}")
    meta = json.loads((vision_dir / "reconstruction_metadata.json").read_text(encoding="utf-8"))
    frame = meta.get("frame")
    if not frame:
        raise SystemExit(f"[validation] {vision_dir} was made before the output frame was "
                         "fixed (mirrored, z_axis_points_down): re-run vision on this sample.")
    det_min = (frame.get("check") or {}).get("camera_rotation_det_min", 1.0)
    if det_min < 0:
        raise SystemExit(f"[validation] {vision_dir}: reflected output frame (det < 0)")
    aruco = json.loads((vision_dir / "aruco_markers_3d.json").read_text(encoding="utf-8"))
    corners = {int(k): np.asarray(v["corners_3d"], dtype=np.float64).reshape(4, 3) * 1000.0
               for k, v in aruco["markers"].items()}
    cams = np.load(vision_dir / "camera_poses_output_frame.npy").astype(np.float64)
    cams[:, :3, 3] *= 1000.0
    K = np.load(vision_dir / "intrinsics_original_frame.npy").astype(np.float64)
    timings = {}
    if (vision_dir / "timings.json").is_file():
        t = json.loads((vision_dir / "timings.json").read_text(encoding="utf-8"))
        timings = {"total_s": t.get("total_seconds")}
        for p in t.get("phases", []):
            if "backend" in p.get("label", "").lower():
                timings["backend_s"] = p.get("seconds")
    return {"meta": meta, "frame": frame, "corners": corners,
            "edge_mm": float(aruco.get("edge_length_m", float("nan"))) * 1000.0,
            "used_ids": [int(i) for i in aruco.get("used_marker_ids", corners)],
            "cams": cams, "K": K, "timings": timings}


def _image_size(vision_dir: Path, K: np.ndarray) -> tuple[int, int]:
    import cv2

    for p in sorted((vision_dir / "inputs_exif_canonical").glob("*")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is not None:
            return im.shape[1], im.shape[0]
    return int(round(2 * K[0, 0, 2])), int(round(2 * K[0, 1, 2]))


def _roi_box(phantom_cfg, v_cad: np.ndarray, marker_z: float, margin: float) -> list[float]:
    """XY box (CAD) of the phantom region: roi_mm, else the STL above the marker plane."""
    if phantom_cfg.roi_mm:
        return [float(x) for x in phantom_cfg.roi_mm]
    above = v_cad[v_cad[:, 2] > marker_z + 1.0]
    if len(above) == 0:
        raise SystemExit("[validation] nothing of the STL is above the marker plane: set "
                         "validation.phantoms.<name>.roi_mm")
    lo, hi = above[:, :2].min(axis=0), above[:, :2].max(axis=0)
    return [float(lo[0] - margin), float(hi[0] + margin), float(lo[1] - margin), float(hi[1] + margin)]


def in_xy(p: np.ndarray, box) -> np.ndarray:
    return (p[:, 0] >= box[0]) & (p[:, 0] <= box[1]) & (p[:, 1] >= box[2]) & (p[:, 1] <= box[3])


def _in_box(box, margin: float = 0.0):
    x0, x1, y0, y1 = box

    def keep(p):
        return ((p[:, 0] >= x0 - margin) & (p[:, 0] <= x1 + margin)
                & (p[:, 1] >= y0 - margin) & (p[:, 1] <= y1 + margin))
    return keep


def _view_geometry(cam_T: np.ndarray, K: np.ndarray, image_wh, p: np.ndarray) -> dict:
    """Where the cameras were, seen from the phantom centre ``p`` (CAD frame, mm,
    Z up): distance, elevation above the marker plane, azimuth, phantom in image."""
    W, H = image_wh
    C, R = cam_T[:, :3, 3], cam_T[:, :3, :3]
    v = C - p
    dist = np.linalg.norm(v, axis=1)
    elev = np.degrees(np.arcsin(np.clip(v[:, 2] / np.maximum(dist, 1e-9), -1, 1)))
    azim = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    Xc = np.einsum("vji,vj->vi", R, p - C)                  # R^T (p - C), per view
    z = np.where(Xc[:, 2] > 1e-9, Xc[:, 2], np.inf)
    u = K[:, 0, 0] * Xc[:, 0] / z + K[:, 0, 2]
    w = K[:, 1, 1] * Xc[:, 1] / z + K[:, 1, 2]
    sees = (Xc[:, 2] > 0) & (u >= 0) & (u < W) & (w >= 0) & (w < H)
    a = np.sort(azim[sees])
    coverage = float(360.0 - np.max(np.diff(np.r_[a, a[0] + 360.0]))) if len(a) > 1 else 0.0

    def mmm(x):
        return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())} if len(x) else {}

    return {
        "phantom_centre_cad_mm": [float(x) for x in p],
        "n_views": int(len(C)), "n_views_seeing_phantom": int(sees.sum()),
        "distance_mm": mmm(dist[sees]), "elevation_deg": mmm(elev[sees]),
        "azimuth_coverage_deg": coverage,
        "per_view": [{"view": i, "distance_mm": float(dist[i]), "elevation_deg": float(elev[i]),
                      "azimuth_deg": float(azim[i]), "sees_phantom": bool(sees[i])}
                     for i in range(len(C))],
    }


def _colors(d: np.ndarray, vmax: float) -> np.ndarray:
    import cv2

    idx = np.clip(np.abs(d) / max(vmax, 1e-9) * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(idx.reshape(-1, 1), cv2.COLORMAP_VIRIDIS)[:, 0, ::-1]
    return np.ascontiguousarray(rgb)


def validate_vision(cfg, vision_dir: Path, out_dir: Path) -> Dict[str, Any]:
    from . import data_dir

    t0 = time.time()
    vcfg, opt = cfg.validation, cfg.validation.vision
    sample = cfg.sample_name
    case = _case_info(vcfg, sample)
    phantom_cfg = vcfg.phantoms[case["phantom"]]
    ddir = data_dir(cfg)
    print(f"[validation] vision — sample {sample} (phantom {case['phantom']}), run {vision_dir}")

    run = _read_vision_run(vision_dir)
    fixture = read_fixture(ddir / vcfg.fixture_table)
    v_cad, f_cad = load_mesh(ddir / phantom_cfg.stl)
    holes = find_holes(v_cad, f_cad)
    warnings = list(fixture.warnings) + [
        f"vision: {w}" for w in (run["meta"].get("aruco_alignment") or {}).get("warnings", [])]
    if len(holes) == 0:
        warnings.append(f"no fiducial holes (bersagli) found in {phantom_cfg.stl}")
    if fixture.edge_mm and abs(run["edge_mm"] / fixture.edge_mm - 1.0) > 0.005:
        warnings.append(f"marker edge: vision used {run['edge_mm']:.3f} mm, the table says "
                        f"{fixture.edge_mm:.3f} mm ({fixture.edge_source})")

    # ---------------------------------------------------------- alignment --
    common = sorted(set(run["corners"]) & set(fixture.corners_mm))
    not_in_table = sorted(set(run["corners"]) - set(fixture.corners_mm))
    if not_in_table:
        warnings.append(f"reconstructed markers not in the table (ignored): {not_in_table}")
    if len(common) < 2:
        raise SystemExit(f"[validation] only {common} of the reconstructed markers are in the "
                         "table: at least 2 are needed")
    A = np.concatenate([fixture.corners_mm[i] for i in common])
    B = np.concatenate([run["corners"][i] for i in common])
    T_k = G.kabsch(A, B)
    res = np.linalg.norm(G.apply(T_k, A) - B, axis=1)
    loo = {}
    if len(common) >= 3:
        for k, mid in enumerate(common):
            others = np.ones(len(common) * 4, dtype=bool)
            others[4 * k:4 * k + 4] = False
            T_o = G.kabsch(A[others], B[others])
            loo[str(mid)] = float(np.linalg.norm(G.apply(T_o, A[~others]) - B[~others], axis=1).mean())
    align: Dict[str, Any] = {
        "markers_used": common,
        "kabsch": {"corner_rms_mm": float(np.sqrt(np.mean(res ** 2))), "corner_max_mm": float(res.max()),
                   "leave_one_marker_out_mm": loo},
    }
    transforms = {"kabsch": T_k}

    fr = run["frame"]
    origin_ids, x_ids = fr.get("origin_marker_ids"), fr.get("x_axis_marker_ids")
    used = [i for i in run["used_ids"] if i in fixture.corners_mm]
    if fr.get("origin") != "frame_marker_ids" or not x_ids:
        align["frame"] = {"skipped": "the run has no fixed frame (vision.aruco.frame_marker_ids / "
                                     "x_axis_marker_ids not set): only kabsch / icp"}
    elif not set(origin_ids or []) | set(x_ids) <= set(fixture.corners_mm):
        align["frame"] = {"skipped": f"frame markers {origin_ids} / {x_ids} not all in the table"}
    else:
        T_f, _ = cad_frame(fixture.corners_mm, marker_ids=used, origin_ids=origin_ids, x_ids=x_ids)
        transforms["frame"] = T_f
        res_f = np.linalg.norm(G.apply(T_f, A) - B, axis=1)
        align["frame"] = {"corner_rms_mm": float(np.sqrt(np.mean(res_f ** 2))),
                          "corner_max_mm": float(res_f.max()),
                          "vs_kabsch_rotation_deg": G.rot_angle_deg(T_f @ G.inv(T_k))}

    # ------------------------------------------------------------ samples --
    marker_z = float(np.mean([c[:, 2].mean() for c in fixture.corners_mm.values()]))
    box = _roi_box(phantom_cfg, v_cad, marker_z, opt.roi_margin_mm)
    centre_cad = np.array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2, marker_z])
    above = v_cad[(v_cad[:, 2] > marker_z + 1.0) & in_xy(v_cad, box)]
    phantom_centre = ((above.min(axis=0) + above.max(axis=0)) / 2 if len(above)
                      else centre_cad)
    if "frame" in transforms:
        align["frame"]["vs_kabsch_shift_at_roi_mm"] = float(np.linalg.norm(
            G.apply(transforms["frame"], centre_cad[None]) - G.apply(T_k, centre_cad[None])))
    s = float(opt.sample_spacing_mm)
    in_roi = _in_box(box)
    # STL sampled a little beyond the box, so that reconstruction points at its
    # border still find their surface; completeness uses only the box.
    stl = G.sample_surface(v_cad, f_cad, s, keep=_in_box(box, 3.0))
    stl_in_box = in_roi(stl.points)
    occ = G.sample_surface(v_cad, f_cad, 2 * s, keep=_in_box(box, opt.occluder_margin_mm)).points
    stl_dist = G.SurfaceDistance(stl, s)

    T_k_inv = G.inv(T_k)

    def roi_out(p):                         # output-frame points in the CAD box (kabsch)
        return in_roi(G.apply(T_k_inv, p))

    v_rec = f_rec = None
    if opt.source == "mesh":
        v_rec, f_rec = load_mesh(vision_dir / "surface_mesh.ply")
        v_rec = v_rec * 1000.0
        rec = G.sample_surface(v_rec, f_rec, s, keep=roi_out, coarse=0.0)
    else:
        import trimesh

        pc = np.asarray(trimesh.load(vision_dir / "cloud.ply", process=False).vertices) * 1000.0
        rec = G.Samples(pc[roi_out(pc)], None)
    if len(rec.points) < 100:
        raise SystemExit(f"[validation] only {len(rec.points)} reconstructed samples in the "
                         f"phantom region {box} (CAD, mm): check the alignment / roi_mm")
    print(f"[validation] {len(common)} markers, kabsch corner RMS "
          f"{align['kabsch']['corner_rms_mm']:.3f} mm; {len(rec.points):,} reconstruction / "
          f"{int(stl_in_box.sum()):,} STL samples at {s:g} mm")

    # ---------------------------------------------------------------- ICP --
    if opt.icp.enabled:
        T_c, icp_info = G.icp_point_to_plane(G.apply(T_k_inv, rec.points), stl_dist,
                                             opt.icp.max_corr_mm, opt.icp.iters)
        transforms["icp"] = T_k @ G.inv(T_c)
        icp_info.update(rotation_deg=G.rot_angle_deg(T_c),
                        shift_at_roi_mm=float(np.linalg.norm(G.apply(T_c, centre_cad[None]) - centre_cad)))
        align["icp"] = icp_info

    # --------------------------------------------------------- visibility --
    cam_cad = np.einsum("ij,vjk->vik", T_k_inv, run["cams"])
    wh = _image_size(vision_dir, run["K"])
    visible = stl_in_box & G.visible_mask(stl, occ, cam_cad, run["K"], wh, tol=opt.visibility_tol_mm)
    views = _view_geometry(cam_cad, run["K"], wh, phantom_centre)

    # ---------------------------------------------------------- fiducials --
    fids, fid_note = [], None
    if not opt.fiducials.enabled:
        fid_note = "disabled (validation.vision.fiducials.enabled)"
    elif len(holes):
        fids, fid_note = _measure_fiducials(vision_dir, run, holes, T_k, v_rec, f_rec, occ, wh, opt)
        if fid_note:
            warnings.append(f"fiducials: {fid_note}")

    # ------------------------------------------------------------ metrics --
    thr = list(opt.thresholds_mm)
    heights = rec.points[:, 2]              # output frame: markers at z = 0, Z up
    metrics: Dict[str, Any] = {}
    acc_by = {}
    fid_err: Dict[str, list] = {}
    for name, T in transforms.items():
        Ti = G.inv(T)
        rec_cad = G.Samples(G.apply(Ti, rec.points),
                            None if rec.normals is None else G.rotate(Ti, rec.normals))
        acc = stl_dist.signed(rec_cad.points)
        comp = G.SurfaceDistance(rec_cad, s).signed(stl.points[visible])
        a_st, c_st = G.dist_stats(acc, thr), G.dist_stats(comp, thr)
        c_st["visible_fraction"] = float(visible.sum() / max(stl_in_box.sum(), 1))
        metrics[name] = {
            "accuracy": a_st, "completeness": c_st,
            "hausdorff_p95_mm": float(max(a_st.get("p95_abs_mm", np.nan), c_st.get("p95_abs_mm", np.nan))),
            "height": G.height_profile(heights, acc, opt.height_bin_mm),
        }
        acc_by[name] = acc
        if fids:
            fid_err[name], metrics[name]["fiducials"] = F.errors(fids, holes, T)
    primary = opt.primary if opt.primary in transforms else "kabsch"
    if primary != opt.primary:
        warnings.append(f"primary alignment {opt.primary!r} not available: using {primary!r}")

    # ------------------------------------------------------------ outputs --
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"metrics": "metrics.json"}
    if opt.save_error_map and v_rec is not None:
        files["error_map"] = _write_error_map(out_dir / "error_map.ply", v_rec, f_rec, roi_out,
                                              transforms[primary], stl_dist, opt.error_map_max_mm)
    if fids and opt.fiducials.save_sheet:
        sheet = F.write_sheet(out_dir / "fiducials.png", fids)
        if sheet:
            files["fiducials"] = sheet
    if opt.save_rrd:
        try:
            files["rerun"] = _write_rrd(out_dir / "validation.rrd", sample, v_cad, f_cad,
                                        transforms[primary], rec.points, acc_by[primary],
                                        fixture, run["corners"], common, opt.error_map_max_mm,
                                        holes, fids)
        except Exception as exc:          # the metrics are what matters
            warnings.append(f"validation.rrd not written: {exc}")

    meta = run["meta"]
    model = vcfg.model_label or "-".join(str(x) for x in (
        meta.get("backend"), (meta.get("alignment_info") or {}).get("pipeline_variant")) if x)
    result = {
        "sample": sample, **case, "model": model,
        "date": Path(out_dir).parents[2].name if len(Path(out_dir).parents) > 2 else None,
        "vision_dir": str(vision_dir), "units": "mm",
        "primary_alignment": primary,
        "roi_box_cad_mm": box, "sample_spacing_mm": s, "source": opt.source,
        "marker_edge_mm": {"vision": run["edge_mm"], "table": fixture.edge_mm,
                           "table_source": fixture.edge_source},
        "vision_frame": run["frame"], "timings": run["timings"],
        "scale": "kept: the ArUco scale set by vision; every alignment is rigid (no rescaling)",
        "views": views,
        "fiducials": _fiducials(holes, transforms[primary], primary, fids, fid_err, fid_note),
        "alignment": align, "metrics": metrics, "files": files, "warnings": warnings,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    p = metrics[primary]
    acc_p, comp_p = p["accuracy"], p["completeness"]
    print(f"[validation] {primary}: accuracy RMSE {acc_p['rmse_mm']:.3f} mm "
          f"(signed mean {acc_p['mean_signed_mm']:+.3f}), "
          + ", ".join(f"<{t:g} mm {acc_p[f'pct_lt_{t:g}mm']:.1f}%" for t in thr)
          + f"; completeness <{thr[0]:g} mm {comp_p.get(f'pct_lt_{thr[0]:g}mm', float('nan')):.1f}% "
          f"of the visible STL; Hausdorff p95 {p['hausdorff_p95_mm']:.3f} mm")
    print(f"[validation] views: {views['n_views_seeing_phantom']}/{views['n_views']} see the phantom, "
          f"distance {views['distance_mm'].get('mean', float('nan')):.0f} mm, elevation "
          f"{views['elevation_deg'].get('min', float('nan')):.0f}-{views['elevation_deg'].get('max', float('nan')):.0f} deg, "
          f"azimuth coverage {views['azimuth_coverage_deg']:.0f} deg")
    if fids:
        fs = p.get("fiducials", {})
        print(f"[validation] fiducials (bersagli): {fs.get('n_found', 0)}/{len(holes)} found"
              + (f"; {primary}: error RMSE {fs['norm_rmse_mm']:.3f} mm (along the normal "
                 f"{fs['normal_rmse_mm']:.3f}, in plane {fs['in_plane_rmse_mm']:.3f}), max "
                 f"{fs['norm_max_mm']:.3f} mm" if fs.get("n_found") else ""))
    for w in warnings:
        print(f"[validation] warning: {w}")
    print(f"[validation] outputs in {out_dir} ({result['elapsed_s']} s)")
    return result


def _r(a) -> list:
    return [round(float(x), 4) for x in a]


def _measure_fiducials(vision_dir: Path, run: dict, holes: Holes, T_k: np.ndarray, v_rec, f_rec,
                       occ: np.ndarray, wh, opt):
    """(fiducials, None) or ([], why not)."""
    images, why = F.view_images(vision_dir, run["meta"], len(run["cams"]))
    if images is None:
        return [], why
    if v_rec is None:
        if not (vision_dir / "surface_mesh.ply").is_file():
            return [], "surface_mesh.ply missing"
        v_rec, f_rec = load_mesh(vision_dir / "surface_mesh.ply")
        v_rec = v_rec * 1000.0
    fids = F.measure(holes, T_pred=T_k, cams=run["cams"], K=run["K"], image_paths=images,
                     mesh_v=v_rec, mesh_f=f_rec, occluders_cad=occ, image_wh=wh, opt=opt.fiducials,
                     visibility_tol=opt.visibility_tol_mm, keep_crops=opt.fiducials.save_sheet)
    return fids, None


def _fiducials(holes: Holes, T: np.ndarray, alignment: str, fids=None, errs=None, note=None) -> dict:
    """The bersagli: position / axis in CAD, where they fall in the output frame
    under ``T`` (the primary alignment) and, when measured, where the
    reconstruction puts them and the error per alignment."""
    mo, ax = G.apply(T, holes.mouth), G.rotate(T, holes.axis)
    items = [{"id": i + 1, "cad_mm": _r(holes.mouth[i]), "axis_cad": _r(holes.axis[i]),
              "output_frame_mm": _r(mo[i]), "axis_output_frame": _r(ax[i]),
              "diameter_mm": round(float(holes.diameter[i]), 4),
              "depth_mm": round(float(holes.depth[i]), 4)}
             for i in range(len(holes))]
    if fids:
        for it, js in zip(items, F.to_json(fids)):
            it.update({k: v for k, v in js.items() if k != "id"})
            it["error"] = {name: e[it["id"] - 1] for name, e in (errs or {}).items()}
    return {
        "what": "blind round holes of the phantom STL; point = centre of the hole mouth "
                "(on the surface), axis = out of the hole",
        "n": len(holes), "output_frame_alignment": alignment, "measured": bool(fids),
        "note": note, "items": items,
    }


def _write_error_map(path: Path, v_rec, f_rec, roi_out, T, stl_dist, vmax) -> str:
    import trimesh

    f = f_rec[roi_out(v_rec[f_rec].mean(axis=1))]
    used, f = np.unique(f, return_inverse=True)
    v = v_rec[used]
    d = stl_dist.signed(G.apply(G.inv(T), v))
    m = trimesh.Trimesh(v, f.reshape(-1, 3), process=False)
    m.visual.vertex_colors = np.c_[_colors(d, vmax), np.full(len(v), 255, np.uint8)]
    m.export(path)
    np.save(path.with_suffix(".npy"), d.astype(np.float32))
    return path.name


def _write_rrd(path: Path, sample, v_cad, f_cad, T, rec_pts, acc, fixture, rec_corners,
               common, vmax, holes: Holes = None, fids=None) -> str:
    import rerun as rr

    rec = rr.RecordingStream("spectra_validation", recording_id=f"{sample}_validation")
    rec.save(path)
    rec.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rec.log("stl", rr.Mesh3D(vertex_positions=G.apply(T, v_cad), triangle_indices=f_cad,
                             albedo_factor=[190, 190, 200, 120]), static=True)
    step = max(1, len(rec_pts) // 300_000)
    rec.log("reconstruction", rr.Points3D(rec_pts[::step], colors=_colors(acc[::step], vmax),
                                          radii=rr.Radius.ui_points(1.5)), static=True)
    cad = np.concatenate([G.apply(T, fixture.corners_mm[i]) for i in common])
    rc = np.concatenate([rec_corners[i] for i in common])
    rec.log("markers/cad", rr.Points3D(cad, colors=[255, 200, 0], radii=rr.Radius.ui_points(4)),
            static=True)
    rec.log("markers/reconstructed", rr.Points3D(rc, colors=[255, 0, 80], radii=rr.Radius.ui_points(4)),
            static=True)
    if holes is not None and len(holes):
        mo = G.apply(T, holes.mouth)
        rec.log("fiducials/expected", rr.Points3D(mo, colors=[0, 220, 255], radii=rr.Radius.ui_points(5),
                                                  labels=[str(i + 1) for i in range(len(mo))]), static=True)
        rec.log("fiducials/axis", rr.Arrows3D(origins=mo, vectors=G.rotate(T, holes.axis) * 2.0,
                                              colors=[0, 220, 255]), static=True)
        got = [(i, fd.measured) for i, fd in enumerate(fids or []) if fd.measured is not None]
        if got:
            rec.log("fiducials/measured", rr.Points3D(np.array([m for _, m in got]), colors=[0, 230, 0],
                                                      radii=rr.Radius.ui_points(5),
                                                      labels=[str(i + 1) for i, _ in got]), static=True)
            rec.log("fiducials/error", rr.LineStrips3D([[mo[i], m] for i, m in got], colors=[255, 60, 60]),
                    static=True)
    rec.flush()
    rec.disconnect()
    return path.name
