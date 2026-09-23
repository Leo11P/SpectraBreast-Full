"""
pipeline_roi.py — Pipeline con modalita' ROI
=============================================

Wrapper attorno a run_full_pipeline() originale che gestisce l'acquisizione
HSI duale (PNG LiveView con ArUco + cubo ROI senza ArUco).

Flusso ROI:
  1. Detection ArUco sulla PNG LiveView (cv2.imread grayscale)
  2. Stima T_roi_to_png via SIFT/ORB + RANSAC (modulo roi_align)
  3. Calcolo H_png_to_render normalmente
  4. Composizione H_roi_to_render = H_png_to_render @ T_roi_to_png
  5. Per la pipeline a valle (lookup 3D, errori 3D ROI, point cloud spettrale)
     viene usato H_roi_to_render con il cubo ROI come "HSI".

Per evitare di duplicare 1400+ righe, riusa tutte le primitive di pipeline.py
e ridefinisce solo il punto di iniezione della 2D image + dell'omografia.

API:
    run_full_pipeline_roi(
        hsi_hdr_path     = ".../cube_roi.hdr",
        liveview_png_path= ".../sample_liveview.png",
        mesh_path        = ...,
        aruco_json_path  = ...,
        output_dir       = ...,
        roi_align_cfg    = {'method': 'sift', 'min_matches': 20,
                            'ransac_thresh': 3.0, 'save_match_viz': True},
        ... [stessi parametri di run_full_pipeline]
    )

    Se liveview_png_path e' None -> chiama direttamente run_full_pipeline()
    (comportamento standard, retrocompatibile).
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from .pipeline import (
    run_full_pipeline,
    load_envi,
    extract_2d_from_hsi,
    detect_aruco,
    load_aruco_3d_json,
    load_mesh,
    render_orthographic_topview,
    project_aruco3d_to_render,
    match_aruco_hsi_render,
    compute_homography,
    reprojection_error_2d,
    reprojection_error_3d_json,
    lookup_3d,
    save_excel_errors,
    visualize_registration,
    visualize_coverage,
    save_render,
    save_turbo_render,
    build_and_export_pointcloud_streaming,
    _export_csv_streaming,
    parse_wavelengths,
    _px_per_mm_from_data,
)
from .roi_align import compute_roi_to_png_homography, load_liveview_png


# =============================================================================
# Default config per il matching ROI -> PNG
# =============================================================================

_DEFAULT_ROI_ALIGN_CFG = {
    'method'         : 'sift',
    'min_matches'    : 20,
    'ransac_thresh'  : 3.0,
    'lowe_ratio'     : 0.75,
    'fallback_orb'   : True,
    'save_match_viz' : True,
}


def _merge_roi_cfg(user_cfg: Optional[dict]) -> dict:
    out = dict(_DEFAULT_ROI_ALIGN_CFG)
    if user_cfg:
        out.update(user_cfg)
    return out


# =============================================================================
# Pipeline ROI
# =============================================================================

def run_full_pipeline_roi(
    hsi_hdr_path,
    mesh_path,
    aruco_json_path,
    output_dir,
    liveview_png_path        = None,   # PNG con ArUco (None = modalita' standard)
    roi_align_cfg            = None,
    # Registration
    hsi_extraction_method    = 'mean',
    aruco_dict_type          = cv2.aruco.DICT_4X4_50,
    suspicious_pixels_hsi    = None,
    render_resolution_mm     = 2.0,
    render_resolution_reg_mm = None,
    render_margin_mm         = 10.0,
    chunk_size               = 500_000,
    marker_side_mm           = 9.0,
    use_subpix               = True,
    subpix_winsize           = 5,
    equalize_method          = 'global',
    tune_detector            = False,
    # Point cloud
    border_px                = 2,
    reflectance_norm         = True,
    export_ply_file          = True,
    export_npy_file          = True,
    export_csv_file          = False,
    csv_max_points           = 500_000,
    pc_chunk_size            = 100_000,
    # Export flags
    save_pointcloud          = True,
    save_images              = True,
    sample_name              = 'sample',
    precomputed_render       = None,
    precomputed_render_reg   = None,
):
    """
    Pipeline completa con supporto modalita' ROI.

    Se liveview_png_path e' None -> esegue run_full_pipeline() standard.
    Se liveview_png_path e' fornito -> modalita' ROI:
        - ArUco detection sulla PNG LiveView
        - T_roi_to_png stimata via SIFT+RANSAC tra mean(ROI) e PNG
        - H_roi_to_render = H_png_to_render @ T_roi_to_png
        - point cloud, lookup 3D, ecc. usano il cubo ROI

    Parameters
    ----------
    liveview_png_path : str | None
        Path alla PNG LiveView. Se None la pipeline funziona in modalita'
        standard (l'HSI deve contenere i 4 ArUco visibili).
    roi_align_cfg : dict | None
        Config per il matching ROI->PNG. Chiavi accettate:
        - method         : 'sift' | 'orb'           (default 'sift')
        - min_matches    : int                       (default 20)
        - ransac_thresh  : float, px                 (default 3.0)
        - lowe_ratio     : float                     (default 0.75)
        - fallback_orb   : bool                      (default True)
        - save_match_viz : bool                      (default True)

    Tutti gli altri parametri sono identici a run_full_pipeline().

    Returns
    -------
    dict identico a run_full_pipeline(), con campi extra:
        - 'T_roi_to_png'    : (3,3) omografia ROI->PNG, None in modalita' standard
        - 'roi_align_info'  : dict con metriche del matching, None in standard
        - 'is_roi_mode'     : bool
    """
    # ── Caso 1: modalita' standard -> delego completamente ────────────────────
    if liveview_png_path is None:
        result = run_full_pipeline(
            hsi_hdr_path             = hsi_hdr_path,
            mesh_path                = mesh_path,
            aruco_json_path          = aruco_json_path,
            output_dir               = output_dir,
            hsi_extraction_method    = hsi_extraction_method,
            aruco_dict_type          = aruco_dict_type,
            suspicious_pixels_hsi    = suspicious_pixels_hsi,
            render_resolution_mm     = render_resolution_mm,
            render_resolution_reg_mm = render_resolution_reg_mm,
            render_margin_mm         = render_margin_mm,
            chunk_size               = chunk_size,
            marker_side_mm           = marker_side_mm,
            use_subpix               = use_subpix,
            subpix_winsize           = subpix_winsize,
            equalize_method          = equalize_method,
            tune_detector            = tune_detector,
            border_px                = border_px,
            reflectance_norm         = reflectance_norm,
            export_ply_file          = export_ply_file,
            export_npy_file          = export_npy_file,
            export_csv_file          = export_csv_file,
            csv_max_points           = csv_max_points,
            pc_chunk_size            = pc_chunk_size,
            save_pointcloud          = save_pointcloud,
            save_images              = save_images,
            sample_name              = sample_name,
            precomputed_render       = precomputed_render,
            precomputed_render_reg   = precomputed_render_reg,
        )
        result['T_roi_to_png']   = None
        result['roi_align_info'] = None
        result['is_roi_mode']    = False
        return result

    # ── Caso 2: modalita' ROI ────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    roi_cfg = _merge_roi_cfg(roi_align_cfg)

    res_reg = render_resolution_reg_mm if render_resolution_reg_mm is not None \
              else render_resolution_mm
    res_pc  = render_resolution_mm
    dual    = abs(res_reg - res_pc) > 1e-6

    print("=" * 68)
    print("FULL PIPELINE — ROI MODE")
    print(f"  Sample           : {sample_name}")
    print(f"  Cubo ROI (.hdr)  : {hsi_hdr_path}")
    print(f"  LiveView PNG     : {liveview_png_path}")
    print(f"  Res registro     : {res_reg} mm/px")
    print(f"  Res point cloud  : {res_pc} mm/px")
    print(f"  SubPix refine    : {'ON' if use_subpix else 'OFF'}")
    print(f"  ROI match method : {roi_cfg['method']}")
    print(f"  Output dir       : {output_dir}")
    print("=" * 68)

    # ── Step 0: ArUco 3D ──────────────────────────────────────────────────────
    print("\n[Step 0] Loading ArUco 3D from JSON...")
    aruco_3d, _ = load_aruco_3d_json(aruco_json_path, scale_m_to_mm=True)

    # ── Step 1: Carico cubo ROI ──────────────────────────────────────────────
    print("\n[Step 1] Loading ROI HSI cube...")
    cube, meta = load_envi(hsi_hdr_path)
    print(f"  Cube shape (ROI): {cube.shape}")
    # Mean del cubo ROI (qualunque method qui produce un'immagine 2D,
    # ma 'mean' e' la piu' robusta per feature matching)
    roi_2d = extract_2d_from_hsi(cube, meta, method=hsi_extraction_method,
                                 equalize_method=equalize_method)

    # ── Step 1b: Carico PNG LiveView ──────────────────────────────────────────
    print(f"\n[Step 1b] Loading LiveView PNG: {liveview_png_path}")
    png_2d = load_liveview_png(liveview_png_path)
    print(f"  PNG shape       : {png_2d.shape}")

    # ── Step 1c: Stima T_roi_to_png ──────────────────────────────────────────
    print("\n[Step 1c] Stima omografia ROI -> PNG (feature matching)...")
    T_roi_to_png, roi_info = compute_roi_to_png_homography(
        roi_image    = roi_2d,
        png_image    = png_2d,
        method       = roi_cfg['method'],
        min_matches  = roi_cfg['min_matches'],
        ransac_thresh= roi_cfg['ransac_thresh'],
        lowe_ratio   = roi_cfg['lowe_ratio'],
        fallback_orb = roi_cfg['fallback_orb'],
        output_dir   = output_dir if save_images else None,
        sample_name  = sample_name,
        save_viz     = roi_cfg['save_match_viz'] and save_images,
    )

    # ── Step 2: Render(s) ─────────────────────────────────────────────────────
    # Render REG
    if precomputed_render_reg is not None:
        print("\n[Step 2a] Render registrazione pre-calcolato (GPU).")
        render_rgb_reg, depth_map_reg, xyz_map_reg, origin_xy_reg, res_reg_actual = \
            precomputed_render_reg
    elif precomputed_render is not None and not dual:
        print("\n[Step 2a] Render unico pre-calcolato — usato per registrazione.")
        render_rgb_reg, depth_map_reg, xyz_map_reg, origin_xy_reg, res_reg_actual = \
            precomputed_render
    else:
        print(f"\n[Step 2a] Render registrazione CPU ({res_reg} mm/px)...")
        mesh = load_mesh(mesh_path, scale_m_to_mm=True)
        render_rgb_reg, depth_map_reg, xyz_map_reg, origin_xy_reg, res_reg_actual = \
            render_orthographic_topview(
                mesh,
                resolution_mm_per_px = res_reg,
                margin_mm             = render_margin_mm,
                chunk_size            = chunk_size,
            )
        if save_images:
            save_render(render_rgb_reg, depth_map_reg, output_dir=output_dir,
                        prefix=f'{sample_name}_render_reg')
            save_turbo_render(xyz_map_reg,
                              os.path.join(output_dir,
                                           f'{sample_name}_render_reg_turbo.png'))

    # Render PC
    if not dual:
        render_rgb_pc  = render_rgb_reg
        depth_map_pc   = depth_map_reg
        xyz_map_pc     = xyz_map_reg
        origin_xy_pc   = origin_xy_reg
        res_pc_actual  = res_reg_actual
        print(f"[Step 2b] Risoluzione reg = pc ({res_pc} mm/px) — render condiviso.")
    elif precomputed_render is not None:
        print("\n[Step 2b] Render point cloud pre-calcolato (GPU).")
        render_rgb_pc, depth_map_pc, xyz_map_pc, origin_xy_pc, res_pc_actual = \
            precomputed_render
    else:
        print(f"\n[Step 2b] Render point cloud CPU ({res_pc} mm/px)...")
        if 'mesh' not in dir():
            mesh = load_mesh(mesh_path, scale_m_to_mm=True)
        render_rgb_pc, depth_map_pc, xyz_map_pc, origin_xy_pc, res_pc_actual = \
            render_orthographic_topview(
                mesh,
                resolution_mm_per_px = res_pc,
                margin_mm             = render_margin_mm,
                chunk_size            = chunk_size,
            )
        if save_images:
            save_render(render_rgb_pc, depth_map_pc, output_dir=output_dir,
                        prefix=f'{sample_name}_render_pc')
            save_turbo_render(xyz_map_pc,
                              os.path.join(output_dir,
                                           f'{sample_name}_render_pc_turbo.png'))

    # ── Step 3: ArUco sulla PNG LiveView (NON sulla ROI) ─────────────────────
    print("\n[Step 3] Detecting ArUco on LiveView PNG...")
    data_png, _ = detect_aruco(png_2d, aruco_dict_type=aruco_dict_type,
                                use_subpix=use_subpix,
                                subpix_winsize=subpix_winsize,
                                tune_detector=tune_detector)
    print(f"  PNG markers found: {sorted(data_png.keys())}")
    if len(data_png) < 1:
        raise RuntimeError(
            "Nessun ArUco rilevato sulla PNG LiveView. "
            "Verifica che la PNG sia quella giusta e contenga marker visibili."
        )

    # ── Step 3b: Proietto JSON 3D corners -> render_reg ──────────────────────
    print("\n[Step 3b] Projecting JSON ArUco corners -> render pixels (reg)...")
    data_render_reg = project_aruco3d_to_render(
        aruco_3d, origin_xy_reg, res_reg_actual
    )

    # ── Step 4: Omografia PNG -> render_reg ──────────────────────────────────
    print("\n[Step 4] Computing homography PNG -> render_reg...")
    pts_png, pts_render_reg, common_ids = match_aruco_hsi_render(
        data_png, data_render_reg
    )
    H_png_to_render, _ = compute_homography(
        pts_png, pts_render_reg, tag=' PNG->render_reg'
    )

    # ── Step 4b: Composizione H_roi_to_render = H_png_to_render @ T_roi_to_png
    H = H_png_to_render @ T_roi_to_png   # (3,3): ROI px -> render_reg px

    # Stessa logica per la griglia PC (scala uniforme tra le due griglie render)
    scale_r2p = res_reg_actual / res_pc_actual
    S = np.array([[scale_r2p, 0,         0],
                  [0,         scale_r2p, 0],
                  [0,         0,         1]], dtype=np.float64)
    H_pc     = S @ H                            # ROI px -> render_pc px
    H_inv    = np.linalg.inv(H)                 # render_reg px -> ROI px
    H_pc_inv = np.linalg.inv(H_pc)              # render_pc  px -> ROI px

    # Per l'errore 2D ho bisogno anche di proiettare i corner PNG attesi.
    # Costruisco data_roi (corner ArUco "virtuali" sulla ROI) applicando
    # T_png_to_roi = inv(T_roi_to_png) ai corner della PNG. Serve per
    # visualizzazione e per la metrica 2D che viene calcolata da
    # reprojection_error_2d. Tuttavia, qui la 2D piu' significativa e' quella
    # PNG->render_reg (gli ArUco SONO sulla PNG), quindi calcoliamo l'errore
    # 2D usando pts_png e pts_render_reg con H_png_to_render.
    T_png_to_roi = np.linalg.inv(T_roi_to_png)
    pts_roi_virtual = cv2.perspectiveTransform(
        pts_png.reshape(-1, 1, 2).astype(np.float32),
        T_png_to_roi.astype(np.float32),
    ).reshape(-1, 2)

    # Costruisco data_roi dict per coerenza con la visualizzazione
    data_roi = {}
    for mid in data_png:
        corners_png = data_png[mid].reshape(-1, 1, 2).astype(np.float32)
        corners_roi = cv2.perspectiveTransform(
            corners_png, T_png_to_roi.astype(np.float32)
        ).reshape(-1, 2)
        data_roi[mid] = corners_roi.astype(np.float32)

    # ── Step 5: Suspicious points (in coordinate ROI) ────────────────────────
    pts_render_h     = None
    coords_3d_h_bil  = None
    coords_3d_h_bic  = None
    if suspicious_pixels_hsi is not None and len(suspicious_pixels_hsi) > 0:
        print(f"\n[Step 5] Projecting {len(suspicious_pixels_hsi)} suspicious point(s)...")
        # NOTA: in modalita' ROI i suspicious_pixels devono essere in coordinate
        # ROI (cioe' indici del cubo HSI ROI). Vengono proiettati via H_pc
        # (ROI -> render_pc) e poi cercati nella xyz_map_pc.
        pts_render_h = cv2.perspectiveTransform(
            suspicious_pixels_hsi.reshape(-1, 1, 2).astype(np.float32), H_pc
        ).reshape(-1, 2)
        coords_3d_h_bil, coords_3d_h_bic = lookup_3d(
            pts_render_h, xyz_map_pc, mode='both', verbose=True
        )

    # ── Step 6: Error metrics ────────────────────────────────────────────────
    print("\n[Step 6] Computing registration errors...")
    # L'errore 2D e' calcolato sui corner della PNG (dove gli ArUco sono
    # effettivamente visibili), proiettati via H_png_to_render.
    err2d_px = reprojection_error_2d(
        pts_png, pts_render_reg, H_png_to_render, tag='PNG->render_reg'
    )
    # Gli errori 3D usano i pixel della PNG (dove gli ArUco sono validi) e
    # H_png_to_render per arrivare al render, e poi la xyz_map.
    # Da notare: la pipeline standard usa data_hsi che e' un dict marker_id ->
    # corners. Qui passiamo data_png (corner sulla PNG) come "data_hsi".
    err3d_dict = reprojection_error_3d_json(
        aruco_3d, data_png, data_render_reg, xyz_map_reg, H_png_to_render, tag='reg'
    )

    if dual:
        print("\n[Step 6b] Computing 3D errors on PC grid...")
        # Per la griglia PC l'omografia da usare e' la PNG->render_pc, che e'
        # semplicemente S @ H_png_to_render (stessa scalatura usata sopra).
        H_png_to_render_pc = S @ H_png_to_render
        err3d_dict_pc = reprojection_error_3d_json(
            aruco_3d, data_png, data_render_reg, xyz_map_pc,
            H_png_to_render_pc, tag='pc'
        )
    else:
        err3d_dict_pc = err3d_dict

    best_method = 'bilinear'
    if err3d_dict is not None:
        eb = err3d_dict['bilinear']; ec = err3d_dict['bicubic']
        eb_v = eb[~np.isnan(eb)] if eb.size else np.array([])
        ec_v = ec[~np.isnan(ec)] if ec.size else np.array([])
        m_bil = float(eb_v.mean()) if eb_v.size else float('inf')
        m_bic = float(ec_v.mean()) if ec_v.size else float('inf')
        best_method = 'bicubic' if m_bic < m_bil else 'bilinear'
        print(f"\n[Method] Best 3D lookup: {best_method.upper()}  "
              f"(bilinear={m_bil:.4f} mm, bicubic={m_bic:.4f} mm)")

    err3d_mm = err3d_dict[best_method] if err3d_dict is not None else None
    coords_3d_h = (coords_3d_h_bic if best_method == 'bicubic'
                   else coords_3d_h_bil)
    if coords_3d_h is not None:
        print(f"\n  Suspicious points -> 3D (method={best_method}):")
        for i in range(len(coords_3d_h)):
            ph  = suspicious_pixels_hsi[i]
            xyz = coords_3d_h[i]
            print(f"  T{i}: ROI({ph[0]:.1f},{ph[1]:.1f}) -> "
                  f"X={xyz[0]:.3f} Y={xyz[1]:.3f} Z={xyz[2]:.3f} mm")

    # ── Step 7: Excel ────────────────────────────────────────────────────────
    excel_path = os.path.join(output_dir, f'{sample_name}_registration_errors.xlsx')
    print(f"\n[Step 7] Saving Excel error report -> {excel_path}")
    # Per la scala px/mm usiamo i corner della PNG (e' li' che gli ArUco
    # esistono realmente). Il marker_side_mm rimane invariato.
    save_excel_errors(
        err2d_px       = err2d_px,
        err3d_dict     = err3d_dict,
        data_hsi       = data_png,           # PNG e' la sorgente ArUco
        common_ids     = common_ids,
        marker_side_mm = marker_side_mm,
        output_path    = excel_path,
        err3d_dict_pc  = err3d_dict_pc if dual else None,
        res_reg        = res_reg_actual,
        res_pc         = res_pc_actual  if dual else None,
    )

    # ── Step 8: Registration visualisation ───────────────────────────────────
    if save_images:
        print("\n[Step 8] Saving registration visualisation...")
        # In modalita' ROI mostriamo:
        #   - a sinistra la ROI (con i corner ArUco "virtuali" proiettati)
        #   - a destra il render_reg con i marker proiettati dal JSON
        visualize_registration(
            render_rgb            = render_rgb_reg,
            hsi_2d                = roi_2d,
            pts_render_h          = pts_render_h,
            data_hsi              = data_roi,             # virtuali sulla ROI
            data_render           = data_render_reg,
            suspicious_pixels_hsi = suspicious_pixels_hsi,
            coords_3d_h           = coords_3d_h,
            output_path           = os.path.join(
                output_dir, f'{sample_name}_registration.png'
            ),
        )
        # Salvo anche una visualizzazione sulla PNG (dove gli ArUco sono veri)
        visualize_registration(
            render_rgb            = render_rgb_reg,
            hsi_2d                = png_2d,
            pts_render_h          = None,
            data_hsi              = data_png,
            data_render           = data_render_reg,
            suspicious_pixels_hsi = None,
            coords_3d_h           = None,
            output_path           = os.path.join(
                output_dir, f'{sample_name}_registration_png.png'
            ),
        )
    else:
        print("\n[Step 8] Skipped (save_images=False).")

    # ── Step 9+10: Point cloud spettrale (usa il cubo ROI) ───────────────────
    wavelengths = parse_wavelengths(meta)
    if wavelengths:
        print(f"  Wavelengths: {wavelengths[0]:.1f} ... {wavelengths[-1]:.1f} nm "
              f"({len(wavelengths)} bands)")

    if save_pointcloud:
        print("\n[Step 9] Building spectral point cloud (ROI cube)...")
        base = os.path.join(output_dir, f'spectral_pointcloud_{sample_name}')
        valid_mask, n_valid = build_and_export_pointcloud_streaming(
            cube              = cube,
            xyz_map           = xyz_map_pc,
            H_inv             = H_pc_inv,        # render_pc px -> ROI px
            output_ply_path   = (base + '.ply')  if export_ply_file else None,
            output_npz_prefix = base              if export_npy_file else None,
            wavelengths       = wavelengths,
            border_px         = border_px,
            reflectance_norm  = reflectance_norm,
            chunk_size        = pc_chunk_size,
        )
        if export_csv_file and n_valid > 0:
            print(f"\n[Step 10] Exporting CSV (max {csv_max_points:,} sampled points)...")
            _export_csv_streaming(
                cube=cube, xyz_flat=xyz_map_pc.reshape(-1, 3),
                valid_idx=(np.where(valid_mask.ravel())[0]).astype(np.int32),
                H_inv32=H_pc_inv.astype(np.float32),
                W_r=xyz_map_pc.shape[1], cols_hsi=cube.shape[1],
                rows_hsi=cube.shape[0],
                wavelengths=wavelengths, output_path=base + '.csv',
                max_points=csv_max_points, reflectance_norm=reflectance_norm,
            )
    else:
        print("\n[Step 9+10] Point cloud skipped (save_pointcloud=False).")
        valid_mask = ~np.isnan(xyz_map_pc[:, :, 0])
        n_valid    = int(valid_mask.sum())

    points_xyz = None
    spectra    = None

    # ── Step 11: Coverage (sulla ROI) ────────────────────────────────────────
    if save_images:
        print("\n[Step 11] Saving coverage map (ROI)...")
        visualize_coverage(
            render_rgb  = render_rgb_pc,
            valid_mask  = valid_mask,
            hsi_2d      = roi_2d,
            H_inv       = H_pc_inv,
            output_path = os.path.join(output_dir, f'{sample_name}_coverage.png'),
        )
    else:
        print("\n[Step 11] Skipped (save_images=False).")

    # ── Summary ──────────────────────────────────────────────────────────────
    px_per_mm = _px_per_mm_from_data(data_png, marker_side_mm)
    print("\n" + "=" * 68)
    print("PIPELINE COMPLETED (ROI MODE)")
    print("=" * 68)
    print(f"  Res. registrazione : {res_reg_actual} mm/px")
    print(f"  Res. point cloud   : {res_pc_actual} mm/px")
    print(f"  ROI->PNG inliers   : {roi_info['n_inliers']}/{roi_info['n_good_matches']}  "
          f"reproj={roi_info['reproj_error_mean_px']:.3f} px")
    print(f"  2D error mean      : {err2d_px.mean():.4f} px  "
          f"= {err2d_px.mean()/px_per_mm:.4f} mm")
    print(f"  2D error max       : {err2d_px.max():.4f} px  "
          f"= {err2d_px.max()/px_per_mm:.4f} mm")
    if err3d_dict is not None:
        for tag, arr in (('bilinear', err3d_dict['bilinear']),
                         ('bicubic',  err3d_dict['bicubic'])):
            v = arr[~np.isnan(arr)]
            if v.size:
                best_lbl = ' <-- best' if tag == best_method else ''
                print(f"  3D REG {tag:<8s} mean: {v.mean():.4f} mm  "
                      f"median: {np.median(v):.4f} mm{best_lbl}")
    if dual and err3d_dict_pc is not None:
        for tag, arr in (('bilinear', err3d_dict_pc['bilinear']),
                         ('bicubic',  err3d_dict_pc['bicubic'])):
            v = arr[~np.isnan(arr)]
            if v.size:
                print(f"  3D PC  {tag:<8s} mean: {v.mean():.4f} mm  "
                      f"median: {np.median(v):.4f} mm")
    n_bands = len(wavelengths) if wavelengths else cube.shape[2]
    print(f"  Point cloud        : {n_valid:,} points  x {n_bands} bands")
    print(f"  Output dir         : {output_dir}")

    return {
        # Omografie
        'H': H, 'H_inv': H_inv,
        'H_pc': H_pc, 'H_pc_inv': H_pc_inv,
        'H_png_to_render': H_png_to_render,
        'T_roi_to_png'   : T_roi_to_png,
        'roi_align_info' : roi_info,
        'is_roi_mode'    : True,
        # Registration
        'pts_hsi': pts_png, 'pts_render': pts_render_reg,
        'common_ids': common_ids,
        'data_hsi': data_png,            # ArUco sulla PNG (sorgente vera)
        'data_roi': data_roi,            # ArUco "virtuali" sulla ROI
        'data_render': data_render_reg,
        'err2d_px': err2d_px,
        'err3d_mm': err3d_mm,
        'err3d_dict': err3d_dict,
        'err3d_dict_pc': err3d_dict_pc if dual else None,
        'best_method': best_method,
        'pts_render_h': pts_render_h,
        'coords_3d_h': coords_3d_h,
        'coords_3d_h_bilinear': coords_3d_h_bil,
        'coords_3d_h_bicubic':  coords_3d_h_bic,
        # Render
        'render_rgb_reg': render_rgb_reg, 'depth_map_reg': depth_map_reg,
        'xyz_map_reg': xyz_map_reg, 'origin_xy_reg': origin_xy_reg,
        'render_rgb': render_rgb_pc, 'depth_map': depth_map_pc,
        'xyz_map': xyz_map_pc, 'origin_xy': origin_xy_pc,
        # HSI
        'cube': cube, 'hsi_2d': roi_2d, 'png_2d': png_2d,
        'wavelengths': wavelengths,
        # Point cloud
        'xyz': points_xyz, 'spectra': spectra,
        'valid_mask': valid_mask, 'n_valid': n_valid,
        'bands': cube.shape[2],
    }