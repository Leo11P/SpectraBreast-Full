"""
render_gpu.py — Ray casting ortografico accelerato su GPU
==========================================================
Drop-in replacement per render_orthographic_topview() di Pipeline_registrazione_2D3D_new.py

Backend (priorità automatica):
  1. CuPy   — kernel CUDA custom via RawModule (massima velocità, richiede cupy)
  2. PyTorch — Möller–Trumbore vettorizzato su GPU (richiede torch + CUDA)
  3. CPU     — trimesh BVH (fallback, nessuna GPU richiesta)

Requisiti minimi:
    pip install trimesh
    pip install torch          # per il backend PyTorch GPU (raccomandato)
    pip install cupy-cuda12x   # per il backend CuPy (opzionale, più veloce)

Utilizzo:
    from render_gpu import render_orthographic_topview_gpu

    render_rgb, depth_map, xyz_map, origin_xy, res = render_orthographic_topview_gpu(
        mesh,
        resolution_mm_per_px = 2.0,
        margin_mm             = 10.0,
        device                = 'cuda',   # 'cuda', 'cuda:1', 'cpu', o None (auto)
    )

L'output è identico alla versione CPU — il resto della pipeline non cambia.
"""

import numpy as np
import cv2
import trimesh

try:
    import cupy as cp
    CUPY_OK = True
except ImportError:
    CUPY_OK = False

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# =============================================================================
# CUDA kernel: Möller–Trumbore ray-triangle intersection
# =============================================================================
# Ogni thread gestisce un raggio (un pixel del render).
# La BVH non è implementata qui: iteriamo su tutti i triangoli (brute-force).
# Per mesh grandi (>500k facce) considera di passare a OptiX o di pre-filtrare.

_KERNEL_SRC = r"""
extern "C" __global__
void raytrace_ortho(
    const float* __restrict__ verts,   // (N_verts, 3) float32
    const int*   __restrict__ faces,   // (N_faces, 3) int32
    const float* __restrict__ ray_ox,  // (H*W,) X origine raggi
    const float* __restrict__ ray_oy,  // (H*W,) Y origine raggi
    const float  ray_oz,               // Z origine (costante, sopra la mesh)
    const int    n_faces,
    const int    n_rays,
    float* __restrict__ hit_x,         // output (H*W,) X hit
    float* __restrict__ hit_y,         // output (H*W,) Y hit
    float* __restrict__ hit_z          // output (H*W,) Z hit — NaN se no hit
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rays) return;

    // Raggio: origine (ox, oy, oz), direzione (0, 0, -1)
    float ox = ray_ox[idx];
    float oy = ray_oy[idx];
    float oz = ray_oz;

    float best_t = 1e30f;
    float best_x = 0.f, best_y = 0.f, best_z = 0.f;
    bool  found  = false;

    for (int fi = 0; fi < n_faces; fi++) {
        int i0 = faces[fi * 3 + 0];
        int i1 = faces[fi * 3 + 1];
        int i2 = faces[fi * 3 + 2];

        float v0x = verts[i0*3+0], v0y = verts[i0*3+1], v0z = verts[i0*3+2];
        float v1x = verts[i1*3+0], v1y = verts[i1*3+1], v1z = verts[i1*3+2];
        float v2x = verts[i2*3+0], v2y = verts[i2*3+1], v2z = verts[i2*3+2];

        // Edge vectors
        float e1x = v1x-v0x, e1y = v1y-v0y, e1z = v1z-v0z;
        float e2x = v2x-v0x, e2y = v2y-v0y, e2z = v2z-v0z;

        // dir = (0, 0, -1)  =>  h = dir x e2
        float hx =  e2y;   // (0*e2z - (-1)*e2y)
        float hy = -e2x;   // ((-1)*e2x - 0*e2z)
        float hz =  0.f;   // (0*e2y - 0*e2x)

        float a = e1x*hx + e1y*hy; // e1z*hz = 0

        if (a > -1e-8f && a < 1e-8f) continue;  // raggio parallelo

        float f  = 1.f / a;
        float sx = ox - v0x, sy = oy - v0y, sz = oz - v0z;
        float u  = f * (sx*hx + sy*hy);          // sz*hz = 0

        if (u < 0.f || u > 1.f) continue;

        // q = s x e1
        float qx = sy*e1z - sz*e1y;
        float qy = sz*e1x - sx*e1z;
        float qz = sx*e1y - sy*e1x;

        // dir = (0,0,-1)  =>  dir·q = -qz
        float v = f * (-qz);
        if (v < 0.f || u + v > 1.f) continue;

        // e2·q
        float t = f * (e2x*qx + e2y*qy + e2z*qz);
        if (t < 1e-6f || t >= best_t) continue;

        best_t = t;
        best_x = ox;
        best_y = oy;
        best_z = oz - t;   // dir=(0,0,-1) => hit_z = oz - t
        found  = true;
    }

    if (found) {
        hit_x[idx] = best_x;
        hit_y[idx] = best_y;
        hit_z[idx] = best_z;
    } else {
        hit_x[idx] = __int_as_float(0x7FC00000);  // NaN
        hit_y[idx] = __int_as_float(0x7FC00000);
        hit_z[idx] = __int_as_float(0x7FC00000);
    }
}
"""


def _compile_kernel():
    mod    = cp.RawModule(code=_KERNEL_SRC)
    kernel = mod.get_function("raytrace_ortho")
    return kernel


# =============================================================================
# BVH-lite: suddivide la mesh in celle XY per ridurre il numero di triangoli
# che ogni thread deve testare (opzionale, attivo se n_faces > THRESHOLD)
# =============================================================================

def _split_mesh_xy(mesh, n_tiles=8):
    """
    Suddivide le facce della mesh in una griglia n_tiles x n_tiles sull'asse XY.
    Ritorna una lista di sub-mesh trimesh (può essere vuota per alcune celle).
    Non è una BVH vera, ma riduce il lavoro per mesh molto grandi.
    """
    bounds   = mesh.bounds
    xmin, xmax = bounds[0,0], bounds[1,0]
    ymin, ymax = bounds[0,1], bounds[1,1]
    dx = (xmax - xmin) / n_tiles
    dy = (ymax - ymin) / n_tiles

    centroids = mesh.triangles_center
    tiles     = []
    for ix in range(n_tiles):
        for iy in range(n_tiles):
            x0, x1 = xmin + ix*dx, xmin + (ix+1)*dx
            y0, y1 = ymin + iy*dy, ymin + (iy+1)*dy
            mask    = ((centroids[:,0] >= x0) & (centroids[:,0] < x1) &
                       (centroids[:,1] >= y0) & (centroids[:,1] < y1))
            if not mask.any():
                continue
            sub = trimesh.Trimesh(
                vertices = mesh.vertices,
                faces    = mesh.faces[mask],
                process  = False,
            )
            tiles.append((x0, x1, y0, y1, sub))
    return tiles


# =============================================================================
# Backend PyTorch: Möller–Trumbore vettorizzato
# =============================================================================

def _render_torch_tiles(mesh, xs, ys, W, H, zmax, resolution_mm_per_px,
                         device, n_tiles=8, chunk_rays=4000):
    """
    Ray casting ortografico top-view via PyTorch GPU (o CPU).
    Direzione raggi fissa: (0, 0, -1).

    Usa una griglia n_tiles x n_tiles sull'asse XY per ridurre il lavoro
    per tile (approx. BVH). Dentro ogni tile i raggi vengono processati
    in chunk di chunk_rays per controllare l'uso di memoria GPU.

    Returns hx_np, hy_np, hz_np — tutti (H, W) float32 numpy.
    """
    dev = torch.device(device)

    # Carica mesh su GPU
    verts    = torch.tensor(mesh.vertices.astype(np.float32), device=dev)
    faces_t  = torch.tensor(mesh.faces.astype(np.int32),     device=dev).long()
    v0 = verts[faces_t[:, 0]]   # (F, 3)
    v1 = verts[faces_t[:, 1]]
    v2 = verts[faces_t[:, 2]]
    del verts, faces_t

    e1 = v1 - v0   # (F, 3)
    e2 = v2 - v0   # (F, 3)
    del v1, v2

    # Pre-calcola costanti Möller–Trumbore per dir=(0,0,-1):
    #   h = dir × e2  =>  hx = e2y,  hy = -e2x,  hz = 0
    #   a = e1 · h
    #   sz = zmax - v0_z   (costante per tutti i raggi)
    hx_f   =  e2[:, 1]                          # (F,)
    hy_f   = -e2[:, 0]                          # (F,)
    a_f    = e1[:, 0] * hx_f + e1[:, 1] * hy_f # (F,)
    not_par = a_f.abs() > 1e-8                  # (F,) bool
    f_f    = torch.zeros_like(a_f)
    f_f[not_par] = 1.0 / a_f[not_par]
    sz_f   = float(zmax) - v0[:, 2]             # (F,)

    # Griglia raggi (numpy, per maschere tile veloci)
    xx, yy   = np.meshgrid(xs, ys)
    ox_np    = xx.ravel().astype(np.float32)
    oy_np    = yy.ravel().astype(np.float32)
    del xx, yy
    ray_ox   = torch.tensor(ox_np, device=dev)
    ray_oy   = torch.tensor(oy_np, device=dev)

    hit_z_all = torch.full((H * W,), float('nan'), dtype=torch.float32, device=dev)

    # Tiling XY
    centroids_np = mesh.triangles_center    # (F, 3) numpy
    bounds = mesh.bounds
    xmin_m, xmax_m = bounds[0, 0], bounds[1, 0]
    ymin_m, ymax_m = bounds[0, 1], bounds[1, 1]
    dx = (xmax_m - xmin_m) / n_tiles
    dy = (ymax_m - ymin_m) / n_tiles
    pad = resolution_mm_per_px

    n_faces   = e1.shape[0]
    use_tiles = n_faces > 20_000
    total_tiles = n_tiles * n_tiles if use_tiles else 1
    ti = 0

    def _process_tile(face_mask_np, ray_mask_np):
        face_idx_np = np.where(face_mask_np)[0]
        ray_idx_np  = np.where(ray_mask_np)[0]
        if face_idx_np.size == 0 or ray_idx_np.size == 0:
            return

        fi      = torch.tensor(face_idx_np, device=dev)
        v0_t    = v0[fi]       # (Ft, 3)
        e1_t    = e1[fi]
        e2_t    = e2[fi]
        hx_t    = hx_f[fi]    # (Ft,)
        hy_t    = hy_f[fi]
        f_t     = f_f[fi]
        np_t    = not_par[fi]
        sz_t    = sz_f[fi]     # (Ft,)  costante

        ray_idx = torch.tensor(ray_idx_np, device=dev)
        ox_t    = ray_ox[ray_idx]   # (Nr,)
        oy_t    = ray_oy[ray_idx]
        Nr      = ox_t.shape[0]

        local_best = torch.full((Nr,), float('inf'),  dtype=torch.float32, device=dev)
        local_hz   = torch.full((Nr,), float('nan'),  dtype=torch.float32, device=dev)

        for s in range(0, Nr, chunk_rays):
            en  = min(s + chunk_rays, Nr)
            oxc = ox_t[s:en, None]   # (C, 1)
            oyc = oy_t[s:en, None]

            # s-vector: (C, Ft)
            sx = oxc - v0_t[None, :, 0]
            sy = oyc - v0_t[None, :, 1]
            # sz_t: (Ft,)  — broadcast automatico a (C, Ft)

            u      = f_t * (sx * hx_t + sy * hy_t)    # (C, Ft)

            qx     = sy * e1_t[None, :, 2] - sz_t * e1_t[None, :, 1]
            qy     = sz_t * e1_t[None, :, 0] - sx * e1_t[None, :, 2]
            qz     = sx * e1_t[None, :, 1] - sy * e1_t[None, :, 0]

            v_coord = f_t * (-qz)
            e2q     = (e2_t[None, :, 0] * qx +
                       e2_t[None, :, 1] * qy +
                       e2_t[None, :, 2] * qz)
            t_val   = f_t * e2q                        # (C, Ft)

            hit = (np_t &
                   (u >= 0) & (u <= 1) &
                   (v_coord >= 0) & ((u + v_coord) <= 1) &
                   (t_val > 1e-6))                     # (C, Ft) bool

            inf_t   = torch.where(hit, t_val, torch.tensor(float('inf'), device=dev))
            min_t, _ = inf_t.min(dim=1)               # (C,)

            has     = min_t < float('inf')
            improve = has & (min_t < local_best[s:en])
            local_best[s:en] = torch.where(improve, min_t,          local_best[s:en])
            local_hz  [s:en] = torch.where(improve, float(zmax) - min_t, local_hz[s:en])

        # Aggiorna array globale (tiene il hit con Z più alto = più vicino)
        has_hit = ~torch.isnan(local_hz)
        prev_z  = hit_z_all[ray_idx]
        improve_g = has_hit & (torch.isnan(prev_z) | (local_hz > prev_z))
        good = torch.where(improve_g)[0]
        if len(good) > 0:
            hit_z_all[ray_idx[good]] = local_hz[good]

        return int(has_hit.sum())

    if use_tiles:
        print(f"[render_torch] Split {n_tiles}x{n_tiles} tiles  "
              f"(device={device}, chunk_rays={chunk_rays})")
        for ix in range(n_tiles):
            for iy in range(n_tiles):
                tx0 = xmin_m + ix * dx;     tx1 = xmin_m + (ix + 1) * dx
                ty0 = ymin_m + iy * dy;     ty1 = ymin_m + (iy + 1) * dy
                face_mask = ((centroids_np[:, 0] >= tx0) & (centroids_np[:, 0] < tx1) &
                             (centroids_np[:, 1] >= ty0) & (centroids_np[:, 1] < ty1))
                ray_mask  = ((ox_np >= tx0 - pad) & (ox_np <= tx1 + pad) &
                             (oy_np >= ty0 - pad) & (oy_np <= ty1 + pad))
                ti += 1
                n_hits = _process_tile(face_mask, ray_mask)
                print(f"  Tile {ti}/{total_tiles}  hits={n_hits or 0:,}", end='\r')
        print()
    else:
        print(f"[render_torch] Singola tile  (device={device}, chunk_rays={chunk_rays})")
        n_hits = _process_tile(np.ones(n_faces, dtype=bool),
                               np.ones(H * W,   dtype=bool))
        print(f"  hits={n_hits or 0:,}")

    total_hits = int((~torch.isnan(hit_z_all)).sum())
    print(f"[render_torch] Hits totali: {total_hits:,} / {H * W:,}")

    # Per raggi ortogonali: hit_x = ox, hit_y = oy (solo dove c'è un hit)
    nan_t   = torch.tensor(float('nan'), device=dev)
    has_all = ~torch.isnan(hit_z_all)
    hx_np   = torch.where(has_all, ray_ox, nan_t).cpu().numpy().reshape(H, W)
    hy_np   = torch.where(has_all, ray_oy, nan_t).cpu().numpy().reshape(H, W)
    hz_np   = hit_z_all.cpu().numpy().reshape(H, W)
    return hx_np, hy_np, hz_np


# =============================================================================
# Render GPU principale
# =============================================================================

def render_orthographic_topview_gpu(
    mesh,
    resolution_mm_per_px = 2.0,
    margin_mm            = 10.0,
    block_size           = 256,      # thread per blocco (CuPy backend)
    use_tiles            = True,     # suddividi mesh in tile XY
    n_tiles              = 8,        # griglia n_tiles x n_tiles
    fallback_cpu         = True,     # se nessun backend GPU → usa CPU trimesh
    device               = None,     # 'cuda', 'cuda:1', 'cpu', None (auto)
    chunk_rays           = 4000,     # raggi per chunk (PyTorch backend)
    xmin_override=None, xmax_override=None,
    ymin_override=None, ymax_override=None,
):
    """
    Render ortografico top-view via ray casting GPU/CPU.
    Interfaccia identica a render_orthographic_topview() CPU.

    Priorità backend (automatica):
      1. CuPy   — kernel CUDA custom (se cupy installato e device GPU)
      2. PyTorch — Möller–Trumbore vettorizzato (se torch installato e device GPU)
      3. CPU     — trimesh BVH (fallback)

    Parameters
    ----------
    device : str | None
        'cuda' | 'cuda:0' | 'cuda:1' ecc. → forza GPU
        'cpu'                              → forza CPU trimesh
        None                               → auto (GPU se disponibile)

    Returns
    -------
    render_rgb  : (H, W, 3) uint8
    depth_map   : (H, W)    float32  Z in mm, NaN = no hit
    xyz_map     : (H, W, 3) float32
    origin_xy   : (xmin, ymin)
    res         : resolution_mm_per_px
    """
    # ── Determinazione device ────────────────────────────────────────────────
    dev_str    = str(device) if device is not None else ''
    want_gpu   = device != 'cpu'   # False solo se esplicitamente 'cpu'

    # Auto-detect: se device=None, preferisci GPU
    if device is None:
        if CUPY_OK:
            effective_device = 'cuda'
        elif TORCH_OK and torch.cuda.is_available():
            effective_device = 'cuda'
        else:
            effective_device = 'cpu'
            want_gpu = False
    else:
        effective_device = device

    # ── Griglia raggi (comune a tutti i backend) ─────────────────────────────
    bounds = mesh.bounds
    xmin = xmin_override if xmin_override is not None else bounds[0, 0] - margin_mm
    xmax = xmax_override if xmax_override is not None else bounds[1, 0] + margin_mm
    ymin = ymin_override if ymin_override is not None else bounds[0, 1] - margin_mm
    ymax = ymax_override if ymax_override is not None else bounds[1, 1] + margin_mm
    zmax = float(bounds[1, 2]) + 10.0

    xs = np.arange(xmin, xmax, resolution_mm_per_px, dtype=np.float32)
    ys = np.arange(ymin, ymax, resolution_mm_per_px, dtype=np.float32)
    W, H  = len(xs), len(ys)
    n_rays = W * H

    print(f"[render_gpu] Grid  : {W} x {H} px  ({resolution_mm_per_px} mm/px)")
    print(f"             Rays  : {n_rays:,}")

    def _build_output(hx_np, hy_np, hz_np):
        depth_map = hz_np.astype(np.float32)
        xyz_map   = np.stack([hx_np, hy_np, hz_np], axis=2).astype(np.float32)
        z_valid   = depth_map[~np.isnan(depth_map)]
        if z_valid.size > 0:
            z_min, z_max = z_valid.min(), z_valid.max()
            denom = (z_max - z_min) if z_max > z_min else 1.0
            gray  = np.where(np.isnan(depth_map), 0,
                             (depth_map - z_min) / denom * 255).astype(np.uint8)
        else:
            gray = np.zeros((H, W), dtype=np.uint8)
        render_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        print(f"[render_gpu] Done. origin_xy=({xmin:.2f}, {ymin:.2f}) mm")
        return render_rgb, depth_map, xyz_map, (xmin, ymin), resolution_mm_per_px

    # ── Backend 1: CuPy (kernel CUDA custom) ────────────────────────────────
    if CUPY_OK and want_gpu:
        try:
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            print(f"[render_gpu] Backend: CuPy  ({gpu_name})")

            xx, yy  = np.meshgrid(xs, ys)
            ray_ox  = cp.asarray(xx.ravel())
            ray_oy  = cp.asarray(yy.ravel())
            del xx, yy

            hit_x = cp.full(n_rays, np.nan, dtype=np.float32)
            hit_y = cp.full(n_rays, np.nan, dtype=np.float32)
            hit_z = cp.full(n_rays, np.nan, dtype=np.float32)

            kernel = _compile_kernel()

            def _launch(sub_mesh, ox_gpu=None, oy_gpu=None,
                        hit_x_out=None, hit_y_out=None, hit_z_out=None, n=None):
                verts_gpu = cp.asarray(sub_mesh.vertices.astype(np.float32))
                faces_gpu = cp.asarray(sub_mesh.faces.astype(np.int32))
                n_f = sub_mesh.faces.shape[0]
                n_r = n if n is not None else n_rays
                grid = (n_r + block_size - 1) // block_size
                kernel(
                    (grid,), (block_size,),
                    (verts_gpu, faces_gpu,
                     ox_gpu if ox_gpu is not None else ray_ox,
                     oy_gpu if oy_gpu is not None else ray_oy,
                     np.float32(zmax), np.int32(n_f), np.int32(n_r),
                     hit_x_out if hit_x_out is not None else hit_x,
                     hit_y_out if hit_y_out is not None else hit_y,
                     hit_z_out if hit_z_out is not None else hit_z,
                    )
                )

            if use_tiles and mesh.faces.shape[0] > 20_000:
                print(f"[render_gpu] Split {n_tiles}x{n_tiles} tiles...")
                tiles = _split_mesh_xy(mesh, n_tiles)
                print(f"[render_gpu] {len(tiles)} tile(s) non vuote.")
                ox_np = ray_ox.get()
                oy_np = ray_oy.get()
                for ti, (tx0, tx1, ty0, ty1, sub) in enumerate(tiles):
                    pad  = resolution_mm_per_px
                    mask = ((ox_np >= tx0 - pad) & (ox_np <= tx1 + pad) &
                            (oy_np >= ty0 - pad) & (oy_np <= ty1 + pad))
                    idx  = np.where(mask)[0]
                    if len(idx) == 0:
                        continue
                    ox_t = cp.asarray(ox_np[idx])
                    oy_t = cp.asarray(oy_np[idx])
                    hx_t = cp.full(len(idx), np.nan, dtype=np.float32)
                    hy_t = cp.full(len(idx), np.nan, dtype=np.float32)
                    hz_t = cp.full(len(idx), np.nan, dtype=np.float32)
                    _launch(sub, ox_gpu=ox_t, oy_gpu=oy_t,
                            hit_x_out=hx_t, hit_y_out=hy_t, hit_z_out=hz_t,
                            n=len(idx))
                    idx_gpu = cp.asarray(idx)
                    has_hit = ~cp.isnan(hz_t)
                    prev_z  = hit_z[idx_gpu]
                    improve = has_hit & (cp.isnan(prev_z) | (hz_t > prev_z))
                    good    = cp.where(improve)[0]
                    if len(good) > 0:
                        hit_x[idx_gpu[good]] = hx_t[good]
                        hit_y[idx_gpu[good]] = hy_t[good]
                        hit_z[idx_gpu[good]] = hz_t[good]
                    print(f"  Tile {ti+1}/{len(tiles)}  facce={sub.faces.shape[0]:,}  "
                          f"hits={int(has_hit.sum()):,}", end='\r')
                print()
            else:
                print(f"[render_gpu] Kernel su tutta la mesh ({mesh.faces.shape[0]:,} facce)...")
                _launch(mesh)

            cp.cuda.Stream.null.synchronize()
            hx_np = hit_x.get().reshape(H, W)
            hy_np = hit_y.get().reshape(H, W)
            hz_np = hit_z.get().reshape(H, W)
            del hit_x, hit_y, hit_z, ray_ox, ray_oy
            print(f"[render_gpu] Hits: {int(np.sum(~np.isnan(hz_np))):,} / {n_rays:,}")
            return _build_output(hx_np, hy_np, hz_np)

        except Exception as e:
            print(f"[render_gpu] CuPy fallito ({e}) — provo PyTorch...")

    # ── Backend 2: PyTorch GPU ───────────────────────────────────────────────
    if TORCH_OK and want_gpu:
        torch_dev = effective_device if effective_device != 'cpu' else 'cuda'
        try:
            if not torch.cuda.is_available() and torch_dev != 'cpu':
                raise RuntimeError("CUDA non disponibile per PyTorch")
            print(f"[render_gpu] Backend: PyTorch  ({torch_dev})")
            hx_np, hy_np, hz_np = _render_torch_tiles(
                mesh, xs, ys, W, H, zmax,
                resolution_mm_per_px, torch_dev, n_tiles, chunk_rays,
            )
            return _build_output(hx_np, hy_np, hz_np)
        except Exception as e:
            print(f"[render_gpu] PyTorch fallito ({e}) — provo CPU...")

    # ── Backend 3: CPU trimesh BVH ───────────────────────────────────────────
    if fallback_cpu:
        print("[render_gpu] Backend: CPU trimesh BVH")
        from .pipeline import render_orthographic_topview
        return render_orthographic_topview(
            mesh, resolution_mm_per_px, margin_mm,
            xmin_override=xmin_override, xmax_override=xmax_override,
            ymin_override=ymin_override, ymax_override=ymax_override,
        )

    raise RuntimeError(
        "Nessun backend disponibile. "
        "Installa torch (pip install torch) o cupy (pip install cupy-cuda12x), "
        "oppure imposta fallback_cpu=True."
    )


# =============================================================================
# Test standalone
# =============================================================================

if __name__ == "__main__":
    import time, sys

    if len(sys.argv) < 2:
        print("Uso: python3 render_gpu.py <mesh.ply> [resolution_mm]")
        sys.exit(1)

    mesh_path = sys.argv[1]
    res       = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    print(f"[Test] Carico mesh: {mesh_path}")
    import trimesh
    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh.apply_scale(1000.0)
    print(f"[Test] Mesh: {len(mesh.vertices)} vertici, {len(mesh.faces)} facce")

    t0 = time.time()
    render_rgb, depth_map, xyz_map, origin_xy, res_out = \
        render_orthographic_topview_gpu(mesh, resolution_mm_per_px=res)
    elapsed = time.time() - t0

    print(f"\n[Test] Tempo: {elapsed:.2f}s")
    cv2.imwrite("test_render_gpu.png", render_rgb)
    print("[Test] Salvato: test_render_gpu.png")
