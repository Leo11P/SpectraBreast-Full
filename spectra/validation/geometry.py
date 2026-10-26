# -*- coding: utf-8 -*-
"""Geometry for the validation: rigid transforms, Kabsch, surface sampling,
signed point-to-surface distance, point-to-plane ICP, visibility from the
cameras and error statistics. numpy + scipy + trimesh only. Units: mm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


# ----------------------------------------------------------- transforms ---
def to_h(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]


def rotate(T: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    return np.asarray(vecs, dtype=np.float64) @ T[:3, :3].T


def inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    return to_h(R.T, -R.T @ T[:3, 3])


def rot_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R[:3, :3]) - 1.0) / 2.0))))


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Proper rigid T (no scale, no reflection) minimising |T(src) - dst|^2."""
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0])
    R = Vt.T @ D @ U.T
    return to_h(R, md - R @ ms)


# ------------------------------------------------------------- sampling ---
@dataclass
class Samples:
    points: np.ndarray                 # (N, 3)
    normals: Optional[np.ndarray]      # (N, 3) unit, or None (point cloud)


def sample_surface(vertices: np.ndarray, faces: np.ndarray, spacing: float,
                   keep: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                   coarse: float = 5.0) -> Samples:
    """Centroids + normals of the faces after splitting them to edges <= spacing
    (an even sampling of the surface). ``keep(points) -> bool mask`` crops the
    surface (applied on a coarse split first, then on the samples)."""
    from trimesh.remesh import subdivide_to_size

    v, f = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    if keep is not None:
        if coarse > spacing:
            v, f = subdivide_to_size(v, f, max_edge=coarse, max_iter=20)
        f = f[keep(v[f].mean(axis=1))]
        if len(f) == 0:
            return Samples(np.zeros((0, 3)), np.zeros((0, 3)))
        used, f = np.unique(f, return_inverse=True)
        v, f = v[used], f.reshape(-1, 3)
    v, f = subdivide_to_size(v, f, max_edge=spacing, max_iter=30)
    tri = v[f]
    c = tri.mean(axis=1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-12
    if keep is not None:
        ok &= keep(c)
    return Samples(c[ok], n[ok] / norm[ok, None])


class SurfaceDistance:
    """Signed distance to a densely sampled surface. Among the k nearest
    samples, those the point projects onto within one sample spacing (flat
    neighbourhood) give the distance along their normal, and the smallest
    one is taken; if there are none (edges, far points), the Euclidean
    distance to the nearest sample, with the sign of its normal side.
    Positive = on the side the normals point to (outside, for an STL)."""

    def __init__(self, samples: Samples, spacing: float, k: int = 6):
        from scipy.spatial import cKDTree

        self.s = samples
        self.spacing = float(spacing)
        self.k = int(k)
        self.tree = cKDTree(samples.points)

    def signed(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        k = min(self.k, len(self.s.points))
        d, j = self.tree.query(pts, k=k, workers=-1)
        d, j = d.reshape(len(pts), k), j.reshape(len(pts), k)
        if self.s.normals is None:
            return d[:, 0]
        diff = pts[:, None, :] - self.s.points[j]                 # (N, k, 3)
        along = np.einsum("nkj,nkj->nk", diff, self.s.normals[j])
        lateral = np.sqrt(np.maximum(d * d - along * along, 0.0))
        cand = np.where(lateral <= self.spacing, np.abs(along), np.inf)
        best = np.argmin(cand, axis=1)
        rows = np.arange(len(pts))
        flat = np.isfinite(cand[rows, best])
        along0 = along[:, 0]
        euclid = np.where(along0 < 0, -1.0, 1.0) * d[:, 0]
        return np.where(flat, along[rows, best], euclid)


# ------------------------------------------------------------------ ICP ---
def _small_rotation(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def icp_point_to_plane(src: np.ndarray, target: SurfaceDistance, max_corr: float,
                       iters: int = 50, tol: float = 1e-7) -> tuple[np.ndarray, dict]:
    """Rigid T (no scale) moving ``src`` onto the target surface, minimising the
    point-to-plane distance of the pairs closer than ``max_corr``."""
    T = np.eye(4)
    pts_t, nrm_t = target.s.points, target.s.normals
    info = {"iterations": 0, "pairs": 0, "rms_mm": float("nan"), "converged": False}
    for it in range(1, iters + 1):
        p = apply(T, src)
        d, j = target.tree.query(p, distance_upper_bound=max_corr, workers=-1)
        ok = np.isfinite(d)
        if ok.sum() < 6:
            break
        p, q, n = p[ok], pts_t[j[ok]], nrm_t[j[ok]]
        r = np.einsum("ij,ij->i", p - q, n)
        A = np.hstack([np.cross(p, n), n])
        x = np.linalg.solve(A.T @ A + 1e-9 * np.eye(6), -A.T @ r)
        T = to_h(_small_rotation(x[:3]), x[3:]) @ T
        info.update(iterations=it, pairs=int(ok.sum()), rms_mm=float(np.sqrt(np.mean(r * r))))
        if np.linalg.norm(x) < tol:
            info["converged"] = True
            break
    return T, info


# ------------------------------------------------------------ visibility ---
def _project(X: np.ndarray, R: np.ndarray, C: np.ndarray, K: np.ndarray, W: int, H: int):
    """Pixel (u, v), depth z and in-image mask of world points for one camera."""
    Xc = (X - C) @ R
    z = Xc[:, 2]
    zs = np.where(z > 1e-6, z, np.inf)
    u = K[0, 0] * Xc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zs + K[1, 2]
    return u, v, z, (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def depth_buffer(occluders: np.ndarray, R: np.ndarray, C: np.ndarray, K: np.ndarray,
                 W: int, H: int, buffer_scale: float) -> np.ndarray:
    """Nearest depth of the occluder points per buffer pixel (flat, inf = empty)."""
    Wb, Hb = int(W * buffer_scale) + 1, int(H * buffer_scale) + 1
    u, v, z, ok = _project(occluders, R, C, K, W, H)
    buf = np.full(Wb * Hb, np.inf)
    np.minimum.at(buf, (v[ok] * buffer_scale).astype(np.int64) * Wb
                  + (u[ok] * buffer_scale).astype(np.int64), z[ok])
    return buf


def _seen_in_view(P: np.ndarray, N: np.ndarray, buf: np.ndarray, R: np.ndarray, C: np.ndarray,
                  K: np.ndarray, W: int, H: int, buffer_scale: float, tol: float,
                  min_cos: float) -> np.ndarray:
    """Points in front of the camera, inside its image, facing it (cos > min_cos)
    and not behind the depth buffer ``buf`` (see depth_buffer)."""
    Wb = int(W * buffer_scale) + 1
    u, v, z, ok = _project(P, R, C, K, W, H)
    view = C - P
    cos = np.einsum("ij,ij->i", N, view) / np.maximum(np.linalg.norm(view, axis=1), 1e-12)
    ok &= cos > min_cos
    footprint = z / (K[0, 0] * buffer_scale)                      # mm per buffer pixel
    tan = np.sqrt(np.clip(1 - cos * cos, 0, 1)) / np.maximum(cos, min_cos)
    thr = tol + np.minimum(footprint * tan, 5.0)
    idx = np.flatnonzero(ok)
    zb = buf[(v[idx] * buffer_scale).astype(np.int64) * Wb + (u[idx] * buffer_scale).astype(np.int64)]
    seen = np.zeros(len(P), dtype=bool)
    seen[idx[z[idx] <= zb + thr[idx]]] = True
    return seen


def visible_mask(samples: Samples, occluders: np.ndarray, cam_T: np.ndarray, K: np.ndarray,
                 image_wh: Sequence[int], buffer_scale: float = 0.5, tol: float = 1.0,
                 min_cos: float = 0.1) -> np.ndarray:
    """Samples seen by at least one camera: in front of it, inside its image,
    facing it (cos > min_cos) and not hidden by ``occluders`` (dense points of
    the surroundings) in a depth buffer at ``buffer_scale`` of the image.
    ``cam_T``: [V, 4, 4] camera-to-world (OpenCV axes), same frame and unit as
    the points; ``K``: [V, 3, 3]."""
    W, H = int(image_wh[0]), int(image_wh[1])
    P, N = samples.points, samples.normals
    seen = np.zeros(len(P), dtype=bool)
    for T, Kv in zip(cam_T, K):
        todo = np.flatnonzero(~seen)
        if len(todo) == 0:
            break
        R, C = T[:3, :3], T[:3, 3]
        buf = depth_buffer(occluders, R, C, Kv, W, H, buffer_scale)
        seen[todo[_seen_in_view(P[todo], N[todo], buf, R, C, Kv, W, H, buffer_scale, tol, min_cos)]] = True
    return seen


def visible_per_view(points: np.ndarray, normals: np.ndarray, occluders: np.ndarray,
                     cam_T: np.ndarray, K: np.ndarray, image_wh: Sequence[int],
                     buffer_scale: float = 0.5, tol: float = 1.0, min_cos: float = 0.1) -> np.ndarray:
    """[V, N] bool: point n seen by camera v (same test as visible_mask)."""
    W, H = int(image_wh[0]), int(image_wh[1])
    out = np.zeros((len(cam_T), len(points)), dtype=bool)
    for i, (T, Kv) in enumerate(zip(cam_T, K)):
        R, C = T[:3, :3], T[:3, 3]
        buf = depth_buffer(occluders, R, C, Kv, W, H, buffer_scale)
        out[i] = _seen_in_view(points, normals, buf, R, C, Kv, W, H, buffer_scale, tol, min_cos)
    return out


def ray_first_hit(origin: np.ndarray, direction: np.ndarray, tri: np.ndarray) -> Optional[float]:
    """Distance along the ray to the nearest triangle of ``tri`` [M, 3, 3] it
    crosses (Moller-Trumbore), or None."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = 1.0 / np.where(ok, det, 1.0)
    s = origin - tri[:, 0]
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return float(t[hit].min()) if hit.any() else None


# ------------------------------------------------------------ statistics ---
def dist_stats(d: np.ndarray, thresholds: Sequence[float]) -> dict:
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return {"n": 0}
    a = np.abs(d)
    out = {
        "n": int(d.size),
        "rmse_mm": float(np.sqrt(np.mean(d * d))),
        "mean_signed_mm": float(d.mean()),
        "mean_abs_mm": float(a.mean()),
        "median_abs_mm": float(np.median(a)),
        "p95_abs_mm": float(np.percentile(a, 95)),
        "max_abs_mm": float(a.max()),
    }
    for t in thresholds:
        out[f"pct_lt_{t:g}mm"] = float(100.0 * np.mean(a < t))
    return out


def height_profile(h: np.ndarray, d: np.ndarray, bin_mm: float) -> dict:
    """Signed error vs height above the marker plane: per-bin stats + linear fit."""
    h, d = np.asarray(h, dtype=np.float64), np.asarray(d, dtype=np.float64)
    if h.size == 0:
        return {"bins": []}
    lo = math.floor(h.min() / bin_mm) * bin_mm
    edges = np.arange(lo, h.max() + bin_mm, bin_mm)
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (h >= a) & (h < b)
        if m.any():
            bins.append({"z_from_mm": float(a), "z_to_mm": float(b), "n": int(m.sum()),
                         "rmse_mm": float(np.sqrt(np.mean(d[m] ** 2))),
                         "mean_signed_mm": float(d[m].mean())})
    out = {"bins": bins}
    if h.size > 10 and np.ptp(h) > bin_mm:
        slope, intercept = np.polyfit(h, d, 1)
        out.update(slope_mm_per_mm=float(slope), intercept_mm=float(intercept))
    return out
