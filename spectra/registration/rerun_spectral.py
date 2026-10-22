"""Spectral point cloud → Rerun recording (last step of the registration stage).

Reads the point cloud the registration has just exported to ``output_dir``
(memmap files ``spectral_pointcloud_<sample>_{xyz,spectra}.bin`` + ``_meta.npy``
+ ``_wavelengths.npy`` when ``export.npy`` is on, otherwise the binary
``spectral_pointcloud_<sample>.ply``) and writes:

    <output_dir>/spectral.rrd   data + layout, open with ``rerun spectral.rrd``
    <output_dir>/spectral.rbl   layout only, ``rerun spectral.rrd spectral.rbl``

Layout (both views spin around the sample's vertical axis; the vertical axis
follows the vision run's ``z_axis_points_down``, read from its
``reconstruction_metadata.json``, so the sample is never shown upside down):

    top    : one band at a time, in ``colormap`` (default gray) — pick it on
             the ``wavelength_nm`` timeline (or ``band`` = band index) in the
             time panel at the bottom
    bottom : first three principal components of the spectra as RGB
             (PC1 -> R, PC2 -> G, PC3 -> B)

Size control (``registration.rerun`` in the YAML): every logged band costs
``n_points x 4`` bytes, so the recording is ``max_points`` random points x
every ``band_step``-th band. 693,974 points x 290 bands is ~810 MB of colours,
which is what made the hand-made recordings too heavy for the viewer.

Never raises from :func:`maybe_write_spectral_rerun`: the point cloud is
already on disk, so a failure here only prints a WARNING.
"""

from __future__ import annotations

import re
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

APPLICATION_ID = "spectra_spectral"
TIMELINE_NM = "wavelength_nm"
TIMELINE_BAND = "band"

_CV2_COLORMAPS = {
    "gray": None,                       # identity LUT, no OpenCV colormap
    "viridis": "COLORMAP_VIRIDIS",
    "turbo": "COLORMAP_TURBO",
    "inferno": "COLORMAP_INFERNO",
    "magma": "COLORMAP_MAGMA",
    "plasma": "COLORMAP_PLASMA",
    "cividis": "COLORMAP_CIVIDIS",
    "jet": "COLORMAP_JET",
}


# =============================================================================
# Readers
# =============================================================================

def _cloud_base(output_dir: Path, sample_name: str) -> Path:
    return Path(output_dir) / f"spectral_pointcloud_{sample_name}"


def _read_memmap(base: Path):
    """``<base>_{xyz,spectra}.bin`` + ``_meta.npy`` + ``_wavelengths.npy``."""
    meta = np.load(f"{base}_meta.npy").tolist()          # [n, 3, n, bands]
    n, bands = int(meta[0]), int(meta[3])
    xyz_path, spec_path = Path(f"{base}_xyz.bin"), Path(f"{base}_spectra.bin")
    for p, expected in ((xyz_path, n * 3 * 4), (spec_path, n * bands * 4)):
        size = p.stat().st_size
        if size != expected:
            raise ValueError(
                f"{p.name} is {size:,} bytes, expected {expected:,} "
                f"({n:,} points x {bands} bands): truncated or incomplete file."
            )
    xyz = np.memmap(xyz_path, dtype=np.float32, mode="r", shape=(n, 3))
    spectra = np.memmap(spec_path, dtype=np.float32, mode="r", shape=(n, bands))
    wl_path = Path(f"{base}_wavelengths.npy")
    wl = np.load(wl_path).astype(np.float64) if wl_path.exists() else np.zeros(0)
    return xyz, spectra, (wl if wl.size == bands else None)


def _read_ply(path: Path):
    """Binary little-endian PLY with float ``x y z`` + one float per band."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path.name}: no 'end_header'.")
            header.append(line.decode("ascii", errors="replace").strip())
            if header[-1] == "end_header":
                break
        offset = f.tell()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path.name}: expected a binary_little_endian PLY.")
    n = int(next(h for h in header if h.startswith("element vertex")).split()[-1])
    props = [h.split() for h in header if h.startswith("property")]
    if {p[1] for p in props} != {"float"} or [p[2] for p in props[:3]] != ["x", "y", "z"]:
        raise ValueError(f"{path.name}: expected float x, y, z + float bands.")
    names = [p[2] for p in props[3:]]
    bands = len(names)
    expected = offset + n * (3 + bands) * 4
    if path.stat().st_size != expected:
        raise ValueError(
            f"{path.name} is {path.stat().st_size:,} bytes, expected {expected:,}: "
            "truncated or incomplete file."
        )
    data = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(n, 3 + bands))
    wl = []
    for nm in names:
        m = re.fullmatch(r"b\d+_(\d+(?:\.\d+)?)nm", nm)
        if not m:
            wl = None
            break
        wl.append(float(m.group(1)))
    return data[:, :3], data[:, 3:], (np.asarray(wl) if wl else None)


def find_and_read_cloud(output_dir: Path, sample_name: str):
    """Return ``(xyz, spectra, wavelengths_or_None, source_path)`` (memmaps)."""
    base = _cloud_base(output_dir, sample_name)
    ply = Path(f"{base}.ply")
    if Path(f"{base}_meta.npy").exists() and Path(f"{base}_spectra.bin").exists():
        try:
            return (*_read_memmap(base), Path(f"{base}_spectra.bin"))
        except ValueError as exc:
            if not ply.exists():
                raise
            print(f"[Rerun spectral] WARNING: {exc} Falling back to {ply.name}.")
    if ply.exists():
        return (*_read_ply(ply), ply)
    raise FileNotFoundError(
        f"no spectral point cloud for '{sample_name}' in {output_dir} "
        f"(looked for {base.name}_spectra.bin and {base.name}.ply)."
    )


# =============================================================================
# Colour helpers
# =============================================================================

def _lut(name: str) -> np.ndarray:
    """256 x 3 uint8 RGB lookup table from OpenCV's colormaps."""
    if name == "gray":
        return np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    import cv2

    code = getattr(cv2, _CV2_COLORMAPS[name])
    bgr = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), code)
    return np.ascontiguousarray(bgr[:, 0, ::-1])


def _stretch_u8(values: np.ndarray) -> np.ndarray:
    """1st–99th percentile stretch to 0..255."""
    lo, hi = np.percentile(values, [1.0, 99.0])
    t = (values - lo) / (hi - lo) if hi > lo else np.zeros_like(values)
    return np.clip(np.round(t * 255.0), 0, 255).astype(np.uint8)


def _colorize(values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """1st–99th percentile stretch, then colormap."""
    return lut[_stretch_u8(values)]


def _pca_rgb(spectra: np.ndarray):
    """First three principal components, each stretched, as R, G, B.

    Returns ``(rgb uint8 [n, 3], [explained-variance ratio of each PC])``.
    """
    x = spectra - spectra.mean(axis=0, dtype=np.float64).astype(np.float32)
    cov = (x.T @ x).astype(np.float64) / max(len(x) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)                  # ascending
    total = float(evals.sum())
    rgb = np.zeros((len(x), 3), dtype=np.uint8)
    ratios = []
    for c in range(min(3, evecs.shape[1])):
        v = evecs[:, -1 - c]
        if v[np.argmax(np.abs(v))] < 0:                 # deterministic sign
            v = -v
        rgb[:, c] = _stretch_u8(x @ v.astype(np.float32))
        ratios.append(float(evals[-1 - c]) / total if total > 0 else float("nan"))
    return rgb, ratios


# =============================================================================
# Writer
# =============================================================================

def write_spectral_rerun(
    output_dir: Path,
    sample_name: str,
    *,
    max_points: Optional[int] = 150_000,
    band_step: int = 2,
    colormap: str = "gray",
    point_radius_px: float = 2.0,
    spin_speed: float = 0.3,
    save_rbl: bool = True,
    rrd_basename: str = "spectral.rrd",
    rbl_basename: str = "spectral.rbl",
    z_down: bool = True,
) -> Path:
    """Write ``spectral.rrd`` (+ ``spectral.rbl``) next to the point cloud."""
    import rerun as rr
    import rerun.blueprint as rrb

    t0 = time.time()
    output_dir = Path(output_dir)
    xyz_all, spec_all, wl, src = find_and_read_cloud(output_dir, sample_name)
    n_all, n_bands = spec_all.shape
    print(f"[Rerun spectral] Source: {src.name}  ({n_all:,} points x {n_bands} bands)")

    # -- subsample (random, fixed seed; sorted for sequential disk reads) ----
    if max_points and n_all > max_points:
        idx = np.sort(np.random.default_rng(0).choice(n_all, max_points, replace=False))
    else:
        idx = np.arange(n_all)
    xyz = np.asarray(xyz_all[idx], dtype=np.float32)
    spec = np.asarray(spec_all[idx], dtype=np.float32)
    ok = np.isfinite(xyz).all(axis=1) & np.isfinite(spec).all(axis=1)
    xyz, spec = xyz[ok], spec[ok]
    n = len(xyz)
    if n == 0:
        raise ValueError("no finite points to display.")
    if (~ok).any():
        print(f"[Rerun spectral] Dropped {int((~ok).sum()):,} points with NaN/inf.")

    band_idx = np.arange(0, n_bands, band_step)
    est_mb = n * 4 * (len(band_idx) + 1 + 3) / 1e6
    print(f"[Rerun spectral] Logging {n:,} points x {len(band_idx)} bands "
          f"(every {band_step}) → ~{est_mb:.0f} MB uncompressed")

    lut = _lut(colormap)
    pca_rgb, ratios = _pca_rgb(spec)
    var = "/".join(f"{r:.0%}" for r in ratios)

    # -- camera: orbit (and spin) around the sample's centre ------------------
    # Vision's output frame has +Z pointing DOWN by default
    # (vision.output.z_axis_points_down): "up" is then -Z.
    up = np.array([0.0, 0.0, -1.0 if z_down else 1.0], dtype=np.float32)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    centre = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    eye_pos = centre + diag * (np.array([0.0, -0.9, 0.0], dtype=np.float32) + 0.7 * up)

    def _eye():
        return rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=eye_pos.tolist(),
            look_target=centre.tolist(),
            eye_up=up.tolist(),
            spin_speed=float(spin_speed),
        )

    timeline = TIMELINE_NM if wl is not None else TIMELINE_BAND
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(
                name=f"Band — choose it on the '{timeline}' timeline",
                origin="/", contents="/band/**", eye_controls=_eye(),
            ),
            rrb.Spatial3DView(
                name=f"PCA — PC1/PC2/PC3 as R/G/B ({var} of variance)",
                origin="/", contents="/pca/**", eye_controls=_eye(),
            ),
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded", timeline=timeline, play_state="paused"),
    )

    rrd_path = output_dir / rrd_basename
    rec = rr.RecordingStream(APPLICATION_ID, recording_id=f"{sample_name}_spectral")
    rec.save(rrd_path, default_blueprint=blueprint)
    radius = rr.Radius.ui_points(float(point_radius_px))

    rec.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN if z_down
            else rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rec.log("pca/cloud", rr.Points3D(xyz, colors=pca_rgb, radii=radius), static=True)
    rec.log("band/cloud", rr.Points3D(xyz, radii=radius), static=True)
    for bi in band_idx:
        if wl is not None:
            rec.set_time(TIMELINE_NM, sequence=int(round(float(wl[bi]))))
        rec.set_time(TIMELINE_BAND, sequence=int(bi))
        rec.log("band/cloud", rr.Points3D.from_fields(colors=_colorize(spec[:, bi], lut)))
    rec.flush()
    rec.disconnect()

    if save_rbl:
        blueprint.save(APPLICATION_ID, str(output_dir / rbl_basename))

    size_mb = rrd_path.stat().st_size / 1e6
    print(f"[Rerun spectral] Saved {rrd_path} ({size_mb:.0f} MB, "
          f"PC1/PC2/PC3 = {var} of variance) in {time.time() - t0:.1f}s")
    if save_rbl:
        print(f"[Rerun spectral] Layout {output_dir / rbl_basename}")
    print(f"[Rerun spectral] Open with: rerun {rrd_basename}"
          + (f" {rbl_basename}" if save_rbl else ""))
    return rrd_path


def _z_axis_points_down(mesh_path) -> bool:
    """``z_axis_points_down`` of the vision run that produced ``mesh_path``."""
    import json

    meta = Path(mesh_path).parent / "reconstruction_metadata.json" if mesh_path else None
    try:
        return bool(json.loads(meta.read_text(encoding="utf-8"))["z_axis_points_down"])
    except Exception:
        print(f"[Rerun spectral] z_axis_points_down not found in {meta}: "
              "assuming true (vision default).")
        return True


def maybe_write_spectral_rerun(cfg, output_dir, sample_name: str,
                               mesh_path=None) -> Optional[Path]:
    """Run :func:`write_spectral_rerun` if ``cfg.rerun.enabled``; never raises.

    ``cfg`` is a :class:`spectra.config.RegistrationConfig`; ``mesh_path``
    defaults to ``cfg.paths.mesh`` (batch passes each sample's own mesh).
    """
    rc = cfg.rerun
    if not rc.enabled:
        return None
    if not cfg.export.save_pointcloud:
        print("[Rerun spectral] Skipped: export.save_pointcloud is false.")
        return None
    print("\n[Rerun spectral] Writing spectral recording...")
    try:
        return write_spectral_rerun(
            Path(output_dir), sample_name,
            max_points=rc.max_points,
            band_step=rc.band_step,
            colormap=rc.colormap,
            point_radius_px=rc.point_radius_px,
            spin_speed=rc.spin_speed,
            save_rbl=rc.save_rbl,
            rrd_basename=rc.rrd_basename,
            rbl_basename=rc.rbl_basename,
            z_down=_z_axis_points_down(mesh_path or cfg.paths.mesh),
        )
    except Exception as exc:  # the point cloud is already on disk
        print(f"[Rerun spectral] WARNING: recording NOT written: {exc}")
        traceback.print_exc()
        return None
