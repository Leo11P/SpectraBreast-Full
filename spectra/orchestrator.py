"""End-to-end orchestrator: ``spectra full`` (vision → registration).

Responsibilities
================
1. **Path resolution.** For a given ``UnifiedConfig`` and (optionally) a
   sample name, fill in any per-sample path that the user left ``None`` in
   the YAML, following the layout convention documented below and in
   ``configs/default.yaml``.

2. **Stage skipping.** When ``spectra full`` is invoked and the vision
   outputs for the current run date already exist, the vision stage is
   skipped automatically (``--force-vision`` overrides).

3. **Cross-stage I/O.** Which vision run registration consumes
   (``surface_mesh.ply`` + ``aruco_markers_3d.json``), unless the user set
   ``registration.paths.mesh`` / ``aruco_json`` explicitly:

   * ``spectra full`` with the vision stage: exactly the run that vision just
     produced (or, when vision was skipped, the run for this date that caused
     the skip). Never a fallback: if those files are missing, it is an error.
   * ``spectra registration`` (or ``full`` without the vision stage): this
     date's run if complete, else the most recent dated run that is, with a
     visible warning naming the run that was picked.

   The ArUco marker edge length has ONE source: ``vision.aruco.marker_edge_length_mm``.
   Registration always inherits it (setting it under ``registration:`` is
   rejected), and refuses a vision run whose ``aruco_markers_3d.json`` was
   built with a different edge length.

Layout convention
=================
Inputs — stage first, sample second, for *every* mode including ``batch``::

    DATA/
      input_vision/<sample>[_<device>]/     # e.g. SB021_iphone, SB021_realsense
        rgb/                                #   images (or directly in the folder)
        poses/                              #   optional
        camera_parameters/                  #   optional
      input_registration/<sample>/
        <sample>_raw.hdr                    #   ENVI cube (+ its .img/.raw)
        <sample>_raw.png                    #   ROI mode only

When several ``<sample>_*`` vision folders exist, the one whose suffix
contains ``iphone`` wins; otherwise the first in alphabetical order.

Outputs — grouped by run date (``dd-mm-yyyy``), then by sample::

    RESULTS/<dd-mm-yyyy>/<sample>/vision/
    RESULTS/<dd-mm-yyyy>/<sample>/registration/
    RESULTS/<dd-mm-yyyy>/_batch/batch_summary.xlsx

Re-running any command on the same day **overwrites** that day's folder
rather than creating a sibling. Pin a different day with
``results_date: "dd-mm-yyyy"`` in the YAML.

Absolute paths always win: anything you set explicitly under
``vision.input.*``, ``vision.output.*`` or ``registration.paths.*`` is used
verbatim and never rewritten, so a config can point anywhere on disk and
ignore the convention entirely.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import (
    UnifiedConfig,
)
from .vision.aruco import IMAGE_EXTENSIONS


# =============================================================================
# Layout convention
# =============================================================================
# These small helpers exist as a SINGLE source of truth: changing any of them
# changes path conventions globally.

DATE_FOLDER_FORMAT = "%d-%m-%Y"

#: Vision input folders may carry a device suffix (``<sample>_iphone``). When
#: more than one candidate matches, a suffix containing one of these wins.
PREFERRED_VISION_SUFFIXES = ("iphone",)


def today_folder() -> str:
    """Current run-date folder name, ``dd-mm-yyyy``."""
    return datetime.now().strftime(DATE_FOLDER_FORMAT)


def is_date_folder(name: str) -> bool:
    """True if ``name`` parses as a ``dd-mm-yyyy`` run-date folder."""
    try:
        datetime.strptime(name, DATE_FOLDER_FORMAT)
    except ValueError:
        return False
    return True


def run_root(results_root: Path, date: Optional[str] = None) -> Path:
    """``<results_root>/<dd-mm-yyyy>`` — the per-day root for all outputs."""
    return Path(results_root) / (date or today_folder())


# ----- inputs ----------------------------------------------------------------

def vision_input_dir(data_root: Path, sample: str) -> Path:
    """Resolve the vision input folder for ``sample``.

    Prefers an exact ``input_vision/<sample>``. Otherwise looks for
    ``input_vision/<sample>_*`` and returns the candidate whose suffix
    contains one of :data:`PREFERRED_VISION_SUFFIXES` (``iphone`` first),
    falling back to the first match alphabetically.

    Returns the exact (possibly non-existent) path when nothing matches, so
    error messages point at the canonical location.
    """
    base = Path(data_root) / "input_vision"
    exact = base / sample
    if exact.is_dir():
        return exact
    if not base.is_dir():
        return exact

    prefix = f"{sample}_"
    candidates = sorted(
        p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)
    )
    if not candidates:
        return exact

    for preferred in PREFERRED_VISION_SUFFIXES:
        for candidate in candidates:
            if preferred in candidate.name[len(prefix):].lower():
                return candidate
    return candidates[0]


def vision_rgb_dir(data_root: Path, sample: str) -> Path:
    """Folder holding the RGB images for ``sample``.

    Canonical location is ``<vision_input_dir>/rgb``. If that folder is
    absent but the sample folder itself contains images, the sample folder is
    used instead — both historical conventions therefore keep working.
    """
    sample_dir = vision_input_dir(data_root, sample)
    rgb_dir = sample_dir / "rgb"
    if rgb_dir.is_dir():
        return rgb_dir
    if sample_dir.is_dir() and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in sample_dir.iterdir()
    ):
        return sample_dir
    return rgb_dir


def vision_pose_dir(data_root: Path, sample: str) -> Path:
    return vision_input_dir(data_root, sample) / "poses"


def vision_camera_params_dir(data_root: Path, sample: str) -> Path:
    return vision_input_dir(data_root, sample) / "camera_parameters"


def registration_input_dir(data_root: Path, sample: str) -> Path:
    return Path(data_root) / "input_registration" / sample


def default_hsi_hdr_path(data_root: Path, sample: str) -> Path:
    return registration_input_dir(data_root, sample) / f"{sample}_raw.hdr"


def default_liveview_png_path(data_root: Path, sample: str) -> Path:
    return registration_input_dir(data_root, sample) / f"{sample}_raw.png"


# ----- outputs ---------------------------------------------------------------

def sample_output_dir(results_root: Path, sample: str, date: Optional[str] = None) -> Path:
    return run_root(results_root, date) / sample


def vision_output_dir(results_root: Path, sample: str, date: Optional[str] = None) -> Path:
    return sample_output_dir(results_root, sample, date) / "vision"


def registration_output_dir(results_root: Path, sample: str, date: Optional[str] = None) -> Path:
    return sample_output_dir(results_root, sample, date) / "registration"


def batch_summary_dir(results_root: Path, date: Optional[str] = None) -> Path:
    return run_root(results_root, date) / "_batch"


def vision_mesh_path(results_root: Path, sample: str, date: Optional[str] = None) -> Path:
    return vision_output_dir(results_root, sample, date) / "surface_mesh.ply"


def vision_aruco_path(results_root: Path, sample: str, date: Optional[str] = None) -> Path:
    return vision_output_dir(results_root, sample, date) / "aruco_markers_3d.json"


# =============================================================================
# Locating an existing vision run
# =============================================================================

def vision_dir_complete(vision_dir: Path) -> bool:
    """True iff both cross-stage handoff files exist in ``vision_dir``."""
    vision_dir = Path(vision_dir)
    return (
        (vision_dir / "surface_mesh.ply").is_file()
        and (vision_dir / "aruco_markers_3d.json").is_file()
    )


def find_vision_dir(
    results_root: Path,
    sample: str,
    date: Optional[str] = None,
) -> tuple[Path, bool]:
    """Locate the vision outputs registration should consume.

    Prefers the requested run date (today by default). If that folder has no
    usable vision run, falls back to the most recent dated folder that does,
    so a registration-only run still works the day after a vision run.

    Returns ``(vision_dir, is_fallback)``. When nothing usable exists the
    canonical path for ``date`` is returned with ``is_fallback=False``, and
    the downstream validator reports it as missing.
    """
    preferred = vision_output_dir(results_root, sample, date)
    if vision_dir_complete(preferred):
        return preferred, False

    root = Path(results_root)
    if root.is_dir():
        dated = sorted(
            (p for p in root.iterdir() if p.is_dir() and is_date_folder(p.name)),
            key=lambda p: datetime.strptime(p.name, DATE_FOLDER_FORMAT),
            reverse=True,
        )
        for day in dated:
            candidate = day / sample / "vision"
            if candidate != preferred and vision_dir_complete(candidate):
                return candidate, True

    return preferred, False


def vision_outputs_present(
    results_root: Path,
    sample: str,
    date: Optional[str] = None,
) -> bool:
    """True iff a usable vision run exists for ``date`` (no fallback)."""
    return vision_dir_complete(vision_output_dir(results_root, sample, date))


def check_marker_edge(aruco_json: Path, marker_edge_length_mm: float) -> None:
    """Refuse a vision run reconstructed with a different marker edge length.

    The mesh and the 3D markers are metric-scaled by the edge length vision
    used (recorded as ``edge_length_m`` in ``aruco_markers_3d.json``), so
    registering them against a different ``vision.aruco.marker_edge_length_mm``
    would silently mix two scales.
    """
    aruco_json = Path(aruco_json)
    if not aruco_json.is_file():
        return  # reported as missing by the registration input validation
    recorded_m = json.loads(aruco_json.read_text(encoding="utf-8")).get("edge_length_m")
    if recorded_m is None:
        raise ValueError(
            f"{aruco_json} does not record the marker edge length (edge_length_m); "
            "re-run vision for this sample."
        )
    recorded_mm = float(recorded_m) * 1000.0
    if not math.isclose(recorded_mm, float(marker_edge_length_mm), rel_tol=1e-6):
        raise ValueError(
            f"Marker edge mismatch: the vision run {aruco_json.parent} was built with "
            f"{recorded_mm:g} mm markers, but vision.aruco.marker_edge_length_mm is "
            f"{float(marker_edge_length_mm):g} mm. Re-run vision for this sample, or "
            "point registration at a vision run made with the same marker size."
        )


# =============================================================================
# Path resolution
# =============================================================================

def _is_unset(value: Any) -> bool:
    """Treat None and empty strings as unset; everything else is user-provided."""
    if value is None:
        return True
    if isinstance(value, (str, Path)) and str(value).strip() == "":
        return True
    return False


def resolve_unified_paths(
    cfg: UnifiedConfig,
    *,
    vision_dir: Optional[Path] = None,
    resolve_vision_handoff: bool = True,
) -> UnifiedConfig:
    """Fill in every unset path and inherited marker parameter.

    Covers ``vision.input.*``, ``vision.output.*`` and ``registration.paths.*``
    using the layout convention, plus the ArUco marker size (always taken from
    ``vision.aruco``) and dictionary (inherited from ``vision.aruco`` unless
    overridden).

    ``registration.paths.mesh`` / ``aruco_json``, when unset, come from:

    * ``vision_dir`` if given (``spectra full``: the run vision just made);
    * nothing, if ``resolve_vision_handoff`` is False (left unset, so a later
      call can still fill them — used before vision has run);
    * otherwise this date's vision run, falling back to the most recent
      complete one (``spectra registration``).

    Returns a new ``UnifiedConfig``; the input is not mutated. Explicitly-set
    paths — including absolute paths — are always left untouched.
    """
    sample = cfg.sample_name
    date = cfg.results_date
    overrides: dict[str, Any] = {}

    # ------- vision.input -------------------------------------------------
    v_in = cfg.vision.input
    if _is_unset(v_in.rgb_dir):
        overrides["vision.input.rgb_dir"] = str(vision_rgb_dir(cfg.data_root, sample))
    # pose_dir / camera_params_dir auto-fill only if the folders exist.
    if _is_unset(v_in.pose_dir):
        candidate = vision_pose_dir(cfg.data_root, sample)
        if candidate.is_dir():
            overrides["vision.input.pose_dir"] = str(candidate)
    if _is_unset(v_in.camera_params_dir):
        candidate = vision_camera_params_dir(cfg.data_root, sample)
        if candidate.is_dir():
            overrides["vision.input.camera_params_dir"] = str(candidate)

    # ------- vision.output ------------------------------------------------
    if _is_unset(cfg.vision.output.root):
        overrides["vision.output.root"] = str(
            sample_output_dir(cfg.results_root, sample, date)
        )
    if _is_unset(cfg.vision.output.run_name):
        overrides["vision.output.run_name"] = "vision"

    # ------- registration.paths ------------------------------------------
    r_paths = cfg.registration.paths
    if _is_unset(r_paths.hsi_hdr):
        overrides["registration.paths.hsi_hdr"] = str(
            default_hsi_hdr_path(cfg.data_root, sample)
        )

    # mesh + aruco come from a vision run (see the docstring for which one).
    # Never resolved before vision has run in `spectra full`: an unset path is
    # treated as "not chosen yet", and a stale fallback written here would be
    # mistaken for a user-set path later on.
    if resolve_vision_handoff and (_is_unset(r_paths.mesh) or _is_unset(r_paths.aruco_json)):
        if vision_dir is None:
            vision_dir, _ = find_vision_dir(cfg.results_root, sample, date)
        vision_dir = Path(vision_dir)
        if _is_unset(r_paths.mesh):
            overrides["registration.paths.mesh"] = str(vision_dir / "surface_mesh.ply")
        if _is_unset(r_paths.aruco_json):
            overrides["registration.paths.aruco_json"] = str(
                vision_dir / "aruco_markers_3d.json"
            )

    if _is_unset(r_paths.output_dir):
        overrides["registration.paths.output_dir"] = str(
            registration_output_dir(cfg.results_root, sample, date)
        )

    # liveview_png: only auto-derive when the run actually needs it, so
    # validation stays clean for non-ROI runs.
    needs_liveview = (
        cfg.registration.mode == "roi"
        or (cfg.registration.mode == "sweep" and cfg.registration.sweep.roi_mode)
        or (cfg.registration.mode == "batch" and cfg.registration.batch.roi_mode)
    )
    if needs_liveview and _is_unset(r_paths.liveview_png):
        overrides["registration.paths.liveview_png"] = str(
            default_liveview_png_path(cfg.data_root, sample)
        )

    # ------- marker geometry: one source of truth -------------------------
    # The HSI cube and the 3D reconstruction see the SAME physical markers.
    # The edge length is ALWAYS vision.aruco.marker_edge_length_mm (setting it
    # under `registration:` is rejected at load time); the dictionary is
    # inherited unless overridden.
    overrides["registration.detection.marker_edge_length_mm"] = float(
        cfg.vision.aruco.marker_edge_length_mm
    )
    if _is_unset(cfg.registration.detection.aruco_dict):
        overrides["registration.detection.aruco_dict"] = cfg.vision.aruco.dictionary

    return cfg.with_overrides(overrides) if overrides else cfg


# =============================================================================
# Stage runners (thin wrappers — used by CLI and by run_full)
# =============================================================================

def run_vision_stage(
    cfg: UnifiedConfig,
    *,
    sample_name: Optional[str] = None,
) -> dict[str, Any]:
    """Run only the vision stage. Returns ``{"run_dir": Path, "skipped": False}``."""
    from .vision.pipeline import run_reconstruction

    # Vision does not consume the registration handoff paths: leave them unset.
    cfg = resolve_unified_paths(cfg, resolve_vision_handoff=False)
    sample = sample_name or cfg.sample_name
    print(f"\n[orchestrator] Vision stage for sample={sample!r}")
    t0 = time.time()
    result = run_reconstruction(cfg.vision)
    dt = time.time() - t0
    print(f"[orchestrator] Vision finished in {dt:.1f}s — run_dir={result.run_dir}")
    return {"run_dir": Path(result.run_dir), "skipped": False, "elapsed_s": dt}


def run_registration_stage(
    cfg: UnifiedConfig,
    *,
    sample_name: Optional[str] = None,
    force_cpu: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run only the registration stage (mode is read from cfg.registration.mode)."""
    from .registration.runner import run_registration

    # Paths already set (by `run_full`, or by the user) are kept; unset ones
    # fall back to the most recent complete vision run.
    cfg = resolve_unified_paths(cfg)
    sample = sample_name or cfg.sample_name
    print(f"\n[orchestrator] Registration stage for sample={sample!r} "
          f"(mode={cfg.registration.mode!r})")
    print(f"[orchestrator] Marker edge (vision.aruco.marker_edge_length_mm): "
          f"{cfg.vision.aruco.marker_edge_length_mm:g} mm")
    if cfg.registration.mode != "batch":
        # batch picks (and checks) each sample's vision run in discover_samples
        mesh_dir = Path(cfg.registration.paths.mesh).parent
        expected = vision_output_dir(cfg.results_root, sample, cfg.results_date)
        if mesh_dir == expected:
            print(f"[orchestrator] Vision run used by registration: {mesh_dir}")
        else:
            print(f"[orchestrator] WARNING: registration uses the vision run in {mesh_dir}, "
                  f"not this run date's ({expected}).")
        check_marker_edge(cfg.registration.paths.aruco_json, cfg.vision.aruco.marker_edge_length_mm)
    t0 = time.time()
    result = run_registration(
        cfg.registration,
        sample_name=sample,
        data_root=cfg.data_root,
        results_root=cfg.results_root,
        results_date=cfg.results_date or today_folder(),
        force_cpu=force_cpu,
        dry_run=dry_run,
    )
    dt = time.time() - t0
    print(f"[orchestrator] Registration finished in {dt:.1f}s")
    result["elapsed_s"] = dt
    return result


# =============================================================================
# Full pipeline
# =============================================================================

def run_full(
    cfg: UnifiedConfig,
    *,
    sample_name: Optional[str] = None,
    force_cpu: bool = False,
    force_vision: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run vision (if needed) and then registration on a single sample.

    Parameters
    ----------
    cfg :
        Unified config. Path resolution is performed internally — anything
        ``None`` is filled in via the layout convention.
    sample_name :
        Overrides ``cfg.sample_name`` for this call.
    force_cpu :
        Forwarded to registration.
    force_vision :
        If True, runs vision even when this day's outputs already exist.
    dry_run :
        If True, validates configs and prints what would happen, no execution.

    Returns
    -------
    dict
        ``{"vision": ..., "registration": ..., "sample": ..., "skipped_vision": bool}``
    """
    # The registration handoff (mesh / aruco_json) is NOT resolved here: vision
    # has not run yet, so this date's folder may be missing and the fallback
    # would pick an older run. It is filled in after vision, below.
    cfg = resolve_unified_paths(cfg, resolve_vision_handoff=False)
    sample = sample_name or cfg.sample_name
    date = cfg.results_date or today_folder()

    print("\n" + "=" * 68)
    print(f"  SPECTRA FULL — sample={sample!r}")
    print(f"  data_root    : {cfg.data_root}")
    print(f"  results_root : {cfg.results_root}")
    print(f"  run date     : {date}")
    print(f"  stages       : {cfg.stages}")
    print(f"  reg mode     : {cfg.registration.mode}")
    print("=" * 68)

    do_vision = "vision" in cfg.stages
    do_registration = "registration" in cfg.stages

    out: dict[str, Any] = {"sample": sample, "date": date, "skipped_vision": False}

    if do_vision:
        if not force_vision and vision_outputs_present(cfg.results_root, sample, cfg.results_date):
            print(f"\n[orchestrator] Vision outputs already exist at "
                  f"{vision_output_dir(cfg.results_root, sample, cfg.results_date)} "
                  f"— skipping vision stage.")
            print("  (use --force-vision to re-run and overwrite them)")
            out["skipped_vision"] = True
            out["vision"] = {
                "skipped": True,
                "run_dir": vision_output_dir(cfg.results_root, sample, cfg.results_date),
            }
        else:
            if dry_run:
                print("[orchestrator][dry-run] Would run vision stage.")
                out["vision"] = {"skipped": False, "dry_run": True}
            else:
                out["vision"] = run_vision_stage(cfg, sample_name=sample)
    else:
        print("[orchestrator] Vision stage disabled (not in cfg.stages).")

    if do_registration:
        if do_vision:
            # Registration consumes exactly the vision run of THIS invocation
            # (just produced, or the one for this date that caused the skip).
            # No fallback to an older run: a missing output is an error.
            # Paths set explicitly by the user are still left untouched.
            vision_dir = Path(out["vision"].get(
                "run_dir", vision_output_dir(cfg.results_root, sample, cfg.results_date)))
            needs_handoff = (_is_unset(cfg.registration.paths.mesh)
                             or _is_unset(cfg.registration.paths.aruco_json))
            if needs_handoff and not dry_run and not vision_dir_complete(vision_dir):
                raise FileNotFoundError(
                    f"Vision run {vision_dir} has no surface_mesh.ply + aruco_markers_3d.json; "
                    "refusing to register against an older vision run."
                )
            cfg = resolve_unified_paths(cfg, vision_dir=vision_dir)
        else:
            # Registration only: this date's vision run, else the most recent one.
            cfg = resolve_unified_paths(cfg)
        if dry_run:
            print("[orchestrator][dry-run] Would run registration stage.")
            out["registration"] = {"dry_run": True}
        else:
            out["registration"] = run_registration_stage(
                cfg, sample_name=sample, force_cpu=force_cpu, dry_run=False,
            )
    else:
        print("[orchestrator] Registration stage disabled (not in cfg.stages).")

    print("\n" + "=" * 68)
    print("SPECTRA FULL COMPLETED")
    print("=" * 68)
    print(f"  Sample        : {sample}")
    print(f"  Run date      : {date}")
    print(f"  Vision skipped: {out['skipped_vision']}")
    if "vision" in out and "run_dir" in out["vision"]:
        print(f"  Vision output : {out['vision']['run_dir']}")
    if "registration" in out:
        print(f"  Reg. output   : {cfg.registration.paths.output_dir}")
    print("=" * 68)

    return out


__all__ = [
    "DATE_FOLDER_FORMAT",
    "PREFERRED_VISION_SUFFIXES",
    "batch_summary_dir",
    "check_marker_edge",
    "default_hsi_hdr_path",
    "default_liveview_png_path",
    "find_vision_dir",
    "is_date_folder",
    "registration_input_dir",
    "registration_output_dir",
    "resolve_unified_paths",
    "run_full",
    "run_registration_stage",
    "run_root",
    "run_vision_stage",
    "sample_output_dir",
    "today_folder",
    "vision_aruco_path",
    "vision_camera_params_dir",
    "vision_dir_complete",
    "vision_input_dir",
    "vision_mesh_path",
    "vision_output_dir",
    "vision_pose_dir",
    "vision_outputs_present",
    "vision_rgb_dir",
]
