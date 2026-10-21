"""Typer CLI for the unified Spectra pipeline.

Top-level commands::

    spectra vision        # MASt3R-SfM + ArUco reconstruction (legacy `recon`)
    spectra registration  # HSI→mesh registration (single/roi/batch/sweep)
    spectra full          # acquisition → vision → registration on one sample
                          # (the stages listed in `stages:`)
    spectra detect        # standalone 2D ArUco detection (unchanged)
    spectra viewer        # local 3D viewer (unchanged)
    spectra calibrate-intrinsics  # checkerboard calibration (unchanged)
    spectra acquire       # robotic acquisition only (Fairino FR5 + RealSense)
    spectra calibrate     # hand-eye + intrinsics of the robot camera (+ validation)

Where vision / registration run is decided by `mufasa.enabled` in the YAML:
true → they are submitted to Mufasa (SLURM) by `spectra/acquisition/remote.py`
and RESULTS/ is downloaded back; false → they run on this PC. The acquisition
always runs on this PC.

`vision` and `recon` are aliases. Vision-only YAML configs from the legacy
repo (top-level fields without a `vision:` wrapper) are still accepted by
the `vision` and `recon` commands; the unified loader is only used when the
YAML has a `vision:` sub-section (which is the new default).

Outputs are grouped by run date: `RESULTS/<dd-mm-yyyy>/<sample>/<stage>/`.
`--date dd-mm-yyyy` targets a different day (useful to run registration
against an earlier vision run); re-running on the same day overwrites that
day's stage folder rather than creating a sibling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from rich import print
from rich.markup import escape

from .config import (
    UnifiedConfig,
    VisionConfig,
    load_unified_config,
)
from .vision.aruco import ARUCO_DICTIONARIES, detect_folder
from .vision.calibration import calibrate_intrinsics
from .vision.config import DEFAULT_CONFIG_PATH, load_config as load_vision_only_config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Unified Spectra pipeline: 3D reconstruction (vision) + HSI registration.",
)


# =============================================================================
# Shared helpers
# =============================================================================

def _parse_override(value: str) -> tuple[str, object]:
    """Parse 'a.b.c=value' overrides. Value is JSON-parsed when possible."""
    if "=" not in value:
        raise typer.BadParameter(f"Override {value!r} must be 'key.path=VALUE'")
    dotted_key, raw_value = value.split("=", 1)
    dotted_key = dotted_key.strip()
    raw_value = raw_value.strip()
    try:
        parsed: object = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = raw_value
    return dotted_key, parsed


def _looks_unified(yaml_data: dict) -> bool:
    """Heuristic: a 'vision:' top-level field means the YAML is unified."""
    return isinstance(yaml_data, dict) and "vision" in yaml_data


def _read_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise typer.BadParameter(f"YAML at {path} must be a mapping.")
    return data


def _load_for_vision_command(config: Path) -> UnifiedConfig:
    """Accept both legacy vision-only YAML and unified YAML for `vision` / `recon`."""
    data = _read_yaml(config)
    if _looks_unified(data):
        return load_unified_config(config)
    # Legacy: wrap a vision-only config in a UnifiedConfig with default extras.
    vcfg = load_vision_only_config(config)
    return UnifiedConfig.model_validate({
        "sample_name": vcfg.output.run_name or "SAMPLE1",
        "vision": vcfg.model_dump(mode="python"),
    })


def _cli_overrides(
    sample: Optional[str],
    date: Optional[str],
    overrides: List[str],
    extra: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    """--sample / --date / command options / --set as one dotted-key dict.

    The same dict is applied to the local config and, when the run goes to
    Mufasa, to the run config sent with the job.
    """
    merged: dict[str, object] = {}
    if sample:
        merged["sample_name"] = sample
    if date:
        merged["results_date"] = date
    merged.update(extra or {})
    for ov in overrides:
        key, value = _parse_override(ov)
        if key.endswith("marker_edge_length_mm"):
            raise typer.BadParameter(
                f"--set {key} is not allowed: the marker edge length is set only in "
                f"{DEFAULT_CONFIG_PATH} (vision.aruco.marker_edge_length_mm)."
            )
        if key == "acquisition" or key.startswith("acquisition."):
            raise typer.BadParameter(
                f"--set {key} is not allowed: the acquisition reads its section "
                "from the YAML file, edit it there."
            )
        merged[key] = value
    return merged


def _in_slurm_job() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID"))


def _mufasa_on(cfg: UnifiedConfig) -> bool:
    """True when vision / registration must be submitted to Mufasa. Inside a
    SLURM job we already are on the cluster: always run here."""
    if not (cfg.mufasa is not None and cfg.mufasa.enabled):
        return False
    if _in_slurm_job():
        print("[yellow]mufasa.enabled is true but this is a SLURM job: running here.[/yellow]")
        return False
    return True


def _load_script(name: str):
    """Load a standalone script of spectra/acquisition/ by path (flat sibling
    imports, robot SDK in that folder)."""
    import importlib.util

    script = Path(__file__).resolve().parent / "acquisition" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"spectra_{name}_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module, script


def _run_on_mufasa(
    config: Path,
    overrides: dict[str, object],
    stages: List[str],
    *,
    dry_run: bool = False,
    force_cpu: bool = False,
) -> None:
    """Submit `stages` to Mufasa (mufasa.enabled is true) and wait for them."""
    remote, _ = _load_script("remote")
    print(f"[bold]{' + '.join(stages)} → Mufasa[/bold] (mufasa.enabled: true)")
    try:
        jobid, state = remote.submit(
            str(config), stages=stages, overrides=overrides, dry_run=dry_run,
            extra_args=["--cpu"] if force_cpu else [],
        )
    except SystemExit as exc:          # remote.py reports errors as SystemExit(msg)
        print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1)
    if state not in ("COMPLETED", "SUBMITTED", "DRY-RUN"):
        print(f"[red]Mufasa job {jobid} ended in state {state}: see the job log above.[/red]")
        raise typer.Exit(code=1)


#: Packages each stage imports when it runs on THIS machine (pyproject extras).
_STAGE_PACKAGES = {
    "vision": ("torch", "PIL"),
    "registration": ("trimesh", "openpyxl"),
}


def _missing(packages) -> list[str]:
    import importlib.util

    return [p for p in packages if importlib.util.find_spec(p) is None]


def _require_local_processing(stages: List[str]) -> None:
    """mufasa.enabled is false: vision / registration run on this PC. Check,
    before anything runs, that it has the `processing` extra (+ MASt3R) and,
    for vision, a CUDA GPU."""
    missing = sorted({p for s in stages for p in _missing(_STAGE_PACKAGES.get(s, ()))})
    if "vision" in stages:
        mast3r_dir = Path(__file__).resolve().parent.parent / "mast3r"
        if not mast3r_dir.is_dir() and _missing(["mast3r"]):
            missing.append("mast3r (clone it in the repo, README §1.3)")
    if missing:
        print(f"[red]mufasa.enabled is false, so {' + '.join(stages)} must run on this PC, "
              f"but it lacks {missing}.[/red]\n"
              "Either install them:  uv sync --extra acquisition --extra processing  "
              "(+ MASt3R / ASMK, README §1.3)\n"
              "or set `mufasa.enabled: true` to run them on Mufasa.")
        raise typer.Exit(code=1)
    if "vision" in stages:
        import torch

        if not torch.cuda.is_available():
            print("[red]Vision needs a CUDA GPU and this PC has none.[/red] "
                  "Set `mufasa.enabled: true` in the config to run it on Mufasa.")
            raise typer.Exit(code=1)


def _require_acquisition_packages() -> None:
    missing = _missing(("pyrealsense2", "fcl", "trimesh"))
    if missing:
        print(f"[red]The acquisition needs {missing}.[/red] "
              "Install them on the robot PC:  uv sync --extra acquisition")
        raise typer.Exit(code=1)


# =============================================================================
# vision (= recon)
# =============================================================================

def _execute_vision(
    config: Path,
    sample: Optional[str],
    date: Optional[str],
    overrides: List[str],
    dry_run: bool,
) -> None:
    cfg = _load_for_vision_command(config)
    ovr = _cli_overrides(sample, date, overrides)
    if ovr:
        cfg = cfg.with_overrides(ovr)
    if _mufasa_on(cfg):
        _run_on_mufasa(config, ovr, ["vision"], dry_run=dry_run)
        return

    from .orchestrator import run_vision_stage, resolve_unified_paths

    cfg = resolve_unified_paths(cfg, resolve_vision_handoff=False)
    if dry_run:
        print("[bold]Dry-run — vision config resolved:[/bold]")
        print(cfg.vision.model_dump(mode="python"))
        return
    _require_local_processing(["vision"])
    result = run_vision_stage(cfg)
    print(f"[green]Vision finished:[/green] {result['run_dir']}")


@app.command("vision")
def vision_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True,
                                help="YAML config (unified or vision-only)."),
    sample: Optional[str] = typer.Option(None, "--sample",
                                          help="Override sample_name for this run."),
    date: Optional[str] = typer.Option(None, "--date",
                                        help="Run-date folder dd-mm-yyyy under results_root (default: today)."),
    set_override: List[str] = typer.Option([], "--set", "-s",
                                            help="Dotted override: --set vision.aruco.min_views_per_marker=3"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                  help="Resolve paths and print the config without running."),
) -> None:
    """Run only the MASt3R-SfM + ArUco vision stage (on Mufasa if mufasa.enabled)."""
    _execute_vision(config, sample, date, set_override, dry_run)


@app.command("recon", hidden=True)
def recon_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    sample: Optional[str] = typer.Option(None, "--sample"),
    date: Optional[str] = typer.Option(None, "--date"),
    set_override: List[str] = typer.Option([], "--set", "-s"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Deprecated alias for `spectra vision`."""
    _execute_vision(config, sample, date, set_override, dry_run)


# =============================================================================
# registration
# =============================================================================

@app.command("registration")
def registration_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True,
                                help="Unified YAML config (must have a `registration:` section)."),
    sample: Optional[str] = typer.Option(None, "--sample",
                                          help="Override sample_name (also used as output filename prefix)."),
    date: Optional[str] = typer.Option(None, "--date",
                                        help="Run-date folder dd-mm-yyyy under results_root (default: today)."),
    mode: Optional[str] = typer.Option(None, "--mode",
                                        help="Override registration.mode (single|roi|batch|sweep)."),
    set_override: List[str] = typer.Option([], "--set", "-s",
                                            help="Dotted override: --set registration.render.resolution_mm_per_px=1.0"),
    force_cpu: bool = typer.Option(False, "--cpu",
                                    help="Force CPU raycasting even if a GPU is available."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                  help="Validate paths and print the resolved config, no execution."),
) -> None:
    """Run only the registration stage (on Mufasa if mufasa.enabled; mode from YAML or `--mode`)."""
    if mode and mode not in ("single", "roi", "batch", "sweep"):
        raise typer.BadParameter(f"Invalid --mode {mode!r}; use single|roi|batch|sweep.")
    ovr = _cli_overrides(sample, date, set_override,
                         {"registration.mode": mode} if mode else None)
    cfg = load_unified_config(config).with_overrides(ovr)
    if _mufasa_on(cfg):
        _run_on_mufasa(config, ovr, ["registration"], dry_run=dry_run, force_cpu=force_cpu)
        return

    _require_local_processing(["registration"])
    from .orchestrator import run_registration_stage

    run_registration_stage(cfg, force_cpu=force_cpu, dry_run=dry_run)


# =============================================================================
# full
# =============================================================================

@app.command("full")
def full_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True,
                                help="Unified YAML config with both `vision:` and `registration:`."),
    sample: Optional[str] = typer.Option(None, "--sample",
                                          help="Override sample_name. Required if YAML uses placeholder."),
    date: Optional[str] = typer.Option(None, "--date",
                                        help="Run-date folder dd-mm-yyyy under results_root (default: today)."),
    set_override: List[str] = typer.Option([], "--set", "-s",
                                            help="Dotted override (e.g. --set registration.mode=roi)"),
    force_cpu: bool = typer.Option(False, "--cpu", help="Force CPU raycasting in registration."),
    force_vision: bool = typer.Option(False, "--force-vision",
                                       help="Re-run vision even if its outputs already exist "
                                            "(this PC only; on Mufasa vision always runs)."),
    overwrite: bool = typer.Option(False, "--overwrite",
                                    help="Acquisition: replace an existing acquisition for the same sample."),
    no_viz: bool = typer.Option(False, "--no-viz", help="Acquisition: disable the Rerun viewer."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                  help="Print the plan (and the Mufasa run config), "
                                       "no robot, no upload, no execution."),
) -> None:
    """Run the `stages:` of the YAML on one sample: acquisition (this PC) →
    vision → registration (this PC, or Mufasa when mufasa.enabled is true)."""
    ovr = _cli_overrides(sample, date, set_override)
    cfg = load_unified_config(config).with_overrides(ovr)
    acquire = "acquisition" in cfg.stages
    if acquire and _in_slurm_job():
        print("[yellow]stages contains 'acquisition' but this is a SLURM job (no robot "
              "here): skipping the acquisition.[/yellow]")
        acquire = False
    compute = [s for s in ("vision", "registration") if s in cfg.stages]
    on_mufasa = _mufasa_on(cfg)
    if acquire and not cfg.acquisition:
        raise typer.BadParameter("stages contains 'acquisition' but the config has no "
                                 "`acquisition:` section.")

    where = "Mufasa (SLURM)" if on_mufasa else "this PC"
    print(f"[bold]spectra full[/bold] — sample {cfg.sample_name!r}")
    if acquire:
        print("  acquisition  : this PC (robot + RealSense)")
    for s in compute:
        print(f"  {s:<13}: {where}")

    if dry_run:
        if acquire:
            print("\\[dry-run] the acquisition would run here (robot not moved).")
        if compute and on_mufasa:
            _run_on_mufasa(config, ovr, compute, dry_run=True, force_cpu=force_cpu)
        elif compute:
            from .orchestrator import run_full
            run_full(cfg, force_cpu=force_cpu, force_vision=force_vision, dry_run=True)
        return

    # Everything that can fail is checked BEFORE the robot moves.
    if compute and on_mufasa:
        remote, _ = _load_script("remote")
        try:
            remote.preflight(str(config), ovr)
        except SystemExit as exc:
            print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(code=1)
    elif compute:
        _require_local_processing(compute)

    if acquire:
        n_saved = _run_acquire(config, [], sample=cfg.sample_name,
                               overwrite=overwrite, no_viz=no_viz)
        if not n_saved:
            print("[red]Nothing acquired: vision / registration not started.[/red]")
            raise typer.Exit(code=1)

    if not compute:
        return
    if on_mufasa:
        _run_on_mufasa(config, ovr, compute, force_cpu=force_cpu)
    else:
        from .orchestrator import run_full
        # A fresh acquisition makes today's vision outputs (if any) stale.
        run_full(cfg, force_cpu=force_cpu, force_vision=force_vision or acquire)


# =============================================================================
# acquire — thin wrapper around spectra/acquisition/acquire.py
# =============================================================================

_ACQUIRE_MODES = {
    "run": [],                 # full acquisition (robot moves)
    "verify": ["--verify"],    # check the transformation chain, robot does not move
    "dry-run": ["--dry-run"],  # bootstrap + pose generation, robot does not move
    "reset": ["--reset"],      # reset robot errors and exit
}


@app.command("acquire")
def acquire_cmd(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c", exists=True,
                                readable=True, help="Unified YAML (needs an `acquisition:` section)."),
    mode: str = typer.Option("run", "--mode", "-m",
                             help="run | verify | dry-run | reset"),
    sample: Optional[str] = typer.Option(None, "--sample",
                                          help="Override sample_name: output in DATA/input_vision/<sample>/."),
    ip: Optional[str] = typer.Option(None, "--ip", help="Robot controller IP (default: acquisition.robot.ip)."),
    no_viz: bool = typer.Option(False, "--no-viz", help="Disable the Rerun viewer."),
    overwrite: bool = typer.Option(False, "--overwrite",
                                    help="Replace an existing acquisition for the same sample."),
) -> None:
    """Robotic acquisition only (never submits): same as `python spectra/acquisition/acquire.py`.
    Use `spectra full` to chain vision + registration (here or on Mufasa)."""
    if mode not in _ACQUIRE_MODES:
        raise typer.BadParameter(f"--mode must be one of {list(_ACQUIRE_MODES)}, got {mode!r}")
    _run_acquire(config, _ACQUIRE_MODES[mode], sample=sample, ip=ip,
                 no_viz=no_viz, overwrite=overwrite)


def _run_acquire(
    config: Path,
    mode_args: List[str],
    *,
    sample: Optional[str] = None,
    ip: Optional[str] = None,
    no_viz: bool = False,
    overwrite: bool = False,
):
    """Run acquire.py's main() with the equivalent argv; returns its result
    (number of saved image/pose pairs for a real acquisition)."""
    import sys

    argv = ["--config", str(config), *mode_args]
    if sample:
        argv += ["--sample", sample]
    if ip:
        argv += ["--ip", ip]
    if no_viz:
        argv.append("--no-viz")
    if overwrite:
        argv.append("--overwrite")
    _require_acquisition_packages()
    module, script = _load_script("acquire")
    old_argv = sys.argv
    sys.argv = [str(script), *argv]
    try:
        return module.main()
    finally:
        sys.argv = old_argv


# =============================================================================
# detect / viewer / calibrate — kept as-is from Vision
# =============================================================================

@app.command("detect")
def detect_cmd(
    input_folder: Path = typer.Argument(..., exists=True, file_okay=False,
                                          help="Folder of RGB images."),
    output_folder: Path = typer.Argument(...,
                                          help="Receives json/ and annotated/."),
    dictionary: str = typer.Option("4x4_50", "--dict",
                                    help=f"ArUco dictionary. One of: {sorted(ARUCO_DICTIONARIES.keys())}"),
    draw_scale: float = typer.Option(2.0, "--draw-scale"),
) -> None:
    """Detect ArUco markers in images (2D only)."""
    results = detect_folder(
        rgb_dir=input_folder,
        out_dir=output_folder,
        dictionary=dictionary,
        draw_scale=max(0.1, draw_scale),
    )
    if not results:
        print(f"[yellow]No images found in:[/yellow] {input_folder}")
        return
    for stem, detections in results.items():
        print(f"{stem}: [green]{len(detections)}[/green] marker(s)")
    print(f"JSON: [green]{output_folder / 'json'}[/green]")
    print(f"Annotated: [green]{output_folder / 'annotated'}[/green]")


@app.command("viewer")
def viewer_cmd(
    results_dir: Path = typer.Option(Path("RESULTS"), "--results-dir", "-r"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    share: bool = typer.Option(False, "--share/--no-share"),
    no_browser: bool = typer.Option(False, "--no-browser"),
) -> None:
    """Local 3D viewer for runs under `results_dir`."""
    try:
        from .vision.viewer import run_viewer
    except ImportError as exc:
        print(f"[red]Viewer import failed:[/red] {exc}")
        print("[yellow]Install gradio and trimesh.[/yellow]")
        raise typer.Exit(code=1)
    run_viewer(
        results_dir=results_dir, host=host, port=port,
        share=share, inbrowser=not no_browser,
    )


# =============================================================================
# calibrate — hand-eye + intrinsics of the robot camera (spectra/calibration)
# =============================================================================

@app.command("calibrate")
def calibrate_cmd(
    step: Optional[str] = typer.Argument(
        None, help="With --modality calibration: acquire | calibrate | full; "
                   "with --modality validation: acquire | validate | full (default full)."),
    modality: Optional[str] = typer.Option(
        None, "--modality", "-m",
        help="calibration | validation | intrinsics. Omitted: the whole procedure."),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c", exists=True,
                                readable=True, help="Unified YAML (needs a `calibration:` section)."),
) -> None:
    """Camera calibration on the robot PC. Without --modality: calibration
    acquisition → intrinsics → hand-eye → validation acquisition → validation.
    Data in DATA/camera_calibration/."""
    from . import calibration
    from .calibration import common

    try:
        steps = calibration.plan(modality, step)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    cfg = load_unified_config(config)
    if cfg.calibration is None:
        raise typer.BadParameter(f"{config} has no `calibration:` section.")
    if any(s == "acquire" for _, s in steps):
        sdk = Path(__file__).resolve().parent / "acquisition" / "fairino" / "Robot.py"
        need = _missing(("pyrealsense2",)) + ([] if sdk.is_file() else [str(sdk)])
        if need:
            print(f"[red]The calibration acquisition needs {need}.[/red] Install "
                  "`uv sync --extra acquisition` and copy the Fairino SDK (README §1.7).")
            raise typer.Exit(code=1)
    print("[bold]spectra calibrate[/bold] — " + " → ".join(
        m + (f" {s}" if s else "") for m, s in steps))
    try:
        calibration.run(common.settings(cfg), steps)
    except SystemExit as exc:          # calibration modules report errors as SystemExit(msg)
        if exc.code not in (None, 0):
            print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(code=1)


@app.command("calibrate-intrinsics")
def calibrate_intrinsics_cmd(
    image_dir: Path = typer.Option(Path("checkerboard"), "--image-dir"),
    output_dir: Path = typer.Option(Path("intrinsics"), "--output-dir"),
    checkerboard_cols: int = typer.Option(10, "--checkerboard-cols"),
    checkerboard_rows: int = typer.Option(7, "--checkerboard-rows"),
    square_size_m: float = typer.Option(0.024, "--square-size-m"),
) -> None:
    """Camera-intrinsic calibration from checkerboard images."""
    try:
        mtx, dist = calibrate_intrinsics(
            image_dir=image_dir,
            output_dir=output_dir,
            checkerboard_size=(int(checkerboard_cols), int(checkerboard_rows)),
            square_size_m=float(square_size_m),
        )
    except ValueError as exc:
        print(f"[red]Calibration failed:[/red] {exc}")
        raise typer.Exit(code=1)
    print(f"[green]Saved:[/green] {output_dir / 'intrinsics.npy'}")
    print(f"[green]Saved:[/green] {output_dir / 'distortions.npy'}")
    print("Camera Matrix:"); print(mtx)
    print("Distortion Coefficients:"); print(dist)


# =============================================================================
# Entry point
# =============================================================================

def main(argv: Optional[list[str]] = None) -> None:
    if argv is None:
        app(prog_name="spectra")
    else:
        app(args=argv, prog_name="spectra")


if __name__ == "__main__":
    main()
