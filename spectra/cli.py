"""Typer CLI for the unified Spectra pipeline.

Top-level commands::

    spectra vision        # MASt3R-SfM + ArUco reconstruction (legacy `recon`)
    spectra registration  # HSI→mesh registration (single/roi/batch/sweep)
    spectra full          # vision → registration end-to-end on a single sample
    spectra detect        # standalone 2D ArUco detection (unchanged)
    spectra viewer        # local 3D viewer (unchanged)
    spectra calibrate-intrinsics  # checkerboard calibration (unchanged)

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
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from rich import print

from .config import (
    UnifiedConfig,
    VisionConfig,
    load_unified_config,
)
from .vision.aruco import ARUCO_DICTIONARIES, detect_folder
from .vision.calibration import calibrate_intrinsics
from .vision.config import load_config as load_vision_only_config

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


def _apply_cli_overrides(cfg: UnifiedConfig, overrides: List[str]) -> UnifiedConfig:
    if not overrides:
        return cfg
    merged: dict[str, object] = {}
    for ov in overrides:
        key, value = _parse_override(ov)
        merged[key] = value
    return cfg.with_overrides(merged)


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
    if sample:
        cfg = cfg.with_overrides({"sample_name": sample})
    if date:
        cfg = cfg.with_overrides({"results_date": date})
    cfg = _apply_cli_overrides(cfg, overrides)

    from .orchestrator import run_vision_stage, resolve_unified_paths

    cfg = resolve_unified_paths(cfg)
    if dry_run:
        print("[bold]Dry-run — vision config resolved:[/bold]")
        print(cfg.vision.model_dump(mode="python"))
        return
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
                                            help="Dotted override: --set vision.aruco.marker_edge_length_mm=30"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                  help="Resolve paths and print the config without running."),
) -> None:
    """Run only the MASt3R-SfM + ArUco vision stage."""
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
    """Run only the registration stage (mode read from YAML or `--mode`)."""
    cfg = load_unified_config(config)

    if sample:
        cfg = cfg.with_overrides({"sample_name": sample})
    if date:
        cfg = cfg.with_overrides({"results_date": date})
    if mode:
        if mode not in ("single", "roi", "batch", "sweep"):
            raise typer.BadParameter(f"Invalid --mode {mode!r}; use single|roi|batch|sweep.")
        cfg = cfg.with_overrides({"registration.mode": mode})
    cfg = _apply_cli_overrides(cfg, set_override)

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
                                       help="Re-run vision even if its outputs already exist."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                  help="Resolve config + paths and print plan, no execution."),
) -> None:
    """Run vision (skipped if already done) and then registration on one sample."""
    cfg = load_unified_config(config)
    if sample:
        cfg = cfg.with_overrides({"sample_name": sample})
    if date:
        cfg = cfg.with_overrides({"results_date": date})
    cfg = _apply_cli_overrides(cfg, set_override)

    from .orchestrator import run_full

    run_full(cfg, force_cpu=force_cpu, force_vision=force_vision, dry_run=dry_run)


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
