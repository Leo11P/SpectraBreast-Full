# -*- coding: utf-8 -*-
"""Validation of the pipeline against known ground truth (3D-printed
phantoms in a cassette with ArUco markers at known positions).
CLI: ``spectra validate``. Protocol: Protocollo_sperimentale_HSI.docx.

Modalities (one per protocol phase):

    vision        Phase 1: the reconstruction (surface_mesh.ply) against the
                  phantom STL — accuracy, completeness, Hausdorff, error vs height.
    registration  Phase 2: HSI -> 3D, leave-one-marker-out against the marker
                  corners reconstructed by vision — reprojection (px, mm), TRE
                  3D, in plane and in depth. Needs the sample's HSI cube
                  (DATA/input_registration/); skipped without it unless asked for.

Steps:

    run      validate one sample (its vision run) -> RESULTS/<date>/<sample>/validation/<modality>/
             with --all: every sample with a vision run whose name says its
             phantom (validation.sample_pattern / cases), then the summary
    summary  every validated sample under RESULTS -> RESULTS/_validation/<modality>_summary.xlsx

Settings: ``validation:`` in the unified config. Data, fixed layout:

    <data_root>/validation/
        Tabella_marker_cassetta.xlsx   marker positions on the cassette (CAD frame, mm)
        Target1+Cassetta.stl, ...      cassette + phantom assemblies (CAD frame, mm)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

MODALITIES = ("vision", "registration")
STEPS = ("run", "summary")


def plan(modality: Optional[str] = None, step: Optional[str] = None) -> list[tuple[str, str]]:
    """[(modality, step)] to run. Raises ValueError on a wrong combination."""
    if modality is not None and modality not in MODALITIES:
        raise ValueError(f"--modality must be one of {list(MODALITIES)}, got {modality!r}")
    step = step or "run"
    if step not in STEPS:
        raise ValueError(f"step must be one of {list(STEPS)}, got {step!r}")
    return [(m, step) for m in ((modality,) if modality else MODALITIES)]


def data_dir(cfg) -> Path:
    return Path(cfg.data_root) / "validation"


def output_dir(cfg, modality: str = "vision") -> Path:
    from ..orchestrator import sample_output_dir

    return sample_output_dir(cfg.results_root, cfg.sample_name, cfg.results_date) / "validation" / modality


def run_validation_stage(cfg, vision_dir: Optional[Path] = None,
                         modalities=MODALITIES) -> dict[str, Any]:
    """Validate the sample of ``cfg``. ``vision_dir`` defaults to this date's
    vision run, else the most recent one (same rule as registration)."""
    if cfg.validation is None:
        raise SystemExit("[validation] `validation:` section missing in the config")
    out: dict[str, Any] = {}
    if vision_dir is None:
        from ..orchestrator import find_vision_dir

        vision_dir, fallback = find_vision_dir(cfg.results_root, cfg.sample_name, cfg.results_date)
        if fallback:
            print(f"[validation] no vision run for this date: using {vision_dir}")
    if "vision" in modalities:
        from .vision import validate_vision

        out["vision"] = validate_vision(cfg, Path(vision_dir), output_dir(cfg, "vision"))
    if "registration" in modalities:
        from .registration import missing_inputs, validate_registration

        missing = missing_inputs(cfg, Path(vision_dir))
        if missing and list(modalities) != ["registration"]:
            # Every modality by default: a sample with no HSI cube only skips phase 2.
            print(f"[validation] registration skipped for {cfg.sample_name}: missing {missing}")
            out["registration"] = {"skipped": missing}
        else:
            out["registration"] = validate_registration(cfg, Path(vision_dir),
                                                        output_dir(cfg, "registration"))
    return out


def is_case(vcfg, sample: str) -> bool:
    """True if `sample` is a validation case: in validation.cases, or named as
    validation.sample_pattern with a phantom listed in validation.phantoms."""
    if sample in vcfg.cases:
        return True
    m = re.match(vcfg.sample_pattern, sample)
    return bool(m) and m.group("phantom") in vcfg.phantoms


def find_cases(cfg) -> dict[str, Path]:
    """{sample: vision run} of every validation case under results_root: the
    run of results_date when set and present, else the most recent one."""
    from datetime import datetime

    from ..orchestrator import DATE_FOLDER_FORMAT, is_date_folder, vision_dir_complete

    root = Path(cfg.results_root)
    days = sorted((d for d in root.iterdir() if d.is_dir() and is_date_folder(d.name)),
                  key=lambda d: datetime.strptime(d.name, DATE_FOLDER_FORMAT)) if root.is_dir() else []
    runs: dict[str, list[Path]] = {}
    for day in days:                                   # oldest -> newest
        for sdir in sorted(p for p in day.iterdir() if p.is_dir()):
            if is_case(cfg.validation, sdir.name) and vision_dir_complete(sdir / "vision"):
                runs.setdefault(sdir.name, []).append(sdir / "vision")
    cases: dict[str, Path] = {}
    for sample, vdirs in runs.items():
        pinned = [v for v in vdirs if v.parent.parent.name == cfg.results_date]
        cases[sample] = pinned[0] if pinned else vdirs[-1]
    return cases


def run_all(cfg, modalities=MODALITIES) -> dict[str, Any]:
    """Validate every case (find_cases), then the summary. A failing sample is
    reported and skipped."""
    cases = find_cases(cfg)
    if not cases:
        raise SystemExit(f"[validation] no vision run of a validation case under {cfg.results_root} "
                         "(names as validation.sample_pattern, e.g. T1_P1_R1, or validation.cases)")
    print(f"[validation] {len(cases)} cases: {', '.join(sorted(cases))}")
    done, failed = [], {}
    for sample, vdir in sorted(cases.items()):
        print("\n" + "-" * 68 + f"\n  {sample}\n" + "-" * 68)
        try:
            run_validation_stage(cfg.with_overrides({"sample_name": sample}), vision_dir=vdir,
                                 modalities=modalities)
            done.append(sample)
        except SystemExit as exc:
            failed[sample] = str(exc)
            print(f"[validation] {sample} FAILED: {exc}")
        except Exception as exc:               # keep going with the other samples
            failed[sample] = f"{type(exc).__name__}: {exc}"
            print(f"[validation] {sample} FAILED: {type(exc).__name__}: {exc}")
    print(f"\n[validation] validated {len(done)}/{len(cases)}"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    summary = run_summary(cfg, modalities) if done else {}
    return {"validated": done, "failed": failed, "summary": summary}


def run_summary(cfg, modalities=MODALITIES) -> dict[str, Any]:
    from .registration import summarize as summarize_registration
    from .report import summarize

    out: dict[str, Any] = {}
    for m in modalities:
        try:
            out[m] = summarize_registration(cfg) if m == "registration" else summarize(cfg, m)
        except SystemExit as exc:           # no validation of this modality yet
            if list(modalities) == [m]:
                raise
            print(f"{exc}")
    return out
