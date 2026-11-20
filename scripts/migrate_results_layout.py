#!/usr/bin/env python3
"""Move results saved in the old layout into the per-model layout.

    old:  RESULTS/<dd-mm-yyyy>/<sample>/{vision, registration, validation, _job, ...}
    new:  RESULTS/<dd-mm-yyyy>/<sample>/<model>/{vision, registration, validation, _job, ...}

<model> = <backend>[-<variant>] (e.g. mast3r-sfm, mast3r-dense), as the
pipeline names it (spectra/vision/backends/registry.py::model_tag). It is read,
in this order, from:
  1. <sample>/vision/reconstruction_metadata.json ("model"), or the vision
     config saved with the run (<sample>/vision/run.json: backend + variant);
  2. the config of the last Mufasa job, <sample>/_job/run_config.yaml;
  3. --model, for the samples where neither exists (e.g. a registration made
     on the vision run of another day).

Dry run by default: prints what it would move. --apply moves. Nothing is
deleted; an entry already present in the model folder is left where it is and
reported. Run it where results are stored (this PC and, on Mufasa, in
mufasa.repo_dir).

Not moved: RESULTS/<date>/_batch/batch_summary.xlsx (its model is not written
in it; the new runs go to _batch/<model>/), RESULTS/_validation/ (the summary:
rebuild it with `spectra validate --summary`).

    python scripts/migrate_results_layout.py [--results-root RESULTS] [--model mast3r-sfm] [--apply]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD_LAYOUT = ("vision", "registration", "validation", "_job")


def _registry():
    path = REPO / "spectra" / "vision" / "backends" / "registry.py"
    spec = importlib.util.spec_from_file_location("spectra_backend_registry", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


REG = _registry()


def _is_date(name: str) -> bool:
    try:
        datetime.strptime(name, "%d-%m-%Y")
    except ValueError:
        return False
    return True


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def model_of(sample_dir: Path) -> tuple[str | None, str]:
    """(model tag, where it was read) of an old-layout sample folder."""
    vision = sample_dir / "vision"
    meta = _read_json(vision / "reconstruction_metadata.json") or {}
    if meta.get("model"):
        return str(meta["model"]), "vision/reconstruction_metadata.json"
    run = _read_json(vision / "run.json")
    if isinstance(run, dict):
        return REG.model_tag(run), "vision/run.json"
    job_cfg = sample_dir / "_job" / "run_config.yaml"
    if job_cfg.is_file():
        try:
            import yaml

            full = yaml.safe_load(job_cfg.read_text(encoding="utf-8")) or {}
            return REG.model_tag(full.get("vision") or {}), "_job/run_config.yaml"
        except Exception as exc:
            return None, f"_job/run_config.yaml unreadable ({exc})"
    return None, "no vision run and no _job/run_config.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-root", default=str(REPO / "RESULTS"))
    ap.add_argument("--model", default=None,
                    help="model for the samples whose model cannot be read (e.g. mast3r-sfm)")
    ap.add_argument("--apply", action="store_true", help="move (default: only print)")
    args = ap.parse_args()

    root = Path(args.results_root)
    if not root.is_dir():
        print(f"{root} does not exist")
        return 1
    if args.model:
        known = sorted(REG.BACKENDS)
        if args.model.split("-")[0] not in known:
            print(f"--model {args.model!r}: the backend must be one of {known}")
            return 1

    moved = skipped = 0
    unknown: list[Path] = []
    for day in sorted(p for p in root.iterdir() if p.is_dir() and _is_date(p.name)):
        for sample_dir in sorted(p for p in day.iterdir() if p.is_dir() and not p.name.startswith("_")):
            if not any((sample_dir / n).exists() for n in OLD_LAYOUT):
                continue                                   # already per-model (or empty)
            model, source = model_of(sample_dir)
            if model is None:
                if not args.model:
                    unknown.append(sample_dir)
                    print(f"?  {sample_dir}: model unknown ({source}); rerun with --model")
                    continue
                model, source = args.model, "--model"
            target = sample_dir / model
            entries = [e for e in sorted(sample_dir.iterdir()) if e.name != model]
            # other model folders already there (new layout next to old): keep them
            entries = [e for e in entries
                       if not (e.is_dir() and any((e / n).exists() for n in OLD_LAYOUT))]
            print(f"{'->' if args.apply else 'would move'} {sample_dir} -> {target.name}/  "
                  f"(model from {source}): {[e.name for e in entries]}")
            if args.apply:
                target.mkdir(exist_ok=True)
            for e in entries:
                dst = target / e.name
                if dst.exists():
                    print(f"   !  {dst} exists: {e.name} left where it is")
                    skipped += 1
                    continue
                if args.apply:
                    shutil.move(str(e), str(dst))
                moved += 1

    batch = sorted(root.glob("*/_batch/batch_summary.xlsx"))
    if batch:
        print(f"\nNot moved (model not recorded): {len(batch)} _batch/batch_summary.xlsx, "
              f"e.g. {batch[0]}")
    if (root / "_validation").is_dir():
        print(f"Not moved: {root / '_validation'} (rebuild with `spectra validate --summary`)")
    print(f"\n{'Moved' if args.apply else 'Would move'} {moved} entries; {skipped} left (already "
          f"there); {len(unknown)} sample folders with unknown model.")
    if not args.apply and moved:
        print("Dry run: nothing changed. Rerun with --apply to move.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
