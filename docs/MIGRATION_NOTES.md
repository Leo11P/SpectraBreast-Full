# Migration notes — Registration files

Each of these files is a **byte-for-byte copy** of the corresponding file in the
legacy `SpectraBreast-Registration` repo, with one mechanical change:

| Legacy import                            | New import       |
| ---------------------------------------- | ---------------- |
| `from spectrabreast.pipeline import ...` | `from .pipeline import ...` |
| `from spectrabreast.pipeline_roi import ...` | `from .pipeline_roi import ...` |
| `from spectrabreast.roi_align import ...` | `from .roi_align import ...` |
| `from spectrabreast.render_gpu import ...` | `from .render_gpu import ...` |
| `from spectrabreast import ...`         | `from . import ...` |

No logic is touched. The CLI entry points (`__main__`) inside `pipeline.py`,
`render_gpu.py`, and `roi_align.py` are kept so the files remain runnable as
standalone scripts during debugging.

Files migrated unchanged (modulo the import rewrite):

- `spectra/registration/pipeline.py`       ← `spectrabreast/pipeline.py`
- `spectra/registration/pipeline_roi.py`   ← `spectrabreast/pipeline_roi.py`
- `spectra/registration/roi_align.py`      ← `spectrabreast/roi_align.py`
- `spectra/registration/render_gpu.py`     ← `spectrabreast/render_gpu.py`

New files:

- `spectra/registration/__init__.py`       — public API + lazy `render_gpu` import
- `spectra/registration/runner.py`         — mode dispatcher (replaces `main.py`)
- `spectra/registration/batch.py`          — refactored discovery for `DATA/<sample>/`
- `spectra/registration/sweep.py`          — refactor of the legacy sweep (no behavior change)
