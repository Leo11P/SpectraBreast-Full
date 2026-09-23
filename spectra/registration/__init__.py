"""HSI → mesh registration sub-package.

Public API:

- :func:`run_registration` — single dispatcher that reads a ``RegistrationConfig``
  and runs the right backend (single / roi / batch / sweep). This is what
  the ``spectra registration`` CLI calls.

- The individual modules (``pipeline``, ``pipeline_roi``, ``batch``,
  ``sweep``, ``roi_align``, ``render_gpu``) remain importable and are
  unchanged in behavior compared to the legacy ``spectrabreast/`` package.
"""

from .pipeline import (
    extract_suspicious_centroids,
    load_mesh,
    run_full_pipeline,
    save_render,
    save_turbo_render,
)
from .pipeline_roi import run_full_pipeline_roi
from .roi_align import compute_roi_to_png_homography, load_liveview_png
from .runner import run_registration

# render_gpu is optional (requires torch / cupy)
try:
    from .render_gpu import render_orthographic_topview_gpu
except ImportError:
    render_orthographic_topview_gpu = None

__all__ = [
    "compute_roi_to_png_homography",
    "extract_suspicious_centroids",
    "load_liveview_png",
    "load_mesh",
    "render_orthographic_topview_gpu",
    "run_full_pipeline",
    "run_full_pipeline_roi",
    "run_registration",
    "save_render",
    "save_turbo_render",
]
