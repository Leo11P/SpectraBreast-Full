"""HSI → mesh registration sub-package.

Public API:

- :func:`run_registration` — single dispatcher that reads a ``RegistrationConfig``
  and runs the right backend (single / roi / batch / sweep). This is what
  the ``spectra registration`` CLI calls.

- The individual modules (``pipeline``, ``pipeline_roi``, ``batch``,
  ``sweep``, ``roi_align``, ``render_gpu``) remain importable and are
  unchanged in behavior compared to the legacy ``spectrabreast/`` package.
"""

# Every backend needs trimesh / openpyxl (and torch for GPU rendering):
# imported on first use, so ``spectra.registration.<module>`` imports stay
# explicit and the package itself imports with only the core dependencies.
_LAZY = {
    "extract_suspicious_centroids": ".pipeline",
    "load_mesh": ".pipeline",
    "run_full_pipeline": ".pipeline",
    "save_render": ".pipeline",
    "save_turbo_render": ".pipeline",
    "run_full_pipeline_roi": ".pipeline_roi",
    "compute_roi_to_png_homography": ".roi_align",
    "load_liveview_png": ".roi_align",
    "run_registration": ".runner",
    "render_orthographic_topview_gpu": ".render_gpu",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        try:
            value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        except ImportError:
            if name != "render_orthographic_topview_gpu":
                raise
            value = None          # optional: requires torch / cupy
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
