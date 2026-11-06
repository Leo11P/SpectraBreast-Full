# -*- coding: utf-8 -*-
"""Summary of every validated sample: RESULTS/_validation/<modality>_summary.xlsx
(+ .json); and write_sample_xlsx, the metrics of one sample in a readable
workbook (validation/vision/metrics.xlsx, next to metrics.json).

Sheets: Cases (one row per sample, every metric of every alignment),
Descriptive (per model: n, mean, std, median, Q1, Q3), Per phantom (same, per
model and phantom, primary alignment), Tests (models compared on the samples
they all have: Wilcoxon signed-rank for 2 models, Friedman + pairwise
Wilcoxon with Holm correction for 3 or more; needs scipy).

Only the newest run of each (sample, model) is kept (date folder).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

#: Metrics compared between models (primary alignment).
KEY_METRICS = ("accuracy.rmse_mm", "accuracy.mean_signed_mm", "accuracy.pct_lt_1mm",
               "accuracy.pct_lt_2mm", "completeness.median_abs_mm", "completeness.pct_lt_1mm",
               "completeness.pct_lt_2mm", "hausdorff_p95_mm", "fiducials.norm_rmse_mm",
               "fiducials.normal_rmse_mm", "fiducials.in_plane_rmse_mm")


def _date(name: str):
    try:
        return datetime.strptime(name, "%d-%m-%Y")
    except (TypeError, ValueError):
        return datetime.min


def collect(results_root: Path, modality: str = "vision") -> List[dict]:
    newest: Dict[tuple, dict] = {}
    for p in sorted(Path(results_root).glob(f"*/*/validation/{modality}/metrics.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[validation] skipped {p}: {exc}")
            continue
        r["date"] = p.parents[3].name
        key = (r.get("sample"), r.get("model"))
        if key not in newest or _date(r["date"]) > _date(newest[key]["date"]):
            newest[key] = r
    return list(newest.values())


def _flat(prefix: str, d: Any, out: dict) -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            _flat(f"{prefix}{k}.", v, out)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix[:-1]] = float(d)


def row_of(r: dict) -> dict:
    row = {k: r.get(k) for k in ("sample", "phantom", "pose", "rep", "model", "date",
                                 "primary_alignment", "scale_from")}
    row["kabsch.corner_rms_mm"] = r["alignment"]["kabsch"]["corner_rms_mm"]
    for name, m in r["metrics"].items():
        flat: dict = {}
        _flat("", {k: v for k, v in m.items() if k != "height"}, flat)
        for k, v in flat.items():
            row[f"{name}.{k}"] = v
        h = m.get("height", {})
        if "slope_mm_per_mm" in h:
            row[f"{name}.height.slope_mm_per_mm"] = h["slope_mm_per_mm"]
    for k in ("backend_s", "total_s"):
        if (r.get("timings") or {}).get(k) is not None:
            row[f"time.{k}"] = float(r["timings"][k])
    v = r.get("views") or {}
    for k in ("n_views", "n_views_seeing_phantom", "azimuth_coverage_deg"):
        if k in v:
            row[f"views.{k}"] = float(v[k])
    for k in ("distance_mm", "elevation_deg"):
        for s, x in (v.get(k) or {}).items():
            row[f"views.{k}.{s}"] = float(x)
    return row


def _desc(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "median": float(med), "q1": float(q1), "q3": float(q3)}


def _holm(ps: List[float]) -> List[float]:
    order = np.argsort(ps)
    adj, prev = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = prev
    return adj


def _block(r: dict):
    """Same physical case across models: (phantom, pose, rep), else the sample name."""
    key = (r.get("phantom"), r.get("pose"), r.get("rep"))
    return key if all(key) else (r.get("sample"),)


def tests(rows: List[dict], primary_of: Dict[str, str]) -> List[dict]:
    models = sorted({r["model"] for r in rows})
    if len(models) < 2:
        return [{"note": f"one model only ({', '.join(models)}): no comparison"}]
    try:
        from scipy import stats
    except ImportError:
        return [{"note": "scipy not installed: no statistical tests"}]
    by_model = {m: {_block(r): r for r in rows if r["model"] == m} for m in models}
    blocks = sorted(set.intersection(*(set(v) for v in by_model.values())), key=str)

    def wilcoxon(a, b):
        return float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0

    out = []
    for metric in KEY_METRICS:
        data = {m: np.array([by_model[m][b].get(f"{primary_of[m]}.{metric}", np.nan)
                             for b in blocks], dtype=float) for m in models}
        ok = np.all([np.isfinite(v) for v in data.values()], axis=0) if blocks else np.zeros(0, bool)
        n = int(ok.sum())
        if n < 3:
            out.append({"metric": metric, "note": f"only {n} samples common to all models"})
            continue
        if len(models) == 2:
            a, b = (data[m][ok] for m in models)
            out.append({"metric": metric, "test": "Wilcoxon", "models": " vs ".join(models),
                        "n": n, "p": wilcoxon(a, b)})
            continue
        cols = [data[m][ok] for m in models]
        if all(np.array_equal(cols[0], c) for c in cols[1:]):
            out.append({"metric": metric, "note": "identical values for every model"})
            continue
        p = float(stats.friedmanchisquare(*cols).pvalue)
        out.append({"metric": metric, "test": "Friedman", "models": ", ".join(models), "n": n, "p": p})
        pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
        raw = [wilcoxon(data[a][ok], data[b][ok]) for a, b in pairs]
        for (a, b), pr, pa in zip(pairs, raw, _holm(raw)):
            out.append({"metric": metric, "test": "Wilcoxon (Holm)", "models": f"{a} vs {b}",
                        "n": n, "p": pr, "p_holm": pa})
    return out


def _sheet(wb, title: str, rows: List[dict]) -> None:
    from openpyxl.styles import Font, PatternFill

    ws = wb.create_sheet(title)
    cols: List[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", start_color="305496")
    for r in rows:
        ws.append([round(v, 5) if isinstance(v, float) else v for v in (r.get(c) for c in cols)])
    for i, c in enumerate(cols, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = max(10, min(40, len(c) + 2))
    ws.freeze_panes = "B2"


# ------------------------------------------------- one sample: metrics.xlsx --
_FMT = {"mm": "0.00", "%": "0.0", "mm/mm": "0.000", "deg": "0.00", "n": "0", "s": "0"}


def _sample_rows(metrics: dict) -> List[tuple]:
    """(label, unit, dotted key in metrics[<alignment>], factor) or (section,)."""
    first = next(iter(metrics.values()))
    pct = sorted((k for k in first.get("accuracy", {}) if k.startswith("pct_lt_")),
                 key=lambda k: float(k[7:-2]))

    def dist(part):
        return ([("Median |error|", "mm", f"{part}.median_abs_mm", 1),
                 ("Mean |error|", "mm", f"{part}.mean_abs_mm", 1),
                 ("Mean signed error (+ = outside the STL)", "mm", f"{part}.mean_signed_mm", 1),
                 ("RMSE", "mm", f"{part}.rmse_mm", 1),
                 ("95th percentile of |error|", "mm", f"{part}.p95_abs_mm", 1),
                 ("Max |error|", "mm", f"{part}.max_abs_mm", 1)]
                + [(f"Within {k[7:-2]} mm", "%", f"{part}.{k}", 1) for k in pct]
                + [("Samples", "n", f"{part}.n", 1)])

    return ([("Surface accuracy: reconstruction -> STL",)] + dist("accuracy")
            + [("Completeness: STL seen by the cameras -> reconstruction",)] + dist("completeness")
            + [("Part of the STL seen by the cameras", "%", "completeness.visible_fraction", 100)]
            + [("Hausdorff",), ("95th percentile (larger of the two)", "mm", "hausdorff_p95_mm", 1)]
            + [("Error vs height above the markers (linear fit)",),
               ("Slope", "mm/mm", "height.slope_mm_per_mm", 1),
               ("At the marker plane", "mm", "height.intercept_mm", 1)]
            + [("Bersagli (fiducials)",),
               ("Found", "n", "fiducials.n_found", 1), ("In the STL", "n", "fiducials.n", 1),
               ("Error RMSE", "mm", "fiducials.norm_rmse_mm", 1),
               ("Error median", "mm", "fiducials.norm_median_mm", 1),
               ("Error max", "mm", "fiducials.norm_max_mm", 1),
               ("Along the normal, RMSE", "mm", "fiducials.normal_rmse_mm", 1),
               ("Along the normal, mean (+ = outside)", "mm", "fiducials.normal_mean_signed_mm", 1),
               ("In the surface plane, RMSE", "mm", "fiducials.in_plane_rmse_mm", 1)])


def _get(d: Any, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _zoom_text(nz) -> str:
    if not nz or float(nz.get("zoom") or 1.0) <= 1.0:
        return "1 (whole image)"
    w, h = nz["crop_px"][2:]
    W, H = nz.get("image_wh") or (None, None)
    return f"{float(nz['zoom']):.2f}x ({nz.get('mode')}): central {w}x{h} px of {W}x{H}"


def write_sample_xlsx(path: Path, r: dict) -> str:
    """The metrics of one validated sample (the dict of metrics.json) as a
    readable workbook: Summary (case, metrics per alignment, checks),
    Bersagli, Error vs height, Views, Markers, Warnings."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    head_font, head_fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", start_color="305496")
    sec_font, sec_fill = Font(bold=True), PatternFill("solid", start_color="D9E1F2")
    prim_fill = PatternFill("solid", start_color="FFF2CC")
    wb = Workbook()

    def header(ws, row, values):
        for j, v in enumerate(values, start=1):
            c = ws.cell(row, j, v)
            c.font, c.fill = head_font, head_fill
            c.alignment = Alignment(wrap_text=True, vertical="center")

    def widths(ws, w):
        for j, x in enumerate(w, start=1):
            ws.column_dimensions[get_column_letter(j)].width = x

    def num(c, v, unit):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            c.value = float(v) if isinstance(v, float) else v
            c.number_format = _FMT.get(unit, "0.00")
        else:
            c.value = v

    # ------------------------------------------------------------ Summary --
    ws = wb.active
    ws.title = "Summary"
    ws.cell(1, 1, f"Validation of {r.get('sample')} — {r.get('model')}, {r.get('date')}").font = \
        Font(bold=True, size=14)
    v, me = r.get("views") or {}, r.get("marker_edge_mm") or {}
    dist_, elev = v.get("distance_mm") or {}, v.get("elevation_deg") or {}
    si = r.get("scale_info") or {}
    scale_txt = r.get("scale_from", "markers")
    if scale_txt == "centers" and si.get("scale_from") == "centers":
        scale_txt += (f": centres fitted on the table, residual RMS {1000 * si['residual_rms_m']:.2f} mm; "
                      f"the marker edge would give {si['edge_vs_centers_pct']:+.2f}%")
    elif scale_txt == "markers":
        scale_txt += ": from the marker edge"
    info = [("Sample", r.get("sample")), ("Phantom", r.get("phantom")), ("Pose", r.get("pose")),
            ("Repetition", r.get("rep")), ("Model", r.get("model")), ("Date", r.get("date")),
            ("Primary alignment", r.get("primary_alignment")), ("Scale", scale_txt),
            ("MASt3R input zoom", _zoom_text(r.get("network_zoom"))),
            ("Marker edge, vision / table (mm)",
             f"{me.get('vision', float('nan')):.2f} / {me.get('table', float('nan')):.2f} "
             f"({me.get('table_source')})"),
            ("Views seeing the phantom", f"{v.get('n_views_seeing_phantom')} / {v.get('n_views')}"),
            ("Camera distance from the phantom (mm)",
             f"{dist_.get('mean', float('nan')):.0f} (min {dist_.get('min', float('nan')):.0f}, "
             f"max {dist_.get('max', float('nan')):.0f})" if dist_ else None),
            ("Camera elevation above the markers (deg)",
             f"{elev.get('min', float('nan')):.0f} - {elev.get('max', float('nan')):.0f}" if elev else None),
            ("Azimuth coverage (deg)", round(v["azimuth_coverage_deg"]) if "azimuth_coverage_deg" in v else None),
            ("Reconstruction time (s)", round((r.get("timings") or {}).get("backend_s") or 0) or None),
            ("Warnings", len(r.get("warnings") or [])), ("Vision run", r.get("vision_dir"))]
    row = 3
    for k, x in info:
        ws.cell(row, 1, k).font = sec_font
        ws.cell(row, 2, x)
        row += 1

    names = list(r.get("metrics") or {})
    primary = r.get("primary_alignment")
    row += 1
    header(ws, row, ["Metric", "Unit"] + [f"{n} (primary)" if n == primary else n for n in names])
    row += 1
    for spec in _sample_rows(r["metrics"]) if names else []:
        if len(spec) == 1:
            ws.cell(row, 1, spec[0]).font = sec_font
            for j in range(1, len(names) + 3):
                ws.cell(row, j).fill = sec_fill
        else:
            label, unit, key, factor = spec
            ws.cell(row, 1, label)
            ws.cell(row, 2, unit)
            for j, n in enumerate(names, start=3):
                x = _get(r["metrics"][n], key)
                c = ws.cell(row, j)
                num(c, x * factor if isinstance(x, (int, float)) and factor != 1 else x, unit)
                if n == primary:
                    c.fill = prim_fill
        row += 1

    al = r.get("alignment") or {}
    checks = [("Markers: kabsch corner residual RMS", "mm", "kabsch.corner_rms_mm"),
              ("Markers: kabsch corner residual max", "mm", "kabsch.corner_max_mm"),
              ("Markers: frame corner residual RMS", "mm", "frame.corner_rms_mm"),
              ("Frame vs kabsch: rotation", "deg", "frame.vs_kabsch_rotation_deg"),
              ("Frame vs kabsch: shift at the phantom", "mm", "frame.vs_kabsch_shift_at_roi_mm"),
              ("ICP correction: rotation", "deg", "icp.rotation_deg"),
              ("ICP correction: shift at the phantom", "mm", "icp.shift_at_roi_mm"),
              ("ICP final RMS", "mm", "icp.rms_mm")]
    row += 1
    header(ws, row, ["Alignment check", "Unit", "Value"])
    row += 1
    for label, unit, key in checks:
        x = _get(al, key)
        if x is None:
            continue
        ws.cell(row, 1, label)
        ws.cell(row, 2, unit)
        num(ws.cell(row, 3), x, unit)
        row += 1
    if r.get("scale_from") == "centers":
        ws.cell(row, 1, "Scale from the marker centres: the marker residuals above are not an "
                        "independent check (phantom and bersagli are)").font = Font(italic=True)
    widths(ws, [44, 8] + [16] * max(len(names), 1))
    ws.column_dimensions["B"].width = 9
    for rr in range(3, 3 + len(info)):
        ws.cell(rr, 2).alignment = Alignment(horizontal="left")

    # ----------------------------------------------------------- Bersagli --
    fid = r.get("fiducials") or {}
    items = fid.get("items") or []
    if items:
        ws = wb.create_sheet("Bersagli")
        # the primary alignment in detail, the others as |error| only
        others = [n for n in names if n != primary]
        cols = ["#", "Status", "Views used", "Views tried", "Spread between views (mm)",
                "CAD x (mm)", "CAD y (mm)", "CAD z (mm)",
                f"{primary}: |error| (mm)", f"{primary}: along the normal (mm)",
                f"{primary}: in plane (mm)", f"{primary}: error x (mm)", f"{primary}: error y (mm)",
                f"{primary}: error z (mm)"] + [f"{n}: |error| (mm)" for n in others]
        header(ws, 1, cols)
        for i, it in enumerate(items, start=2):
            err = it.get("error") or {}
            e = err.get(primary) or {}
            vals = ([it.get("id"), it.get("status"), it.get("n_views_used"), it.get("n_views_tried"),
                     it.get("spread_mm")] + list(it.get("cad_mm") or [None] * 3)
                    + [e.get("norm_mm"), e.get("normal_mm"), e.get("in_plane_mm")]
                    + list(e.get("xyz_mm") or [None] * 3)
                    + [(err.get(n) or {}).get("norm_mm") for n in others])
            for j, x in enumerate(vals, start=1):
                num(ws.cell(i, j), x, "n" if j in (1, 3, 4) else "mm")
                if 9 <= j <= 14:
                    ws.cell(i, j).fill = prim_fill
        ws.cell(len(items) + 3, 1, f"Error = measured - expected (output frame, mm); expected from the "
                                   f"alignment, + along the normal = outside the STL. Search radius "
                                   f"{fid.get('search_radius_mm') or float('nan'):.2f} mm. "
                                   f"{fid.get('note') or ''}").font = Font(italic=True)
        widths(ws, [5, 30, 8, 8, 12] + [10] * (len(cols) - 5))
        ws.freeze_panes = "C2"

    # ---------------------------------------------------- Error vs height --
    if names and (r["metrics"][names[0]].get("height") or {}).get("bins"):
        ws = wb.create_sheet("Error vs height")
        cols = ["Height from (mm)", "Height to (mm)", "Samples"]
        for n in names:
            cols += [f"{n}: mean signed error (mm)", f"{n}: RMSE (mm)"]
        header(ws, 1, cols)
        bins = {n: r["metrics"][n]["height"]["bins"] for n in names}
        for i, b in enumerate(bins[names[0]], start=2):
            vals = [b["z_from_mm"], b["z_to_mm"], b["n"]]
            for n in names:
                bb = bins[n][i - 2] if i - 2 < len(bins[n]) else {}
                vals += [bb.get("mean_signed_mm"), bb.get("rmse_mm")]
            for j, x in enumerate(vals, start=1):
                num(ws.cell(i, j), x, "n" if j == 3 else "mm")
        ws.cell(len(bins[names[0]]) + 3, 1, "Height above the marker plane (output frame); "
                                             "+ = reconstruction outside the STL").font = Font(italic=True)
        widths(ws, [12, 12, 10] + [14] * (len(cols) - 3))
        ws.freeze_panes = "A2"

    # -------------------------------------------------------------- Views --
    if v.get("per_view"):
        ws = wb.create_sheet("Views")
        header(ws, 1, ["View", "Distance from the phantom (mm)", "Elevation above the markers (deg)",
                       "Azimuth (deg)", "Sees the phantom"])
        for i, p in enumerate(v["per_view"], start=2):
            ws.cell(i, 1, p["view"])
            num(ws.cell(i, 2), p["distance_mm"], "s")
            num(ws.cell(i, 3), p["elevation_deg"], "s")
            num(ws.cell(i, 4), p["azimuth_deg"], "s")
            ws.cell(i, 5, "yes" if p["sees_phantom"] else "no")
        widths(ws, [7, 16, 16, 12, 12])
        ws.freeze_panes = "A2"

    # ------------------------------------------------------------ Markers --
    loo = _get(al, "kabsch.leave_one_marker_out_mm") or {}
    res_c = si.get("residual_per_marker_m") or {}
    if loo or res_c:
        ws = wb.create_sheet("Markers")
        cols = ["Marker", "Leave-one-marker-out error (mm)"]
        if res_c:
            cols.append("Centre residual after the scale fit (mm)")
        header(ws, 1, cols)
        for i, m in enumerate(sorted(set(loo) | set(res_c), key=int), start=2):
            ws.cell(i, 1, int(m))
            num(ws.cell(i, 2), loo.get(m), "mm")
            if res_c:
                num(ws.cell(i, 3), None if m not in res_c else 1000 * res_c[m], "mm")
        widths(ws, [8, 18, 22])
        ws.freeze_panes = "A2"

    # ----------------------------------------------------------- Warnings --
    if r.get("warnings"):
        ws = wb.create_sheet("Warnings")
        header(ws, 1, ["Warning"])
        for i, w in enumerate(r["warnings"], start=2):
            ws.cell(i, 1, w).alignment = Alignment(wrap_text=True)
        ws.column_dimensions["A"].width = 120

    for ws in wb.worksheets:                  # printable: landscape, one page wide
        ws.page_setup.orientation = "landscape"
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    wb.save(path)
    return Path(path).name


def summarize(cfg, modality: str = "vision") -> Path:
    from openpyxl import Workbook

    results = collect(cfg.results_root, modality)
    if not results:
        raise SystemExit(f"[validation] no {modality} validation under {cfg.results_root}"
                         f"/*/*/validation/{modality}/: run `spectra validate` first")
    rows = [row_of(r) for r in results]
    primary_of: Dict[str, str] = {}
    for r in results:
        primary_of.setdefault(r["model"], r.get("primary_alignment", "kabsch"))
    numeric = sorted({k for r in rows for k, v in r.items() if isinstance(v, float)})

    desc, per_phantom = [], []
    for m in sorted({r["model"] for r in rows}):
        rm = [r for r in rows if r["model"] == m]
        for k in numeric:
            desc.append({"model": m, "metric": k,
                         **_desc(np.array([r.get(k, np.nan) for r in rm], dtype=float))})
        for ph in sorted({r["phantom"] for r in rm}, key=str):
            rp = [r for r in rm if r["phantom"] == ph]
            for k in KEY_METRICS:
                key = f"{primary_of[m]}.{k}"
                per_phantom.append({"model": m, "phantom": ph, "metric": key,
                                    **_desc(np.array([r.get(key, np.nan) for r in rp], dtype=float))})
    test_rows = tests(rows, primary_of)

    out_dir = Path(cfg.results_root) / "_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = out_dir / f"{modality}_summary.xlsx"
    wb = Workbook()
    wb.remove(wb.active)
    _sheet(wb, "Cases", rows)
    _sheet(wb, "Descriptive", desc)
    _sheet(wb, "Per phantom", per_phantom)
    _sheet(wb, "Tests", test_rows)
    wb.save(xlsx)
    (out_dir / f"{modality}_summary.json").write_text(json.dumps(
        {"cases": rows, "descriptive": desc, "per_phantom": per_phantom, "tests": test_rows},
        indent=2), encoding="utf-8")
    print(f"[validation] {len(rows)} samples, {len({r['model'] for r in rows})} model(s) -> {xlsx}")
    return xlsx
