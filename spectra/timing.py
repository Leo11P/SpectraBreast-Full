# -*- coding: utf-8 -*-
"""Timing of a run: how long every phase and sub-phase took, from the
acquisition to the download of the results.

Standard library only (+ openpyxl for the Excel), so the standalone scripts
of spectra/acquisition/ load it by path. Two files per sample, in
RESULTS/<dd-mm-yyyy>/<sample>/validation/timing/:

    timing_local.json   written on THIS PC: acquisition, upload to Mufasa,
                        SLURM queue and job, download
    timing_job.json     written where the stages ran (Mufasa, or this PC):
                        vision, registration, validation

timing.xlsx merges the two (rebuilt by whoever writes last: the job, then the
download on this PC). Each file: {"run": {...}, "phases": [{phase, where,
start, end, seconds, subphases: [{name, seconds}], note}]}; start / end are
epoch seconds.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

LOCAL = "timing_local.json"
JOB = "timing_job.json"
XLSX = "timing.xlsx"


def timing_dir(results_root, date: str, sample: str) -> Path:
    return Path(results_root) / date / sample / "validation" / "timing"


def _read(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {"run": {}, "phases": []}


def _write(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def reset(path: Path, **run) -> None:
    """Start a new run in this file (the entries of an earlier run are dropped)."""
    _write(path, {"run": {"started": time.time(), **run}, "phases": []})


def add(path: Path, phase: str, start: float, end: float, *, where: str = "",
        subphases: Optional[List[Dict]] = None, note: str = "") -> None:
    """Append a phase (start / end: epoch seconds); subphases: [{name, seconds}]."""
    data = _read(path)
    data.setdefault("phases", []).append({
        "phase": phase, "where": where, "start": start, "end": end,
        "seconds": round(end - start, 3),
        "subphases": [{"name": s["name"], "seconds": round(float(s["seconds"]), 3)}
                      for s in (subphases or []) if s.get("seconds") is not None],
        "note": note})
    _write(path, data)


@contextmanager
def phase(path: Optional[Path], name: str, where: str = ""):
    """Times the block; yields a list to which (name, seconds) sub-phases can be
    appended. path None: times nothing (no run being recorded)."""
    subs: List[Dict] = []
    t0 = time.time()
    try:
        yield subs
    finally:
        if path is not None:
            try:
                add(path, name, t0, time.time(), where=where, subphases=subs)
            except Exception as exc:          # timing must never break a run
                print(f"[timing] not recorded ({name}): {exc}")


def _fmt(t: Optional[float]) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") if t else ""


def merged(tdir: Path) -> List[dict]:
    """Phases of both files, in chronological order."""
    phases = _read(Path(tdir) / LOCAL).get("phases", []) + _read(Path(tdir) / JOB).get("phases", [])
    return sorted(phases, key=lambda p: (p.get("start") or 0))


def write_xlsx(tdir: Path, sample: str = "", date: str = "") -> Optional[Path]:
    """timing.xlsx (+ timing.json, the merged phases) from the two files."""
    tdir = Path(tdir)
    phases = merged(tdir)
    if not phases:
        return None
    t0 = min(p["start"] for p in phases)
    t1 = max(p["end"] for p in phases)
    total = t1 - t0
    rows = []
    for p in phases:
        rows.append(p)
        if p["phase"] == "SLURM job (on Mufasa)":
            inside = sum(q["seconds"] for q in phases
                         if q.get("where") == "Mufasa" and q is not p
                         and p["start"] - 1 <= q["start"] <= p["end"] + 1)
            if inside:
                p = dict(p)
                p.setdefault("subphases", [])
                p["subphases"] = list(p["subphases"]) + [
                    {"name": "stages (vision + registration + validation)", "seconds": round(inside, 3)},
                    {"name": "job overhead (environment, model loading, I/O)",
                     "seconds": round(p["seconds"] - inside, 3)}]
                rows[-1] = p
    (tdir / "timing.json").write_text(json.dumps(
        {"sample": sample, "date": date, "start": t0, "end": t1, "total_s": round(total, 3),
         "phases": rows}, indent=2), encoding="utf-8")

    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    bold, head = Font(bold=True), PatternFill("solid", fgColor="DDEBF7")
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    for k, v in (("Sample", sample), ("Run date", date), ("Start", _fmt(t0)), ("End", _fmt(t1)),
                 ("Total (wall clock) [s]", round(total, 1)),
                 ("Total (wall clock) [min]", round(total / 60, 2))):
        ws.append([k, v])
        ws.cell(ws.max_row, 1).font = bold
    ws.append([])
    ws.append(["Phase", "Where", "Start", "End", "Seconds", "Minutes", "% of total", "Note"])
    for c in ws[ws.max_row]:
        c.font, c.fill = bold, head
    for p in rows:
        ws.append([p["phase"], p.get("where", ""), _fmt(p["start"]), _fmt(p["end"]),
                   round(p["seconds"], 1), round(p["seconds"] / 60, 2),
                   round(100 * p["seconds"] / total, 1) if total else None, p.get("note", "")])
    ws.append([])
    ws.append(["Phases overlap (e.g. the SLURM job contains vision, registration and "
               "validation): the total is the wall clock from the first start to the last end."])

    wd = wb.create_sheet("Sub-phases")
    wd.append(["Phase", "Sub-phase", "Seconds", "Minutes", "% of phase"])
    for c in wd[1]:
        c.font, c.fill = bold, head
    for p in rows:
        wd.append([p["phase"], "(total)", round(p["seconds"], 1), round(p["seconds"] / 60, 2), 100.0])
        wd.cell(wd.max_row, 1).font = bold
        for s in p.get("subphases", []):
            wd.append(["", s["name"], round(s["seconds"], 1), round(s["seconds"] / 60, 2),
                       round(100 * s["seconds"] / p["seconds"], 1) if p["seconds"] else None])
    for sheet, widths in ((ws, (34, 10, 20, 20, 10, 10, 11, 40)), (wd, (34, 52, 10, 10, 11))):
        for i, w in enumerate(widths):
            sheet.column_dimensions[chr(65 + i)].width = w
    out = tdir / XLSX
    wb.save(out)
    print(f"[timing] total {total / 60:.1f} min -> {out}")
    return out
