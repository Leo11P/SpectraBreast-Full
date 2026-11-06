#!/usr/bin/env python3
# =============================================================================
#  remote.py  —  run vision / registration on Mufasa (SLURM) from this PC
# -----------------------------------------------------------------------------
#  Used by the CLI when the config has `mufasa.enabled: true`:
#      spectra full           acquisition here, then vision + registration on Mufasa
#      spectra vision         vision on Mufasa
#      spectra registration   registration on Mufasa
#  or by hand, to (re)submit a sample already on disk:
#      python spectra/acquisition/remote.py --sample SB030 [--stages vision registration]
#  With `mufasa.enabled: false` nothing is ever sent to Mufasa.
#
#  The cluster cannot open connections towards this PC, so the flow is always
#  PC -> Mufasa over SSH/SFTP. On Mufasa everything lives in the repo clone
#  (mufasa.repo_dir), with the same layout as here:
#      <repo_dir>/DATA/input_vision/<sample>/        <repo_dir>/RESULTS/<dd-mm-yyyy>/<sample>/
#      <repo_dir>/DATA/input_registration/<sample>/      vision/  registration/  _job/
#
#    1. one SSH connection to the login node (key from the config, or password)
#    2. inputs, by sample name — only what is missing or changed is uploaded
#       (same size + mtime = already there, like rsync):
#         vision        DATA/input_vision/<sample>/  mirrored from this PC
#                       (stale remote files removed), checked with SHA256SUMS
#         registration  DATA/input_registration/<sample>/ (HSI cube); if it is
#                       not on this PC, the copy already on Mufasa is used.
#                       Without vision in the same job: the vision run already
#                       on Mufasa, or this PC's if newer (mesh + ArUco JSON)
#    3. RESULTS/<date>/<sample>/_job/run_config.yaml = the LOCAL config + the CLI
#       overrides, with sample_name / data_root / results_root / results_date /
#       stages set for the run and the robot-PC sections (`acquisition:`,
#       `calibration:`, `mufasa:`) removed, so the job runs on the GPU node and
#       never submits again. The job sets SPECTRA_DEFAULT_CONFIG to it: for that
#       run it IS the default config (single source of the marker edge); the
#       repo's configs/default.yaml on Mufasa is neither read nor modified.
#    4. sbatch a job script that runs `spectra full` on a GPU node
#    5. download only mufasa.pull_patterns (default: the registration Excel
#       reports, the spectral Rerun recording, the validation and the timing)
#       into the local RESULTS/<date>/<sample>/, from a process DETACHED from
#       the terminal (`--watch`, mufasa.detach_pull): closing the terminal does
#       not stop it. Log: RESULTS/<date>/<sample>/_job/pull_<jobid>.log.
#       Everything else stays on Mufasa.
#
#  Requires: paramiko, pyyaml   (pip install paramiko pyyaml)
# =============================================================================
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import fnmatch
import getpass
import hashlib
import json
import os
import posixpath
import shlex
import stat
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "default.yaml")

COMPUTE_STAGES = ("vision", "registration", "validation")   # the stages run on Mufasa
DATE_FORMAT = "%d-%m-%Y"                         # RESULTS/<dd-mm-yyyy>/
# Vision run files registration reads (first two required, as in the pipeline).
VISION_REQUIRED = ("surface_mesh.ply", "aruco_markers_3d.json")
VISION_HANDOFF = VISION_REQUIRED + ("timings.json", "reconstruction_metadata.json")
# ... and what the validation reads too.
VISION_VALIDATION = VISION_HANDOFF + ("camera_poses_output_frame.npy",
                                      "intrinsics_original_frame.npy")

# SLURM states meaning the job has left the queue.
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                   "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
SKIP_UPLOAD = {"_discarded"}                     # never uploaded, never mirrored

DEFAULTS = {
    "enabled": False,
    "host": None, "port": 22, "user": None, "ssh_key_path": None, "repo_dir": None,
    "slurm_bin_dirs": [],
    "partition": "jobs", "qos": "gpuwide", "gres": "gpu:4g.20gb:1",
    "cpus": 8, "mem": "64G", "time": "12:00:00", "job_name": "spectra",
    "wait": True, "poll_s": 20, "pull_results": True, "detach_pull": True,
    "pull_patterns": ["registration/*.xlsx", "registration/*.rrd", "registration/*.rbl",
                      "validation/vision/*", "validation/registration/*", "validation/timing/*"],
}
# Always downloaded with the stages' results: the timing of the run (spectra/timing.py).
ALWAYS_PULL = ("validation/timing/*",)


def _timing():
    """spectra/timing.py, loaded by path (this script runs standalone)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "spectra_timing", os.path.join(os.path.dirname(ROOT), "timing.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def set_dotted(root, dotted_key, value):
    """root["a"]["b"]["c"] = value for "a.b.c" (same syntax as `--set`)."""
    parts = dotted_key.split(".")
    cur = root
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_settings(config_path, overrides=None):
    """Returns (full_cfg_dict with the overrides applied, mufasa settings,
    sample, local paths)."""
    with open(config_path, "r", encoding="utf-8") as f:
        full = yaml.safe_load(f) or {}
    for key, value in (overrides or {}).items():
        set_dotted(full, key, value)
    if "remote" in (full.get("acquisition") or {}):
        raise SystemExit("[remote] acquisition.remote has moved to the top-level "
                         f"`mufasa:` section: update {config_path}")
    r = dict(DEFAULTS)
    r.update(full.get("mufasa") or {})
    if not r["enabled"]:
        raise SystemExit(f"[remote] mufasa.enabled is false in {config_path}: "
                         "nothing is sent to Mufasa")
    missing = [k for k in ("host", "user", "repo_dir") if not r.get(k)]
    if missing:
        raise SystemExit(f"[remote] mufasa.{missing} missing in {config_path}")
    sample = full.get("sample_name")
    if not sample:
        raise SystemExit("[remote] sample_name missing (config or --sample)")
    data_root = os.path.abspath(str(full.get("data_root", "DATA")))
    paths = {
        "vision": os.path.join(data_root, "input_vision", sample),
        "registration": os.path.join(data_root, "input_registration", sample),
        "validation": os.path.join(data_root, "validation"),
        "results_root": os.path.abspath(str(full.get("results_root", "RESULTS"))),
        # on Mufasa: same layout, inside the repo clone
        "r_data": posixpath.join(r["repo_dir"], "DATA"),
        "r_results": posixpath.join(r["repo_dir"], "RESULTS"),
    }
    return full, r, sample, paths


def marker_edge_mm(config_path, full):
    """Marker edge length (mm) used locally. Like the pipeline, it is read ONLY
    from the repo's configs/default.yaml; another config must not set it."""
    if os.path.abspath(config_path) != os.path.abspath(DEFAULT_CONFIG):
        if ((full.get("vision") or {}).get("aruco") or {}).get("marker_edge_length_mm") is not None:
            raise SystemExit(f"[remote] {config_path}: vision.aruco.marker_edge_length_mm "
                             f"cannot be set here, only in {DEFAULT_CONFIG}")
        with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
    value = ((full.get("vision") or {}).get("aruco") or {}).get("marker_edge_length_mm")
    if value is None:
        raise SystemExit(f"[remote] vision.aruco.marker_edge_length_mm missing in {DEFAULT_CONFIG}")
    return float(value)


def check_portable(full):
    """Paths set by hand point at THIS PC and do not exist on Mufasa."""
    vis, reg = full.get("vision") or {}, full.get("registration") or {}
    bad = [f"vision.input.{k}" for k in ("rgb_dir", "pose_dir", "camera_params_dir")
           if (vis.get("input") or {}).get(k)]
    if (vis.get("output") or {}).get("root"):
        bad.append("vision.output.root")
    bad += [f"registration.paths.{k}" for k in
            ("hsi_hdr", "mesh", "aruco_json", "liveview_png", "output_dir")
            if (reg.get("paths") or {}).get(k)]
    if bad:
        raise SystemExit(f"[remote] {bad} set by hand: on Mufasa every path is derived "
                         "from sample_name. Set them back to null (or mufasa.enabled: false).")


def _day(name):
    try:
        return _dt.datetime.strptime(name, DATE_FORMAT)
    except ValueError:
        return None


def vision_runs(results_root, sample, isfile, listdir, join):
    """{day: vision_dir} of the complete vision runs (mesh + ArUco JSON) of
    `sample`. Works on both sides through isfile / listdir / join."""
    out = {}
    for name in listdir(results_root):
        d = join(results_root, name, sample, "vision")
        if _day(name) and all(isfile(join(d, f)) for f in VISION_REQUIRED):
            out[name] = d
    return out


def vision_files(full, stages):
    """Files of a vision run the job reads: + cloud.ply when registration (or
    its validation) or the vision validation works on the point cloud."""
    files = VISION_VALIDATION if "validation" in stages else VISION_HANDOFF
    reg_cloud = (("registration" in stages or "validation" in stages)
                 and (full.get("registration") or {}).get("source") == "cloud")
    val_cloud = "validation" in stages and (
        ((full.get("validation") or {}).get("vision") or {}).get("source") == "cloud")
    return files + ("cloud.ply",) if (reg_cloud or val_cloud) else files


def registration_inputs(full, sample):
    reg = full.get("registration") or {}
    mode = reg.get("mode", "single")
    roi = mode == "roi" or (mode == "sweep" and (reg.get("sweep") or {}).get("roi_mode"))
    return [f"{sample}_raw.hdr"] + ([f"{sample}_raw.png"] if roi else [])


# --------------------------------------------------------------------------- #
# SSH / SFTP helpers
# --------------------------------------------------------------------------- #
def connect(r):
    try:
        import paramiko
    except ImportError:
        raise SystemExit("[remote] missing dependency: pip install paramiko") from None
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(hostname=r["host"], port=int(r["port"]), username=r["user"],
              timeout=30, look_for_keys=False, allow_agent=False)
    if r.get("ssh_key_path"):
        key = os.path.expanduser(str(r["ssh_key_path"]))
        if not os.path.isfile(key):
            raise SystemExit(f"[remote] SSH key {key} not found (mufasa.ssh_key_path)")
        kw["key_filename"] = key
    else:
        kw["password"] = getpass.getpass(f"Password for {r['user']}@{r['host']}: ")
    print(f"[remote] connecting to {r['user']}@{r['host']} ...")
    try:
        client.connect(**kw)
    except Exception as e:
        raise SystemExit(f"[remote] cannot connect to {r['host']}: {e} "
                         "(Polimi LAN/VPN on? key authorised on Mufasa?)") from None
    return client


def run_cmd(client, r, command):
    """Runs `command` on the login node in an interactive login shell
    (`bash -ilc`), with the SLURM binaries prepended to PATH. Non-interactive
    shells skip most of ~/.bashrc, and sbatch copies the caller's environment
    into the job: without -i the job missed env vars set there (this caused
    `ModuleNotFoundError: mast3r` in the past)."""
    dirs = r.get("slurm_bin_dirs") or []
    prefix = f"export PATH={':'.join(dirs)}:$PATH && " if dirs else ""
    _, stdout, stderr = client.exec_command(f"bash -ilc {shlex.quote(prefix + command)}")
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out.strip(), err.strip()


def check_remote_repo(sftp, r):
    """Fail early (before uploading) if mufasa.repo_dir is wrong."""
    path = posixpath.join(r["repo_dir"], "pyproject.toml")
    try:
        sftp.stat(path)
    except FileNotFoundError:
        raise SystemExit(f"[remote] {path} not found on Mufasa: check mufasa.repo_dir") from None


def r_isfile(sftp, path):
    try:
        return stat.S_ISREG(sftp.stat(path).st_mode)
    except (FileNotFoundError, OSError):
        return False


def r_listdir(sftp, path):
    try:
        return sftp.listdir(path)
    except (FileNotFoundError, OSError):
        return []


def sftp_makedirs(sftp, remote_dir):
    cur = "/"
    for part in remote_dir.strip("/").split("/"):
        cur = posixpath.join(cur, part)
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def local_files(local_dir, only=None):
    """{relative posix path: (absolute path, size, mtime)}."""
    out = {}
    if only is not None:
        names = [n for n in only if os.path.isfile(os.path.join(local_dir, n))]
    else:
        names = []
        for base, dirs, files in os.walk(local_dir):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_UPLOAD)
            names += [os.path.relpath(os.path.join(base, f), local_dir) for f in sorted(files)]
    for rel in names:
        full = os.path.join(local_dir, rel)
        st = os.stat(full)
        out[rel.replace(os.sep, "/")] = (full, st.st_size, int(st.st_mtime))
    return out


def remote_files(sftp, remote_dir, rel=""):
    """{relative posix path: (size, mtime)} under remote_dir (empty if missing)."""
    out = {}
    try:
        entries = sftp.listdir_attr(posixpath.join(remote_dir, rel) if rel else remote_dir)
    except (FileNotFoundError, OSError):
        return out
    for e in entries:
        sub = posixpath.join(rel, e.filename) if rel else e.filename
        if stat.S_ISDIR(e.st_mode):
            if e.filename not in SKIP_UPLOAD:
                out.update(remote_files(sftp, remote_dir, sub))
        else:
            out[sub] = (e.st_size, int(e.st_mtime))
    return out


def local_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_sha256(client, r, remote_dir, names):
    """{name: sha256} of files in remote_dir, computed on Mufasa."""
    if not names:
        return {}
    _, out, _ = run_cmd(client, r, f"cd {shlex.quote(remote_dir)} && sha256sum -- "
                        + " ".join(shlex.quote(n) for n in names))
    res = {}
    for line in out.splitlines():          # the login shell may print other lines
        h, _, name = line.strip().partition("  ")
        if len(h) == 64 and name.lstrip("*") in names:
            res[name.lstrip("*")] = h
    return res


def sync_dir(sftp, local_dir, remote_dir, only=None, mirror=False, dry_run=False, hasher=None):
    """Make remote_dir hold this PC's files, uploading only what is needed.

    Same size and mtime = already there (like rsync). Same size but another
    mtime (e.g. copied on Mufasa with plain `cp` / `scp`) = compared by SHA-256
    through `hasher(remote_dir, names)`; identical files are not re-sent and get
    the local mtime, so the next check is instant. With `mirror`, remote files
    that are not on this PC are deleted. Returns a dict of relative paths:
    missing / changed (uploaded), same, by_checksum (subset of same), removed."""
    loc = local_files(local_dir, only)
    rem = remote_files(sftp, remote_dir)
    if only is not None:
        rem = {n: v for n, v in rem.items() if n in loc}
    missing = [n for n in loc if n not in rem]
    changed = [n for n in loc if n in rem and rem[n][0] != loc[n][1]]
    check = [n for n in loc if n in rem and rem[n][0] == loc[n][1] and rem[n][1] != loc[n][2]]
    by_checksum = []
    if check and hasher is not None:
        rh = hasher(remote_dir, check)
        for n in check:
            (by_checksum if rh.get(n) == local_sha256(loc[n][0]) else changed).append(n)
    else:
        changed += check
    todo = missing + changed
    extra = sorted(set(rem) - set(loc)) if mirror else []
    if not dry_run:
        made = set()
        for n in todo:
            full, _, mtime = loc[n]
            rpath = posixpath.join(remote_dir, n)
            rdir = posixpath.dirname(rpath)
            if rdir not in made:
                sftp_makedirs(sftp, rdir)
                made.add(rdir)
            sftp.put(full, rpath)               # paramiko checks the size after the put
            sftp.utime(rpath, (mtime, mtime))
        for n in by_checksum:
            mtime = loc[n][2]
            sftp.utime(posixpath.join(remote_dir, n), (mtime, mtime))
        for n in extra:
            sftp.remove(posixpath.join(remote_dir, n))
    return {"missing": missing, "changed": changed, "by_checksum": by_checksum,
            "same": [n for n in loc if n not in todo], "removed": extra}


def pull_selected(sftp, remote_base, local_base, patterns):
    """Download the files of remote_base matching `patterns` ("dir/*.ext",
    relative to remote_base) into the same place under local_base."""
    n = 0
    for pat in patterns:
        rdir, glob = posixpath.split(pat)
        for name in sorted(r_listdir(sftp, posixpath.join(remote_base, rdir))):
            if not fnmatch.fnmatch(name, glob):
                continue
            ldir = os.path.join(local_base, *rdir.split("/")) if rdir else local_base
            os.makedirs(ldir, exist_ok=True)
            sftp.get(posixpath.join(remote_base, rdir, name), os.path.join(ldir, name))
            print(f"  <- {posixpath.join(rdir, name)}")
            n += 1
    return n


def job_state(client, r, jobid):
    _, out, _ = run_cmd(client, r, f"squeue -j {jobid} -h -o %T")
    if out:
        return out.splitlines()[0].strip()
    _, out, _ = run_cmd(client, r, f"sacct -j {jobid} -n -X -P --format=State")
    return out.splitlines()[0].strip().split()[0] if out else "UNKNOWN"


# --------------------------------------------------------------------------- #
# Plan, run config, job script
# --------------------------------------------------------------------------- #
def plan_run(sftp, full, sample, paths, stages, date):
    """What the job runs and what to sync for it.
    Returns (stages, [(label, local_dir, remote_dir, only_these_files|None, mirror)])."""
    stages = [s for s in COMPUTE_STAGES if s in stages]   # acquisition never runs there
    if not stages:
        raise SystemExit("[remote] nothing to run on Mufasa: stages must include "
                         "vision, registration and/or validation")
    syncs = []
    r_vis = posixpath.join(paths["r_data"], "input_vision", sample)
    r_reg = posixpath.join(paths["r_data"], "input_registration", sample)
    if "vision" in stages:
        if os.path.isdir(paths["vision"]):
            syncs.append(("input_vision", paths["vision"], r_vis, None, True))
        elif r_listdir(sftp, r_vis):
            print(f"[remote] {paths['vision']} not on this PC: using the copy on Mufasa ({r_vis})")
        else:
            raise SystemExit(f"[remote] input_vision/{sample} is neither on this PC nor on "
                             "Mufasa: nothing to reconstruct")
    if "registration" in stages:
        if (full.get("registration") or {}).get("mode") == "batch":
            raise SystemExit("[remote] registration.mode=batch scans every sample and is not "
                             "submitted from here: use single/roi/sweep, or mufasa.enabled: false")
        need = registration_inputs(full, sample)
        if all(os.path.isfile(os.path.join(paths["registration"], n)) for n in need):
            syncs.append(("input_registration", paths["registration"], r_reg, None, False))
        elif all(r_isfile(sftp, posixpath.join(r_reg, n)) for n in need):
            print(f"[remote] {need} not on this PC: using the copy on Mufasa ({r_reg})")
        elif "vision" in stages:
            print(f"[remote] {need} neither on this PC nor on Mufasa: running VISION only "
                  "(registration needs the HSI cube)")
            stages.remove("registration")
        else:
            raise SystemExit(f"[remote] registration needs input_registration/{sample}/{need}, "
                             "neither on this PC nor on Mufasa")
    if "validation" in stages and "registration" not in stages:
        # The registration validation (phase 2) reads the HSI cube: sent when it is
        # here; without it on Mufasa that modality is skipped, not an error.
        need = registration_inputs(full, sample)
        if all(os.path.isfile(os.path.join(paths["registration"], n)) for n in need):
            syncs.append(("input_registration", paths["registration"], r_reg, None, False))
    # DATA/validation: the marker table + STL of a validation job, and the
    # table that vision reads with vision.aruco.scale_from 'centers'.
    centers = ((full.get("vision") or {}).get("aruco") or {}).get("scale_from") == "centers"
    if "validation" in stages or ("vision" in stages and centers):
        syncs += validation_sync(sftp, paths)
    if ("registration" in stages or "validation" in stages) and "vision" not in stages:
        # The job takes `date`'s vision run, else the most recent one (pipeline
        # rule): send this PC's run only when it is the one that must be used.
        loc = vision_runs(paths["results_root"], sample, os.path.isfile,
                          lambda d: os.listdir(d) if os.path.isdir(d) else [], os.path.join)
        rem = vision_runs(paths["r_results"], sample, lambda p: r_isfile(sftp, p),
                          lambda d: r_listdir(sftp, d), posixpath.join)
        l_day = date if date in loc else max(loc, key=_day, default=None)
        r_day = date if date in rem else max(rem, key=_day, default=None)
        if l_day and (l_day == date or not r_day or (r_day != date and _day(l_day) > _day(r_day))):
            syncs.append(("vision run", loc[l_day],
                          posixpath.join(paths["r_results"], l_day, sample, "vision"),
                          vision_files(full, stages), False))
        elif r_day:
            print(f"[remote] {' + '.join(s for s in stages if s != 'vision')} uses the "
                  f"vision run on Mufasa: {rem[r_day]}")
        else:
            raise SystemExit(f"[remote] no vision run for {sample} on this PC or on Mufasa "
                             f"(needs {list(VISION_REQUIRED)}): run `spectra vision` first")
    return stages, syncs


def validation_sync(sftp, paths):
    """DATA/validation (marker table + STL) for a validation job."""
    r_val = posixpath.join(paths["r_data"], "validation")
    if os.path.isdir(paths["validation"]):
        return [("validation data", paths["validation"], r_val, None, False)]
    if r_listdir(sftp, r_val):
        print(f"[remote] {paths['validation']} not on this PC: using the copy on Mufasa ({r_val})")
        return []
    raise SystemExit(f"[remote] {paths['validation']} (marker table + STL) is neither "
                     "on this PC nor on Mufasa")


def pull_validation_all(sftp, r_results, local_results, date):
    """After `spectra validate --all`: every case's metrics.json of `date` and
    the summary (RESULTS/_validation/). Error maps / .rrd stay on Mufasa."""
    n = 0
    r_day = posixpath.join(r_results, date)
    for sample in sorted(r_listdir(sftp, r_day)):
        for modality in ("vision", "registration"):
            src = posixpath.join(r_day, sample, "validation", modality, "metrics.json")
            if r_isfile(sftp, src):
                dst = os.path.join(local_results, date, sample, "validation", modality)
                os.makedirs(dst, exist_ok=True)
                sftp.get(src, os.path.join(dst, "metrics.json"))
                n += 1
    n += pull_selected(sftp, r_results, local_results, ["_validation/*"])
    return n


def render_run_config(full, sample, paths, stages, marker_mm, date):
    """Local config (+ CLI overrides) with only the run-specific fields changed."""
    cfg = copy.deepcopy(full)
    for section in ("acquisition", "mufasa", "calibration"):   # robot PC only
        cfg.pop(section, None)
    # This file is the run's default config on Mufasa (SPECTRA_DEFAULT_CONFIG),
    # so it carries the marker edge used locally (from configs/default.yaml).
    cfg.setdefault("vision", {}).setdefault("aruco", {})["marker_edge_length_mm"] = marker_mm
    ((cfg.get("registration") or {}).get("detection") or {}).pop("marker_edge_length_mm", None)
    cfg["sample_name"] = sample
    cfg["data_root"] = paths["r_data"]
    cfg["results_root"] = paths["r_results"]
    cfg["results_date"] = date        # pinned: a job that starts after midnight stays here
    cfg["stages"] = list(stages)
    return yaml.safe_dump(cfg, sort_keys=False)


def build_job_script(r, job_dir, run_config, extra_args=(), command="full"):
    extra = "".join(f" {shlex.quote(a)}" for a in extra_args)
    return f"""#!/bin/bash
#SBATCH --job-name={r['job_name']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition={r['partition']}
#SBATCH --qos={r['qos']}
#SBATCH --gres={r['gres']}
#SBATCH --cpus-per-task={r['cpus']}
#SBATCH --mem={r['mem']}
#SBATCH --time={r['time']}
#SBATCH --output={job_dir}/slurm-%j.out
#SBATCH --error={job_dir}/slurm-%j.out

set -euo pipefail
echo "Host: $(hostname)   SLURM_JOB_ID: $SLURM_JOB_ID   Start: $(date)"
cd {shlex.quote(r['repo_dir'])}

export UV_CACHE_DIR="${{HOME}}/.cache/uv"
export PIP_CACHE_DIR="${{HOME}}/.cache/pip"
export HF_HOME="${{HOME}}/.cache/huggingface"      # model weights: persistent cache
export TORCH_HOME="${{HOME}}/.cache/torch"
export TMPDIR="${{HOME}}/.cache/tmp/spectra_${{SLURM_JOB_ID}}"   # per-job scratch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${{UV_CACHE_DIR}}" "${{PIP_CACHE_DIR}}" "${{HF_HOME}}" "${{TORCH_HOME}}" "${{TMPDIR}}"
trap 'rm -rf "${{TMPDIR}}"' EXIT
export PATH="${{HOME}}/.local/bin:${{PATH}}"
# This run's config is its default config (marker edge included): the repo's
# configs/default.yaml on Mufasa is not used.
export SPECTRA_DEFAULT_CONFIG={shlex.quote(run_config)}

uv run --extra processing spectra {command} --config {shlex.quote(run_config)}{extra}
echo "Finished: $(date)"
"""


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def preflight(config_path=DEFAULT_CONFIG, overrides=None):
    """Checks settings, SSH login and mufasa.repo_dir without uploading
    anything: `spectra full` calls it BEFORE moving the robot."""
    _, r, _, _ = load_settings(config_path, overrides)
    client = connect(r)
    try:
        check_remote_repo(client.open_sftp(), r)
    finally:
        client.close()
    print(f"[remote] Mufasa reachable, repo found in {r['repo_dir']}")


def submit(config_path=DEFAULT_CONFIG, *, stages=None, overrides=None,
           wait=None, pull=None, dry_run=False, extra_args=(), validate_all=False,
           new_timing=True):
    """Syncs what `stages` need, submits the SLURM job and (optionally) waits
    for it. The results are downloaded by a DETACHED process (`--watch`), which
    keeps going if this terminal is closed (mufasa.detach_pull; without an SSH
    key it cannot log in alone: then they are downloaded here, as before).
    Returns (jobid, final_state).
    ``validate_all``: one job running `spectra validate --all` on every case
    whose vision run is on Mufasa (only DATA/validation is synced).
    ``new_timing``: start a new timing record of the sample (False: `spectra
    full` already started it with the acquisition)."""
    t_up0 = time.time()
    full, r, sample, paths = load_settings(config_path, overrides)
    if stages is None:
        stages = full.get("stages") or list(COMPUTE_STAGES)
    check_portable(full)
    local_mm = marker_edge_mm(config_path, full)
    wait = r["wait"] if wait is None else wait
    pull = r["pull_results"] if pull is None else pull
    date = full.get("results_date") or time.strftime(DATE_FORMAT)
    r_sample = posixpath.join(paths["r_results"], date, "_validation" if validate_all else sample)
    job_dir = posixpath.join(r_sample, "_job")
    local_job = os.path.join(paths["results_root"], date,
                             "_validation" if validate_all else sample, "_job")
    tm = _timing()
    tfile = None if (validate_all or dry_run) else tm.timing_dir(
        paths["results_root"], date, sample) / tm.LOCAL
    if tfile is not None and new_timing:
        tm.reset(tfile, sample=sample, date=date, stages=list(stages))
    up_subs, t_lap = [], time.time()

    def lap(name):
        nonlocal t_lap
        now = time.time()
        up_subs.append({"name": name, "seconds": now - t_lap})
        t_lap = now

    client = connect(r)
    try:
        sftp = client.open_sftp()
        check_remote_repo(sftp, r)                        # before uploading anything
        lap("connect + checks")
        if validate_all:
            stages, syncs = ["validation"], validation_sync(sftp, paths)
            print(f"[remote] validate --all on Mufasa (every case with a vision run there) -> "
                  f"{paths['r_results']}/{date}/<sample>/validation/")
        else:
            stages, syncs = plan_run(sftp, full, sample, paths, list(stages), date)
            print(f"[remote] sample {sample}: {' + '.join(stages)} on Mufasa -> {r_sample}")
        for label, ldir, rdir, only, mirror in syncs:
            s = sync_dir(sftp, ldir, rdir, only, mirror, dry_run,
                         hasher=lambda d, names: remote_sha256(client, r, d, names))
            up, gone = s["missing"] + s["changed"], s["removed"]
            print(f"[remote] {label}: {'would upload' if dry_run else 'uploaded'} {len(up)}"
                  + (f" ({len(s['missing'])} missing, {len(s['changed'])} different"
                     f"{', e.g. ' + str(up[:3]) if up else ''})" if up else "")
                  + f", already on Mufasa {len(s['same'])}"
                  + (f" ({len(s['by_checksum'])} matched by checksum)" if s["by_checksum"] else "")
                  + (f", {'would remove' if dry_run else 'removed'} {len(gone)} stale "
                     f"{gone[:5]}{' ...' if len(gone) > 5 else ''}" if gone else "")
                  + f"  ({rdir})")
            lap(f"sync {label}: {len(up)} files")

        run_config_text = render_run_config(full, sample, paths, stages, local_mm, date)
        if dry_run:
            print("[remote][dry-run] nothing uploaded or submitted. run_config.yaml would be:\n"
                  + run_config_text)
            return None, "DRY-RUN"

        r_vis = posixpath.join(paths["r_data"], "input_vision", sample)
        if "vision" in stages and r_isfile(sftp, posixpath.join(r_vis, "SHA256SUMS")):
            code, out, err = run_cmd(client, r, f"cd {shlex.quote(r_vis)} && "
                                                "sha256sum -c --quiet SHA256SUMS")
            if code != 0:
                raise SystemExit(f"[remote] checksum mismatch on Mufasa ({r_vis}):\n{out or err}")
            print("[remote] input_vision verified on Mufasa (SHA256SUMS)")
        elif "vision" in stages:
            print("[remote] warning: no SHA256SUMS, images/poses not checksum-verified")
        lap("checksum verification on Mufasa")

        sftp_makedirs(sftp, job_dir)
        run_config = posixpath.join(job_dir, "run_config.yaml")
        with sftp.open(run_config, "w") as fh:
            fh.write(run_config_text)
        job = posixpath.join(job_dir, "job.slurm")
        # RESULTS/ on Mufasa is shared between jobs: vision must not be skipped
        # because an earlier job already wrote this date's vision run
        args = list(extra_args) + (["--force-vision"] if "vision" in stages else [])
        with sftp.open(job, "w") as fh:
            fh.write(build_job_script(r, job_dir, run_config,
                                      ["--all"] if validate_all else args,
                                      command="validate" if validate_all else "full"))

        code, out, err = run_cmd(client, r, f"cd {shlex.quote(job_dir)} && sbatch job.slurm")
        jobid = out.split()[-1] if code == 0 and out else ""
        if not jobid.isdigit():
            raise SystemExit(f"[remote] sbatch failed:\n{err or out}")
        print(f"[remote] submitted job {jobid} (qos {r['qos']}). "
              f"Log on Mufasa: {job_dir}/slurm-{jobid}.out")
        lap("job script + sbatch")
        if tfile is not None:
            tm.add(tfile, "upload to Mufasa + submit", t_up0, time.time(), where="this PC",
                   subphases=up_subs)
        spec = {"config": os.path.abspath(config_path), "overrides": overrides or {},
                "cwd": os.getcwd(), "jobid": jobid, "stages": list(stages), "date": date,
                "sample": sample, "validate_all": validate_all, "r_sample": r_sample,
                "job_dir": job_dir, "local_job": local_job, "pull": bool(pull),
                "timing": str(tfile) if tfile is not None else None}
        watcher = spawn_watcher(spec, r) if pull else None
        if not wait:
            if watcher:
                print(f"[remote] the results will be downloaded in the background when the "
                      f"job ends (log: {watcher['log']})")
            return jobid, "SUBMITTED"

        print(f"[remote] waiting for job {jobid} (poll {r['poll_s']}s; Ctrl-C or closing "
              "the terminal stops waiting here, the job keeps running on Mufasa"
              + (" and the results are still downloaded)" if watcher else ")"))
        state = wait_job(client, r, jobid, verbose=True)
        _, tail, _ = run_cmd(client, r, f"tail -n 30 {job_dir}/slurm-{jobid}.out 2>/dev/null")
        if tail:
            print("--- last lines of the job log ---\n" + tail + "\n---")
        print(f"[remote] results on Mufasa: {r_sample}")
        if not pull:
            return jobid, state
        if watcher:
            print(f"[remote] download in the background (log: {watcher['log']}) ...")
            done = wait_marker(watcher["done"], timeout_s=3600)
            if done:
                print(f"[remote] downloaded {done.get('n_files', 0)} files into "
                      f"{done.get('local_dir')}" + (f" ({done['error']})" if done.get("error") else ""))
            else:
                print(f"[remote] the download is still running in the background: {watcher['log']}")
        else:
            finish(client, sftp, r, spec, state)
        return jobid, state
    finally:
        client.close()


def wait_job(client, r, jobid, verbose=False):
    """Polls the job until it ends; returns its final state."""
    last, state, unknown = None, "UNKNOWN", 0
    while True:
        state = job_state(client, r, jobid)
        if verbose and state != last:
            print(f"  [{time.strftime('%H:%M:%S')}] {state}", flush=True)
            last = state
        # UNKNOWN right after sbatch can be a short accounting delay
        unknown = unknown + 1 if state == "UNKNOWN" else 0
        if state in TERMINAL_STATES or unknown >= 3:
            break
        time.sleep(int(r["poll_s"]))
    if state == "UNKNOWN":
        print(f"[remote] warning: cannot read the state of job {jobid} "
              "(check mufasa.slurm_bin_dirs); it may still be running")
    return state


def job_times(client, r, jobid):
    """(submit, start, end) of the job as epoch seconds (sacct), or None."""
    _, out, _ = run_cmd(client, r, f"sacct -j {jobid} -n -X -P --format=Submit,Start,End")
    try:
        vals = out.splitlines()[0].strip().split("|")
        return tuple(_dt.datetime.strptime(v, "%Y-%m-%dT%H:%M:%S").timestamp() for v in vals[:3])
    except Exception:
        return None


def finish(client, sftp, r, spec, state):
    """After the job: SLURM queue / job times and the download into the timing
    record, the download itself, timing.xlsx. Returns the number of files."""
    tm = _timing()
    paths = load_settings(spec["config"], spec["overrides"])[3]
    tfile = spec.get("timing")
    times = job_times(client, r, spec["jobid"]) if tfile and not spec.get("_timed") else None
    if times:
        sub, start, end = times
        tm.add(tfile, "SLURM queue (pending)", sub, start, where="Mufasa")
        tm.add(tfile, "SLURM job (on Mufasa)", start, end, where="Mufasa", note=state)
        spec["_timed"] = True             # a retry of the download does not add them again
    t0 = time.time()
    date, sample, stages = spec["date"], spec["sample"], spec["stages"]
    if spec["validate_all"]:
        n = pull_validation_all(sftp, paths["r_results"], paths["results_root"], date)
        print(f"[remote] downloaded {n} files (every case's metrics.json of {date} + "
              f"RESULTS/_validation/) into {paths['results_root']}")
        local = paths["results_root"]
    else:
        # only the stages this job ran (older files of other stages stay
        # there) + the timing of the run
        patterns = [p for p in r["pull_patterns"]
                    if p.split("/")[0] in stages or "/" not in p or p in ALWAYS_PULL]
        patterns += [p for p in ALWAYS_PULL if p not in patterns]
        local = os.path.join(paths["results_root"], date, sample)
        n = pull_selected(sftp, spec["r_sample"], local, patterns)
        print(f"[remote] downloaded {n} files into {local} "
              f"(mufasa.pull_patterns of {' + '.join(stages)}: {patterns})")
    if tfile:
        tm.add(tfile, "download to this PC", t0, time.time(), where="this PC",
               subphases=[{"name": f"{n} files", "seconds": time.time() - t0}])
        try:
            tm.write_xlsx(os.path.dirname(tfile), sample=sample, date=date)
        except Exception as exc:
            print(f"[timing] timing.xlsx not written: {exc}")
    return n, local


# --------------------------------------------------------------------------- #
# Download in the background (survives the terminal being closed)
# --------------------------------------------------------------------------- #
def spawn_watcher(spec, r):
    """Starts `remote.py --watch <spec>` detached from this terminal: it waits
    for the job, downloads the results and writes the timing. Returns
    {log, done} paths, or None (then the caller downloads itself)."""
    if not r.get("detach_pull", True):
        return None
    if not r.get("ssh_key_path"):
        print("[remote] no mufasa.ssh_key_path: the download cannot run in the background "
              "(it needs the password), it is done here at the end of the job")
        return None
    os.makedirs(spec["local_job"], exist_ok=True)
    base = os.path.join(spec["local_job"], f"pull_{spec['jobid']}")
    spec_path, log, done = base + ".json", base + ".log", base + ".done"
    for p in (done,):
        if os.path.exists(p):
            os.remove(p)
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, default=str)
    cmd = [sys.executable, os.path.abspath(__file__), "--watch", spec_path]
    logf = open(log, "a", encoding="utf-8")
    kw = dict(cwd=spec["cwd"], stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
              close_fds=True)
    try:
        if os.name == "nt":
            # no console (closing the terminal does not kill it), own process
            # group, and out of the job object of `uv run` (which kills its
            # children when it exits) when that job allows it.
            flags = 0x00000008 | 0x00000200            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            try:
                subprocess.Popen(cmd, creationflags=flags | 0x01000000, **kw)  # | BREAKAWAY_FROM_JOB
            except OSError:
                subprocess.Popen(cmd, creationflags=flags, **kw)
        else:
            subprocess.Popen(cmd, start_new_session=True, **kw)
    except Exception as exc:
        print(f"[remote] background download not started ({exc}): it is done here")
        return None
    finally:
        logf.close()
    print(f"[remote] background download started for job {spec['jobid']} (log: {log})")
    return {"log": log, "done": done}


def wait_marker(path, timeout_s=3600, poll_s=5):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        time.sleep(poll_s)
    return None


def watch(spec_path):
    """Body of the detached process: wait for the job, download, timing.
    Reconnects if the connection drops (VPN, sleep) while waiting."""
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    os.chdir(spec["cwd"])
    print(f"[watch] {time.strftime('%Y-%m-%d %H:%M:%S')} job {spec['jobid']} ({spec['sample']})",
          flush=True)
    r = load_settings(spec["config"], spec["overrides"])[1]
    done = os.path.join(spec["local_job"], f"pull_{spec['jobid']}.done")
    result = {"jobid": spec["jobid"], "n_files": 0, "local_dir": None}
    state, fails = "UNKNOWN", 0
    while True:
        try:
            client = connect(r)
            try:
                state = wait_job(client, r, spec["jobid"])
                print(f"[watch] {time.strftime('%H:%M:%S')} job ended: {state}", flush=True)
                sftp = client.open_sftp()
                result["n_files"], result["local_dir"] = finish(client, sftp, r, spec, state)
            finally:
                client.close()
            break
        except (Exception, SystemExit) as exc:
            fails += 1
            print(f"[watch] {time.strftime('%H:%M:%S')} error ({exc}); retry {fails}/60 in 60 s",
                  flush=True)
            if fails >= 60:
                result["error"] = str(exc)
                break
            time.sleep(60)
    result["state"] = state
    with open(done, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[watch] {time.strftime('%Y-%m-%d %H:%M:%S')} done: {result}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Run vision / registration of a sample on "
                                             "Mufasa (SLURM); needs mufasa.enabled: true")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="unified pipeline config")
    ap.add_argument("--sample", default=None, help="override sample_name")
    ap.add_argument("--stages", nargs="+", choices=COMPUTE_STAGES, default=None,
                    help="stages to run (default: `stages` in the config)")
    ap.add_argument("--no-wait", action="store_true", help="submit and return, do not wait")
    ap.add_argument("--no-pull", action="store_true", help="do not download the results")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be uploaded and the run config, send nothing")
    ap.add_argument("--watch", default=None, metavar="SPEC_JSON",
                    help=argparse.SUPPRESS)       # the detached download (spawn_watcher)
    args = ap.parse_args()
    if args.watch:
        watch(args.watch)
        return
    submit(args.config, stages=args.stages,
           overrides={"sample_name": args.sample} if args.sample else None,
           wait=False if args.no_wait else None,
           pull=False if args.no_pull else None,
           dry_run=args.dry_run)


if __name__ == "__main__":
    main()
