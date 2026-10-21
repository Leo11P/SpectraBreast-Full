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
#    5. optionally wait, then download only mufasa.pull_patterns (default: the
#       registration Excel reports + the spectral Rerun recording) into the
#       local RESULTS/<date>/<sample>/. Everything else stays on Mufasa.
#
#  Requires: paramiko, pyyaml   (pip install paramiko pyyaml)
# =============================================================================
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import fnmatch
import getpass
import os
import posixpath
import shlex
import stat
import time

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "default.yaml")

COMPUTE_STAGES = ("vision", "registration")      # the only stages run on Mufasa
DATE_FORMAT = "%d-%m-%Y"                         # RESULTS/<dd-mm-yyyy>/
# Vision run files registration reads (first two required, as in the pipeline).
VISION_REQUIRED = ("surface_mesh.ply", "aruco_markers_3d.json")
VISION_HANDOFF = VISION_REQUIRED + ("timings.json", "reconstruction_metadata.json")

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
    "wait": True, "poll_s": 20, "pull_results": True,
    "pull_patterns": ["registration/*.xlsx", "registration/*.rrd", "registration/*.rbl"],
}


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
        raise SystemExit("[remote] missing dependency: pip install paramiko")
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
                         "(Polimi LAN/VPN on? key authorised on Mufasa?)")
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
        raise SystemExit(f"[remote] {path} not found on Mufasa: check mufasa.repo_dir")


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


def sync_dir(sftp, local_dir, remote_dir, only=None, mirror=False, dry_run=False):
    """Upload what is missing or changed (size or mtime) in remote_dir; with
    `mirror`, also delete remote files that are not on this PC. The remote
    mtime is set to the local one, so an unchanged file is never re-sent.
    Returns (uploaded, unchanged, removed) lists of relative paths."""
    loc = local_files(local_dir, only)
    rem = remote_files(sftp, remote_dir) if only is None else {
        n: v for n, v in remote_files(sftp, remote_dir).items() if n in loc}
    todo = [n for n, (_, size, mtime) in loc.items() if rem.get(n) != (size, mtime)]
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
        for n in extra:
            sftp.remove(posixpath.join(remote_dir, n))
    return todo, [n for n in loc if n not in todo], extra


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
                         "vision and/or registration")
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
    if "registration" in stages and "vision" not in stages:
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
                          VISION_HANDOFF, False))
        elif r_day:
            print(f"[remote] registration uses the vision run on Mufasa: {rem[r_day]}")
        else:
            raise SystemExit(f"[remote] no vision run for {sample} on this PC or on Mufasa "
                             f"(needs {list(VISION_REQUIRED)}): run `spectra vision` first")
    return stages, syncs


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


def build_job_script(r, job_dir, run_config, extra_args=()):
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

uv run --extra processing spectra full --config {shlex.quote(run_config)}{extra}
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
           wait=None, pull=None, dry_run=False, extra_args=()):
    """Syncs what `stages` need, submits the SLURM job and (optionally) waits
    for it and downloads the selected results. Returns (jobid, final_state)."""
    full, r, sample, paths = load_settings(config_path, overrides)
    if stages is None:
        stages = full.get("stages") or list(COMPUTE_STAGES)
    check_portable(full)
    local_mm = marker_edge_mm(config_path, full)
    wait = r["wait"] if wait is None else wait
    pull = r["pull_results"] if pull is None else pull
    date = full.get("results_date") or time.strftime(DATE_FORMAT)
    r_sample = posixpath.join(paths["r_results"], date, sample)
    job_dir = posixpath.join(r_sample, "_job")

    client = connect(r)
    try:
        sftp = client.open_sftp()
        check_remote_repo(sftp, r)                        # before uploading anything
        stages, syncs = plan_run(sftp, full, sample, paths, list(stages), date)
        print(f"[remote] sample {sample}: {' + '.join(stages)} on Mufasa -> {r_sample}")
        for label, ldir, rdir, only, mirror in syncs:
            up, same, gone = sync_dir(sftp, ldir, rdir, only, mirror, dry_run)
            verb = "would upload" if dry_run else "uploaded"
            print(f"[remote] {label}: {verb} {len(up)}, already on Mufasa {len(same)}"
                  + (f", {'would remove' if dry_run else 'removed'} {len(gone)} stale "
                     f"{gone[:5]}{' ...' if len(gone) > 5 else ''}" if gone else "")
                  + f"  ({rdir})")

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

        sftp_makedirs(sftp, job_dir)
        run_config = posixpath.join(job_dir, "run_config.yaml")
        with sftp.open(run_config, "w") as fh:
            fh.write(run_config_text)
        job = posixpath.join(job_dir, "job.slurm")
        # RESULTS/ on Mufasa is shared between jobs: vision must not be skipped
        # because an earlier job already wrote this date's vision run
        args = list(extra_args) + (["--force-vision"] if "vision" in stages else [])
        with sftp.open(job, "w") as fh:
            fh.write(build_job_script(r, job_dir, run_config, args))

        code, out, err = run_cmd(client, r, f"cd {shlex.quote(job_dir)} && sbatch job.slurm")
        jobid = out.split()[-1] if code == 0 and out else ""
        if not jobid.isdigit():
            raise SystemExit(f"[remote] sbatch failed:\n{err or out}")
        print(f"[remote] submitted job {jobid} (qos {r['qos']}). "
              f"Log on Mufasa: {job_dir}/slurm-{jobid}.out")
        if not wait:
            return jobid, "SUBMITTED"

        print(f"[remote] waiting for job {jobid} (poll {r['poll_s']}s; Ctrl-C stops "
              "waiting, the job keeps running on Mufasa)")
        last, state, unknown = None, "UNKNOWN", 0
        while True:
            state = job_state(client, r, jobid)
            if state != last:
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
        _, tail, _ = run_cmd(client, r, f"tail -n 30 {job_dir}/slurm-{jobid}.out 2>/dev/null")
        if tail:
            print("--- last lines of the job log ---\n" + tail + "\n---")
        print(f"[remote] results on Mufasa: {r_sample}")
        if pull:
            # only the stages this job ran: older files of other stages stay there
            patterns = [p for p in r["pull_patterns"]
                        if p.split("/")[0] in stages or "/" not in p]
            local_sample = os.path.join(paths["results_root"], date, sample)
            n = pull_selected(sftp, r_sample, local_sample, patterns)
            print(f"[remote] downloaded {n} files into {local_sample} "
                  f"(mufasa.pull_patterns of {' + '.join(stages)}: {patterns})")
        return jobid, state
    finally:
        client.close()


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
    args = ap.parse_args()
    submit(args.config, stages=args.stages,
           overrides={"sample_name": args.sample} if args.sample else None,
           wait=False if args.no_wait else None,
           pull=False if args.no_pull else None,
           dry_run=args.dry_run)


if __name__ == "__main__":
    main()
