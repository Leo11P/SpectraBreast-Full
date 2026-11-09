#!/usr/bin/env python3
# =============================================================================
#  cameras.py  —  the cameras the robot acquisition can mount on the flange
# -----------------------------------------------------------------------------
#  One camera at a time, chosen by `acquisition.camera.type` in default.yaml:
#    realsense : Intel RealSense D456 (pyrealsense2), colour + optional depth.
#    gc1020c   : Allied Vision Prosilica GC1020C, GigE Vision, through the
#                Vimba X SDK (vmbpy). Colour only, no factory intrinsics.
#  Both expose the same small interface, used by acquire.py and by the
#  calibration views (spectra/calibration/acquisition.py):
#    w, h                  image size in pixels
#    grab(n_avg)           mean of n_avg colour frames, BGR uint8 (None = no frame)
#    latest()              one fresh colour frame, BGR uint8 (live preview)
#    settle(seconds)       let the stream run while the robot settles
#    grab_depth()          depth aligned to colour, uint16 mm (None = no depth)
#    factory_intrinsics()  (K, D, model), or None if the camera has none
#    stop()
#  Imported "flat" like the other modules of this folder (no `spectra`
#  package); each camera SDK is imported only when that camera is opened.
# =============================================================================
import threading
import time

import numpy as np

CAMERA_TYPES = ("realsense", "gc1020c")
#: Python module each camera needs (checked by the CLI before the robot moves).
SDK_MODULE = {"realsense": "pyrealsense2", "gc1020c": "vmbpy"}


def resolve_camera_cfg(camera):
    """`acquisition.camera` -> flat dict of the selected camera, with 'type'.
    Keys at the top of `camera:` are shared (e.g. n_avg_frames); the block of
    the selected camera (`realsense:` / `gc1020c:`) overrides them. An older
    flat `camera:` block without `type` is the RealSense."""
    camera = dict(camera or {})
    kind = str(camera.get("type") or "realsense").strip().lower()
    if kind not in CAMERA_TYPES:
        raise SystemExit(f"[camera] acquisition.camera.type must be one of "
                         f"{list(CAMERA_TYPES)}, got {kind!r}")
    flat = {k: v for k, v in camera.items() if k not in CAMERA_TYPES}
    flat.update(camera.get(kind) or {})
    flat["type"] = kind
    missing = [k for k in ("width", "height") if flat.get(k) is None]
    if missing:
        raise SystemExit(f"[camera] acquisition.camera.{kind}: {missing} missing")
    return flat


def calibration_dir_name(kind):
    """Folder under data_root with the camera's own calibration (hand-eye +
    intrinsics, written by `spectra calibrate`). The RealSense keeps the
    original camera_calibration/."""
    return "camera_calibration" if kind == "realsense" else f"camera_calibration_{kind}"


def open_camera(cam_cfg, depth=None, rs_color_settings=None):
    """Opens the camera described by cam_cfg (output of resolve_camera_cfg).
    depth: None = cam_cfg['depth'] (RealSense only). rs_color_settings: the
    exposure / gain / white balance of `calibration.camera` (RealSense only)."""
    if cam_cfg["type"] == "realsense":
        return RealSenseCamera(cam_cfg, depth=depth, color_settings=rs_color_settings)
    return VimbaCamera(cam_cfg)


# -----------------------------------------------------------------------------
#  RealSense D456
# -----------------------------------------------------------------------------
class RealSenseCamera:
    """RealSense colour stream. If the device supports it, depth is ALSO
    enabled, aligned to colour (same intrinsics as colour -> consistent with
    intrinsics.npy): grab_depth() returns the depth map."""

    kind = "realsense"

    def __init__(self, cfg, depth=None, color_settings=None):
        import pyrealsense2 as rs
        self.rs = rs
        self.w = int(cfg["width"]); self.h = int(cfg["height"])
        fps = int(cfg.get("fps") or 15)
        self.timeout_ms = int(cfg.get("frame_timeout_ms") or 10000)
        # optional depth (default ON): it must NEVER prevent color acquisition.
        self.depth_enabled = bool(cfg.get("depth", True) if depth is None else depth)
        self.depth_scale = 1.0
        self.align = None

        def _start(with_depth):
            pipe = rs.pipeline()
            conf = rs.config()
            conf.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, fps)
            if with_depth:
                # no forced depth resolution: many cameras (e.g. D45x) do NOT
                # support depth at the color resolution (e.g. 1280x800). Let the
                # SDK pick a valid depth mode; alignment to color brings the
                # map back to the color resolution/intrinsics anyway.
                conf.enable_stream(rs.stream.depth)
            prof = pipe.start(conf)
            return pipe, prof

        self.pipe = None
        if self.depth_enabled:
            try:
                self.pipe, self.profile = _start(True)
                self.depth_scale = float(self.profile.get_device()
                                         .first_depth_sensor().get_depth_scale())
                self.align = rs.align(rs.stream.color)
                print(f"[realsense] depth ON (scale {self.depth_scale:.6f} m/unit, "
                      "aligned to color)")
            except Exception as e:
                print(f"[realsense] warning: depth unavailable ({e}); color only")
                self.depth_enabled = False
                if self.pipe is not None:
                    try: self.pipe.stop()
                    except Exception: pass
                self.pipe = None
        if self.pipe is None:
            self.pipe, self.profile = _start(False)
        if color_settings:
            self._apply_color_settings(color_settings)
        # warm-up (exposure settling)
        for _ in range(30):
            self.pipe.wait_for_frames(self.timeout_ms)

    def _apply_color_settings(self, C):
        """Colour sensor: manual exposure/gain keep the chessboard corners stable
        between frames. Unsupported options are reported and skipped."""
        rs = self.rs
        try:
            sensor = self.profile.get_device().first_color_sensor()
        except Exception as e:
            print(f"  [rs] colour sensor not accessible: {e}")
            return

        def _set(opt, val, name):
            try:
                if sensor.supports(opt):
                    sensor.set_option(opt, float(val))
                    print(f"  [rs] {name} = {val}")
                else:
                    print(f"  [rs] {name}: not supported by the sensor")
            except Exception as e:
                print(f"  [rs] cannot set {name}: {e}")

        _set(rs.option.enable_auto_exposure, 1 if C["auto_exposure"] else 0, "auto_exposure")
        if not C["auto_exposure"]:
            if C["exposure"] is not None:
                _set(rs.option.exposure, C["exposure"], "exposure")
            if C["gain"] is not None:
                _set(rs.option.gain, C["gain"], "gain")
        # Turning auto white balance off WITHOUT a manual value leaves a residual
        # value and a colour cast on the whole image: set white_balance too.
        _set(rs.option.enable_auto_white_balance, 1 if C["auto_white_balance"] else 0,
             "auto_white_balance")
        if not C["auto_white_balance"] and C["white_balance"] is not None:
            _set(rs.option.white_balance, C["white_balance"], "white_balance")

    def grab(self, n_avg=5):
        acc = None; got = 0
        for _ in range(max(1, int(n_avg))):
            try:
                frames = self.pipe.wait_for_frames(self.timeout_ms)
            except RuntimeError:
                continue
            c = frames.get_color_frame()
            if not c:
                continue
            img = np.asanyarray(c.get_data()).astype(np.float32)
            acc = img if acc is None else acc + img
            got += 1
        if got == 0:
            return None
        return (acc / got).round().astype(np.uint8)

    def latest(self):
        try:
            c = self.pipe.wait_for_frames(self.timeout_ms).get_color_frame()
        except RuntimeError:
            return None
        return np.asanyarray(c.get_data()) if c else None

    def settle(self, seconds):
        """Consumes frames for `seconds` (robot settling), so the next grab
        does not read frames queued while the robot was moving."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            try: self.pipe.wait_for_frames(self.timeout_ms)
            except Exception: pass

    def grab_depth(self):
        """Depth map co-registered with color (same resolution/intrinsics).
        Returns a uint16 in MILLIMETRES (0 = no measurement/hole) or None if
        depth is unavailable. Single frameset: at shot time the robot is
        still, so no averaging (averaging depth with the zeros of the holes
        would corrupt it)."""
        if not self.depth_enabled or self.align is None:
            return None
        try:
            frames = self.pipe.wait_for_frames(self.timeout_ms)
        except RuntimeError:
            return None
        try:
            aligned = self.align.process(frames)
            d = aligned.get_depth_frame()
        except Exception:
            return None
        if not d:
            return None
        raw = np.asanyarray(d.get_data())                 # uint16, native units
        depth_mm = raw.astype(np.float32) * self.depth_scale * 1000.0
        return np.clip(depth_mm, 0, 65535).astype(np.uint16)

    def factory_intrinsics(self):
        """Factory intrinsics of the COLOR stream actually running (same
        resolution as the images). Returns (K 3x3, D (5,), model name)."""
        vsp = self.profile.get_stream(self.rs.stream.color).as_video_stream_profile()
        i = vsp.get_intrinsics()
        K = np.array([[i.fx, 0.0, i.ppx], [0.0, i.fy, i.ppy], [0.0, 0.0, 1.0]], float)
        D = np.asarray(list(i.coeffs)[:5], float)
        return K, D, str(i.model)

    def stop(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


# -----------------------------------------------------------------------------
#  Prosilica GC1020C (Allied Vision, GigE Vision) through Vimba X
# -----------------------------------------------------------------------------
class VimbaCamera:
    """Prosilica GC1020C through Allied Vision Vimba X (vmbpy). The camera
    streams continuously into a callback that keeps the newest complete frame,
    converted to BGR8. grab() averages frames that arrive AFTER the call and
    skips the first of them (its exposure may have started while the robot was
    still moving). No depth and no factory intrinsics: K, D come from
    `spectra calibrate --modality intrinsics`.
    Needs Vimba X on the PC (GigE transport layer) and `vmbpy` in the venv.
    Only one program at a time can stream from the camera: close Vimba X
    Viewer / arv-viewer / gc1020c_capture.py first."""

    kind = "gc1020c"
    depth_enabled = False

    def __init__(self, cfg):
        try:
            import vmbpy
        except Exception as e:      # missing package, or VmbC libraries not loadable
            raise SystemExit(f"[gc1020c] cannot load vmbpy ({e}). On the robot PC: install "
                             "Vimba X (GigE transport layer) and `uv sync --extra acquisition`.") from e
        self._v = vmbpy
        self.timeout_s = float(cfg.get("frame_timeout_ms") or 10000) / 1000.0
        self._cond = threading.Condition()
        self._idx, self._img, self._err, self._n_bad = 0, None, None, 0
        self._cam = None
        self._vmb = vmbpy.VmbSystem.get_instance()
        try:
            self._vmb.__enter__()
        except (vmbpy.VmbSystemError, vmbpy.VmbTransportLayerError) as e:   # no GigE TL
            self._vmb = None
            raise SystemExit(f"[gc1020c] Vimba X did not start ({e}): install Vimba X on this "
                             "PC (it provides the GigE transport layer).") from None
        try:
            cam = self._find(cfg.get("id"))
            try:
                cam.__enter__()
            except vmbpy.VmbCameraError as e:
                raise SystemExit(f"[gc1020c] cannot open camera {cam.get_id()} ({e}). Is another "
                                 "program streaming from it, or is this PC's network adapter "
                                 "on another subnet than the camera?") from None
            self._cam = cam
            self._configure(cfg)
            self.w = int(cam.get_feature_by_name("Width").get())
            self.h = int(cam.get_feature_by_name("Height").get())
            if (self.w, self.h) != (int(cfg["width"]), int(cfg["height"])):
                raise SystemExit(f"[gc1020c] the camera streams {self.w}x{self.h} but "
                                 f"acquisition.camera.gc1020c is {cfg['width']}x{cfg['height']}: "
                                 "fix width/height (the intrinsics must match too).")
            cam.start_streaming(handler=self._on_frame, buffer_count=5)
            if self._next(0, max(self.timeout_s, 5.0))[0] is None:
                raise SystemExit(f"[gc1020c] no complete frame from the camera "
                                 f"({self._n_bad} incomplete). Incomplete frames usually mean "
                                 "packets too large for the link: set "
                                 "acquisition.camera.gc1020c.packet_size: 1500.")
        except BaseException:
            self.stop()
            raise
        print(f"[gc1020c] {cam.get_id()} streaming {self.w}x{self.h} ({self._fmt})")

    # ---------------------------------------------------------------- setup
    def _find(self, ident):
        cams = self._vmb.get_all_cameras()
        seen = ", ".join(f"{c.get_id()} ({c.get_model()})" for c in cams) or "none"
        if ident:
            try:
                return self._vmb.get_camera_by_id(str(ident))
            except self._v.VmbCameraError:
                raise SystemExit(f"[gc1020c] camera {ident!r} not found (Vimba X sees: {seen}). "
                                 "Use its IP or the ID shown by Vimba X Viewer.") from None
        if len(cams) == 1:
            return cams[0]
        if cams:
            raise SystemExit(f"[gc1020c] more than one camera ({seen}): set "
                             "acquisition.camera.gc1020c.id")
        raise SystemExit("[gc1020c] no camera found by Vimba X. Check the camera power, that "
                         "Vimba X (GigE transport layer) is installed, and that this PC's "
                         "network adapter is on the camera's subnet (e.g. camera "
                         "192.168.1.121 -> PC 192.168.1.50 / 255.255.255.0). Vimba X Viewer "
                         "must see the camera before the pipeline can.")

    def _set(self, label, candidates):
        """Sets the first feature of `candidates` [(name, value)] the camera has
        (feature names differ between firmware versions)."""
        for name, value in candidates:
            try:
                feat = self._cam.get_feature_by_name(name)
            except self._v.VmbFeatureError:
                continue
            try:
                feat.set(value)
                print(f"[gc1020c] {name} = {value}")
            except Exception as e:
                print(f"[gc1020c] warning: cannot set {name} = {value} ({e})")
            return
        print(f"[gc1020c] {label}: not supported by this camera")

    def _configure(self, cfg):
        v, cam = self._v, self._cam
        # GigE packet size: fixed (1500 = standard MTU, e.g. USB-Ethernet adapters)
        # or negotiated by Vimba X for this link.
        ps = cfg.get("packet_size")
        if ps:
            self._set("packet size", [("GevSCPSPacketSize", int(ps))])
        else:
            try:
                cmd = cam.get_streams()[0].get_feature_by_name("GVSPAdjustPacketSize")
                cmd.run()
                t0 = time.time()
                while not cmd.is_done() and time.time() - t0 < 5.0:
                    time.sleep(0.01)
                print(f"[gc1020c] packet size negotiated: "
                      f"{cam.get_feature_by_name('GevSCPSPacketSize').get()} B")
            except Exception as e:
                print(f"[gc1020c] packet size not negotiated ({e}): camera value kept")

        # pixel format: colour from the camera (BGR8 / RGB8) if available,
        # otherwise Bayer, debayered here by Vimba X into BGR8.
        fmts = cam.get_pixel_formats()
        want = cfg.get("pixel_format")
        if want:
            fmt = getattr(v.PixelFormat, str(want), None)
            if fmt not in fmts:
                raise SystemExit(f"[gc1020c] pixel_format {want!r} not available, the camera "
                                 f"offers {[str(f) for f in fmts]}")
        else:
            fmt = next((f for f in (v.PixelFormat.Bgr8, v.PixelFormat.Rgb8) if f in fmts), None)
            if fmt is None:
                colour = v.intersect_pixel_formats(fmts, v.COLOR_PIXEL_FORMATS)
                if not colour:
                    raise SystemExit(f"[gc1020c] no colour pixel format "
                                     f"({[str(f) for f in fmts]}): is it a mono camera?")
                fmt = colour[0]
        cam.set_pixel_format(fmt)
        self._fmt = str(fmt)

        # exposure / gain: null = automatic (continuous), a number = manual
        exp = cfg.get("exposure_us")
        if exp is None:
            self._set("auto exposure", [("ExposureAuto", "Continuous")])
        else:
            self._set("auto exposure", [("ExposureAuto", "Off")])
            self._set("exposure", [("ExposureTimeAbs", float(exp)), ("ExposureTime", float(exp))])
        gain = cfg.get("gain_db")
        if gain is None:
            self._set("auto gain", [("GainAuto", "Continuous")])
        else:
            self._set("auto gain", [("GainAuto", "Off")])
            self._set("gain", [("Gain", float(gain)), ("GainAbs", float(gain)),
                               ("GainRaw", int(round(float(gain))))])
        wb = "Continuous" if cfg.get("white_balance_auto", True) else "Off"
        self._set("auto white balance", [("BalanceWhiteAuto", wb)])

    # -------------------------------------------------------------- frames
    def _on_frame(self, cam, stream, frame):
        """vmbpy callback (Vimba thread): keeps the newest complete frame (BGR8)."""
        try:
            if frame.get_status() == self._v.FrameStatus.Complete:
                f = frame
                if f.get_pixel_format() != self._v.PixelFormat.Bgr8:
                    f = f.convert_pixel_format(self._v.PixelFormat.Bgr8)
                img = f.as_numpy_ndarray().copy()      # the buffer goes back to the camera
                with self._cond:
                    self._idx += 1
                    self._img = img
                    self._cond.notify_all()
            else:
                self._n_bad += 1
        except Exception as e:
            with self._cond:
                self._err = e
                self._cond.notify_all()
        finally:
            cam.queue_frame(frame)

    def _next(self, after, timeout=None):
        """First frame newer than index `after`: (img, index), else (None, after)."""
        with self._cond:
            self._cond.wait_for(lambda: self._idx > after or self._err is not None,
                                self.timeout_s if timeout is None else timeout)
            if self._err is not None:
                print(f"[gc1020c] frame error: {self._err}")
                self._err = None
                return None, after
            if self._idx <= after:
                return None, after
            return self._img, self._idx

    def grab(self, n_avg=5):
        with self._cond:
            last = self._idx
        img, last = self._next(last)        # skipped: exposed (partly) before the call
        if img is None:
            return None
        acc, got = None, 0
        for _ in range(max(1, int(n_avg))):
            img, last = self._next(last)
            if img is None:
                break
            acc = img.astype(np.float32) if acc is None else acc + img
            got += 1
        if got == 0:
            return None
        return (acc / got).round().astype(np.uint8)

    def latest(self):
        with self._cond:
            last = self._idx
        return self._next(last)[0]

    def settle(self, seconds):
        time.sleep(max(0.0, float(seconds)))   # grab() only uses frames newer than its call

    def grab_depth(self):
        return None

    def factory_intrinsics(self):
        return None

    def stop(self):
        cam, self._cam = self._cam, None
        if cam is not None:
            for fn in (cam.stop_streaming, lambda: cam.__exit__(None, None, None)):
                try: fn()
                except Exception: pass
        vmb, self._vmb = getattr(self, "_vmb", None), None
        if vmb is not None:
            try: vmb.__exit__(None, None, None)
            except Exception: pass
