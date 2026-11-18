#!/usr/bin/env python3
# =============================================================================
#  live_view.py  —  live camera window while the robot acquires
# -----------------------------------------------------------------------------
#  A background thread shows the newest frame of the camera (without taking it
#  away from the acquisition: VimbaCamera.peek() only reads) with the pose
#  being executed, the phase (moving / centring / settling / shot / saved),
#  the number of images saved and a focus value. Closing the window (q / Esc /
#  the X) only closes the viewer: the acquisition goes on. If the camera stops
#  streaming (cable pulled) the window shows NO SIGNAL until it comes back.
#  Only for cameras with a peek() method (the GC1020C).
#  Imported "flat" like the other modules of this folder.
# =============================================================================
import threading
import time

import cv2 as cv
import numpy as np


def focus_score(img):
    """Sharpness of the central half of the image: variance of the Laplacian
    (higher = sharper). Only comparable between frames of the SAME scene,
    lighting and exposure."""
    h, w = img.shape[:2]
    roi = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    return float(cv.Laplacian(gray, cv.CV_64F).var())


def _text(img, txt, org, scale=0.8, color=(255, 255, 255)):
    cv.putText(img, txt, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(img, txt, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA)


class LiveView:
    WIN = "acquisition - camera (q = close this window, the acquisition goes on)"
    STALE_S = 1.5          # no new frame for this long -> NO SIGNAL

    def __init__(self, cam, max_fps=15.0):
        self.cam = cam
        self.period = 1.0 / max_fps
        self._lock = threading.Lock()
        self._st = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-view", daemon=True)

    @classmethod
    def create(cls, cam):
        """Starts the viewer, or returns None (with a message) if the camera cannot
        be read without taking frames from the acquisition."""
        if not hasattr(cam, "peek"):
            print(f"[live] live view not available for the {cam.kind} camera (GC1020C only)")
            return None
        view = cls(cam)
        view._thread.start()
        return view

    def set(self, **state):
        """pose, total, layer, azim, phase, saved: shown in the overlay."""
        with self._lock:
            self._st.update(state)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self):
        scores = []
        try:
            cv.namedWindow(self.WIN, cv.WINDOW_NORMAL)
            cv.resizeWindow(self.WIN, 800, 600)
            while not self._stop.is_set():
                t0 = time.time()
                img, age = self.cam.peek()
                with self._lock:
                    st = dict(self._st)
                stale = img is None or age > self.STALE_S
                if img is None:
                    vis = np.zeros((self.cam.h, self.cam.w, 3), np.uint8)
                else:
                    vis = img.copy()
                    if not stale:
                        scores = (scores + [focus_score(img)])[-8:]
                h, w = vis.shape[:2]
                lines = []
                if st.get("pose"):
                    lines.append(f"pose {st['pose']}/{st.get('total', '?')}   "
                                 f"{st.get('layer', '')} az={st.get('azim', 0):.0f}")
                lines.append(f"{st.get('phase', 'starting')}   saved {st.get('saved', 0)}")
                if scores:
                    lines.append(f"focus {sum(scores) / len(scores):.0f}   {w}x{h}")
                for k, txt in enumerate(lines):
                    _text(vis, txt, (12, 28 + 28 * k))
                if stale:
                    msg = ("NO SIGNAL" if img is None else f"NO SIGNAL ({age:.0f} s)")
                    _text(vis, msg, (w // 2 - 130, h // 2), 1.4, (0, 0, 255))
                cv.imshow(self.WIN, vis)
                key = cv.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv.getWindowProperty(self.WIN, cv.WND_PROP_VISIBLE) < 1:
                    print("[live] window closed; the acquisition goes on.")
                    break
                time.sleep(max(0.0, self.period - (time.time() - t0)))
        except Exception as e:                # the viewer must never stop the acquisition
            print(f"[live] viewer stopped: {e}")
        finally:
            try:
                cv.destroyWindow(self.WIN)
            except Exception:
                pass
