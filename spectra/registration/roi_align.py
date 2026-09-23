"""
roi_align.py — Allineamento ROI -> LiveView via feature matching
=================================================================

Quando l'acquisizione HSI e' "duale":
  - LiveView PNG : immagine 2D completa con i 4 ArUco visibili
  - Cubo ROI     : tensore ENVI ritagliato sul solo campione (ArUco
                   parzialmente o totalmente assenti)

l'omografia che mappa il cubo HSI al render della mesh non puo' essere
calcolata direttamente dalla mean del cubo (gli ArUco non ci sono).
Va invece composta come:

    H_roi_to_render = H_png_to_render @ T_roi_to_png

dove T_roi_to_png e' una 3x3 calcolata UNA VOLTA per coppia (PNG, ROI)
con feature matching SIFT (o ORB) + RANSAC. Non richiede parametri
manuali ne' file esterni: viene stimata da zero confrontando la mean
del cubo ROI con la PNG LiveView.

Le due immagini devono condividere texture/contenuto (tipicamente il
campione e' visibile in entrambe), il che e' garantito dal fatto che
la ROI e' di per se' un crop della scena LiveView.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


# =============================================================================
# Detection feature: SIFT (preferito) o ORB (fallback)
# =============================================================================

def _detect_sift(img_gray: np.ndarray, n_features: int = 5000):
    """Detection SIFT + descrittori. Ritorna (kp, des) oppure (None, None)."""
    try:
        sift = cv2.SIFT_create(nfeatures=n_features)
    except AttributeError:
        # OpenCV vecchio o build senza contrib
        return None, None
    kp, des = sift.detectAndCompute(img_gray, None)
    if des is None or len(kp) < 4:
        return None, None
    return kp, des


def _detect_orb(img_gray: np.ndarray, n_features: int = 10000):
    """Detection ORB + descrittori. Sempre disponibile in opencv-python."""
    orb = cv2.ORB_create(nfeatures=n_features, scoreType=cv2.ORB_HARRIS_SCORE)
    kp, des = orb.detectAndCompute(img_gray, None)
    if des is None or len(kp) < 4:
        return None, None
    return kp, des


def _match_descriptors(des1, des2, matcher_type: str, ratio: float = 0.75):
    """
    Matching con knnMatch + ratio test di Lowe.

    matcher_type : 'flann_kdtree' per SIFT (descrittori float),
                   'bf_hamming'   per ORB  (descrittori binari).
    """
    if matcher_type == 'flann_kdtree':
        index_params  = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif matcher_type == 'bf_hamming':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        raise ValueError(f"matcher_type={matcher_type!r} non supportato")

    raw = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


# =============================================================================
# Pre-processing per dare a SIFT/ORB piu' contrasto sulla mean del cubo
# =============================================================================

def _normalize_for_features(img: np.ndarray) -> np.ndarray:
    """
    Converte un'immagine arbitraria (float, uint8, ...) in uint8 contrastata
    pronta per SIFT/ORB. Usa equalizeHist dopo stretch dei percentili 2-98.
    """
    if img.dtype != np.uint8:
        a = np.nan_to_num(img.astype(np.float32))
        a = np.maximum(a, 0)
        p2, p98 = np.percentile(a, (2, 98))
        if p98 - p2 > 0:
            a = np.clip((a - p2) / (p98 - p2), 0, 1)
        else:
            a = a / (a.max() + 1e-6)
        img8 = (a * 255).astype(np.uint8)
    else:
        img8 = img.copy()
    return cv2.equalizeHist(img8)


# =============================================================================
# Visualizzazione diagnostica
# =============================================================================

def _save_match_viz(roi_img: np.ndarray, png_img: np.ndarray,
                    kp1, kp2, good, mask, out_path: str,
                    max_lines: int = 80) -> None:
    """Salva immagine con le righe di match inlier (campionate)."""
    inliers = [g for i, g in enumerate(good) if mask[i]]
    # Sotto-campiono per leggibilita'
    if len(inliers) > max_lines:
        step = len(inliers) // max_lines
        inliers = inliers[::step]
    viz = cv2.drawMatches(
        roi_img, kp1, png_img, kp2, inliers, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, viz)


def _save_overlay_viz(roi_img: np.ndarray, png_img: np.ndarray,
                      H: np.ndarray, out_path: str) -> None:
    """Warpa la ROI sulla PNG e salva un'immagine sovrapposta 50/50."""
    H32 = H.astype(np.float32)
    warped = cv2.warpPerspective(roi_img, H32, (png_img.shape[1], png_img.shape[0]))
    overlay = cv2.addWeighted(png_img, 0.5, warped, 0.5, 0)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, overlay)


# =============================================================================
# API principale
# =============================================================================

def compute_roi_to_png_homography(
    roi_image: np.ndarray,
    png_image: np.ndarray,
    method: str           = 'sift',
    min_matches: int      = 20,
    ransac_thresh: float  = 3.0,
    lowe_ratio: float     = 0.75,
    n_features_sift: int  = 5000,
    n_features_orb: int   = 10000,
    fallback_orb: bool    = True,
    output_dir: Optional[str] = None,
    sample_name: str      = 'sample',
    save_viz: bool        = True,
) -> tuple[np.ndarray, dict]:
    """
    Stima l'omografia 3x3 che mappa pixel della ROI -> pixel della PNG LiveView.

    Strategia:
        1. Pre-processing (equalizeHist + percentile stretch) per dare contrasto
        2. Detection feature: SIFT se disponibile, altrimenti ORB
        3. knnMatch + ratio test di Lowe
        4. RANSAC su findHomography
        5. (Opzionale) Salvataggio immagine diagnostica

    Parameters
    ----------
    roi_image       : (H, W) numpy array. Immagine 2D derivata dal cubo ROI
                      (es. mean delle bande, o singola banda visibile).
    png_image       : (H, W) numpy array uint8. La PNG LiveView.
    method          : 'sift' o 'orb'. SIFT e' preferito per accuratezza.
    min_matches     : numero minimo di good matches per accettare la stima.
                      Sotto questa soglia -> ValueError.
    ransac_thresh   : soglia px per cv2.findHomography RANSAC.
    lowe_ratio      : soglia ratio test di Lowe (tipico 0.7-0.8).
    fallback_orb    : se SIFT non e' disponibile o fallisce, prova ORB.
    output_dir      : se non None, salva immagini diagnostiche qui dentro.
    sample_name     : prefisso file diagnostici.
    save_viz        : se False non salva nulla anche se output_dir e' valido.

    Returns
    -------
    T_roi_to_png : (3, 3) float64. Omografia tale che
                   pts_png = perspectiveTransform(pts_roi, T_roi_to_png).
    info         : dict con metriche di qualita':
                   - method, n_keypoints_roi, n_keypoints_png,
                     n_good_matches, n_inliers, inlier_ratio,
                     reproj_error_mean_px (sui soli inliers)

    Raises
    ------
    ValueError   : se nessuna stima riesce a superare min_matches.
    """
    if roi_image is None or roi_image.size == 0:
        raise ValueError("roi_image vuoto o None")
    if png_image is None or png_image.size == 0:
        raise ValueError("png_image vuoto o None")

    roi_u8 = _normalize_for_features(roi_image)
    png_u8 = _normalize_for_features(png_image)

    method = method.lower().strip()
    if method not in ('sift', 'orb'):
        raise ValueError(f"method deve essere 'sift' o 'orb', non {method!r}")

    # ── Tentativo principale ─────────────────────────────────────────────────
    used_method = method
    if method == 'sift':
        kp1, des1 = _detect_sift(roi_u8, n_features_sift)
        kp2, des2 = _detect_sift(png_u8, n_features_sift)
        matcher_type = 'flann_kdtree'
        if (des1 is None or des2 is None) and fallback_orb:
            print("[ROI align] SIFT non disponibile o senza descrittori "
                  "-> fallback ORB")
            used_method = 'orb'
            kp1, des1 = _detect_orb(roi_u8, n_features_orb)
            kp2, des2 = _detect_orb(png_u8, n_features_orb)
            matcher_type = 'bf_hamming'
    else:   # 'orb'
        kp1, des1 = _detect_orb(roi_u8, n_features_orb)
        kp2, des2 = _detect_orb(png_u8, n_features_orb)
        matcher_type = 'bf_hamming'

    if des1 is None or des2 is None:
        raise ValueError("[ROI align] Nessun descrittore estratto da ROI o PNG.")

    print(f"[ROI align] Method: {used_method.upper()}  "
          f"keypoints: ROI={len(kp1)}  PNG={len(kp2)}")

    good = _match_descriptors(des1, des2, matcher_type, ratio=lowe_ratio)
    print(f"[ROI align] Good matches (ratio<{lowe_ratio}): {len(good)}")

    if len(good) < min_matches:
        # Ultimo tentativo: rilasso il ratio test
        good = _match_descriptors(des1, des2, matcher_type, ratio=0.85)
        print(f"[ROI align] Retry con ratio=0.85: {len(good)} match")
        if len(good) < min_matches:
            raise ValueError(
                f"[ROI align] Solo {len(good)} match (< {min_matches} richiesti). "
                f"Le due immagini potrebbero non condividere abbastanza texture."
            )

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    T, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if T is None:
        raise ValueError("[ROI align] cv2.findHomography ha restituito None")

    mask_flat = mask.ravel().astype(bool)
    n_inliers = int(mask_flat.sum())
    if n_inliers < min_matches // 2:
        raise ValueError(
            f"[ROI align] Solo {n_inliers} inliers RANSAC "
            f"(<{min_matches//2} minimo). Stima inaffidabile."
        )

    # Errore di reproiezione medio sui soli inliers
    src_in = src[mask_flat]
    dst_in = dst[mask_flat]
    proj   = cv2.perspectiveTransform(src_in, T)
    reproj_err = float(np.mean(np.linalg.norm(
        proj.reshape(-1, 2) - dst_in.reshape(-1, 2), axis=1
    )))

    print(f"[ROI align] RANSAC inliers: {n_inliers}/{len(good)} "
          f"({100*n_inliers/len(good):.1f}%)")
    print(f"[ROI align] Reproj error (inliers): {reproj_err:.3f} px")

    info = {
        'method'              : used_method,
        'n_keypoints_roi'     : len(kp1),
        'n_keypoints_png'     : len(kp2),
        'n_good_matches'      : len(good),
        'n_inliers'           : n_inliers,
        'inlier_ratio'        : n_inliers / len(good) if good else 0.0,
        'reproj_error_mean_px': reproj_err,
    }

    # ── Diagnostica ──────────────────────────────────────────────────────────
    if save_viz and output_dir is not None:
        viz_path = os.path.join(output_dir,
                                f'{sample_name}_roi_alignment_matches.png')
        overlay_path = os.path.join(output_dir,
                                    f'{sample_name}_roi_alignment_overlay.png')
        try:
            _save_match_viz(roi_u8, png_u8, kp1, kp2, good, mask_flat, viz_path)
            print(f"[ROI align] Diagnostica match  -> {viz_path}")
        except Exception as e:
            print(f"[ROI align] WARNING: match viz fallita: {e}")
        try:
            _save_overlay_viz(roi_u8, png_u8, T, overlay_path)
            print(f"[ROI align] Diagnostica overlay -> {overlay_path}")
        except Exception as e:
            print(f"[ROI align] WARNING: overlay viz fallita: {e}")

    return T.astype(np.float64), info


# =============================================================================
# Helper: caricamento PNG LiveView (con check)
# =============================================================================

def load_liveview_png(png_path: str) -> np.ndarray:
    """
    Carica la PNG LiveView come uint8 grayscale.

    Raises
    ------
    FileNotFoundError : se il file non esiste.
    ValueError        : se l'immagine non e' leggibile.
    """
    if not os.path.isfile(png_path):
        raise FileNotFoundError(f"LiveView PNG non trovata: {png_path}")
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossibile leggere LiveView PNG: {png_path}")
    return img


# =============================================================================
# Test standalone
# =============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python3 roi_align.py <roi.png> <png_liveview.png> [output_dir]")
        sys.exit(1)
    roi = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    png = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    out = sys.argv[3] if len(sys.argv) > 3 else '.'
    T, info = compute_roi_to_png_homography(roi, png, output_dir=out)
    print(f"\nT =\n{T}")
    print(f"info = {info}")
