# -*- coding: utf-8 -*-
"""
rs_setup.py
===========
Utility RealSense: impostazione esposizione/gain manuali e cattura mediata su N
frame (riduce il rumore di rilevazione dei corner di ~sqrt(N)).

NON importa pyrealsense2 a livello di modulo: l'oggetto 'rs' viene passato dal
chiamante, cosi' i moduli che importano questo file restano utilizzabili anche
senza la SDK RealSense (es. modalita' offline 'validate').
"""

import numpy as np


def apply_color_settings(rs, profile, auto_exposure=False, exposure=None,
                         gain=None, auto_white_balance=True, white_balance=None,
                         verbose=True):
    """Configura il sensore COLORE. Con auto_exposure=False fissa esposizione e
    gain (corner stabili tra frame). Ogni set e' protetto: se un'opzione non e'
    supportata, avvisa e prosegue."""
    try:
        sensor = profile.get_device().first_color_sensor()
    except Exception as e:
        if verbose:
            print(f"  [rs] sensore colore non accessibile: {e}")
        return

    def _set(opt, val, name):
        try:
            if sensor.supports(opt):
                sensor.set_option(opt, float(val))
                if verbose:
                    print(f"  [rs] {name} = {val}")
            elif verbose:
                print(f"  [rs] {name}: opzione non supportata dal sensore")
        except Exception as e:
            if verbose:
                print(f"  [rs] impossibile impostare {name}: {e}")

    _set(rs.option.enable_auto_exposure, 1 if auto_exposure else 0, "auto_exposure")
    if not auto_exposure:
        if exposure is not None:
            _set(rs.option.exposure, exposure, "exposure")
        if gain is not None:
            _set(rs.option.gain, gain, "gain")
    # Il bilanciamento automatico e' ATTIVO di default: disattivarlo senza un
    # valore manuale lascia la camera su un valore residuo/di default e produce
    # una dominante di colore su TUTTA l'immagine (visto: scacchiera rosa/lavata).
    # Se lo disattivi, passa SEMPRE anche 'white_balance' (Kelvin, es. 4500-5500
    # per luce da interno) tarato col RealSense Viewer.
    _set(rs.option.enable_auto_white_balance, 1 if auto_white_balance else 0,
         "auto_white_balance")
    if not auto_white_balance and white_balance is not None:
        _set(rs.option.white_balance, white_balance, "white_balance")


def grab_averaged(rs, pipeline, timeout_ms, n_avg=1):
    """Media n_avg frame colore consecutivi -> immagine uint8.
    Da usare a ROBOT FERMO. Ritorna None se non arriva alcun frame."""
    n_avg = max(1, int(n_avg))
    acc = None
    got = 0
    for _ in range(n_avg):
        frames = pipeline.wait_for_frames(timeout_ms)
        cf = frames.get_color_frame()
        if not cf:
            continue
        img = np.asanyarray(cf.get_data()).astype(np.float32)
        acc = img if acc is None else acc + img
        got += 1
    if acc is None:
        return None
    return (acc / got).round().astype(np.uint8)
