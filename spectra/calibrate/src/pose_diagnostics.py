# -*- coding: utf-8 -*-
"""
pose_diagnostics.py
===================
Diagnosi dell'OSSERVABILITA' ROTAZIONALE di un set di pose per hand-eye.

Teoria (Tsai-Lenz 1989): nella soluzione di AX=XB, la parte di rotazione di X
(la cam2gripper) e' vincolata SOLO dalla diversita' degli assi di rotazione tra
le pose. Se gli assi delle rotazioni relative sono quasi coplanari o paralleli,
la rotazione attorno alla direzione "mancante" e' mal osservabile: la cam2gripper
esce sbagliata di gradi ANCHE con tante pose e ottima coerenza traslazionale.

Questo modulo NON dipende da OpenCV/robot: legge i posa*.txt (x,y,z,rx,ry,rz in
mm e gradi, Euler ZYX come Fairino) e calcola metriche oggettive + un verdetto.

Metriche:
  1. Apertura del cono di vista (spread delle direzioni d'approccio).
  2. Distribuzione degli angoli di rotazione relativi tra le pose.
  3. OSSERVABILITA' ROTAZIONALE: autovalori della matrice di informazione
     M = sum w_ij * a_ij a_ij^T  (a = asse della rotazione relativa,
     w = peso ~ magnitudine di rotazione). Il rapporto lambda_min/lambda_max
     dice quanto gli assi coprono le 3 direzioni: ~1 isotropo (ottimo),
     ~0 assi coplanari (rotazione mal condizionata).
"""

import os
import glob
import math
import numpy as np


def euler_zyx_to_R(rx, ry, rz):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_angle_axis(R):
    """Ritorna (angolo_rad, asse_unitario) di una matrice di rotazione."""
    ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if ang < 1e-9:
        return 0.0, np.array([0.0, 0.0, 1.0])
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(ax)
    if n < 1e-9:                      # rotazione di ~180 deg: asse dal termine simmetrico
        w, V = np.linalg.eigh((R + np.eye(3)) / 2.0)
        ax = V[:, np.argmax(w)]
        return ang, ax / np.linalg.norm(ax)
    return ang, ax / n


def load_poses(folder):
    poses = []
    for f in sorted(glob.glob(os.path.join(folder, "posa*.txt"))):
        txt = open(f).read().replace("\n", ",")
        v = [float(x) for x in txt.split(",") if x.strip()]
        if len(v) >= 6:
            poses.append(v[:6])
    return poses


def analyze(poses, angle_min_deg=10.0):
    """Analizza una lista di pose [x,y,z,rx,ry,rz] (mm, gradi ZYX)."""
    Rs = [euler_zyx_to_R(*p[3:]) for p in poses]
    n = len(Rs)
    if n < 3:
        raise ValueError("Servono almeno 3 pose")

    # ---- 1) apertura del cono di vista (direzione d'approccio = asse Z flangia) ----
    approach = np.array([R @ np.array([0, 0, 1.0]) for R in Rs])
    cos_max = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            cos_max = min(cos_max, float(approach[i] @ approach[j]))
    cone_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_max))))

    # ---- 2) angoli di rotazione relativi + 3) assi ----
    rel_angles = []
    axes = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            ang, ax = rot_angle_axis(Rs[i].T @ Rs[j])
            angdeg = math.degrees(ang)
            rel_angles.append(angdeg)
            if angdeg >= angle_min_deg:
                axes.append(ax)
                weights.append((math.sin(ang / 2.0)) ** 2)   # peso ~ info di Tsai
    rel_angles = np.array(rel_angles)

    if len(axes) < 2:
        return dict(n=n, cone_deg=cone_deg, rel_angles=rel_angles,
                    usable_pairs=len(axes), eig=None, isotropy=0.0,
                    weak_dir=None, verdict="POCHE ROTAZIONI SIGNIFICATIVE")

    axes = np.array(axes)
    weights = np.array(weights)
    M = np.zeros((3, 3))
    for a, w in zip(axes, weights):
        M += w * np.outer(a, a)
    M /= weights.sum()
    eigval, eigvec = np.linalg.eigh(M)          # crescente
    eig = eigval[::-1]                           # decrescente: lambda1>=lambda2>=lambda3
    isotropy = float(eig[2] / eig[0])            # ~1 isotropo, ~0 assi coplanari
    weak_dir = eigvec[:, 0]                       # direzione meno coperta (autoval minore)

    if isotropy >= 0.30:
        verdict = "BUONA osservabilita' rotazionale (assi ben distribuiti in 3D)"
    elif isotropy >= 0.10:
        verdict = "MEDIOCRE: gli assi coprono male una direzione"
    else:
        verdict = "MAL CONDIZIONATA: assi quasi coplanari -> rotazione instabile"

    return dict(n=n, cone_deg=cone_deg, rel_angles=rel_angles,
                usable_pairs=len(axes), eig=eig, isotropy=isotropy,
                weak_dir=weak_dir, verdict=verdict)


def format_report(res):
    L = []
    L.append(f"Pose analizzate: {res['n']}")
    L.append(f"Apertura cono di vista (max angolo tra direzioni d'approccio): "
             f"{res['cone_deg']:.1f} deg")
    if len(res["rel_angles"]):
        ra = res["rel_angles"]
        L.append(f"Angoli di rotazione tra pose (deg): "
                 f"mediana={np.median(ra):.1f}  min={ra.min():.1f}  max={ra.max():.1f}")
        L.append(f"  coppie con rotazione >= soglia (utili): {res['usable_pairs']}")
    if res["eig"] is not None:
        e = res["eig"]
        L.append(f"Autovalori info-rotazione (norm.): "
                 f"[{e[0]/e[0]:.2f}, {e[1]/e[0]:.2f}, {e[2]/e[0]:.2f}]")
        L.append(f"Isotropia assi (lambda_min/lambda_max): {res['isotropy']:.3f}")
        wd = res["weak_dir"]
        L.append(f"Direzione peggio coperta (asse base): "
                 f"[{wd[0]:+.2f}, {wd[1]:+.2f}, {wd[2]:+.2f}]")
    L.append("")
    L.append(f"VERDETTO: {res['verdict']}")
    if res["eig"] is not None and res["isotropy"] < 0.10:
        L.append("  -> Aggiungi pose che ruotano attorno alla direzione debole: "
                 "servono inclinazioni della camera su ASSI DIVERSI, non solo roll "
                 "attorno all'asse ottico ne' viste sulla stessa calotta.")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    poses = load_poses(folder)
    if not poses:
        print(f"Nessun posa*.txt in {folder}")
        sys.exit(1)
    print(format_report(analyze(poses)))


# ---------------------------------------- residui per-vista (target->base)
def per_view_residuals(G_list, C_list, X):
    """Per ogni vista: target2base_i = G_i @ X @ C_i (deve essere identico per
    tutte, scacchiera ferma). Ritorna, per vista, la deviazione ROTAZIONALE (deg)
    e TRASLAZIONALE (mm) dal consenso (mediana robusta).
    G_list = gripper2base 4x4, C_list = target2cam 4x4, X = cam2gripper 4x4."""
    T = [G @ X @ C for G, C in zip(G_list, C_list)]
    t_all = np.array([Ti[:3, 3] for Ti in T])
    t_ref = np.median(t_all, axis=0)

    # rotazione di consenso: media che minimizza la distanza geodetica (proiezione SVD)
    Rsum = sum(Ti[:3, :3] for Ti in T)
    U, _, Vt = np.linalg.svd(Rsum)
    R_ref = U @ Vt
    if np.linalg.det(R_ref) < 0:
        U[:, -1] *= -1
        R_ref = U @ Vt

    rot_dev, tr_dev = [], []
    for Ti in T:
        ang, _ = rot_angle_axis(R_ref.T @ Ti[:3, :3])
        rot_dev.append(math.degrees(ang))
        tr_dev.append(float(np.linalg.norm(Ti[:3, 3] - t_ref)))
    return np.array(rot_dev), np.array(tr_dev)


def format_residuals(names, rot_dev, tr_dev, rot_sigma=2.0):
    order = np.argsort(rot_dev)[::-1]
    thr = float(np.median(rot_dev) + rot_sigma * np.std(rot_dev))
    L = ["    vista     rot_dev(deg)   tr_dev(mm)   flag",
         "    " + "-" * 46]
    for i in order:
        flag = "  <== OUTLIER ROT" if rot_dev[i] > thr else ""
        L.append(f"    {str(names[i]):>8s}   {rot_dev[i]:9.3f}     {tr_dev[i]:8.2f}{flag}")
    L.append("")
    L.append(f"    rot_dev: mediana={np.median(rot_dev):.3f}  "
             f"max={rot_dev.max():.3f}  soglia_outlier={thr:.3f} deg")
    L.append(f"    tr_dev : mediana={np.median(tr_dev):.2f}  "
             f"max={tr_dev.max():.2f} mm")
    return "\n".join(L)
