#!/usr/bin/env python3
"""Servoing visivo continuo Fairino: mappa 6 ArUco, 3 centri inquadrati e foto.
Unico programma da avviare. Richiede servo_support.py e i moduli originali
acquire.py/collision_guard.py. Default solo piano; --execute abilita il moto.
"""
import argparse
from collections import deque
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import tempfile
import threading
import time

import numpy as np
import cv2 as cv
import yaml
import acquire as ac
from cameras import calibration_dir_name, open_camera, resolve_camera_cfg
from servo_support import (
    Quintic, NominalPath, Packet, PacketBuffer, StreamFault, send_packets,
    pointing_increment, rotation_exp, visible_marker_ids, marker_center_px,
    center_in_image,
)

POSES = [
    [107.2, -140.338, 120.791, -90.847, -99.701, 111.51],
    [104.301, -61.997, 58.383, -48.109, -99.704, 111.509],
    [45.117, -117.472, 144.872, -107.948, -58.329, 51.919],
    [74.398, -126.784, 118.404, -91.587, -85.504, 51.919],
    [62.988, -76.05, 92.915, -72.989, -84.279, 51.919],
]

RITORNO_INIZIALE = True
VELOCITA = 5.0
ID_MARKER_TARGET = ()
MIN_ARUCO_VISIBILI = 3
MARGINE_MARKER_PX = 40.0
FRAME_CONSECUTIVI_OK = 5

def check(code, name):
    """Un codice diverso da zero interrompe la procedura."""
    if code != 0:
        raise RuntimeError(f"{name}: codice {code}")

def read_value(reply, name):
    if not isinstance(reply, (tuple, list)) or len(reply) != 2:
        raise RuntimeError(f"{name}: risposta SDK inattesa: {reply!r}")
    err, result = reply
    check(err, name)
    if result is None:
        raise RuntimeError(f"{name}: valore assente")
    return result

def valid_joints(q):
    return q is not None and np.asarray(q).shape == (6,) and np.isfinite(q).all()

class RobotIO(ac.RobotIO):
    """Riusa i metodi originali, ma abilita il robot solo prima del moto."""
    def __init__(self, ip):
        from fairino import Robot
        self.rpc = Robot.RPC(ip)         # [ORIGINALE] stessa connessione
        # Non chiamiamo il costruttore originale: abilitava subito il robot e
        # forzava is_connect=True. Qui una connessione difettosa deve fallire.

    def inverse_kin_near(self, flange, previous):
        # API nativa; nessuna funzione aggiunta ad acquire.py.
        method = getattr(self.rpc, "GetInverseKinRef", None)
        if callable(method):
            reply = method(0, list(map(float, flange)), list(map(float, previous)))
        else:
            reply = self.rpc.GetInverseKin(0, list(map(float, flange)), -1)
        q = read_value(reply, "Cinematica inversa")
        if not valid_joints(q):
            raise RuntimeError("Cinematica inversa: sei giunti finiti richiesti")
        return list(map(float, q))

    def enable_backstop(self, cfg):
        c = cfg["collision"]
        if not c.get("native_backstop", True):
            raise RuntimeError("Questo STEP richiede collision.native_backstop=true")
        check(self.rpc.SetCollisionDetectionMethod(0, 0), "Metodo anticollisione")
        check(self.rpc.SetAnticollision(0, list(map(float, c.get("native_level", [5]*6))), 0),
              "Livello anticollisione")
        check(self.rpc.SetCollisionStrategy(int(c.get("native_strategy", 2))),
              "Strategia anticollisione")

    def assert_ready(self):
        # [NUOVO] Lettura allarmi, senza reset automatici.
        codes = read_value(self.rpc.GetRobotErrorCode(), "Allarmi robot")
        if not isinstance(codes, (list, tuple)) or any(x != 0 for x in codes):
            raise RuntimeError(f"Allarmi robot: {codes}")
        check(self.safety_code(), "Stato sicurezza")

    def done(self):
        # [NUOVO] 0=in movimento, 1=movimento concluso.
        state = read_value(self.rpc.GetRobotMotionDone(), "Fine movimento")
        if state not in (0, 1):
            raise RuntimeError(f"Stato movimento inatteso: {state}")
        return state == 1

class VideoFrames:
    """[NUOVO] Passa copie delle immagini al bootstrap senza aprire due camere.

    La finestra e la RealSense sono gestite solo dal programma principale.
    Il calcolo del percorso e il robot lavorano separatamente: il video continua.
    """
    def __init__(self, stop):
        self.stop = stop
        self.condition = threading.Condition()
        self.frame = None
        self.number = 0

    def publish(self, frame):
        with self.condition:
            self.frame = frame.copy()
            self.number += 1
            self.condition.notify_all()

    def grab(self, n_avg=1, timeout=10.0):
        images = []
        deadline = time.monotonic() + timeout
        with self.condition:
            previous = self.number
            while len(images) < max(1, n_avg):
                if self.stop.is_set():
                    raise RuntimeError("Interruzione richiesta")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Fotogrammi assenti durante la localizzazione")
                if self.number == previous:
                    self.condition.wait(min(0.1, remaining))
                    continue
                previous = self.number
                images.append(self.frame.copy())
        return np.mean(images, axis=0).astype(np.uint8)


def waypoint_options(full_config, acquisition_cfg, marker_ids=()):
    """Riusa il dizionario normalizzato da acquire e gli ID del frame target."""
    aruco = (full_config.get("vision") or {}).get("aruco") or {}
    name = acquisition_cfg["aruco"]["dictionary"]
    name = str(name).strip().upper()
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    ids = list(marker_ids) if marker_ids else aruco.get("frame_marker_ids")
    if not isinstance(ids, (list, tuple)) or not ids:
        raise SystemExit("vision.aruco.frame_marker_ids deve contenere gli ID del target")
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise SystemExit("Gli ID del target devono essere interi distinti non negativi")
    return {"dictionary": name, "marker_ids": list(ids)}


class WaypointPhotos:
    """Salva il frame appena verificato, senza testo sovrapposto.
    Raccolta di foto e metadati: non sostituisce il dataset completo di
    run_acquire (pose TXT, calibrazioni e manifest della pipeline Vision).
    """
    def __init__(self, root, cfg, K, D, T_c2g):     #prepara la cartella e salva le info generali della sessione
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(prefix="run_", dir=str(root)))
        self.index = 0
        self.cfg = cfg
        metadata = {
            "description": "Foto dopo centratura con Jacobiano visivo, tracking LK e mappa persistente",
            "camera_type": cfg["camera"].get("type"),
            "intrinsics_source": cfg.get("_intrinsics_source", {}),
            "camera_matrix": np.asarray(K).tolist(),
            "distortion": np.asarray(D).tolist(),
            "T_camera_to_gripper_mm": np.asarray(T_c2g).tolist(),
            "euler_order": cfg["robot"]["euler_order"],
            "waypoint_check": cfg["waypoint_check"],
            "margin_px": MARGINE_MARKER_PX,
            "consecutive_frames": FRAME_CONSECUTIVI_OK,
            "minimum_visible_target_markers": MIN_ARUCO_VISIBILI,
            "visibility_criterion": "mean_of_four_image_vertices_inside_margin",
        }
        (self.folder / "sessione.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

    def save(self, frame, name, actual_q, requested_q):     # salva una immagine e salva il suo file json
        index = self.index + 1
        stem = f"foto_{index:04d}"
        image_path = self.folder / f"{stem}.png"
        metadata_path = self.folder / f"{stem}.json"
        ok, encoded = cv.imencode(".png", frame)
        if not ok:
            raise RuntimeError(f"Codifica foto fallita: {name}")
        metadata = json.dumps({
            "waypoint": name,
            "image": image_path.name,
            "actual_joints_deg": list(actual_q),
            "requested_joints_deg": list(requested_q),
            "image_size_px": [int(frame.shape[1]), int(frame.shape[0])],
            "aruco_verified": False,
            "tracked_corners_verified": True,
            "view_source": "persistent_map_lk_tracking_pnp",
            "visual_feedback": self.cfg.get("_last_visual", {}),
        }, indent=2, allow_nan=False)
        # Nomi esclusivi: non sovrascrive foto precedenti.
        with image_path.open("xb") as out:
            out.write(encoded.tobytes())
        with metadata_path.open("x", encoding="utf-8") as out:
            out.write(metadata)
        self.index = index
        return image_path


def require_guard(guard):
    if not (guard.enable and guard.available and guard.validated):
        raise RuntimeError("Controllo collisioni assente o modello non validato: niente moto")


def safe_path(guard, q0, q1):
    require_guard(guard)
    ok, reasons = guard.config_safe(q1)
    if not ok:
        raise RuntimeError(f"Posizione rifiutata: {reasons}")
    ok, reasons = guard.path_safe(q0, q1)
    if not ok:
        raise RuntimeError(f"Trasferimento rifiutato: {reasons}")


def look_at(position, center, reference_y):     # calcola la matrice di trasformazione della camera in modo che punti verso il centro
    # non muove direttamente il robot, ma resistuisce una matrice che rappresenta la posa della camera che guarda il centro target
    # center è il centro del target, position è la posizione in mm desiderata della camera in XYZ, 
    # reference_y è la direzione di riferimento per stabilire l'inclinazione della camera attorno all'asse ottico.
    z = np.asarray(center)-position     # np.asarray(center) converte il centro in un array numpy,
    # e poi sottrae la posizione della camera per ottenere il vettore che punta dal centro della camera al centro del target.
    if np.linalg.norm(z) < 50:  # se ho una distanza tra camera e target inferiore a 50 mm, solleva un errore.
        raise RuntimeError("Camera a meno di 50 mm dal centro")
    z = z/np.linalg.norm(z)     #normalizza z
    x = np.cross(reference_y, z)    #calcola l'asse x tramite il prodotto vettoriale tra la direzione di riferimento e l'asse z. L'ordine conta!
    if np.linalg.norm(x) < 0.1:     # nel caso in cui y e z siano assi quasi paralleli, il prodotto vettoriale sarà vicino a zero e quindi non valido. In tal caso solleva un errore.
        raise RuntimeError("Prodotto vettoriale y x z non valido perche' angolo troppo piccolo")
    x /= np.linalg.norm(x)      #normalizza x
    T = np.eye(4)       # matrice identità 4x4 usata per rapp. le pose della camera.
    T[:3, :3] = np.column_stack((x, np.cross(z, x), z))     #mette i vettori in colonna nella matrice di rotazione della camera. La prima colonna è x, la seconda è y (calcolata come il prodotto vettoriale tra z e x), e la terza è z.
    T[:3, 3] = position
    return T    #restituisce la posa desiderata della camera rispetto alla base del robot.


def persistent_target(robot, bridge, cfg, H, K, D):
    """Mappa ID -> quattro vertici in base/mm; acquisita una volta da fermo."""
    identity = {"robot": cfg["robot"], "K": K.tolist(), "D": D.tolist(),
                "handeye": H.tolist(), "options": cfg["waypoint_check"],
                "marker_size_mm": cfg["_marker_mm"],
                "setup_id": cfg["_setup_id"]}   #.tolist converte gli array numpy in liste, per poterli serializzare in JSON.
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # La fingerprint è una firma della rappresentazione JSON dell'identità, ordinata per chiave. 
    # Serve a verificare che la mappa sia compatibile con la configurazione corrente.
    path = Path(cfg["_map_file"]).expanduser().resolve()
    if path.exists():   # se la mappa esiste già (NON IL NOSTRO CASO), la legge e verifica la compatibilità con la configurazione corrente.
        data = json.loads(path.read_text())
        if data.get("schema") != 1 or data.get("fingerprint") != fingerprint:
            raise RuntimeError("Mappa incompatibile con setup/calibrazione: usare un nuovo file mappa, controllare nello YAML")
    else:   # se non esiste ancora una mappa, prepara il riconoscimento dei marker ArUco e acquisisce le osservazioni necessarie per costruire la mappa.
        options = cfg["waypoint_check"]
        dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, options["dictionary"]))
        detect = (cv.aruco.ArucoDetector(dictionary).detectMarkers if hasattr(cv.aruco, "ArucoDetector")
                  else lambda im: cv.aruco.detectMarkers(im, dictionary))
        size = cfg["_marker_mm"]
        obj = np.array([[-size/2,size/2,0],[size/2,size/2,0],
                        [size/2,-size/2,0],[-size/2,-size/2,0]], dtype=np.float64)
        q0 = np.asarray(robot.get_joints(), float)
        samples = {str(i): [] for i in options["marker_ids"]}
        for _ in range(10):
            if not robot.done() or np.max(np.abs(np.asarray(robot.get_joints())-q0)) > 0.05:
                raise RuntimeError("Il robot deve restare fermo durante la mappatura")
            frame = bridge.grab(n_avg=1)
            corners, ids, _ = detect(frame)
            labels = [] if ids is None else ids.reshape(-1).tolist()
            if len(labels) != len(set(labels)):
                raise RuntimeError("ID ArUco duplicati")
            T = ac.pose_to_T(robot.forward_kin(q0.tolist())) @ H
            for c, i in zip(corners, labels):
                if str(i) not in samples:
                    continue
                pixels = np.asarray(c, float).reshape(4, 2)
                ok, r, t = cv.solvePnP(obj, pixels, K, D, flags=cv.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    raise RuntimeError("PnP fallito")
                camera_points = (cv.Rodrigues(r)[0] @ obj.T).T + t.reshape(3)
                reproj = cv.projectPoints(obj, r, t, K, D)[0].reshape(4, 2)
                if np.any(camera_points[:, 2] <= 0) or np.max(np.linalg.norm(reproj-pixels, axis=1)) > 1.5:
                    raise RuntimeError("Localizzazione marker non affidabile")
                samples[str(i)].append((T[:3, :3] @ camera_points.T).T + T[:3, 3])
        if not robot.done() or np.max(np.abs(np.asarray(robot.get_joints())-q0)) > 0.05:
            raise RuntimeError("Robot mosso durante mappatura")
        markers = {}
        for i, values in samples.items():
            if len(values) < 8:     # per creare la prima mappa serve almeno 8 osservazioni su 10 per ogni marker, altrimenti solleva un errore.
                raise RuntimeError(f"Marker {i}: servono almeno 8 osservazioni su 10")
            mean = np.mean(values, axis=0)
            if np.max(np.linalg.norm(np.asarray(values)-mean, axis=2)) > 3:
                raise RuntimeError(f"Marker {i}: stima instabile oltre 3 mm")
            markers[i] = mean.tolist()
        data = {"schema": 1, "fingerprint": fingerprint, "calibration": identity,
                "units": "mm", "frame": "robot_base", "markers": markers}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:
            json.dump(data, f, indent=2, allow_nan=False)
    if set(data["markers"]) != set(map(str, cfg["waypoint_check"]["marker_ids"])):
        raise RuntimeError("ID nella mappa non corrispondenti")
    points = np.array(list(data["markers"].values()), float)
    if points.shape != (len(data["markers"]), 4, 3) or not np.isfinite(points).all():
        raise RuntimeError("Coordinate mappa non valide")
    cfg["_map_snapshot"] = data
    return {"centroid": points.reshape(-1, 3).mean(axis=0), "corners_base": points.reshape(-1, 3)}  
# qua calcola il centro del provino come media dei 24 vertici degli aruco., salvato come centroid.

#plan_joints prepara e controlla il percorso prima del movimento.
def plan_joints(robot, cfg, target, T_c2g, K, D, stop):
    T_start, q_start = ac.camera_T_base(robot, T_c2g)
    if T_start is None or not valid_joints(q_start):
        raise RuntimeError("Posa iniziale non valida")
    cfg.update(_target=target, _K=K, _D=D, _handeye=T_c2g)  # aggiunge alla config del target la calibrazione della camera e l'hand-eye.
    geometric_view(T_start, cfg)       # verifica che la posa iniziale della camera sia valida rispetto al target e ai marker.
    poses = [{"name": "iniziale", "joints": list(q_start),
              "flange": robot.forward_kin(q_start), "cam_T_base": T_start, "photo": True}]
    # poses conterrà tutte le tappe del percorso
    positions = [(f"P{i}", (ac.pose_to_T(robot.forward_kin(q)) @ T_c2g)[:3, 3])
                 for i, q in enumerate(POSES, 1)]
    # positions ricava le posizioni della camera da poses. [:3, 3] indica che conserva solo la posizione,
    # l'orientamento della camera verrà ricostruito in direzione del target.
    if RITORNO_INIZIALE:
        positions.append(("ritorno iniziale", T_start[:3, 3]))
    previous = np.array(q_start, float)
    current = T_start[:3, 3].copy()
    for name, end in positions:
        count = max(1, int(math.ceil(np.linalg.norm(end-current)/2)))
        start = current.copy()
        for j in range(1, count+1):
            if stop.is_set():   #scorre gli intervalli intermedi e se richiesta interruzione si stoppa.
                raise RuntimeError("Interruzione richiesta")
            T = look_at(start+(end-start)*j/count, target["centroid"], T_start[:3, 1])  
            #qui T definisce position, center e reference_y per la funzione look_at
            flange = ac.gripper_pose_from_camera_T(T, T_c2g)    #trasforma posa desiderata della camera nella posa della flangia.
            q = robot.inverse_kin_near(flange, previous)    #inverse kin.
            if np.max(np.abs(np.array(q)-previous)) > 1:
                raise RuntimeError("Passo articolare oltre 1 grado: percorso rifiutato")
            actual = ac.pose_to_T(robot.forward_kin(q)) @ T_c2g #fa di nuovo la forward kin. per ricostruire la posa della camera che vorrei.
            rotation_error = np.linalg.norm(cv.Rodrigues(actual[:3,:3].T @ T[:3,:3])[0])    # differenza angolare tra orientamento desiderato e quello ricostruito.
            if np.linalg.norm(actual[:3,3]-T[:3,3]) > 1 or rotation_error > math.radians(0.5):
                raise RuntimeError("IK/FK discordanti") # rifiuta il risultato se la discordanza supera 1mm di posizione e 0,5° in orientamento.
            geometric_view(actual, cfg)
            # Verifica anche la proiezione in punti intermedi del tratto articolare.
            for alpha in np.linspace(0, 1, 5):
                qi = previous+(np.array(q)-previous)*alpha
                geometric_view(ac.pose_to_T(robot.forward_kin(qi.tolist())) @ T_c2g, cfg)
            poses.append({"name": name if j==count else f"{name}/passo{j}",
                          "joints": list(q), "flange": list(flange),
                          "cam_T_base": actual, "photo": j==count})
            previous = np.array(q)
        current = end
    from collision_guard import CollisionGuard
    guard = CollisionGuard(cfg, robot, ac.pose_to_T, robot.forward_kin,
                           T_c2g=T_c2g, euler_order=ac._EULER)
    guard.set_target(target["centroid"])
    guard.validate([p["joints"] for p in poses])
    require_guard(guard)
    for a, b in zip(poses, poses[1:]):
        safe_path(guard, a["joints"], b["joints"])
    return poses, guard, list(q_start)


def interrupted_wait(stop, seconds):
    if stop.wait(seconds):
        raise RuntimeError("Interruzione richiesta")


class VisualTracker:
    """Un riconoscimento iniziale; poi LK avanti/indietro + PnP dalla mappa.
    Eseguito sul flusso video anche durante il moto servo. Nessuna RPC qui.
    Un marker perso non viene riassociato automaticamente: si continua solo
    finche' almeno tre marker completi restano seguiti e verificati.

    La classe VisualTracker segue gli ArUco nelle immagini e stima la posizione della camera
    rispetto alla mappa 3D.
    Fa due operazioni diverse:
    - All'avvio: riconosce gli ID e associa ogni marker alle coordinate memorizzate.
    - Nei fotogrammi successivi: segue i quattro vertici con il tracking ottico, senza
    riconoscere nuovamente gli ID.
    """
    def __init__(self, frame, cfg, initial_camera_T_base):
        self.cfg = cfg
        self.map = {int(i): np.asarray(p, float) for i, p in cfg['_map_snapshot']['markers'].items()}
        # self.map legge la mappa e per ogni id conserva i 4 vertici x,y,z nel riferimento della base del robot.
        self.previous = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)   # converte immagine da colore a scala di grigi e confronta fotogramma precedente con successivo.
        # IL TRACKING UTILIZZA VARIAZIONI DI LUMINOSITA' E NON COLORI!
        detector = ac.make_detector(cfg['waypoint_check']['dictionary']) #riconosce il dizionario degli aruco.
        corners, ids = ac.detect_markers(detector, self.previous)
        labels = [] if ids is None else np.asarray(ids).reshape(-1).tolist()    # se trova i marker, ne trasforma gli id in una lista semplice.
        if len(labels) != len(set(labels)):
            raise RuntimeError('ID ArUco duplicati all avvio del tracking')
        self.points = {int(i): np.asarray(c, np.float32).reshape(4, 2)
                       for c, i in zip(corners, labels) if int(i) in self.map}  # selp.points abbina ogni id ai suoi vertici nell'immagine
        if len(self.points) < 3:
            raise RuntimeError('Tracking iniziale: servono almeno 3 dei 6 ID noti')
        B = np.linalg.inv(initial_camera_T_base) #con B passiamo dal riferimento della base a quello della camera.
        self.rvec = cv.Rodrigues(B[:3, :3])[0] #La mappa conosce i marker nello spazio: conserva le loro coordinate XYZ rispetto alla base.
        #L'immagine mostra dove sono i loro vertici in pixel.
        #La PnP (perspective and point per calcolare la posa della camera e orientamento attraverso punti conosciuti)
        #collega queste due informazioni: ricava la posizione e l’orientamento della camera rispetto ai marker conosciuti.
        #Da questa stima calcoliamo dove si trova il centro del target rispetto alla camera, quindi quanto dobbiamo correggere il puntamento.
        self.tvec = B[:3, 3].reshape(3, 1)
        self.estimate()

    def update(self, frame):    #riconosce in più frame un vertice dell'aruco, e lo fa per ogni vertice di ogni marker.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ids = sorted(self.points)   #mette in ordine gli id dei marker ancora seguiti
        old = np.concatenate([self.points[i] for i in ids]).reshape(-1, 1, 2)   #raccoglie le posizioni precedenti dei vertici nel formato richiesto da opencv.
        #adesso cercheremo gli stessi punti nell'immagine nuova:
        new, status, error = cv.calcOpticalFlowPyrLK(self.previous, gray, old, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
        # lucas-kanade cerca dove si sono spostati i punti, confrontando la luminosità attorno a ciascuno.
        # new sono le nuove coord. dei punti, status è l'indicazione di successo per ogni punto, error è l'errore di confronto fotometrico del tracking.
        if new is None or status is None:
            raise RuntimeError('Tracking ottico perso')
        back, status_back, _ = cv.calcOpticalFlowPyrLK(gray, self.previous, new, None,
            winSize=(21, 21), maxLevel=3)
        # winSize ricerca il punto precdente nell'immagine successiva in un'area di confronto larga 21px e alta 21px intorno al punto precendente.
        # maxlevel confronta anche versioni rimpicciolite dell'immagine
        if back is None or status_back is None or error is None:
            raise RuntimeError('Verifica tracking indietro fallita')
        valid = (status.reshape(-1)>0) & (status_back.reshape(-1)>0)
        valid &= np.linalg.norm(back.reshape(-1, 2)-old.reshape(-1, 2), axis=1) < 0.8
        valid &= np.isfinite(new.reshape(-1, 2)).all(axis=1)
        valid &= error.reshape(-1) < 20
        self.points = {i: new.reshape(-1, 2)[4*k:4*k+4].copy()
                       for k, i in enumerate(ids) if np.all(valid[4*k:4*k+4])}
        self.previous = gray
        self.estimate()

    def estimate(self):
        h, w = self.previous.shape
        margin = MARGINE_MARKER_PX
        self.points = {i: p for i, p in self.points.items()
            if center_in_image(p, w, h, margin)
            and cv.isContourConvex(p.astype(np.float32))
            and abs(cv.contourArea(p.astype(np.float32))) >= 25}
        if len(self.points) < 3:
            raise RuntimeError('Meno di 3 centri ArUco affidabili entro i margini: arresto')
        ids = sorted(self.points)
        objects = np.concatenate([self.map[i] for i in ids]).astype(np.float64)
        pixels = np.concatenate([self.points[i] for i in ids]).astype(np.float64)
        ok, r, t, inliers = cv.solvePnPRansac(objects, pixels, self.cfg['_K'], self.cfg['_D'],
            rvec=self.rvec.copy(), tvec=self.tvec.copy(), useExtrinsicGuess=True,
            iterationsCount=100, reprojectionError=2.0, confidence=0.999,
            flags=cv.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None:
            raise RuntimeError('PnP tracking fallita')
        accepted = set(np.asarray(inliers).reshape(-1).tolist())
        ids = [i for k, i in enumerate(ids) if set(range(4*k, 4*k+4)) <= accepted]
        if len(ids) < 3:
            raise RuntimeError('PnP: meno di 3 marker interamente coerenti')
        objects = np.concatenate([self.map[i] for i in ids]).astype(np.float64)
        pixels = np.concatenate([self.points[i] for i in ids]).astype(np.float64)
        ok, r, t = cv.solvePnP(objects, pixels, self.cfg['_K'], self.cfg['_D'],
                               r, t, True, flags=cv.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError('Raffinamento PnP fallito')
        R = cv.Rodrigues(r)[0]
        predicted = cv.projectPoints(objects, r, t, self.cfg['_K'], self.cfg['_D'])[0].reshape(-1, 2)
        camera = (R @ objects.T).T + t.reshape(3)
        if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 1):
            raise RuntimeError('PnP: punti dietro camera o non finiti')
        if np.max(np.linalg.norm(predicted-pixels, axis=1)) > 2.0:
            raise RuntimeError('Tracking incoerente con la mappa (>2 px)')
        B = np.eye(4)
        B[:3, :3], B[:3, 3] = R, t.reshape(3)
        self.camera_T_base = np.linalg.inv(B)
        self.target_camera = R @ np.asarray(self.cfg['_target']['centroid']) + t.reshape(3)
        if not np.isfinite(self.target_camera).all() or self.target_camera[2] <= 1:
            raise RuntimeError('Centro target non valido nella stima visiva')
        self.points = {i: self.points[i] for i in ids}
        self.rvec, self.tvec = r, t
        self.timestamp = time.monotonic()

    def snapshot(self):
        return {'T': self.camera_T_base.copy(), 'target_camera': self.target_camera.copy(),
                'ids': sorted(self.points),
                'centers_px': {str(i): marker_center_px(p).tolist() for i,p in self.points.items()},
                'timestamp': self.timestamp}


class ServoFrames(VideoFrames):
    """Il thread video aggiorna il tracking; il controllo ne legge copie protette."""
    def __init__(self, stop):
        super().__init__(stop)
        self.tracker = None
        self.tracking_error = None
        self.video_time = time.monotonic()

    def publish(self, frame):
        with self.condition:
            self.video_time = time.monotonic()
            if self.tracker is not None and self.tracking_error is None:
                try:
                    self.tracker.update(frame)
                except Exception as exc:
                    self.tracking_error = str(exc)
                    self.stop.set()
            self.frame = frame.copy()
            self.number += 1
            self.condition.notify_all()

    def start_tracking(self, cfg, T):
        with self.condition:
            if self.frame is None or time.monotonic()-self.video_time > 1:
                raise RuntimeError('Nessun frame recente per avviare tracking')
            self.tracker = VisualTracker(self.frame.copy(), cfg, T)

    def grab_measurement(self, timeout=2.0):
        """Immagine e stima dello STESSO frame, copiate sotto il medesimo lock."""
        deadline = time.monotonic()+timeout
        with self.condition:
            previous = self.number
            while self.number == previous:
                if self.tracking_error:
                    raise RuntimeError(self.tracking_error)
                if self.stop.is_set():
                    raise RuntimeError('Interruzione richiesta')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('Timeout frame visivo nuovo')
                self.condition.wait(min(0.1, remaining))
            if self.stop.is_set():
                raise RuntimeError(self.tracking_error or 'Interruzione richiesta')
            return self.frame.copy(), self.measurement()

    def measurement(self):
        with self.condition:
            if self.tracking_error:
                raise RuntimeError(self.tracking_error)
            if self.tracker is None:
                raise RuntimeError('Tracking non inizializzato')
            state = self.tracker.snapshot()
            if time.monotonic()-state['timestamp'] > 1.0:
                raise RuntimeError('Misura visiva vecchia oltre 1 secondo')
            return state


def marker_projection(T, cfg):
    B = np.linalg.inv(T)
    camera, pixels = {}, {}
    for key, points in cfg['_map_snapshot']['markers'].items():
        P = (B[:3, :3] @ np.asarray(points, float).T).T+B[:3, 3]
        camera[int(key)] = P
        if np.all(P[:, 2] > 1):
            pixels[int(key)] = cv.projectPoints(P, np.zeros(3), np.zeros(3), cfg['_K'], cfg['_D'])[0].reshape(4, 2)
    return visible_marker_ids(camera, pixels, cfg['camera']['width'], cfg['camera']['height'], MARGINE_MARKER_PX)


def geometric_view(T, cfg):
    """Vincolo sui centri 3 su 6, calcolati dai quattro vertici proiettati."""
    if len(marker_projection(T, cfg)) < 3:
        raise RuntimeError('Meno di 3 centri ArUco previsti nei margini')
    P = np.linalg.inv(T) @ np.r_[cfg['_target']['centroid'], 1.0]
    if P[2] <= 1 or math.degrees(math.atan2(np.linalg.norm(P[:2]), P[2])) > 8:
        raise RuntimeError('Puntamento previsto oltre 8 gradi o target dietro camera')


def checked_camera(robot, cfg):
    q = robot.get_joints()
    if not valid_joints(q):
        raise RuntimeError('Lettura giunti non valida')
    fk = robot.forward_kin(q)
    if not valid_joints(fk):
        raise RuntimeError('FK nativa non valida')
    return ac.pose_to_T(fk) @ cfg['_handeye'], list(map(float, q))


def measured_state(robot, bridge, cfg):
    """Sincronizzazione a robot fermo: frame nuovo tra due letture dei giunti."""
    if not robot.done():
        raise RuntimeError('Misura per correzione richiesta a robot in moto')
    T, q = checked_camera(robot, cfg)
    frame, state = bridge.grab_measurement(timeout=2)
    T_after, q_after = checked_camera(robot, cfg)
    if not robot.done() or np.max(np.abs(np.array(q_after)-q)) > 0.05:
        raise RuntimeError('Robot mosso durante misura visiva')
    # Limite di plausibilita della catena: non aggiorna la mappa per inseguire errori.
    angle = np.linalg.norm(cv.Rodrigues(T[:3,:3].T @ state['T'][:3,:3])[0])
    if np.linalg.norm(T[:3,3]-state['T'][:3,3]) > 25 or angle > math.radians(8):
        raise RuntimeError('Stima visiva/FK discordanti oltre 25 mm o 8 gradi')
    robot.assert_ready()
    return frame, state, T_after, q_after


def target_error_px(state, cfg):
    P = state['target_camera']
    uv = cv.projectPoints(P.reshape(1,3), np.zeros(3), np.zeros(3), cfg['_K'], cfg['_D'])[0].reshape(2)
    return float(np.linalg.norm(uv-cfg['_K'][:2, 2]))


def solve_camera_pose(robot, cfg, desired, previous):
    flange = ac.gripper_pose_from_camera_T(desired, cfg['_handeye'])
    if abs(abs(flange[4])-90) < float(cfg['visibility']['gimbal_lock_margin_deg']):
        raise RuntimeError('Orientamento nella zona Euler esclusa')
    q = robot.inverse_kin_near(flange, previous)
    actual = ac.pose_to_T(robot.forward_kin(q)) @ cfg['_handeye']
    rotation_error = np.linalg.norm(cv.Rodrigues(actual[:3,:3].T @ desired[:3,:3])[0])
    if np.linalg.norm(actual[:3,3]-desired[:3,3]) > 1 or rotation_error > math.radians(0.5):
        raise RuntimeError('IK/FK discordanti nella correzione')
    return q

PERIOD = .008
CHUNK_SAMPLES = 20
MAX_QUEUE = 40
VISION_MAX_AGE = .4
PACKET_MAX_AGE = .8
VELOCITY = 3.0
ACCELERATION = 10.0
JERK = 100.0
CAMERA_SPEED = 5.0
CORRECTION_RATE = .15

class FastFrames(ServoFrames):
    """Snapshot immutabile pubblicato dal video: il sender non attende il lock LK."""
    def __init__(self, stop):
        super().__init__(stop)
        self.fast_state = None

    def publish(self, frame):
        acquired = time.monotonic()
        super().publish(frame)
        if self.tracker is not None and self.tracking_error is None:
            state = self.tracker.snapshot()
            state['timestamp'] = acquired
            self.fast_state = state   # il dizionario non viene piu modificato

    def start_tracking(self, cfg, T):
        super().start_tracking(cfg,T)
        self.fast_state = self.tracker.snapshot()

    def fast_measurement(self):
        state=self.fast_state
        if self.tracking_error:
            raise StreamFault(self.tracking_error)
        if state is None or time.monotonic()-state['timestamp']>VISION_MAX_AGE:
            raise StreamFault('Video/tracking non aggiornato entro 400 ms')
        if len(state['ids'])<3:
            raise StreamFault('Meno di tre centri ArUco affidabili')
        return state


class CommandHistory:
    """Storia dei riferimenti inviati; nessuna RPC nel thread veloce."""
    def __init__(self, q0):
        self.values=deque([(time.monotonic(),tuple(q0))],maxlen=1000)

    def sent(self,packet,when):
        self.values.append((when,packet.joints))

    def at(self,when):
        values=list(self.values)
        if len(values)==1:
            return list(values[0][1])
        for stamp,q in reversed(values):
            if stamp<=when:
                if when-stamp>.25:
                    raise StreamFault('Storia comandi troppo vecchia per la misura visiva')
                return list(q)
        # Prima del primo invio la camera e ferma nella configurazione iniziale.
        return list(values[0][1])

    def latest(self):
        return np.array(self.values[-1][1],float)


def camera_for(robot, cfg, q):
    flange=robot.forward_kin(list(map(float,q)))
    if not valid_joints(flange):
        raise StreamFault('FK nativa non disponibile')
    return ac.pose_to_T(flange) @ cfg['_handeye']


def validate_curve(curve, robot, cfg, guard, state, reference_T, previous_q):
    """Valida il polinomio e ogni punto ServoJ prima che entri nella coda."""
    curve.require_limits(VELOCITY,ACCELERATION,JERK)
    count=round(curve.duration/PERIOD)
    joints=[curve.at(k*PERIOD) for k in range(1,count+1)]
    prev=np.asarray(previous_q,float)
    require_guard(guard)
    for q in joints:
        if np.max(np.abs(q-prev))>VELOCITY*PERIOD+1e-6:
            raise StreamFault('Incremento ServoJ oltre il limite di velocita')
        # Il guard interpola anche fra due riferimenti, non solo ai capi del blocco.
        ok,reasons=guard.path_safe(prev.tolist(),q.tolist(),n=3)
        if not ok:
            raise StreamFault(f'Collisione prevista nel blocco servo: {reasons}')
        T=camera_for(robot,cfg,q)
        geometric_view(T,cfg)
        measured_prediction=state['T'] @ np.linalg.inv(reference_T) @ T
        visible=marker_projection(measured_prediction,cfg)
        if len(set(visible)&set(state['ids']))<3:
            raise StreamFault('Il blocco lascerebbe meno di tre centri ArUco nei margini')
        prev=q
    return joints


class TrajectoryProducer(threading.Thread):
    """FK/IK e collisioni su una connessione SEPARATA dall'invio ServoJ.

    Una nuova correzione viene raccordata con quintiche C2 alla parte gia
    accodata. Non cambia retroattivamente i riferimenti gia approvati.
    """
    def __init__(self, robot, cfg, bridge, guard, path, q0, bias, history, buffer, stop, log):
        super().__init__(name='servo-planner',daemon=True)
        self.robot,self.cfg,self.bridge,self.guard=robot,cfg,bridge,guard
        self.path,self.history,self.buffer,self.stop,self.log=path,history,buffer,stop,log
        self.q=np.array(q0,float); self.v=np.zeros(6); self.a=np.zeros(6)
        self.bias=bias.copy(); self.final_q=None
        self.telemetry=(time.monotonic(),self.q.copy())
        self.ready=threading.Event()
        self.last_error=None

    def health(self):
        self.bridge.fast_measurement()
        stamp,actual=self.telemetry
        if time.monotonic()-stamp>.5:
            raise StreamFault('Telemetria robot non aggiornata entro 500 ms')
        if self.buffer.fault:
            raise StreamFault(self.buffer.fault)

    def run(self):
        try:
            self.produce()
        except BaseException as exc:
            self.buffer.fault=str(exc)
        finally:
            self.ready.set()

    def produce(self):
        dt=CHUNK_SAMPLES*PERIOD
        index=0; path_t=0.; stable_since=None; settled_frames=0; last_frame=None
        ended_at=None; previous_actual=None; stable_photo=None
        # Primo riferimento uguale ai giunti effettivi: non si salta all'inizio del piano.
        state=self.bridge.fast_measurement()
        self.buffer.put(Packet(index,tuple(self.q),state['timestamp']+PACKET_MAX_AGE))
        index+=1
        while not self.stop.is_set() and not self.buffer.cancel.is_set():
            # Non calcola un orizzonte arbitrariamente lontano dal consumo effettivo.
            while self.buffer.queue.qsize()>MAX_QUEUE-CHUNK_SAMPLES:
                if self.stop.wait(.002) or self.buffer.cancel.is_set():
                    return
            self.robot.assert_ready()
            actual=self.robot.get_joints()
            if not valid_joints(actual):
                raise StreamFault('Giunti effettivi non leggibili durante servo')
            actual=np.array(actual,float)
            if np.max(np.abs(actual-self.history.latest()))>1.0:
                raise StreamFault('Inseguimento articolare oltre 1 grado')
            self.telemetry=(time.monotonic(),actual.copy())
            state=self.bridge.fast_measurement()
            # Corregge l'età di acquisizione solo se misurata/configurata sul sistema reale.
            frame_time=state['timestamp']-self.cfg['_vision_latency']
            reference_q=self.history.at(frame_time)
            reference_T=camera_for(self.robot,self.cfg,reference_q)
            # Controllo di plausibilita sul riferimento corrispondente alla misura.
            rotation_gap=np.linalg.norm(cv.Rodrigues(reference_T[:3,:3].T@state['T'][:3,:3])[0])
            if np.linalg.norm(reference_T[:3,3]-state['T'][:3,3])>25 or rotation_gap>math.radians(8):
                raise StreamFault('Visione e riferimento temporale discordanti')
            next_t=min(path_t+dt,self.path.duration)
            nominal_q=self.path.at(next_t)
            nominal_T=camera_for(self.robot,self.cfg,nominal_q)
            desired=nominal_T.copy()
            desired[:3,:3]=nominal_T[:3,:3]@self.bias
            predicted_visual=state['T'] @ np.linalg.inv(reference_T) @ desired
            P=(np.linalg.inv(predicted_visual)@np.r_[self.cfg['_target']['centroid'],1.])[:3]
            # Correzione a velocita angolare limitata; il blocco la raccorda senza salti.
            delta=pointing_increment(P,gain=.4,max_angle_rad=math.radians(CORRECTION_RATE*dt))
            error=target_error_px(state,self.cfg)
            if path_t>=self.path.duration and error<=5:
                delta=np.zeros(3)
            accepted=None; errors=[]
            for scale in (1.,.5,.25,0.):
                candidate_bias=self.bias@rotation_exp(delta*scale)
                if np.linalg.norm(cv.Rodrigues(candidate_bias)[0])>math.radians(3):
                    raise StreamFault('Correzione accumulata oltre 3 gradi')
                desired[:3,:3]=nominal_T[:3,:3]@candidate_bias
                try:
                    goal=solve_camera_pose(self.robot,self.cfg,desired,self.q.tolist())
                    curve=Quintic(self.q,self.v,self.a,goal,self.path.at(next_t,1),
                                  self.path.at(next_t,2),dt)
                    joints=validate_curve(curve,self.robot,self.cfg,self.guard,state,reference_T,self.q)
                    accepted=(candidate_bias,curve,joints,scale)
                    break
                except (RuntimeError,ValueError) as exc:
                    errors.append(str(exc))
            if accepted is None:
                raise StreamFault(f'Nessun raccordo valido: {errors[-1]}')
            candidate_bias,curve,joints,scale=accepted
            if time.monotonic()-state['timestamp']>VISION_MAX_AGE:
                raise StreamFault('Validazione del blocco troppo lenta rispetto alla visione')
            self.bias=candidate_bias
            self.q,self.v,self.a=curve.at(dt),curve.at(dt,1),curve.at(dt,2)
            path_t=next_t
            is_last=False
            if path_t>=self.path.duration:
                ended_at=time.monotonic() if ended_at is None else ended_at
                if time.monotonic()-ended_at>30:
                    raise StreamFault('Centratura/assestamento alla tappa non conclusi entro 30 s')
                stationary=(previous_actual is not None and np.max(np.abs(actual-previous_actual))<=.03
                            and np.max(np.abs(actual-self.q))<=.15)
                if stationary and error<=5 and np.max(np.abs(self.v))<1e-6 and np.max(np.abs(self.a))<1e-6:
                    stable_since=time.monotonic() if stable_since is None else stable_since
                    if state['timestamp']!=last_frame:
                        settled_frames+=1
                        last_frame=state['timestamp']
                    is_last=(time.monotonic()-stable_since>=.6 and settled_frames>=3)
                else:
                    stable_since=None; settled_frames=0
            previous_actual=actual.copy()
            self.log.append({'time_monotonic':time.monotonic(),'nominal_t':path_t,
                'error_px':error,'ids':state['ids'],'correction_scale':scale,
                'queue_packets':self.buffer.queue.qsize(),
                'bias_rad':cv.Rodrigues(self.bias)[0].reshape(3).tolist()})
            if is_last:
                self.final_q=self.q.copy()
            for k,q in enumerate(joints):
                self.buffer.put(Packet(index,tuple(q),state['timestamp']+PACKET_MAX_AGE,
                                       last=is_last and k==len(joints)-1))
                index+=1
            if self.buffer.queue.qsize()>=CHUNK_SAMPLES+1:
                self.ready.set()
            if is_last:
                self.final_q=self.q.copy()
                self.ready.set()
                return


def stream_leg(sender, planner, cfg, bridge, guard, nodes, bias, stop, messages, log):
    q_actual=sender.get_joints()
    if not valid_joints(q_actual):
        raise StreamFault('Giunti iniziali servo non disponibili')
    path=NominalPath([p['joints'] for p in nodes],
        [p['cam_T_base'][:3,3] for p in nodes],period=PERIOD,camera_speed=CAMERA_SPEED,
        velocity=VELOCITY*.65,acceleration=ACCELERATION*.5,jerk=JERK*.5)
    history=CommandHistory(q_actual); buffer=PacketBuffer(MAX_QUEUE)
    producer=TrajectoryProducer(planner,cfg,bridge,guard,path,q_actual,bias,history,buffer,stop,log)
    active=False; stream_log=[]
    producer.start()
    try:
        deadline=time.monotonic()+10
        while not producer.ready.wait(.02):
            if stop.is_set():
                raise StreamFault('Interruzione richiesta prima della partenza servo')
            if time.monotonic()>deadline:
                raise StreamFault('Preparazione buffer servo troppo lenta')
        if buffer.fault:
            raise StreamFault(buffer.fault)
        producer.health()
        # La telemetria recente non basta: controlla anche il punto da cui si parte.
        q_now=sender.get_joints()
        if not valid_joints(q_now) or np.max(np.abs(np.array(q_now)-q_actual))>.05:
            raise StreamFault('Robot mosso durante preparazione servo')
        sender.assert_ready()
        if stop.is_set():
            raise StreamFault('Interruzione richiesta')
        active=True  # prima della RPC: una risposta persa richiede comunque StopMotion
        cfg['_motion_pending']=True
        check(sender.rpc.ServoMoveStart(),'ServoMoveStart')
        messages.put(f'Servo continuo: {nodes[-1]["name"]}; invio richiesto 125 Hz')
        stream_log=send_packets(sender.rpc,buffer,producer.health,stop,period=PERIOD,
                               on_sent=history.sent, records=stream_log)
        # L'ultimo tratto e gia a velocita e accelerazione nulle e mantiene la posa.
        check(sender.rpc.ServoMoveEnd(),'ServoMoveEnd')
        active=False  # pending resta vero finche la foto conferma il robot fermo
        return producer.bias.copy(),np.asarray(producer.final_q),stream_log
    finally:
        cfg.setdefault('_timing_records',[]).extend(stream_log)
        buffer.cancel.set()
        if active:
            try:
                check(sender.rpc.StopMotion(),'StopMotion')
            except BaseException as exc:
                messages.put(f'STOP NON CONFERMATO: {exc}; verificare arresto fisico')
            try:
                check(sender.rpc.ServoMoveEnd(),'ServoMoveEnd dopo errore')
            except BaseException as exc:
                messages.put(f'Chiusura servo non confermata: {exc}')
            # Il worker esterno puo ripetere lo stop se una chiamata non ha confermato.
        producer.join(timeout=1)
        if producer.is_alive():
            cfg['_planner_stuck']=True
            messages.put('RPC del pianificatore ancora attiva; nessuna ripartenza')


def photo_when_still(sender,bridge,cfg,expected,stop):
    deadline=time.monotonic()+5
    stable=None
    while time.monotonic()<deadline:
        if stop.is_set():
            raise StreamFault('Interruzione richiesta prima della foto')
        sender.assert_ready()
        q=sender.get_joints()
        if not valid_joints(q):
            raise StreamFault('Giunti non disponibili prima della foto')
        if sender.done() and np.max(np.abs(np.array(q)-expected))<=.15:
            stable=time.monotonic() if stable is None else stable
            if time.monotonic()-stable>=1.:
                frame,state,T,actual=measured_state(sender,bridge,cfg)
                if len(state['ids'])<3 or target_error_px(state,cfg)>5:
                    raise StreamFault('Inquadratura/centratura non confermata allo scatto')
                return frame,state,actual
        else:
            stable=None
        stop.wait(.05)
    raise StreamFault('Robot non fermo entro il tempo previsto per la foto')


def photo_groups(poses):
    yield [poses[0],poses[0]]  # centra e stabilizza anche la foto iniziale in ServoJ
    group=[poses[0]]
    for p in poses[1:]:
        group.append(p)
        if p['photo']:
            yield group
            group=[p]
    if len(group)>1:
        raise StreamFault('Percorso privo di tappa fotografica finale')


def worker(cfg,bridge,K,D,stop,messages,plan_only):
    sender=planner=photos=None; log=[]; timing=[]
    cfg.update(_motion_pending=False,_planner_stuck=False,_timing_records=timing)
    try:
        sender=RobotIO(cfg['robot']['ip'])
        sender.assert_ready()
        if not sender.done():
            raise StreamFault('Robot gia in movimento')
        for name in ('ServoMoveStart','ServoJ','ServoMoveEnd','StopMotion'):
            if not callable(getattr(sender.rpc,name,None)):
                raise StreamFault(f'SDK privo di {name}')
        ac._EULER=cfg['robot']['euler_order']
        H=np.asarray(ac.load_T_cam2gripper(cfg),float)
        if (H.shape!=(4,4) or not np.isfinite(H).all() or not np.allclose(H[3],[0,0,0,1]) or
            not np.allclose(H[:3,:3].T@H[:3,:3],np.eye(3),atol=1e-4) or
            not np.isclose(np.linalg.det(H[:3,:3]),1,atol=1e-4)):
            raise StreamFault('Hand-eye non rigida')
        cfg.update(_K=K,_D=D,_handeye=H)
        messages.put('Carico mappa ArUco e verifico il percorso, senza moto')
        cfg['_target']=persistent_target(sender,bridge,cfg,H,K,D)
        initial_T,_=checked_camera(sender,cfg)
        initial_point=(np.linalg.inv(initial_T)@np.r_[cfg['_target']['centroid'],1.])[:3]
        if initial_point[2]<=0 or math.degrees(math.atan2(np.linalg.norm(initial_point[:2]),initial_point[2]))>2:
            raise StreamFault('Partire con camera gia orientata al centro entro 2 gradi')
        poses,guard,q_start=plan_joints(sender,cfg,cfg['_target'],H,K,D,stop)
        T,q=checked_camera(sender,cfg)
        if np.max(np.abs(np.array(q)-q_start))>.05:
            raise StreamFault('Robot mosso durante pianificazione')
        bridge.start_tracking(cfg,T)
        measured_state(sender,bridge,cfg)
        # Verifica geometrica anche dei raccordi nominali C2, non solo dei nodi MoveJ.
        for nodes in photo_groups(poses):
            path=NominalPath([p['joints'] for p in nodes],[p['cam_T_base'][:3,3] for p in nodes],
                period=PERIOD,camera_speed=CAMERA_SPEED,velocity=VELOCITY*.65,
                acceleration=ACCELERATION*.5,jerk=JERK*.5)
            for piece in path.pieces:
                previous=piece.at(0)
                for seconds in np.linspace(0,piece.duration,max(3,math.ceil(piece.duration/.04)+1)):
                    if stop.is_set():
                        raise StreamFault('Interruzione richiesta durante verifica piano')
                    q=piece.at(seconds)
                    safe_path(guard,previous.tolist(),q.tolist())
                    geometric_view(camera_for(sender,cfg,q),cfg)
                    previous=q
        Path('visuo_servo_piano.json').write_text(json.dumps({'mode':'continuous_servoj_visual_feedback',
            'period_s':PERIOD,'joint_speed_deg_s':VELOCITY,'joint_accel_deg_s2':ACCELERATION,
            'joint_jerk_deg_s3':JERK,'minimum_markers':3,
            'photo_waypoints':[p['name'] for p in poses if p['photo']]},indent=2))
        if plan_only:
            messages.put('Piano nominale e tracking verificati; nessuna abilitazione. Tempi servo da misurare sul sistema reale')
            return
        # Due client RPC: il pianificatore non deve bloccare il canale ServoJ.
        planner=RobotIO(cfg['robot']['ip'])
        planner.assert_ready()
        photos=WaypointPhotos(cfg['_photos_dir'],cfg,K,D,H)
        (photos.folder/'mappa_usata.json').write_text(json.dumps(cfg['_map_snapshot'],indent=2))
        messages.put(f'Foto: {photos.folder}. Partenza dopo 5 secondi; Q interrompe')
        interrupted_wait(stop,5)
        q=sender.get_joints()
        if not valid_joints(q) or np.max(np.abs(np.array(q)-q_start))>.05:
            raise StreamFault('Robot mosso durante attesa')
        sender.assert_ready(); bridge.fast_measurement()
        if stop.is_set():
            raise StreamFault('Interruzione richiesta')
        sender.enable_backstop(cfg)
        check(sender.rpc.Mode(0),'Modalita automatica')
        check(sender.rpc.RobotEnable(1),'Abilitazione')
        bias=np.eye(3)
        for nodes in photo_groups(poses):
            if cfg['_planner_stuck']:
                raise StreamFault('Pianificatore non terminato')
            bias,expected,records=stream_leg(sender,planner,cfg,bridge,guard,nodes,bias,stop,messages,log)
            frame,state,q=photo_when_still(sender,bridge,cfg,expected,stop)
            cfg['_motion_pending']=False
            cfg['_last_visual']={'tracked_ids':state['ids'],'marker_centers_px':state['centers_px'],'centering_error_px':target_error_px(state,cfg),
                'estimated_camera_T_base_mm':state['T'].tolist(),'controller':'continuous_ServoJ'}
            saved=photos.save(frame,nodes[-1]['name'],q,nodes[-1]['joints'])
            messages.put(f'{nodes[-1]["name"]}: robot fermo, foto salvata in {saved}')
        messages.put('FINITO: trasferimenti ServoJ continui e foto alle tappe')
    except BaseException as exc:
        messages.put(f'INTERROTTO: {bridge.tracking_error or exc}')
    finally:
        if sender is not None:
            if cfg['_motion_pending']:
                try:
                    check(sender.rpc.StopMotion(),'StopMotion finale')
                except BaseException as exc:
                    messages.put(f'STOP NON CONFERMATO: {exc}')
            sender.close()
        # Non chiude il socket mentre un thread e ancora dentro la sua RPC.
        if planner is not None and not cfg['_planner_stuck']:
            planner.close()
        if photos is not None:
            try:
                (photos.folder/'servo_log.json').write_text(json.dumps(log,indent=2,allow_nan=False))
                (photos.folder/'timing_servoj.json').write_text(json.dumps(timing,indent=2,allow_nan=False))
            except Exception as exc:
                messages.put(f'Log non salvato: {exc}')
        messages.put(None)


def setup_camera_cfg(cfg):
    """[NUOVO] Come load_config dell'acquire.py nuovo: cfg['camera'] diventa la
    camera scelta (acquisition.camera.type), piatta: chiavi comuni + il suo
    blocco. Hand-eye, ingombro e cartella di calibrazione seguono la camera:
      realsense -> acquisition.hand_eye / acquisition.camera_box / DATA/camera_calibration/
      gc1020c   -> acquisition.camera.gc1020c.*            / DATA/camera_calibration_gc1020c/
    """
    cam = resolve_camera_cfg(cfg.get("camera"))
    if cam["type"] == "realsense":      # layout di questa repo: hand_eye / camera_box a parte
        cam.setdefault("T_cam2gripper", (cfg.get("hand_eye") or {}).get("T_cam2gripper"))
        cam.setdefault("camera_box", cfg.get("camera_box"))
    data_root = os.path.dirname(cfg["_calib_dir"])     # load_config: <data_root>/camera_calibration
    cfg["camera"] = cam
    cfg["camera_box"] = cam.get("camera_box")          # letto da collision_guard
    cfg["hand_eye"] = {"T_cam2gripper": cam.get("T_cam2gripper")}   # letto da ac.load_T_cam2gripper
    cfg["_calib_dir"] = os.path.join(data_root, calibration_dir_name(cam["type"]))
    return cam["type"]


def resolve_intrinsics(cfg, camera):
    """[NUOVO] RealSense: ac.resolve_intrinsics (fabbrica o override), invariato.
    GC1020C: nessuna intrinseca di fabbrica -> calibrazione OpenCV della camera,
    <calib dir>/output/intrinsics/intrinsics_opencv.json, o il JSON indicato in
    acquisition.camera.gc1020c.intrinsics. La risoluzione salvata con la
    calibrazione (img_size) deve coincidere con quella dello stream."""
    if camera.factory_intrinsics() is not None:
        return ac.resolve_intrinsics(cfg, camera)
    kind = cfg["camera"]["type"]
    path = cfg["camera"].get("intrinsics") or os.path.join(
        cfg["_calib_dir"], "output", "intrinsics", "intrinsics_opencv.json")
    path = os.path.abspath(str(path))
    if not os.path.isfile(path):
        raise SystemExit(f"[intrinsics] la camera {kind} non ha intrinseche di fabbrica e "
                         f"{path} non esiste: eseguire `spectra calibrate --modality "
                         f"intrinsics` (o impostare acquisition.camera.{kind}.intrinsics)")
    K, D = ac.load_intrinsics_override(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            size = json.load(f).get("img_size")
        size = [int(v) for v in size] if size else None
    except (OSError, ValueError, TypeError, AttributeError):
        size = None
    if size is not None and size != [camera.w, camera.h]:
        raise SystemExit(f"[intrinsics] {path} calcolate a {size[0]}x{size[1]} ma la {kind} "
                         f"trasmette {camera.w}x{camera.h}: rifare le intrinseche o "
                         "correggere width/height")
    src = {"source": "override" if cfg["camera"].get("intrinsics") else "calibration",
           "path": path, "camera": kind, "width": camera.w, "height": camera.h,
           "K": np.asarray(K, float).tolist(), "D": np.asarray(D, float).ravel().tolist()}
    cfg["_intrinsics_source"] = src
    print(f"[intrinsics] {kind}: {path}: fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
          f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    return K, D


def poll_frame(camera, last):
    """[NUOVO] Fotogramma colore BGR nuovo SENZA attendere: (img, indice) o (None, last).

    RealSense: poll_for_frames della pipeline. GC1020C (Vimba X): il callback
    dello stream conserva l'ultimo frame completo; si legge solo se piu nuovo
    di `last` (timeout 0, quindi non blocca la finestra).
    """
    if camera.kind == "realsense":
        frames = camera.pipe.poll_for_frames()
        if not frames:
            return None, last
        color = frames.get_color_frame()
        if not color:
            return None, last
        return np.asanyarray(color.get_data()).copy(), last + 1
    img, idx = camera._next(last, 0.0)
    return (None, last) if img is None else (img.copy(), idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=ac.DEFAULT_CONFIG)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--preview-only", action="store_true", help="Solo camera, senza robot")
    modes.add_argument("--execute", action="store_true", help="Abilita il moto dopo i controlli")
    modes.add_argument("--plan-only", action="store_true", help="Calcola e controlla, senza moto")
    ap.add_argument("--marker-ids", nargs="+", type=int, default=ID_MARKER_TARGET,
                    help="ID dei marker che delimitano il target")
    ap.add_argument("--photos-dir", default="foto_waypoints",
                    help="Cartella foto; crea una nuova sottocartella a ogni esecuzione")
    ap.add_argument("--marker-mm", type=float, default=None, help="Override lato marker; deve coincidere col YAML")
    ap.add_argument("--map-file", default="mappa_aruco.json")
    ap.add_argument("--setup-id", required=True, help="Identifica montaggio fisso base/target")
    ap.add_argument("--vision-latency", type=float, default=0., help="Ritardo video misurato in secondi (0 se non compensato)")
    args = ap.parse_args()
    if not math.isfinite(args.vision_latency) or not 0 <= args.vision_latency <= .2:
        ap.error("--vision-latency deve essere tra 0 e 0.2 s")
    cfg = copy.deepcopy(ac.load_config(args.config))  # [ORIGINALE] stesso YAML
    marker_mm = float(cfg["aruco"]["marker_length_mm"])
    if not math.isfinite(marker_mm) or marker_mm <= 0:
        ap.error("Lato marker YAML non valido")
    if args.marker_mm is not None and not math.isclose(args.marker_mm, marker_mm, abs_tol=1e-6):
        ap.error("--marker-mm deve coincidere con vision.aruco.marker_edge_length_mm")
    if cfg["robot"].get("tool") != 0 or cfg["robot"].get("user") != 0:
        raise SystemExit("Questa versione richiede tool=0 e user=0, come nel file ricevuto")
    if cfg["robot"].get("euler_order") not in ("ZYX", "XYZ"):
        raise SystemExit("Serve la convenzione Euler esplicita, non AUTO")
    if not cfg.get("collision", {}).get("enable"):
        raise SystemExit("collision.enable deve restare true")
    if not args.preview_only:
        with open(args.config, "r", encoding="utf-8") as config_file:
            full_config = yaml.safe_load(config_file) or {}
        cfg["waypoint_check"] = waypoint_options(
            full_config, cfg, marker_ids=args.marker_ids)
        if len(cfg["waypoint_check"]["marker_ids"]) != 6:
            ap.error("Questa versione richiede esattamente 6 ID ArUco")
        print("[waypoint]", cfg["waypoint_check"], flush=True)
    kind = setup_camera_cfg(cfg)
    print(f"[camera] {kind}", flush=True)
    if kind != "realsense" and not cfg.get("camera_box") and not args.preview_only:
        raise SystemExit(f"acquisition.camera.{kind}.camera_box mancante: configurare gli ingombri reali")
    cfg["camera"]["depth"] = False        # [MODIFICATO] solo video colore, niente depth
    cfg["robot"]["vel"] = VELOCITA
    cfg["_photos_dir"] = args.photos_dir
    cfg.update(_marker_mm=marker_mm, _map_file=args.map_file, _setup_id=args.setup_id, _vision_latency=args.vision_latency)
    stop = threading.Event()
    messages = queue.Queue()
    bridge = FastFrames(stop)
    camera = thread = None
    window = "VISUO SERVO CONTINUO - video dal vivo - Q interrompe"
    status = "Apro la telecamera"
    finished = False
    try:
        camera = open_camera(cfg["camera"], depth=False)
        K, D = resolve_intrinsics(cfg, camera)
        cfg["camera"]["width"], cfg["camera"]["height"] = int(camera.w), int(camera.h)
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        cv.resizeWindow(window, 960, int(round(960 * camera.h / camera.w)))
        last_frame = time.monotonic()
        frame = None
        frame_idx = 0
        while not finished:
            new, frame_idx = poll_frame(camera, frame_idx)
            if new is not None:
                frame = new
                last_frame = time.monotonic()
                bridge.publish(frame)
            if time.monotonic()-last_frame > 3:
                raise RuntimeError("Video assente per oltre 3 secondi")
            if frame is not None:
                if thread is None and not args.preview_only:
                    thread = threading.Thread(target=worker,
                        args=(cfg, bridge, K, D, stop, messages, not args.execute))
                    thread.start()      # [NUOVO] parte da solo, senza S o P
                if args.preview_only:
                    status = "SOLO VIDEO: nessuna connessione al robot"
                shown = frame.copy()
                cv.putText(shown, status[:100], (15, 30), cv.FONT_HERSHEY_SIMPLEX,
                           0.65, (0, 255, 0), 2)
                cv.imshow(window, shown)
            while not messages.empty():
                msg = messages.get()
                if msg is None:
                    finished = True
                else:
                    status = msg
                    print(msg, flush=True)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
            if cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stop.set()
        # Non inviamo comandi RPC dal thread della finestra: il worker gestisce stop.
        # Una chiamata SDK bloccata puo ritardarlo: l'arresto fisico resta necessario.
        if thread is not None:
            thread.join(timeout=2)
            if thread.is_alive():
                print("Operazione SDK ancora attiva: verificare il robot; usare arresto fisico se necessario")
        if camera is not None:
            camera.stop()
        cv.destroyAllWindows()
        while not messages.empty():
            msg = messages.get()
            if msg is not None:
                print(msg, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interruzione richiesta")
    except Exception as exc:
        raise SystemExit(f"ERRORE: {exc}")