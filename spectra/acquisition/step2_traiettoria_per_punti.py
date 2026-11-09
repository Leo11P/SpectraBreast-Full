#!/usr/bin/env python3
"""STEP 2: sequenza articolare MoveJ + video + foto ai waypoint, riutilizzando acquire.py.

Copiare accanto ad acquire.py. Dalla RADICE della repository:
  python spectra/acquisition/step2_goto_joints.py --preview-only
  python spectra/acquisition/step2_goto_joints.py --plan-only
  python spectra/acquisition/step2_goto_joints.py

L'ultimo comando abilita ed esegue il moto dopo i controlli e 5 secondi.
Foto PNG: una per tappa verificata, incluso il ritorno iniziale.
Ogni esecuzione crea una sottocartella distinta in foto_waypoints/.
Q richiede interruzione; non sostituisce l'arresto fisico.
"""
# NOI VOGLIAMO UN PROVINO CON SOLI 4 ARUCO!

import argparse
import copy
import json
import math
from pathlib import Path
import queue
import threading
import time

import yaml
import numpy as np
import cv2 as cv
import acquire as ac                     # [ORIGINALE] funzioni del vostro file

# Configurazioni assolute [J1, J2, J3, J4, J5, J6], in gradi.
# Provengono da goto_joints; non sono certificate per questa installazione.
POSES = [
    [66.546,  -74.044, 117.687, -172.105,  -88.877, -82.596],
    [67.327,  -90.308, 127.099,  -88.724,  -33.249, 124.739],
    [73.675, -109.578, 138.704,  -79.536,  -61.214,  95.196],
    [41.587,  -74.039, 123.442, -172.105,  -88.926, -74.788],
    [76.424,  -77.979, 143.171, -181.996, -121.568, -41.438],
]
RITORNO_INIZIALE = True  # anche P5 -> iniziale viene controllato
PAUSA_TAPPA = 1.0
VELOCITA = 5.0                          # parametro percentuale Fairino
ATTESA_PARTENZA = 5.0                   # nessun tasto da premere
TIMEOUT_MOVIMENTO = 120.0
# Di default legge dizionario e ID da vision.aruco nel vostro YAML.
MIN_ARUCO_VISIBILI = 3  # almeno tre qualsiasi degli ID del target
MARGINE_MARKER_PX = 40.0
FRAME_CONSECUTIVI_OK = 5
TIMEOUT_INQUADRATURA = 8.0

# Sì, a ogni tappa raggiunta — P1, P2, P3, P4, P5 e ritorno iniziale — esegue questa sequenza:
# 1. Conferma che il robot abbia raggiunto la posa e sia fermo.
# 2. Attende un secondo.
# 3. Verifica almeno tre ArUco dentro i margini per cinque fotogrammi consecutivi.
# 4. Salva una sola foto, l’ultimo fotogramma verificato.
# 5. Prosegue verso la tappa successiva.
# .png: la fotografia.
# .json accanto alla foto: nome della tappa e angoli dei giunti.
# sessione.json: parametri della camera e del controllo ArUco.

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

    def move_j_async(self, q, cfg):
        # MoveJ nativo; blendT=0 lo rende non bloccante.
        # Si invia UNA tappa e si attende l'arrivo prima di inviare la prossima.
        return self.rpc.MoveJ(list(map(float, q)), int(cfg["robot"]["tool"]),
                              int(cfg["robot"]["user"]), vel=VELOCITA, blendT=0.0)


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


def marker_frame_ok(corners, ids, required_ids, width, height, margin, min_count=3):    # abbiamo richiesto che siano visibili akmeno 3 aruco per rendere valida l'acquisizione.
    """Almeno min_count ID distinti del target, interamente entro i margini.

    Gli altri marker possono essere assenti o fuori margine. Non verifica
    centratura esatta o eventuali occlusioni del campione tra i marker.
    """
    if not 1 <= min_count <= len(set(required_ids)):
        raise ValueError("Soglia ArUco incompatibile con gli ID del target")
    if ids is None:
        return False, "nessun marker rilevato"
    found = {}
    for marker_id, points in zip(ids, corners):
        marker_id = int(marker_id)
        if marker_id not in required_ids:
            continue  # marker estranei non contribuiscono al conteggio, devo avere uno dei markers tra 0,1,2,3,4,5.
        if marker_id in found:  # verifica ulteriore per casi eccezionali
            return False, f"ID duplicato: {marker_id}"  # rifiuta frame con marker multipli dello stesso ID tipo nel caso [0, 1, 1] non accetta l'immagine.
        found[marker_id] = points
    count = 0   # inizia il conteggio del numero totale di marker nell'immagine
    for points in found.values():
        if len(points) != 4:    # se un marker non ha 4 vertici, rifiuta l'immagine, perchè significa che non vede il marker per intero
            continue
        if all(math.isfinite(x) and math.isfinite(y)    # controlla che ogni vertice abbia coordinate x,y finite.
               and margin <= x <= width - 1 - margin    # controlla che la posizione orizz e verticale rispettino il margine
               and margin <= y <= height - 1 - margin for x, y in points):
            count += 1
    reason = f"ArUco del target entro i margini: {count}/{min_count} richiesti"
    return count >= min_count, reason


def waypoint_options(full_config, acquisition_cfg): # serve per impostare il riconoscimento degli aruco del target.
    # full_config è tutto il contenuto del file default.YAML, trasformato in un dizionario python. 
    # acquisition_cfg è la sezione load_config di acquire.py.
    """Riusa il dizionario normalizzato da acquire e gli ID del frame target."""
    aruco = (full_config.get("vision") or {}).get("aruco") or {}
    name = acquisition_cfg["aruco"]["dictionary"]
    name = str(name).strip().upper()
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    ids = aruco.get("frame_marker_ids")
    if not isinstance(ids, (list, tuple)) or not ids:
        raise SystemExit("vision.aruco.frame_marker_ids deve contenere gli ID del target")
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise SystemExit("Gli ID del target devono essere interi distinti non negativi")
    return {"dictionary": name, "marker_ids": list(ids)}


class WaypointViewCheck:
    """Legge frame nuovi solo a robot fermo; non comanda correzioni di posa."""
    def __init__(self, cfg):
        options = cfg["waypoint_check"]
        self.ids = tuple(options["marker_ids"]) # tuple produce una sequenza che non viene modificata aggiungendo e togliendo elementi.
        name = options["dictionary"]
        if not self.ids or not isinstance(name, str) or not name.startswith("DICT_"):
            raise RuntimeError("Controllare dizionario aruco e vision.aruco.frame_marker_ids nel YAML")
        if not 1 <= MIN_ARUCO_VISIBILI <= len(self.ids):    # len(self.ids)=6 nel nostro caso.
            raise RuntimeError("MIN_ARUCO_VISIBILI incompatibile con gli ID del target")
        aruco = getattr(cv, "aruco", None)
        if aruco is None or not hasattr(aruco, name):
            raise RuntimeError(f"Dizionario ArUco non disponibile: {name}")
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, name))
        if any(i < 0 or i >= len(dictionary.bytesList) for i in self.ids):
            raise RuntimeError("ID marker fuori dal dizionario scelto")
        if hasattr(aruco, "ArucoDetector"): # prepara il riconoscitore da usare dopo. se è disponibile arucodetector crea un riconoscitore e conserva il suo metodo detectMarkers in self.detect.
            self.detect = aruco.ArucoDetector(dictionary).detectMarkers
        else:   # se non disponibile arucodetector, prepara una soluzione alternativa lambda.
            self.detect = lambda image: aruco.detectMarkers(image, dictionary)

    def verify(self, robot, expected_q, bridge, stop, messages, name):  # per ogni verifica di una tappa, chiama questa funzione che verifica che il robot sia fermo e che i marker siano visibili.
        deadline = time.monotonic() + TIMEOUT_INQUADRATURA  # time.monotonic misura intervallo di tempo indipendentemente dal clock del pc.
        consecutive = 0
        reason = "nessun frame valido"
        messages.put(f"{name}: robot fermo, verifico il target nel video")
        while time.monotonic() < deadline:
            if stop.is_set():
                raise RuntimeError("Interruzione richiesta")
            robot.assert_ready()
            q = robot.get_joints()
            if (not valid_joints(q) or not robot.done()
                    or max(abs(a-b) for a, b in zip(q, expected_q)) > 0.5):  # tolleranza di 0,5° per ogni giunto
                raise RuntimeError(f"{name}: robot non fermo nella posa prevista")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # grab attende un frame successivo alla chiamata, senza mediare
            # immagini: il riconoscimento usa ogni volta un'immagine reale nuova.
            frame = bridge.grab(n_avg=1, timeout=remaining)
            corners, ids, _ = self.detect(frame)
            points = [c.reshape(4, 2).tolist() for c in corners]
            labels = [] if ids is None else ids.reshape(-1).tolist()
            ok, reason = marker_frame_ok(points, labels, self.ids,
                frame.shape[1], frame.shape[0], MARGINE_MARKER_PX, MIN_ARUCO_VISIBILI)  # decide se l'immagine è valida
            if time.monotonic() > deadline: 
                break
            consecutive = consecutive + 1 if ok else 0  # aggiorna il numero di immagini valide.
            if consecutive >= FRAME_CONSECUTIVI_OK:
                # Conferma che il robot sia rimasto nella posa anche durante
                # la lettura dell'ultima immagine, prima di consegnarla al salvataggio. se non arriva a 5 fotogrammi consecutivi validi, non salva la foto e non passa alla tappa successiva!!!!!!
                if stop.is_set():
                    raise RuntimeError("Interruzione richiesta")
                robot.assert_ready()
                q_after = robot.get_joints()
                if (not valid_joints(q_after) or not robot.done()
                        or max(abs(a-b) for a, b in zip(q_after, expected_q)) > 0.5):   
                    raise RuntimeError(f"{name}: robot spostato durante la verifica")
                messages.put(f"{name}: inquadratura verificata su {consecutive} frame")
                return frame, list(map(float, q_after))
        raise RuntimeError(f"{name}: inquadratura non confermata: {reason}. "
                           "Sequenza interrotta; nessuna tappa successiva")


class WaypointPhotos:
    """Salva foto, pose e calibrazione nello stesso formato di acquire.py."""

    def __init__(self, cfg, K, D, T_c2g):
        for name in (
            "prepare_out_dir",
            "_save_image_pose_pair",
            "finalize_dataset",
        ):
            if not callable(getattr(ac, name, None)):
                raise RuntimeError(
                    f"acquire.py non contiene la funzione richiesta: {name}"
                )

        self.folder = Path(cfg["_out_dir"]).resolve()
        ac.prepare_out_dir(str(self.folder), overwrite=False)

        self.folder.mkdir(parents=True, exist_ok=True)

        for name in ("rgb", "poses", "depth", "camera_parameters"):
            (self.folder / name).mkdir(exist_ok=False)

        self.manifest = []
        camparam = self.folder / "camera_parameters"

        np.save(
            camparam / "intrinsics.npy",
            np.asarray(K, float),
        )
        np.save(
            camparam / "distortions.npy",
            np.asarray(D, float).ravel(),
        )

        (camparam / "intrinsics_source.json").write_text(
            json.dumps(cfg.get("_intrinsics_source", {}), indent=2),
            encoding="utf-8",
        )

        # Come in acquire.py: traslazione camera->flangia in metri.
        cam2ee = np.asarray(T_c2g, float).copy()
        cam2ee[:3, 3] /= 1000.0
        np.save(camparam / "camera2ee.npy", cam2ee)

        self._write_manifest()

    def _write_manifest(self):
        temporary = self.folder / ".manifest.json.tmp"

        temporary.write_text(
            json.dumps(self.manifest, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        temporary.replace(self.folder / "manifest.json")

    def save(self, frame, waypoint, actual_q, robot):
        # FK dei giunti letti dopo la verifica a robot fermo.
        flange = robot.forward_kin(actual_q)

        if not valid_joints(flange):
            raise RuntimeError(
                "FK reale non valida: foto non salvata senza posa"
            )

        T_flange = ac.pose_to_T(flange)

        index = len(self.manifest) + 1
        tag = f"{index:04d}"

        image_rel = f"rgb/img_{tag}.png"
        pose_rel = f"poses/pose_{tag}.txt"

        entry = {
            "index": index,
            "image": image_rel,
            "depth": None,
            "pose_txt": pose_rel,
            "pose_frame": "flange",
            "layer": "joint_waypoints",
            "azim_deg": None,
            "flange_cmd": list(map(float, waypoint["flange"])),
            "joints_cmd": list(map(float, waypoint["joints"])),
            "joints_deg": list(map(float, actual_q)),
            "waypoint": waypoint["name"],
            "aruco_verified": True,
        }

        # Controlla i metadati prima di scrivere la coppia.
        json.dumps(entry, allow_nan=False)

        if not ac._save_image_pose_pair(
            frame,
            str(self.folder / image_rel),
            T_flange,
            str(self.folder / pose_rel),
        ):
            raise RuntimeError(
                f"Salvataggio immagine/posa fallito: {waypoint['name']}"
            )

        self.manifest.append(entry)
        self._write_manifest()

        return self.folder / image_rel

    def close(self):
        # Verifica coppie foto/posa e genera SHA256SUMS.
        self.manifest = ac.finalize_dataset(
            str(self.folder),
            self.manifest,
        )
        self._write_manifest()

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


def plan_joints(robot, cfg, target, T_c2g, K, D, stop): # prepara e controlla l'intera sequenza prima che il robot inizi a muoversi. Se tutto ok, ritorna le pose e il guard validato.
    """Pose insegnate: verifica FK, visibilita alle tappe e trasferimenti.

    Non avvolge gli angoli a +/-180: il giro del polso resta quello richiesto.
    La campionatura del guard non certifica il percorso reale del controller.
    """
    T_start, q_start = ac.camera_T_base(robot, T_c2g)
    if T_start is None or not valid_joints(q_start):
        raise RuntimeError("Posizione iniziale non leggibile")
    q_start = list(map(float, q_start))
    center = np.asarray(target["centroid"], float)
    fk_start = robot.forward_kin(q_start)
    if not valid_joints(fk_start):
        raise RuntimeError("Cinematica diretta iniziale non valida")
    poses = [{"name": "iniziale", "joints": q_start,
              "flange": list(map(float, fk_start)),
              "cam_T_base": T_start}]
    targets = [(f"P{i}", q) for i, q in enumerate(POSES, 1)]
    if RITORNO_INIZIALE:
        targets.append(("ritorno iniziale", q_start))
    for name, q in targets:
        if stop.is_set():
            raise RuntimeError("Interruzione richiesta")
        if not valid_joints(q):
            raise RuntimeError(f"{name}: servono sei angoli finiti")
        q = list(map(float, q))
        flange = robot.forward_kin(q)
        if not valid_joints(flange):
            raise RuntimeError(f"{name}: cinematica diretta non valida")
        flange = list(map(float, flange))
        # tool=0/user=0: FK della flangia e trasformazione hand-eye.
        T = ac.pose_to_T(flange) @ T_c2g
        distance = np.linalg.norm(center - T[:3, 3])
        if not np.isfinite(distance) or distance < 1:
            raise RuntimeError(f"{name}: distanza del bersaglio non valida")
        point = ac.base_points_to_cam(center, T)[0]
        if point[2] <= 1:
            raise RuntimeError("Bersaglio dietro la camera")
        uv, _ = cv.projectPoints(point.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
        u, v = uv.reshape(2)
        vis = cfg["visibility"]
        margin = float(vis.get("margin_px", 40))
        if vis.get("margin_px_proportional", False):
            margin *= float(vis.get("margin_ref_distance_mm", 150))/distance
            margin = float(np.clip(margin, vis.get("margin_px_min", 25), vis.get("margin_px_max", 180)))
        if not (margin <= u <= cfg["camera"]["width"]-margin and
                margin <= v <= cfg["camera"]["height"]-margin):
            raise RuntimeError("Centro del bersaglio fuori dal margine immagine")
        if abs(abs(flange[4])-90) < float(cfg["visibility"]["gimbal_lock_margin_deg"]):
            raise RuntimeError("Orientamento troppo vicino alla zona Euler esclusa")
        coverage = ac._surface_coverage(target["corners_base"], T, K, D,
            int(cfg["camera"]["width"]), int(cfg["camera"]["height"]),
            float(cfg["visibility"].get("margin_px", 40)))
        if coverage < float(cfg["visibility"].get("min_marker_coverage", 0)):
            raise RuntimeError("Marker fuori dall'inquadratura prevista")
        poses.append({"name": name, "joints": q, "flange": flange,
                      "cam_T_base": T})

    from collision_guard import CollisionGuard
    guard = CollisionGuard(cfg, robot, ac.pose_to_T, robot.forward_kin,
                           T_c2g=T_c2g, euler_order=ac._EULER)
    if not (guard.enable and guard.available):
        raise RuntimeError("Modulo collisioni non disponibile: niente moto")
    guard.set_target(center)
    guard.validate([p["joints"] for p in poses])
    require_guard(guard)
    previous = q_start
    for p in poses[1:]:
        if stop.is_set():
            raise RuntimeError("Interruzione richiesta")
        safe_path(guard, previous, p["joints"])
        previous = p["joints"]
    return poses, guard, q_start


def interrupted_wait(stop, seconds):
    if stop.wait(seconds):
        raise RuntimeError("Interruzione richiesta")


def worker(cfg, bridge, K, D, stop, messages, plan_only):
    robot = None
    moving = False
    photos = None
    try:    # il programma tenta di eseguire le istruzioni seguenti, se una fallisce si passa al blocco except. A prescindere esegue comunque il finally.
        view_check = WaypointViewCheck(cfg)
        robot = RobotIO(cfg["robot"]["ip"])
        robot.assert_ready()
        if not robot.done():
            raise RuntimeError("Robot gia in movimento")
        ac._EULER = cfg["robot"]["euler_order"]  # ZYX esplicito del nostro YAML
        T_c2g = np.asarray(ac.load_T_cam2gripper(cfg), float)
        if (T_c2g.shape != (4, 4) or not np.isfinite(T_c2g).all()
                or not np.allclose(T_c2g[3], [0, 0, 0, 1])
                or not np.allclose(T_c2g[:3, :3].T @ T_c2g[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(T_c2g[:3, :3]), 1, atol=1e-4)):
            raise RuntimeError("Matrice hand-eye non rigida o non valida")
        messages.put("Leggo i marker: il robot resta fermo")
        target = ac.bootstrap(robot, bridge, cfg, T_c2g, K, D)  # [ORIGINALE]
        messages.put("Controllo sequenza giunti e collisioni: robot fermo")
        poses, guard, q_start = plan_joints(robot, cfg, target, T_c2g, K, D, stop)
        Path("step1_giunti_piano.json").write_text(json.dumps({
            "trajectory": "joint_waypoints", "speed_percent": VELOCITA,
            "return_to_start": RITORNO_INIZIALE,
            "target_mm": np.asarray(target["centroid"]).tolist(),
            "waypoints": [{"name": p["name"], "joints_deg": p["joints"],
                           "flange_mm_deg": p["flange"]} for p in poses]}, indent=2),
            encoding="utf-8")
        if plan_only:
            messages.put("Controlli conclusi. Solo piano: nessuna abilitazione o movimento")
            return
        photos = WaypointPhotos(cfg, K, D, T_c2g)
        messages.put(f"Foto della sessione: {photos.folder}")
        view_check.verify(robot, q_start, bridge, stop, messages, "iniziale")
        messages.put("Controlli passati. Partenza automatica fra 5 secondi. Q interrompe")
        interrupted_wait(stop, ATTESA_PARTENZA)
        q_now = robot.get_joints()
        if not valid_joints(q_now) or max(abs(a-b) for a,b in zip(q_now,q_start)) > 0.5:     # richiede che i giunti siano entro 0.5° dalla configurazione iniziale.
            raise RuntimeError("Il robot si e spostato durante i controlli: piano annullato")
        robot.assert_ready()
        if stop.is_set():
            raise RuntimeError("Interruzione richiesta")
        robot.enable_backstop(cfg)
        check(robot.rpc.Mode(0), "Modalita automatica")      # [ORIGINALE, spostato]
        check(robot.rpc.RobotEnable(1), "Abilitazione")       # [ORIGINALE, spostato]
        # La prima tappa e la posizione attuale: non c'e bisogno di comandarla.
        previous = q_start
        for p in poses[1:]:
            if stop.is_set():
                raise RuntimeError("Interruzione richiesta")
            robot.assert_ready()
            q_now = robot.get_joints()
            if not valid_joints(q_now):
                raise RuntimeError("Giunti attuali non leggibili")
            if max(abs(a-b) for a,b in zip(q_now, previous)) > 0.5:
                raise RuntimeError("Robot fuori dalla tappa precedente: piano annullato")
            safe_path(guard, q_now, p["joints"])
            messages.put(f"Movimento verso {p['name']}")
            moving = True                # anche una risposta persa deve richiedere stop
            check(robot.move_j_async(p["joints"], cfg), "MoveJ")
            start = time.monotonic()
            stable_since = None
            while True:
                interrupted_wait(stop, 0.05)
                robot.assert_ready()
                q = robot.get_joints()
                if not valid_joints(q):
                    raise RuntimeError("Lettura giunti fallita durante il moto")
                now = time.monotonic()
                if now-start > TIMEOUT_MOVIMENTO:
                    raise RuntimeError("Tempo massimo movimento superato")
                reached = max(abs(a-b) for a,b in zip(q,p["joints"])) <= 0.5
                if robot.done() and reached:
                    if stable_since is None:
                        stable_since = now
                    if now-stable_since >= 0.5: # attende mezzo secondo di stabilità prima di passare alla verifica del target.
                        break
                else:
                    stable_since = None
            moving = False  # considera concluso il movimento.
            previous = p["joints"]
            interrupted_wait(stop, PAUSA_TAPPA)
            frame, actual_q = view_check.verify(
                robot, p["joints"], bridge, stop, messages, p["name"])
            if stop.is_set():
                raise RuntimeError("Interruzione richiesta")
            photo_path = photos.save(frame, p, actual_q, robot) # salva foto e posa nel formato richiesto da acquire.py.
            messages.put(f"{p['name']}: foto salvata in {photo_path}")
        messages.put("FINITO: robot tornato alla configurazione iniziale"
                     if RITORNO_INIZIALE else "FINITO: ultima tappa raggiunta")
    except BaseException as exc:
        # Anche SystemExit dei controlli originali deve apparire nella finestra/log.
        messages.put(f"INTERROTTO: {exc}")
    finally:
        if robot is not None:
            if moving:
                try:
                    check(robot.rpc.StopMotion(), "StopMotion")  # [NUOVO]
                    messages.put("Stop richiesto: verificare arresto sul controller")
                except BaseException as exc:
                    messages.put(f"STOP NON CONFERMATO: {exc}. Usare arresto fisico")
            try:    # chiude la connessione robot.
                robot.close()
            except BaseException as exc:
                messages.put(
                    f"Errore chiusura connessione: {exc}"
                )

        if photos is not None:
            try:
                photos.close()
                messages.put(
                    f"Dataset verificato: {len(photos.manifest)} "
                    f"coppie in {photos.folder}"
                )
            except BaseException as exc:
                messages.put(
                    f"ERRORE FINALIZZAZIONE DATASET: {exc}"
                )
        messages.put(None)
# !!! MOVIMENTO E SALVATAGGIO VENGONO FATTI NEL WORKER, NEL MAIN MANTENGO SOLO ATTIVA LA FINESTRA!!!

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=ac.DEFAULT_CONFIG)  # permette di scegliere il file di configurazione.
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--preview-only", action="store_true", help="Solo camera, senza robot")
    modes.add_argument("--plan-only", action="store_true", help="Calcola e controlla, senza moto")
    ap.add_argument("--sample", default=None,
                    help="Nome campione come in acquire.py")
    args = ap.parse_args()
    cfg = copy.deepcopy(ac.load_config(args.config, sample=args.sample))  # [ORIGINALE] stesso YAML
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
            full_config, cfg)
        print("[waypoint]", cfg["waypoint_check"], flush=True)
    cfg["camera"]["depth"] = False        # [MODIFICATO] solo video colore, niente depth
    cfg["robot"]["vel"] = VELOCITA
    if not args.preview_only and not args.plan_only:
        ac.prepare_out_dir(cfg["_out_dir"], overwrite=False)  # [ORIGINALE] prepara la cartella di output
    stop = threading.Event()    # crea gli strumenti di comunicazione tra le due attività: camera real time video e worker che gestisce il robot e salva le foto.
    messages = queue.Queue()    # crea la coda di messaggi tra il worker e il main thread della finestra.
    bridge = VideoFrames(stop)  # crea collegamento attraverso cui il main pubblica le immagini e il worker le legge.
    camera = thread = None  # inizialmente non sono ancora stati creati nè video(camera) nè thread del robot.
    window = "STEP 2 GIUNTI - video dal vivo - Q interrompe"
    status = "Apro la telecamera"
    finished = False
    try:
        camera = ac.RS(cfg)              # [ORIGINALE] risoluzione e fps del YAML
        K, D = ac.resolve_intrinsics(cfg, camera)  # [ORIGINALE] parametri ottici della camera
        cv.namedWindow(window, cv.WINDOW_NORMAL)    # crea della finestra video
        cv.resizeWindow(window, 960, 600)   # grandezza della finestra video e non è la risoluzione delle immagini.
        last_frame = time.monotonic()
        frame = None
        while not finished:
            frames = camera.pipe.poll_for_frames()  # [NUOVO] non attende un frame
            if frames:
                color = frames.get_color_frame()
                if color:
                    frame = np.asanyarray(color.get_data()).copy()
                    last_frame = time.monotonic()
                    bridge.publish(frame)
            if time.monotonic()-last_frame > 3:
                raise RuntimeError("Video assente per oltre 3 secondi")
            if frame is not None:
                if thread is None and not args.preview_only:
                    thread = threading.Thread(target=worker,
                        args=(cfg, bridge, K, D, stop, messages, args.plan_only))   # dopo la prima immagine avvia il worker.
                    thread.start()      # [NUOVO] parte da solo, senza S o P
                if args.preview_only:
                    status = "SOLO VIDEO: nessuna connessione al robot"
                shown = frame.copy()
                cv.putText(shown, status[:100], (15, 30), cv.FONT_HERSHEY_SIMPLEX,
                           0.65, (0, 255, 0), 2)
                cv.imshow(window, shown)
            while not messages.empty(): # prende i messaggi del worker.
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