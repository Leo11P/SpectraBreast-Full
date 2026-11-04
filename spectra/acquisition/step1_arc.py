#!/usr/bin/env python3
"""STEP 1: movimento automatico + video, riutilizzando acquire.py.

Copiare accanto ad acquire.py. Dalla RADICE della repository:
  python spectra/acquisition/step1_arco.py --preview-only
  python spectra/acquisition/step1_arco.py --plan-only
  python spectra/acquisition/step1_arco.py

L'ultimo comando abilita ed esegue il moto dopo i controlli e 5 secondi.
Digita "q" per richiedere interruzione; non sostituisce l'arresto fisico.
"""
# fare: python -m pip install "opencv-contrib-python>=4.7,<5" numpy pyrealsense2 PyYAML python-fcl trimesh.
# ricordare di avere nel folder la libreria SDK Fairino.

import argparse
import copy
import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np
import cv2 as cv
import acquire as ac                     # [ORIGINALE] funzioni del vostro file

# [NUOVO] Scelte del percorso. Sono angoli attorno al bersaglio, NON dei giunti.
ANGOLI_ARCO = (0, 5, 10, 5, 0)
VELOCITA = 5.0                          # parametro percentuale Fairino
ATTESA_PARTENZA = 5.0 #sec              # nessun tasto da premere
MAX_SALTO_GIUNTO = 15.0                 # rifiuta cambi di postura troppo grandi
TIMEOUT_MOVIMENTO = 120.0               # tempo massimo controllato dal ciclo di attesa di ogni tappa
                                        # se una tappa supera il timeout_movimento, allora si verifica stopmotion.

def check(code, name):
    """Un codice diverso da zero interrompe la procedura."""
    if code != 0:
        raise RuntimeError(f"{name}: codice {code}")
# richiede un codice di successo =0

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
# controlla che q sia una lista/array di 6 numeri finiti, non None, non NaN, non inf.

class RobotIO(ac.RobotIO):  # serve per comunicare con il robot e deriva da acquire.py
    """Riusa i metodi originali, ma abilita il robot solo prima del moto."""
    def __init__(self, ip):     # ip è l'indirizzo del controller del robot.
        from fairino import Robot
        self.rpc = Robot.RPC(ip)         # [ORIGINALE] stessa connessione, self.rpc è il nome con cui il programma conserva il collegamento al robot.
        # Non chiamiamo il costruttore originale: abilitava subito il robot e
        # forzava is_connect=True. Qui una connessione difettosa deve fallire.

    def inverse_kin_near(self, flange, previous): # flange=posizione eorientamento desiderati della camera, previous=angoli dei giunti della tappa precedente.
        # API nativa; nessuna funzione aggiunta ad acquire.py. Serve a trovare i giunti.
        method = getattr(self.rpc, "GetInverseKinRef", None)
        if callable(method):
            reply = method(0, list(map(float, flange)), list(map(float, previous)))
        else:
            reply = self.rpc.GetInverseKin(0, list(map(float, flange)), -1)
        q = read_value(reply, "Cinematica inversa")
        if not valid_joints(q):
            raise RuntimeError("Cinematica inversa: sei giunti finiti richiesti")
        return list(map(float, q))

    def enable_backstop(self, cfg): # imposta l'anticollisione del controller, come richiesto dal file YAML. Se non e' impostato native_backstop=true, fallisce.
        c = cfg["collision"]
        if not c.get("native_backstop", True):
            raise RuntimeError("Questo STEP richiede collision.native_backstop=true")
        check(self.rpc.SetCollisionDetectionMethod(0, 0), "Metodo anticollisione")
        check(self.rpc.SetAnticollision(0, list(map(float, c.get("native_level", [5]*6))), 0),
              "Livello anticollisione")
        check(self.rpc.SetCollisionStrategy(int(c.get("native_strategy", 2))),
              "Strategia anticollisione")

    def assert_ready(self):
        # [NUOVO] Lettura allarmi, senza reset automatici degli allarmi.
        codes = read_value(self.rpc.GetRobotErrorCode(), "Allarmi robot")
        if not isinstance(codes, (list, tuple)) or any(x != 0 for x in codes):
            raise RuntimeError(f"Allarmi robot: {codes}")
        check(self.safety_code(), "Stato sicurezza")
# legge i codici di errore e richiede che siano tutti 0.

    def done(self):
        # [NUOVO] 0=in movimento, 1=movimento concluso.
        state = read_value(self.rpc.GetRobotMotionDone(), "Fine movimento")
        if state not in (0, 1):
            raise RuntimeError(f"Stato movimento inatteso: {state}")
        return state == 1

    def move_j_async(self, q, cfg):
        # MoveJ nativo; blendT=0 implica che il robot non riceve il movimento successivo fino a quando non ha raggiunto la tappa e ci è rimasto per almeno 0.5sec.
        # Quindi si invia UNA tappa e si attende l'arrivo prima di inviare la prossima.
        return self.rpc.MoveJ(list(map(float, q)), int(cfg["robot"]["tool"]),
                              int(cfg["robot"]["user"]), vel=VELOCITA, blendT=0.0)  # blendT e' un parametro di raccordo di movimento


class VideoFrames:
    """[NUOVO] Passa copie delle immagini al bootstrap senza aprire due camere.

    La finestra e la RealSense sono gestite solo dal programma principale.
    Il worker calcola il percorso e gestisce il robot in un thread separato;
    il thread principale continua a gestire il video.
    """
    def __init__(self, stop):   
        self.stop = stop    # segnale di interruzione
        self.condition = threading.Condition()  # strumento per coordinare i due threads
        self.frame = None   # conserva l'ultimo fotogramma disponibile
        self.number = 0  # contatore dei fotogrammi.

    def publish(self, frame):   # pubblicare una nuova immagine
        with self.condition:
            self.frame = frame.copy()
            self.number += 1
            self.condition.notify_all()

    def grab(self, n_avg=1):    # consente di ottenere nuove immagini.
        images = []
        deadline = time.monotonic() + 10.0
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
        return np.mean(images, axis=0).astype(np.uint8) # calcola la media delle immagini raccolte e la restituisce come immagine finale a 8 bit.


def rotation_about(axis, degrees):
    """[NUOVO] Rotazione geometrica attorno alla normale dei marker."""
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm < 1e-8:
        raise RuntimeError("Direzione dei marker non valida")
    return cv.Rodrigues(axis / norm * math.radians(degrees))[0] # produce matrice di rotazione 3x3 usata per ruotare la posa della camera attorno all'asse dei marker.


def arc_camera_poses(T_start, center, up):
    """Mantiene distanza, inclinazione e orientamento relativo iniziali.

    Ruota sia posizione sia orientamento iniziale, senza un ripuntamento iniziale.
    Se inizialmente il campione e centrato, resta nella stessa direzione nelle
    pose desiderate. Non e' una garanzia sul tratto MoveJ fra queste pose.
    """
    center = np.asarray(center, float)
    offset = T_start[:3, 3] - center
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    radius = np.linalg.norm(offset - np.dot(offset, up) * up)
    if not np.isfinite(radius) or radius < 10:
        raise RuntimeError("Camera quasi sopra il centro: arco troppo piccolo. "
                           "Serve una posizione iniziale obliqua, gia verificata.")
    out = []
    for angle in ANGOLI_ARCO:
        R = rotation_about(up, angle)
        T = T_start.copy()
        T[:3, 3] = center + R @ offset
        T[:3, :3] = R @ T_start[:3, :3]
        out.append(T)
    return out


def require_guard(guard):   # per controllo collisioni
    if not (guard.enable and guard.available and guard.validated):
        raise RuntimeError("Controllo collisioni assente o modello non validato: niente moto")


def safe_path(guard, q0, q1):   # q0=giunti della tappa precedente, q1=giunti della tappa successiva.
    require_guard(guard)
    ok, reasons = guard.config_safe(q1)
    if not ok:
        raise RuntimeError(f"Posizione rifiutata: {reasons}")
    ok, reasons = guard.path_safe(q0, q1)
    if not ok:
        raise RuntimeError(f"Trasferimento rifiutato: {reasons}")


def plan_arc(robot, cfg, target, T_c2g, K, D, stop):    # la variabile robot è definita in worker, che si trova nelle righe dopo rispetto a questa def.
    """[NUOVO] Pianifica solo l'arco scelto; nessun riordino o deviazione.
    1. Legge la posa iniziale della camera e i giunti iniziali.
    2. Calcola distanza e direzione del bersaglio.
    3. Controlla il puntamento iniziale della camera.
   Se la differenza angolare supera la soglia configurata, rifiuta il piano. Non centra automaticamente il bersaglio prima dell'arco.
    4. Costruisce le pose della camera.
    5. Esamina ogni posa.
   Per ciascuna:
   - controlla se è stata richiesta l'interruzione;
   - ricava la posa necessaria della flangia;
   - calcola i giunti tramite cinematica inversa.
   Per la prima e l'ultima posa usa direttamente q_start, così il ritorno richiede gli stessi giunti iniziali.
    6. Ricontrolla i giunti con la cinematica diretta.
   Calcola quale posa produrrebbero e la confronta con quella desiderata. Rifiuta differenze superiori a 2 mm o 1°.
    7. Controlla l'inquadratura prevista.
   Verifica che il centro sia davanti alla camera e dentro i margini dell'immagine.
    8. Controlla il salto articolare.
   Nessun giunto deve variare di oltre 15° tra tappe consecutive.
    9. Applica il filtro sull'orientamento Euler.
   Esclude una zona vicina a ±90° della componente angolare controllata. Questo non rappresenta un controllo completo di tutte le singolarità del robot.
    10. Controlla la copertura prevista del bersaglio usando _surface_coverage() di acquire.py.
    11. Salva la tappa nella lista poses.
    Dopo aver elaborato tutte le pose, crea il modulo CollisionGuard, ne richiede la validazione e controlla tutti i trasferimenti.
    Alla fine restituisce:
    return poses, guard, q_start
    cioè il piano, il controllo collisioni e i giunti iniziali."""
    T_start, q_start = ac.camera_T_base(robot, T_c2g)  # [ORIGINALE] legge la posa della camera
    if T_start is None or not valid_joints(q_start):
        raise RuntimeError("Posizione iniziale non leggibile")
    center = np.asarray(target["centroid"], float)
    direction = center - T_start[:3, 3]
    distance = np.linalg.norm(direction)
    if not np.isfinite(distance) or distance < 1:
        raise RuntimeError("Distanza del bersaglio non valida")
    pointing = math.degrees(math.acos(float(np.clip(
        np.dot(T_start[:3, 2], direction / distance), -1, 1)))) # pointing è l'angolo in gradi tra la direzione della camera e la direzione che la collega al centro del bersaglio.
    if pointing > float(cfg["robot"].get("pointing_gate_deg", 15)):
        raise RuntimeError(f"Camera iniziale non puntata sul bersaglio: {pointing:.1f} gradi")

    Ts = arc_camera_poses(T_start, center, target["up"])
    poses = []
    previous = q_start
    for index, (angle, T) in enumerate(zip(ANGOLI_ARCO, Ts)):
        if stop.is_set():
            raise RuntimeError("Interruzione richiesta")
        flange = ac.gripper_pose_from_camera_T(T, T_c2g)  # [ORIGINALE]
        if index in (0, len(Ts)-1):
            q = list(q_start)            # ritorna agli stessi giunti, non altra IK
        else:
            q = robot.inverse_kin_near(flange, previous)  # [ORIGINALE]
        if not valid_joints(q):
            raise RuntimeError(f"Nessuna soluzione per la tappa a {angle} gradi")
        # Verifica che IK e FK native descrivano la stessa posa della flangia.
        fk = robot.forward_kin(q)
        if fk is None:
            raise RuntimeError("Cinematica diretta non leggibile")
        actual = ac.pose_to_T(fk)
        desired = ac.pose_to_T(flange)
        delta = actual[:3, :3].T @ desired[:3, :3]
        rot_err = math.degrees(math.acos(float(np.clip((np.trace(delta)-1)/2, -1, 1))))
        if np.linalg.norm(actual[:3, 3]-desired[:3, 3]) > 2 or rot_err > 1:
            raise RuntimeError("IK/FK discordanti: oltre 2 mm o 1 grado")
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
        if max(abs(a-b) for a, b in zip(q, previous)) > MAX_SALTO_GIUNTO:
            raise RuntimeError("Salto articolare troppo grande: percorso rifiutato")
        if abs(abs(flange[4])-90) < float(cfg["visibility"]["gimbal_lock_margin_deg"]):
            raise RuntimeError("Orientamento troppo vicino alla zona Euler esclusa")
        coverage = ac._surface_coverage(target["corners_base"], T, K, D,
            int(cfg["camera"]["width"]), int(cfg["camera"]["height"]),
            float(cfg["visibility"].get("margin_px", 40)))
        if coverage < float(cfg["visibility"].get("min_marker_coverage", 0)):
            raise RuntimeError("Marker fuori dall'inquadratura prevista")
        poses.append({"angle": angle, "joints": list(q), "flange": flange,
                      "cam_T_base": T})
        previous = q

    # [ORIGINALE] Stesse API del guard richiamate da run_acquire.
    # Modulo non allegato: la sua presenza/validazione e obbligatoria a runtime.
    from collision_guard import CollisionGuard
    guard = CollisionGuard(cfg, robot, ac.pose_to_T, robot.forward_kin,
                           T_c2g=T_c2g, euler_order=ac._EULER)
    if not (guard.enable and guard.available):
        raise RuntimeError("Modulo collisioni non disponibile: niente moto")
    guard.set_target(center)
    guard.validate([list(q_start)] + [p["joints"] for p in poses])
    require_guard(guard)
    previous = q_start
    for p in poses:
        safe_path(guard, previous, p["joints"])
        previous = p["joints"]
    return poses, guard, q_start    # piano, controllo collisioni, giunti iniziali.


def interrupted_wait(stop, seconds):    # stop e' definito in main() e passato a worker() e poi a plan_arc(). Serve per interrompere l'attesa se viene premuto il tasto "q".
    if stop.wait(seconds):
        raise RuntimeError("Interruzione richiesta")


def worker(cfg, bridge, K, D, stop, messages, plan_only):
    # worker coordina pianificazione e movimento
    """Dentro il try, procede così:
    1. Crea la connessione al robot.
    2. Controlla gli allarmi.
    3. Verifica che il robot non risulti già in movimento.
    4. Imposta la convenzione degli angoli di Eulero (data dal file yaml: ZYX).
    5. Carica T_c2g, la calibrazione tra camera e flangia.
    6. Controlla che la matrice di calibrazione sia matematicamente valida.
    7. Richiama ac.bootstrap() per localizzare il bersaglio attraverso i marker.
    8. Richiama plan_arc() per calcolare e controllare il percorso.
    9. Salva il piano in step1_piano.json.
    10. Se plan_only, termina qui. altrimenti procede con l'abilitazione e il movimento del robot.
    11. Attende 5 secondi prima di partire, con possibilità di interruzione.
    12. Controlla che il robot non si sia mosso durante i controlli.
    13. Abilita l'anticollisione del robot.
    14. Abilita la modalità automatica e l'abilitazione del robot.
    15. Inizia il ciclo di movimento per ogni tappa del piano"""
    robot = None    # non c'e' ancora una connessione al robot
    moving = False  # non e' stato inviato un movimento al robot
    try:
        robot = RobotIO(cfg["robot"]["ip"])
        robot.assert_ready()
        if not robot.done():
            raise RuntimeError("Robot gia in movimento")
        ac._EULER = cfg["robot"]["euler_order"]  # ZYX esplicito del vostro YAML
        T_c2g = np.asarray(ac.load_T_cam2gripper(cfg), float)   # legge la calibrazione camera-flangia dal file YAML.
        if (T_c2g.shape != (4, 4) or not np.isfinite(T_c2g).all()
                or not np.allclose(T_c2g[3], [0, 0, 0, 1])
                or not np.allclose(T_c2g[:3, :3].T @ T_c2g[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(T_c2g[:3, :3]), 1, atol=1e-4)):
            raise RuntimeError("Matrice hand-eye non rigida o non valida")
        messages.put("Leggo i marker: il robot resta fermo")
        target = ac.bootstrap(robot, bridge, cfg, T_c2g, K, D)  # [ORIGINALE] localizza il bersaglio mediante i markers.
        messages.put("Calcolo arco e controllo collisioni: robot fermo")
        poses, guard, q_start = plan_arc(robot, cfg, target, T_c2g, K, D, stop) # calcola e controlla il percorso.
        Path("step1_piano.json").write_text(json.dumps({
            "angles_deg": ANGOLI_ARCO, "speed_percent": VELOCITA,
            "target_mm": np.asarray(target["centroid"]).tolist(),
            "waypoints": [{"angle_deg": p["angle"], "joints_deg": p["joints"],
                           "flange_mm_deg": p["flange"]} for p in poses]}, indent=2),
            encoding="utf-8")   # salva il piano con il nome step1_piano.json.
        if plan_only:
            messages.put("Controlli conclusi. Solo piano: nessuna abilitazione o movimento")
            return
        messages.put("Controlli passati. Partenza automatica fra 5 secondi. Q interrompe")
        interrupted_wait(stop, ATTESA_PARTENZA)
        q_now = robot.get_joints()
        if not valid_joints(q_now) or max(abs(a-b) for a,b in zip(q_now,q_start)) > 0.5:
            raise RuntimeError("Il robot si e spostato durante i controlli: piano annullato")
        robot.assert_ready()
        if stop.is_set():
            raise RuntimeError("Interruzione richiesta")
        robot.enable_backstop(cfg)
        check(robot.rpc.Mode(0), "Modalita automatica")      # [ORIGINALE, spostato]
        check(robot.rpc.RobotEnable(1), "Abilitazione")       # [ORIGINALE, spostato]
        # La prima tappa e la posizione attuale: non c'e bisogno di comandarla.
        previous = q_start
        for p in poses[1:]: # qui inizia il ciclo di movimenti.
            if stop.is_set():
                raise RuntimeError("Interruzione richiesta")
            robot.assert_ready()
            q_now = robot.get_joints()
            if not valid_joints(q_now):
                raise RuntimeError("Giunti attuali non leggibili")
            if max(abs(a-b) for a,b in zip(q_now, previous)) > 0.5:
                raise RuntimeError("Robot fuori dalla tappa precedente: piano annullato")
            safe_path(guard, q_now, p["joints"])
            messages.put(f"Movimento verso {p['angle']} gradi dell'arco")
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
                    if now-stable_since >= 0.5:
                        break
                else:
                    stable_since = None
            moving = False
            previous = p["joints"]
        messages.put("FINITO: robot tornato alla configurazione iniziale")
    except BaseException as exc:    # i blocchi except gestiscono i problemi e stampano i messaggi di errore nella finestra/log.
        # Anche SystemExit dei controlli originali deve apparire nella finestra/log.
        messages.put(f"INTERROTTO: {exc}")
    finally:    # pulizia finale eseguita dopo un completamento normale o un errore. Chiude la connessione al robot e segnala la fine del thread.
        if robot is not None:
            if moving:
                try:
                    check(robot.rpc.StopMotion(), "StopMotion")  # [NUOVO]
                    messages.put("Stop richiesto: verificare arresto sul controller")
                except BaseException as exc:
                    messages.put(f"STOP NON CONFERMATO: {exc}. Usare arresto fisico")
            robot.close()                # [ORIGINALE] non equivale a fermare il moto
        messages.put(None)


def main(): # prepara il programma e mantiene il video.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=ac.DEFAULT_CONFIG)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--preview-only", action="store_true", help="Solo camera, senza robot")
    modes.add_argument("--plan-only", action="store_true", help="Calcola e controlla, senza moto")
    args = ap.parse_args()
    cfg = copy.deepcopy(ac.load_config(args.config))  # [ORIGINALE] stesso YAML
    if cfg["robot"].get("tool") != 0 or cfg["robot"].get("user") != 0:
        raise SystemExit("Questa versione richiede tool=0 e user=0, come nel file ricevuto")
    if cfg["robot"].get("euler_order") not in ("ZYX", "XYZ"):
        raise SystemExit("Serve la convenzione Euler esplicita, non AUTO")
    if not cfg.get("collision", {}).get("enable"):
        raise SystemExit("collision.enable deve restare true")
    cfg["camera"]["depth"] = False        # [MODIFICATO] solo video colore, niente depth
    cfg["robot"]["vel"] = VELOCITA
    # i seguenti tre comandi interrompono, scambiano messaggi e condividono immagini.
    stop = threading.Event()
    messages = queue.Queue()
    bridge = VideoFrames(stop)  # Passi stop a videoframe per permettergli di interrompere l'attesa di nuove immagini quando chiedi di uscire.
    camera = thread = None
    window = "STEP 1 - video dal vivo - Q interrompe"
    status = "Apro la telecamera"
    finished = False
    try:
        camera = ac.RS(cfg)              # [ORIGINALE] risoluzione e fps del YAML, apre la camera.
        K, D = ac.resolve_intrinsics(cfg, camera)  # [ORIGINALE] parametri ottici K, D della camera.
        # K descrive come la camera proietta i punti 3D sul piano immagine, D descrive le distorsioni ottiche.
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        cv.resizeWindow(window, 960, 600)
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
                        args=(cfg, bridge, K, D, stop, messages, args.plan_only))
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
    finally:    # chiude la camera, il thread e la finestra, anche se ci sono stati errori o interruzioni.
        stop.set()
        # Non inviamo comandi RPC dal thread della finestra: il worker gestisce stop.
        # Una chiamata SDK bloccata puo ritardarlo: l'arresto fisico resta necessario.
        if thread is not None:
            thread.join(timeout=2)  # aspetta fino a 2 sec che il worker termini e non lo forza a terminare.
            if thread.is_alive():
                print("Operazione SDK ancora attiva: verificare il robot; usare arresto fisico se necessario")
        if camera is not None:
            camera.stop()
        cv.destroyAllWindows()
        while not messages.empty():
            msg = messages.get()
            if msg is not None:
                print(msg, flush=True)

# le seguenti righe servono per avviare tutto.
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interruzione richiesta")
    except Exception as exc:
        raise SystemExit(f"ERRORE: {exc}")