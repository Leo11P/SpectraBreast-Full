# Le definizioni vengono lette dall'alto; l'esecuzione vera parte dal main in fondo.
# Lessico: q=sei giunti in gradi; XYZ=coordinate in mm; uv=coordinate in pixel.
# Una matrice T 4x4 contiene rotazione 3x3 e posizione XYZ nell'ultima colonna.
# L'operatore @ moltiplica matrici; .T trasponde; [:3,3] prende XYZ.
# self indica questa istanza della classe; cfg e un dizionario di configurazione.
# Un metodo ereditato puo essere sostituito nella sottoclasse: nel main si crea FastFrames.
# PREVISIONE: centro proiettato dalla mappa e dalla FK. OSSERVAZIONE: centro misurato nel frame.
# SOLO MAPPA continua senza nuova osservazione: non puo confermare sfocatura, occlusione o spostamenti fisici.
# Durante il trasferimento si punta il target; tre centri PREVISTI sono richiesti soltanto alle foto.
# Le correzioni del puntamento sono calcolate anche DURANTE il moto in questa versione.
# Questa copia non cambia soglie, pose, algoritmi o impostazioni camera.
"""Servoing continuo Fairino: target durante il moto, 3 centri previsti alle foto.
Richiede servo_support.py e i moduli originali
Default solo piano; --execute abilita il moto."""
# Strumento per leggere le opzioni del comando, per esempio --execute e --map-file.
import argparse
# Coda con memoria limitata: conserva riferimenti e pose recenti, eliminando quelli piu vecchi.
from collections import deque
# Serve per copiare la configurazione senza modificare involontariamente il dizionario originale.
import copy
# Calcola una firma ripetibile delle impostazioni per controllare il riuso della mappa.
import hashlib
# Legge e scrive mappe, calibrazioni e informazioni delle fotografie come testo strutturato.
import json
# Funzioni matematiche e conversioni tra gradi e radianti.
import math
# Gestisce percorsi di cartelle e file della calibrazione.
import os
# Rappresenta percorsi e consente lettura, scrittura e creazione di cartelle.
from pathlib import Path
# Code thread-safe per messaggi o riferimenti, secondo il punto del programma.
import queue
# Crea una cartella run_... distinta per ogni sessione, senza cancellarla automaticamente.
import tempfile
# Thread, eventi e lock per coordinare video, pianificazione e invio.
import threading
# Orologio monotono e attese: serve alle scadenze e alle verifiche di freschezza.
import time

# NumPy: array e matrici; np e soltanto un nome abbreviato.
import numpy as np
# OpenCV: immagini, ArUco, confronto patch, proiezioni e PnP; cv e l’abbreviazione.
import cv2 as cv
# Legge il file di configurazione YAML; non contiene i comandi del robot.
import yaml
# Riusa le funzioni originali del progetto: ac.nome richiama una funzione di acquire.py.
import acquire as ac
# Gestione camera del progetto: selezione configurazione, apertura e cartella calibrazione.
from cameras import calibration_dir_name, open_camera, resolve_camera_cfg
# Importa i calcoli e il ciclo di invio commentati nel secondo file di questo pacchetto.
from servo_support import (
    Quintic, NominalPath, Packet, PacketBuffer, StreamFault, send_packets,
    pointing_increment, rotation_exp, visible_marker_ids, marker_center_px,
    center_in_image,
)

# Cinque configurazioni [J1..J6] in gradi. Sono quelle dell’ultimo allegato; il piano ne usa gli XYZ camera e ricalcola l’orientamento.
POSES = [
    [107.2, -140.338, 120.791, -90.847, -99.701, 111.51],
    [104.301, -61.997, 58.383, -48.109, -99.704, 111.509],
    [45.117, -117.472, 144.872, -107.948, -58.329, 51.919],
    [74.398, -126.784, 118.404, -91.587, -85.504, 51.919],
    [62.988, -76.05, 92.915, -72.989, -84.279, 51.919],
]

# True aggiunge la destinazione XYZ iniziale della camera. Il roll e i giunti possono differire dopo le correzioni.
RITORNO_INIZIALE = True
# Campo percentuale legacy scritto nella configurazione. Non e il limite velocita del flusso ServoJ.
VELOCITA = 50.0
# Tupla vuota: senza --marker-ids usa gli ID del YAML.
ID_MARKER_TARGET = ()
# Numero minimo di ID distinti PREVISTI ai controlli fotografici; non e richiesto durante il moto.
MIN_ARUCO_VISIBILI = 3
# Distanza minima del CENTRO dai bordi allo scatto. Non richiede che tutto il quadrato entri nel margine.
MARGINE_MARKER_PX = 40.0
# Valore legacy riportato in sessione.json; non e il numero di frame usato dall’assestamento continuo attuale.
FRAME_CONSECUTIVI_OK = 5

# FUNZIONE: check
# Controlla il codice restituito da un comando Fairino.
# INGRESSO: code e il codice SDK; name e il nome leggibile dell'operazione.
# Zero significa successo. Un valore diverso da zero genera un errore.
# Questa funzione non arresta da sola il robot: lo stop viene richiesto
# nei blocchi di gestione degli errori del movimento.
def check(code, name):
    """Un codice diverso da zero interrompe la procedura."""
    if code != 0:
        raise RuntimeError(f"{name}: codice {code}")


# FUNZIONE: read_value
# Estrae il valore da una risposta SDK nel formato (errore, valore).
# Prima verifica struttura, codice di successo e presenza del valore.
# RISULTATO: il dato richiesto, per esempio giunti o stato movimento.
# Una risposta malformata non viene interpretata come una lettura valida.
def read_value(reply, name):
    if not isinstance(reply, (tuple, list)) or len(reply) != 2:
        raise RuntimeError(f"{name}: risposta SDK inattesa: {reply!r}")
    err, result = reply
    check(err, name)
    if result is None:
        raise RuntimeError(f"{name}: valore assente")
    return result


# FUNZIONE: valid_joints
# Controllo di formato: servono esattamente sei numeri finiti.
# Non verifica collisioni, limiti meccanici o raggiungibilita.
# Nel programma e riutilizzato anche per pose [x,y,z,rx,ry,rz],
# perche anch'esse hanno sei componenti: il nome e quindi un po generico.
def valid_joints(q):
    return q is not None and np.asarray(q).shape == (6,) and np.isfinite(q).all()


# CLASSE: RobotIO
# Interfaccia del nostro programma con il controller Fairino.
# Eredita letture, FK e chiusura da acquire.RobotIO. Qui cambiamo
# il costruttore per non abilitare il robot appena ci colleghiamo.
# RPC e l'oggetto dell'SDK usato per scambiare richieste con il controller.
class RobotIO(ac.RobotIO):
    """Riusa i metodi originali, ma abilita il robot solo prima del moto."""
    # FUNZIONE: RobotIO.__init__
    # INGRESSO: ip e l'indirizzo di rete del controller.
    # Crea il client SDK ma non invia Mode o RobotEnable.
    # Non forza il flag is_connect: una connessione non valida deve fallire.
    def __init__(self, ip):
        # SDK del produttore: contiene la classe RPC e le API reali del controller.
        from fairino import Robot
        # Crea il client di rete. Non e una cinematica: e l’oggetto usato per parlare al controller.
        self.rpc = Robot.RPC(ip)         # [ORIGINALE] stessa connessione
        # Non chiamiamo il costruttore originale: abilitava subito il robot e
        # forzava is_connect=True. Qui una connessione difettosa deve fallire.

    # FUNZIONE: RobotIO.inverse_kin_near
    # Calcola i giunti per una posa della FLANGIA desiderata.
    # flange: [x,y,z,rx,ry,rz] in mm e gradi; previous: sei giunti in gradi.
    # Preferisce GetInverseKinRef per chiedere una soluzione vicina ai giunti
    # precedenti. Il fallback GetInverseKin non offre la stessa preferenza;
    # per questo il programma controlla anche gli eventuali salti articolari.
    # Non muove il robot: restituisce una soluzione cinematica.
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

    # FUNZIONE: RobotIO.enable_backstop
    # Configura l'anticollisione nativa del controller usando il YAML.
    # E una protezione reattiva basata sulle grandezze del robot, distinta
    # dal controllo geometrico predittivo del percorso. Ogni comando deve
    # restituire successo; non vengono azzerati automaticamente gli allarmi.
    def enable_backstop(self, cfg):
        c = cfg["collision"]
        if not c.get("native_backstop", True):
            raise RuntimeError("Questo STEP richiede collision.native_backstop=true")
        check(self.rpc.SetCollisionDetectionMethod(0, 0), "Metodo anticollisione")
        check(self.rpc.SetAnticollision(0, list(map(float, c.get("native_level", [5]*6))), 0),
              "Livello anticollisione")
        check(self.rpc.SetCollisionStrategy(int(c.get("native_strategy", 2))),
              "Strategia anticollisione")

    # FUNZIONE: RobotIO.assert_ready
    # Legge i codici di errore e lo stato di sicurezza.
    # Se e presente un allarme, genera un errore: non cerca di cancellarlo.
    # Questa funzione controlla lo stato; non abilita o muove il braccio.
    def assert_ready(self):
        # [NUOVO] Lettura allarmi, senza reset automatici.
        codes = read_value(self.rpc.GetRobotErrorCode(), "Allarmi robot")
        if not isinstance(codes, (list, tuple)) or any(x != 0 for x in codes):
            raise RuntimeError(f"Allarmi robot: {codes}")
        check(self.safety_code(), "Stato sicurezza")

    # FUNZIONE: RobotIO.done
    # Interroga il segnale di fine movimento del controller.
    # RISULTATO: True solo per stato 1; False per stato 0.
    # Da solo questo segnale non prova di aver raggiunto la posa richiesta:
    # prima dello scatto viene controllata anche la distanza dai giunti attesi.
    def done(self):
        # [NUOVO] 0=in movimento, 1=movimento concluso.
        state = read_value(self.rpc.GetRobotMotionDone(), "Fine movimento")
        if state not in (0, 1):
            raise RuntimeError(f"Stato movimento inatteso: {state}")
        return state == 1


# CLASSE: VideoFrames
# Contenitore condiviso tra chi legge la camera e chi usa le immagini.
# Non apre una seconda camera: conserva una copia dell'ultimo frame.
# Condition coordina l'accesso ai dati. Il contatore number permette
# di aspettare un'immagine NUOVA anziche riutilizzare quella precedente.
class VideoFrames:
    """[NUOVO] Passa copie delle immagini al bootstrap senza aprire due camere.

    La finestra e la RealSense sono gestite solo dal programma principale.
    Il calcolo del percorso e il robot lavorano separatamente: il video continua.
    """
    # FUNZIONE: VideoFrames.__init__
    # Prepara segnale stop, protezione dei dati e contatore.
    # frame=None significa che non e ancora arrivata un'immagine.
    # stop e condiviso: quando viene attivato, le attese devono terminare.
    def __init__(self, stop):
        self.stop = stop
        self.condition = threading.Condition()
        self.frame = None
        self.number = 0

    # FUNZIONE: VideoFrames.publish
    # Riceve un'immagine, ne conserva una copia e incrementa number.
    # La copia evita che il buffer originale della camera cambi dopo la lettura.
    # notify_all sveglia chi aspetta: non esegue il controllo robot.
    def publish(self, frame):
        # Accesso protetto al contenitore condiviso: impedisce aggiornamenti spezzati mentre un altro thread legge.
        with self.condition:
            # Copia il frame per non dipendere dal buffer che la camera potrebbe riutilizzare.
            self.frame = frame.copy()
            # Ogni publish incrementa il contatore; chi aspetta puo riconoscere un fotogramma successivo.
            self.number += 1
            # Avvisa le attese sulla Condition; potranno leggere i dati dopo il rilascio della protezione.
            self.condition.notify_all()

    # FUNZIONE: VideoFrames.grab
    # Aspetta immagini successive a quella disponibile alla chiamata.
    # n_avg=1 restituisce un frame; valori maggiori fanno una media dei frame.
    # La mappatura usa n_avg=1. Il timeout impedisce attese infinite del video.
    # Condition.wait rilascia temporaneamente il lock, permettendo a publish
    # di entrare e consegnare il frame che questa funzione sta aspettando.
    def grab(self, n_avg=1, timeout=10.0):
        images = []
        deadline = time.monotonic() + timeout
        # Accesso protetto al contenitore condiviso: impedisce aggiornamenti spezzati mentre un altro thread legge.
        with self.condition:
            # Memorizza il numero presente alla chiamata, cosi non restituisce subito lo stesso vecchio frame.
            previous = self.number
            while len(images) < max(1, n_avg):
                if self.stop.is_set():
                    raise RuntimeError("Interruzione richiesta")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Fotogrammi assenti durante la localizzazione")
                if self.number == previous:
                    # Attesa breve che RILASCIA il lock; publish puo quindi consegnare il nuovo frame.
                    self.condition.wait(min(0.1, remaining))
                    continue
                # Memorizza il numero presente alla chiamata, cosi non restituisce subito lo stesso vecchio frame.
                previous = self.number
                images.append(self.frame.copy())
        return np.mean(images, axis=0).astype(np.uint8)


# FUNZIONE: waypoint_options
# Recupera dizionario ArUco dal YAML normalizzato da acquire.
# Gli ID vengono dall'opzione --marker-ids, se presente, altrimenti dal YAML.
# Controlla che gli ID siano interi, distinti e non negativi.
# La tupla vuota di default non significa nessun target: significa usa il YAML.
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


# CLASSE: WaypointPhotos
# Gestisce soltanto cartelle, PNG e JSON. Non comanda il robot,
# non esegue tracking e non mette a fuoco: riceve il frame gia controllato.
# Una nuova istanza corrisponde a una nuova cartella di sessione.
class WaypointPhotos:
    """Salva il frame appena verificato, senza testo sovrapposto.

    Raccolta di foto e metadati: non sostituisce il dataset completo di
    run_acquire (pose TXT, calibrazioni e manifest della pipeline Vision).
    """
    # FUNZIONE: WaypointPhotos.__init__
    # root e la cartella principale, per default foto_waypoints.
    # Il percorso relativo parte dalla cartella da cui lanci il programma.
    # Crea una sottocartella run_... e scrive sessione.json con K, D, hand-eye
    # ed impostazioni. Non salva ancora fotografie. cfg viene conservato
    # come riferimento per leggere i risultati aggiornati prima di ogni scatto.
    def __init__(self, root, cfg, K, D, T_c2g):
        # Un percorso relativo e risolto dalla cartella corrente del terminale, non dalla posizione di questo Python.
        root = Path(root).expanduser().resolve()
        # Crea la cartella foto principale; non fallisce soltanto perche era gia presente.
        root.mkdir(parents=True, exist_ok=True)
        # Nuova sottocartella univoca run_...; questo codice non la cancella automaticamente a fine sessione.
        self.folder = Path(tempfile.mkdtemp(prefix="run_", dir=str(root)))
        # Prima del primo scatto il contatore e zero; il primo nome sara foto_0001.
        self.index = 0
        self.cfg = cfg
        metadata = {
            "description": "Foto dopo centratura: centri previsti dalla mappa, feedback visivo opzionale",
            # Tipo camera effettivamente selezionato nel YAML, ad esempio gc1020c.
            "camera_type": cfg["camera"].get("type"),
            # Origine della calibrazione: JSON o fabbrica, con percorso e dimensioni disponibili.
            "intrinsics_source": cfg.get("_intrinsics_source", {}),
            # K: intrinseche della camera; da array NumPy a liste JSON.
            "camera_matrix": np.asarray(K).tolist(),
            # D: coefficienti di distorsione della calibrazione.
            "distortion": np.asarray(D).tolist(),
            # Trasformazione camera->flangia; traslazione in mm, orientamento come matrice.
            "T_camera_to_gripper_mm": np.asarray(T_c2g).tolist(),
            # Convenzione delle conversioni degli angoli del controller, ad esempio ZYX.
            "euler_order": cfg["robot"]["euler_order"],
            # ID del target e dizionario usato per il riconoscimento iniziale.
            "waypoint_check": cfg["waypoint_check"],
            # Margine richiesto per il centro ArUco allo scatto, non per tutti i suoi vertici.
            "margin_px": MARGINE_MARKER_PX,
            # Campo legacy: registra 5, ma NON governa l’assestamento continuo; il produttore usa tre frame e tempi separati.
            "consecutive_frames": FRAME_CONSECUTIVI_OK,
            # Tre centri della MAPPA nei margini allo scatto, anche se sfocati.
            "minimum_predicted_markers_at_photo": MIN_ARUCO_VISIBILI,
            # Zero: il trasferimento non richiede un numero minimo di marker inquadrati.
            "minimum_markers_during_motion": 0,
            # Specifica se il vincolo e geometrico e in quale fase del programma e richiesto.
            "visibility_criterion": "target_during_motion_three_predicted_centers_at_photo",
        }
        (self.folder / "sessione.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

    # FUNZIONE: WaypointPhotos.save
    # Salva un frame PNG e il JSON con lo stesso numero progressivo.
    # name e la tappa; actual_q sono i giunti letti; requested_q sono i giunti
    # nominali della tappa, che possono differire per il puntamento corretto.
    # Il metodo si fida delle verifiche fatte prima: non le ripete qui.
    # Le due scritture non sono atomiche: se fallisce il JSON, il PNG puo gia esistere.
    def save(self, frame, name, actual_q, requested_q):
        # Numero del nuovo scatto, aggiornato definitivamente solo dopo entrambe le scritture.
        index = self.index + 1
        # Formato 04d: almeno quattro cifre con zeri iniziali, per esempio foto_0007.
        stem = f"foto_{index:04d}"
        # Percorso PNG di questa foto; il JSON verra salvato accanto con lo stesso nome di base.
        image_path = self.folder / f"{stem}.png"
        # Percorso del file di informazioni della singola foto, diverso da sessione.json.
        metadata_path = self.folder / f"{stem}.json"
        # Comprimi il frame in PNG in memoria; non aggiunge il testo visto nella finestra e non mette a fuoco.
        ok, encoded = cv.imencode(".png", frame)
        if not ok:
            raise RuntimeError(f"Codifica foto fallita: {name}")
        metadata = json.dumps({
            # Nome della tappa a cui appartiene questa fotografia.
            "waypoint": name,
            # Solo il nome del PNG, nella stessa cartella del JSON.
            "image": image_path.name,
            # Giunti effettivamente letti alla tappa, in gradi.
            "actual_joints_deg": list(actual_q),
            # Giunti nominali della tappa, prima delle correzioni effettive di puntamento.
            "requested_joints_deg": list(requested_q),
            # Ordine [larghezza,altezza], mentre frame.shape usa [altezza,larghezza,canali].
            "image_size_px": [int(frame.shape[1]), int(frame.shape[0])],
            # False: non ridecodifica gli ID ad ogni scatto. Non significa marker assente.
            "aruco_verified": False,
            # False: questa versione non segue i quattro vertici durante il moto.
            "tracked_corners_verified": False,
            # Dipende da visual_feedback_fresh; in SOLO MAPPA resta False anche se la foto viene ammessa.
            "tracked_centers_verified": bool(self.cfg.get("_last_visual", {}).get("visual_feedback_fresh", False)),
            # True: il criterio della foto e la proiezione della mappa, non la leggibilita del codice ArUco.
            "predicted_centers_checked": True,
            # Indica se la posa usata viene dalla sola FK o da FK con ultima correzione visiva.
            "view_source": self.cfg.get("_last_visual", {}).get("pose_source", "robot_fk_only"),
            # Copia i dettagli aggiornati dal worker prima di chiamare save.
            "visual_feedback": self.cfg.get("_last_visual", {}),
        }, indent=2, allow_nan=False)
        # Nomi esclusivi: non sovrascrive foto precedenti.
        # x impedisce sovrascritture; b scrive byte. with chiude il file anche uscendo per un errore.
        with image_path.open("xb") as out:
            # Scrive i byte del PNG sul disco: e qui che la fotografia viene effettivamente salvata.
            out.write(encoded.tobytes())
        # Crea il JSON nuovo con testo UTF-8; un file gia esistente causa errore anziche essere sovrascritto.
        with metadata_path.open("x", encoding="utf-8") as out:
            out.write(metadata)
        # Registra il nuovo numero soltanto dopo che PNG e JSON sono stati scritti.
        self.index = index
        # Permette al chiamante di mostrare il percorso completo della foto nel terminale.
        return image_path


# FUNZIONE: require_guard
# Richiede che il modello collisioni sia attivo, caricato e validato.
# Se manca, il programma rifiuta il movimento. Questo evita di accettare
# la risposta permissiva del guard quando il modello non e disponibile.
# La validazione software non certifica la geometria fisica dell'intera cella.
def require_guard(guard):
    if not (guard.enable and guard.available and guard.validated):
        raise RuntimeError("Controllo collisioni assente o modello non validato: niente moto")


# FUNZIONE: safe_path
# Controlla configurazione di arrivo e percorso tra q0 e q1.
# I sei valori sono giunti in gradi, non coordinate della camera.
# Il guard campiona il percorso con il proprio modello geometrico;
# non e una garanzia matematica su ogni istante del movimento reale.
def safe_path(guard, q0, q1):
    require_guard(guard)
    ok, reasons = guard.config_safe(q1)
    if not ok:
        raise RuntimeError(f"Posizione rifiutata: {reasons}")
    ok, reasons = guard.path_safe(q0, q1)
    if not ok:
        raise RuntimeError(f"Trasferimento rifiutato: {reasons}")


# FUNZIONE: look_at
# Costruisce una posa CAMERA->BASE che guarda center dalla position.
# position e center sono XYZ in mm nella base del robot.
# reference_y e la direzione Y della camera iniziale: guida il roll.
# Z della camera punta al target; X e Y vengono ricostruiti perpendicolari.
# Restituisce una matrice 4x4; NON muove il robot e NON usa immagini.
def look_at(position, center, reference_y):
    # Vettore camera->target in base: sottrae gli XYZ della camera dagli XYZ del target.
    z = np.asarray(center)-position
    # Soglia geometrica scelta nel codice, 50 mm; non deriva dalla camera e non sostituisce gli ingombri collisioni.
    if np.linalg.norm(z) < 50:
        raise RuntimeError("Camera a meno di 50 mm dal centro")
    # Normalizza: mantiene la direzione verso il target e porta la lunghezza a 1.
    z = z/np.linalg.norm(z)
    # Prodotto vettoriale: ottiene l’asse X perpendicolare a riferimento Y e nuova direzione ottica Z.
    x = np.cross(reference_y, z)
    # Se riferimento Y e Z sono quasi paralleli, la scelta dell’orientamento diventa instabile.
    if np.linalg.norm(x) < 0.1:
        raise RuntimeError("Direzione di puntamento singolare")
    # Rende unitario l’asse X; /= significa riassegna x diviso la sua lunghezza.
    x /= np.linalg.norm(x)
    # Prepara una trasformazione 4x4: identita, poi orientamento nelle prime tre colonne e posizione nell’ultima.
    T = np.eye(4)
    # Colonne [X, Y, Z] della camera nella base. Y=cross(Z,X) rende perpendicolari tutti e tre gli assi.
    T[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    # L’ultima colonna riceve XYZ camera; la posizione non viene modificata da look_at.
    T[:3, 3] = position
    return T


# FUNZIONE: persistent_target
# Crea la mappa iniziale del target statico.
# A robot fermo legge dieci frame e richiede osservazioni dei 6 ID.
# Usa i vertici iniziali per stimare XYZ nella camera, poi li porta nella base.
# Salva anche il centro XYZ per ID. Il centroid del target e la media dei 24
# vertici, equivalente alla media dei 6 centri: non misura il provino.
# Spostamenti fisici non osservati non sono rilevati dalla firma del file.
def persistent_target(robot, bridge, cfg, H, K, D):
    """Mappa ID -> quattro vertici in base/mm; acquisita una volta da fermo."""
    # Raccoglie i dati che devono essere coerenti tra creazione e riuso della mappa.
    identity = {"robot": cfg["robot"], "K": K.tolist(), "D": D.tolist(),
                "handeye": H.tolist(), "options": cfg["waypoint_check"],
                "marker_size_mm": cfg["_marker_mm"]
                }
    # Impronta delle impostazioni: confronta la configurazione, non rileva spostamenti fisici.
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # Percorso assoluto del JSON; un nome relativo parte dalla directory da cui avvii il comando.
    path = Path(cfg["_map_file"]).expanduser().resolve()
    # ID e dizionario arrivano dal YAML normalizzato, oppure dall’override degli ID.
    options = cfg["waypoint_check"]
    # Prepara il dizionario solo per la localizzazione iniziale, non per il conteggio geometrico futuro.
    dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, options["dictionary"]))
    # Seleziona l’API compatibile con la versione OpenCV installata.
    detect = (cv.aruco.ArucoDetector(dictionary).detectMarkers if hasattr(cv.aruco, "ArucoDetector")
                else lambda im: cv.aruco.detectMarkers(im, dictionary))
    # Lato fisico del quadrato in millimetri: una misura errata falsa la scala della mappa.
    size = cfg["_marker_mm"]
    # Quattro vertici locali del marker attorno al suo centro (0,0,0); sono XYZ in mm, non pixel.
    obj = np.array([[-size/2,size/2,0],[size/2,size/2,0],
                    [size/2,-size/2,0],[-size/2,-size/2,0]], dtype=np.float64)
    # Legge sei giunti effettivi e li conserva come riferimento per verificare che il robot resti fermo.
    q0 = np.asarray(robot.get_joints(), float)
    # Un elenco di osservazioni per ID: ciascuna osservazione conterra quattro vertici XYZ nella base.
    samples = {str(i): [] for i in options["marker_ids"]}
    # Usa dieci immagini distinte per ridurre l’effetto delle oscillazioni della stima iniziale.
    for _ in range(10):
        # Non accetta la misura se il controller indica moto o un giunto cambia oltre 0,05 gradi.
        if not robot.done() or np.max(np.abs(np.asarray(robot.get_joints())-q0)) > 0.05:
            raise RuntimeError("Il robot deve restare fermo durante la mappatura")
        # Aspetta un frame successivo alla chiamata; n_avg=1 evita medie tra immagini diverse.
        frame = bridge.grab(n_avg=1)
        # Solo qui nella mappatura individua i quattro vertici in pixel e i rispettivi ID.
        corners, ids, _ = detect(frame)
        # Se non ci sono ID usa []; altrimenti appiattisce l’array OpenCV in una lista di numeri.
        labels = [] if ids is None else ids.reshape(-1).tolist()
        # Un ID ripetuto nello stesso frame renderebbe ambigua l’associazione con la mappa.
        if len(labels) != len(set(labels)):
            raise RuntimeError("ID ArUco duplicati")
        # Forward Kinematics fornisce FLANGIA->BASE; moltiplicare per H (CAMERA->FLANGIA) produce CAMERA->BASE.
        T = ac.pose_to_T(robot.forward_kin(q0.tolist())) @ H
        # Abbina i quattro punti immagine c al loro ID i, senza perdere la corrispondenza.
        for c, i in zip(corners, labels):
            # Ignora marker estranei: non devono cambiare la geometria del target.
            if str(i) not in samples:
                continue
            # Matrice 4x2: ogni riga e [u,v] di un vertice in pixel.
            pixels = np.asarray(c, float).reshape(4, 2)
            # PnP iniziale del quadrato: confronta forma XYZ nota e quattro vertici osservati con K e D.
            ok, r, t = cv.solvePnP(obj, pixels, K, D, flags=cv.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                raise RuntimeError("PnP fallito")
            # P_camera = R * P_marker + t: trasforma i vertici locali in XYZ rispetto alla camera.
            camera_points = (cv.Rodrigues(r)[0] @ obj.T).T + t.reshape(3)
            # Riproduce in pixel la misura prevista dalla posa appena calcolata per confrontarla con i vertici veri.
            reproj = cv.projectPoints(obj, r, t, K, D)[0].reshape(4, 2)
            # Profondità positiva e riproiezione entro 1,5 pixel: controlli di coerenza, non prova di esattezza assoluta.
            #if np.any(camera_points[:, 2] <= 0) or np.max(np.linalg.norm(reproj-pixels, axis=1)) > 1.5:
             #   raise RuntimeError("Localizzazione marker non affidabile")
            # Porta P_camera nella BASE con T e conserva questa osservazione; la traslazione e in mm.
            samples[str(i)].append((T[:3, :3] @ camera_points.T).T + T[:3, 3])
    # Non accetta la misura se il controller indica moto o un giunto cambia oltre 0,05 gradi.
    if not robot.done() or np.max(np.abs(np.asarray(robot.get_joints())-q0)) > 0.05:
        raise RuntimeError("Robot mosso durante mappatura")
    markers = {}
    # Verifica separatamente ogni ID; la prima mappa deve contenere tutti i sei marker configurati.
    for i, values in samples.items():
        # Un marker deve comparire con stima valida in almeno otto dei dieci frame iniziali.
        if len(values) < 8:
            raise RuntimeError(f"Marker {i}: servono almeno 8 osservazioni su 10")
        # Media per vertice: rimangono quattro punti XYZ, non si mescolano i quattro angoli tra loro.
        mean = np.mean(values, axis=0)
        # Controlla la distanza 3D di ogni osservazione dalla sua media; oltre 3 mm rifiuta la mappa.
        #if np.max(np.linalg.norm(np.asarray(values)-mean, axis=2)) > 3:
        #    raise RuntimeError(f"Marker {i}: stima instabile oltre 3 mm")
        # Salva i quattro vertici medi dell’ID i in un formato serializzabile nel JSON.
        markers[i] = mean.tolist()
    # Include vertici originali e centri XYZ: i vertici rimangono per la calibrazione, non vengono seguiti nel moto.
    data = {"schema": 1, "fingerprint": fingerprint, "calibration": identity,
            "units": "mm", "frame": "robot_base", "markers": markers,
            "centers_base_mm": {i:np.asarray(p,float).mean(axis=0).tolist() for i,p in markers.items()}}
    # Crea, se non esiste, la cartella del file mappa, comprese le eventuali cartelle superiori.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Modalita x: crea un nuovo file e rifiuta di sovrascriverne uno gia presente.
    with path.open("x") as f:
        # Scrive la mappa leggibile; NaN e infinito sono rifiutati per non salvare coordinate non valide.
        json.dump(data, f, indent=2, allow_nan=False)

    if set(data["markers"]) != set(map(str, cfg["waypoint_check"]["marker_ids"])):
        raise RuntimeError("ID nella mappa non corrispondenti")
    # Array atteso 6x4x3: sei ID, quattro vertici ciascuno, tre coordinate nella base.
    points = np.array(list(data["markers"].values()), float)
    if points.shape != (len(data["markers"]), 4, 3) or not np.isfinite(points).all():
        raise RuntimeError("Coordinate mappa non valide")
    # Per ciascun ID fa la media dei quattro vertici 3D gia noti: nessuna nuova misura della camera.
    derived_centers = {i:np.asarray(p,float).mean(axis=0).tolist()
                       for i,p in data["markers"].items()}
    # Se i centri sono gia memorizzati, verifica che non siano stati modificati incoerentemente.
    if "centers_base_mm" in data:
        if set(data["centers_base_mm"]) != set(derived_centers):
            raise RuntimeError("Centri della mappa incompatibili con gli ID")
        for i,c in derived_centers.items():
            stored=np.asarray(data["centers_base_mm"][i],float)
            if stored.shape!=(3,) or not np.isfinite(stored).all() or not np.allclose(stored,c,atol=1e-6,rtol=0):
                raise RuntimeError("Centri della mappa incoerenti con la calibrazione iniziale")
    # Rende disponibili i centri anche caricando una vecchia mappa che conteneva solo i vertici.
    data["centers_base_mm"] = derived_centers
    # Conserva in memoria una copia della mappa da usare per tracking, previsioni e metadati delle foto.
    cfg["_map_snapshot"] = data
    # Il target e la media dei 24 vertici, equivalente alla media dei sei centri: non il centro misurato del provino.
    return {"centroid": points.reshape(-1, 3).mean(axis=0), "corners_base": points.reshape(-1, 3)}


# FUNZIONE: plan_joints
# Prepara e controlla il percorso senza muovere il robot.
# Legge la posa iniziale, converte le POSES articolari in posizioni camera,
# poi ricostruisce gli orientamenti verso il target con look_at.
# Inserisce nodi a distanza camera <=2 mm e usa IK vicina al nodo precedente.
# Controlla IK/FK, salti e collisioni. photo=True identifica le tappe di scatto.
# I tre centri previsti sono richiesti solo alle tappe foto, non ai nodi intermedi.
# I raccordi continui saranno costruiti dopo: questi nodi non sono soste.
def plan_joints(robot, cfg, target, T_c2g, K, D, stop):
    # Legge l’inizio reale: giunti attuali e posa CAMERA->BASE ottenuta da FK e hand-eye.
    T_start, q_start = ac.camera_T_base(robot, T_c2g)
    if T_start is None or not valid_joints(q_start):
        raise RuntimeError("Posa iniziale non valida")
    # Condivide i dati geometrici con le funzioni che proiettano il target e i centri.
    cfg.update(_target=target, _K=K, _D=D, _handeye=T_c2g)
    # La partenza e una tappa foto: gia nel piano richiede tre centri previsti, senza muovere il robot.
    geometric_view(T_start, cfg, require_markers=True)
    # Il primo nodo descrive la configurazione attuale e porta photo=True per la foto iniziale.
    poses = [{"name": "iniziale", "joints": list(q_start),
              "flange": robot.forward_kin(q_start), "cam_T_base": T_start, "photo": True}]
    # Dalle sei articolazioni di ogni POSES estrae SOLO la posizione XYZ della CAMERA; ricalcolera l’orientamento.
    positions = [(f"P{i}", (ac.pose_to_T(robot.forward_kin(q)) @ T_c2g)[:3, 3])
                 for i, q in enumerate(POSES, 1)]
    # Aggiunge una destinazione finale all’XYZ iniziale della camera, non necessariamente agli stessi giunti.
    
    alignment_name = "allineamento iniziale"
    aligned_T = look_at(
        T_start[:3, 3], target["centroid"], T_start[:3, 1]
    )
    alignment_rotvec = cv.Rodrigues(
        T_start[:3, :3].T @ aligned_T[:3, :3]
    )[0].reshape(3)

    positions.insert(
        0, (alignment_name, T_start[:3, 3].copy())
    )
    if RITORNO_INIZIALE:
        positions.append(("ritorno iniziale", T_start[:3, 3]))
    # Memoria dei giunti del nodo precedente, necessaria per scegliere una IK vicina e limitare salti.
    previous = np.array(q_start, float)
    # XYZ iniziale della camera, cioe origine del primo trasferimento cartesiano nominale.
    current = T_start[:3, 3].copy()
    # Costruisce le tappe nell’ordine P1, P2, ... e ritorno, senza riordinarle automaticamente.
    for name, end in positions:
        # Divide la distanza camera per 2 mm e arrotonda in alto: nodi intermedi distanti al massimo 2 mm.
        count = max(1, int(math.ceil(np.linalg.norm(end-current)/2)))
        
        aligning = name == alignment_name
        if aligning:
            count = max(
                count,
                int(math.ceil(
                    math.degrees(np.linalg.norm(alignment_rotvec)) / 0.25
                ))
            )

        # Fissa l’XYZ iniziale di questo trasferimento, indipendentemente dal nodo intermedio corrente.
        start = current.copy()
        # Costruisce un nodo alla volta, fino al nodo finale che coincide con end.
        for j in range(1, count+1):
            if stop.is_set():
                raise RuntimeError("Interruzione richiesta")
            # Interpolazione XYZ lungo la retta e Z camera verso target; T_start[:3,1] fornisce l’Y iniziale come guida.
            T = look_at(start+(end-start)*j/count, target["centroid"], T_start[:3, 1])

            if aligning:
                T = T_start.copy()
                T[:3, :3] = (
                    T_start[:3, :3]
                    @ cv.Rodrigues(alignment_rotvec * (j / count))[0]
                )
            
            # Passa dalla posa CAMERA desiderata alla posa FLANGIA usando l’inversa della hand-eye.
            flange = ac.gripper_pose_from_camera_T(T, T_c2g)
            # Calcolo IK del nodo, senza movimento; cerca una soluzione prossima ai giunti precedenti.
            q = robot.inverse_kin_near(flange, previous)
            # Rifiuta piu di 1 grado su un giunto fra nodi vicini; non limita a 1 grado tutto il trasferimento.
                      
            print("Variazione giunti [gradi]:", np.round(delta, 3), flush=True)
            if np.max(np.abs(np.array(q)-previous)) > 2:
                raise RuntimeError("Passo articolare oltre 2 gradi: percorso rifiutato")
            # Ricalcola la posa tramite FK dei giunti proposti. actual qui NON significa fisicamente raggiunta.
            actual = ac.pose_to_T(robot.forward_kin(q)) @ T_c2g
            # Modulo del vettore di rotazione relativo: differenza angolare fra orientamento desiderato e ricostruito, in radianti.
            rotation_error = np.linalg.norm(cv.Rodrigues(actual[:3,:3].T @ T[:3,:3])[0])
            # Confronto tra due calcoli IK/FK: soglie 1 mm e 0,5 gradi in questa copia, non precisione misurata sul robot.
            if np.linalg.norm(actual[:3,3]-T[:3,3]) > 1 or rotation_error > math.radians(0.5):
                raise RuntimeError("IK/FK discordanti")
            # Tre centri richiesti solo al nodo fotografico finale; nei nodi intermedi basta il puntamento del target.
            geometric_view(actual, cfg, require_markers=(j==count))
            # Verifica anche la proiezione in punti intermedi del tratto articolare.
            # Controlla anche cinque configurazioni tra i due nodi, non soltanto la destinazione.
            for alpha in np.linspace(0, 1, 5):
                # Interpolazione dei sei giunti per la verifica intermedia: non e un comando inviato al robot.
                qi = previous+(np.array(q)-previous)*alpha
                geometric_view(ac.pose_to_T(robot.forward_kin(qi.tolist())) @ T_c2g, cfg)
            # Salva il nodo; solo j==count riceve photo=True, gli altri non sono soste fotografiche.
            poses.append({"name": name if j==count else f"{name}/passo{j}",
                          "joints": list(q), "flange": list(flange),
                          "cam_T_base": actual, "photo": j==count and not aligning})
            # Il nodo appena approvato diventa il riferimento articolare per la prossima IK.
            previous = np.array(q)
        # Finito un trasferimento, la sua destinazione diventa la partenza del successivo.
        current = end
    # Carica il modulo geometrico originale del progetto, non incluso nei due file nuovi.
    from collision_guard import CollisionGuard
    # Passa FK nativa e hand-eye al modello per posizionare braccio e camera coerentemente.
    guard = CollisionGuard(cfg, robot, ac.pose_to_T, robot.forward_kin,
                           T_c2g=T_c2g, euler_order=ac._EULER)
    # Imposta il centro usato dal volume di esclusione del campione nel guard.
    guard.set_target(target["centroid"])
    # Confronta la cinematica del modello con il controller; non valida automaticamente ogni ostacolo fisico.
    guard.validate([p["joints"] for p in poses])
    require_guard(guard)
    # Abbina ogni nodo a quello immediatamente successivo, includendo il primo trasferimento.
    for a, b in zip(poses, poses[1:]):
        safe_path(guard, a["joints"], b["joints"])
    # Consegna piano, guard e giunti iniziali al coordinatore: nessuna abilitazione eseguita qui.
    return poses, guard, list(q_start)


# FUNZIONE: interrupted_wait
# Attende un intervallo in secondi, ma termina prima se stop viene attivato.
# Il risultato dell'attesa segnala l'interruzione e viene trasformato in errore.
# Non e un comando di arresto inviato al robot.
def interrupted_wait(stop, seconds):
    if stop.wait(seconds):
        raise RuntimeError("Interruzione richiesta")


# CLASSE: VisualTracker
# Feedback opzionale dalle immagini: segue PATCH ATTORNO AI CENTRI.
# Questa NON e la vecchia classe con LK sui quattro vertici.
# I vertici sono usati solo all'avvio per associare ID e centro iniziale.
# Durante update si confrontano piccole immagini attorno ai centri previsti.
# La PnP usa centri XYZ e centri in pixel, non dodici o ventiquattro vertici.
# Se questo tracker perde i centri, FastFrames puo continuare con la mappa.
class VisualTracker:
    """Vertici solo all'avvio. Poi segue patch centrate sui marker, non vertici.
    La mappa dei centri XYZ resta fissa. Le posizioni 2D sono misure ottenute
    confrontando piccole immagini, mai aggiornamenti dei quattro vertici.
    Una previsione geometrica non viene passata alla PnP come osservazione."""
    
    SEARCH = 35
    MIN_SCORE = 0.80

    # FUNZIONE: VisualTracker.initial_center
    # INGRESSO: quattro vertici in pixel, nell'ordine del quadrilatero.
    # Calcola l'intersezione delle diagonali: nel modello proiettivo e il
    # centro fisico del quadrato nell'immagine. Non usa la media di tre vertici.
    # Serve solo all'inizializzazione delle patch, non a ogni frame.
    # Distorsione residua e rumore restano limiti di questa inizializzazione.
    @staticmethod
    # Questa operazione di calcolo non dipende da niente di ciò che il tracciatore ha salvato in memoria. 
    # Definirlo statico garantisce che questo metodo sia un puro "calcolatore di supporto" isolato e sicuro, 
    # senza effetti collaterali sul resto del programma.
    def initial_center(corners):
        # Intersezione diagonali: proiezione del centro fisico del quadrato.
        p = np.asarray(corners,float).reshape(4,2)
        # Sistema 2x2 per trovare l’incrocio tra le diagonali 0->2 e 1->3 del quadrilatero iniziale.
        A = np.column_stack((p[2]-p[0], -(p[3]-p[1])))
        if abs(np.linalg.det(A)) < 1e-6:
            raise RuntimeError('Marker iniziale degenere')
        # Risolve le frazioni lungo le due diagonali: entrambe devono cadere tra 0 e 1.
        a,b = np.linalg.solve(A,p[1]-p[0])
        if not (0<a<1 and 0<b<1):
            raise RuntimeError('Centro iniziale fuori dal marker')
        # Restituisce il punto d’incrocio in pixel, non coordinate nella base del robot.
        return p[0]+a*(p[2]-p[0])

    # FUNZIONE: VisualTracker.__init__
    # Carica i centri XYZ dalla mappa e identifica gli ID nell'immagine iniziale.
    # Per ogni ID costruisce una patch centrata sul marker e ne controlla il contrasto.
    # La patch deve contenere dettagli distinguibili: il solo pixel centrale
    # potrebbe essere uniforme e non essere seguibile. Memorizza immagine locale,
    # dimensione e ID, poi esegue la prima stima con i centri osservati.
    # Se non riesce, FastFrames ammette comunque l'avvio SOLO MAPPA.
    def __init__(self, frame, cfg, initial_camera_T_base):
        self.cfg = cfg
        self.map = {int(i):np.asarray(c,float) for i,c in cfg['_map_snapshot']['centers_base_mm'].items()}
        self.previous = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        detector = ac.make_detector(cfg['waypoint_check']['dictionary'])
        corners,ids = ac.detect_markers(detector,self.previous)
        # Se non ci sono ID usa []; altrimenti appiattisce l’array OpenCV in una lista di numeri.
        labels = [] if ids is None else np.asarray(ids).reshape(-1).tolist()
        # Un ID ripetuto nello stesso frame renderebbe ambigua l’associazione con la mappa.
        if len(labels)!=len(set(labels)):
            raise RuntimeError('ID duplicati all’avvio')
        self.centers={}; self.templates={}; self.sizes={}; self.scores={}
        for c,i in zip(corners,labels):
            i=int(i)
            if i not in self.map:
                continue
            p=np.asarray(c,float).reshape(4,2)
            center=self.initial_center(p)
            # Dimensione della patch iniziale: proporzionale al marker, limitata fra 15 e 81 pixel prima dell’arrotondamento dispari.
            side=min(81,max(15,int(np.max(np.ptp(p,axis=0))*.8)))
            # Rende dispari il lato per avere un centro ben definito del ritaglio.
            side=side if side%2 else side+1
            # Ritaglia l’immagine locale intorno al centro; non estrae quattro vertici da seguire.
            patch=self.patch(self.previous,center,side)
            # La patch deve entrare nel frame e avere contrasto; una zona uniforme non e affidabile per il confronto.
            if patch is not None and np.std(patch)>=5:
                self.centers[i]=center
                # Memorizza i pixel della patch associata all’ID; non e la coordinata XYZ del marker.
                self.templates[i]=patch
                # Conserva il lato della patch per le ricerche successive dello stesso ID.
                self.sizes[i]=side
                self.scores[i]=1.
        # Il tracker VISIVO richiede tre misure; FastFrames gestisce questo errore continuando con la mappa.
        if len(self.centers)<3:
            raise RuntimeError('Avvio: almeno 3 centri con patch distinguibili e ID riconosciuti')
        # Inverte CAMERA->BASE per ottenere BASE->CAMERA: il riferimento richiesto dalla proiezione/PnP.
        B=np.linalg.inv(initial_camera_T_base)
        # Converte la matrice di rotazione nel vettore di rotazione richiesto da OpenCV.
        self.rvec=cv.Rodrigues(B[:3,:3])[0]; self.tvec=B[:3,3].reshape(3,1)
        # Verifica la geometria dei centri osservati e ne ricava la posa camera, senza inviare comandi robot.
        self.estimate()

    # FUNZIONE: VisualTracker.patch
    # Estrae un quadrato d'immagine di lato side, centrato su center in pixel.
    # Il centro puo avere coordinate non intere: getRectSubPix interpola i pixel.
    # Se il quadrato non entra interamente nel frame restituisce None.
    # Questo limita il feedback della patch, NON il conteggio geometrico del marker.
    @staticmethod
    def patch(gray, center, side):
        h,w=gray.shape; half=side//2
        u,v=np.asarray(center,float)
        # Un ritaglio che esce dall’immagine non e valido per il confronto, ma il centro puo restare geometricamente previsto.
        if not np.isfinite([u,v]).all() or u-half<0 or v-half<0 or u+half>=w or v+half>=h:
            return None
        # Ritaglio sub-pixel: campiona anche centri non interi e restituisce una copia indipendente.
        return cv.getRectSubPix(gray,(side,side),(float(u),float(v))).copy()

    # FUNZIONE: VisualTracker.match
    # Cerca una piccola immagine template vicino al centro predicted.
    # Non decodifica l'ID e non cerca i quattro vertici. Confronta la luminosita
    # normalizzata delle patch; il picco migliore deve essere forte e distinguibile
    # da altri picchi. Restituisce (centro trovato, punteggio), oppure None.
    # Non garantisce riconoscimento sotto ogni sfocatura o cambio di prospettiva.
    # Nel vecchio codice: LK segue i vertici POI PnP stima la posa della camera
    # Nel nuovo codice: Template matching segue le zone attorno ai centri POI Ottiene i centri in pixel
    # POI PnP confronta quei centri con i centri XYZ della mappa POI Stima la posa della camera 
    @classmethod
    def match(cls, gray, template, predicted):
        # Cerca una patch simile intorno al CENTRO previsto. Nessuna ricerca vertici.
        h,w=gray.shape; side=template.shape[0]; half=side//2
        u,v=np.asarray(predicted,float)
        x0=max(0,int(np.floor(u-half-cls.SEARCH)))
        y0=max(0,int(np.floor(v-half-cls.SEARCH)))
        x1=min(w,int(np.ceil(u+half+cls.SEARCH+1)))
        y1=min(h,int(np.ceil(v+half+cls.SEARCH+1)))
        # Finestra di ricerca locale attorno al centro previsto: non analizza indiscriminatamente tutta l’immagine.
        region=gray[y0:y1,x0:x1]
        # La regione deve essere almeno grande quanto la patch da confrontare.
        if min(region.shape)<side:
            return None
        # Per ogni posizione candidata confronta la patch normalizzando la luminosita e produce una mappa di punteggi.
        scores=cv.matchTemplate(region,template,cv.TM_CCOEFF_NORMED)
        if not np.isfinite(scores).all():
            return None
        # peak e il punteggio migliore; where e l’angolo del ritaglio corrispondente nella regione di ricerca.
        _,peak,_,where=cv.minMaxLoc(scores)
        # Sotto 0,80 rifiuta la misura visiva; il centro previsto dalla mappa rimane comunque disponibile.
        if peak<cls.MIN_SCORE:
            return None
        # Rifiuta corrispondenze ambigue, con un altro picco quasi equivalente.
        # Prepara la ricerca di un secondo picco, per evitare di accettare due zone quasi indistinguibili.
        other=scores.copy(); x,y=where
        # Esclude il vicinato del miglior picco, che appartiene alla stessa corrispondenza locale.
        other[max(0,y-4):y+5,max(0,x-4):x+5]=-1
        # Richiede un vantaggio di almeno 0,04 sul secondo picco: altrimenti la corrispondenza e ambigua.
        if other.size and peak-float(other.max())<.04:
            return None
        # Dall’angolo del ritaglio torna al CENTRO in coordinate pixel dell’immagine completa, piu il punteggio.
        return np.array([x0+x+half,y0+y+half],float),float(peak)

    # FUNZIONE: VisualTracker.update
    # Riceve il nuovo frame, prevede le posizioni dei centri dalla mappa
    # e dalla posa corrente, e cerca le patch soltanto in quelle zone.
    # Se il centro era osservato prima, controlla anche la corrispondenza inversa.
    # Le posizioni trovate sono misure: le previsioni non diventano misure inventate.
    # Stima la posa dai centri accettati e aggiorna le patch dopo il controllo PnP.
    # Non contiene calcOpticalFlowPyrLK: non sta seguendo i vertici.
    def update(self, frame):
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        if gray.shape!=self.previous.shape:
            raise RuntimeError('Risoluzione video cambiata')
        # Calcola dove dovrebbero apparire i centri XYZ noti: questa e una previsione, non una lettura del frame.
        predicted=cv.projectPoints(np.array([self.map[i] for i in sorted(self.templates)]),
            self.rvec,self.tvec,self.cfg['_K'],self.cfg['_D'])[0].reshape(-1,2)
        found={}; scores={}; patches={}
        # Cerca ogni patch in prossimita del proprio centro previsto, mantenendo l’ID inizialmente associato.
        for i,guess in zip(sorted(self.templates),predicted):
            # Misura la posizione del centro tramite la somiglianza della patch, non tramite vertici o nuova decodifica ID.
            matched=self.match(gray,self.templates[i],guess)
            # Nessuna misura accettata per questo ID in questo frame; non inventa una corrispondenza.
            if matched is None:
                continue
            center,score=matched
            # Ritaglia l’immagine locale intorno al centro; non estrae quattro vertici da seguire.
            patch=self.patch(gray,center,self.sizes[i])
            if patch is None or np.std(patch)<5:
                continue
            # Verifica inversa della patch, quando il centro era misurato prima.
            if i in self.centers:
                # Verifica inversa: la patch attuale deve ritrovare il centro nella vecchia immagine.
                backward=self.match(self.previous,patch,self.centers[i])
                # Se il ritorno differisce piu di 1,5 pixel, rifiuta la nuova misura del centro.
                if backward is None or np.linalg.norm(backward[0]-self.centers[i])>1.5:
                    continue
            found[i]=center; scores[i]=score; patches[i]=patch
        # Conserva solo i centri misurati in questo frame, separatamente dalla mappa XYZ che resta fissa.
        self.centers=found; self.scores=scores
        # Il frame attuale diventa quello precedente per la verifica inversa al prossimo aggiornamento.
        self.previous=gray
        # Verifica la geometria dei centri osservati e ne ricava la posa camera, senza inviare comandi robot.
        self.estimate()
        # Aggiorna solo le patch il cui centro ha superato anche la verifica PnP.
        for i in self.centers:
            # Memorizza i pixel della patch associata all’ID; non e la coordinata XYZ del marker.
            self.templates[i]=patches[i]

    # FUNZIONE: VisualTracker.estimate
    # Stima base->camera usando i centri XYZ noti e i centri 2D misurati.
    # Richiede tre centri per ottenere un aggiornamento VISIVO, non per far
    # continuare il moto: la mancanza viene gestita da FastFrames in SOLO MAPPA.
    # Controlla margini, non allineamento, riproiezione e salti della soluzione.
    # Con tre centri la posa puo essere ambigua: usa una stima iniziale e limiti,
    # ma questo non rende il problema globalmente univoco.
    def estimate(self):
        h,w=self.previous.shape; margin=MARGINE_MARKER_PX
        self.centers={i:c for i,c in self.centers.items()
                      if np.isfinite(c).all() and margin<=c[0]<w-margin and margin<=c[1]<h-margin}
        # Il tracker VISIVO richiede tre misure; FastFrames gestisce questo errore continuando con la mappa.
        if len(self.centers)<3:
            raise RuntimeError('Meno di 3 centri misurati affidabili: nessun vertice viene cercato')
        # Allinea il criterio fisico: centro XYZ della mappa e centro 2D della patch.
        # Con tre punti la soluzione può essere ambigua: usa la posa precedente
        # e controlli di variazione, non promette una stima univoca globale.
        ids=sorted(self.centers)
        objects=np.array([self.map[i] for i in ids],np.float64)
        pixels=np.array([self.centers[i] for i in ids],np.float64)
        # Misura come sono distribuiti i centri 3D; se quasi allineati, stimare la posa diventa instabile.
        singular=np.linalg.svd(objects-objects.mean(axis=0),compute_uv=False)
        if singular[1]<1:
            raise RuntimeError('Centri 3D troppo allineati per stimare la posa')
        # Controlla che anche i centri in pixel non risultino tutti lungo una stessa linea.
        image_singular=np.linalg.svd(pixels-pixels.mean(axis=0),compute_uv=False)
        if image_singular[1]<2:
            raise RuntimeError('Centri immagine quasi allineati')
        r=self.rvec.copy(); t=self.tvec.copy()
        # Tentativi limitati di raffinamento/esclusione di un centro incoerente: non un ciclo infinito.
        for _ in range(4):
            objects=np.array([self.map[i] for i in ids],np.float64)
            pixels=np.array([self.centers[i] for i in ids],np.float64)
            # PnP con i soli centri: oggetti XYZ nella base contro misure pixel. Usa la posa precedente come inizializzazione.
            ok,r,t=cv.solvePnP(objects,pixels,self.cfg['_K'],self.cfg['_D'],
                r,t,True,flags=cv.SOLVEPNP_ITERATIVE)
            if not ok:
                raise RuntimeError('PnP dei soli centri fallita')
            R=cv.Rodrigues(r)[0]
            # Porta i centri 3D nel riferimento camera con rotazione e traslazione stimate.
            points=(R@objects.T).T+t.reshape(3)
            if not np.isfinite(points).all() or np.any(points[:,2]<=1):
                raise RuntimeError('Posa dai centri non valida')
            # Riproietta i centri dalla posa stimata per controllare lo scarto rispetto ai centri misurati.
            uv=cv.projectPoints(objects,r,t,self.cfg['_K'],self.cfg['_D'])[0].reshape(-1,2)
            # Distanza in pixel per ciascun centro tra previsione PnP e misura della patch.
            error=np.linalg.norm(uv-pixels,axis=1)
            # Tutti i centri devono essere coerenti entro 2,5 pixel per accettare la stima visiva.
            if np.isfinite(error).all() and error.max()<=2.5:
                break
            if len(ids)<=3 or not np.isfinite(error).all():
                raise RuntimeError('Centri osservati incoerenti con la mappa')
            # Se ci sono piu di tre misure elimina la peggiore e riprova; non puo scendere sotto tre per il feedback visivo.
            del ids[int(np.argmax(error))]
        else:
            raise RuntimeError('Stima dai centri non convergente')
        # Orientamento precedente usato come controllo di continuita, non come nuova misura.
        old_R=cv.Rodrigues(self.rvec)[0]
        # Differenza angolare fra vecchia e nuova soluzione PnP; puo intercettare alcuni salti di soluzione.
        rotation_change=np.linalg.norm(cv.Rodrigues(old_R.T@R)[0])
        # Rifiuta salti stimati oltre 25 mm o 5 gradi; non risolve ogni possibile ambiguita della posa.
        if np.linalg.norm(t-self.tvec)>25 or rotation_change>math.radians(5):
            raise RuntimeError('Salto della posa stimata dai centri: soluzione ambigua o tracking perso')
        # Rimuove dal risultato i centri esclusi nel raffinamento, senza cancellare i centri XYZ della mappa.
        self.centers={i:self.centers[i] for i in ids}
        B=np.eye(4); B[:3,:3],B[:3,3]=R,t.reshape(3)
        # Inverte BASE->CAMERA per ottenere la posa CAMERA->BASE da consegnare al resto del programma.
        self.camera_T_base=np.linalg.inv(B)
        # Esprime il centro TARGET fisso nella camera: non calcola la media dei soli marker osservati.
        self.target_camera=R@np.asarray(self.cfg['_target']['centroid'])+t.reshape(3)
        if not np.isfinite(self.target_camera).all() or self.target_camera[2]<=1:
            raise RuntimeError('Target dietro camera')
        self.rvec,self.tvec=r,t
        # Tempo PC dell’aggiornamento, non timestamp hardware certificato dell’esposizione.
        self.timestamp=time.monotonic()

    # FUNZIONE: VisualTracker.snapshot
    # Consegna copie dei risultati del tracker, senza inviare comandi robot.
    # Qui ids indica i centri OSSERVATI accettati da questo tracker.
    # FastFrames costruira poi uno stato diverso: nel suo ids mette i centri
    # PREVISTI nei margini e tiene observed_ids separato.
    # La distinzione evita di confondere una previsione con una lettura visiva.
    def snapshot(self):
        # Calcola dove dovrebbero apparire i centri XYZ noti: questa e una previsione, non una lettura del frame.
        predicted=cv.projectPoints(np.array(list(self.map.values())),self.rvec,self.tvec,
            self.cfg['_K'],self.cfg['_D'])[0].reshape(-1,2)
        return {'T':self.camera_T_base.copy(),'target_camera':self.target_camera.copy(),
                'ids':sorted(self.centers),
                'centers_px':{str(i):p.tolist() for i,p in self.centers.items()},
                # Proiezioni geometriche in pixel, non punti direttamente trovati nel frame.
                'predicted_centers_px':{str(i):p.tolist() for i,p in zip(self.map,predicted)},
                # Punteggio della patch per le misure visive disponibili; nessun punteggio inventato in SOLO MAPPA.
                'center_scores':{str(i):self.scores[i] for i in self.centers},
                'tracking_source':'center_patch_matching_no_runtime_corners',
                # Istante PC pertinente a questo snapshot; nel bridge combinato e il tempo di arrivo del video.
                'timestamp':self.timestamp}


# CLASSE: ServoFrames
# Estensione del contenitore video con un tracker opzionale.
# La versione base di publish richiederebbe stop per perdita tracking.
# ATTENZIONE: il main crea FastFrames, che eredita da questa classe ma
# SOSTITUISCE publish, start_tracking e measurement per ammettere SOLO MAPPA.
# Leggere queste funzioni base non basta a descrivere il comportamento finale.
class ServoFrames(VideoFrames):
    """Il thread video aggiorna il tracking; il controllo ne legge copie protette."""
    # FUNZIONE: ServoFrames.__init__
    # Inizializza il contenitore madre VideoFrames e i campi del tracker.
    # None indica nessun tracker/errore ancora disponibile.
    # L'orologio e monotono: e un tempo del PC, non l'istante certificato di esposizione.
    def __init__(self, stop):
        super().__init__(stop)
        self.tracker = None
        self.tracking_error = None
        # Registra quando arriva un frame, indipendentemente dal fatto che i marker siano leggibili.
        self.video_time = time.monotonic()

    # FUNZIONE: ServoFrames.publish
    # Versione BASE: aggiorna il tracker sotto protezione e registra un errore.
    # Qui stop.set sarebbe richiesto se update fallisse. Nel programma attuale
    # FastFrames.publish sovrascrive questo metodo e NON ferma per il solo blur.
    # Il frame viene copiato e chi attende un numero nuovo viene avvisato.
    def publish(self, frame):
        # Accesso protetto al contenitore condiviso: impedisce aggiornamenti spezzati mentre un altro thread legge.
        with self.condition:
            # Registra quando arriva un frame, indipendentemente dal fatto che i marker siano leggibili.
            self.video_time = time.monotonic()
            if self.tracker is not None and self.tracking_error is None:
                try:
                    # Passa il nuovo frame al tracker; il tipo concreto di publish decide come gestire un eventuale errore.
                    self.tracker.update(frame)
                except Exception as exc:
                    # Conserva il messaggio dell’errore del tracker per comunicarlo al chiamante.
                    self.tracking_error = str(exc)
                    # Richiede interruzione al resto del programma; non e una RPC StopMotion e non garantisce un arresto istantaneo.
                    self.stop.set()
            # Copia il frame per non dipendere dal buffer che la camera potrebbe riutilizzare.
            self.frame = frame.copy()
            # Ogni publish incrementa il contatore; chi aspetta puo riconoscere un fotogramma successivo.
            self.number += 1
            # Avvisa le attese sulla Condition; potranno leggere i dati dopo il rilascio della protezione.
            self.condition.notify_all()

    # FUNZIONE: ServoFrames.start_tracking
    # Versione BASE: crea VisualTracker su un frame recente.
    # Nel main effettivo questa funzione e sostituita da FastFrames.start_tracking,
    # che gestisce anche il caso in cui non sia possibile osservare i centri.
    def start_tracking(self, cfg, T):
        # Accesso protetto al contenitore condiviso: impedisce aggiornamenti spezzati mentre un altro thread legge.
        with self.condition:
            if self.frame is None or time.monotonic()-self.video_time > 1:
                raise RuntimeError('Nessun frame recente per avviare tracking')
            self.tracker = VisualTracker(self.frame.copy(), cfg, T)

    # FUNZIONE: ServoFrames.grab_measurement
    # Aspetta un frame successivo a quello presente alla chiamata.
    # Restituisce immagine e stato sotto la stessa Condition, evitando una lettura
    # spezzata durante publish. L'attesa rilascia il lock per far arrivare il video.
    # self.measurement usa il metodo dell'oggetto reale: con FastFrames riceve
    # lo stato della mappa corretto opzionalmente dalla visione, non sempre una PnP nuova.
    def grab_measurement(self, timeout=2.0):
        """Immagine e stima dello STESSO frame, copiate sotto il medesimo lock."""
        deadline = time.monotonic()+timeout
        # Accesso protetto al contenitore condiviso: impedisce aggiornamenti spezzati mentre un altro thread legge.
        with self.condition:
            # Memorizza il numero presente alla chiamata, cosi non restituisce subito lo stesso vecchio frame.
            previous = self.number
            # Resta in attesa finche publish non aumenta il contatore oppure arriva un errore/timeout.
            while self.number == previous:
                if self.tracking_error:
                    raise RuntimeError(self.tracking_error)
                if self.stop.is_set():
                    raise RuntimeError('Interruzione richiesta')
                # Secondi residui prima della scadenza: se sono finiti non aspetta indefinitamente.
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('Timeout frame visivo nuovo')
                # Attesa breve che RILASCIA il lock; publish puo quindi consegnare il nuovo frame.
                self.condition.wait(min(0.1, remaining))
            if self.stop.is_set():
                raise RuntimeError(self.tracking_error or 'Interruzione richiesta')
            # Restituisce immagine e stato sotto la protezione; measurement e quello dell’oggetto concreto FastFrames.
            return self.frame.copy(), self.measurement()

    # FUNZIONE: ServoFrames.measurement
    # Versione BASE: legge una misura del tracker recente entro un secondo.
    # FastFrames la sostituisce: il programma finale legge lo snapshot combinato
    # mappa/FK, ammettendo l'assenza di una misura visiva nuova.
    def measurement(self):
        # Accesso protetto al contenitore condiviso: impedisce aggiornamenti spezzati mentre un altro thread legge.
        with self.condition:
            if self.tracking_error:
                raise RuntimeError(self.tracking_error)
            if self.tracker is None:
                raise RuntimeError('Tracking non inizializzato')
            # Copia i risultati correnti del tracker. Questa e la versione base, sostituita nel bridge finale.
            state = self.tracker.snapshot()
            if time.monotonic()-state['timestamp'] > 1.0:
                raise RuntimeError('Misura visiva vecchia oltre 1 secondo')
            return state


# FUNZIONE: marker_projection
# Proietta i centri XYZ memorizzati su un'immagine prevista dalla posa T.
# T e CAMERA->BASE: si inverte per portare i centri BASE->CAMERA.
# Controlla profondita e margine di 40 px. Restituisce gli ID previsti dentro.
# Non apre la camera, non cerca vertici e non conferma occlusioni o nitidezza.
def marker_projection(T, cfg):
    """Proietta SOLO i centri XYZ memorizzati: nessun vertice usato durante il moto."""
    # Inverte CAMERA->BASE per ottenere BASE->CAMERA: il riferimento richiesto dalla proiezione/PnP.
    B=np.linalg.inv(T); ids=[]
    w,h=cfg['camera']['width'],cfg['camera']['height']; m=MARGINE_MARKER_PX
    # Esamina ogni centro XYZ della mappa, compresi i marker sfocati/coperti nella nuova immagine.
    for key,center in cfg['_map_snapshot']['centers_base_mm'].items():
        # Porta il centro XYZ dalla base nel riferimento camera per poterlo proiettare in pixel.
        P=B[:3,:3]@np.asarray(center,float)+B[:3,3]
        # Non considera proiettabile un centro non valido o dietro/troppo vicino al piano camera.
        if not np.isfinite(P).all() or P[2]<=1:
            continue
        uv=cv.projectPoints(P.reshape(1,3),np.zeros(3),np.zeros(3),cfg['_K'],cfg['_D'])[0].reshape(2)
        if np.isfinite(uv).all() and m<=uv[0]<w-m and m<=uv[1]<h-m:
            ids.append(int(key))
    return ids


# FUNZIONE: geometric_view
# Controllo geometrico della camera: target davanti, dentro l'immagine
# ed errore angolare previsto non oltre 8 gradi. Sono soglie del codice,
# non garanzie di precisione fisica. require_markers=False durante il moto.
# Solo se require_markers=True richiede anche tre centri previsti:
# questa opzione e usata per le tappe fotografiche nominali.
def geometric_view(T, cfg, require_markers=False):
    """Trasferimenti: solo target. Tappe foto: anche almeno tre centri previsti."""
    # Esprime il centro target dalla base nella camera, aggiungendo la coordinata omogenea 1.
    P = np.linalg.inv(T) @ np.r_[cfg['_target']['centroid'], 1.0]
    if (not np.isfinite(P).all() or P[2] <= 1 or
        math.degrees(math.atan2(np.linalg.norm(P[:2]), P[2])) > 8):
        raise RuntimeError('Puntamento previsto oltre 8 gradi o target dietro camera')
    uv = cv.projectPoints(P[:3].reshape(1,3),np.zeros(3),np.zeros(3),cfg['_K'],cfg['_D'])[0].reshape(2)
    if (not np.isfinite(uv).all() or not
        (0<=uv[0]<cfg['camera']['width'] and 0<=uv[1]<cfg['camera']['height'])):
        raise RuntimeError('Centro del target previsto fuori immagine')
    # Il numero minimo di marker e condizionale: non e richiesto di default durante il trasferimento.
    if require_markers and len(marker_projection(T,cfg)) < MIN_ARUCO_VISIBILI:
        raise RuntimeError('Tappa foto: meno di 3 centri ArUco previsti nei margini')


# FUNZIONE: verify_photo_state
# Ultimo criterio di inquadratura prima di SALVARE la foto.
# Richiede almeno tre ID distinti PREVISTI nei margini e target entro 5 px
# dal punto principale (cx,cy) della camera. Non richiede ID leggibili.
# Non cerca un'altra posa se fallisce: genera errore, niente foto e niente tappa successiva.
def verify_photo_state(state, cfg):
    """Criterio allo scatto, valido anche con marker sfocati/non osservati."""
    # Conta ID distinti PREVISTI, non il numero di vertici e non il numero di ID riconosciuti dall’immagine.
    if len(set(state['ids'])) < MIN_ARUCO_VISIBILI:
        raise StreamFault('Foto non acquisita: meno di 3 centri ArUco PREVISTI nei margini')
    # Errore di centratura del target in pixel: puo provenire dal modello se manca nuova osservazione.
    error=target_error_px(state,cfg)
    # Foto ammessa entro 5 pixel dal punto principale; non significa allineamento matematicamente perfetto.
    if not math.isfinite(error) or error > 5:
        raise StreamFault('Foto non acquisita: centro del target non centrato entro 5 pixel')


# FUNZIONE: checked_camera
# Legge i giunti effettivi, chiama FK nativa e applica la hand-eye.
# Restituisce (T_camera_to_base, giunti) in mm e gradi.
# E una posa derivata dal robot e dalla calibrazione, non una misura dalle immagini.
def checked_camera(robot, cfg):
    q = robot.get_joints()
    if not valid_joints(q):
        raise RuntimeError('Lettura giunti non valida')
    fk = robot.forward_kin(q)
    if not valid_joints(fk):
        raise RuntimeError('FK nativa non valida')
    return ac.pose_to_T(fk) @ cfg['_handeye'], list(map(float, q))


# FUNZIONE: measured_state
# Utilizzata a robot fermo per ottenere frame e stato corrente.
# Aggiorna la FK del bridge, aspetta un frame nuovo, poi rilegge i giunti.
# Se si sono mossi oltre 0,05 gradi o la posa stimata e troppo discordante, fallisce.
# Il nome non implica sempre una misura VISIVA: lo stato puo essere SOLO MAPPA
# ed espone pose_source e visual_feedback_fresh per dichiararlo.
def measured_state(robot, bridge, cfg):
    """Sincronizzazione a robot fermo: frame nuovo tra due letture dei giunti."""
    if not robot.done():
        raise RuntimeError('Misura per correzione richiesta a robot in moto')
    # Prima lettura robot per prendere una misura a braccio fermo.
    T, q = checked_camera(robot, cfg)
    # Aggiorna la proiezione della mappa anche se nessun centro e visivamente riconoscibile.
    bridge.update_robot_pose(T)
    # Aspetta un frame nuovo con stato della mappa/feedback coerente alla lettura, senza richiedere ArUco leggibili.
    frame, state = bridge.grab_measurement(timeout=2)
    # Seconda lettura: verifica che il robot non si sia spostato durante l’acquisizione del frame.
    T_after, q_after = checked_camera(robot, cfg)
    # Non accetta la misura se il controller indica moto o un giunto cambia oltre 0,05 gradi.
    if not robot.done() or np.max(np.abs(np.array(q_after)-q)) > 0.05:
        raise RuntimeError('Robot mosso durante misura visiva')
    # Limite di plausibilita della catena: non aggiorna la mappa per inseguire errori.
    angle = np.linalg.norm(cv.Rodrigues(T[:3,:3].T @ state['T'][:3,:3])[0])
    if np.linalg.norm(T[:3,3]-state['T'][:3,3]) > 25 or angle > math.radians(8):
        raise RuntimeError('Stima visiva/FK discordanti oltre 25 mm o 8 gradi')
    robot.assert_ready()
    return frame, state, T_after, q_after


# FUNZIONE: target_error_px
# Proietta il centro target corrente e ne misura la distanza in pixel
# da (cx,cy), il punto principale della calibrazione K.
# Questo punto puo differire dal centro geometrico del fotogramma.
# L'errore e previsto se manca un feedback visivo fresco; non e sempre osservato.
def target_error_px(state, cfg):
    P = state['target_camera']
    # Proietta il centro target con distorsione K/D, quindi lo confronta con il punto principale della camera.
    uv = cv.projectPoints(P.reshape(1,3), np.zeros(3), np.zeros(3), cfg['_K'], cfg['_D'])[0].reshape(2)
    # Distanza euclidea in pixel da (cx,cy); non dalle coordinate geometriche width/2,height/2.
    return float(np.linalg.norm(uv-cfg['_K'][:2, 2]))


# FUNZIONE: solve_camera_pose
# Converte una posa CAMERA desiderata nella posa FLANGIA usando hand-eye,
# poi chiede alla IK nativa una soluzione vicina ai giunti precedenti.
# Controlla zona Euler esclusa e coerenza IK/FK. Non muove il robot.
# Le soglie attualmente nel file sono 1 mm e 0,5 gradi: questa copia commentata
# non applica cambiamenti fatti soltanto a mano su un altro computer.
def solve_camera_pose(robot, cfg, desired, previous):
    # Passa dalla posa CAMERA desiderata alla posa FLANGIA usando l’inversa della hand-eye.
    flange = ac.gripper_pose_from_camera_T(desired, cfg['_handeye'])
    # Esclude una zona Euler vicina alla singolarita della rappresentazione; e distinta dalla collisione.
    if abs(abs(flange[4])-90) < float(cfg['visibility']['gimbal_lock_margin_deg']):
        raise RuntimeError('Orientamento nella zona Euler esclusa')
    # Calcolo IK del nodo, senza movimento; cerca una soluzione prossima ai giunti precedenti.
    q = robot.inverse_kin_near(flange, previous)
    # Ricalcola la posa tramite FK dei giunti proposti. actual qui NON significa fisicamente raggiunta.
    actual = ac.pose_to_T(robot.forward_kin(q)) @ cfg['_handeye']
    # Modulo del vettore di rotazione relativo: differenza angolare fra orientamento desiderato e ricostruito, in radianti.
    rotation_error = np.linalg.norm(cv.Rodrigues(actual[:3,:3].T @ desired[:3,:3])[0])
    # Confronto tra due calcoli IK/FK: soglie 1 mm e 0,5 gradi in questa copia, non precisione misurata sul robot.
    if np.linalg.norm(actual[:3,3]-desired[:3,3]) > 1 or rotation_error > math.radians(0.5):
        raise RuntimeError('IK/FK discordanti nella correzione')
    return q

# Secondi per riferimento: 0,008 s = 125 Hz richiesti. Il sistema non garantisce hard real-time.
PERIOD = .008
# Venti punti per blocco: 20*0,008 = 0,160 s. Non sono venti foto.
CHUNK_SAMPLES = 20
# Massimo quaranta punti pronti, cioe 320 ms piu l’eventuale blocco in elaborazione; introduce ritardo nelle correzioni.
MAX_QUEUE = 40
# 400 ms massimi senza video recente. Il frame puo essere sfocato: il requisito riguarda l’arrivo del video.
VISION_MAX_AGE = .4
# Scadenza dei punti rispetto al frame di base: 800 ms. Non trasforma una previsione in una misura visiva.
PACKET_MAX_AGE = .8
# Massima velocita articolare del riferimento, 3 gradi/s. Non e una percentuale e non e mm/s.
VELOCITY = 6.0
# Massima accelerazione articolare del riferimento, 10 gradi/s².
ACCELERATION = 10.0
# Massima variazione dell’accelerazione articolare, 100 gradi/s³, per limitare cambiamenti troppo bruschi.
JERK = 100.0
# 5 mm/s usati per stimare le durate nominali; non certificano il limite cartesiano sulla curva articolare corretta.
CAMERA_SPEED = 5.0
# 0,15 gradi/s come limite angolare del correttore; l’incremento per blocco dipende dalla sua durata.
CORRECTION_RATE = .15

# CLASSE: FastFrames
# Bridge realmente creato dal main. Tiene separati previsione e osservazione.
# ids: centri PREVISTI nei margini, anche sfocati/coperti.
# observed_ids: centri misurati nel frame corrente accettato.
# Se manca il feedback conserva la precedente correzione e propaga con FK.
# Durante il movimento non impone tre ArUco; allo scatto il vincolo viene
# applicato da verify_photo_state. Video e posa devono restare aggiornati.
class FastFrames(ServoFrames):
    """Conta i CENTRI PREVISTI dalla mappa, anche senza osservazione visiva.

    I centri osservati servono solo per aggiornare la correzione visiva della
    posa. Se mancano, conserva l'ultima correzione e usa la FK attuale.
    Il sender legge uno snapshot gia pronto, senza RPC o calcoli pesanti.
    """
    # FUNZIONE: FastFrames.__init__
    # Prepara snapshot, memoria delle pose e correzione visiva opzionale.
    # pose_data contiene una coppia (matrice camera da FK, istante PC).
    # visual_data contiene (stato osservato, correzione rispetto a FK, istante frame).
    # pose_lock protegge la breve copia della storia, non blocca l'invio ServoJ.
    def __init__(self, stop):
        super().__init__(stop)
        # Nessuno snapshot combinato disponibile prima di aver collegato mappa e posa robot.
        self.fast_state=None
        # Nessuna posa FK corrente ancora fornita al bridge.
        self.pose_data=None
        # Nessuna correzione visiva precedente: inizialmente la previsione sara solo FK.
        self.visual_data=None
        # Storia breve delle pose reali derivate dai giunti, utile per abbinarle al tempo del video.
        self.pose_history=deque(maxlen=100)
        # Protezione breve della storia pose; il sender non usa questo lock per inviare i comandi.
        self.pose_lock=threading.Lock()
        self.model_cfg=None
        # Cancella la nota di mancato feedback quando una correzione valida viene accettata.
        self.visual_warning=None

    # FUNZIONE: FastFrames.update_robot_pose
    # Riceve una nuova posa camera calcolata dai giunti effettivi.
    # Non fa RPC qui: il chiamante ha gia letto il robot e calcolato la FK.
    # Salva posa e tempo nella storia e rigenera le proiezioni dei centri.
    # Anche con zero centri osservati la previsione segue il movimento del robot.
    def update_robot_pose(self, T, timestamp=None):
        T=np.asarray(T,float)
        if T.shape!=(4,4) or not np.isfinite(T).all():
            raise StreamFault('Posa robot non valida per proiettare i centri')
        stamp=time.monotonic() if timestamp is None else float(timestamp)
        data=(T.copy(),stamp)
        # Copia/aggiorna la storia delle pose in modo coerente tra thread video e pianificatore.
        with self.pose_lock:
            # Sostituisce insieme matrice e tempo della FK corrente, senza inventare una nuova osservazione visiva.
            self.pose_data=data
            # Conserva la lettura per l’abbinamento temporale con eventuali immagini accettate.
            self.pose_history.append(data)
        self.refresh_state()

    # FUNZIONE: FastFrames.accept_visual
    # Decide se usare una stima proveniente dalle immagini.
    # La abbina alla posa robot temporalmente piu vicina, compensando solo
    # l'eventuale ritardo video MISURATO e configurato. Se tempi o pose sono
    # troppo discordanti, non aggiorna la correzione e resta sulla previsione.
    # La correzione conservata non e una nuova osservazione quando il video e sfocato.
    def accept_visual(self, state, frame_time):
        # Copia/aggiorna la storia delle pose in modo coerente tra thread video e pianificatore.
        with self.pose_lock:
            history=list(self.pose_history)
        if not history:
            return
        # Compensa soltanto il ritardo configurato: non presume automaticamente una sincronizzazione hardware.
        exposure_time=frame_time-self.model_cfg.get('_vision_latency',0.)
        # Sceglie la posa FK temporalmente piu vicina al tempo immagine stimato.
        robot_T,stamp=min(history,key=lambda pair:abs(pair[1]-exposure_time))
        # Se la differenza supera 250 ms rifiuta la correzione visiva; la previsione della mappa resta disponibile.
        if abs(stamp-exposure_time)>.25:
            self.visual_warning='Misura visiva senza posa robot temporalmente vicina: uso previsione'
            return
        # Differenza trasformativa fra posa ottenuta dalla visione e posa FK abbinata: conserva anche l’eventuale traslazione.
        correction=state['T']@np.linalg.inv(robot_T)
        angle=np.linalg.norm(cv.Rodrigues(correction[:3,:3])[0])
        if np.linalg.norm(state['T'][:3,3]-robot_T[:3,3])>25 or angle>math.radians(5):
            self.visual_warning='Misura visiva discordante: uso previsione'
            return
        # Questa correzione resta congelata se il feedback viene perso.
        # Memorizza una correzione ACCETTATA; non verra ringiovanita solo perche arrivano altri frame sfocati.
        self.visual_data=(state,correction,frame_time)
        # Cancella la nota di mancato feedback quando una correzione valida viene accettata.
        self.visual_warning=None

    # FUNZIONE: FastFrames.start_tracking
    # Collega mappa e posa iniziale al video gia arrivato.
    # Prova a inizializzare il tracking dei centri; se non riesce non richiede stop:
    # con una mappa nota puo partire SOLO MAPPA. Non ricrea continuamente il
    # rilevatore dei vertici. La prima mappa, invece, richiede ancora sei ID riconoscibili.
    def start_tracking(self, cfg, T):
        # Accesso protetto al contenitore condiviso: impedisce aggiornamenti spezzati mentre un altro thread legge.
        with self.condition:
            if self.frame is None or time.monotonic()-self.video_time>1:
                raise StreamFault('Nessun video recente per inizializzare la mappa')
            frame=self.frame.copy(); frame_time=self.video_time
        # Collega il bridge ai centri della mappa e alle intrinseche.
        self.model_cfg=cfg
        self.update_robot_pose(T)
        try:
            # Prova una sola inizializzazione del tracking delle patch dei centri per questa sessione.
            tracker=VisualTracker(frame,cfg,T)
            self.accept_visual(tracker.snapshot(),frame_time)
            self.tracker=tracker
        except Exception as exc:
            # Una mappa gia nota permette di partire anche senza ID osservabili.
            # Non si reinizializza ripetutamente il rilevatore dei vertici.
            # Nel fallback nessun tracking attivo; la previsione geometrica puo comunque funzionare con la mappa nota.
            self.tracker=None
            self.visual_warning=f'Tracking non inizializzato; previsione dalla mappa: {exc}'
        self.refresh_state()

    # FUNZIONE: FastFrames.publish
    # Gestisce ogni frame reale. Se il tracker esiste, guida la ricerca con
    # la previsione FK e prova ad accettare una correzione visiva.
    # Un fallimento di tracking registra una nota ma NON attiva stop.
    # Aggiorna immagine, numero e snapshot; il video deve comunque continuare
    # ad arrivare, anche se tutti i marker sono sfocati o coperti.
    def publish(self, frame):
        # Tempo di ricezione del frame sul PC: resta distinto dall’ultimo successo del tracking.
        acquired=time.monotonic()
        # Accesso protetto al contenitore condiviso: impedisce aggiornamenti spezzati mentre un altro thread legge.
        with self.condition:
            self.video_time=acquired
        if self.tracker is not None:
            try:
                # Guida la ricerca dei centri con la posa prevista corrente.
                if self.pose_data is not None:
                    T,_=self.pose_data
                    # Se non c’e mai stata una correzione visiva usa identita; altrimenti conserva l’ultima accettata.
                    correction=np.eye(4) if self.visual_data is None else self.visual_data[1]
                    # Inverte CAMERA->BASE per ottenere BASE->CAMERA: il riferimento richiesto dalla proiezione/PnP.
                    B=np.linalg.inv(correction@T)
                    self.tracker.rvec=cv.Rodrigues(B[:3,:3])[0]
                    self.tracker.tvec=B[:3,3].reshape(3,1)
                # Passa il nuovo frame al tracker; il tipo concreto di publish decide come gestire un eventuale errore.
                self.tracker.update(frame)
                # Accetta il nuovo feedback solo se supera i controlli temporali e di discordanza con il robot.
                self.accept_visual(self.tracker.snapshot(),acquired)
            except Exception as exc:
                # La perdita visiva non invalida la posizione geometrica dei marker.
                # La perdita del tracking non attiva stop in questa versione: viene dichiarato il passaggio al solo modello.
                self.visual_warning=f'Nessuna nuova correzione visiva: {exc}'
        # Accesso protetto al contenitore condiviso: impedisce aggiornamenti spezzati mentre un altro thread legge.
        with self.condition:
            self.frame=frame.copy()
            self.number+=1
            self.refresh_state()
            # Avvisa le attese sulla Condition; potranno leggere i dati dopo il rilascio della protezione.
            self.condition.notify_all()

    # FUNZIONE: FastFrames.refresh_state
    # Costruisce lo stato che il sender leggerà senza calcoli pesanti.
    # Combina l'ultima correzione visiva accettata con la FK attuale.
    # Proietta OGNI centro della mappa e conta quelli nei margini.
    # Distingue osservati e previsti; se la nuova immagine non conferma i centri,
    # non presenta le vecchie osservazioni come attuali. Non impone qui un minimo.
    # In SOLO MAPPA la geometria puo essere corretta solo quanto mappa e calibrazione.
    def refresh_state(self):
        # Non puo proiettare nulla finche mancano la mappa/calibrazione o una posa robot valida.
        if self.model_cfg is None or self.pose_data is None:
            return
        cfg=self.model_cfg
        # FK attuale e suo tempo; questi sono dati del robot, non una stima ricavata dal frame.
        robot_T,robot_time=self.pose_data
        visual=self.visual_data
        # Se non c’e mai stata una correzione visiva usa identita; altrimenti conserva l’ultima accettata.
        correction=np.eye(4) if visual is None else visual[1]
        # Propaga l’ultima correzione visiva sulla FK attuale. Senza correzione coincide con la sola FK.
        T=correction@robot_T
        # Inverte CAMERA->BASE per ottenere BASE->CAMERA: il riferimento richiesto dalla proiezione/PnP.
        B=np.linalg.inv(T)
        h,w=cfg['camera']['height'],cfg['camera']['width']
        m=MARGINE_MARKER_PX
        predicted={};visible=[]
        # Esamina ogni centro XYZ della mappa, compresi i marker sfocati/coperti nella nuova immagine.
        for key,center in cfg['_map_snapshot']['centers_base_mm'].items():
            # Porta il centro XYZ dalla base nel riferimento camera per poterlo proiettare in pixel.
            P=B[:3,:3]@np.asarray(center,float)+B[:3,3]
            # Non considera proiettabile un centro non valido o dietro/troppo vicino al piano camera.
            if not np.isfinite(P).all() or P[2]<=1:
                continue
            uv=cv.projectPoints(P.reshape(1,3),np.zeros(3),np.zeros(3),cfg['_K'],cfg['_D'])[0].reshape(2)
            if not np.isfinite(uv).all():
                continue
            # Memorizza il centro PREVISTO in pixel anche senza misurarlo dall’immagine.
            predicted[str(key)]=uv.tolist()
            # Questo e il criterio di inquadratura del centro: margine sui due lati orizzontali e verticali.
            if m<=uv[0]<w-m and m<=uv[1]<h-m:
                # Aggiunge l’ID ai geometricamente previsti dentro, NON ai marker confermati visivamente.
                visible.append(int(key))
        # Feedback fresco solo se corrisponde al frame corrente ed e abbastanza recente; vecchie misure non valgono come nuove.
        fresh=(visual is not None and time.monotonic()-visual[2]<=VISION_MAX_AGE
               and abs(visual[2]-self.video_time)<1e-6)
        # Se il frame non ha feedback accettato, il gruppo delle misure osservate viene svuotato.
        raw=visual[0] if fresh else {}
        # Centri osservati correnti che sono anche nei margini previsti, separati dagli ID solo mappa.
        observed=sorted(set(raw.get('ids',[]))&set(visible))
        target=B[:3,:3]@np.asarray(cfg['_target']['centroid'],float)+B[:3,3]
        # Pubblica uno snapshot completo: il sender lo legge senza eseguire proiezioni o chiamate SDK.
        self.fast_state={
            'T':T.copy(), 'target_camera':target,
            'ids':sorted(visible),  # criterio richiesto: centri PREVISTI nei margini
            # ID con osservazione attuale accettata; non coincide necessariamente con il conteggio usato per la foto.
            'observed_ids':observed,
            # ID previsti nei margini senza conferma nel frame corrente.
            'prediction_only_ids':sorted(set(visible)-set(observed)),
            'centers_px':{str(i):raw.get('centers_px',{}).get(str(i)) for i in observed},
            # Proiezioni geometriche in pixel, non punti direttamente trovati nel frame.
            'predicted_centers_px':predicted,
            # Punteggio della patch per le misure visive disponibili; nessun punteggio inventato in SOLO MAPPA.
            'center_scores':{str(i):raw.get('center_scores',{}).get(str(i)) for i in observed},
            # True solo per una correzione visiva accettata relativa al frame attuale, non per una vecchia misura conservata.
            'visual_feedback_fresh':bool(fresh),
            # Istante dell’ultima correzione accettata: non viene aggiornato dai frame privi di misure.
            'last_visual_timestamp':None if visual is None else visual[2],
            # L’origine della posa distingue previsione da sola FK e FK con correzione visiva conservata.
            'pose_source':'robot_fk_only' if visual is None else 'robot_fk_with_last_visual_correction',
            'visual_warning':self.visual_warning,
            # Tempo dell’ultimo aggiornamento della posa dai giunti, distinto da quello del video.
            'robot_pose_timestamp':robot_time,
            # Matrice FK attuale: il produttore la usa per propagare la correzione sulla posa futura.
            'robot_T':robot_T.copy(),
            # Istante PC pertinente a questo snapshot; nel bridge combinato e il tempo di arrivo del video.
            'timestamp':self.video_time,
            # Specifica se il vincolo e geometrico e in quale fase del programma e richiesto.
            'visibility_criterion':'predicted_map_centers',
        }

    # FUNZIONE: FastFrames.fast_measurement
    # Legge lo snapshot gia pronto e controlla freschezza video/posa e target.
    # Non fa RPC, non aggiorna il tracker e non calcola una IK nel sender veloce.
    # require_markers e False per default: durante il trasferimento sono ammessi
    # zero centri previsti. Il requisito tre viene controllato prima dello scatto.
    # La perdita del video vero o della telemetria resta motivo di errore.
    def fast_measurement(self, require_markers=False):
        state=self.fast_state
        if state is None:
            raise StreamFault('Previsione centri non inizializzata')
        now=time.monotonic()
        # Anche in SOLO MAPPA richiede arrivo del video: oltre 400 ms senza frame il controllo fallisce.
        if now-state['timestamp']>VISION_MAX_AGE:
            raise StreamFault('Video non aggiornato entro 400 ms')
        # Una vecchia FK non basta a seguire la camera in movimento: oltre 500 ms genera errore.
        if now-state['robot_pose_timestamp']>.5:
            raise StreamFault('Posa robot non aggiornata entro 500 ms')
        # Il numero minimo di marker e condizionale: non e richiesto di default durante il trasferimento.
        if require_markers and len(set(state['ids']))<MIN_ARUCO_VISIBILI:
            raise StreamFault('Tappa foto: meno di 3 centri PREVISTI entro i margini')
        if not np.isfinite(state['target_camera']).all() or state['target_camera'][2]<=1:
            raise StreamFault('Target previsto dietro camera')
        return state

    # FUNZIONE: FastFrames.measurement
    # Sostituisce la versione base: restituisce lo stato combinato attuale.
    # Un frame recente non implica una nuova osservazione ArUco; leggere sempre
    # i campi observed_ids, prediction_only_ids e pose_source per distinguerli.
    def measurement(self):
        # Le misure restituite sono marcate come previste o corrette visivamente.
        # Non usa il timestamp dell'ultimo tracking come se fosse una nuova misura.
        # Legge lo snapshot combinato, non forza una nuova PnP e non attende il prossimo frame.
        return self.fast_measurement()


# CLASSE: CommandHistory
# Conserva i riferimenti articolari inviati e il loro tempo PC.
# Non e una registrazione delle pose fisicamente raggiunte.
# Nel produttore attuale la propagazione della posa usa robot_T dalla FK;
# latest resta usato per verificare l'inseguimento dei giunti.
class CommandHistory:
    """Storia dei riferimenti inviati; nessuna RPC nel thread veloce."""
    # FUNZIONE: CommandHistory.__init__
    # Memorizza la configurazione iniziale e prepara una coda di lunghezza limitata.
    # Quando la coda supera 1000 elementi, i piu vecchi vengono eliminati.
    def __init__(self, q0):
        self.values=deque([(time.monotonic(),tuple(q0))],maxlen=1000)

    # FUNZIONE: CommandHistory.sent
    # Callback eseguita dopo una RPC ServoJ riuscita.
    # Conserva il riferimento inviato e il tempo di ritorno della chiamata;
    # non prova che il robot abbia gia raggiunto quei giunti.
    def sent(self,packet,when):
        self.values.append((when,packet.joints))

    # FUNZIONE: CommandHistory.at
    # Cerca il riferimento inviato non successivo al tempo richiesto.
    # Prima del primo invio conserva la configurazione iniziale ferma.
    # Il controllo di eta impedisce associazioni temporali troppo vecchie.
    # E un'utilita rimasta disponibile; il produttore corrente usa robot_T per la propagazione.
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

    # FUNZIONE: CommandHistory.latest
    # Restituisce gli ultimi giunti COMANDATI come array NumPy.
    # Il produttore li confronta con quelli EFFETTIVI letti dal robot.
    def latest(self):
        return np.array(self.values[-1][1],float)


# FUNZIONE: camera_for
# Calcola la posa camera per un vettore articolare q qualsiasi tramite FK.
# q puo essere un riferimento futuro, non necessariamente la lettura attuale.
# La matrice restituita e CAMERA->BASE con traslazione in mm.
def camera_for(robot, cfg, q):
    flange=robot.forward_kin(list(map(float,q)))
    if not valid_joints(flange):
        raise StreamFault('FK nativa non disponibile')
    return ac.pose_to_T(flange) @ cfg['_handeye']


# FUNZIONE: validate_curve
# Approva un blocco prima che i suoi punti entrino nella coda ServoJ.
# Controlla limiti analitici, incrementi, collisioni campionate e puntamento target.
# Non richiede tre centri ArUco durante il trasferimento, ne sulla FK nominale
# ne sulla previsione corretta visivamente. Non invia nessun comando robot.
def validate_curve(curve, robot, cfg, guard, state, reference_T, previous_q):
    """Valida il polinomio e ogni punto ServoJ prima che entri nella coda."""
    # Verifica i massimi analitici della quintica, anche tra i riferimenti campionati.
    curve.require_limits(VELOCITY,ACCELERATION,JERK)
    # Numero di riferimenti ServoJ del blocco; la durata e un multiplo del periodo.
    count=round(curve.duration/PERIOD)
    # Campiona il blocco ai tempi di invio previsti; il suo primo confine e gia noto dal blocco precedente.
    joints=[curve.at(k*PERIOD) for k in range(1,count+1)]
    prev=np.asarray(previous_q,float)
    require_guard(guard)
    for q in joints:
        # Evita un incremento fra due riferimenti incompatibile con la velocita massima configurata.
        if np.max(np.abs(q-prev))>VELOCITY*PERIOD+1e-6:
            raise StreamFault('Incremento ServoJ oltre il limite di velocita')
        # Il guard interpola anche fra due riferimenti, non solo ai capi del blocco.
        # Controlla collisioni anche tra due riferimenti con tre campioni geometrici.
        ok,reasons=guard.path_safe(prev.tolist(),q.tolist(),n=3)
        if not ok:
            raise StreamFault(f'Collisione prevista nel blocco servo: {reasons}')
        T=camera_for(robot,cfg,q)
        # Nel trasferimento controlla il TARGET: require_markers resta False.
        geometric_view(T,cfg)
        # Prevede la posa del nuovo riferimento applicando la correzione corrente; il nome non significa sempre misura visiva.
        measured_prediction=state['T'] @ np.linalg.inv(reference_T) @ T
        # Anche la posa corretta dal feedback deve puntare il target.
        # Durante il trasferimento NON viene imposto un numero minimo di ArUco.
        # Anche sulla previsione corretta richiede il puntamento target, non tre ArUco nel trasferimento.
        geometric_view(measured_prediction,cfg)
        prev=q
    return joints


# CLASSE: TrajectoryProducer
# Thread che prepara blocchi futuri su un client RPC separato dal sender.
# Puo eseguire FK/IK e guard collisioni senza inserirli nel ciclo veloce.
# Le nuove correzioni vengono raccordate C2 ai punti gia accodati,
# non cambiano retroattivamente comandi gia approvati/inviati.
# Se e troppo lento, il sender interrompe per buffer/scadenza invece di recuperare a raffica.
class TrajectoryProducer(threading.Thread):
    """FK/IK e collisioni su una connessione SEPARATA dall'invio ServoJ.

    Una nuova correzione viene raccordata con quintiche C2 alla parte gia
    accodata. Non cambia retroattivamente i riferimenti gia approvati.
    """
    # FUNZIONE: TrajectoryProducer.__init__
    # Collega robot pianificatore, mappa, video, percorso e coda.
    # q/v/a sono posizione, velocita e accelerazione al confine dell'ultimo blocco.
    # bias e la correzione di orientamento accumulata. ready segnala che sono
    # pronti i primi punti; telemetry memorizza una lettura robot con il suo tempo.
    def __init__(self, robot, cfg, bridge, guard, path, q0, bias, history, buffer, stop, log):
        super().__init__(name='servo-planner',daemon=True)
        self.robot,self.cfg,self.bridge,self.guard=robot,cfg,bridge,guard
        self.path,self.history,self.buffer,self.stop,self.log=path,history,buffer,stop,log
        self.q=np.array(q0,float); self.v=np.zeros(6); self.a=np.zeros(6)
        self.bias=bias.copy(); self.final_q=None
        self.telemetry=(time.monotonic(),self.q.copy())
        # Segnale che sblocca stream_leg quando ci sono i primi punti o un errore da gestire.
        self.ready=threading.Event()
        self.last_error=None

    # FUNZIONE: TrajectoryProducer.health
    # Controllo rapido chiamato anche dal sender: legge snapshot e tempi.
    # Non deve leggere il robot via RPC o elaborare immagini qui.
    # Un errore del produttore invalida anche eventuali punti rimasti nella coda.
    def health(self):
        # Controllo rapido di freschezza e target; non richiede centri osservati e non impone tre nel moto.
        self.bridge.fast_measurement()
        stamp,actual=self.telemetry
        if time.monotonic()-stamp>.5:
            raise StreamFault('Telemetria robot non aggiornata entro 500 ms')
        # Qualunque errore di pianificazione deve arrivare al sender anche se la coda ha ancora dati.
        if self.buffer.fault:
            raise StreamFault(self.buffer.fault)

    # FUNZIONE: TrajectoryProducer.run
    # Punto di ingresso del thread Python: esegue produce.
    # Se qualcosa fallisce, scrive buffer.fault per informare subito il sender.
    # Sblocca ready anche per un errore, evitando che il chiamante aspetti invano.
    def run(self):
        try:
            # Il corpo effettivo del lavoro del thread viene eseguito qui.
            self.produce()
        except BaseException as exc:
            # Comunica il guasto al ciclo di invio invece di far morire silenziosamente il produttore.
            self.buffer.fault=str(exc)
        finally:
            self.ready.set()

    # FUNZIONE: TrajectoryProducer.produce
    # CICLO DI PIANIFICAZIONE: un blocco ha 20 riferimenti da 8 ms, cioe 160 ms.
    # Legge robot, aggiorna la previsione dei centri e prende il prossimo punto nominale.
    # Il Jacobiano corregge SOLO il puntamento del centro target, non cerca di far
    # entrare tre ArUco. Applica IK e costruisce una quintica raccordata.
    # Se una correzione e troppo forte prova a ridurla; se nessun blocco e valido fallisce.
    # Alla tappa mantiene la posa e aspetta centratura e assestamento; il requisito
    # ArUco finale sara controllato dopo ServoMoveEnd da photo_when_still.
    def produce(self):
        # 20 riferimenti per 0,008 s = blocco nominale di 0,160 s.
        dt=CHUNK_SAMPLES*PERIOD
        # Inizializza numerazione pacchetti, tempo del percorso e contatori di assestamento.
        index=0; path_t=0.; stable_since=None; settled_frames=0; last_frame=None
        ended_at=None; previous_actual=None; stable_photo=None
        # Primo riferimento uguale ai giunti effettivi: non si salta all'inizio del piano.
        state=self.bridge.fast_measurement()
        # Primo riferimento uguale ai giunti attuali: non salta direttamente a una posa lontana.
        self.buffer.put(Packet(index,tuple(self.q),state['timestamp']+PACKET_MAX_AGE))
        index+=1
        # Continua a produrre solo finche non arriva stop utente o annullamento del trasferimento.
        while not self.stop.is_set() and not self.buffer.cancel.is_set():
            # Non calcola un orizzonte arbitrariamente lontano dal consumo effettivo.
            # Non pianifica troppo in anticipo: attende che il sender consumi parte della coda.
            while self.buffer.queue.qsize()>MAX_QUEUE-CHUNK_SAMPLES:
                # Attesa breve interrompibile per non occupare CPU aspettando spazio nella coda.
                if self.stop.wait(.002) or self.buffer.cancel.is_set():
                    return
            self.robot.assert_ready()
            # Qui actual contiene giunti EFFETTIVI letti, diversamente da actual nella sola pianificazione IK/FK.
            actual=self.robot.get_joints()
            if not valid_joints(actual):
                raise StreamFault('Giunti effettivi non leggibili durante servo')
            actual=np.array(actual,float)
            # Confronta giunti letti e ultimo riferimento inviato: oltre 1 grado di inseguimento genera errore.
            if np.max(np.abs(actual-self.history.latest()))>1.0:
                raise StreamFault('Inseguimento articolare oltre 1 grado')
            self.telemetry=(time.monotonic(),actual.copy())
            # Aggiorna la previsione con FK dei giunti letti: funziona anche quando nessuna patch viene osservata.
            self.bridge.update_robot_pose(camera_for(self.robot,self.cfg,actual))
            state=self.bridge.fast_measurement()
            # Corregge l'età di acquisizione solo se misurata/configurata sul sistema reale.
            # Lo stato corrente propaga l'ultima correzione visiva sulla FK
            # effettiva, anche se l'immagine non fornisce nuovi centri osservati.
            # Usa la FK effettiva associata allo snapshot, non interpreta la previsione come nuova osservazione.
            reference_T=state['robot_T']
            # Controllo di plausibilita sul riferimento corrispondente alla misura.
            rotation_gap=np.linalg.norm(cv.Rodrigues(reference_T[:3,:3].T@state['T'][:3,:3])[0])
            if np.linalg.norm(reference_T[:3,3]-state['T'][:3,3])>25 or rotation_gap>math.radians(8):
                raise StreamFault('Visione e riferimento temporale discordanti')
            # Avanza il tempo nominale di un blocco; alla fine resta sulla destinazione per assestarsi.
            next_t=min(path_t+dt,self.path.duration)
            # Sei giunti del percorso nominale al prossimo confine di blocco.
            nominal_q=self.path.at(next_t)
            # Dai giunti nominali torna alla posa camera per applicare una correzione di puntamento.
            nominal_T=camera_for(self.robot,self.cfg,nominal_q)
            desired=nominal_T.copy()
            # Conserva la correzione gia accumulata, anziche annullarla tornando all’orientamento nominale.
            desired[:3,:3]=nominal_T[:3,:3]@self.bias
            # Propaga la correzione di posa sul riferimento futuro: con video perso e una previsione da FK piu correzione congelata.
            predicted_visual=state['T'] @ np.linalg.inv(reference_T) @ desired
            # Centro target nel riferimento della camera futura: e questo punto che il Jacobiano vuole centrare.
            P=(np.linalg.inv(predicted_visual)@np.r_[self.cfg['_target']['centroid'],1.])[:3]
            # Correzione a velocita angolare limitata; il blocco la raccorda senza salti.
            # Corregge SOLO il puntamento: gain effettivo 0,4 e angolo limitato da CORRECTION_RATE per la durata del blocco.
            delta=pointing_increment(P,gain=.4,max_angle_rad=math.radians(CORRECTION_RATE*dt))
            error=target_error_px(state,self.cfg)
            # Alla destinazione, quando il target e gia entro 5 pixel, non continua a introdurre piccole rotazioni.
            if path_t>=self.path.duration and error<=5:
                # Incremento nullo: mantiene l’orientamento per poter verificare la stabilita.
                delta=np.zeros(3)
            accepted=None; errors=[]
            # Prova correzione intera, meta, un quarto e infine nessuna nuova correzione per trovare un raccordo ammissibile.
            for scale in (1.,.5,.25,0.):
                # Compone la nuova piccola rotazione con quelle gia accettate; non e ancora un comando al robot.
                candidate_bias=self.bias@rotation_exp(delta*scale)
                # Non consente che la correzione totale si accumuli oltre 3 gradi rispetto al nominale.
                if np.linalg.norm(cv.Rodrigues(candidate_bias)[0])>math.radians(3):
                    raise StreamFault('Correzione accumulata oltre 3 gradi')
                desired[:3,:3]=nominal_T[:3,:3]@candidate_bias
                try:
                    # Converte posa camera corretta in sei giunti, verificando la coerenza IK/FK.
                    goal=solve_camera_pose(self.robot,self.cfg,desired,self.q.tolist())
                    # Nuovo blocco con stessi q/v/a del confine precedente: il raccordo non impone uno stop a ogni blocco.
                    curve=Quintic(self.q,self.v,self.a,goal,self.path.at(next_t,1),
                                  self.path.at(next_t,2),dt)
                    # Solo se limiti, collisioni e puntamento passano, i punti potranno entrare nel buffer.
                    joints=validate_curve(curve,self.robot,self.cfg,self.guard,state,reference_T,self.q)
                    # Conserva il primo candidato ammissibile e termina i tentativi con scale piu piccole.
                    accepted=(candidate_bias,curve,joints,scale)
                    break
                except (RuntimeError,ValueError) as exc:
                    # Registra il motivo del rifiuto e prova una correzione ridotta; non nasconde il fallimento se nessuna passa.
                    errors.append(str(exc))
            # Nessun raccordo valido: non inventa una deviazione e non prosegue con un comando non verificato.
            if accepted is None:
                raise StreamFault(f'Nessun raccordo valido: {errors[-1]}')
            candidate_bias,curve,joints,scale=accepted
            # Se i calcoli hanno consumato troppo tempo, non approva un blocco basato su dati video ormai vecchi.
            if time.monotonic()-state['timestamp']>VISION_MAX_AGE:
                raise StreamFault('Validazione del blocco troppo lenta rispetto alla visione')
            self.bias=candidate_bias
            # Memorizza lo stato finale calcolato del blocco per raccordare quello successivo.
            self.q,self.v,self.a=curve.at(dt),curve.at(dt,1),curve.at(dt,2)
            path_t=next_t
            is_last=False
            if path_t>=self.path.duration:
                ended_at=time.monotonic() if ended_at is None else ended_at
                # Dopo l’arrivo nominale concede al massimo 30 s per centratura/assestamento; poi interrompe.
                if time.monotonic()-ended_at>30:
                    raise StreamFault('Centratura/assestamento alla tappa non conclusi entro 30 s')
                # Richiede letture articolari quasi ferme e vicine al riferimento finale: controllo numerico di assestamento.
                stationary=(previous_actual is not None and np.max(np.abs(actual-previous_actual))<=.03
                            and np.max(np.abs(actual-self.q))<=.15)
                # Conferma anche target centrato e velocita/accelerazione finali del riferimento quasi nulle.
                if stationary and error<=5 and np.max(np.abs(self.v))<1e-6 and np.max(np.abs(self.a))<1e-6:
                    stable_since=time.monotonic() if stable_since is None else stable_since
                    # Conta frame distinti, non piu letture ripetute della stessa immagine.
                    if state['timestamp']!=last_frame:
                        settled_frames+=1
                        last_frame=state['timestamp']
                    # Prima di chiudere il servo richiede almeno 0,6 s stabili e tre frame distinti, non tre ArUco osservati.
                    is_last=(time.monotonic()-stable_since>=.6 and settled_frames>=3)
                else:
                    stable_since=None; settled_frames=0
            previous_actual=actual.copy()
            # Registra quale parte era prevista/osservata e come e stata ridotta la correzione.
            self.log.append({'time_monotonic':time.monotonic(),'nominal_t':path_t,
                'target_error_px':error,'visible_predicted_ids':state['ids'],
                # ID con osservazione attuale accettata; non coincide necessariamente con il conteggio usato per la foto.
                'observed_ids':state['observed_ids'],
                # ID previsti nei margini senza conferma nel frame corrente.
                'prediction_only_ids':state['prediction_only_ids'],
                # True solo per una correzione visiva accettata relativa al frame attuale, non per una vecchia misura conservata.
                'visual_feedback_fresh':state['visual_feedback_fresh'],
                # L’origine della posa distingue previsione da sola FK e FK con correzione visiva conservata.
                'pose_source':state['pose_source'],'correction_scale':scale,
                'queue_packets':self.buffer.queue.qsize(),
                'bias_rad':cv.Rodrigues(self.bias)[0].reshape(3).tolist()})
            if is_last:
                self.final_q=self.q.copy()
            # Accoda in ordine i riferimenti gia approvati, con scadenza e numero progressivo.
            for k,q in enumerate(joints):
                self.buffer.put(Packet(index,tuple(q),state['timestamp']+PACKET_MAX_AGE,
                                       last=is_last and k==len(joints)-1))
                index+=1
            # Segnala che il primo riferimento piu un blocco sono pronti: il sender puo partire.
            if self.buffer.queue.qsize()>=CHUNK_SAMPLES+1:
                self.ready.set()
            if is_last:
                self.final_q=self.q.copy()
                self.ready.set()
                return


# FUNZIONE: stream_leg
# Esegue un trasferimento verso una tappa fotografica con ServoJ.
# Prepara percorso, produttore e buffer; verifica che il robot non si sia mosso
# prima di avviare il servo. Il primo riferimento e la configurazione attuale.
# Poi ServoMoveStart -> send_packets -> ServoMoveEnd.
# In caso di errore richiede StopMotion anche se si perde la risposta di avvio.
# A fine normale pending resta True finche la foto conferma il robot fermo.
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
    # Avvia il thread di preparazione; questo non abilita ancora la modalita servo del controller.
    producer.start()
    try:
        deadline=time.monotonic()+10
        # Aspetta brevemente la preparazione iniziale, ricontrollando stop e timeout.
        while not producer.ready.wait(.02):
            if stop.is_set():
                raise StreamFault('Interruzione richiesta prima della partenza servo')
            # Una preparazione che non termina entro il tempo ammesso non autorizza la partenza.
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
        # Imposta lo stato PRIMA di ServoMoveStart: anche una risposta persa deve portare al tentativo di stop.
        active=True  # prima della RPC: una risposta persa richiede comunque StopMotion
        # Memorizza che il moto puo essere stato avviato e deve essere gestito in caso di errore.
        cfg['_motion_pending']=True
        # Richiede al controller l’ingresso nella modalita servo; un errore non viene ignorato.
        check(sender.rpc.ServoMoveStart(),'ServoMoveStart')
        messages.put(f'Servo continuo: {nodes[-1]["name"]}; invio richiesto 125 Hz')
        # Il controllo resta qui durante l’invio temporizzato dei riferimenti; nessuna IK nel ciclo veloce.
        stream_log=send_packets(sender.rpc,buffer,producer.health,stop,period=PERIOD,
                               on_sent=history.sent, records=stream_log)
        # L'ultimo tratto e gia a velocita e accelerazione nulle e mantiene la posa.
        # Richiede l’uscita dal servo dopo il tratto finale fermo; non scatta ancora la foto.
        check(sender.rpc.ServoMoveEnd(),'ServoMoveEnd')
        active=False  # pending resta vero finche la foto conferma il robot fermo
        return producer.bias.copy(),np.asarray(producer.final_q),stream_log
    finally:
        cfg.setdefault('_timing_records',[]).extend(stream_log)
        # Fa smettere il produttore di inserire punti anche se attendeva spazio nella coda.
        buffer.cancel.set()
        if active:
            try:
                # Richiesta SOFTWARE di arresto: la comunicazione o una RPC bloccata possono ritardarla.
                check(sender.rpc.StopMotion(),'StopMotion')
            except BaseException as exc:
                messages.put(f'STOP NON CONFERMATO: {exc}; verificare arresto fisico')
            try:
                # Richiede l’uscita dal servo dopo il tratto finale fermo; non scatta ancora la foto.
                check(sender.rpc.ServoMoveEnd(),'ServoMoveEnd dopo errore')
            except BaseException as exc:
                messages.put(f'Chiusura servo non confermata: {exc}')
            # Il worker esterno puo ripetere lo stop se una chiamata non ha confermato.
        # Aspetta al massimo un secondo la chiusura del produttore; non puo uccidere una RPC bloccata.
        producer.join(timeout=1)
        # Non autorizza altri trasferimenti se il vecchio produttore e ancora impegnato.
        if producer.is_alive():
            cfg['_planner_stuck']=True
            messages.put('RPC del pianificatore ancora attiva; nessuna ripartenza')


# FUNZIONE: keep_pose_fresh
# Con il robot FERMO aggiorna periodicamente la FK del bridge. Senza questo,
# la storia delle pose invecchia e accept_visual rifiuta ogni misura visiva
# (posa piu vicina oltre 250 ms dal frame): osservati = 0 anche con marker visibili.
# Va chiamata solo dal thread del worker, che e l'unico a usare il client RPC sender.
# seconds=None: continua fino a stop (usato in plan-only per tenere vivo il tracking).
def keep_pose_fresh(sender, bridge, cfg, stop, seconds=None, period=.05):
    end = None if seconds is None else time.monotonic() + seconds
    while not stop.is_set() and (end is None or time.monotonic() < end):
        T_now, _ = checked_camera(sender, cfg)
        bridge.update_robot_pose(T_now)
        stop.wait(period)
    if seconds is not None and stop.is_set():
        raise RuntimeError("Interruzione richiesta")


# FUNZIONE: photo_when_still
# DOPO il trasferimento aspetta stato fermo e giunti vicini a quelli finali.
# Richiede stabilita per un secondo, poi prende un nuovo frame con stato coerente.
# Solo qui impone target entro 5 pixel e tre centri ArUco PREVISTI nei margini.
# Se fallisce non cerca un'altra inquadratura e non muove di nuovo la camera:
# restituisce un errore e la sequenza non avanza alla posa successiva.
def photo_when_still(sender,bridge,cfg,expected,stop):
    deadline=time.monotonic()+5
    stable=None
    while time.monotonic()<deadline:
        if stop.is_set():
            raise StreamFault('Interruzione richiesta prima della foto')
        sender.assert_ready()
        # Nella sosta aggiorna anche la posa FK: il produttore del trasferimento e gia terminato.
        actual_T,q=checked_camera(sender,cfg)
        bridge.update_robot_pose(actual_T)
        if not valid_joints(q):
            raise StreamFault('Giunti non disponibili prima della foto')
        # Richiede fine movimento e giunti entro 0,15 gradi dal riferimento finale.
        if sender.done() and np.max(np.abs(np.array(q)-expected))<=.15:
            stable=time.monotonic() if stable is None else stable
            # Il criterio di robot fermo deve durare almeno un secondo prima della verifica fotografica.
            if time.monotonic()-stable>=1.:
                # Prende un frame successivo e ricontrolla che il robot non si muova durante la lettura.
                frame,state,T,actual=measured_state(sender,bridge,cfg)
                # QUI verifica target entro 5 pixel E almeno tre centri PREVISTI; se fallisce non si salva la foto.
                verify_photo_state(state,cfg)
                # Consegna al chiamante frame e dati approvati: il salvataggio su disco avviene subito dopo nel worker.
                return frame,state,actual
        else:
            stable=None
        stop.wait(.05)
    raise StreamFault('Robot non fermo entro il tempo previsto per la foto')


# FUNZIONE: photo_groups
# Raggruppa i molti nodi intermedi in trasferimenti tra foto.
# Crea prima un tratto iniziale a posizione nominale invariata per centrare
# ed assestare la prima foto. Ogni gruppo successivo termina con photo=True.
# Non e un ordinamento ottimizzato: mantiene l'ordine delle tappe.
def photo_groups(poses):
    # Primo gruppo a XYZ nominale invariato, per centrare/assestare e fotografare la posizione iniziale.
    yield [poses[0],poses[0]]  # centra e stabilizza anche la foto iniziale in ServoJ
    group=[poses[0]]
    for p in poses[1:]:
        group.append(p)
        # Chiude un trasferimento soltanto sulle tappe fotografiche, non a ogni nodo intermedio.
        if p['photo']:
            yield group
            group=[p]
    if len(group)>1:
        raise StreamFault('Percorso privo di tappa fotografica finale')


# FUNZIONE: worker
# COORDINATORE DEL ROBOT, eseguito separatamente dal ciclo della finestra.
# Ordine: controlli -> mappa -> piano -> tracking opzionale -> verifica raccordi.
# In plan-only termina senza abilitazione o moto. Con execute prepara foto,
# apre il secondo client RPC, aspetta cinque secondi e abilita.
# Per ogni gruppo esegue ServoJ, conferma lo scatto e salva PNG/JSON.
# Il finally gestisce stop se necessario, chiusure e log anche dopo errori.
def worker(cfg,bridge,K,D,stop,messages,plan_only):
    sender=planner=photos=None; log=[]; timing=[]
    cfg.update(_motion_pending=False,_planner_stuck=False,_timing_records=timing)
    try:
        # Client per i comandi servo e i controlli di avvio; non abilita subito il robot.
        sender=RobotIO(cfg['robot']['ip'])
        sender.assert_ready()
        if not sender.done():
            raise StreamFault('Robot gia in movimento')
        # Verifica che le API necessarie esistano nell’SDK importato, prima di tentare un moto.
        for name in ('ServoMoveStart','ServoJ','ServoMoveEnd','StopMotion'):
            if not callable(getattr(sender.rpc,name,None)):
                raise StreamFault(f'SDK privo di {name}')
        # Imposta esplicitamente la convenzione delle conversioni Euler del progetto.
        ac._EULER=cfg['robot']['euler_order']
        # Carica CAMERA->FLANGIA della camera selezionata; non e la trasformazione della camera nella base.
        H=np.asarray(ac.load_T_cam2gripper(cfg),float)
        if (H.shape!=(4,4) or not np.isfinite(H).all() or not np.allclose(H[3],[0,0,0,1]) or
            not np.allclose(H[:3,:3].T@H[:3,:3],np.eye(3),atol=1e-4) or
            not np.isclose(np.linalg.det(H[:3,:3]),1,atol=1e-4)):
            raise StreamFault('Hand-eye non rigida')
        cfg.update(_K=K,_D=D,_handeye=H)
        messages.put('Carico mappa: target durante il moto, almeno 3 centri previsti alle foto')
        # Crea o carica la mappa; il robot resta fermo durante la prima misura dei marker.
        cfg['_target']=persistent_target(sender,bridge,cfg,H,K,D)
        initial_T,_=checked_camera(sender,cfg)
        initial_point=(np.linalg.inv(initial_T)@np.r_[cfg['_target']['centroid'],1.])[:3]
        # La camera iniziale deve gia guardare il target entro 2 gradi: non esegue una grande rotazione automatica di ricerca.
        #if initial_point[2]<=0 or math.degrees(math.atan2(np.linalg.norm(initial_point[:2]),initial_point[2]))>2:
         #   raise StreamFault('Partire con camera gia orientata al centro entro 2 gradi')
        # Tutto il percorso viene pianificato prima dell’abilitazione del robot.
        poses,guard,q_start=plan_joints(sender,cfg,cfg['_target'],H,K,D,stop)
        T,q=checked_camera(sender,cfg)
        if np.max(np.abs(np.array(q)-q_start))>.05:
            raise StreamFault('Robot mosso durante pianificazione')
        # Collega mappa, FK e video. Se il tracking non parte, la mappa nota consente il fallback geometrico.
        bridge.start_tracking(cfg,T)
        measured_state(sender,bridge,cfg)
        # Verifica geometrica anche dei raccordi nominali C2, non solo dei nodi MoveJ.
        # Esamina un trasferimento fotografico alla volta: puo essere la verifica anticipata o l’esecuzione, secondo il blocco.
        last_pose=0.
        for nodes in photo_groups(poses):
            path=NominalPath([p['joints'] for p in nodes],[p['cam_T_base'][:3,3] for p in nodes],
                period=PERIOD,camera_speed=CAMERA_SPEED,velocity=VELOCITY*.65,
                acceleration=ACCELERATION*.5,jerk=JERK*.5)
            # Controlla anche le quintiche nominali tra nodi, non si limita alla sicurezza dei nodi stessi.
            for piece in path.pieces:
                previous=piece.at(0)
                for seconds in np.linspace(0,piece.duration,max(3,math.ceil(piece.duration/.04)+1)):
                    if stop.is_set():
                        raise StreamFault('Interruzione richiesta durante verifica piano')
                    # La verifica puo durare secondi: tiene fresca la FK per il tracking visivo.
                    if time.monotonic()-last_pose>.1:
                        T_now,_=checked_camera(sender,cfg)
                        bridge.update_robot_pose(T_now)
                        last_pose=time.monotonic()
                    q=piece.at(seconds)
                    safe_path(guard,previous.tolist(),q.tolist())
                    geometric_view(camera_for(sender,cfg,q),cfg)
                    previous=q
        # Scrive il riepilogo del piano nella cartella da cui e stato avviato il comando.
        Path('visuo_servo_piano.json').write_text(json.dumps({'mode':'continuous_servoj_visual_feedback',
            'period_s':PERIOD,'joint_speed_deg_s':VELOCITY,'joint_accel_deg_s2':ACCELERATION,
            'joint_jerk_deg_s3':JERK,'minimum_markers_during_motion':0,
            # Tre centri della MAPPA nei margini allo scatto, anche se sfocati.
            'minimum_predicted_markers_at_photo':MIN_ARUCO_VISIBILI,
            'photo_waypoints':[p['name'] for p in poses if p['photo']]},indent=2))
        # Termina prima di RobotEnable/Mode/ServoMoveStart: il piano non e un’autorizzazione di movimento.
        if plan_only:
            messages.put('Piano nominale e tracking verificati; nessuna abilitazione. Tempi servo da misurare sul sistema reale (Q per uscire)')
            # Robot fermo: continua ad aggiornare la FK cosi' il tracking visivo resta valutabile.
            keep_pose_fresh(sender,bridge,cfg,stop)
            return
        # Due client RPC: il pianificatore non deve bloccare il canale ServoJ.
        # Secondo client RPC per FK/IK del produttore: evita di condividere lo stesso client XML-RPC col sender.
        planner=RobotIO(cfg['robot']['ip'])
        planner.assert_ready()
        # Crea adesso cartella sessione e metadati; le foto saranno salvate alle tappe verificate.
        photos=WaypointPhotos(cfg['_photos_dir'],cfg,K,D,H)
        (photos.folder/'mappa_usata.json').write_text(json.dumps(cfg['_map_snapshot'],indent=2))
        messages.put(f'Foto: {photos.folder}. Partenza dopo 5 secondi; Q interrompe')
        # Cinque secondi di attesa interrompibile prima dell’abilitazione prevista dal flusso.
        keep_pose_fresh(sender,bridge,cfg,stop,5)
        q=sender.get_joints()
        if not valid_joints(q) or np.max(np.abs(np.array(q)-q_start))>.05:
            raise StreamFault('Robot mosso durante attesa')
        sender.assert_ready()
        actual_T,_=checked_camera(sender,cfg)
        bridge.update_robot_pose(actual_T)
        bridge.fast_measurement()
        if stop.is_set():
            raise StreamFault('Interruzione richiesta')
        # Configura la protezione reattiva del controller prima di iniziare i trasferimenti.
        sender.enable_backstop(cfg)
        # Richiede modalita automatica al controller, verificando il codice di risposta.
        check(sender.rpc.Mode(0),'Modalita automatica')
        # Abilita il robot solo dopo controlli, piano e attesa; non esegue reset degli allarmi.
        check(sender.rpc.RobotEnable(1),'Abilitazione')
        # Inizia senza correzione angolare aggiuntiva: matrice identita 3x3.
        bias=np.eye(3)
        # Esamina un trasferimento fotografico alla volta: puo essere la verifica anticipata o l’esecuzione, secondo il blocco.
        for nodes in photo_groups(poses):
            if cfg['_planner_stuck']:
                raise StreamFault('Pianificatore non terminato')
            # Trasferimento servo continuo e assestamento finale; restituisce correzione conservata e giunti finali.
            bias,expected,records=stream_leg(sender,planner,cfg,bridge,guard,nodes,bias,stop,messages,log)
            # Dopo ServoMoveEnd, ultimo controllo prima dello scatto; in caso di errore non passa al salvataggio.
            frame,state,q=photo_when_still(sender,bridge,cfg,expected,stop)
            # Puo azzerare lo stato di possibile moto solo dopo la conferma fotografica del robot fermo.
            cfg['_motion_pending']=False
            # Salva separatamente centri previsti, osservati e origine della stima; non etichetta SOLO MAPPA come nuova misura.
            cfg['_last_visual']={'visible_predicted_ids':state['ids'],
                # ID con osservazione attuale accettata; non coincide necessariamente con il conteggio usato per la foto.
                'observed_ids':state['observed_ids'],
                # ID previsti nei margini senza conferma nel frame corrente.
                'prediction_only_ids':state['prediction_only_ids'],
                # True solo per una correzione visiva accettata relativa al frame attuale, non per una vecchia misura conservata.
                'visual_feedback_fresh':state['visual_feedback_fresh'],
                # L’origine della posa distingue previsione da sola FK e FK con correzione visiva conservata.
                'pose_source':state['pose_source'],
                'visual_warning':state['visual_warning'],'marker_centers_px':state['centers_px'],
                # Proiezioni geometriche in pixel, non punti direttamente trovati nel frame.
                'predicted_centers_px':state['predicted_centers_px'],
                # Punteggio della patch per le misure visive disponibili; nessun punteggio inventato in SOLO MAPPA.
                'center_scores':state['center_scores'],
                # Permette di capire se l’errore allo scatto e previsto dal modello anziche sostenuto da nuova visione.
                'centering_error_source':state['pose_source'],
                # Scarto del centro target da (cx,cy): deve essere <=5 alla foto; non e errore di messa a fuoco.
                'centering_error_px':target_error_px(state,cfg),
                'estimated_camera_T_base_mm':state['T'].tolist(),'controller':'continuous_ServoJ'}
            # Scrive il PNG non annotato e il JSON della tappa approvata.
            saved=photos.save(frame,nodes[-1]['name'],q,nodes[-1]['joints'])
            messages.put(f'{nodes[-1]["name"]}: robot fermo, foto salvata in {saved}')
        messages.put('FINITO: trasferimenti ServoJ continui e foto alle tappe')
    except BaseException as exc:
        messages.put(f'INTERROTTO: {bridge.tracking_error or exc}')
    finally:
        if sender is not None:
            # Se una fase attiva non e stata conclusa con conferma di fermo, tenta StopMotion nel cleanup.
            if cfg['_motion_pending']:
                try:
                    # Richiesta SOFTWARE di arresto: la comunicazione o una RPC bloccata possono ritardarla.
                    check(sender.rpc.StopMotion(),'StopMotion finale')
                except BaseException as exc:
                    messages.put(f'STOP NON CONFERMATO: {exc}')
            sender.close()
        # Non chiude il socket mentre un thread e ancora dentro la sua RPC.
        # Non chiude il client di un produttore ancora bloccato nella propria RPC.
        if planner is not None and not cfg['_planner_stuck']:
            planner.close()
        if photos is not None:
            try:
                (photos.folder/'servo_log.json').write_text(json.dumps(log,indent=2,allow_nan=False))
                (photos.folder/'timing_servoj.json').write_text(json.dumps(timing,indent=2,allow_nan=False))
            except Exception as exc:
                messages.put(f'Log non salvato: {exc}')
        messages.put(None)


# FUNZIONE: setup_camera_cfg
# Seleziona e appiattisce il blocco della camera nel YAML tramite cameras.py.
# Per GC1020C prende hand-eye e camera_box dal suo blocco, non dalla RealSense.
# Aggiorna la cartella di calibrazione specifica. Non apre ancora la camera.
# I valori hardware effettivi dipendono da cameras.py, che non e incluso qui.
def setup_camera_cfg(cfg):
    """[NUOVO] Come load_config dell'acquire.py nuovo: cfg['camera'] diventa la
    camera scelta (acquisition.camera.type), piatta: chiavi comuni + il suo
    blocco. Hand-eye, ingombro e cartella di calibrazione seguono la camera:
      gc1020c   -> acquisition.camera.gc1020c.*            / DATA/camera_calibration_gc1020c/
    """
    # Seleziona le chiavi comuni e il blocco del tipo di camera indicato nel YAML.
    cam = resolve_camera_cfg(cfg.get("camera"))
    if cam["type"] != "gc1020c":      
        raise SystemExit("Configurare acquisition.camera.type: gc1020c")
    # Risale dalla cartella calibrazione della configurazione alla cartella dati del progetto.
    data_root = os.path.dirname(cfg["_calib_dir"])     # load_config: <data_root>/camera_calibration
    cfg["camera"] = cam
    # Passa al guard l’ingombro della camera selezionata, ottica e supporto secondo le misure configurate.
    cfg["camera_box"] = cam.get("camera_box")          # letto da collision_guard
    # Assicura che ac.load_T_cam2gripper legga la hand-eye di questa camera, non quella di un’altra.
    cfg["hand_eye"] = {"T_cam2gripper": cam.get("T_cam2gripper")}   # letto da ac.load_T_cam2gripper
    # La GC1020C usa la propria cartella di calibrazione, distinta dalla RealSense.
    cfg["_calib_dir"] = os.path.join(data_root, calibration_dir_name(cam["type"]))
    return cam["type"]


# FUNZIONE: resolve_intrinsics
# Se la camera offre intrinseche di fabbrica usa il resolver di acquire.
# Per GC1020C carica K e D dal JSON configurato o dalla cartella di calibrazione.
# Se il JSON riporta img_size, richiede che coincida con lo stream.
# Se la dimensione non e presente nel JSON, questo controllo non puo verificarla.
# Salva la provenienza delle intrinseche nei metadati; non esegue una nuova calibrazione.
def resolve_intrinsics(cfg, camera):
    """
    GC1020C: nessuna intrinseca di fabbrica -> calibrazione OpenCV della camera,
    <calib dir>/output/intrinsics/intrinsics_opencv.json, o il JSON indicato in
    acquisition.camera.gc1020c.intrinsics. La risoluzione salvata con la
    calibrazione (img_size) deve coincidere con quella dello stream."""

    def poll_frame(camera, last):
        """Legge un nuovo fotogramma della GC1020C senza attendere."""
        # last identifica l'ultimo fotogramma già letto.
        # Il timeout 0.0 evita di aspettare un'immagine nuova.
        img, idx = camera._next(last, 0.0)
        # Se non ci sono immagini nuove, conserva l'indice precedente.
        if img is None:
            return None, last
        # Restituisce una copia dell'immagine e il suo indice.
        return img.copy(), idx
    
    kind = cfg["camera"]["type"]
    # Prima usa un JSON esplicito, altrimenti il percorso di calibrazione convenzionale della camera.
    path = cfg["camera"].get("intrinsics") or os.path.join(
        cfg["_calib_dir"], "output", "calibration", "intrinsics_opencv.json")
    path = os.path.abspath(str(path))
    if not os.path.isfile(path):
        raise SystemExit(f"[intrinsics] la camera {kind} non ha intrinseche di fabbrica e "
                         f"{path} non esiste: eseguire `spectra calibrate --modality "
                         f"intrinsics` (o impostare acquisition.camera.{kind}.intrinsics)")
    # Legge la calibrazione gia esistente; questa funzione non stima K e D da nuove immagini.
    K, D = ac.load_intrinsics_override(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            size = json.load(f).get("img_size")
        size = [int(v) for v in size] if size else None
    except (OSError, ValueError, TypeError, AttributeError):
        size = None
    # Se img_size e disponibile deve coincidere con la risoluzione stream; non scala le intrinseche silenziosamente.
    if size is not None and size != [camera.w, camera.h]:
        raise SystemExit(f"[intrinsics] {path} calcolate a {size[0]}x{size[1]} ma la {kind} "
                         f"trasmette {camera.w}x{camera.h}: rifare le intrinseche o "
                         "correggere width/height")
    src = {"source": "override" if cfg["camera"].get("intrinsics") else "calibration",
           "path": path, "camera": kind, "width": camera.w, "height": camera.h,
           "K": np.asarray(K, float).tolist(), "D": np.asarray(D, float).ravel().tolist()}
    # Conserva percorso e valori letti per renderli rintracciabili nei metadati della sessione.
    cfg["_intrinsics_source"] = src
    print(f"[intrinsics] {kind}: {path}: fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
          f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    return K, D


# FUNZIONE: poll_frame
# Legge solo un'immagine nuova, senza bloccare aspettando un frame.
# RealSense usa poll_for_frames; GC1020C usa camera._next(last, 0.0).
# Restituisce (immagine BGR copiata, indice), oppure (None, last).
# L'interfaccia GC1020C e quella del cameras.py originale del progetto.
def poll_frame(camera, last):
    """[NUOVO] Fotogramma colore BGR nuovo SENZA attendere: (img, indice) o (None, last).

    RealSense: poll_for_frames della pipeline. GC1020C (Vimba X): il callback
    dello stream conserva l'ultimo frame completo; si legge solo se piu nuovo
    di `last` (timeout 0, quindi non blocca la finestra).
    """
    # Ramo della pipeline RealSense; la GC1020C segue la lettura _next piu sotto.
    if camera.kind == "realsense":
        frames = camera.pipe.poll_for_frames()
        if not frames:
            return None, last
        color = frames.get_color_frame()
        if not color:
            return None, last
        return np.asanyarray(color.get_data()).copy(), last + 1
    # Lettura non bloccante GC1020C: timeout zero e indice dell’ultimo frame gia letto.
    img, idx = camera._next(last, 0.0)
    return (None, last) if img is None else (img.copy(), idx)


# FUNZIONE: main
# AVVIO OPERATIVO: legge gli argomenti e la configurazione, apre la camera,
# crea la finestra e pubblica i frame. Al primo frame avvia worker in un thread.
# Senza execute passa plan_only=True: nessuna abilitazione o movimento.
# Il thread video mostra separatamente centri previsti e osservati.
# Q o chiusura finestra attivano stop; una RPC bloccata puo ritardarne la gestione.
# Alla fine chiude camera e finestra, non forza lo spegnimento del robot.
def main():
    # Prepara la lettura delle opzioni del comando; non si collega ancora a camera o robot.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=ac.DEFAULT_CONFIG)
    # Impedisce di selezionare contemporaneamente preview-only, plan-only ed execute.
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
    # ap.add_argument("--setup-id", required=True, help="Identifica montaggio fisso base/target")
    ap.add_argument("--vision-latency", type=float, default=0., help="Ritardo video misurato in secondi (0 se non compensato)")
    # Legge davvero gli argomenti digitati nel terminale e controlla che siano riconosciuti.
    args = ap.parse_args()
    if not math.isfinite(args.vision_latency) or not 0 <= args.vision_latency <= .2:
        ap.error("--vision-latency deve essere tra 0 e 0.2 s")
    # Copia indipendente della configurazione: il programma aggiungera dati temporanei senza modificare il YAML sul disco.
    cfg = copy.deepcopy(ac.load_config(args.config))  # [ORIGINALE] stesso YAML
    # Dimensione fisica del marker, unica fonte nel YAML; e in mm, non in metri.
    marker_mm = float(cfg["aruco"]["marker_length_mm"])
    if not math.isfinite(marker_mm) or marker_mm <= 0:
        ap.error("Lato marker YAML non valido")
    # Un override differente dal YAML e rifiutato per evitare incoerenze di scala.
    if args.marker_mm is not None and not math.isclose(args.marker_mm, marker_mm, abs_tol=1e-6):
        ap.error("--marker-mm deve coincidere con vision.aruco.marker_edge_length_mm")
    if cfg["robot"].get("tool") != 0 or cfg["robot"].get("user") != 0:
        raise SystemExit("Questa versione richiede tool=0 e user=0, come nel file ricevuto")
    if cfg["robot"].get("euler_order") not in ("ZYX", "XYZ"):
        raise SystemExit("Serve la convenzione Euler esplicita, non AUTO")
    if not cfg.get("collision", {}).get("enable"):
        raise SystemExit("collision.enable deve restare true")
    # Solo l’anteprima puo evitare la configurazione degli ID e la connessione robot nel worker.
    if not args.preview_only:
        with open(args.config, "r", encoding="utf-8") as config_file:
            full_config = yaml.safe_load(config_file) or {}
        cfg["waypoint_check"] = waypoint_options(
            full_config, cfg, marker_ids=args.marker_ids)
        if len(cfg["waypoint_check"]["marker_ids"]) != 6:
            ap.error("Questa versione richiede esattamente 6 ID ArUco")
        print("[waypoint]", cfg["waypoint_check"], flush=True)
    # Prepara impostazioni, hand-eye e calibrazione del tipo camera selezionato.
    kind = setup_camera_cfg(cfg)
    cfg["_calib_dir"]=r"C:\Users\spflg\Desktop\SpectraBreast-Full\DATA\camera_calibration_gc1020c"
    print(f"[camera] {kind}", flush=True)
    # Senza ingombro GC1020C non accetta modalita robot: il guard non potrebbe collocare la camera correttamente.
    if not cfg.get("camera_box") and not args.preview_only:
        raise SystemExit(f"acquisition.camera.{kind}.camera_box mancante: configurare gli ingombri reali")
    # Usa lo stream colore; non acquisisce una mappa di profondita in questo programma.
    cfg["camera"]["depth"] = False        # [MODIFICATO] solo video colore, niente depth
    # Campo legacy di configurazione; non definisce la velocita di questo stream ServoJ.
    cfg["robot"]["vel"] = VELOCITA
    # Cartella principale foto indicata da --photos-dir; default foto_waypoints.
    cfg["_photos_dir"] = args.photos_dir
    cfg.update(_marker_mm=marker_mm, _map_file=args.map_file, _vision_latency=args.vision_latency)
    # Segnale condiviso: non e un interruttore fisico, viene controllato dalle attivita software.
    stop = threading.Event()
    # Coda dei messaggi da worker alla finestra, distinta dalla coda dei pacchetti ServoJ.
    messages = queue.Queue()
    # ISTANZA REALE del bridge: usa il comportamento con centri previsti, non il vecchio ServoFrames.publish.
    bridge = FastFrames(stop)
    camera = thread = None
    window = "VISUO SERVO CONTINUO - video dal vivo - Q interrompe"
    status = "Apro la telecamera"
    finished = False
    try:
        # Apre la camera tramite cameras.py del progetto e avvia lo stream colore.
        camera = open_camera(cfg["camera"], depth=False)
        # Risoluzione e calibrazione sono controllate prima di avviare il worker sul primo frame.
        K, D = resolve_intrinsics(cfg, camera)
        # I controlli di inquadratura useranno le dimensioni effettive della camera aperta.
        cfg["camera"]["width"], cfg["camera"]["height"] = int(camera.w), int(camera.h)
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        cv.resizeWindow(window, 960, int(round(960 * camera.h / camera.w)))
        last_frame = time.monotonic()
        frame = None
        frame_idx = 0
        while not finished:
            # Cerca un nuovo frame senza bloccare il ciclo della finestra.
            new, frame_idx = poll_frame(camera, frame_idx)
            if new is not None:
                frame = new
                last_frame = time.monotonic()
                # Consegnando il frame aggiorna anche le previsioni e, se disponibile, il tracking opzionale dei centri.
                bridge.publish(frame)
            if time.monotonic()-last_frame > 3:
                raise RuntimeError("Video assente per oltre 3 secondi")
            if frame is not None:
                # Avvia il worker una sola volta, dopo il primo frame; in anteprima non apre il robot.
                if thread is None and not args.preview_only:
                    thread = threading.Thread(target=worker,
                        args=(cfg, bridge, K, D, stop, messages, not args.execute))
                    thread.start()      # [NUOVO] parte da solo, senza S o P
                if args.preview_only:
                    status = "SOLO VIDEO: nessuna connessione al robot"
                # Aggiunge il testo a una COPIA per la finestra; il frame delle fotografie rimane senza scritte.
                shown = frame.copy()
                cv.putText(shown, status[:100], (15, 30), cv.FONT_HERSHEY_SIMPLEX,
                           0.65, (0, 255, 0), 2)
                # Mostra i conteggi per trasparenza: meno di tre durante il trasferimento adesso e ammesso.
                if bridge.fast_state is not None:
                    current = bridge.fast_state
                    mode = "VISIONE + MAPPA" if current["visual_feedback_fresh"] else "SOLO MAPPA"
                    info = f"Centri previsti: {len(current['ids'])} | osservati: {len(current['observed_ids'])} | {mode}"
                    cv.putText(shown, info, (15, 60), cv.FONT_HERSHEY_SIMPLEX,
                               0.55, (0, 255, 255), 2)
                    # Mostra il motivo per cui il feedback visivo non viene accettato (vuoto se tutto ok).
                    if current.get("visual_warning"):
                        cv.putText(shown, str(current["visual_warning"])[:110], (15, 90),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv.imshow(window, shown)
            while not messages.empty():
                msg = messages.get()
                if msg is None:
                    finished = True
                else:
                    status = msg
                    print(msg, flush=True)
            # Legge Q dalla finestra e avvia l’uscita; non e un arresto hardware.
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
            if cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stop.set()
        # Non inviamo comandi RPC dal thread della finestra: il worker gestisce stop.
        # Una chiamata SDK bloccata puo ritardarlo: l'arresto fisico resta necessario.
        # Aspetta brevemente il worker nel cleanup senza pretendere di poter interrompere una RPC bloccata.
        if thread is not None:
            thread.join(timeout=2)
            if thread.is_alive():
                print("Operazione SDK ancora attiva: verificare il robot; usare arresto fisico se necessario")
        if camera is not None:
            # Ferma la camera e libera lo stream; non e un comando di arresto del braccio.
            camera.stop()
        # Chiude le finestre OpenCV create dal programma.
        cv.destroyAllWindows()
        while not messages.empty():
            msg = messages.get()
            if msg is not None:
                print(msg, flush=True)


# Python esegue main solo avviando questo file direttamente, non quando viene importato come modulo.
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interruzione richiesta")
    except Exception as exc:
        raise SystemExit(f"ERRORE: {exc}")