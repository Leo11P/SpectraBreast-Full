"""Matematica e traiettorie del servoing: nessuna connessione hardware."""
from dataclasses import dataclass
import math
import queue
import threading
import time
import numpy as np


def interaction_matrix(x, y, depth_mm):
    """L del punto normalizzato: twist [mm, mm, mm, rad, rad, rad].

    Per una scena fissa, s_dot = L @ v_camera, riferito alla camera.
    Il comando applicato dal programma usa solo le colonne omega_x/omega_y.
    """
    if not np.isfinite([x, y, depth_mm]).all() or depth_mm <= 0:
        raise ValueError("Punto camera non finito o dietro la camera")
    return np.array([[-1/depth_mm, 0, x/depth_mm, x*y, -(1+x*x), y],
                     [0, -1/depth_mm, y/depth_mm, 1+y*y, -x*y, -x]], float)


def pointing_increment(point_camera_mm, gain=0.55, damping=0.03,
                       max_angle_rad=np.deg2rad(0.35)):
    """Incremento angolare camera, NON velocita: DLS a passi discreti.

    Centro desiderato = asse ottico, ovvero s*=[0,0]. Non corregge la
    posizione della camera o il roll: queste componenti sono del percorso.
    """
    P = np.asarray(point_camera_mm, float)
    if P.shape != (3,) or not np.isfinite(P).all() or P[2] <= 0:
        raise ValueError("Centro target camera non valido")
    if not (0 < gain <= 1 and damping > 0 and 0 < max_angle_rad < 0.1):
        raise ValueError("Parametri controllo non validi")
    e = P[:2]/P[2]
    A = interaction_matrix(e[0], e[1], P[2])[:, 3:5]
    delta_xy = -gain * A.T @ np.linalg.solve(A @ A.T + damping**2*np.eye(2), e)
    delta = np.array([delta_xy[0], delta_xy[1], 0.0])
    norm = np.linalg.norm(delta)
    if norm > max_angle_rad:
        delta *= max_angle_rad/norm
    return delta


def rotation_exp(w):
    """Rodrigues, radianti; la matrice viene moltiplicata a DESTRA di R_camera."""
    w = np.asarray(w, float)
    if w.shape != (3,) or not np.isfinite(w).all():
        raise ValueError("Rotazione non valida")
    angle = np.linalg.norm(w)
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if angle < 1e-8:
        return np.eye(3) + W + 0.5*W@W
    return np.eye(3) + np.sin(angle)/angle*W + (1-np.cos(angle))/angle**2*(W@W)


def marker_center_px(vertices):
    """Centro richiesto: media aritmetica dei quattro vertici nell'immagine."""
    p = np.asarray(vertices, float)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError('Servono quattro vertici 2D finiti per il centro')
    return p.mean(axis=0)


def center_in_image(vertices, width, height, margin=40):
    """Solo il centro deve stare nei margini; i vertici non sono vincolati al margine."""
    c = marker_center_px(vertices)
    return bool(margin <= c[0] < width-margin and margin <= c[1] < height-margin)


def visible_marker_ids(points_camera, pixels, width, height, margin=40):
    """Conta i CENTRI dai quattro vertici proiettati, mantenendo profondita valida."""
    result = []
    for i, vertices in pixels.items():
        P = np.asarray(points_camera[i], float)
        if P.shape != (4, 3) or not np.isfinite(P).all() or np.any(P[:, 2] <= 1):
            continue
        try:
            if center_in_image(vertices, width, height, margin):
                result.append(i)
        except ValueError:
            continue
    return result


class StreamFault(RuntimeError):
    pass


class Quintic:
    """Polinomio su tempo normalizzato, con posizione/velocita/accelerazione ai capi."""
    def __init__(self, q0, v0, a0, q1, v1, a1, duration):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('Durata non valida')
        values = np.array([q0,v0,a0,q1,v1,a1],float)
        if values.shape != (6,6) or not np.isfinite(values).all():
            raise ValueError('Sei vettori articolari finiti richiesti')
        self.duration = float(duration)
        T = self.duration
        self.c = np.zeros((6,6))
        self.c[:3] = [values[0], values[1]*T, values[2]*T*T/2]
        rhs = np.array([values[3]-self.c[:3].sum(axis=0),
                        values[4]*T-self.c[1]-2*self.c[2],
                        values[5]*T*T-2*self.c[2]])
        self.c[3:] = np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20]],float),rhs)

    def at(self, seconds, derivative=0):
        u = float(np.clip(seconds/self.duration,0,1))
        c = self.c.copy()
        for _ in range(derivative):
            c = c[1:]*np.arange(1,len(c))[:,None]/self.duration
        return np.polynomial.polynomial.polyval(u,c)

    def peak(self, derivative):
        """Massimo analitico per giunto: capi e radici della derivata successiva."""
        result = []
        for j in range(6):
            c = self.c[:,j].copy()
            for _ in range(derivative):
                c = np.arange(1,len(c))*c[1:]/self.duration
            dc = np.arange(1,len(c))*c[1:]
            roots = np.polynomial.polynomial.polyroots(dc) if np.any(dc) else []
            candidates = [0.,1.]+[float(r.real) for r in roots if abs(r.imag)<1e-9 and 0<r.real<1]
            result.append(max(abs(np.polynomial.polynomial.polyval(u,c)) for u in candidates))
        return np.array(result)

    def require_limits(self, velocity, acceleration, jerk):
        for order, limit, label in [(1,velocity,'velocita'),(2,acceleration,'accelerazione'),(3,jerk,'jerk')]:
            if np.max(self.peak(order)) > limit+1e-8:
                raise StreamFault(f'Limite {label} articolare superato')


class NominalPath:
    """Interpola i nodi densi senza fermarsi ai nodi interni; v=a=0 alle foto."""
    def __init__(self, joints, positions, period=.008, camera_speed=5., velocity=3., acceleration=10., jerk=100.):
        q=np.asarray(joints,float); p=np.asarray(positions,float)
        if q.ndim!=2 or q.shape[1]!=6 or len(q)<2 or p.shape!=(len(q),3):
            raise ValueError('Nodi percorso non validi')
        durations=np.maximum(.16,np.maximum(np.linalg.norm(np.diff(p,axis=0),axis=1)/camera_speed,
                        np.max(np.abs(np.diff(q,axis=0)),axis=1)/velocity*2))
        for _ in range(10):
            # Durate multiple del periodo: nessun pacchetto duplicato ai confini.
            durations=np.ceil(durations/period)*period
            slopes=np.diff(q,axis=0)/durations[:,None]
            v=np.zeros_like(q)
            for k in range(1,len(q)-1):
                same=slopes[k-1]*slopes[k]>0
                v[k,same]=2/(1/slopes[k-1,same]+1/slopes[k,same])
            pieces=[Quintic(q[k],v[k],np.zeros(6),q[k+1],v[k+1],np.zeros(6),durations[k])
                    for k in range(len(q)-1)]
            factor=max(1.,max(np.max(x.peak(1))/velocity for x in pieces),
                       max(math.sqrt(np.max(x.peak(2))/acceleration) for x in pieces),
                       max((np.max(x.peak(3))/jerk)**(1/3) for x in pieces))
            if factor<=1+1e-8:
                self.pieces=pieces; self.ends=np.cumsum(durations); self.duration=float(self.ends[-1]); return
            durations*=factor*1.02
        raise StreamFault('Impossibile temporizzare il percorso')

    def at(self,t,derivative=0):
        index=min(int(np.searchsorted(self.ends,t,side='right')),len(self.pieces)-1)
        start=0. if index==0 else self.ends[index-1]
        return self.pieces[index].at(t-start,derivative)


@dataclass(frozen=True)
class Packet:
    index: int
    joints: tuple
    valid_until: float
    last: bool = False


class PacketBuffer:
    """Coda limitata. Gli errori del produttore non vengono nascosti dai pacchetti rimasti."""
    def __init__(self, capacity=60):
        self.queue=queue.Queue(capacity)
        self.fault=None
        self.cancel=threading.Event()

    def put(self, packet):
        while not self.cancel.is_set():
            try:
                self.queue.put(packet,timeout=.02); return
            except queue.Full:
                pass
        raise StreamFault('Produzione annullata')

    def take_now(self):
        if self.fault:
            raise StreamFault(self.fault)
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            raise StreamFault('Buffer servo esaurito: niente invii in ritardo o recuperi a raffica')


def send_packets(rpc, buffer, health, stop, period=.008, lateness=.004,
                 clock=time.monotonic, wait=None, on_sent=None, records=None):
    """Solo ServoJ e controlli rapidi; nessuna IK/FK/immagine in questo ciclo.

    Il chiamante gestisce ServoMoveStart/End e StopMotion anche per risposte perse.
    Il primo pacchetto deve essere la posizione attuale. Non salta pacchetti.
    """
    if not (0.001 <= period <= .016 and 0<lateness<period):
        raise ValueError('Periodo o tolleranza temporale non validi')
    if wait is None:
        wait=lambda seconds:stop.wait(seconds)
    deadline=clock(); expected=0
    records=[] if records is None else records
    while True:
        if stop.is_set():
            raise StreamFault('Interruzione richiesta')
        health()
        packet=buffer.take_now()
        if packet.index!=expected:
            raise StreamFault('Sequenza pacchetti non consecutiva')
        remaining=deadline-clock()
        if remaining>0:
            wait(remaining)
        now=clock()
        if stop.is_set():
            raise StreamFault('Interruzione richiesta')
        if now-deadline>lateness:
            raise StreamFault('Scadenza ServoJ mancata')
        if now>packet.valid_until:
            raise StreamFault('Pacchetto basato su misura visiva scaduta')
        # Ricontrollo immediatamente prima dell'invio, dopo l'attesa.
        health()
        began=clock()
        if began-deadline>lateness:
            raise StreamFault('Scadenza mancata durante i controlli')
        if began>packet.valid_until:
            raise StreamFault('Pacchetto scaduto durante i controlli')
        result=rpc.ServoJ(list(packet.joints),[0.,0.,0.,0.],cmdT=period)
        finished=clock()
        if result!=0:
            raise StreamFault(f'ServoJ: codice {result}')
        if finished-began>period:
            raise StreamFault('RPC ServoJ piu lenta del periodo')
        if on_sent:
            on_sent(packet,finished)
        records.append({'index':expected,'lateness_s':began-deadline,'rpc_s':finished-began})
        if packet.last:
            return records
        expected+=1; deadline+=period