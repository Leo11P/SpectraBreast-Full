"""
Crea un file .rrd con la point cloud spettrale SpectraBreast, apribile con il viewer Rerun.

Uso:  python visualizza_spectrabreast.py <cartella_con_i_file> [nome_campione]
      es. python visualizza_spectrabreast.py RESULTS/SB021/registration spectral_pointcloud_SB021

Nel viewer:
  - vista "Banda selezionata": colore = riflettanza alla lunghezza d'onda scelta sulla timeline
    "lunghezza_onda_nm" (colormap viridis, contrasto stirato per banda tra 1° e 99° percentile)
  - vista "RGB approssimato": bande ~640/550/460 nm
"""
import sys
from pathlib import Path
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
name = sys.argv[2] if len(sys.argv) > 2 else "spectral_pointcloud_SAMPLE2"

# meta = [n_punti, 3, n_punti, n_bande]
n, _, _, n_bands = np.load(folder / f"{name}_meta.npy").tolist()
wl = np.load(folder / f"{name}_wavelengths.npy")
xyz = np.fromfile(folder / f"{name}_xyz.bin", dtype=np.float32).reshape(n, 3)
spectra = np.memmap(folder / f"{name}_spectra.bin", dtype=np.float32, mode="r", shape=(n, n_bands))

_viridis = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], np.float32)
def colormap(v):
    v = np.clip(np.nan_to_num(v), 0, 1) * (len(_viridis) - 1)
    i = np.minimum(v.astype(int), len(_viridis) - 2)
    t = (v - i)[:, None]
    return ((1 - t) * _viridis[i] + t * _viridis[i + 1]).astype(np.uint8)

def stretch(v):
    lo, hi = np.nanpercentile(v, [1, 99])
    return (v - lo) / max(hi - lo, 1e-6)

def band(nm):
    return int(np.argmin(np.abs(wl - nm)))

TIMELINE = "lunghezza_onda_nm"
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(name="Banda selezionata", origin="/", contents="/per_banda"),
        rrb.Spatial3DView(name="RGB approssimato", origin="/", contents="/pseudo_rgb"),
    ),
    rrb.TimePanel(expanded=True, timeline=TIMELINE),
    collapse_panels=True,
)

rr.init("spectrabreast_spectral")
out = folder / f"{name}_spettrale.rrd"
rr.save(out, default_blueprint=blueprint)

radius = rr.Radius.ui_points(2.0)

# RGB approssimato (statico)
rgb = np.stack([stretch(np.asarray(spectra[:, band(b)])) for b in (640, 550, 460)], axis=1)
rr.log("pseudo_rgb", rr.Points3D(xyz, colors=(np.clip(np.nan_to_num(rgb), 0, 1) * 255).astype(np.uint8), radii=radius), static=True)

# Geometria statica + un colore per ogni banda sulla timeline in nm
rr.log("per_banda", rr.Points3D(xyz, radii=radius), static=True)
for i in range(n_bands):
    rr.set_time(TIMELINE, sequence=int(round(float(wl[i]))))
    rr.log("per_banda", rr.Points3D.from_fields(colors=colormap(stretch(np.asarray(spectra[:, i])))))

print("Creato:", out)
