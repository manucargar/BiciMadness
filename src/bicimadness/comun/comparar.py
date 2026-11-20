"""Lectura de ejecuciones guardadas y T-test pareado, para comparar_estudios.py y comparar_algoritmos.py."""
import csv
import json

import numpy as np
from scipy.stats import ttest_rel

from .datos import RESULTADOS


def ejecuciones(algoritmo):
    """Ejecuciones terminadas de results/<algoritmo>/, en orden cronológico: [{resumen..., "carpeta": ruta}]."""
    res = []
    for c in sorted((RESULTADOS / algoritmo).glob("*/resumen.csv")):
        with open(c, encoding="utf-8") as f:
            fila = next(csv.DictReader(f))
        for k in ("mae", "rmse", "r2", "mae_referencia", "segundos_total", "segundos_entreno_por_bloque"):
            fila[k] = float(fila[k])
        fila.setdefault("id_estacion", "sí" if fila["modo"] == "global" else "no")
        fila["carpeta"] = c.parent
        res.append(fila)
    return res


def algoritmos():
    """Carpetas de results/ que contienen ejecuciones con el formato nuevo (config.json + resumen.csv)."""
    return sorted(d.name for d in RESULTADOS.iterdir()
                  if d.is_dir() and any(d.glob("*/config.json")) and any(d.glob("*/resumen.csv")))


def mae_por_estacion_bloque(carpeta):
    """{(estación, bloque): MAE} de una ejecución (y lo mismo de la referencia de ayer)."""
    modelo, ref = {}, {}
    with open(carpeta / "resultados_folds.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            modelo[r["id_estacion"], r["bloque"]] = float(r["mae"])
            ref[r["id_estacion"], r["bloque"]] = float(r["mae_referencia"])
    return modelo, ref


def ttest(a, b):
    """T-test pareado sobre las (estación, bloque) que tienen las dos ejecuciones. -> (t, p, n)."""
    claves = sorted(a.keys() & b.keys())
    if len(claves) < 2:
        return float("nan"), float("nan"), len(claves)
    t, p = ttest_rel([a[c] for c in claves], [b[c] for c in claves])
    return float(t), float(p), len(claves)


def corto(datos):
    return datos.removeprefix("salidas_por_hora_").removesuffix(".csv")


def escribir(ruta, filas):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def tabla(filas, columnas):
    """Imprime filas (dicts) con las columnas dadas, alineadas."""
    anchos = {c: max(len(c), *(len(_fmt(f[c])) for f in filas)) for c in columnas}
    print("  ".join(f"{c:<{anchos[c]}}" for c in columnas))
    for f in filas:
        print("  ".join(f"{_fmt(f[c]):<{anchos[c]}}" for c in columnas))


def _fmt(v):
    if isinstance(v, float):
        return "" if np.isnan(v) else f"{v:.4g}" if abs(v) < 1e-3 and v else f"{v:.3f}"
    return str(v)


def leer_config(carpeta):
    return json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
