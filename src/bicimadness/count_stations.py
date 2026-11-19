"""Salidas por hora, nombre y coordenadas de TODAS las estaciones de BiciMAD (2021-2023).

Es el mismo recorrido de preprocess.py (mismos lectores y filtro de 2 minutos), pero sin
quedarse con las 50 con más salidas. Lo usa select_stations.py para elegir las 60 estaciones.

Salidas (en data/cache/, se regeneran con  python src/bicimadness/count_stations.py):
  - estaciones_todas.csv     id, nombre, lat, lon (mediana de las posiciones de desbloqueo),
                             salidas totales y salidas por mes
  - salidas_todas.csv.gz     (fecha_hora, id_estacion, salidas) solo de las horas con alguna salida
"""
import csv
import gzip
import io
import json
import re
import zipfile
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

from preprocess import CARPETAS, MADRID, MIN_SEGUNDOS, PATRON, PROYECTO, ficheros_json

CACHE = PROYECTO / "data" / "cache"
COORDS = re.compile(r"'coordinates':\s*\[(-?\d+\.\d+),\s*(-?\d+\.\d+)\]")  # la 1ª es la de desbloqueo
MAX_PUNTOS = 1000  # posiciones guardadas por estación para la mediana (de sobra)

conteo = Counter()  # (hora 'YYYY-MM-DDTHH', estación) -> salidas
nombres = {}
puntos = defaultdict(list)  # estación -> [(lat, lon)]


def leer_csv(fichero):
    """Como preprocess.leer_csv, pero también guarda la posición de desbloqueo."""
    with open(fichero, encoding="utf-8", errors="replace") as f:
        corrupto = f.readline().lstrip("﻿").startswith("fecha,")
        for linea in f:
            if corrupto:
                linea = linea.replace(";", "").replace('"', "")
            m = PATRON.search(linea)
            if not m or float(m.group(1)) * 60 < MIN_SEGUNDOS:
                continue
            est = m.group(3)
            c = COORDS.search(linea)
            if c and len(puntos[est]) < MAX_PUNTOS:
                lon, lat = float(c.group(1)), float(c.group(2))
                if 40.2 < lat < 40.6 and -3.9 < lon < -3.5:  # descarta GPS sin señal (0,0) o fuera de Madrid
                    puntos[est].append((lat, lon))
            yield m.group(2), est, re.sub(r"^\d+ - ", "", m.group(4).strip())


def leer_json(fichero):
    """Como preprocess.leer_json, pero leyendo línea a línea (el JSON entero no cabe en 3 GB de RAM)."""
    if fichero.suffix == ".zip":
        z = zipfile.ZipFile(fichero)
        nombre = next(n for n in z.namelist() if n.endswith(".json") and not n.startswith("__MACOSX"))
        lineas = io.TextIOWrapper(z.open(nombre), encoding="utf-8")
    else:
        lineas = open(fichero, encoding="utf-8")
    with lineas:
        for linea in lineas:
            if not linea.strip():
                continue
            r = json.loads(linea)
            if r["travel_time"] >= MIN_SEGUNDOS:
                utc = datetime.strptime(r["unplug_hourTime"][:13], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
                yield utc.astimezone(MADRID).strftime("%Y-%m-%dT%H"), str(r["idunplug_station"]), None


def sumar(fichero, viajes):
    n = 0
    for hora, est, nombre in viajes:
        conteo[hora, est] += 1
        if nombre:
            nombres.setdefault(est, nombre)
        n += 1
    print(f"{fichero.name}: {n} viajes", flush=True)


def main():
    for fichero in sorted(f for c in CARPETAS for f in c.glob("*.csv")):
        sumar(fichero, leer_csv(fichero))
    for fichero in ficheros_json({h[:7] for h, _ in conteo}):
        sumar(fichero, leer_json(fichero))

    CACHE.mkdir(exist_ok=True)
    total, por_mes = Counter(), defaultdict(Counter)
    for (hora, est), n in conteo.items():
        total[est] += n
        por_mes[est][hora[:7]] += n
    meses = sorted({h[:7] for h, _ in conteo})

    with open(CACHE / "estaciones_todas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id_estacion", "nombre", "lat", "lon", "n_posiciones", "salidas_totales", *meses])
        for est, _ in total.most_common():
            p = puntos.get(est, [])
            lat = round(statistics.median(x for x, _ in p), 6) if p else ""
            lon = round(statistics.median(y for _, y in p), 6) if p else ""
            w.writerow([est, nombres.get(est, ""), lat, lon, len(p), total[est], *(por_mes[est][m] for m in meses)])

    with gzip.open(CACHE / "salidas_todas.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fecha_hora", "id_estacion", "salidas"])
        for (hora, est), n in sorted(conteo.items()):
            w.writerow([hora, est, n])
    print(f"{len(total)} estaciones, {len(meses)} meses -> {CACHE}/", flush=True)


if __name__ == "__main__":
    main()
