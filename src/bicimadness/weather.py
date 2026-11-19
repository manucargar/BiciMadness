"""Temperatura y precipitación por hora en Madrid-Retiro (enero 2021 - febrero 2023).

Fuente: Meteostat (https://meteostat.net), estación Madrid / Retiro (WMO 08222), que redistribuye
las observaciones de AEMET. Licencia CC BY-NC 4.0 (hay que citarla). Los datos vienen en UTC y se
pasan a hora de Madrid, igual que salidas_por_hora_top50.csv: '2021-06-01 08:00' = de 08:00 a 08:59.

Salida: data/raw/meteo_horario.csv con
  fecha_hora    hora local de Madrid
  temperatura   ºC
  precipitacion mm (= l/m2) caídos en esa hora
"""
import csv
import gzip
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PROYECTO = Path(__file__).resolve().parents[2]  # raíz del repositorio
CARPETA = PROYECTO / "data" / "raw" / "meteo"  # ficheros anuales descargados
SALIDA = PROYECTO / "data" / "raw" / "meteo_horario.csv"
ESTACION = "08222"  # Madrid / Retiro
INICIO, FIN = datetime(2021, 1, 1), datetime(2023, 2, 28, 23)
URL = "https://bulk.meteostat.net/v2/hourly/{anio}/{estacion}.csv.gz"
MADRID = ZoneInfo("Europe/Madrid")


def descargar():
    CARPETA.mkdir(exist_ok=True)
    for anio in range(INICIO.year, FIN.year + 1):
        fichero = CARPETA / f"{ESTACION}_{anio}.csv.gz"
        if not fichero.exists():
            print(f"descargando {anio}")
            urllib.request.urlretrieve(URL.format(anio=anio, estacion=ESTACION), fichero)


def leer():
    """-> {hora local 'YYYY-MM-DD HH:00': ([temperaturas], [precipitaciones])}"""
    datos = defaultdict(lambda: ([], []))
    for fichero in sorted(CARPETA.glob(f"{ESTACION}_*.csv.gz")):
        with gzip.open(fichero, "rt") as f:
            # columnas: date, hour (UTC), temp, dwpt, rhum, prcp, snow, wdir, wspd, wpgt, pres, tsun, coco
            for fila in csv.reader(f):
                utc = datetime.strptime(fila[0], "%Y-%m-%d").replace(hour=int(fila[1]), tzinfo=timezone.utc)
                temp, prec = datos[utc.astimezone(MADRID).strftime("%Y-%m-%d %H:00")]
                if fila[2]:
                    temp.append(float(fila[2]))
                if fila[5]:
                    prec.append(float(fila[5]))
    return datos


def main():
    descargar()
    datos = leer()
    filas, huecos = [], []
    t = INICIO
    while t <= FIN:
        clave = t.strftime("%Y-%m-%d %H:00")
        temp, prec = datos.get(clave, ([], []))
        # en el cambio de hora de octubre dos horas UTC caen en la misma hora local: media de temperatura, suma de lluvia
        fila = [round(sum(temp) / len(temp), 1) if temp else None, round(sum(prec), 1) if prec else None]
        if None in fila:
            huecos.append(clave)
        filas.append([clave, *fila])
        t += timedelta(hours=1)

    # huecos (horas sin observación o las 02:00 que no existen en marzo): valor de la hora anterior,
    # o de la siguiente si falta al principio
    for col in (1, 2):
        for orden in (filas, filas[::-1]):
            anterior = None
            for fila in orden:
                if fila[col] is None:
                    fila[col] = anterior
                anterior = fila[col]

    with open(SALIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fecha_hora", "temperatura", "precipitacion"])
        w.writerows(filas)
    print(f"{SALIDA.name}: {len(filas)} horas, {len(huecos)} rellenadas con la hora más cercana")


if __name__ == "__main__":
    main()
