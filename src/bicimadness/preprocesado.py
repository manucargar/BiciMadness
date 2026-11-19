"""Salidas por hora de las 50 estaciones de BiciMAD con más salidas (2021-2023).

Salida: salidas_por_hora_top50.csv -> una fila por (hora, estación) con sus salidas,
más variables de calendario (día de la semana, mes, festivo 0/1) para el windowing.
Sábado y domingo no se marcan aparte: ya los da dia_semana.

Fuentes:
  - CSV trips_AA_MM_Mes.csv (junio 2021 - febrero 2023), hora local de Madrid
  - JSON AAAAMM_movements.json / .zip (formato antiguo, enero - mayo 2021), hora en UTC
Todos en las CARPETAS (bicimad_2021, bicimad_2022, bicimad_2023).
Solo cuentan viajes de MIN_SEGUNDOS o más: los CSV traen muchos desbloqueos fallidos de
segundos que los JSON no incluyen (con el filtro, junio 2021 cuadra entre ambos formatos).
"""
import csv
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

RAIZ = Path(__file__).resolve().parents[2]
DATOS_BRUTOS = RAIZ / "data" / "raw"
DATOS_PROCESADOS = RAIZ / "data" / "processed"
CARPETAS = [Path(c) for c in sys.argv[1:]] or [
    DATOS_BRUTOS / f"bicimad_{anio}" for anio in (2021, 2022, 2023)
]
SALIDA = DATOS_PROCESADOS / "salidas_por_hora_top50.csv"
CALENDARIO = DATOS_BRUTOS / "calendario_laboral.csv"
TOP_N = 50
MIN_SEGUNDOS = 120
MADRID = ZoneInfo("Europe/Madrid")

# fecha -> laborable / sabado / domingo / festivo (calendario laboral del Ayuntamiento de Madrid)
with open(CALENDARIO, encoding="utf-8-sig") as f:
    TIPO_DIA = {datetime.strptime(fila[0], "%d/%m/%Y").date(): fila[2].strip().lower()
                for fila in csv.reader(f, delimiter=";") if fila and fila[0][:1].isdigit()}

# trip_minutes;geolocation_unlock;...;unlock_date;locktype;unlocktype;...;lock_date;station_unlock;dock_unlock;unlock_station_name
# ponytail: regex en vez de csv porque el fichero de octubre viene corrupto (CSV con comas troceado por ';')
PATRON = re.compile(
    r"([\d.]+)[;,]\{'type'.*?"                                       # trip_minutes
    r"(\d{4}-\d\d-\d\dT\d\d):\d\d:\d\d+[;,][A-Z_]*[;,][A-Z_]*[;,]"  # unlock_date (hora) + locktype + unlocktype
    r".*?\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d+[;,]"                        # geo/address lock + lock_date (octubre: '15;0' -> '150')
    r"(\d+)[;,]\d*[;,]([^;,]*)"                                      # station_unlock, dock, nombre
)


def leer_csv(fichero):
    """-> (hora local 'YYYY-MM-DDTHH', estación, nombre) de cada viaje desde una estación."""
    with open(fichero, encoding="utf-8", errors="replace") as f:
        cabecera = f.readline()
        corrupto = cabecera.lstrip("\ufeff").startswith("fecha,")  # octubre 2021: separador ',' y ';' metidos en medio del texto
        for linea in f:
            if corrupto:
                linea = linea.replace(";", "").replace('"', "")
            m = PATRON.search(linea)
            if m and float(m.group(1)) * 60 >= MIN_SEGUNDOS:  # desde 2022 el nombre viene como '53 - Plaza de Lavapiés'
                yield m.group(2), m.group(3), re.sub(r"^\d+ - ", "", m.group(4).strip())


def leer_json(fichero):
    """Igual que leer_csv para el formato antiguo (una línea JSON por viaje, hora en UTC)."""
    if fichero.suffix == ".zip":
        with zipfile.ZipFile(fichero) as z:
            nombre = next(n for n in z.namelist() if n.endswith(".json") and not n.startswith("__MACOSX"))
            lineas = z.read(nombre).decode("utf-8").splitlines()
    else:
        lineas = fichero.read_text(encoding="utf-8").splitlines()
    for linea in lineas:
        if not linea.strip():
            continue
        r = json.loads(linea)
        if r["travel_time"] >= MIN_SEGUNDOS:
            utc = datetime.strptime(r["unplug_hourTime"][:13], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            yield utc.astimezone(MADRID).strftime("%Y-%m-%dT%H"), str(r["idunplug_station"]), None


def ficheros_json(meses_csv):
    """Un fichero *_movements por mes (zip o json), solo de meses que no estén ya en los CSV."""
    elegidos = {}
    for f in sorted(f for c in CARPETAS for f in c.glob("*_movements*")):
        mes = f.name[:4] + "-" + f.name[4:6]
        if f.suffix in (".json", ".zip") and mes not in meses_csv:
            elegidos.setdefault(mes, f)
    return [elegidos[m] for m in sorted(elegidos)]


conteo = Counter()  # (hora 'YYYY-MM-DDTHH', estación) -> salidas
total = Counter()
nombres = {}


def sumar(fichero, viajes):
    n = 0
    for hora, est, nombre in viajes:
        conteo[hora, est] += 1
        total[est] += 1
        if nombre:
            nombres.setdefault(est, nombre)
        n += 1
    print(f"{fichero.name}: {n} viajes")


for fichero in sorted(f for c in CARPETAS for f in c.glob("*.csv")):
    sumar(fichero, leer_csv(fichero))
meses_csv = {h[:7] for h, _ in conteo}
for fichero in ficheros_json(meses_csv):
    sumar(fichero, leer_json(fichero))

top = [est for est, _ in total.most_common(TOP_N)]
horas = sorted({h for h, _ in conteo})
inicio, fin = (datetime.strptime(h, "%Y-%m-%dT%H") for h in (horas[0], horas[-1]))

DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)
with open(SALIDA, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["fecha_hora", "id_estacion", "dia_semana", "mes", "festivo", "salidas"])
    t = inicio
    while t <= fin:  # todas las horas, también las que no tienen viajes (0 salidas)
        clave = t.strftime("%Y-%m-%dT%H")
        for e in top:  # 50 filas por hora, una por estación
            w.writerow([t.strftime("%Y-%m-%d %H:00"), e, t.isoweekday(), t.month, int(TIPO_DIA[t.date()] == "festivo"), conteo[clave, e]])
        t += timedelta(hours=1)

with open(SALIDA.with_name("estaciones_top50.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["columna", "id_estacion", "nombre", "salidas_totales"])
    w.writerows([f"est_{e}", e, nombres.get(e, ""), total[e]] for e in top)

print(f"{SALIDA.name}: {int((fin - inicio).total_seconds() // 3600) + 1} horas x {len(top)} estaciones")
