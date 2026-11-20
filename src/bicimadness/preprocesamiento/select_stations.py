"""Elige 60 estaciones variadas de BiciMAD para la práctica final (6 grupos x 10).

Criterio del profesor: un conjunto limitado pero diverso según tres factores, cada uno con dos extremos.

  Factor                 Grupo                Cómo se elige
  Demanda                demanda_alta         las 10 con más salidas
                         demanda_baja         las 10 con menos salidas (de las candidatas)
  Ubicación              centrica             la más cercana a cada uno de 10 lugares emblemáticos
                         afueras              las 10 más lejos de la Puerta del Sol, separadas >= 1 km entre sí
  Dificultad predictiva  facil / dificil      primera ronda: una RNA por estación (rna/entrenar.py, todas las variables)
                                              sobre TODAS las candidatas libres; las 10 con mayor y menor R²
                                              por estación, calculado solo con los bloques de test anteriores a
                                              CORTE_GRUPOS (comun/datos.py, 2022-08-01). Así, en errores_por_grupo.csv
                                              el periodo desde_2022-08-01 mide estos grupos con semanas que no se
                                              usaron para elegirlos.

Candidatas: estaciones con al menos 30 salidas en cada uno de los 26 meses (ene 2021 - feb 2023).
Así se quitan las que abrieron, cerraron o cambiaron de número a mitad del periodo, cuya serie
tendría meses enteros a 0. Una estación solo entra en un grupo: los grupos se rellenan en el orden
de la tabla y, si una ya está elegida, se pasa a la siguiente de la lista.

Uso:  python src/bicimadness/preprocesamiento/count_stations.py    (una vez: cuenta todas las estaciones -> data/cache/)
      python src/bicimadness/preprocesamiento/select_stations.py   (la primera ronda tarda; si ya hay resultados, los reutiliza)
      python src/bicimadness/preprocesamiento/select_stations.py --repetir     (vuelve a entrenar la primera ronda)
      python src/bicimadness/preprocesamiento/select_stations.py --muestra 60  (primera ronda con 60 al azar, más rápida)

Salidas:
  data/processed/estaciones_seleccionadas.csv   las 60 estaciones con su grupo, el motivo y sus datos
  data/processed/salidas_por_hora_sel60.csv     salidas por hora de las 60, mismo formato que salidas_por_hora_top50.csv:
                                    python src/bicimadness/rna/entrenar.py sel60
  data/cache/salidas_por_hora_primera_ronda.csv datos de la primera ronda
  results/primera_ronda/                        resultados de rna/entrenar.py en la primera ronda
"""
import argparse
import csv
import gzip
import math
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from preprocess import PROYECTO, TIPO_DIA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from bicimadness.comun.datos import CORTE_GRUPOS  # noqa: E402

CACHE = PROYECTO / "data" / "cache"
POR_GRUPO = 10
MIN_SALIDAS_MES = 30
SEMILLA = 42
SEPARACION_AFUERAS_KM = 1.0
RADIO_EMBLEMATICO_KM = 0.5
SOL = (40.41694, -3.70361)  # km 0, Puerta del Sol
# lugares emblemáticos (lat, lon); los dos últimos son de reserva por si alguno no tiene estación libre cerca
EMBLEMATICOS = {
    "Puerta del Sol": (40.41694, -3.70361),
    "Plaza Mayor": (40.41552, -3.70742),
    "Plaza de España": (40.42330, -3.71220),
    "Callao / Gran Vía": (40.41994, -3.70598),
    "Palacio Real / Ópera": (40.41799, -3.71432),
    "Cibeles": (40.41930, -3.69300),
    "Puerta de Alcalá / Retiro": (40.41999, -3.68877),
    "Museo del Prado": (40.41378, -3.69214),
    "Atocha": (40.40650, -3.68990),
    "Plaza de Colón": (40.42530, -3.69040),
    "Templo de Debod": (40.42400, -3.71780),
    "Plaza de Santa Ana": (40.41460, -3.70060),
}
MUESTRA = PROYECTO / "data" / "cache" / "salidas_por_hora_primera_ronda.csv"
PRIMERA_RONDA = PROYECTO / "results" / "primera_ronda"
RNA = Path(__file__).resolve().parents[1] / "rna" / "entrenar.py"
SALIDA_EST = PROYECTO / "data" / "processed" / "estaciones_seleccionadas.csv"
SALIDA_HORAS = PROYECTO / "data" / "processed" / "salidas_por_hora_sel60.csv"


def km(a, b):
    """Distancia en km entre dos puntos (lat, lon) (fórmula del haversine)."""
    (la1, lo1), (la2, lo2) = (map(math.radians, p) for p in (a, b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def cargar_estaciones():
    with open(CACHE / "estaciones_todas.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    meses = [k for k in filas[0] if k[:2] == "20"]
    est = {}
    for r in filas:
        if r["lat"] and all(int(r[m]) >= MIN_SALIDAS_MES for m in meses):
            pos = (float(r["lat"]), float(r["lon"]))
            est[r["id_estacion"]] = {"id_estacion": r["id_estacion"], "nombre": r["nombre"], "lat": pos[0], "lon": pos[1],
                                     "salidas_totales": int(r["salidas_totales"]), "dist_sol_km": km(pos, SOL)}
    print(f"{len(filas)} estaciones en los datos, {len(est)} candidatas (>= {MIN_SALIDAS_MES} salidas en los {len(meses)} meses)")
    return est


def cargar_conteo():
    conteo = defaultdict(dict)  # estación -> {hora 'YYYY-MM-DDTHH': salidas}
    with gzip.open(CACHE / "salidas_todas.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            conteo[r["id_estacion"]][r["fecha_hora"]] = int(r["salidas"])
    horas = sorted({h for c in conteo.values() for h in c})
    return conteo, *(datetime.strptime(h, "%Y-%m-%dT%H") for h in (horas[0], horas[-1]))


def escribir_salidas(fichero, ids, conteo, inicio, fin):
    """Mismo formato que salidas_por_hora_top50.csv: todas las horas, también las de 0 salidas."""
    with open(fichero, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fecha_hora", "id_estacion", "dia_semana", "mes", "festivo", "salidas"])
        t = inicio
        while t <= fin:
            clave = t.strftime("%Y-%m-%dT%H")
            for e in ids:
                w.writerow([t.strftime("%Y-%m-%d %H:00"), e, t.isoweekday(), t.month,
                            int(TIPO_DIA[t.date()] == "festivo"), conteo[e].get(clave, 0)])
            t += timedelta(hours=1)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--repetir", action="store_true", help="vuelve a entrenar la primera ronda aunque ya haya resultados")
    args.add_argument("--jobs", type=int, help="bloques de la primera ronda en paralelo (por defecto todos los núcleos)")
    args.add_argument("--modo", choices=["estacion", "global"], default="estacion",
                      help="primera ronda con una red por estación (por defecto) o una red global; la global con "
                           "~200 estaciones no cabe en 8 GB de RAM")
    args.add_argument("--muestra", type=int, default=0,
                      help="primera ronda con N candidatas al azar en vez de todas (0 = todas, por defecto)")
    args = args.parse_args()

    est = cargar_estaciones()
    elegidas = {}  # id -> fila con grupo y motivo

    def elegir(e, grupo, motivo):
        elegidas[e] = {**est[e], "grupo": grupo, "motivo": motivo}

    libres = lambda: [e for e in est if e not in elegidas]

    # 1-2. Demanda
    for e in sorted(libres(), key=lambda e: -est[e]["salidas_totales"])[:POR_GRUPO]:
        elegir(e, "demanda_alta", f"{est[e]['salidas_totales']} salidas")
    for e in sorted(libres(), key=lambda e: est[e]["salidas_totales"])[:POR_GRUPO]:
        elegir(e, "demanda_baja", f"{est[e]['salidas_totales']} salidas")

    # 3. Céntricas: la estación libre más cercana a cada lugar emblemático
    n = 0
    for lugar, pos in EMBLEMATICOS.items():
        if n == POR_GRUPO:
            break
        cerca = sorted(libres(), key=lambda e: km(pos, (est[e]["lat"], est[e]["lon"])))
        d = km(pos, (est[cerca[0]]["lat"], est[cerca[0]]["lon"]))
        if d <= RADIO_EMBLEMATICO_KM:
            elegir(cerca[0], "centrica", f"{lugar} ({d * 1000:.0f} m)")
            n += 1
        else:
            print(f"  {lugar}: ninguna estación libre a menos de {RADIO_EMBLEMATICO_KM * 1000:.0f} m, se usa la reserva")

    # 4. Afueras: las más lejos de Sol, separadas entre sí para no coger 10 del mismo barrio
    afueras = []
    for e in sorted(libres(), key=lambda e: -est[e]["dist_sol_km"]):
        pos = (est[e]["lat"], est[e]["lon"])
        if all(km(pos, (est[o]["lat"], est[o]["lon"])) >= SEPARACION_AFUERAS_KM for o in afueras):
            afueras.append(e)
            elegir(e, "afueras", f"{est[e]['dist_sol_km']:.1f} km de Sol")
        if len(afueras) == POR_GRUPO:
            break

    # 5-6. Dificultad predictiva: primera ronda con la RNA sobre las candidatas libres
    muestra = sorted(libres(), key=int)
    if args.muestra:
        muestra = sorted(random.Random(SEMILLA).sample(muestra, args.muestra), key=int)
    conteo, inicio, fin = cargar_conteo()
    errores = PRIMERA_RONDA / "error_por_estacion.csv"
    hecha = errores.exists() and {r["id_estacion"] for r in csv.DictReader(open(errores, encoding="utf-8"))} == set(muestra)
    if args.repetir or not hecha:
        escribir_salidas(MUESTRA, muestra, conteo, inicio, fin)
        print(f"Primera ronda: RNA ({args.modo}) sobre {len(muestra)} estaciones, bloques antes de {CORTE_GRUPOS:%Y-%m-%d}...",
              flush=True)
        subprocess.run([sys.executable, str(RNA), str(MUESTRA), "--modo", args.modo, "--variables", "todas",
                        "--hasta", f"{CORTE_GRUPOS:%Y-%m-%d}", "--carpeta", str(PRIMERA_RONDA),
                        "--nombre", "primera_ronda", *(["--jobs", str(args.jobs)] if args.jobs else [])], check=True)
    else:
        print(f"Primera ronda: se reutiliza {errores.relative_to(PROYECTO)} (--repetir para entrenar de nuevo)")
    with open(errores, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    r2 = {r["id_estacion"]: float(r["r2"]) for r in filas}
    for e in est:
        est[e]["r2_primera_ronda"] = r2.get(e, "")
    orden = sorted(muestra, key=lambda e: -r2[e])
    for e in orden[:POR_GRUPO]:
        elegir(e, "facil", f"R² = {r2[e]:.3f}")
    for e in orden[::-1][:POR_GRUPO]:
        elegir(e, "dificil", f"R² = {r2[e]:.3f}")

    # Resultado
    horas_totales = int((fin - inicio).total_seconds() // 3600) + 1
    campos = ["id_estacion", "nombre", "grupo", "motivo", "salidas_totales", "media_salidas_hora",
              "dist_sol_km", "lat", "lon", "r2_primera_ronda"]
    with open(SALIDA_EST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for e, r in elegidas.items():
            w.writerow({**r, "media_salidas_hora": round(r["salidas_totales"] / horas_totales, 3),
                        "dist_sol_km": round(r["dist_sol_km"], 2), "r2_primera_ronda": est[e].get("r2_primera_ronda", "")})
    escribir_salidas(SALIDA_HORAS, list(elegidas), conteo, inicio, fin)

    print(f"\n{len(elegidas)} estaciones elegidas:")
    for grupo in dict.fromkeys(r["grupo"] for r in elegidas.values()):
        print(f"  {grupo:13}", ", ".join(f"{r['nombre']} ({e})" for e, r in elegidas.items() if r["grupo"] == grupo))
    print(f"\nGuardado: {SALIDA_EST.name} y {SALIDA_HORAS.name}")


if __name__ == "__main__":
    main()
