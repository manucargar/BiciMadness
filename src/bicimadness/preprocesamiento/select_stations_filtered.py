"""Crea una revisión de sel60 sin modificar la selección original.

Descarta candidatas con dos semanas completas consecutivas por debajo del
10 % de su mediana semanal. La semana de Filomena (4-10 enero 2021) y las
semanas parciales no intervienen en el filtro. Conserva las estaciones de la
selección original que pasan; rellena la plaza de afueras por distancia a Sol
y separación, y las tres plazas difíciles por el R² de la primera ronda.

Uso: python src/bicimadness/preprocesamiento/select_stations_filtered.py
"""
import csv
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

import select_stations as base


SALIDA_EST = base.PROYECTO / "data" / "processed" / "estaciones_seleccionadas_filtradas.csv"
SALIDA_HORAS = base.PROYECTO / "data" / "processed" / "salidas_por_hora_sel60_filtrada.csv"
FILOMENA = datetime(2021, 1, 4)
UMBRAL = 0.10
EXCLUIDAS_ESPERADAS = {"174", "218", "233", "269"}
GRUPOS = ("demanda_alta", "demanda_baja", "centrica", "afueras", "facil", "dificil")
CAMPOS = ("id_estacion", "nombre", "grupo", "motivo", "salidas_totales",
          "media_salidas_hora", "dist_sol_km", "lat", "lon", "r2_primera_ronda")


def semanas_completas(inicio, fin):
    """Lunes de las semanas enteras presentes en la serie, salvo Filomena."""
    lunes = datetime(inicio.year, inicio.month, inicio.day) - timedelta(days=inicio.weekday())
    if lunes < inicio:
        lunes += timedelta(days=7)
    while lunes + timedelta(days=7) <= fin + timedelta(hours=1):
        if lunes != FILOMENA:
            yield lunes
        lunes += timedelta(days=7)


def estaciones_activas(estaciones, conteo, inicio, fin):
    lunes = list(semanas_completas(inicio, fin))
    if not lunes:
        raise ValueError("no hay semanas completas para calcular el filtro")
    activos = set()
    incidencias = {}
    for id_estacion in estaciones:
        totales = defaultdict(int)
        for hora, salidas in conteo[id_estacion].items():
            fecha = datetime.fromisoformat(hora)
            semana = datetime(fecha.year, fecha.month, fecha.day) - timedelta(days=fecha.weekday())
            totales[semana] += salidas
        valores = [totales[semana] for semana in lunes]  # las semanas sin viajes valen 0
        mediana = statistics.median(valores)
        bajas = [n < UMBRAL * mediana for n in valores]
        parejas = [(lunes[i], lunes[i + 1]) for i in range(len(lunes) - 1)
                   if lunes[i + 1] - lunes[i] == timedelta(days=7) and bajas[i] and bajas[i + 1]]
        if parejas:
            incidencias[id_estacion] = (mediana, parejas)
        else:
            activos.add(id_estacion)
    return activos, incidencias, len(lunes)


def primera_ronda():
    ruta = base.PRIMERA_RONDA / "error_por_estacion.csv"
    with open(ruta, newline="", encoding="utf-8") as f:
        return {r["id_estacion"]: float(r["r2"]) for r in csv.DictReader(f)}


def main():
    estaciones = base.cargar_estaciones()  # requisito mensual original
    conteo, inicio, fin = base.cargar_conteo()
    activos, incidencias, n_semanas = estaciones_activas(estaciones, conteo, inicio, fin)
    with open(base.SALIDA_EST, newline="", encoding="utf-8") as f:
        originales = list(csv.DictReader(f))
    if len(originales) != 60 or len({r["id_estacion"] for r in originales}) != 60:
        raise ValueError("la selección original debe tener 60 estaciones distintas")

    excluidas = {r["id_estacion"] for r in originales} - activos
    if excluidas != EXCLUIDAS_ESPERADAS:
        raise ValueError(f"el filtro excluye {sorted(excluidas)}, se esperaban {sorted(EXCLUIDAS_ESPERADAS)}; "
                         "revisar datos antes de cambiar la selección")
    filas = [r.copy() if r["id_estacion"] in activos else None for r in originales]
    ocupadas = {r["id_estacion"] for r in filas if r is not None}
    r2 = primera_ronda()
    horas_totales = int((fin - inicio).total_seconds() // 3600) + 1

    def nueva_fila(id_estacion, grupo, motivo):
        e = estaciones[id_estacion]
        return {"id_estacion": id_estacion, "nombre": e["nombre"], "grupo": grupo,
                "motivo": motivo, "salidas_totales": e["salidas_totales"],
                "media_salidas_hora": round(e["salidas_totales"] / horas_totales, 3),
                "dist_sol_km": round(e["dist_sol_km"], 2), "lat": e["lat"], "lon": e["lon"],
                "r2_primera_ronda": r2.get(id_estacion, "")}

    # Se mantienen las nueve afueras originales válidas; se completa con la
    # candidata libre más lejana que respeta 1 km con todas ellas.
    afueras = [r["id_estacion"] for r in filas if r and r["grupo"] == "afueras"]
    candidatas = sorted(activos - ocupadas, key=lambda e: (-estaciones[e]["dist_sol_km"], int(e)))
    sustituta_afueras = next((e for e in candidatas if all(
        base.km((estaciones[e]["lat"], estaciones[e]["lon"]),
                (estaciones[o]["lat"], estaciones[o]["lon"])) >= base.SEPARACION_AFUERAS_KM
        for o in afueras)), None)
    if sustituta_afueras is None:
        raise ValueError("no hay sustituta de afueras con la separación requerida")
    ocupadas.add(sustituta_afueras)
    i_afueras = next(i for i, r in enumerate(originales) if r["id_estacion"] == "269")
    filas[i_afueras] = nueva_fila(sustituta_afueras, "afueras",
                                 f"{estaciones[sustituta_afueras]['dist_sol_km']:.1f} km de Sol")

    # Las plazas difíciles se rellenan con las puntuaciones más bajas de la
    # misma primera ronda usada para construir la selección original.
    posibles = sorted((e for e in activos - ocupadas if e in r2), key=lambda e: (r2[e], int(e)))
    indices_dificiles = [i for i, r in enumerate(originales)
                         if r["grupo"] == "dificil" and filas[i] is None]
    if len(posibles) < len(indices_dificiles):
        raise ValueError("faltan candidatas puntuadas en la primera ronda")
    sustitutas_dificiles = posibles[:len(indices_dificiles)]
    for i, e in zip(indices_dificiles, sustitutas_dificiles):
        filas[i] = nueva_fila(e, "dificil", f"R² = {r2[e]:.3f}")

    ids = [r["id_estacion"] for r in filas]
    if (len(ids) != 60 or len(set(ids)) != 60 or any(
            sum(r["grupo"] == grupo for r in filas) != 10 for grupo in GRUPOS)):
        raise ValueError("la nueva selección no tiene seis grupos de diez estaciones distintas")
    cambios = {r["id_estacion"] for r in filas} - {r["id_estacion"] for r in originales}
    if cambios != {sustituta_afueras, *sustitutas_dificiles} or len(cambios) != 4:
        raise ValueError("la selección debe cambiar exactamente cuatro estaciones")

    # Se escriben únicamente ficheros con nombres nuevos. La selección y el
    # CSV horario originales permanecen disponibles para reproducir resultados.
    with open(SALIDA_EST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    base.escribir_salidas(SALIDA_HORAS, ids, conteo, inicio, fin)

    print(f"Filtro: {n_semanas} semanas completas, umbral {UMBRAL:.0%} de la mediana; "
          f"{len(activos)}/{len(estaciones)} candidatas válidas")
    for e in sorted(excluidas, key=int):
        mediana, parejas = incidencias[e]
        print(f"  Sale {estaciones[e]['nombre']} ({e}): mediana {mediana:g}, "
              f"primeras semanas bajas {parejas[0][0]:%Y-%m-%d} y {parejas[0][1]:%Y-%m-%d}")
    for e in [sustituta_afueras, *sustitutas_dificiles]:
        print(f"  Entra {estaciones[e]['nombre']} ({e})")
    print(f"Guardado: {SALIDA_EST.relative_to(base.PROYECTO)}")
    print(f"Guardado: {SALIDA_HORAS.relative_to(base.PROYECTO)}")


if __name__ == "__main__":
    main()
