"""Datos comunes a todos los algoritmos: CSV de salidas, variables, ventana y bloques de la sliding window.

Cualquier CSV con el formato de salidas_por_hora_top50.csv sirve (fecha_hora, id_estacion, ..., salidas):
todas las estaciones con las mismas horas, una fila por (hora, estación).
"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PROYECTO = Path(__file__).resolve().parents[3]  # raíz del repositorio
PROCESADOS = PROYECTO / "data" / "processed"
METEO = PROYECTO / "data" / "raw" / "meteo_horario.csv"
RESULTADOS = PROYECTO / "results"

# alias cortos para --datos
ALIAS = {
    "top50": PROCESADOS / "salidas_por_hora_top50.csv",
    "sel60": PROCESADOS / "salidas_por_hora_sel60.csv",
}
# CSV de estaciones con columna "grupo" (para errores_por_grupo.csv), por nombre del CSV de salidas
GRUPOS = {"salidas_por_hora_sel60.csv": PROCESADOS / "estaciones_seleccionadas.csv"}

VARIABLES = {
    "hora": "hora del día, nominal (24 columnas 0/1)",
    "dia_semana": "día de la semana, nominal (7 columnas 0/1)",
    "mes": "mes, nominal (12 columnas 0/1)",
    "festivo": "1 si es festivo",
    "temperatura": "ºC en Madrid-Retiro",
    "precipitacion": "mm de lluvia en una hora en Madrid-Retiro",
}
METEOROLOGICAS = {"temperatura", "precipitacion"}

# Sliding window: entrena con las `semanas` anteriores al bloque y prueba con las TEST_H horas siguientes.
# El primer bloque empieza siempre en la misma hora (deja sitio a la ventana y al entrenamiento más largos
# permitidos), así TODAS las ejecuciones sobre un mismo CSV usan los mismos bloques y se pueden comparar.
TEST_H = 2 * 7 * 24          # 2 semanas por bloque
MAX_VENTANA = 7 * 24         # hasta 1 semana de historial
MAX_SEMANAS = 52             # hasta 1 año de entrenamiento
VAL_H = 7 * 24               # validación interna (parada temprana, elegir alpha): última semana de cada entrenamiento
INICIO_TEST = MAX_VENTANA + MAX_SEMANAS * 7 * 24
# Fácil/difícil se clasifica con los bloques que empiezan antes de esta fecha; errores_por_grupo.csv
# da también el error de los bloques posteriores, que no se usaron para clasificar.
CORTE_GRUPOS = datetime(2022, 8, 1)


def ruta_datos(valor):
    """'sel60', 'top50' o una ruta (relativa a la raíz del repositorio o absoluta)."""
    ruta = ALIAS.get(valor, PROYECTO / valor)
    if not ruta.exists():
        raise SystemExit(f"no existe {ruta}. Usa {', '.join(ALIAS)} o la ruta de un CSV")
    return ruta


def cargar(ruta):
    """-> ({estación: salidas por hora}, [fechas], festivo por hora)."""
    series, fechas, festivo = {}, [], []
    with open(ruta, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            est = fila["id_estacion"]
            if not series:
                primera = est
            series.setdefault(est, []).append(int(fila["salidas"]))
            if est == primera:  # las fechas son las mismas para todas las estaciones
                fechas.append(datetime.strptime(fila["fecha_hora"], "%Y-%m-%d %H:%M"))
                festivo.append(fila["festivo"] == "1")
    return {e: np.array(s, dtype=np.float32) for e, s in series.items()}, fechas, np.array(festivo, dtype=np.float32)


def variables(nombres, fechas, festivo, meteo="actual"):
    """Matriz (horas x columnas) con las variables de la hora a predecir.

    meteo="actual": temperatura y lluvia de la hora que se predice (equivale a un pronóstico perfecto).
    meteo="anterior": las de la hora anterior, lo que ya se ha observado al hacer la predicción.
    """
    columnas = {
        "hora": lambda: np.eye(24)[[f.hour for f in fechas]],  # nominales: una columna 0/1 por valor
        "dia_semana": lambda: np.eye(7)[[f.weekday() for f in fechas]],
        "mes": lambda: np.eye(12)[[f.month - 1 for f in fechas]],
        "festivo": lambda: festivo[:, None],
    }
    if METEOROLOGICAS & set(nombres):
        with open(METEO, encoding="utf-8") as f:
            obs = {r["fecha_hora"]: (float(r["temperatura"]), float(r["precipitacion"])) for r in csv.DictReader(f)}
        tp = np.array([obs[f.strftime("%Y-%m-%d %H:00")] for f in fechas])
        if meteo == "anterior":
            tp = np.vstack([tp[:1], tp[:-1]])  # fila t = meteorología de t-1
        columnas |= {"temperatura": lambda: tp[:, :1], "precipitacion": lambda: tp[:, 1:]}
    if not nombres:
        return np.zeros((len(fechas), 0), dtype=np.float32)
    return np.hstack([columnas[n]() for n in nombres]).astype(np.float32)


def ventana(y, cal, n, objetivos):
    """Fila t = [y[t-n] ... y[t-1], variables de t]  ->  y[t]."""
    sw = sliding_window_view(y, n)  # sw[i] = y[i : i+n]
    return np.hstack([sw[objetivos - n], cal[objetivos]]), y[objetivos]


def bloques(n_horas, hasta=None, fechas=None):
    """[(k, inicio)] de cada bloque de test de TEST_H horas. `hasta`: solo los que empiezan antes de esa fecha."""
    res = list(enumerate(range(INICIO_TEST, n_horas - TEST_H + 1, TEST_H)))
    return [(k, s) for k, s in res if hasta is None or fechas[s] < hasta]
