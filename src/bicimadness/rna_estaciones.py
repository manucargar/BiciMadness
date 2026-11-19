"""RNA por estación para predecir las salidas de la hora siguiente (BiciMAD).

Uso:  python src/bicimadness/model_per_station.py
      python src/bicimadness/model_per_station.py --ventanas 24 48 --semanas 26 52

Por cada estación y cada ventana:
  - objetivo: salidas de la hora siguiente
  - entradas: salidas de las N horas anteriores (windowing) + hora, día de la semana y mes
    de la hora a predecir como variables nominales (one-hot) + festivo 0/1
  - validación sliding window: entrena con las ENTRENOS semanas anteriores, prueba con las TEST_H
    horas siguientes y desliza (todas las configs se prueban sobre las mismas semanas)
  - métricas MAE, RMSE y R², comparadas con una referencia sencilla (misma hora del día anterior)
Al final: T-test pareado entre configuraciones y análisis del error por hora y por estación.
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data" / "processed" / "salidas_por_hora_top50.csv"
RESULTADOS = RAIZ / "results" / "per_station"
_args = argparse.ArgumentParser()
_args.add_argument("--ventanas", type=int, nargs="+", default=[24], help="horas anteriores que entran a la red")
_args.add_argument("--semanas", type=int, nargs="+", default=[52], help="semanas de entrenamiento (52 = 1 año)")
_args = _args.parse_known_args()[0]
VENTANAS = _args.ventanas
ENTRENOS = _args.semanas
TEST_H = 2 * 7 * 24     # 2 semanas de test; la ventana se desliza esto cada vez
OCULTAS = (32, 16)      # neuronas de las capas ocultas


def cargar():
    series, fechas, festivo = defaultdict(list), [], []
    with open(DATOS, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            series[fila["id_estacion"]].append(int(fila["salidas"]))
            if fila["id_estacion"] == next(iter(series)):  # las fechas son las mismas para todas las estaciones
                fechas.append(datetime.strptime(fila["fecha_hora"], "%Y-%m-%d %H:%M"))
                festivo.append(fila["festivo"] == "1")
    return {e: np.array(s, dtype=float) for e, s in series.items()}, fechas, np.array(festivo, dtype=float)


def calendario(fechas):
    """Hora, día de la semana y mes como variables nominales (one-hot: una columna 0/1 por valor)."""
    h = np.array([f.hour for f in fechas])
    d = np.array([f.weekday() for f in fechas])
    m = np.array([f.month - 1 for f in fechas])
    return np.hstack([np.eye(24)[h], np.eye(7)[d], np.eye(12)[m]]), h


def ventana(y, cal, n, objetivos):
    """Fila t = [y[t-n] ... y[t-1], calendario de t]  ->  y[t]."""
    X = np.array([np.concatenate([y[t - n:t], cal[t]]) for t in objetivos])
    return X, y[objetivos]


def metricas(real, pred):
    return (mean_absolute_error(real, pred), mean_squared_error(real, pred) ** 0.5, r2_score(real, pred))


def evaluar_estacion(est, y, cals, horas):
    # mismas semanas de test para todas las configs -> comparación justa en el T-test
    inicio = max(VENTANAS) + max(ENTRENOS) * 7 * 24
    folds = range(inicio, len(y) - TEST_H + 1, TEST_H)
    filas, errores_hora = [], defaultdict(list)  # (config, hora) -> errores absolutos
    for k, s in enumerate(folds):
        te = np.arange(s, s + TEST_H)
        preds = {"referencia_24h": y[te - 24]}  # método sencillo: lo que salió ayer a la misma hora
        for semanas in ENTRENOS:
            tr = np.arange(s - semanas * 7 * 24, s)  # las semanas justo antes del test
            for n in VENTANAS:
                for sufijo, cal in cals.items():
                    Xtr, ytr = ventana(y, cal, n, tr)
                    Xte, _ = ventana(y, cal, n, te)
                    rna = make_pipeline(StandardScaler(), MLPRegressor(
                        hidden_layer_sizes=OCULTAS, early_stopping=True, max_iter=500, random_state=0))
                    # no hay salidas negativas
                    preds[f"rna_{n}h_{semanas}sem{sufijo}"] = np.clip(rna.fit(Xtr, ytr).predict(Xte), 0, None)
        for config, p in preds.items():
            filas.append([config, est, k, *metricas(y[te], p)])
            for hora, e in zip(horas[te], np.abs(y[te] - p)):
                errores_hora[config, hora].append(e)
    return filas, errores_hora


def main():
    series, fechas, festivo = cargar()
    cal, horas = calendario(fechas)
    # festivo 0/1 (sábado y domingo ya los da dia_semana)
    cals = {"": np.column_stack([cal, festivo])}
    print(f"{len(series)} estaciones, {len(fechas)} horas, ventanas {VENTANAS}, {int(festivo.sum() / 24)} días festivos")
    res = Parallel(n_jobs=-1)(delayed(evaluar_estacion)(e, y, cals, horas) for e, y in series.items())

    RESULTADOS.mkdir(exist_ok=True)
    filas = [f for r, _ in res for f in r]
    with open(RESULTADOS / "resultados_folds.csv", "w", newline="") as f:
        csv.writer(f).writerows([["config", "id_estacion", "fold", "mae", "rmse", "r2"], *filas])

    # tabla[config][(estacion, fold)] = (mae, rmse, r2)
    tabla = defaultdict(dict)
    for config, est, k, *m in filas:
        tabla[config][est, k] = m
    configs = sorted(tabla)
    claves = sorted(tabla[configs[0]])

    print(f"\nMedia sobre {len(claves)} (estación, fold):")
    print(f"{'config':20}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    with open(RESULTADOS / "resumen.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "mae", "rmse", "r2", "n_estacion_fold"])
        for c in configs:
            mae, rmse, r2 = np.mean([tabla[c][k] for k in claves], axis=0)
            print(f"{c:20}{mae:8.3f}{rmse:8.3f}{r2:8.3f}")
            w.writerow([c, round(mae, 4), round(rmse, 4), round(r2, 4), len(claves)])

    print("\nT-test pareado sobre el MAE (p < 0.05 -> diferencia significativa):")
    with open(RESULTADOS / "ttest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_a", "config_b", "mae_a", "mae_b", "t", "p"])
        for a, b in combinations(configs, 2):
            ma, mb = (np.array([tabla[c][k][0] for k in claves]) for c in (a, b))
            t, p = ttest_rel(ma, mb)
            print(f"  {a} vs {b}: MAE {ma.mean():.3f} vs {mb.mean():.3f}  t={t:.2f}  p={p:.4f}")
            w.writerow([a, b, round(ma.mean(), 4), round(mb.mean(), 4), round(t, 3), p])

    errores_hora = defaultdict(list)
    for _, eh in res:
        for clave, e in eh.items():
            errores_hora[clave].extend(e)
    with open(RESULTADOS / "error_por_hora.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hora", *(f"{m}_{c}" for c in configs for m in ("mae", "rmse"))])
        for h in range(24):
            e = {c: np.array(errores_hora[c, h]) for c in configs}
            w.writerow([h, *(round(v, 4) for c in configs for v in (e[c].mean(), np.sqrt((e[c] ** 2).mean())))])

    with open(RESULTADOS / "error_por_estacion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_estacion", *(f"{m}_{c}" for c in configs for m in ("mae", "rmse", "r2"))])
        for est in series:
            medias = [np.mean([m for (e, _), m in tabla[c].items() if e == est], axis=0) for c in configs]
            w.writerow([est, *(round(v, 4) for m in medias for v in m)])
    print(f"\nGuardado en {RESULTADOS}/: resumen.csv, ttest.csv, resultados_folds.csv, "
          "error_por_hora.csv, error_por_estacion.csv")


if __name__ == "__main__":
    main()
