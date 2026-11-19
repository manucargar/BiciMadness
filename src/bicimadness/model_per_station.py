"""RNA para predecir las salidas de la hora siguiente en cada estación (BiciMAD).

Uso:  python src/bicimadness/model_per_station.py                                  -> ventana 24h, entrenando con 1 año
      python src/bicimadness/model_per_station.py --ventanas 24 48 --semanas 26 52  -> compara todas las combinaciones
      python src/bicimadness/model_per_station.py --atributos hora,dia_semana,mes,festivo hora,dia_semana,mes,festivo,temperatura,precipitacion
                                                                 -> compara dos conjuntos de atributos
      python src/bicimadness/model_per_station.py --modelo estacion global          -> una red por estación vs una red para todas

Modelos: "estacion" entrena una red por estación; "global" una sola red con los datos de las 50
estaciones, donde la estación entra como atributo nominal (one-hot).

Por cada estación (o la red global) y cada ventana:
  - objetivo: salidas de la hora siguiente
  - entradas: salidas de las N horas anteriores (windowing) + los atributos elegidos de la hora a predecir
    (ver ATRIBUTOS; hora, dia_semana y mes entran como nominales one-hot)
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

PROYECTO = Path(__file__).resolve().parents[2]  # raíz del repositorio
DATOS = PROYECTO / "data" / "processed" / "salidas_por_hora_top50.csv"
METEO = PROYECTO / "data" / "raw" / "meteo_horario.csv"
RESULTADOS = PROYECTO / "results" / "per_station"
# atributos que se pueden meter en la red (de la hora a predecir)
ATRIBUTOS = {
    "hora": "hora del día, nominal (24 columnas 0/1)",
    "dia_semana": "día de la semana, nominal (7 columnas 0/1)",
    "mes": "mes, nominal (12 columnas 0/1)",
    "festivo": "1 si es festivo",
    "temperatura": "ºC en Madrid-Retiro",
    "precipitacion": "mm de lluvia en esa hora en Madrid-Retiro",
}
_args = argparse.ArgumentParser()
_args.add_argument("--ventanas", type=int, nargs="+", default=[24], help="horas anteriores que entran a la red")
_args.add_argument("--semanas", type=int, nargs="+", default=[52], help="semanas de entrenamiento (52 = 1 año)")
_args.add_argument("--atributos", nargs="+", default=["hora,dia_semana,mes,festivo"],
                   help="conjuntos de atributos separados por comas; varios conjuntos se comparan entre sí. "
                        "Disponibles: " + ", ".join(ATRIBUTOS))
_args.add_argument("--modelo", nargs="+", default=["estacion"], choices=["estacion", "global"],
                   help="estacion = una red por estación; global = una sola red para las 50 estaciones "
                        "(la estación entra como atributo nominal). Con los dos se comparan.")
_args.add_argument("--datos", default=str(DATOS.relative_to(PROYECTO)),
                   help="CSV de salidas por hora (por defecto las 50 estaciones con más salidas)")
_args.add_argument("--resultados", default=str(RESULTADOS.relative_to(PROYECTO)),
                   help="carpeta donde se guardan los resultados (rutas relativas: desde la raíz del repositorio)")
_args.add_argument("--ocultas-global", type=int, nargs="+", default=[64, 32],
                   help="neuronas de cada capa oculta de la red global (64 32 = dos capas de 64 y 32)")
_args.add_argument("--jobs", type=int, default=4,
                   help="folds de la red global en paralelo (cada uno ocupa memoria: bajadlo si hay poca RAM)")
_args = _args.parse_known_args()[0]
DATOS = PROYECTO / _args.datos  # una ruta absoluta se queda como está
RESULTADOS = PROYECTO / _args.resultados
MODELOS = _args.modelo
VENTANAS = _args.ventanas
ENTRENOS = _args.semanas
CONJUNTOS = [c.split(",") for c in _args.atributos]
for _a in {a for c in CONJUNTOS for a in c} - set(ATRIBUTOS):
    raise SystemExit(f"atributo desconocido '{_a}'. Disponibles: {', '.join(ATRIBUTOS)}")
TEST_H = 2 * 7 * 24     # 2 semanas de test; la ventana se desliza esto cada vez
OCULTAS = (32, 16)         # neuronas de las capas ocultas (red por estación)
OCULTAS_GLOBAL = tuple(_args.ocultas_global)  # la red global ve 50 veces más datos -> algo más grande


def cargar():
    series, fechas, festivo = defaultdict(list), [], []
    with open(DATOS, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            series[fila["id_estacion"]].append(int(fila["salidas"]))
            if fila["id_estacion"] == next(iter(series)):  # las fechas son las mismas para todas las estaciones
                fechas.append(datetime.strptime(fila["fecha_hora"], "%Y-%m-%d %H:%M"))
                festivo.append(fila["festivo"] == "1")
    return {e: np.array(s, dtype=float) for e, s in series.items()}, fechas, np.array(festivo, dtype=float)


def atributos(fechas, festivo):
    """{nombre: matriz (horas x columnas)} solo de los atributos que se usan en algún conjunto."""
    usados = {a for c in CONJUNTOS for a in c}
    h = np.array([f.hour for f in fechas])
    columnas = {
        "hora": lambda: np.eye(24)[h],  # nominales: una columna 0/1 por valor
        "dia_semana": lambda: np.eye(7)[[f.weekday() for f in fechas]],
        "mes": lambda: np.eye(12)[[f.month - 1 for f in fechas]],
        "festivo": lambda: festivo[:, None],
    }
    if usados & {"temperatura", "precipitacion"}:
        with open(METEO, encoding="utf-8") as f:
            meteo = {r["fecha_hora"]: (float(r["temperatura"]), float(r["precipitacion"])) for r in csv.DictReader(f)}
        tp = np.array([meteo[f.strftime("%Y-%m-%d %H:00")] for f in fechas])
        columnas |= {"temperatura": lambda: tp[:, :1], "precipitacion": lambda: tp[:, 1:]}
    return {a: columnas[a]() for a in usados}, h


def ventana(y, cal, n, objetivos):
    """Fila t = [y[t-n] ... y[t-1], calendario de t]  ->  y[t]."""
    X = np.array([np.concatenate([y[t - n:t], cal[t]]) for t in objetivos])
    return X, y[objetivos]


def metricas(real, pred):
    return (mean_absolute_error(real, pred), mean_squared_error(real, pred) ** 0.5, r2_score(real, pred))


def folds(n_horas):
    """(k, inicio) de cada bloque de test. Mismas semanas para todas las configs -> comparación justa en el T-test."""
    inicio = max(VENTANAS) + max(ENTRENOS) * 7 * 24
    return list(enumerate(range(inicio, n_horas - TEST_H + 1, TEST_H)))


def anotar(filas, errores_hora, config, est, k, real, pred, horas_te):
    filas.append([config, est, k, *metricas(real, pred)])
    for hora, e in zip(horas_te, np.abs(real - pred)):
        errores_hora[config, hora].append(e)


def evaluar_estacion(est, y, cals, horas):
    """Una red por estación, y la referencia sencilla."""
    filas, errores_hora = [], defaultdict(list)  # (config, hora) -> errores absolutos
    for k, s in folds(len(y)):
        te = np.arange(s, s + TEST_H)
        anotar(filas, errores_hora, "referencia_24h", est, k, y[te], y[te - 24], horas[te])  # lo que salió ayer
        if "estacion" not in MODELOS:
            continue
        for semanas in ENTRENOS:
            tr = np.arange(s - semanas * 7 * 24, s)  # las semanas justo antes del test
            for n in VENTANAS:
                for sufijo, cal in cals.items():
                    Xtr, ytr = ventana(y, cal, n, tr)
                    Xte, _ = ventana(y, cal, n, te)
                    rna = make_pipeline(StandardScaler(), MLPRegressor(
                        hidden_layer_sizes=OCULTAS, early_stopping=True, max_iter=500, random_state=0))
                    pred = np.clip(rna.fit(Xtr, ytr).predict(Xte), 0, None)  # no hay salidas negativas
                    anotar(filas, errores_hora, f"rna_{n}h_{semanas}sem{sufijo}", est, k, y[te], pred, horas[te])
    return filas, errores_hora


def evaluar_global(k, s, series, cals, horas):
    """Una sola red para todas las estaciones en el fold k; la estación entra como atributo nominal (one-hot)."""
    filas, errores_hora = [], defaultdict(list)
    te = np.arange(s, s + TEST_H)
    onehot = np.eye(len(series), dtype=np.float32)
    for semanas in ENTRENOS:
        tr = np.arange(s - semanas * 7 * 24, s)
        for n in VENTANAS:
            for sufijo, cal in cals.items():
                Xtr, ytr, Xte = [], [], {}
                for i, (est, y) in enumerate(series.items()):
                    X, yy = ventana(y, cal, n, tr)
                    Xtr.append(np.hstack([X, np.tile(onehot[i], (len(X), 1))]).astype(np.float32))
                    ytr.append(yy)
                    X, _ = ventana(y, cal, n, te)
                    Xte[est] = np.hstack([X, np.tile(onehot[i], (len(X), 1))]).astype(np.float32)
                rna = make_pipeline(StandardScaler(), MLPRegressor(
                    hidden_layer_sizes=OCULTAS_GLOBAL, early_stopping=True, max_iter=200, random_state=0))
                rna.fit(np.vstack(Xtr), np.concatenate(ytr))
                for est, y in series.items():
                    pred = np.clip(rna.predict(Xte[est]), 0, None)
                    anotar(filas, errores_hora, f"rna_global_{n}h_{semanas}sem{sufijo}", est, k, y[te], pred, horas[te])
    return filas, errores_hora


def main():
    series, fechas, festivo = cargar()
    columnas, horas = atributos(fechas, festivo)
    # un nombre por conjunto solo si se comparan varios (rna_24h_52sem_hora+mes+temperatura)
    cals = {("_" + "+".join(c) if len(CONJUNTOS) > 1 else ""): np.hstack([columnas[a] for a in c])
            for c in CONJUNTOS}
    print(f"{DATOS.name}: {len(series)} estaciones, {len(fechas)} horas, ventanas {VENTANAS}, semanas {ENTRENOS}")
    for c in CONJUNTOS:
        print("  atributos:", ", ".join(c))
    res = Parallel(n_jobs=-1)(delayed(evaluar_estacion)(e, y, cals, horas) for e, y in series.items())
    if "global" in MODELOS:
        # ponytail: --jobs (4 por defecto) porque cada fold global junta los datos de todas las estaciones (~0,2 GB con 50)
        res += Parallel(n_jobs=_args.jobs)(delayed(evaluar_global)(k, s, series, cals, horas) for k, s in folds(len(fechas)))

    RESULTADOS.mkdir(parents=True, exist_ok=True)
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
    ancho = max(map(len, configs)) + 2
    print(f"{'config':{ancho}}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    with open(RESULTADOS / "resumen.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "mae", "rmse", "r2", "n_estacion_fold"])
        for c in configs:
            mae, rmse, r2 = np.mean([tabla[c][k] for k in claves], axis=0)
            print(f"{c:{ancho}}{mae:8.3f}{rmse:8.3f}{r2:8.3f}")
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
