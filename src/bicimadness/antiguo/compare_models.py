"""[ANTIGUO] Comparación de modelos para predecir las salidas de la hora siguiente en cada estación (BiciMAD).

Todos los modelos se evalúan en las mismas condiciones, para que el T-test sea justo:
  - las mismas estaciones, la misma ventana (24 h anteriores), los mismos atributos
  - los mismos folds de la sliding window de model_per_station.py (entrena con 52 semanas, prueba con
    las 2 siguientes), uno de cada FOLDS_CADA para que la LSTM tarde minutos y no horas

Modelos (--modelos para elegir algunos):
  ref_ayer            misma hora del día anterior (la referencia de model_per_station.py)
  ref_semana          misma hora de la semana anterior
  media_historica     media de la estación a esa hora y día de la semana en las semanas de entrenamiento
  regresion_lineal    regresión lineal (Ridge, alfa=1, por la colinealidad de las columnas nominales)
  gradient_boosting   árboles de decisión potenciados (HistGradientBoostingRegressor)
  mlp                 la red global de model_per_station.py (perceptrón multicapa 64-32)
  lstm                red recurrente LSTM global: lee las 24 h anteriores como secuencia y une a su salida
                      los atributos de la hora a predecir y la estación (PyTorch)

Los modelos entrenados (lineal, gradient boosting, mlp, lstm) son globales: una sola red para todas las
estaciones, con la estación como atributo nominal (one-hot), y reciben las mismas entradas.

Se mantiene hasta que cada algoritmo tenga su carpeta propia (como rna/) y se compare con
comparar_algoritmos.py. Sus resultados no usan el formato nuevo de results/<algoritmo>/.

Uso:  python src/bicimadness/antiguo/compare_models.py                         -> las 60 estaciones elegidas, con meteorología
      python src/bicimadness/antiguo/compare_models.py --modelos mlp lstm      -> solo algunos modelos
      python src/bicimadness/antiguo/compare_models.py --folds-cada 1          -> todos los folds (la LSTM tarda mucho más)

Cada fold se guarda en <resultados>/folds/ al terminar: si se corta, al volver a ejecutarlo continúa
donde se quedó (borrad esa carpeta para empezar de cero).

Salidas en <resultados>/ (mismo formato que model_per_station.py, así sirve analyze_groups.py):
  resumen.csv, ttest.csv, resultados_folds.csv, error_por_hora.csv, error_por_estacion.csv
  tiempos.csv     segundos medios de entrenamiento y de predicción por fold de cada modelo
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import ttest_rel
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROYECTO = Path(__file__).resolve().parents[3]  # raíz del repositorio
MODELOS = ["ref_ayer", "ref_semana", "media_historica", "regresion_lineal", "gradient_boosting", "mlp", "lstm"]

args = argparse.ArgumentParser()
args.add_argument("--datos", default="data/processed/salidas_por_hora_sel60.csv")
args.add_argument("--resultados", default="results/model_comparison")
args.add_argument("--atributos", default="hora,dia_semana,mes,festivo,temperatura,precipitacion",
                  help="atributos de la hora a predecir, separados por comas (los de model_per_station.py)")
args.add_argument("--modelos", nargs="+", default=MODELOS, choices=MODELOS)
args.add_argument("--folds-cada", type=int, default=3, help="usa uno de cada N folds (1 = todos)")
args.add_argument("--epocas-lstm", type=int, default=10, help="épocas máximas de la LSTM (para antes si no mejora)")
args = args.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from bicimadness.comun import datos as D  # noqa: E402  (carga, variables y bloques: los mismos que rna/)

N_VENTANA = 24
SEMANAS = 52
RESULTADOS = PROYECTO / args.resultados
SEMILLA = 0


def anotar(filas, errores_hora, config, est, k, real, pred, horas_te):
    filas.append([config, est, k, mean_absolute_error(real, pred), mean_squared_error(real, pred) ** 0.5,
                  r2_score(real, pred)])
    for hora, e in zip(horas_te, np.abs(real - pred)):
        errores_hora[config, hora].append(e)


def ventanas(y, cal, objetivos):
    """Igual que D.ventana pero vectorizado: fila t = [y[t-24] ... y[t-1], calendario de t] -> y[t]."""
    sw = sliding_window_view(y, N_VENTANA)  # sw[i] = y[i : i+24]
    return np.hstack([sw[objetivos - N_VENTANA], cal[objetivos]]).astype(np.float32), y[objetivos]


def matrices(series, cal, tr, te):
    """X e y globales (todas las estaciones apiladas) con la estación en one-hot al final."""
    onehot = np.eye(len(series), dtype=np.float32)
    Xtr, ytr, Xte = [], [], []
    for i, y in enumerate(series.values()):
        for idx, Xs, ys in ((tr, Xtr, ytr), (te, Xte, None)):
            X, yy = ventanas(y, cal, idx)
            Xs.append(np.hstack([X, np.tile(onehot[i], (len(X), 1))]))
            if ys is not None:
                ys.append(yy)
    return np.vstack(Xtr), np.concatenate(ytr).astype(np.float32), Xte


# ---------- LSTM ----------
def entrenar_lstm(Xtr, ytr, n_estatico):
    """Xtr = [24 salidas anteriores | atributos de la hora | one-hot estación]. Devuelve una función predict."""
    import torch
    from torch import nn

    torch.manual_seed(SEMILLA)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    sec, est = Xtr[:, :N_VENTANA], Xtr[:, N_VENTANA:]
    m_sec, s_sec = sec.mean(), sec.std() + 1e-6
    m_est, s_est = est.mean(axis=0), est.std(axis=0) + 1e-6

    def tensores(X):
        s = torch.from_numpy(((X[:, :N_VENTANA] - m_sec) / s_sec).astype(np.float32)).unsqueeze(-1)
        e = torch.from_numpy(((X[:, N_VENTANA:] - m_est) / s_est).astype(np.float32))
        return s, e

    class Red(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
            self.cabeza = nn.Sequential(nn.Linear(32 + n_estatico, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, s, e):
            _, (h, _) = self.lstm(s)  # último estado oculto: resumen de las 24 h
            return self.cabeza(torch.cat([h[-1], e], dim=1)).squeeze(1)

    red = Red()
    opt = torch.optim.Adam(red.parameters(), lr=1e-3)
    # validación para parar a tiempo: 10 % de las filas al azar, como early_stopping del MLP
    n = len(Xtr)
    rng = np.random.default_rng(SEMILLA)
    val = rng.random(n) < 0.1
    s_tr, e_tr = tensores(Xtr[~val]); y_tr = torch.from_numpy(ytr[~val])
    s_va, e_va = tensores(Xtr[val]); y_va = torch.from_numpy(ytr[val])
    mejor, mejor_estado, sin_mejora = np.inf, None, 0
    for _ in range(args.epocas_lstm):
        red.train()
        orden = torch.randperm(len(y_tr))
        for i in range(0, len(orden), 1024):
            b = orden[i:i + 1024]
            opt.zero_grad()
            perdida = nn.functional.mse_loss(red(s_tr[b], e_tr[b]), y_tr[b])
            perdida.backward()
            opt.step()
        red.eval()
        with torch.no_grad():
            pv = torch.cat([red(s_va[i:i + 8192], e_va[i:i + 8192]) for i in range(0, len(y_va), 8192)])
            mse = nn.functional.mse_loss(pv, y_va).item()
        if mse < mejor - 1e-4:
            mejor, sin_mejora = mse, 0
            mejor_estado = {k: v.clone() for k, v in red.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora == 2:  # para si en 2 épocas seguidas no mejora en validación
                break
    red.load_state_dict(mejor_estado)
    red.eval()

    def predict(X):
        s, e = tensores(X)
        with torch.no_grad():
            return torch.cat([red(s[i:i + 8192], e[i:i + 8192]) for i in range(0, len(X), 8192)]).numpy()

    return predict


def evaluar_fold(k, s, series, cal, fechas):
    """{modelo: {"pred": {estación: predicción}, "t_entreno": s, "t_pred": s}} de un fold."""
    tr = np.arange(s - SEMANAS * 7 * 24, s)
    te = np.arange(s, s + D.TEST_H)
    out = {}

    def apuntar(nombre, entrenar, predecir):
        t0 = time.time()
        modelo = entrenar()
        t1 = time.time()
        pred = predecir(modelo)
        out[nombre] = {"pred": {e: np.clip(p, 0, None).tolist() for e, p in pred.items()},
                       "t_entreno": t1 - t0, "t_pred": time.time() - t1}
        print(f"  fold {k:2d} {nombre:18s} entreno {t1 - t0:6.1f} s", flush=True)

    # referencias (no se entrenan)
    if "ref_ayer" in args.modelos:
        apuntar("ref_ayer", lambda: None, lambda _: {e: y[te - 24] for e, y in series.items()})
    if "ref_semana" in args.modelos:
        apuntar("ref_semana", lambda: None, lambda _: {e: y[te - 168] for e, y in series.items()})
    if "media_historica" in args.modelos:
        clave = np.array([f.hour * 7 + f.weekday() for f in fechas])  # 168 casillas: hora x día de la semana

        def medias():
            return {e: np.bincount(clave[tr], weights=y[tr], minlength=168) / np.bincount(clave[tr], minlength=168)
                    for e, y in series.items()}
        apuntar("media_historica", medias, lambda m: {e: m[e][clave[te]] for e in series})

    entrenados = [m for m in ("regresion_lineal", "gradient_boosting", "mlp", "lstm") if m in args.modelos]
    if not entrenados:
        return out
    Xtr, ytr, Xte = matrices(series, cal, tr, te)
    ests = list(series)
    por_est = lambda f: {e: f(X) for e, X in zip(ests, Xte)}  # noqa: E731
    if "regresion_lineal" in entrenados:
        apuntar("regresion_lineal", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr),
                lambda m: por_est(m.predict))
    if "gradient_boosting" in entrenados:
        apuntar("gradient_boosting", lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, early_stopping=True, random_state=SEMILLA).fit(Xtr, ytr),
            lambda m: por_est(m.predict))
    if "mlp" in entrenados:
        apuntar("mlp", lambda: make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=SEMILLA)).fit(Xtr, ytr),
            lambda m: por_est(m.predict))
    if "lstm" in entrenados:
        apuntar("lstm", lambda: entrenar_lstm(Xtr, ytr, Xtr.shape[1] - N_VENTANA), lambda f: por_est(f))
    return out


def main():
    ruta = D.ruta_datos(args.datos)
    series, fechas, festivo = D.cargar(ruta)
    horas = np.array([f.hour for f in fechas])
    atributos = args.atributos.split(",")
    cal = D.variables(atributos, fechas, festivo)
    todos = D.bloques(len(fechas))
    elegidos = todos[::args.folds_cada]
    print(f"{ruta.name}: {len(series)} estaciones, {len(fechas)} horas")
    print(f"atributos: {', '.join(atributos)}  |  folds: {[k for k, _ in elegidos]} de {len(todos)}")
    print(f"modelos: {', '.join(args.modelos)}", flush=True)

    carpeta_folds = RESULTADOS / "folds"
    carpeta_folds.mkdir(parents=True, exist_ok=True)
    filas, errores_hora, tiempos = [], defaultdict(list), defaultdict(list)
    for k, s in elegidos:
        fichero = carpeta_folds / f"fold_{k:02d}.json"
        hecho = json.loads(fichero.read_text()) if fichero.exists() else {}
        faltan = [m for m in args.modelos if m not in hecho]
        if faltan:
            print(f"Fold {k} ({fechas[s]:%Y-%m-%d} a {fechas[s + D.TEST_H - 1]:%Y-%m-%d})", flush=True)
            guardar = args.modelos
            args.modelos = faltan
            hecho |= evaluar_fold(k, s, series, cal, fechas)
            args.modelos = guardar
            fichero.write_text(json.dumps(hecho))
        te = np.arange(s, s + D.TEST_H)
        for m in args.modelos:
            for e, y in series.items():
                anotar(filas, errores_hora, m, e, k, y[te], np.array(hecho[m]["pred"][e]), horas[te])
            tiempos[m].append((hecho[m]["t_entreno"], hecho[m]["t_pred"]))

    escribir(filas, errores_hora, tiempos, list(series))


def escribir(filas, errores_hora, tiempos, estaciones):
    tabla = defaultdict(dict)  # tabla[modelo][(estación, fold)] = (mae, rmse, r2)
    for m, e, k, *met in filas:
        tabla[m][e, k] = met
    modelos = [m for m in MODELOS if m in tabla]
    claves = sorted(tabla[modelos[0]])

    with open(RESULTADOS / "resultados_folds.csv", "w", newline="") as f:
        csv.writer(f).writerows([["config", "id_estacion", "fold", "mae", "rmse", "r2"], *filas])

    print(f"\nMedia sobre {len(claves)} (estación, fold):")
    print(f"{'modelo':20}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'entreno (s)':>13}")
    with open(RESULTADOS / "resumen.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "mae", "rmse", "r2", "n_estacion_fold"])
        for m in modelos:
            mae, rmse, r2 = np.mean([tabla[m][c] for c in claves], axis=0)
            print(f"{m:20}{mae:8.3f}{rmse:8.3f}{r2:8.3f}{np.mean([t[0] for t in tiempos[m]]):13.1f}")
            w.writerow([m, round(mae, 4), round(rmse, 4), round(r2, 4), len(claves)])

    with open(RESULTADOS / "tiempos.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "segundos_entreno_por_fold", "segundos_prediccion_por_fold", "n_folds"])
        for m in modelos:
            w.writerow([m, *(round(float(np.mean(v)), 2) for v in zip(*tiempos[m])), len(tiempos[m])])

    print("\nT-test pareado sobre el MAE (p < 0.05 -> diferencia significativa):")
    with open(RESULTADOS / "ttest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_a", "config_b", "mae_a", "mae_b", "t", "p"])
        for a, b in combinations(modelos, 2):
            ma, mb = (np.array([tabla[m][c][0] for c in claves]) for m in (a, b))
            t, p = ttest_rel(ma, mb)
            print(f"  {a:18} vs {b:18}: MAE {ma.mean():.3f} vs {mb.mean():.3f}  t={t:7.2f}  p={p:.4g}")
            w.writerow([a, b, round(ma.mean(), 4), round(mb.mean(), 4), round(t, 3), p])

    with open(RESULTADOS / "error_por_hora.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hora", *(f"{x}_{m}" for m in modelos for x in ("mae", "rmse"))])
        for h in range(24):
            e = {m: np.array(errores_hora[m, h]) for m in modelos}
            w.writerow([h, *(round(v, 4) for m in modelos for v in (e[m].mean(), np.sqrt((e[m] ** 2).mean())))])

    with open(RESULTADOS / "error_por_estacion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_estacion", *(f"{x}_{m}" for m in modelos for x in ("mae", "rmse", "r2"))])
        for est in estaciones:
            medias = [np.mean([v for (e, _), v in tabla[m].items() if e == est], axis=0) for m in modelos]
            w.writerow([est, *(round(v, 4) for me in medias for v in me)])
    print(f"\nGuardado en {RESULTADOS.name}/: resumen.csv, ttest.csv, tiempos.csv, resultados_folds.csv, "
          "error_por_hora.csv, error_por_estacion.csv")


if __name__ == "__main__":
    main()
