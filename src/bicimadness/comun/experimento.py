"""Ejecución de un estudio, igual para todos los algoritmos.

Cada algoritmo (rna/, ridge/, lstm/...) solo tiene que dar una función
    entrenar(X, y, args) -> modelo con .predict(X)
(en modo multivariante, y y predict(X) tienen una columna por estación)
y llamar a ejecutar(). Aquí se hace todo lo demás: leer el CSV, construir las entradas (ventana +
variables; en modo global la estación NO entra, para que el modelo valga para cualquier estación,
salvo con --con-id-estacion), recorrer los bloques de la sliding window,
medir tiempos, calcular métricas y guardar los resultados en results/<algoritmo>/<fecha>_<nombre>/.

Ficheros de cada ejecución:
  config.json            argumentos, fechas, commit de git y versiones: qué se ejecutó exactamente
  resumen.csv            MAE, RMSE, R² medios (y los de la referencia "misma hora de ayer") y tiempos
  resultados_folds.csv   métricas de cada estación en cada bloque
  tiempos.csv            segundos de entrenamiento y de predicción de cada bloque
  error_por_estacion.csv error medio de cada estación
  error_por_hora.csv     error según la hora del día
  errores_por_grupo.csv  error por grupo (solo si el CSV tiene grupos, como sel60)
  predicciones.csv.gz    fecha, estación, bloque, valor real y predicción
"""
import argparse
import csv
import gzip
import json
import platform
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import datos as D

COMUNES = {"datos", "modo", "con_id_estacion", "variables", "meteo", "ventana", "semanas", "nombre", "semilla", "jobs", "hasta", "carpeta"}


def argumentos(descripcion, multivariante=False):
    """Parser con los argumentos comunes; cada algoritmo le añade los suyos.

    multivariante=True añade --modo multivariante (y lo pone por defecto): un modelo que recibe el historial
    de TODAS las estaciones a la vez y da una predicción por estación. Solo para algoritmos que lo soportan."""
    p = argparse.ArgumentParser(description=descripcion, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datos", nargs="?", default="sel60",
                   help=f"CSV de salidas por hora: {', '.join(D.ALIAS)} o una ruta (por defecto sel60)")
    modos = ["global", "estacion"] + (["multivariante"] if multivariante else [])
    defecto = "multivariante" if multivariante else "global"
    p.add_argument("--modo", choices=modos, default=defecto,
                   help="global = un modelo para todas las estaciones (no sabe qué estación es: vale para cualquiera); "
                        "estacion = un modelo por estación"
                        + ("; multivariante = un modelo que ve el historial de todas las estaciones a la vez "
                           "y da todas las predicciones (la estación es su posición, sin one-hot)" if multivariante else "")
                        + f" (por defecto {defecto})")
    p.add_argument("--con-id-estacion", action="store_true",
                   help="en modo global, meter la estación en one-hot (solo para comparar: el modelo queda atado a "
                        "las estaciones del entrenamiento; por defecto no se mete)")
    p.add_argument("--variables", default="todas",
                   help="variables de la hora a predecir separadas por comas, 'todas' o 'ninguna' "
                        f"(solo el historial). Disponibles: {', '.join(D.VARIABLES)}")
    p.add_argument("--meteo", choices=["actual", "anterior"], default="actual",
                   help="actual = meteorología de la hora que se predice (pronóstico perfecto); "
                        "anterior = la de la hora anterior, ya observada (por defecto actual)")
    p.add_argument("--ventana", type=int, default=24, help=f"horas anteriores que entran al modelo (1-{D.MAX_VENTANA})")
    p.add_argument("--semanas", type=int, default=52, help=f"semanas de entrenamiento en cada bloque (1-{D.MAX_SEMANAS})")
    p.add_argument("--nombre", help="nombre del estudio (por defecto el del CSV; si se repite se añade _1, _2...)")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--jobs", type=int,
                   help="bloques en paralelo (por defecto 2 en modo global, que ocupa memoria, y todos los núcleos por estación)")
    p.add_argument("--hasta", help="solo los bloques de test que empiezan antes de esta fecha (AAAA-MM-DD)")
    p.add_argument("--carpeta", help="guarda en esta carpeta en vez de results/<algoritmo>/<fecha>_<nombre>")
    return p


def _validar(args):
    if not 1 <= args.ventana <= D.MAX_VENTANA:
        raise SystemExit(f"--ventana debe estar entre 1 y {D.MAX_VENTANA}")
    if not 1 <= args.semanas <= D.MAX_SEMANAS:
        raise SystemExit(f"--semanas debe estar entre 1 y {D.MAX_SEMANAS}")
    nombres = {"todas": list(D.VARIABLES), "ninguna": []}.get(args.variables, args.variables.split(","))
    for n in set(nombres) - set(D.VARIABLES):
        raise SystemExit(f"variable desconocida '{n}'. Disponibles: {', '.join(D.VARIABLES)}")
    args.lista_variables = nombres
    args.hasta_fecha = datetime.strptime(args.hasta, "%Y-%m-%d") if args.hasta else None


def escalar_en_sitio(X, trozo=200_000):
    """Estandariza X (media 0, desviación 1 por columna) sin copiarla; -> (media, desviación).

    Con muchas estaciones la matriz global ocupa GB: StandardScaler haría otra copia entera."""
    n = len(X)
    suma = sum(X[i:i + trozo].sum(axis=0, dtype=np.float64) for i in range(0, n, trozo))
    media = suma / n
    cuad = sum(((X[i:i + trozo] - media) ** 2).sum(axis=0) for i in range(0, n, trozo))
    desv = np.sqrt(cuad / n)
    desv[desv == 0] = 1
    for i in range(0, n, trozo):
        X[i:i + trozo] -= media.astype(X.dtype)
        X[i:i + trozo] /= desv.astype(X.dtype)
    return media.astype(np.float32), desv.astype(np.float32)


def _bloque(k, s, series, cal, args, entrenar):
    """Entrena y predice un bloque. -> (k, {estación: predicción}, segundos entreno, segundos predicción)."""
    tr = np.arange(s - args.semanas * 7 * 24, s)
    te = np.arange(s, s + D.TEST_H)
    ests = list(series)
    preds, t_ent, t_pred = {}, 0.0, 0.0
    if args.modo == "estacion":
        for e in ests:
            Xtr, ytr = D.ventana(series[e], cal, args.ventana, tr)
            Xte, _ = D.ventana(series[e], cal, args.ventana, te)
            t0 = time.perf_counter()
            modelo = entrenar(Xtr.astype(np.float32), ytr, args)
            t1 = time.perf_counter()
            preds[e] = np.clip(modelo.predict(Xte.astype(np.float32)), 0, None)  # no hay salidas negativas
            t_ent, t_pred = t_ent + t1 - t0, t_pred + time.perf_counter() - t1
        return k, preds, t_ent, t_pred

    if args.modo == "multivariante":
        # fila t = [historial de las n horas anteriores de todas las estaciones (hora a hora), variables de t]
        # objetivo = salidas de todas las estaciones en t; el modelo devuelve (filas x estaciones)
        Y = np.stack([series[e] for e in ests], axis=1)
        sw = sliding_window_view(Y, args.ventana, axis=0)  # sw[i] = Y[i:i+n].T  (estaciones x n)

        def matriz(idx):
            hist = sw[idx - args.ventana].transpose(0, 2, 1).reshape(len(idx), -1)
            return np.hstack([hist, cal[idx]]).astype(np.float32), Y[idx]

        Xtr, ytr = matriz(tr)
        Xte, _ = matriz(te)
        t0 = time.perf_counter()
        modelo = entrenar(Xtr, ytr, args)
        t1 = time.perf_counter()
        P = np.clip(modelo.predict(Xte), 0, None)
        return k, {e: P[:, i] for i, e in enumerate(ests)}, t1 - t0, time.perf_counter() - t1

    # global: todas las estaciones apiladas en una matriz reservada de una vez (sin copias intermedias)
    n_est, n_tr = len(ests), len(tr)
    n_id = n_est if args.con_id_estacion else 0
    columnas = args.ventana + cal.shape[1] + n_id
    Xtr = np.zeros((n_est * n_tr, columnas), dtype=np.float32)
    ytr = np.empty(n_est * n_tr, dtype=np.float32)
    Xte = {}
    for i, e in enumerate(ests):
        filas = slice(i * n_tr, (i + 1) * n_tr)
        X, ytr[filas] = D.ventana(series[e], cal, args.ventana, tr)
        Xtr[filas, :X.shape[1]] = X
        if n_id:
            Xtr[filas, X.shape[1] + i] = 1  # estación en one-hot
        X, _ = D.ventana(series[e], cal, args.ventana, te)
        Xte[e] = np.zeros((len(te), columnas), dtype=np.float32)
        Xte[e][:, :X.shape[1]] = X
        if n_id:
            Xte[e][:, X.shape[1] + i] = 1
    t0 = time.perf_counter()
    modelo = entrenar(Xtr, ytr, args)
    t1 = time.perf_counter()
    del Xtr
    preds = {e: np.clip(modelo.predict(Xte[e]), 0, None) for e in ests}
    return k, preds, t1 - t0, time.perf_counter() - t1


def _metricas(real, pred):
    return mean_absolute_error(real, pred), mean_squared_error(real, pred) ** 0.5, r2_score(real, pred)


def _carpeta(algoritmo, args, inicio):
    if args.carpeta:
        return D.PROYECTO / args.carpeta, args.nombre or args.ruta_datos.stem
    base = D.RESULTADOS / algoritmo
    nombre = args.nombre or args.ruta_datos.stem.removeprefix("salidas_por_hora_")
    patron = re.compile(r"\d{4}-\d\d-\d\d_\d{4}_" + re.escape(nombre) + r"(_\d+)?")
    repetidos = [d for d in base.glob("*") if d.is_dir() and patron.fullmatch(d.name)] if base.exists() else []
    if repetidos:
        nombre += f"_{len(repetidos)}"
    return base / f"{inicio:%Y-%m-%d_%H%M}_{nombre}", nombre


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=D.PROYECTO, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except OSError:
        return None


def ejecutar(algoritmo, args, entrenar):
    """Ejecuta el estudio descrito por args con la función entrenar del algoritmo y guarda los resultados."""
    _validar(args)
    inicio = datetime.now()
    t_inicio = time.perf_counter()
    args.ruta_datos = D.ruta_datos(args.datos)
    carpeta, nombre = _carpeta(algoritmo, args, inicio)

    series, fechas, festivo = D.cargar(args.ruta_datos)
    cal = D.variables(args.lista_variables, fechas, festivo, args.meteo)
    lista = D.bloques(len(fechas), args.hasta_fecha, fechas)
    if not lista:
        raise SystemExit("no hay ningún bloque de test con esas opciones")
    jobs = args.jobs or (-1 if args.modo == "estacion" else 2)
    print(f"{algoritmo} | {args.ruta_datos.name}: {len(series)} estaciones, {len(lista)} bloques de 2 semanas "
          f"({fechas[lista[0][1]]:%Y-%m-%d} a {fechas[lista[-1][1] + D.TEST_H - 1]:%Y-%m-%d})")
    print(f"  modo {args.modo}{' con id de estación' if args.con_id_estacion and args.modo == 'global' else ''}, ventana {args.ventana} h, {args.semanas} semanas, meteo {args.meteo}, "
          f"variables: {', '.join(args.lista_variables) or 'ninguna'}", flush=True)

    res = Parallel(n_jobs=jobs, verbose=5)(delayed(_bloque)(k, s, series, cal, args, entrenar) for k, s in lista)
    segundos_total = time.perf_counter() - t_inicio
    guardar(carpeta, algoritmo, nombre, args, inicio, segundos_total, series, fechas, lista, res)
    return carpeta


def guardar(carpeta, algoritmo, nombre, args, inicio, segundos_total, series, fechas, lista, res):
    carpeta.mkdir(parents=True, exist_ok=True)
    inicio_de = dict(lista)
    horas = np.array([f.hour for f in fechas])

    folds, por_hora, por_hora_ref, tiempos = [], defaultdict(list), defaultdict(list), []
    with gzip.open(carpeta / "predicciones.csv.gz", "wt", newline="", encoding="utf-8") as fz:
        wz = csv.writer(fz)
        wz.writerow(["fecha_hora", "id_estacion", "bloque", "real", "prediccion"])
        for k, preds, t_ent, t_pred in sorted(res, key=lambda r: r[0]):
            s = inicio_de[k]
            te = np.arange(s, s + D.TEST_H)
            tiempos.append([k, f"{fechas[s]:%Y-%m-%d}", round(t_ent, 2), round(t_pred, 2)])
            for e, y in series.items():
                real, pred, ref = y[te], preds[e], y[te - 24]  # referencia: lo que salió ayer a esa hora
                folds.append([k, f"{fechas[s]:%Y-%m-%d}", e, *_metricas(real, pred), *_metricas(real, ref),
                              float(real.mean())])
                for h, a, b in zip(horas[te], np.abs(real - pred), np.abs(real - ref)):
                    por_hora[h].append(a)
                    por_hora_ref[h].append(b)
                wz.writerows([f"{fechas[t]:%Y-%m-%d %H:00}", e, k, int(r), round(float(p), 3)]
                             for t, r, p in zip(te, real, pred))

    cab = ["bloque", "inicio_test", "id_estacion", "mae", "rmse", "r2", "mae_referencia", "rmse_referencia",
           "r2_referencia", "salidas_hora"]
    _escribir(carpeta / "resultados_folds.csv", cab, [[*f[:3], *(round(v, 4) for v in f[3:])] for f in folds])
    _escribir(carpeta / "tiempos.csv", ["bloque", "inicio_test", "segundos_entreno", "segundos_prediccion"], tiempos)
    m = np.array([f[3:] for f in folds], dtype=float)  # columnas de cab[3:]

    por_est = defaultdict(list)
    for f, fila in zip(folds, m):
        por_est[f[2]].append(fila)
    _escribir(carpeta / "error_por_estacion.csv", ["id_estacion", *cab[3:]],
              [[e, *np.round(np.mean(v, axis=0), 4)] for e, v in por_est.items()])
    _escribir(carpeta / "error_por_hora.csv", ["hora", "mae", "rmse", "mae_referencia", "rmse_referencia"],
              [[h, *(round(float(x), 4) for x in (np.mean(por_hora[h]), np.sqrt(np.mean(np.square(por_hora[h]))),
                                                    np.mean(por_hora_ref[h]), np.sqrt(np.mean(np.square(por_hora_ref[h])))))]
               for h in range(24)])
    _grupos(carpeta, args, folds, m, fechas)

    t = np.array([r[2:] for r in tiempos], dtype=float)
    medias = m.mean(axis=0)
    extra = {k: v for k, v in vars(args).items() if k not in COMUNES and k not in ("ruta_datos", "lista_variables", "hasta_fecha")}
    resumen = {
        "algoritmo": algoritmo, "nombre": nombre, "fecha": f"{inicio:%Y-%m-%d %H:%M}", "datos": args.ruta_datos.name,
        "modo": args.modo, "id_estacion": "sí" if args.modo == "global" and args.con_id_estacion else "no",
        "variables": "+".join(args.lista_variables) or "ninguna", "meteo": args.meteo,
        "ventana": args.ventana, "semanas": args.semanas, "parametros": json.dumps(extra, ensure_ascii=False),
        "n_estaciones": len(series), "n_bloques": len(lista),
        "mae": round(medias[0], 4), "rmse": round(medias[1], 4), "r2": round(medias[2], 4),
        "mae_referencia": round(medias[3], 4), "rmse_referencia": round(medias[4], 4), "r2_referencia": round(medias[5], 4),
        "segundos_total": round(segundos_total, 1),
        "segundos_entreno_por_bloque": round(float(t[:, 0].mean()), 2),
        "segundos_prediccion_por_bloque": round(float(t[:, 1].mean()), 3),
    }
    _escribir(carpeta / "resumen.csv", list(resumen), [list(resumen.values())])

    config = {
        "algoritmo": algoritmo, "nombre": nombre, "inicio": inicio.isoformat(timespec="seconds"),
        "fin": datetime.now().isoformat(timespec="seconds"), "datos": args.ruta_datos.name,
        "ruta_datos": str(args.ruta_datos.relative_to(D.PROYECTO)) if args.ruta_datos.is_relative_to(D.PROYECTO) else str(args.ruta_datos),
        "argumentos": {k: v for k, v in vars(args).items() if k not in ("ruta_datos", "hasta_fecha")},
        "bloques": [{"bloque": k, "inicio_test": f"{fechas[s]:%Y-%m-%d %H:%M}"} for k, s in lista],
        "commit_git": _commit(), "python": platform.python_version(),
    }
    (carpeta / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nMAE {resumen['mae']}  RMSE {resumen['rmse']}  R² {resumen['r2']}   "
          f"(referencia ayer: MAE {resumen['mae_referencia']})")
    print(f"tiempo total {segundos_total / 60:.1f} min, entrenamiento medio por bloque {resumen['segundos_entreno_por_bloque']} s")
    print(f"Guardado en {carpeta.relative_to(D.PROYECTO) if carpeta.is_relative_to(D.PROYECTO) else carpeta}/")


def _grupos(carpeta, args, folds, m, fechas):
    """errores_por_grupo.csv: media por grupo en todos los bloques y en los posteriores a CORTE_GRUPOS."""
    fichero = D.GRUPOS.get(args.ruta_datos.name)
    if not fichero or not fichero.exists():
        return
    with open(fichero, encoding="utf-8") as f:
        grupo = {r["id_estacion"]: r["grupo"] for r in csv.DictReader(f) if r.get("grupo")}
    corte = f"{D.CORTE_GRUPOS:%Y-%m-%d}"
    periodos = {"todos": lambda f: True, f"desde_{corte}": lambda f: f[1] >= corte}
    filas = []
    for g in [*dict.fromkeys(grupo.values()), "todas"]:
        for periodo, dentro in periodos.items():
            sel = [fila for f, fila in zip(folds, m) if (g == "todas" or grupo.get(f[2]) == g) and dentro(f)]
            if sel:
                v = np.mean(sel, axis=0)
                n_est = len({f[2] for f in folds if g == "todas" or grupo.get(f[2]) == g})
                filas.append([g, periodo, n_est, round(v[6], 3), *np.round(v[:4], 4)])
    _escribir(carpeta / "errores_por_grupo.csv",
              ["grupo", "periodo", "n_estaciones", "salidas_hora", "mae", "rmse", "r2", "mae_referencia"], filas)


def _escribir(ruta, cabecera, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cabecera)
        w.writerows(filas)
