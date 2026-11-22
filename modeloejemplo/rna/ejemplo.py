"""Entrenar una RNA reutilizable y consultar horas históricas de sel60."""
import argparse
import calendar
import csv
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from bicimadness.comun import datos as D
from bicimadness.rna.entrenar import entrenar

CARPETA = Path(__file__).resolve().parent
MODELO = CARPETA / "modelo.joblib"
VENTANA = 24
INICIO = datetime(2021, 1, 2, 1)
FIN = datetime(2022, 1, 2, 1)  # extremo excluido: 365 días completos


def cargar():
    ruta = D.ruta_datos("sel60")
    # Comprobar alineación: cargar() del proyecto presupone que todas las estaciones
    # tienen exactamente las mismas horas, en orden y sin huecos.
    horas = {}
    with ruta.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            horas.setdefault(r["id_estacion"], []).append(r["fecha_hora"])
    primera = next(iter(horas.values()))
    if len(horas) != 60 or any(h != primera for h in horas.values()):
        raise ValueError("sel60 debe contener 60 estaciones con las mismas horas ordenadas.")
    series, fechas, festivo = D.cargar(ruta)
    if any(b - a != timedelta(hours=1) for a, b in zip(fechas, fechas[1:])):
        raise ValueError("Hay horas duplicadas, desordenadas o ausentes en sel60.")
    cal = D.variables(list(D.VARIABLES), fechas, festivo, meteo="actual")
    if not np.isfinite(cal).all() or any(not np.isfinite(y).all() or (y < 0).any() for y in series.values()):
        raise ValueError("Los datos contienen valores no válidos.")
    return series, fechas, cal


def indices(fechas, inicio, fin):
    if inicio.minute or inicio.second or fin.minute or fin.second or fin <= inicio:
        raise ValueError("Indica horas completas y un intervalo de duración positiva.")
    if inicio < fechas[VENTANA] or fin > fechas[-1] + timedelta(hours=1):
        raise ValueError(f"Periodo fuera de los datos: desde {fechas[VENTANA]} hasta {fechas[-1]} inclusive.")
    mapa = {f: i for i, f in enumerate(fechas)}
    if inicio not in mapa:
        raise ValueError("La fecha inicial no existe en los datos.")
    s = mapa[inicio]
    return np.arange(s, s + int((fin - inicio).total_seconds() / 3600))


def siguiente_mes(fecha):
    anio = fecha.year + (fecha.month == 12)
    mes = fecha.month % 12 + 1
    return fecha.replace(year=anio, month=mes, day=min(fecha.day, calendar.monthrange(anio, mes)[1]))


def escribir_csv(ruta, cabecera, filas):
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cabecera)
        w.writerows(filas)


def redondear(pred):
    """Salidas enteras no negativas; los empates .5 se redondean hacia arriba."""
    return np.floor(np.maximum(0, pred) + 0.5).astype(np.int64)


def wape(real, pred):
    """Error absoluto total como porcentaje del volumen real; incluye horas cero."""
    real, pred = np.asarray(real, dtype=float), np.asarray(pred, dtype=float)
    volumen = np.abs(real).sum()
    return float(100 * np.abs(real - pred).sum() / volumen) if volumen > 0 else None


def predecir(artefacto, datos, inicio, fin, destino):
    series, fechas, cal = datos
    if inicio < datetime.fromisoformat(artefacto["fin_entrenamiento_exclusivo"]):
        raise ValueError("Elige una fecha posterior al entrenamiento para evitar evaluar horas aprendidas.")
    if list(series) != artefacto["estaciones"]:
        raise ValueError("Las estaciones no coinciden con las del modelo guardado.")
    idx = indices(fechas, inicio, fin)
    reales, predichos, referencias = [], [], []
    filas = []
    for est, y in series.items():
        # Cada fila usa las 24 observaciones REALES anteriores a su propia hora.
        # Evaluar el mes en lote equivale a ir actualizando el historial cada hora.
        X, real = D.ventana(y, cal, VENTANA, idx)
        pred = redondear(artefacto["modelo"].predict(X.astype(np.float32)))
        ref = y[idx - 24]
        reales.extend(real)
        predichos.extend(pred)
        referencias.extend(ref)
        filas.extend((fechas[t].strftime("%Y-%m-%d %H:%M"), est, int(p), int(r),
                      int(abs(r - p)), int(a)) for t, p, r, a in zip(idx, pred, real, ref))
    resumen = {"inicio": inicio.isoformat(), "fin_exclusivo": fin.isoformat(),
               "n_horas": len(idx), "n_estaciones": len(series),
               "meteorologia": "real observada de la hora objetivo; no es un pronóstico meteorológico",
               "historial": "24 horas reales anteriores, actualizado cada hora",
               "redondeo": "entero más cercano, .5 hacia arriba; aplicado antes de todas las métricas",
               "definicion_wape": "100 * suma(|real-prediccion|)/suma(|real|); null si el volumen real total es cero"}
    for sufijo, valores in (("", predichos), ("_referencia", referencias)):
        resumen.update({"mae" + sufijo: float(mean_absolute_error(reales, valores)),
                        "rmse" + sufijo: float(mean_squared_error(reales, valores) ** 0.5),
                        "r2" + sufijo: float(r2_score(reales, valores))})
        resumen["wape_pct" + sufijo] = wape(reales, valores)
    destino.mkdir(parents=True, exist_ok=True)
    escribir_csv(destino / "predicciones.csv",
                 ["fecha_hora", "id_estacion", "prediccion", "real", "error_absoluto", "referencia_ayer"],
                 sorted(filas, key=lambda r: (r[0], int(r[1]))))
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(idx)} horas × {len(series)} estaciones. MAE={resumen['mae']:.4f}, "
          f"RMSE={resumen['rmse']:.4f}, R²={resumen['r2']:.4f}")
    porcentaje = resumen["wape_pct"]
    print("WAPE: " + (f"{porcentaje:.2f} %" if porcentaje is not None
                      else "no definido (volumen real total cero)"))
    print(f"Predicciones y comparación real: {destino}")
    if len(idx) == 1:
        print("Estación  Predicción  Real")
        for _, est, pred, real, *_ in sorted(filas, key=lambda r: int(r[1])):
            print(f"{est:>8}  {pred:>10d}  {real:>4}")
    return resumen


def entrenar_ejemplo(datos):
    series, fechas, cal = datos
    idx = indices(fechas, INICIO, FIN)
    X = np.empty((len(series) * len(idx), VENTANA + cal.shape[1]), dtype=np.float32)
    y = np.empty(len(X), dtype=np.float32)
    for i, serie in enumerate(series.values()):
        tramo = slice(i * len(idx), (i + 1) * len(idx))
        X[tramo], y[tramo] = D.ventana(serie, cal, VENTANA, idx)
    print(f"Entrenando RNA global: {len(idx)} horas × {len(series)} estaciones; "
          f"desde {INICIO} hasta {FIN} (excluido).", flush=True)
    t = time.perf_counter()
    args = SimpleNamespace(modo="global", ocultas=[64, 32], iteraciones=200, semilla=0)
    modelo = entrenar(X, y, args)
    meta = {"algoritmo": "RNA global sel60", "inicio_entrenamiento": INICIO.isoformat(),
            "fin_entrenamiento_exclusivo": FIN.isoformat(), "ventana": VENTANA,
            "n_horas_entrenamiento": len(idx), "n_filas_entrenamiento": len(y),
            "estaciones": list(series), "variables": list(D.VARIABLES), "meteo": "actual",
            "con_id_estacion": False, "ocultas": [64, 32], "semilla": 0,
            "iteraciones_maximas": 200, "iteraciones_realizadas": int(modelo.red.n_iter_),
            "early_stopping": True, "fraccion_validacion_interna": 0.1,
            "sklearn": sklearn.__version__, "segundos_entrenamiento": time.perf_counter() - t}
    artefacto = {**meta, "modelo": modelo}
    joblib.dump(artefacto, MODELO)
    (CARPETA / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo y escalado guardados en {MODELO}", flush=True)
    predecir(artefacto, datos, FIN, siguiente_mes(FIN), CARPETA / "evaluacion_mes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="accion", required=True)
    sub.add_parser("entrenar", help="Entrena, guarda el modelo y evalúa el mes posterior")
    p = sub.add_parser("predecir", help="Consulta una hora o simula un periodo con historial real actualizado")
    p.add_argument("--fecha", required=True, help='Hora inicial: "2022-01-15 18:00"')
    duracion = p.add_mutually_exclusive_group()
    duracion.add_argument("--horas", type=int, default=1)
    duracion.add_argument("--mes", action="store_true", help="Hasta la misma fecha del mes siguiente")
    args = parser.parse_args()
    try:
        inicio = None
        if args.accion == "predecir":
            inicio = datetime.strptime(args.fecha, "%Y-%m-%d %H:%M")
            if args.horas < 1:
                raise ValueError("--horas debe ser positivo.")
            if not MODELO.exists():
                raise ValueError("Primero ejecuta el comando entrenar.")
        with threadpool_limits(limits=2):
            datos = cargar()
            if args.accion == "entrenar":
                entrenar_ejemplo(datos)
            else:
                artefacto = joblib.load(MODELO)
                if artefacto["sklearn"] != sklearn.__version__:
                    raise ValueError(f"Usa scikit-learn {artefacto['sklearn']}, la versión del entrenamiento.")
                fin = siguiente_mes(inicio) if args.mes else inicio + timedelta(hours=args.horas)
                destino = CARPETA / "consultas" / f"{inicio:%Y%m%d_%H%M}_{fin:%Y%m%d_%H%M}"
                predecir(artefacto, datos, inicio, fin, destino)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
