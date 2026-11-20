"""Prepara las salidas horarias y la meteorología de 60 estaciones para entrenar el LSTM.

The script creates one wide matrix of next-hour targets and one wide feature matrix,
plus the station/timestamp metadata needed to construct chronological sequences.
It uses only meteo_horario.csv as the weather source, for consistency with existing MLP runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
DATOS = RAIZ / "data" / "processed" / "salidas_por_hora_sel60.csv"
METEO = RAIZ / "data" / "raw" / "meteo_horario.csv"
ESTACIONES = RAIZ / "data" / "processed" / "estaciones_seleccionadas.csv"
NPZ_SALIDA = RAIZ / "data" / "processed" / "lstm_model_data.npz"
MANIFEST_SALIDA = RAIZ / "data" / "processed" / "lstm_model_data.json"


def leer_tiempos_y_estaciones() -> tuple[list[datetime], list[str], dict[str, dict[str, str]]]:
    tiempos: list[datetime] = []
    tiempo_visto: set[datetime] = set()
    estaciones: list[str] = []
    estacion_vista: set[str] = set()

    with DATOS.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            estacion = fila["id_estacion"]
            if estacion not in estacion_vista:
                estacion_vista.add(estacion)
                estaciones.append(estacion)

            tiempo = datetime.strptime(fila["fecha_hora"], "%Y-%m-%d %H:%M")
            if tiempo not in tiempo_visto:
                tiempo_visto.add(tiempo)
                tiempos.append(tiempo)

    tiempos.sort()

    metadata: dict[str, dict[str, str]] = {}
    with ESTACIONES.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            metadata[fila["id_estacion"]] = dict(fila)

    return tiempos, estaciones, metadata


def cargar_objetivos(tiempos: list[datetime], estaciones: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tiempo_a_fila = {tiempo: i for i, tiempo in enumerate(tiempos)}
    estacion_a_columna = {estacion: i for i, estacion in enumerate(estaciones)}
    objetivos = np.full((len(tiempos), len(estaciones)), np.nan, dtype=np.float32)
    festivo = np.zeros(len(tiempos), dtype=np.float32)

    seen: set[tuple[datetime, str]] = set()
    with DATOS.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            tiempo = datetime.strptime(fila["fecha_hora"], "%Y-%m-%d %H:%M")
            estacion = fila["id_estacion"]
            key = (tiempo, estacion)
            if key in seen:
                raise ValueError(f"Duplicate target row for station {estacion} at {tiempo}")
            seen.add(key)

            row = tiempo_a_fila[tiempo]
            column = estacion_a_columna[estacion]
            objetivo = float(fila["salidas"])
            if objetivo < 0:
                raise ValueError(f"Negative target at station {estacion}, timestamp {tiempo}")
            objetivos[row, column] = objetivo
            festivo[row] = 1.0 if fila["festivo"] == "1" else 0.0

    if np.isnan(objetivos).any():
        raise ValueError("The target matrix has missing station-hour cells")

    return objetivos, festivo


def cargar_meteo(tiempos: list[datetime]) -> np.ndarray:
    weather: dict[datetime, tuple[float, float]] = {}
    duplicates: set[datetime] = set()

    with METEO.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            tiempo = datetime.strptime(fila["fecha_hora"], "%Y-%m-%d %H:%M")
            if tiempo in weather:
                duplicates.add(tiempo)
            weather[tiempo] = (float(fila["temperatura"]), float(fila["precipitacion"]))

    if duplicates:
        raise ValueError(f"Duplicate timestamps in {METEO.name}: {len(duplicates)}")

    meteo = np.empty((len(tiempos), 2), dtype=np.float32)
    missing: list[datetime] = []
    for i, tiempo in enumerate(tiempos):
        if tiempo not in weather:
            missing.append(tiempo)
            continue
        meteo[i] = weather[tiempo]

    if missing:
        preview = ", ".join(t.strftime("%Y-%m-%d %H:%M") for t in missing[:5])
        raise ValueError(f"Missing meteo_horario.csv rows for target timestamps: {preview}")

    if not np.isfinite(meteo).all():
        raise ValueError("meteo_horario.csv contains missing or non-finite values")

    return meteo


def crear_exogenas(tiempos: list[datetime], meteo: np.ndarray, festivo: np.ndarray) -> np.ndarray:
    hora = np.array([tiempo.hour for tiempo in tiempos], dtype=np.float32)
    dia_semana = np.array([tiempo.weekday() for tiempo in tiempos], dtype=np.float32)
    mes = np.array([tiempo.month - 1 for tiempo in tiempos], dtype=np.float32)

    features = np.column_stack([
        meteo[:, 0],
        meteo[:, 1],
        np.sin(2 * np.pi * hora / 24),
        np.cos(2 * np.pi * hora / 24),
        np.sin(2 * np.pi * dia_semana / 7),
        np.cos(2 * np.pi * dia_semana / 7),
        np.sin(2 * np.pi * mes / 12),
        np.cos(2 * np.pi * mes / 12),
        festivo,
    ]).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Generated exogenous feature matrix contains non-finite values")
    return features


def validar_timeline(tiempos: list[datetime]) -> None:
    """Fail if the target timeline is duplicated or has a gap in hourly chronology."""
    if len(tiempos) != len(set(tiempos)):
        raise ValueError("The target timeline contains duplicate timestamps")
    gaps = [
        (a, b)
        for a, b in zip(tiempos, tiempos[1:])
        if (b - a).total_seconds() != 3600
    ]
    if gaps:
        preview = ", ".join(f"{a:%Y-%m-%d %H:%M}->{b:%Y-%m-%d %H:%M}" for a, b in gaps[:5])
        raise ValueError(f"The target timeline is not consecutive hourly: {preview}")


def fechas_objetivo(tiempos: list[datetime], ventana: int) -> list[datetime]:
    """Return the timestamp associated with each supervised target row."""
    if ventana < 1:
        raise ValueError("window must be at least 1")
    if len(tiempos) <= ventana:
        raise ValueError("the window is longer than the available timeline")
    return tiempos[ventana:].copy()


def construir_secuencias(
    objetivos: np.ndarray,
    exogenas: np.ndarray,
    ventana: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build LSTM samples: past 60-station targets, exogenous features for t, and y[t]."""
    if ventana < 1:
        raise ValueError("window must be at least 1")
    if len(objetivos) != len(exogenas):
        raise ValueError("targets and exogenous features must have the same row count")

    n_samples = len(objetivos) - ventana
    if n_samples <= 0:
        raise ValueError("the window is longer than the available timeline")

    X = np.stack([objetivos[t - ventana:t] for t in range(ventana, len(objetivos))])
    exogenas_objetivo = exogenas[ventana:].copy()
    y = objetivos[ventana:].copy()
    return X, exogenas_objetivo, y


def crear_folds(
    n_horas: int,
    ventanas: list[int],
    semanas_entrenamiento: list[int],
    test_h: int = 336,
) -> list[dict[str, int]]:
    """Create fair chronological fold boundaries shared by every window/training configuration."""
    if test_h < 1:
        raise ValueError("test_h must be at least 1")
    if not ventanas or not semanas_entrenamiento:
        raise ValueError("at least one window and one training length are required")

    primer_test = max(ventanas) + max(semanas_entrenamiento) * 7 * 24
    folds = []
    for fold, test_start in enumerate(range(primer_test, n_horas - test_h + 1, test_h)):
        folds.append({
            "fold": fold,
            "test_start": test_start,
            "test_end": test_start + test_h,
        })
    if not folds:
        raise ValueError("the requested windows/training lengths leave no complete test fold")
    return folds


def ajustar_escalado(
    objetivos: np.ndarray,
    exogenas: np.ndarray,
    inicio: int,
    fin: int,
) -> dict[str, np.ndarray]:
    """Fit station and exogenous scalers only on one chronological training interval."""
    if inicio < 0 or fin > len(objetivos) or inicio >= fin:
        raise ValueError("invalid scaling interval")

    objetivo_medio = objetivos[inicio:fin].mean(axis=0, dtype=np.float64).astype(np.float32)
    objetivo_std = objetivos[inicio:fin].std(axis=0, dtype=np.float64).astype(np.float32)
    objetivo_std = np.where(objetivo_std > 1e-8, objetivo_std, 1.0).astype(np.float32)

    exog_medio = exogenas[inicio:fin].mean(axis=0, dtype=np.float64).astype(np.float32)
    exog_std = exogenas[inicio:fin].std(axis=0, dtype=np.float64).astype(np.float32)
    exog_std = np.where(exog_std > 1e-8, exog_std, 1.0).astype(np.float32)

    return {
        "objetivo_medio": objetivo_medio,
        "objetivo_std": objetivo_std,
        "exog_medio": exog_medio,
        "exog_std": exog_std,
    }


def escalar_objetivos(objetivos: np.ndarray, escalado: dict[str, np.ndarray]) -> np.ndarray:
    return ((objetivos - escalado["objetivo_medio"]) / escalado["objetivo_std"]).astype(np.float32)


def escalar_exogenas(exogenas: np.ndarray, escalado: dict[str, np.ndarray]) -> np.ndarray:
    return ((exogenas - escalado["exog_medio"]) / escalado["exog_std"]).astype(np.float32)


def desescalar_objetivos(objetivos_escalados: np.ndarray, escalado: dict[str, np.ndarray]) -> np.ndarray:
    return (
        objetivos_escalados * escalado["objetivo_std"] + escalado["objetivo_medio"]
    ).astype(np.float32)


def dividir_fold(
    objetivos: np.ndarray,
    exogenas: np.ndarray,
    ventana: int,
    test_start: int,
    test_end: int,
    semanas_entrenamiento: int,
    horas_validacion: int = 168,
) -> dict[str, np.ndarray]:
    """Prepare one leakage-free fold: early train, final-week validation, and test arrays."""
    training_start = test_start - semanas_entrenamiento * 7 * 24
    if training_start < ventana:
        raise ValueError("training interval begins before the first complete supervised window")
    if test_end > len(objetivos):
        raise ValueError("test interval exceeds the available timeline")
    if horas_validacion < 1 or horas_validacion >= test_start - training_start:
        raise ValueError("invalid inner-validation interval")

    fit_end = test_start - horas_validacion
    escalado = ajustar_escalado(objetivos, exogenas, training_start, fit_end)

    train_target_times = range(training_start, fit_end)
    val_target_times = range(fit_end, test_start)
    test_target_times = range(test_start, test_end)

    def secuencia(indices: range) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_indices = list(indices)
        past = np.stack([
            objetivos[t - ventana:t]
            for t in target_indices
        ])
        target_exog = exogenas[target_indices]
        target = objetivos[target_indices]
        past = escalar_objetivos(past, escalado)
        target_exog = escalar_exogenas(target_exog, escalado)
        target = escalar_objetivos(target, escalado)
        return past, target_exog, target

    X_train, exog_train, y_train = secuencia(train_target_times)
    X_val, exog_val, y_val = secuencia(val_target_times)
    X_test, exog_test, y_test = secuencia(test_target_times)

    return {
        **escalado,
        "X_train": X_train,
        "exog_train": exog_train,
        "y_train": y_train,
        "X_val": X_val,
        "exog_val": exog_val,
        "y_val": y_val,
        "X_test": X_test,
        "exog_test": exog_test,
        "y_test": y_test,
        "train_start": np.array(training_start),
        "fit_end": np.array(fit_end),
        "test_start": np.array(test_start),
        "test_end": np.array(test_end),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data for the 60-station LSTM")
    parser.add_argument("--ventana", type=int, default=24, help="window used for sequence validation")
    parser.add_argument("--ventanas", type=int, nargs="+", default=[24, 48, 168], help="candidate windows documented in the manifest")
    parser.add_argument("--semanas", type=int, nargs="+", default=[26, 52], help="candidate chronological training lengths")
    parser.add_argument("--test-h", type=int, default=336, help="two-week test block length")
    parser.add_argument("--validacion-h", type=int, default=168, help="inner validation hours held from the end of training")
    parser.add_argument("--npz", default=str(NPZ_SALIDA), help="output .npz path")
    parser.add_argument("--manifest", default=str(MANIFEST_SALIDA), help="output JSON manifest path")
    args = parser.parse_args()

    npz_salida = Path(args.npz)
    if not npz_salida.is_absolute():
        npz_salida = RAIZ / npz_salida
    manifest_salida = Path(args.manifest)
    if not manifest_salida.is_absolute():
        manifest_salida = RAIZ / manifest_salida

    tiempos, estaciones, metadata = leer_tiempos_y_estaciones()
    validar_timeline(tiempos)
    objetivos, festivo = cargar_objetivos(tiempos, estaciones)
    meteo = cargar_meteo(tiempos)
    exogenas = crear_exogenas(tiempos, meteo, festivo)
    X_demo, exogenas_demo, y_demo = construir_secuencias(objetivos, exogenas, args.ventana)
    folds = crear_folds(len(tiempos), args.ventanas, args.semanas, args.test_h)

    npz_salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_salida,
        timestamps=np.array([t.strftime("%Y-%m-%d %H:%M") for t in tiempos]),
        station_ids=np.array(estaciones),
        targets=objetivos,
        exogenous=exogenas,
    )

    station_metadata = [metadata.get(estacion, {"id_estacion": estacion}) for estacion in estaciones]
    feature_names = [
        "temperatura",
        "precipitacion",
        "hora_sin",
        "hora_cos",
        "dia_semana_sin",
        "dia_semana_cos",
        "mes_sin",
        "mes_cos",
        "festivo",
    ]
    manifest = {
        "weather_source": str(METEO.relative_to(RAIZ)),
        "target_source": str(DATOS.relative_to(RAIZ)),
        "station_metadata_source": str(ESTACIONES.relative_to(RAIZ)),
        "hours": len(tiempos),
        "stations": len(estaciones),
        "start": tiempos[0].strftime("%Y-%m-%d %H:%M"),
        "end": tiempos[-1].strftime("%Y-%m-%d %H:%M"),
        "target_shape": list(objetivos.shape),
        "exogenous_shape": list(exogenas.shape),
        "exogenous_features": feature_names,
        "station_ids": estaciones,
        "station_metadata": station_metadata,
        "missing_targets": 0,
        "negative_targets": 0,
        "missing_weather": 0,
        "target_semantics": "y[t] contains all 60 departures in hour t; exogenous[t] is weather/calendar for hour t",
        "candidate_windows": args.ventanas,
        "candidate_training_weeks": args.semanas,
        "test_hours": args.test_h,
        "inner_validation_hours": args.validacion_h,
        "folds": folds,
        "sequence_shapes": {
            str(ventana): {
                "past_targets": [len(objetivos) - ventana, ventana, len(estaciones)],
                "target_exogenous": [len(objetivos) - ventana, len(feature_names)],
                "target": [len(objetivos) - ventana, len(estaciones)],
                "target_timestamp_first": fechas_objetivo(tiempos, ventana)[0].strftime("%Y-%m-%d %H:%M"),
                "target_timestamp_last": fechas_objetivo(tiempos, ventana)[-1].strftime("%Y-%m-%d %H:%M"),
            }
            for ventana in args.ventanas
        },
        "demo_window": args.ventana,
        "demo_past_targets_shape": list(X_demo.shape),
        "demo_target_exogenous_shape": list(exogenas_demo.shape),
        "demo_target_shape": list(y_demo.shape),
    }
    manifest_salida.parent.mkdir(parents=True, exist_ok=True)
    manifest_salida.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {npz_salida}")
    print(f"Wrote {manifest_salida}")


if __name__ == "__main__":
    main()
