"""Analiza las predicciones guardadas del LSTM y las compara con la referencia de 24 horas."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

RAIZ = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = RAIZ / "results" / "lstm_selected_60"


def mae(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(real - pred)))


def rmse(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((real - pred) ** 2)))


def r2(real: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((real - pred) ** 2)
    ss_tot = np.sum((real - np.mean(real, axis=0, keepdims=True)) ** 2)
    return 1.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def metricas(real: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    return mae(real, pred), rmse(real, pred), r2(real, pred)


def load_station_groups() -> dict[str, str]:
    path = RAIZ / "data" / "processed" / "estaciones_seleccionadas.csv"
    groups = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            groups[row["id_estacion"]] = row["grupo"]
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza predicciones guardadas del LSTM de 60 estaciones")
    parser.add_argument("--resultados", default=str(DEFAULT_RESULTS))
    parser.add_argument(
        "--rna-folds",
        help="CSV results_folds.csv generado por rna_estaciones_sel60_adaptada.py",
    )
    args = parser.parse_args()

    resultados = Path(args.resultados)
    if not resultados.is_absolute():
        resultados = RAIZ / resultados
    if not resultados.exists():
        raise SystemExit(f"Results directory does not exist: {resultados}")

    files = sorted(resultados.glob("*_prediction.npz"))
    if not files:
        raise SystemExit(f"No *_prediction.npz files found in {resultados}")

    predictions_by_config_fold = defaultdict(dict)
    groups = load_station_groups()

    for file in files:
        name = file.name.replace("_prediction.npz", "")
        parts = name.rsplit("_fold_", 1)
        if len(parts) != 2:
            continue
        config, fold_text = parts
        fold = int(fold_text)
        data = np.load(file, allow_pickle=True)
        predictions_by_config_fold[config][fold] = {
            "timestamps": data["timestamps"],
            "station_ids": data["station_ids"],
            "real": data["real"],
            "prediction": data["prediction"],
            "yesterday": data["yesterday"],
        }

    station_ids = next(iter(next(iter(predictions_by_config_fold.values())).values()))["station_ids"]
    station_ids = [str(value) for value in station_ids]
    reference_folds = next(iter(predictions_by_config_fold.values()))
    for config, folds in predictions_by_config_fold.items():
        for fold_index in set(reference_folds) & set(folds):
            reference = reference_folds[fold_index]
            candidate = folds[fold_index]
            if not np.array_equal(reference["timestamps"], candidate["timestamps"]):
                raise ValueError(
                    f"Test timestamps differ for fold {fold_index}: "
                    f"{next(iter(predictions_by_config_fold))} vs {config}"
                )
            if not np.array_equal(reference["real"], candidate["real"]):
                raise ValueError(f"Actual targets differ for fold {fold_index}: {config}")
            if not np.array_equal(reference["yesterday"], candidate["yesterday"]):
                raise ValueError(f"Yesterday baseline differs for fold {fold_index}: {config}")

    summary_rows = []
    station_rows = []
    hour_rows = []
    group_rows = []
    day_type_rows = []
    ttest_rows = []

    configs = sorted(predictions_by_config_fold)

    for config in configs:
        folds = predictions_by_config_fold[config]
        real_parts = [fold["real"] for fold in folds.values()]
        pred_parts = [fold["prediction"] for fold in folds.values()]
        base_parts = [fold["yesterday"] for fold in folds.values()]
        real = np.concatenate(real_parts, axis=0)
        pred = np.concatenate(pred_parts, axis=0)
        base = np.concatenate(base_parts, axis=0)
        fold_metrics = [
            np.mean(
                [
                    metricas(values["real"][:, station], values["prediction"][:, station])
                    for station in range(values["real"].shape[1])
                ],
                axis=0,
            )
            for values in folds.values()
        ]
        summary_rows.append([
            config,
            float(np.mean([values[0] for values in fold_metrics])),
            float(np.mean([values[1] for values in fold_metrics])),
            float(np.mean([values[2] for values in fold_metrics])),
            len(folds),
        ])

        baseline_mae_fold = []
        lstm_mae_fold = []
        common_folds = sorted(set(folds) & set(predictions_by_config_fold[configs[0]]))
        for fold in common_folds:
            values = folds[fold]
            lstm_mae_fold.append(mae(values["real"], values["prediction"]))
            baseline_mae_fold.append(mae(values["real"], values["yesterday"]))
        if len(common_folds) >= 2:
            t, p = ttest_rel(lstm_mae_fold, baseline_mae_fold)
        else:
            t, p = float("nan"), float("nan")
        ttest_rows.append([config, "referencia_24h", float(np.mean(lstm_mae_fold)), float(np.mean(baseline_mae_fold)), float(t), float(p), len(common_folds)])

        for station_index, station_id in enumerate(station_ids):
            station_real = real[:, station_index]
            station_pred = pred[:, station_index]
            for prediction_name, station_prediction in (
                (config, station_pred),
                ("referencia_24h", base[:, station_index]),
            ):
                station_rows.append([
                    station_id,
                    groups.get(station_id, ""),
                    prediction_name,
                    mae(station_real, station_prediction),
                    rmse(station_real, station_prediction),
                    r2(station_real, station_prediction),
                ])

        timestamps = np.concatenate([fold["timestamps"] for fold in folds.values()], axis=0)
        hours = np.array([timestamp.hour for timestamp in timestamps])
        laborable = np.array([timestamp.weekday() < 5 for timestamp in timestamps])
        for hour in range(24):
            mask = hours == hour
            if mask.any():
                for prediction_name, hour_prediction in ((config, pred), ("referencia_24h", base)):
                    hour_rows.append([
                        hour,
                        prediction_name,
                        mae(real[mask], hour_prediction[mask]),
                        rmse(real[mask], hour_prediction[mask]),
                    ])

        for day_name, day_mask in (("entre_semana", laborable), ("fin_de_semana", ~laborable)):
            if day_mask.any():
                for prediction_name, day_prediction in ((config, pred), ("referencia_24h", base)):
                    day_type_rows.append([
                        day_name,
                        prediction_name,
                        mae(real[day_mask], day_prediction[day_mask]),
                        rmse(real[day_mask], day_prediction[day_mask]),
                        r2(real[day_mask], day_prediction[day_mask]),
                        int(day_mask.sum()),
                    ])

        for group in sorted(set(groups.get(station_id, "") for station_id in station_ids)):
            indices = [i for i, station_id in enumerate(station_ids) if groups.get(station_id, "") == group]
            for prediction_name, group_prediction in ((config, pred), ("referencia_24h", base)):
                group_mae, group_rmse, group_r2 = metricas(real[:, indices], group_prediction[:, indices])
                group_rows.append([
                    group,
                    prediction_name,
                    group_mae,
                    group_rmse,
                    group_r2,
                    len(indices),
                ])

    baseline_folds = next(iter(predictions_by_config_fold.values()))
    baseline_metrics = [
        np.mean(
            [
                metricas(fold["real"][:, station], fold["yesterday"][:, station])
                for station in range(fold["real"].shape[1])
            ],
            axis=0,
        )
        for fold in baseline_folds.values()
    ]
    summary_rows.append([
        "referencia_24h",
        float(np.mean([values[0] for values in baseline_metrics])),
        float(np.mean([values[1] for values in baseline_metrics])),
        float(np.mean([values[2] for values in baseline_metrics])),
        len(baseline_folds),
    ])

    for i, config_a in enumerate(configs):
        for config_b in configs[i + 1:]:
            common_folds = sorted(
                set(predictions_by_config_fold[config_a])
                & set(predictions_by_config_fold[config_b])
            )
            mae_a = [
                mae(
                    predictions_by_config_fold[config_a][fold]["real"],
                    predictions_by_config_fold[config_a][fold]["prediction"],
                )
                for fold in common_folds
            ]
            mae_b = [
                mae(
                    predictions_by_config_fold[config_b][fold]["real"],
                    predictions_by_config_fold[config_b][fold]["prediction"],
                )
                for fold in common_folds
            ]
            if len(common_folds) >= 2:
                t, p = ttest_rel(mae_a, mae_b)
            else:
                t, p = float("nan"), float("nan")
            ttest_rows.append([
                config_a,
                config_b,
                float(np.mean(mae_a)),
                float(np.mean(mae_b)),
                float(t),
                float(p),
                len(common_folds),
            ])

    if args.rna_folds:
        rna_path = Path(args.rna_folds)
        if not rna_path.is_absolute():
            rna_path = RAIZ / rna_path
        mlp_records = defaultdict(lambda: defaultdict(list))
        with rna_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["config"] != "rna_global_24h_52sem":
                    continue
                mlp_records[int(row["fold"])][row["id_estacion"]].append(
                    (float(row["mae"]), float(row["rmse"]), float(row["r2"]))
                )
        mlp_fold_metrics = {}
        for fold, station_records in mlp_records.items():
            metrics_by_station = [
                np.mean(values, axis=0)
                for values in station_records.values()
            ]
            if metrics_by_station:
                mlp_fold_metrics[fold] = np.mean(metrics_by_station, axis=0)
        if not mlp_fold_metrics:
            raise SystemExit(f"No rna_global_24h_52sem rows found in {rna_path}")

        mlp_metrics = np.mean(list(mlp_fold_metrics.values()), axis=0)
        summary_rows.append([
            "rna_global_24h_52sem",
            *[float(value) for value in mlp_metrics],
            len(mlp_fold_metrics),
        ])
        lstm_candidates = [
            config for config in configs
            if config.startswith("lstm_24h_52sem_con_meteo")
        ]
        if lstm_candidates:
            lstm_config = lstm_candidates[0]
            folds = sorted(set(mlp_fold_metrics) & set(predictions_by_config_fold[lstm_config]))
            lstm_fold_mae = [
                mae(
                    predictions_by_config_fold[lstm_config][fold]["real"],
                    predictions_by_config_fold[lstm_config][fold]["prediction"],
                )
                for fold in folds
            ]
            mlp_fold_mae = [float(mlp_fold_metrics[fold][0]) for fold in folds]
            if len(folds) >= 2:
                t, p = ttest_rel(lstm_fold_mae, mlp_fold_mae)
            else:
                t, p = float("nan"), float("nan")
            ttest_rows.append([
                lstm_config,
                "rna_global_24h_52sem",
                float(np.mean(lstm_fold_mae)),
                float(np.mean(mlp_fold_mae)),
                float(t),
                float(p),
                len(folds),
            ])

    with (resultados / "lstm_comparison_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "mae", "rmse", "r2", "n_folds"])
        writer.writerows(summary_rows)

    with (resultados / "lstm_paired_ttest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "baseline", "mae_lstm", "mae_baseline", "t", "p", "n_folds"])
        writer.writerows(ttest_rows)

    with (resultados / "lstm_error_by_station.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_estacion", "grupo", "config", "mae", "rmse", "r2"])
        writer.writerows(station_rows)

    with (resultados / "lstm_error_by_hour.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hora", "config", "mae", "rmse"])
        writer.writerows(hour_rows)

    with (resultados / "lstm_error_por_grupo.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["grupo_estacion", "config", "mae", "rmse", "r2", "n_estaciones"])
        writer.writerows(group_rows)

    with (resultados / "lstm_error_por_tipo_dia.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tipo_dia", "config", "mae", "rmse", "r2", "n_horas"])
        writer.writerows(day_type_rows)

    station_diagnostics = {}
    for config in [*configs, "referencia_24h"]:
        rows = [row for row in station_rows if row[2] == config]
        ordered = sorted(rows, key=lambda row: row[3])
        station_diagnostics[config] = {
            "best_stations": [{"id_estacion": row[0], "grupo": row[1], "mae": row[3]} for row in ordered[:5]],
            "worst_stations": [{"id_estacion": row[0], "grupo": row[1], "mae": row[3]} for row in ordered[-5:][::-1]],
        }

    fold_diagnostics = {}
    for config in configs:
        folds = predictions_by_config_fold[config]
        values = {
            str(fold): mae(data["real"], data["prediction"])
            for fold, data in folds.items()
        }
        fold_diagnostics[config] = {
            "fold_mae": values,
            "best_fold": min(values, key=values.get),
            "worst_fold": max(values, key=values.get),
        }

    (resultados / "lstm_resumen_diagnostico.json").write_text(
        json.dumps(
            {
                "estaciones": station_diagnostics,
                "folds": fold_diagnostics,
                "metric_definition": "MAE, RMSE, and R2 are computed per station and fold, then averaged equally over stations and folds.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Saved LSTM comparison, paired T-test, station, and hourly analysis under", resultados)
    for row in summary_rows:
        print(f"{row[0]:25} MAE={row[1]:.4f} RMSE={row[2]:.4f} R2={row[3]:.4f}")


if __name__ == "__main__":
    main()
