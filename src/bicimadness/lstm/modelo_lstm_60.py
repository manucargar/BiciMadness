"""LSTM multivariante global para predecir salidas de 60 estaciones BiciMAD.

Uses the preprocessing and chronological folds in preparar_lstm.py. The input is a
window of 60-station departure values; target-hour temperature, precipitation,
calendar, and holiday features are appended to the LSTM representation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preparar_lstm import (
    crear_folds,
    cargar_meteo,
    cargar_objetivos,
    crear_exogenas,
    desescalar_objetivos,
    dividir_fold,
    leer_tiempos_y_estaciones,
    validar_timeline,
)

RAIZ = Path(__file__).resolve().parents[3]
RESULTADOS = RAIZ / "results" / "lstm_selected_60"


class LSTMSelected60(nn.Module):
    def __init__(
        self,
        n_stations: int,
        n_exogenous: int,
        hidden_size: int = 64,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_stations,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_exogenous, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_stations),
        )

    def forward(self, past_targets: torch.Tensor, target_exogenous: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(past_targets)
        representation = hidden[-1]
        combined = torch.cat([representation, target_exogenous], dim=1)
        return self.head(combined)


def fijar_semilla(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metricas(real: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = pred - real
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((real - np.mean(real, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": float(r2)}


def entrenar_fold(
    fold_data: dict[str, np.ndarray],
    n_stations: int,
    n_exogenous: int,
    hidden_size: int,
    n_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[LSTMSelected60, dict[str, float], dict[str, list[float]]]:
    model = LSTMSelected60(n_stations, n_exogenous, hidden_size, n_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.L1Loss()

    train_dataset = TensorDataset(
        torch.from_numpy(fold_data["X_train"]),
        torch.from_numpy(fold_data["exog_train"]),
        torch.from_numpy(fold_data["y_train"]),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(fold_data["X_val"]),
        torch.from_numpy(fold_data["exog_val"]),
        torch.from_numpy(fold_data["y_val"]),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_val_loss = float("inf")
    stale_epochs = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for past_targets, target_exogenous, target in train_loader:
            past_targets = past_targets.to(device)
            target_exogenous = target_exogenous.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(past_targets, target_exogenous)
            loss = loss_function(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()) * len(target))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for past_targets, target_exogenous, target in val_loader:
                past_targets = past_targets.to(device)
                target_exogenous = target_exogenous.to(device)
                target = target.to(device)
                prediction = model(past_targets, target_exogenous)
                loss = loss_function(prediction, target)
                val_losses.append(float(loss.detach().cpu()) * len(target))

        train_loss = sum(train_losses) / len(train_dataset)
        val_loss = sum(val_losses) / len(val_dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, {"val_loss_scaled_mae": best_val_loss}, history


def predecir(
    model: LSTMSelected60,
    past_targets: np.ndarray,
    target_exogenous: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions = []
    dataset = TensorDataset(torch.from_numpy(past_targets), torch.from_numpy(target_exogenous))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for past_targets_batch, target_exogenous_batch in loader:
            prediction = model(past_targets_batch.to(device), target_exogenous_batch.to(device))
            predictions.append(prediction.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena un LSTM global para 60 estaciones BiciMAD")
    parser.add_argument("--ventanas", type=int, nargs="+", default=[24])
    parser.add_argument("--semanas", type=int, nargs="+", default=[52])
    parser.add_argument(
        "--ancla-folds",
        type=int,
        default=None,
        help="window used only to align fold boundaries across different window configurations",
    )
    parser.add_argument(
        "--meteorologia",
        nargs="+",
        choices=["con", "sin"],
        default=["con"],
        help="compara el uso de temperatura/precipitación; calendario se mantiene en ambas",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="fold indices to run; default is all folds")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validacion-h", type=int, default=168)
    parser.add_argument("--test-h", type=int, default=336)
    parser.add_argument("--resultados", default=str(RESULTADOS))
    args = parser.parse_args()

    resultados = Path(args.resultados)
    if not resultados.is_absolute():
        resultados = RAIZ / resultados
    resultados.mkdir(parents=True, exist_ok=True)

    fijar_semilla(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tiempos, estaciones, _ = leer_tiempos_y_estaciones()
    validar_timeline(tiempos)
    objetivos, festivo = cargar_objetivos(tiempos, estaciones)
    meteo = cargar_meteo(tiempos)
    exogenas = crear_exogenas(tiempos, meteo, festivo)

    ventanas_ancladas = [*args.ventanas]
    if args.ancla_folds is not None:
        ventanas_ancladas.append(args.ancla_folds)
    all_folds = crear_folds(len(tiempos), ventanas_ancladas, args.semanas, args.test_h)
    selected_folds = all_folds if args.folds is None else [fold for fold in all_folds if fold["fold"] in args.folds]
    if not selected_folds:
        raise SystemExit("No fold indices matched --folds")

    config_rows = []
    fold_rows = []
    histories = []

    for ventana in args.ventanas:
        for semanas in args.semanas:
            for meteorologia in args.meteorologia:
                config = f"lstm_{ventana}h_{semanas}sem_{meteorologia}_meteo"
                for fold in selected_folds:
                    fijar_semilla(args.seed + fold["fold"])
                    started = time.time()
                    fold_data = dividir_fold(
                        objetivos,
                        exogenas,
                        ventana,
                        fold["test_start"],
                        fold["test_end"],
                        semanas,
                        args.validacion_h,
                    )
                    if meteorologia == "sin":
                        for split in ("train", "val", "test"):
                            fold_data[f"exog_{split}"] = fold_data[f"exog_{split}"][:, 2:].copy()
                    model, training_metrics, history = entrenar_fold(
                        fold_data=fold_data,
                        n_stations=len(estaciones),
                        n_exogenous=fold_data["exog_train"].shape[1],
                        hidden_size=args.hidden,
                        n_layers=args.layers,
                        dropout=args.dropout,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        patience=args.patience,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        device=device,
                    )

                    scaled_prediction = predecir(
                        model,
                        fold_data["X_test"],
                        fold_data["exog_test"],
                        args.batch_size,
                        device,
                    )
                    prediction = np.clip(desescalar_objetivos(scaled_prediction, fold_data), 0.0, None)
                    real = objetivos[fold["test_start"]:fold["test_end"]]
                    metrics = metricas(real, prediction)

                    fold_rows.append({
                        "config": config,
                        "fold": fold["fold"],
                        "window": ventana,
                        "weeks": semanas,
                        "meteorologia": meteorologia,
                        "seconds": round(time.time() - started, 2),
                        **metrics,
                        **training_metrics,
                    })
                    histories.append({
                        "config": config,
                        "fold": fold["fold"],
                        "history": history,
                    })
                    print(
                        f"{config} fold={fold['fold']} MAE={metrics['mae']:.4f} "
                        f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}"
                    )

                    torch.save(model.state_dict(), resultados / f"{config}_fold_{fold['fold']}_model.pt")
                    np.savez_compressed(
                        resultados / f"{config}_fold_{fold['fold']}_prediction.npz",
                        timestamps=np.array(tiempos[fold["test_start"]:fold["test_end"]]),
                        station_ids=np.array(estaciones),
                        real=real,
                        prediction=prediction,
                        yesterday=objetivos[fold["test_start"] - 24:fold["test_end"] - 24],
                    )

    with (resultados / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)

    summary = []
    for config in dict.fromkeys(row["config"] for row in fold_rows):
        rows = [row for row in fold_rows if row["config"] == config]
        summary.append({
            "config": config,
            "mae": float(np.mean([row["mae"] for row in rows])),
            "rmse": float(np.mean([row["rmse"] for row in rows])),
            "r2": float(np.mean([row["r2"] for row in rows])),
            "n_folds": len(rows),
        })
    with (resultados / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    run_config = vars(args).copy()
    run_config["device"] = str(device)
    run_config["torch_version"] = torch.__version__
    run_config["station_ids"] = estaciones
    run_config["weather_source"] = "data/raw/meteo_horario.csv"
    (resultados / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (resultados / "learning_history.json").write_text(json.dumps(histories, indent=2, ensure_ascii=False), encoding="utf-8")

    for row in summary:
        print(f"{row['config']}: MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} R2={row['r2']:.4f}")


if __name__ == "__main__":
    main()
