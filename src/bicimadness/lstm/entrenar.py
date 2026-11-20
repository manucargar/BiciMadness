"""LSTM global para predecir las salidas de la hora siguiente en cada estación (formato común de los algoritmos).

Mismas entradas, bloques, validación y métricas que rna/entrenar.py (todo lo común está en comun/experimento.py).
La LSTM lee las --ventana salidas anteriores de la estación como secuencia; el resto de columnas (calendario,
meteorología y estación en one-hot) se unen a su último estado oculto y pasan por dos capas densas (64 y 32).
Pérdida MSE, Adam, parada temprana con la última semana del entrenamiento (como la RNA, el boosting y Ridge).

No confundir con los scripts anteriores de esta carpeta (modelo_lstm_60.py...): son otro experimento (la LSTM
recibe las 60 estaciones a la vez) y no usan los mismos bloques ni métricas.

Uso:  python src/bicimadness/lstm/entrenar.py
      python src/bicimadness/lstm/entrenar.py sel60 --ocultas 64 --iteraciones 30 --nombre sel60_64
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, para importar bicimadness

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from bicimadness.comun.experimento import argumentos, ejecutar, escalar_en_sitio  # noqa: E402

LOTE = 1024


class Red(nn.Module):
    def __init__(self, n_fijas, ocultas):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=ocultas, batch_first=True)
        self.cabeza = nn.Sequential(nn.Linear(ocultas + n_fijas, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, X, v):
        _, (h, _) = self.lstm(X[:, :v].unsqueeze(-1))  # último estado oculto: resumen de las v horas
        return self.cabeza(torch.cat([h[-1], X[:, v:]], dim=1)).squeeze(1)


class LSTMEscalada:
    def __init__(self, media, desv, red, v):
        self.media, self.desv, self.red, self.v = media, desv, red, v

    def predict(self, X):
        self.red.eval()
        X = torch.from_numpy(((X - self.media) / self.desv).astype(np.float32))
        with torch.no_grad():
            return torch.cat([self.red(X[i:i + 8192], self.v) for i in range(0, len(X), 8192)]).numpy()


def _error(red, X, y, v, idx):
    red.eval()
    with torch.no_grad():
        p = torch.cat([red(X[idx[i:i + 8192]], v) for i in range(0, len(idx), 8192)])
    return float(((p - y[idx]) ** 2).mean())


def entrenar(X, y, args, val):
    torch.manual_seed(args.semilla)
    torch.set_num_threads(args.hilos or max(1, (os.cpu_count() or 2) // (args.jobs or 2)))
    media, desv = escalar_en_sitio(X)  # X queda estandarizada en sitio
    Xt, yt, v = torch.from_numpy(X), torch.from_numpy(np.asarray(y, dtype=np.float32)), args.ventana
    tr, va = torch.from_numpy(np.flatnonzero(~val)), torch.from_numpy(np.flatnonzero(val))
    red = Red(X.shape[1] - v, args.ocultas)
    opt = torch.optim.Adam(red.parameters(), lr=args.tasa)
    mejor, mejor_estado, sin_mejora = float("inf"), None, 0
    for _ in range(args.iteraciones):
        red.train()
        orden = tr[torch.randperm(len(tr))]
        for i in range(0, len(orden), LOTE):
            b = orden[i:i + LOTE]
            opt.zero_grad()
            nn.functional.mse_loss(red(Xt[b], v), yt[b]).backward()
            opt.step()
        error = _error(red, Xt, yt, v, va)
        if error < mejor - 1e-4:
            mejor, sin_mejora = error, 0
            mejor_estado = {k: t.clone() for k, t in red.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora == args.paciencia:
                break
    red.load_state_dict(mejor_estado)
    return LSTMEscalada(media, desv, red, v)


def main():
    p = argumentos(__doc__)
    p.add_argument("--ocultas", type=int, default=32, help="unidades de la capa LSTM (por defecto 32)")
    p.add_argument("--iteraciones", type=int, default=20, help="épocas máximas (por defecto 20)")
    p.add_argument("--paciencia", type=int, default=3, help="épocas seguidas sin mejorar en validación antes de parar (por defecto 3)")
    p.add_argument("--tasa", type=float, default=1e-3, help="learning rate de Adam")
    p.add_argument("--hilos", type=int, help="hilos de PyTorch por bloque (por defecto núcleos / bloques en paralelo)")
    ejecutar("lstm", p.parse_args(), entrenar)


if __name__ == "__main__":
    main()
