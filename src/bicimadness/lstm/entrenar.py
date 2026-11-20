"""LSTM para predecir las salidas de la hora siguiente en cada estación de BiciMAD (modelo de Manu).

Por defecto es multivariante, como el LSTM de Manu (antiguo/lstm_manu/modelo_lstm_60.py): en cada una de las
N horas anteriores entra un vector con las salidas de TODAS las estaciones; el último estado de la LSTM se
junta con las variables de la hora a predecir (calendario y meteorología) y una cabeza densa
(oculta -> ReLU -> dropout -> una salida por estación) da todas las predicciones a la vez. La estación no
entra en one-hot: cada una tiene su posición en la entrada y su neurona de salida. Así la red puede usar lo
que pasa en las demás estaciones, pero solo sirve para el conjunto de estaciones con el que se entrena.

Con --modo global o --modo estacion cada fila es una sola estación (secuencia de su propio historial), como
la RNA y Ridge: global = un modelo compartido que no sabe qué estación es (vale para cualquiera; con
--con-id-estacion se le da en one-hot); estacion = un modelo por estación.

Ajustes de Manu: estado de 64, 1 capa, dropout 0,1, AdamW (lr 1e-3, weight decay 1e-5), pérdida L1 (MAE),
lotes de 256, hasta 30 épocas con paciencia 5, recorte del gradiente a 1, validación con la última semana
del entrenamiento (cronológica) y, en multivariante, normalización por estación.
Mismos bloques, métricas y ficheros que el resto (comun/experimento.py):
  python src/bicimadness/comparar_algoritmos.py

Uso (por defecto: sel60, multivariante, todas las variables, ventana 24 h, 52 semanas):
  python src/bicimadness/lstm/entrenar.py
  python src/bicimadness/lstm/entrenar.py sel60 --modo global
  python src/bicimadness/lstm/entrenar.py sel60 --modo estacion
  python src/bicimadness/lstm/entrenar.py sel60 --oculta 32 --epocas 10 --hasta 2022-03-01
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, para importar bicimadness

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from bicimadness.comun.experimento import argumentos, ejecutar  # noqa: E402

# En la GPU del Mac (mps) la LSTM se queda sin memoria con 8 GB: CPU por defecto (LSTM_DISPOSITIVO=mps para probar)
DISPOSITIVO = os.environ.get("LSTM_DISPOSITIVO", "cpu")
LOTE_PREDICCION = 8192


class Red(nn.Module):
    def __init__(self, n_canales, n_estaticas, n_salidas, oculta, capas, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_canales, oculta, num_layers=capas, batch_first=True,
                            dropout=dropout if capas > 1 else 0.0)
        self.salida = nn.Sequential(nn.Linear(oculta + n_estaticas, oculta), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(oculta, n_salidas))

    def forward(self, seq, estaticas):
        _, (h, _) = self.lstm(seq)
        return self.salida(torch.cat([h[-1], estaticas], dim=1))


class LSTMEscalada:
    """La red con la normalización calculada al entrenar.

    X: [historial (n horas x c canales, hora a hora), variables]; c = 1 (una estación) o nº de estaciones."""

    def __init__(self, red, n, c, media_y, desv_y, media_e, desv_e):
        self.red, self.n, self.c = red, n, c
        self.media_y, self.desv_y, self.media_e, self.desv_e = media_y, desv_y, media_e, desv_e

    def tensores(self, X):
        nc = self.n * self.c
        seq = (X[:, :nc].reshape(len(X), self.n, self.c) - self.media_y) / self.desv_y
        est = (X[:, nc:] - self.media_e) / self.desv_e
        return (torch.from_numpy(np.ascontiguousarray(seq, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(est, dtype=np.float32)))

    def predict(self, X):
        seq, est = self.tensores(X)
        self.red.eval()
        with torch.no_grad():
            p = [self.red(seq[i:i + LOTE_PREDICCION].to(DISPOSITIVO), est[i:i + LOTE_PREDICCION].to(DISPOSITIVO)).cpu()
                 for i in range(0, len(seq), LOTE_PREDICCION)]
        p = torch.cat(p).numpy() * self.desv_y + self.media_y
        return p if self.c > 1 else p[:, 0]


def _validacion(n_filas, args):
    """Máscara de las últimas args.validacion_h horas del entrenamiento (de cada estación en modo global).

    Las filas van en orden de tiempo; en modo global, apiladas por estación (semanas*168 cada una)."""
    n_tr = args.semanas * 7 * 24
    return (np.arange(n_filas) % n_tr) >= n_tr - args.validacion_h


def entrenar(X, y, args):
    torch.manual_seed(args.semilla)
    if args.modo == "estacion":
        torch.set_num_threads(1)  # muchos modelos pequeños en paralelo (joblib): un hilo cada uno
    y = y.reshape(len(y), -1)  # (filas x estaciones); 1 columna salvo en multivariante
    n, c = args.ventana, y.shape[1]
    val = _validacion(len(y), args)
    ent = ~val
    # escalado solo con la parte de entrenamiento; historial y objetivo con la misma escala (por estación)
    media_y, desv_y = y[ent].mean(axis=0), y[ent].std(axis=0)
    desv_y[desv_y < 1e-8] = 1
    media_e, desv_e = X[ent, n * c:].mean(axis=0), X[ent, n * c:].std(axis=0)
    desv_e[desv_e < 1e-8] = 1
    red = Red(c, X.shape[1] - n * c, c, args.oculta, args.capas, args.dropout).to(DISPOSITIVO)
    modelo = LSTMEscalada(red, n, c, media_y, desv_y, media_e, desv_e)
    seq, est = (t.to(DISPOSITIVO) for t in modelo.tensores(X))
    obj = torch.from_numpy(((y - media_y) / desv_y).astype(np.float32)).to(DISPOSITIVO)
    i_val = torch.from_numpy(np.flatnonzero(val)).to(DISPOSITIVO)
    i_ent = torch.from_numpy(np.flatnonzero(ent)).to(DISPOSITIVO)
    gen = torch.Generator().manual_seed(args.semilla)
    opt = torch.optim.AdamW(red.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    perdida_l1 = nn.L1Loss()
    inicio = time.perf_counter()

    mejor, estado, sin_mejora = float("inf"), None, 0
    for epoca in range(1, args.epocas + 1):
        red.train()
        for lote in i_ent[torch.randperm(len(i_ent), generator=gen).to(DISPOSITIVO)].split(args.lote):
            opt.zero_grad(set_to_none=True)
            perdida = perdida_l1(red(seq[lote], est[lote]), obj[lote])
            perdida.backward()
            nn.utils.clip_grad_norm_(red.parameters(), max_norm=1.0)
            opt.step()
        red.eval()
        with torch.no_grad():
            pv = torch.cat([red(seq[l], est[l]) for l in i_val.split(LOTE_PREDICCION)])
        error = float((pv - obj[i_val]).abs().mean())
        if error < mejor:
            mejor, sin_mejora = error, 0
            estado = {k: v.detach().clone() for k, v in red.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                break
    red.load_state_dict(estado)
    if args.modo != "estacion":
        print(f"  bloque entrenado: {epoca} épocas, {time.perf_counter() - inicio:.0f} s", file=sys.stderr, flush=True)
    return modelo


def main():
    p = argumentos(__doc__, multivariante=True)
    p.add_argument("--oculta", type=int, default=64, help="tamaño del estado de la LSTM y de la capa densa (64)")
    p.add_argument("--capas", type=int, default=1, help="capas de LSTM apiladas (1)")
    p.add_argument("--dropout", type=float, default=0.1, help="dropout de la cabeza (y entre capas LSTM si hay varias)")
    p.add_argument("--epocas", type=int, default=30, help="épocas máximas (30)")
    p.add_argument("--paciencia", type=int, default=5, help="para si estas épocas seguidas no mejoran la validación (5)")
    p.add_argument("--lote", type=int, default=256, help="ejemplos por paso de entrenamiento (256)")
    p.add_argument("--lr", type=float, default=1e-3, help="tasa de aprendizaje de AdamW")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay de AdamW")
    p.add_argument("--validacion-h", type=int, default=168,
                   help="últimas horas del entrenamiento que se usan para parar (168 = 1 semana)")
    args = p.parse_args()
    if not 1 <= args.validacion_h < args.semanas * 7 * 24:
        raise SystemExit("--validacion-h debe ser menor que las horas de entrenamiento")
    if args.jobs is None and args.modo == "global":
        args.jobs = 1  # un bloque cada vez, usando todos los núcleos
    ejecutar("lstm", args, entrenar)


if __name__ == "__main__":
    main()
