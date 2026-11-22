"""Gradient boosting (árboles potenciados) para predecir las salidas de la hora siguiente en cada estación.

Mismas entradas, bloques y métricas que rna/entrenar.py (todo lo común está en comun/experimento.py):
solo cambia el modelo (HistGradientBoostingRegressor de scikit-learn). No necesita estandarizar las entradas.
Modo global sin id de estación por defecto; --con-id-estacion añade el one-hot.
Se entrena con todo menos la última semana (--validacion-h) y se elige el número de
árboles que da menor error en esa semana.

Uso:  python src/bicimadness/gradient_boosting/entrenar.py
      python src/bicimadness/gradient_boosting/entrenar.py sel60 --perdida poisson --nombre sel60_poisson
      python src/bicimadness/gradient_boosting/entrenar.py sel60 --iteraciones 500 --tasa 0.05
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, para importar bicimadness

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_poisson_deviance  # noqa: E402

from bicimadness.comun.experimento import argumentos, ejecutar, opciones_validacion, validacion_cronologica  # noqa: E402


class PrimerosArboles:
    """El modelo usando solo los n primeros árboles (los que dieron menor error en validación)."""

    def __init__(self, modelo, n):
        self.modelo, self.n = modelo, n

    def predict(self, X):
        for i, p in enumerate(self.modelo.staged_predict(X), 1):
            if i == self.n:
                return p


def _error(perdida, real, pred):
    if perdida == "absolute_error":
        return float(np.mean(np.abs(real - pred)))
    if perdida == "poisson":
        return float(mean_poisson_deviance(real, np.clip(pred, 1e-6, None)))
    return float(np.mean((real - pred) ** 2))


def entrenar(X, y, args):
    val = validacion_cronologica(len(y), args)
    parametros = dict(loss=args.perdida, max_iter=args.iteraciones, learning_rate=args.tasa,
                      max_depth=args.profundidad, random_state=args.semilla)
    Xa, ya, Xv, yv = X[~val], y[~val], X[val], y[val]
    modelo = HistGradientBoostingRegressor(early_stopping=False, **parametros).fit(Xa, ya)
    errores = [_error(args.perdida, yv, p) for p in modelo.staged_predict(Xv)]
    return PrimerosArboles(modelo, int(np.argmin(errores)) + 1)


def main():
    p = argumentos(__doc__)
    p.add_argument("--perdida", choices=["squared_error", "absolute_error", "poisson"], default="squared_error",
                   help="función de pérdida (por defecto squared_error, la misma que la RNA; poisson es la natural para conteos)")
    p.add_argument("--iteraciones", type=int, default=300, help="árboles máximos")
    p.add_argument("--tasa", type=float, default=0.1, help="learning rate")
    p.add_argument("--profundidad", type=int, default=None, help="profundidad máxima de cada árbol (por defecto sin límite)")
    opciones_validacion(p)
    args = p.parse_args()
    ejecutar("gradient_boosting", args, entrenar)


if __name__ == "__main__":
    main()
