"""RNA (perceptrón multicapa) para predecir las salidas de la hora siguiente en cada estación de BiciMAD.

Cada ejecución es un estudio: una configuración concreta (CSV, modo, variables, ventana...). Los resultados
se guardan en results/rna/<fecha>_<nombre>/ y se comparan con  python src/bicimadness/comparar_estudios.py rna

Uso (sin argumentos: sel60, red global sin id de estación, todas las variables, ventana 24 h, 52 semanas):
  python src/bicimadness/rna/entrenar.py
  python src/bicimadness/rna/entrenar.py top50
  python src/bicimadness/rna/entrenar.py sel60 --modo estacion
  python src/bicimadness/rna/entrenar.py sel60 --variables hora,dia_semana,mes,festivo --nombre sel60_calendario
  python src/bicimadness/rna/entrenar.py sel60 --meteo anterior --ventana 48 --ocultas 128 64
  .venv/bin/python src/bicimadness/rna/entrenar.py sel60_filtrada --perdida poisson --meteo anterior

Para crear otro algoritmo (lstm/, gradient_boosting/...): copiar esta carpeta y cambiar entrenar() y sus
argumentos propios. Todo lo demás (bloques, tiempos, métricas, ficheros) lo hace comun/experimento.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, para importar bicimadness

import copy
import inspect

import numpy as np
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.metrics import mean_poisson_deviance  # noqa: E402

from bicimadness.comun.experimento import (
    argumentos, ejecutar, escalar_en_sitio, opciones_validacion, validacion_cronologica,
)  # noqa: E402

OCULTAS = {"global": [64, 32], "estacion": [32, 16]}  # la red global ve muchos más datos -> algo más grande
ITERACIONES = {"global": 200, "estacion": 500}


class RedEscalada:
    """La red con la estandarización de las entradas que se calculó al entrenar."""

    def __init__(self, media, desv, red):
        self.media, self.desv, self.red = media, desv, red

    def predict(self, X):
        return self.red.predict((X - self.media) / self.desv)


def entrenar(X, y, args):
    val = validacion_cronologica(len(y), args)
    media, desv = escalar_en_sitio(X, entrenamiento=~val)
    opciones = dict(hidden_layer_sizes=tuple(args.ocultas or OCULTAS[args.modo]),
                    max_iter=1, random_state=args.semilla)
    if args.perdida == "poisson":
        opciones["loss"] = "poisson"
    red = MLPRegressor(**opciones)
    Xa, ya, Xv, yv = X[~val], y[~val], X[val], y[val]
    mejor, estado, sin_mejora = np.inf, None, 0
    for _ in range(args.iteraciones or ITERACIONES[args.modo]):
        red.partial_fit(Xa, ya)
        pred_val = red.predict(Xv)
        error = (float(mean_poisson_deviance(yv, np.maximum(pred_val, 1e-12)))
                 if args.perdida == "poisson" else float(np.mean((pred_val - yv) ** 2)))
        if error < mejor - 1e-4:
            mejor, sin_mejora = error, 0
            estado = copy.deepcopy((red.coefs_, red.intercepts_))
        else:
            sin_mejora += 1
            if sin_mejora >= 10:
                break
    red.coefs_, red.intercepts_ = estado
    return RedEscalada(media, desv, red)


def main():
    p = argumentos(__doc__)
    p.add_argument("--ocultas", type=int, nargs="+",
                   help="neuronas de cada capa oculta (por defecto 64 32 en modo global y 32 16 por estación)")
    p.add_argument("--iteraciones", type=int, help="épocas máximas (por defecto 200 global, 500 por estación)")
    p.add_argument("--perdida", choices=["squared_error", "poisson"], default="squared_error",
                   help="pérdida de entrenamiento y validación (por defecto squared_error; poisson requiere scikit-learn >= 1.7)")
    opciones_validacion(p)
    args = p.parse_args()
    if args.perdida == "poisson" and "loss" not in inspect.signature(MLPRegressor).parameters:
        p.error("Poisson requiere scikit-learn >= 1.7; ejecuta el script con .venv/bin/python")
    if args.iteraciones is not None and args.iteraciones < 1:
        p.error("--iteraciones debe ser al menos 1")
    args.ocultas = args.ocultas or OCULTAS[args.modo]  # se guarda en config.json el valor usado
    args.iteraciones = args.iteraciones or ITERACIONES[args.modo]
    ejecutar("rna", args, entrenar)


if __name__ == "__main__":
    main()
