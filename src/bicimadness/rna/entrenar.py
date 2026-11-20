"""RNA (perceptrón multicapa) para predecir las salidas de la hora siguiente en cada estación de BiciMAD.

Cada ejecución es un estudio: una configuración concreta (CSV, modo, variables, ventana...). Los resultados
se guardan en results/rna/<fecha>_<nombre>/ y se comparan con  python src/bicimadness/comparar_estudios.py rna

Uso (sin argumentos: sel60, red global, todas las variables, ventana 24 h, 52 semanas):
  python src/bicimadness/rna/entrenar.py
  python src/bicimadness/rna/entrenar.py top50
  python src/bicimadness/rna/entrenar.py sel60 --modo estacion
  python src/bicimadness/rna/entrenar.py sel60 --variables hora,dia_semana,mes,festivo --nombre sel60_calendario
  python src/bicimadness/rna/entrenar.py sel60 --meteo anterior --ventana 48 --ocultas 128 64

Para crear otro algoritmo (lstm/, gradient_boosting/...): copiar esta carpeta y cambiar entrenar() y sus
argumentos propios. Todo lo demás (bloques, tiempos, métricas, ficheros) lo hace comun/experimento.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, para importar bicimadness

import copy  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

from bicimadness.comun.experimento import argumentos, ejecutar, escalar_en_sitio, partir  # noqa: E402

OCULTAS = {"global": [64, 32], "estacion": [32, 16]}  # la red global ve muchos más datos -> algo más grande
ITERACIONES = {"global": 200, "estacion": 500}
PACIENCIA = 10  # épocas seguidas sin mejorar en validación


class RedEscalada:
    """La red con la estandarización de las entradas que se calculó al entrenar."""

    def __init__(self, media, desv, red):
        self.media, self.desv, self.red = media, desv, red

    def predict(self, X):
        return self.red.predict((X - self.media) / self.desv)


def entrenar(X, y, args, val):
    media, desv = escalar_en_sitio(X)  # X ya no se usa después: se estandariza sin copiarla
    capas, max_iter = tuple(args.ocultas or OCULTAS[args.modo]), args.iteraciones or ITERACIONES[args.modo]
    if args.validacion == "aleatoria":  # forma anterior: 10 % de filas al azar (las ventanas solapadas se cuelan en la validación)
        red = MLPRegressor(hidden_layer_sizes=capas, early_stopping=True, max_iter=max_iter, random_state=args.semilla)
        return RedEscalada(media, desv, red.fit(X, y))
    # cronológica: se entrena con todo menos la última semana, y se para cuando el error en esa semana deja de mejorar
    Xa, ya, Xv, yv = partir(X, y, val)
    red = MLPRegressor(hidden_layer_sizes=capas, max_iter=1, random_state=args.semilla)
    mejor, mejor_estado, sin_mejora = np.inf, None, 0
    for _ in range(max_iter):
        red.partial_fit(Xa, ya)  # una época
        error = float(np.mean((red.predict(Xv) - yv) ** 2))
        if error < mejor - 1e-4:
            mejor, sin_mejora = error, 0
            mejor_estado = copy.deepcopy((red.coefs_, red.intercepts_))
        else:
            sin_mejora += 1
            if sin_mejora == PACIENCIA:
                break
    red.coefs_, red.intercepts_ = mejor_estado
    return RedEscalada(media, desv, red)


def main():
    p = argumentos(__doc__)
    p.add_argument("--ocultas", type=int, nargs="+",
                   help="neuronas de cada capa oculta (por defecto 64 32 en modo global y 32 16 por estación)")
    p.add_argument("--iteraciones", type=int, help="épocas máximas (por defecto 200 global, 500 por estación)")
    p.add_argument("--validacion", choices=["cronologica", "aleatoria"], default="cronologica",
                   help="validación para parar el entrenamiento: última semana del entrenamiento (por defecto) o 10 %% al azar")
    args = p.parse_args()
    args.ocultas = args.ocultas or OCULTAS[args.modo]  # se guarda en config.json el valor usado
    args.iteraciones = args.iteraciones or ITERACIONES[args.modo]
    ejecutar("rna", args, entrenar)


if __name__ == "__main__":
    main()
