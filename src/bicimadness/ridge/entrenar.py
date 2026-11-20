"""Ridge (regresión lineal con regularización) con las mismas entradas que la RNA.

Sirve de contraste: si Ridge iguala a la RNA, la no linealidad no aporta (idea de Juan, defeat_linearity.py).
La regularización se elige sola entre 0,01 y 1000 (RidgeCV: validación leave-one-out eficiente con la SVD de X;
"eigen" construiría una matriz filas x filas, imposible en modo global).

Uso (mismos argumentos que rna/entrenar.py):
  python src/bicimadness/ridge/entrenar.py sel60
  python src/bicimadness/ridge/entrenar.py sel60 --modo estacion
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, para importar bicimadness

import numpy as np  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402

from bicimadness.comun.experimento import argumentos, ejecutar, escalar_en_sitio  # noqa: E402

ALFAS = np.logspace(-2, 3, 6)


class RidgeEscalado:
    def __init__(self, media, desv, modelo):
        self.media, self.desv, self.modelo = media, desv, modelo

    def predict(self, X):
        return self.modelo.predict((X - self.media) / self.desv)


def entrenar(X, y, args):
    if args.modo == "estacion":
        X = X.astype(np.float64)  # pocas filas: en float32 el SVD de RidgeCV puede no converger
    media, desv = escalar_en_sitio(X)
    return RidgeEscalado(media, desv, RidgeCV(alphas=ALFAS, gcv_mode="svd").fit(X, y))


def main():
    args = argumentos(__doc__).parse_args()
    ejecutar("ridge", args, entrenar)


if __name__ == "__main__":
    main()
