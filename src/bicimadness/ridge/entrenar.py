"""Regresión lineal regularizada (Ridge) para predecir las salidas de la hora siguiente en cada estación.

Mismas entradas, bloques, validación y métricas que rna/entrenar.py (todo lo común está en comun/experimento.py):
solo cambia el modelo. El parámetro de regularización alpha se elige en cada bloque entre --alphas con el error
en la última semana del entrenamiento, y el modelo final se ajusta con el resto (igual que los demás algoritmos).

Uso:  python src/bicimadness/ridge/entrenar.py
      python src/bicimadness/ridge/entrenar.py sel60 --meteo anterior --nombre sel60_meteo_anterior
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, para importar bicimadness

import numpy as np  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from bicimadness.comun.experimento import ModeloEscalado, argumentos, ejecutar, escalar_en_sitio, partir  # noqa: E402

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def entrenar(X, y, args, val):
    media, desv = escalar_en_sitio(X)
    Xa, ya, Xv, yv = partir(X, y, val)
    modelos = [Ridge(alpha=a).fit(Xa, ya) for a in args.alphas]
    errores = [float(np.mean((m.predict(Xv) - yv) ** 2)) for m in modelos]
    return ModeloEscalado(media, desv, modelos[int(np.argmin(errores))])


def main():
    p = argumentos(__doc__)
    p.add_argument("--alphas", type=float, nargs="+", default=ALPHAS, help=f"valores de regularización a probar (por defecto {ALPHAS})")
    ejecutar("ridge", p.parse_args(), entrenar)


if __name__ == "__main__":
    main()
