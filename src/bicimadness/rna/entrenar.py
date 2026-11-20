"""RNA (perceptrón multicapa) para predecir las salidas de la hora siguiente en cada estación de BiciMAD.

Cada ejecución es un estudio: una configuración concreta (CSV, modo, variables, ventana...). Los resultados
se guardan en results/rna/<fecha>_<nombre>/ y se comparan con  python src/bicimadness/comparar_estudios.py rna

Uso (sin argumentos: sel60, red global sin id de estación, todas las variables, ventana 24 h, 52 semanas):
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

from sklearn.neural_network import MLPRegressor  # noqa: E402

from bicimadness.comun.experimento import argumentos, ejecutar, escalar_en_sitio  # noqa: E402

OCULTAS = {"global": [64, 32], "estacion": [32, 16]}  # la red global ve muchos más datos -> algo más grande
ITERACIONES = {"global": 200, "estacion": 500}


class RedEscalada:
    """La red con la estandarización de las entradas que se calculó al entrenar."""

    def __init__(self, media, desv, red):
        self.media, self.desv, self.red = media, desv, red

    def predict(self, X):
        return self.red.predict((X - self.media) / self.desv)


def entrenar(X, y, args):
    media, desv = escalar_en_sitio(X)  # X ya no se usa después: se estandariza sin copiarla
    red = MLPRegressor(hidden_layer_sizes=tuple(args.ocultas or OCULTAS[args.modo]), early_stopping=True,
                       max_iter=args.iteraciones or ITERACIONES[args.modo], random_state=args.semilla)
    return RedEscalada(media, desv, red.fit(X, y))


def main():
    p = argumentos(__doc__)
    p.add_argument("--ocultas", type=int, nargs="+",
                   help="neuronas de cada capa oculta (por defecto 64 32 en modo global y 32 16 por estación)")
    p.add_argument("--iteraciones", type=int, help="épocas máximas (por defecto 200 global, 500 por estación)")
    args = p.parse_args()
    args.ocultas = args.ocultas or OCULTAS[args.modo]  # se guarda en config.json el valor usado
    args.iteraciones = args.iteraciones or ITERACIONES[args.modo]
    ejecutar("rna", args, entrenar)


if __name__ == "__main__":
    main()
