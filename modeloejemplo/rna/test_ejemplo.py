import contextlib
import io
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ejemplo import indices, predecir, siguiente_mes, wape, redondear


class UltimaObservacion:
    def predict(self, X):
        return X[:, 23]


class EjemploTest(unittest.TestCase):
    def test_wape_y_ceros(self):
        self.assertAlmostEqual(wape([2, 4], [1, 5]), 100 / 3)
        self.assertEqual(wape([0, 2, 4], [10, 1, 5]), 200)
        self.assertEqual(wape([10, 0, 10], [8, 3, 15]), 50)
        self.assertEqual(wape([0, 2], [0, 2]), 0)
        self.assertIsNone(wape([0, 0], [0, 2]))
        self.assertIsNone(wape([0, 0], [0, 0]))

    def test_redondeo(self):
        self.assertEqual(redondear(np.array([-1, 0, 0.49, 0.5, 2.5, 2.9])).tolist(), [0, 0, 0, 1, 3, 3])

    def setUp(self):
        self.fechas = [datetime(2022, 1, 1) + timedelta(hours=h) for h in range(80)]

    def test_limites_historial_y_periodo(self):
        self.assertEqual(indices(self.fechas, self.fechas[24], self.fechas[25]).tolist(), [24])
        for inicio, fin in [(self.fechas[23], self.fechas[25]),
                            (self.fechas[24], self.fechas[-1] + timedelta(hours=2)),
                            (self.fechas[24] + timedelta(minutes=1), self.fechas[25])]:
            with self.assertRaises(ValueError):
                indices(self.fechas, inicio, fin)

    def test_mes_calendario(self):
        self.assertEqual(siguiente_mes(datetime(2024, 1, 31, 18)), datetime(2024, 2, 29, 18))
        self.assertEqual(siguiente_mes(datetime(2022, 12, 15)), datetime(2023, 1, 15))

    def test_historial_real_actualizado_sin_hora_objetivo(self):
        datos = ({"43": np.arange(80, dtype=np.float32)}, self.fechas,
                 np.zeros((80, 0), dtype=np.float32))
        artefacto = {"modelo": UltimaObservacion(), "estaciones": ["43"],
                    "fin_entrenamiento_exclusivo": self.fechas[24].isoformat()}
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            resumen = predecir(artefacto, datos, self.fechas[24], self.fechas[48], Path(d))
            # Si se reutilizara la predicción, la diferencia crecería; si se filtrara
            # la hora objetivo, sería cero. Con el historial real siempre vale uno.
            self.assertEqual(resumen["mae"], 1.0)
            self.assertEqual(resumen["mae_referencia"], 24.0)
            with self.assertRaises(ValueError):
                predecir(artefacto, datos, self.fechas[23], self.fechas[25], Path(d))


if __name__ == "__main__":
    unittest.main()
