"""Pruebas de separación temporal y escalado sin datos de validación."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bicimadness.comun import datos as D, experimento as E
from bicimadness.rna.entrenar import entrenar as rna
from bicimadness.lstm.entrenar import entrenar as lstm
from bicimadness.gradient_boosting.entrenar import entrenar as boosting, _error


class ValidacionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args = SimpleNamespace(semanas=2, validacion_h=168, modo='global', ventana=2,
                                    semilla=0, ocultas=[4], iteraciones=2, oculta=4, capas=1,
                                    dropout=0., lr=.001, weight_decay=1e-5, epocas=1,
                                    lote=128, paciencia=2, con_id_estacion=False,
                                    perdida='squared_error', tasa=.1, profundidad=None)

    def test_folds_and_modes(self):
        series = {'a': np.arange(18676, dtype=np.float32),
                  'b': np.arange(18676, dtype=np.float32) + 100000}
        folds = D.bloques(18676)
        self.assertEqual(len(folds), 29)
        self.assertTrue(all(b-a == D.TEST_H for (_, a), (_, b) in zip(folds, folds[1:])))
        for mode in ('global', 'estacion', 'multivariante'):
            self.args.modo = mode
            for k, s in folds:
                def fit(X, y, args):
                    val = E.validacion_cronologica(len(y), args)
                    times = y % 100000
                    self.assertLess(times[~val].max(), times[val].min())
                    self.assertEqual(times[val].max(), s-1)
                    self.assertEqual(times[~val].min(), s-336)
                    c = y.shape[1] if y.ndim == 2 else 1
                    np.testing.assert_array_equal(X[:, c:2*c].reshape(y.shape), y-1)
                    class Model:
                        def predict(self, Xt):
                            p = Xt[:, c:2*c]+1
                            return p if y.ndim == 2 else p[:, 0]
                    return Model()
                _, pred, _, _ = E._bloque(k, s, series, np.zeros((18676, 0)), self.args, fit)
                for e in series:
                    np.testing.assert_array_equal(pred[e], series[e][s:s+D.TEST_H])

    def test_invalid_validation(self):
        for weeks, hours in ((2, 0), (2, 336), (2, 337), (1, 168)):
            self.args.semanas, self.args.validacion_h = weeks, hours
            with self.assertRaises(ValueError):
                E.validacion_cronologica(weeks*168, self.args)

    def test_scaling_and_ridge_compatibility(self):
        X = np.array([[1, 4], [3, 4], [1000, 9000]], dtype=np.float32)
        scaled = X.copy()
        mean, std = E.escalar_en_sitio(scaled, trozo=1, entrenamiento=np.array([True, True, False]))
        np.testing.assert_allclose(mean, [2, 4])
        np.testing.assert_allclose(std, [1, 1])
        np.testing.assert_allclose(scaled, (X-mean)/std)
        mean, std = E.escalar_en_sitio(X.copy(), trozo=1)
        np.testing.assert_allclose(mean, X.mean(0))
        np.testing.assert_allclose(std, X.std(0), rtol=1e-6)

    def test_neural_scaling_excludes_validation(self):
        rng = np.random.default_rng(0)
        for mode, rows, c in [('global', 672, 1), ('estacion', 336, 1), ('multivariante', 336, 2)]:
            self.args.modo = mode
            X = rng.normal(size=(rows, 2*c+1)).astype(np.float32)
            y = rng.normal(size=(rows, c)).astype(np.float32)
            val = E.validacion_cronologica(rows, self.args)
            X[val] += 100
            y[val] += 100
            model = lstm(X, y if c > 1 else y[:, 0], self.args)
            for actual, expected in [(model.media_y, y[~val].mean(0)), (model.desv_y, y[~val].std(0)),
                                     (model.media_e, X[~val, 2*c:].mean(0)), (model.desv_e, X[~val, 2*c:].std(0))]:
                np.testing.assert_allclose(actual, expected, atol=1e-6)
            self.assertTrue(np.isfinite(model.predict(X[:4])).all())
            if c == 1:
                model = rna(X.copy(), y[:, 0], self.args)
                np.testing.assert_allclose(model.media, X[~val].mean(0), atol=1e-6)
                np.testing.assert_allclose(model.desv, X[~val].std(0), atol=1e-6)
                self.assertTrue(np.isfinite(model.predict(X[:4])).all())

    def test_boosting_validation_selection(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(672, 3)).astype(np.float32)
        y = rng.poisson(2, size=len(X)).astype(np.float32)
        val = E.validacion_cronologica(len(X), self.args)
        for loss in ('squared_error', 'absolute_error', 'poisson'):
            self.args.perdida = loss
            model = boosting(X, y, self.args)
            errors = [_error(loss, y[val], p) for p in model.modelo.staged_predict(X[val])]
            self.assertEqual(model.n, np.argmin(errors)+1)
            self.assertTrue(np.isfinite(model.predict(X[:4])).all())


if __name__ == '__main__':
    unittest.main()
