# BiciMadness

**Idioma / Language:** [Español](#español) · [English](#english)

## Español

BiciMadness predice **cuántas bicicletas saldrán de una estación de BiciMAD durante la hora siguiente**. Es un proyecto de investigación en Python para *Inteligencia Artificial en las Organizaciones* (UC3M, 2026/27), orientado a apoyar la planificación de la redistribución de bicicletas en Madrid.

Transforma viajes históricos en recuentos horarios, entrena varios algoritmos (red neuronal, Ridge, gradient boosting, LSTM) y los evalúa **en las mismas condiciones**: mismas entradas, mismos bloques de test, misma validación interna y mismas métricas. El experimento principal usa **60 estaciones elegidas para representar distintos niveles de demanda, ubicaciones y dificultad de predicción**; un experimento anterior usa las 50 con más salidas.

El repositorio contiene experimentos por línea de comandos e informes CSV. No incluye aplicación web, API de predicción en directo, predicción a 24 horas, optimizador de redistribución ni modelos guardados.

### Índice

[Inicio rápido](#inicio-rápido) · [Qué predice el modelo](#qué-predice-el-modelo) · [Estructura](#estructura-del-proyecto) · [Datos](#datos-y-preprocesado) · [Selección de estaciones](#selección-de-las-60-estaciones) · [Algoritmos](#algoritmos-y-referencias) · [Evaluación](#cómo-funciona-la-evaluación) · [Ejecutar](#ejecutar-experimentos) · [Resultados](#interpretar-los-resultados) · [Regenerar datos](#regenerar-los-datos) · [Problemas](#solución-de-problemas) · [Limitaciones](#limitaciones-y-trabajo-futuro)

### Inicio rápido

Ejecuta los comandos desde la raíz del proyecto. Requiere **Python 3.10 o superior**.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt`.

Los datos preparados ya están en el repositorio (`data/processed/salidas_por_hora_sel60.csv`, `estaciones_seleccionadas.csv`, `salidas_por_hora_top50.csv`, `estaciones_top50.csv`, `data/raw/meteo_horario.csv` y `calendario_laboral.csv`), así que no hace falta descargar los viajes originales. Para entrenar la red neuronal con la configuración por defecto:

```powershell
python src/bicimadness/rna/entrenar.py
```

Cada ejecución crea su carpeta `results/<algoritmo>/<fecha>_<nombre>/` (nunca se sobrescribe una anterior). Abre `resumen.csv` para las métricas globales y `errores_por_grupo.csv` para el análisis por grupo de estaciones. Un algoritmo tarda de ~30 s por bloque (RNA, Ridge, boosting) a ~3 min por bloque (LSTM); los 29 bloques completos van de decenas de minutos a ~1,5 h. No hay interfaz gráfica.

### Qué predice el modelo

Una **salida** es un viaje que empieza en una estación. Para cada estación y hora, el objetivo es el número de salidas en esa hora. Para predecir las 09:00–09:59 el modelo recibe:

- Las salidas observadas de esa estación desde las 09:00 de ayer hasta las 08:00 de hoy (**ventana**, 24 h por defecto).
- Calendario de la hora objetivo: hora, día de la semana, mes y festivo.
- Temperatura y precipitación (ver `--meteo` más abajo).
- La estación en one-hot, si el modelo es global.

```text
entradas: salidas[t-24], ..., salidas[t-1], variables[t]
objetivo: salidas[t]
```

La salida es un número no negativo (las predicciones negativas se sustituyen por 0; no se redondean). La ventana de 24 h es distinta del **horizonte**: el sistema predice **una hora hacia delante**; no predice las próximas 24 horas desde un instante.

### Estructura del proyecto

```text
.
├── README.md
├── requirements.txt
├── src/bicimadness/
│   ├── preprocesamiento/        # viajes → datos horarios; selección de estaciones; meteorología
│   ├── comun/
│   │   ├── datos.py             # carga, variables, ventana y bloques de test (compartido)
│   │   ├── experimento.py       # ejecuta un estudio y guarda resultados (compartido)
│   │   └── comparar.py          # lectura de ejecuciones y t-test pareado
│   ├── rna/entrenar.py          # red neuronal (MLP)
│   ├── ridge/entrenar.py        # regresión lineal regularizada
│   ├── gradient_boosting/entrenar.py
│   ├── lstm/entrenar.py         # LSTM (PyTorch)
│   ├── comparar_estudios.py     # compara los estudios de un algoritmo
│   ├── comparar_algoritmos.py   # compara el mejor estudio de cada algoritmo
│   └── antiguo/, defeat_linearity.py, lstm/{preparar,modelo,analizar}_lstm.py, lstm/rna_estaciones_sel60_adaptada.py
│                                # código anterior, conservado de momento; usa otros bloques y métricas, no comparar con lo nuevo
├── data/{raw,processed,cache}/
├── results/                     # salidas generadas (ignoradas por git)
└── docs/                        # propuesta, seguimiento e índice de la memoria
```

**Cómo se añade un algoritmo:** cada uno solo define `entrenar(X, y, args, val) -> modelo con .predict(X)` y llama a `ejecutar()`. Todo lo demás (bloques, escalado, tiempos, métricas, ficheros) vive en `comun/experimento.py`, de modo que todos se evalúan y se guardan igual. Copia una carpeta existente y cambia `entrenar()` y sus opciones propias.

### Datos y preprocesado

- **Viajes:** [Datos abiertos de Madrid — BiciMAD](https://datos.madrid.es/dataset/900034-0-bicimad-viajes-estaciones). Los datos preparados abarcan de enero de 2021 al 18 de febrero de 2023.
- **Festivos:** calendario laboral de Madrid (`data/raw/calendario_laboral.csv`).
- **Meteorología:** [Meteostat](https://meteostat.net), estación Madrid-Retiro 08222 (observaciones de AEMET, licencia CC BY-NC 4.0: cita la fuente).

Pasos: leer CSV y JSON antiguos → descartar viajes CSV de menos de 120 s (desbloqueos fallidos) → pasar las fechas UTC de los JSON a hora local → contar salidas por estación y hora, rellenando con 0 las horas vacías → añadir día de la semana, mes y festivo.

Formato de los CSV horarios:

| Columna | Significado |
|---|---|
| `fecha_hora` | Inicio de la hora, hora local de Madrid (`YYYY-MM-DD HH:00`) |
| `id_estacion` | Identificador de estación |
| `dia_semana` | ISO: 1 = lunes … 7 = domingo |
| `mes` | 1–12 |
| `festivo` | 1 si es festivo |
| `salidas` | Salidas en esa hora (objetivo) |

Los modelos calculan hora, día y mes a partir de `fecha_hora` y leen `festivo` del CSV. La meteorología se cruza por fecha con `meteo_horario.csv` (`temperatura` en ºC, `precipitacion` en mm). Todas las estaciones deben tener la misma serie horaria completa y ordenada. El script meteorológico rellena huecos con el valor anterior y, al cambiar la hora en otoño, promedia temperatura y suma lluvia.

### Selección de las 60 estaciones

Seis grupos de diez estaciones, sin repetir y rellenados en este orden:

| Grupo | Criterio |
|---|---|
| `demanda_alta` | Más salidas totales |
| `demanda_baja` | Menos salidas totales entre las candidatas |
| `centrica` | La más cercana (< 500 m) a cada lugar emblemático (Sol, Prado, Atocha…) |
| `afueras` | Las más alejadas de Sol, separadas ≥ 1 km entre sí |
| `facil` | Mayor R² en una primera ronda (RNA por estación) sobre todas las candidatas libres |
| `dificil` | Menor R² en esa primera ronda |

Es candidata la estación con ≥ 30 salidas en cada mes del periodo (240 de 294 en los datos originales). El R² de la primera ronda se calcula **solo con bloques que empiezan antes de 2022-08-01** (`CORTE_GRUPOS`); `errores_por_grupo.csv` informa también de los bloques posteriores, no usados para elegir los grupos. `--muestra N` usa una muestra aleatoria (semilla 42), más rápida. Las coordenadas son la mediana de las posiciones GPS de desbloqueo. `estaciones_seleccionadas.csv` recoge nombre, grupo, motivo, demanda, distancia a Sol y coordenadas.

### Algoritmos

Todos usan la misma entrada `[ventana | variables | estación en one-hot]`, el mismo estandarizado por bloque, `clip(pred, 0, ∞)` y la **misma validación interna**: con `--validacion cronologica` (por defecto) se entrena con todo menos la última semana de cada estación, que sirve para parar el entrenamiento (RNA, LSTM), elegir el número de árboles (boosting) o elegir `alpha` (Ridge). `--validacion aleatoria` (RNA y boosting) usa el 10 % de filas al azar, como en versiones anteriores.

| Carpeta | Modelo | Opciones propias |
|---|---|---|
| `rna/` | MLP de scikit-learn | `--ocultas` (64 32 global / 32 16 por estación), `--iteraciones` (200 / 500) |
| `ridge/` | Ridge | `--alphas` |
| `gradient_boosting/` | `HistGradientBoostingRegressor` | `--perdida` (`squared_error`, `absolute_error`, `poisson`), `--iteraciones`, `--tasa`, `--profundidad` |
| `lstm/` | LSTM (32 unidades) + capas de 64 y 32 | `--ocultas`, `--iteraciones`, `--paciencia`, `--tasa`, `--hilos` |

Opciones comunes:

| Opción | Por defecto | Significado |
|---|---|---|
| `datos` (posicional) | `sel60` | `sel60`, `top50` o ruta de un CSV con el mismo formato |
| `--modo` | `global` | `global` = un modelo para todas (estación en one-hot); `estacion` = uno por estación |
| `--sin-id-estacion` | — | en modo global, sin la estación en one-hot |
| `--variables` | `todas` | lista separada por comas (`hora,dia_semana,mes,festivo,temperatura,precipitacion`), `todas` o `ninguna` |
| `--meteo` | `actual` | `actual` = meteo de la hora que se predice (pronóstico perfecto); `anterior` = la de la hora anterior |
| `--ventana` | 24 | horas anteriores (1–168) |
| `--semanas` | 52 | semanas de entrenamiento por bloque (1–52) |
| `--semilla` | 0 | semilla |
| `--nombre` | nombre del CSV | nombre del estudio (se añade `_1`, `_2`… si se repite) |
| `--jobs` | 2 global / todos por estación | bloques en paralelo |
| `--hasta` | — | solo bloques que empiezan antes de esa fecha |
| `--carpeta` | — | guardar en esta carpeta en vez de `results/<algoritmo>/…` |

### Cómo funciona la evaluación

**Ventana deslizante** cronológica: se entrena con las 52 semanas anteriores (por defecto), se evalúa con las 2 siguientes (336 h), se avanza dos semanas y se reentrena. El primer bloque empieza siempre en la misma hora, así que **todas las ejecuciones sobre un mismo CSV usan los mismos 29 bloques** (enero 2022 – febrero 2023 en `sel60`) y son comparables. En el test, cada predicción usa las observaciones reales anteriores a su hora, incluidas las del mismo bloque: son predicciones sucesivas a una hora vista, no dos semanas predichas de golpe.

### Ejecutar experimentos

```powershell
python src/bicimadness/rna/entrenar.py                                   # sel60, global, todas las variables
python src/bicimadness/rna/entrenar.py sel60 --modo estacion
python src/bicimadness/rna/entrenar.py sel60 --meteo anterior --nombre sel60_meteo_anterior
python src/bicimadness/ridge/entrenar.py
python src/bicimadness/gradient_boosting/entrenar.py --perdida poisson --nombre sel60_poisson
python src/bicimadness/lstm/entrenar.py
python src/bicimadness/rna/entrenar.py --help
```

Para probar rápido, añade `--hasta 2022-02-10` (3 bloques). Para comparar:

```powershell
python src/bicimadness/comparar_estudios.py rna --datos sel60      # estudios de un algoritmo: ordena por MAE y t-test contra el mejor
python src/bicimadness/comparar_algoritmos.py --datos sel60        # mejor estudio de cada algoritmo; escribe results/comparacion_algoritmos*.csv
```

### Interpretar los resultados

- **MAE**: error absoluto medio en salidas por estación y hora. **RMSE**: igual pero penaliza más los errores grandes. **R²**: ajuste respecto a predecir la media; 1 es perfecto, 0 iguala la media y puede ser negativo.
- `resumen.csv` trae **dos versiones**. `mae`, `rmse`, `r2` son la media de las métricas de cada (estación, bloque): cada estación pesa igual, y el R² es muy severo con series con muchos ceros. `mae_agregado`, `rmse_agregado`, `r2_agregado` se calculan con todas las predicciones juntas (R² respecto a la media global del test). Con la misma predicción pueden diferir mucho, así que **compara siempre dentro de la misma versión** y cita cuál usas.
- `mae_referencia` etc. son los de la referencia «misma hora de ayer».

| Archivo de cada ejecución | Contenido |
|---|---|
| `config.json` | argumentos, bloques, commit de git y versión de Python |
| `resumen.csv` | métricas (ambas versiones), referencia y tiempos |
| `resultados_folds.csv` | métricas por estación y bloque |
| `metricas_por_bloque.csv` | MAE, RMSE y R² de cada bloque (todas las estaciones juntas) |
| `tiempos.csv` | segundos de entrenamiento y predicción por bloque |
| `error_por_estacion.csv`, `error_por_hora.csv` | errores por estación y por hora del día |
| `errores_por_grupo.csv` | por grupo de estaciones, en todos los bloques y desde 2022-08-01 (solo `sel60`) |
| `predicciones.csv.gz` | fecha, estación, bloque, real y predicción |

El t-test pareado de `comparar_*.py` compara el MAE de las mismas parejas (estación, bloque). Esas parejas **no son independientes** (misma estación y misma semana, meteo común), de modo que los p-valores salen optimistas; no demuestran relevancia práctica y no se corrigen las comparaciones múltiples.

### Regenerar los datos

Opcional. Los viajes originales (~2,2 GB) se descargan aparte y se colocan con sus nombres originales:

```text
data/raw/bicimad_2021/  202101_movements.zip … 202105_movements-json.zip, trips_21_06_June.csv … trips_21_12_December.csv
data/raw/bicimad_2022/  trips_22_01_January.csv … trips_22_12_December.csv
data/raw/bicimad_2023/  trips_23_01_January.csv, trips_23_02_February.csv
```

```powershell
python src/bicimadness/preprocesamiento/preprocess.py        # top 50
python src/bicimadness/preprocesamiento/count_stations.py    # recuentos y coordenadas de todas (data/cache/)
python src/bicimadness/preprocesamiento/select_stations.py   # 60 estaciones (entrena la primera ronda; --repetir, --muestra N, --modo)
python src/bicimadness/preprocesamiento/weather.py           # meteorología (descarga si falta; necesita Internet)
```

La primera ronda entrena una red por estación para unas 200 estaciones (10–20 min con 8 núcleos); una red global con tantas estaciones no cabe en 8 GB de RAM. Asegúrate de que existe `data/processed/`.

### Solución de problemas

| Síntoma | Qué comprobar |
|---|---|
| `ModuleNotFoundError` | Activa `.venv` e instala con ese `python -m pip`. |
| Falta el CSV horario o la meteorología | Regenera el conjunto; la meteo debe cubrir todas las horas objetivo. |
| Poca memoria en modo global | `--jobs 1`. |
| LSTM muy lenta | `--hasta`, `--iteraciones` menor o `--hilos`. |
| `ConvergenceWarning` (RNA) | Se alcanzó el máximo de épocas; revisa el estudio antes de fiarte. |
| No hay bloques de test | El CSV necesita historia para la ventana, el entrenamiento y 336 h de test. |

### Limitaciones y trabajo futuro

- **Horizonte:** solo una hora hacia delante; la predicción a 24 h para ~20 estaciones está pendiente.
- **Meteo:** con `--meteo actual` se usa el tiempo observado de la hora objetivo (pronóstico perfecto, optimista); `anterior` usa solo lo ya observado. No hay previsión real ni eventos (partidos, conciertos).
- **Sesgo de selección:** los grupos fácil/difícil se eligen por R² con parte de las mismas series; mira solo `desde_2022-08-01` para evaluarlos con semanas no usadas. Una estación del grupo difícil (id 174) tiene R² muy negativo y domina los promedios. Elegir y evaluar la configuración sobre los mismos 29 bloques da errores algo optimistas.
- **Estadística:** una sola semilla por estudio; t-test con observaciones dependientes (arriba).
- **Cobertura:** los viajes terminan en febrero de 2023; no se ha establecido el rendimiento en la red actual. Una sola estación meteorológica para toda la ciudad; el cambio horario se simplifica.
- **Demanda frente a salidas:** las salidas observadas no miden la demanda no atendida (estación vacía) ni incluyen llegadas, stock, rutas ni despliegue.

Documentación adicional: [propuesta](docs/propuesta.pdf), [seguimiento del 20/11](docs/seguimiento_20nov.md) (partes desfasadas respecto al código actual) e [índice de la memoria](docs/indice_memoria.md). Para entender el código: `preprocesamiento/preprocess.py`, luego `ventana()` y `bloques()` en `comun/datos.py`, `_bloque()` y `guardar()` en `comun/experimento.py` y `rna/entrenar.py`.

[Read in English](#english)

---

## English

BiciMadness predicts **how many bicycles will leave a BiciMAD station during the next hour**. It is a Python research project for *Inteligencia Artificial en las Organizaciones* (UC3M, 2026/27) that supports bicycle redistribution planning in Madrid.

It turns historical trips into hourly departure counts, trains several algorithms (neural network, Ridge, gradient boosting, LSTM) and evaluates them **under identical conditions**: same inputs, same test blocks, same internal validation and same metrics. The main experiment uses **60 stations selected to span demand levels, locations and prediction difficulty**; an earlier one uses the 50 busiest. There is no web app, live API, 24-hour forecast, redistribution optimizer or saved model.

**Quick start** (Python ≥ 3.10):

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: py -m venv .venv; .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/bicimadness/rna/entrenar.py               # global MLP on 60 stations; results in results/rna/<date>_<name>/
```

Prepared data is already in `data/processed/` and `data/raw/`, so the original trips (~2.2 GB) are only needed to rebuild it.

**What is predicted.** For each station and hour, the number of departures. Inputs: the previous 24 hourly counts of that station, calendar (hour, weekday, month, holiday), temperature and rainfall, and the station ID (one-hot) in global models. The horizon is **one hour ahead**; the 24-hour input window is not a 24-hour forecast.

**Algorithms** (one folder each, all via `comun/experimento.py`): `rna/` (MLP), `ridge/`, `gradient_boosting/`, `lstm/` (PyTorch). Each defines only `entrenar(X, y, args, val)`; folds, scaling, timing, metrics and files are shared. Internal validation is the **last chronological week** of each training period (early stopping for MLP/LSTM, number of trees for boosting, `alpha` for Ridge); `--validacion aleatoria` restores the old random 10 % split for MLP and boosting.

**Evaluation.** Chronological sliding window: train on the previous 52 weeks, test on the next two (336 h), advance two weeks, retrain. Every run on the same CSV uses the same **29 blocks** (Jan 2022 – Feb 2023 for `sel60`), so runs are comparable. Predictions are successive one-hour-ahead forecasts using real past observations.

**Metrics.** MAE, RMSE, R². `resumen.csv` has two versions: the mean over (station, block) pairs (`mae`, `rmse`, `r2`; R² is harsh on sparse series) and the pooled one over all predictions (`*_agregado`). Compare only within the same version. `comparar_estudios.py` and `comparar_algoritmos.py` run paired t-tests on per-(station, block) MAE; those pairs are not independent, so p-values are optimistic.

**Commands:**

```bash
python src/bicimadness/ridge/entrenar.py
python src/bicimadness/gradient_boosting/entrenar.py --perdida poisson --nombre sel60_poisson
python src/bicimadness/lstm/entrenar.py
python src/bicimadness/comparar_algoritmos.py --datos sel60
```

Use `--hasta 2022-02-10` for a quick 3-block test, `--meteo anterior` to use only already-observed weather, and `--help` on any script for all options (see the Spanish tables above).

**Station selection.** Six groups of ten: high/low demand, central/outlying (distance from Puerta del Sol), and easy/hard to predict (first-round R² of one network per candidate station, on blocks before 2022-08-01 only). Candidates need ≥ 30 departures in every month.

**Limitations.** One-hour horizon only; target-hour weather is a perfect forecast (optimistic) and no events are modelled; easy/hard groups are partly selected on the evaluated series (use the `desde_2022-08-01` rows) and station 174 has a very negative R²; one seed per study; dependent observations in the t-test; trips end in February 2023; observed departures do not measure unmet demand and exclude arrivals and stock.

[Volver al inicio](#bicimadness) · [Ir a Español](#español)
