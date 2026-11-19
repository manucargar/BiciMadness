# BiciMadness

**Idioma / Language:** [Español](#español) · [English](#english)

## English

BiciMadness predicts **how many bicycles will leave a BiciMAD station during the next hour**. It is a Python research project for *Inteligencia Artificial en las Organizaciones* at UC3M (2026/27), intended to support decisions about redistributing bicycles around Madrid.

The project turns historical trips into hourly departure counts, trains forecasting models, and measures their performance on later periods. The main experiment uses **60 stations selected to represent different demand levels, locations, and prediction difficulties**. An earlier experiment uses the 50 busiest stations.

This repository contains command-line experiments and CSV reports. It does not currently provide a web application, a live prediction API, a redistribution optimizer, or saved models for deployment.

### Contents

- [Quick start](#quick-start)
- [What the model predicts](#what-the-model-predicts)
- [Project structure](#project-structure)
- [Data and preprocessing](#data-and-preprocessing)
- [How the 60 stations are selected](#how-the-60-stations-are-selected)
- [Models and features](#models-and-features)
- [How evaluation works](#how-evaluation-works)
- [Running experiments](#running-experiments)
- [Understanding the results](#understanding-the-results)
- [Rebuilding the datasets](#rebuilding-the-datasets)
- [Troubleshooting](#troubleshooting)
- [Limitations and future work](#limitations-and-future-work)
- [Further reading](#further-reading)

### Quick start

Run commands from the project root: the folder containing this README and `requirements.txt`. Python **3.10 or later** is required by the current dependencies; Python 3.10 or 3.11 is a practical starting point.

#### 1. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements install NumPy, scikit-learn, SciPy, Joblib, PyTorch, and timezone data. PyTorch is used only by the LSTM, but the full requirements file installs it anyway. No GPU is required; the current LSTM implementation runs on CPU.

#### 2. Check the prepared inputs

The current workspace already contains:

```text
data/processed/salidas_por_hora_sel60.csv
data/processed/estaciones_seleccionadas.csv
data/processed/salidas_por_hora_top50.csv
data/processed/estaciones_top50.csv
data/raw/meteo_horario.csv
data/raw/calendario_laboral.csv
```

If these files are available, you can train without downloading or processing the original trips. See [Rebuilding the datasets](#rebuilding-the-datasets) if they are missing.

#### 3. Run the main experiment

```bash
python src/bicimadness/rna/entrenar.py
```

With no arguments this trains and evaluates a global neural network on the 60 selected stations (`sel60`), using the previous 24 hours of departures, calendar features, and weather, with 52 weeks of training.

Each run is saved in its own dated folder, such as `results/rna/2026-11-19_1530_sel60/`. Open `resumen.csv` for the overall metrics and timing, and `errores_por_grupo.csv` for the group analysis. Training can take tens of minutes, depending on your computer. Output appears in the terminal; there is no browser interface.

### What the model predicts

A **departure** is a trip starting at a station. For each station and each hour, the target is the number of departures during that hour.

For example, to predict departures between **09:00 and 09:59**, a model with a 24-hour window receives:

- The observed departure counts from 09:00 yesterday through 08:00 today.
- Calendar information for today's 09:00 hour.
- Temperature and rainfall for that target hour, when weather features are enabled.
- The station identifier, when using a global model.

The output is a nonnegative numeric estimate, such as `3.7` departures. Predictions are clipped at zero but are not rounded to whole bicycles before evaluation.

In notation, the training example is:

```text
inputs:  departures[t-24], ..., departures[t-1], features[t]
target:  departures[t]
```

The 24-hour **input window** is different from the **forecast horizon**: the current system predicts one hour ahead. It does not forecast the next 24 hours from a single starting point.

### Project structure

```text
.
├── README.md
├── requirements.txt
├── src/bicimadness/
│   ├── preprocesamiento/
│   │   ├── preprocess.py        # Raw trips → hourly data for the top 50 stations
│   │   ├── count_stations.py    # Raw trips → counts and coordinates for all stations
│   │   ├── select_stations.py   # Select 60 stations and write their hourly dataset
│   │   └── weather.py           # Download and prepare hourly Madrid-Retiro weather
│   ├── comun/
│   │   ├── datos.py             # Loading, features, input window, and test blocks (shared)
│   │   ├── experimento.py       # Runs one study and saves its results (shared)
│   │   └── comparar.py          # Reads saved runs; paired t-test
│   ├── rna/
│   │   └── entrenar.py          # Neural network (MLP): one study per run
│   ├── comparar_estudios.py     # Compare the studies of one algorithm
│   ├── comparar_algoritmos.py   # Compare the best study of each algorithm
│   └── antiguo/                 # Previous all-in-one comparison (Ridge, boosting, LSTM...)
├── data/
│   ├── raw/                   # Original trips, holiday calendar, and weather
│   ├── processed/             # Hourly datasets and station selection tables
│   └── cache/                 # Intermediate counts; regenerated as needed
├── results/                   # Experiment outputs; generated locally
└── docs/                      # Proposal, progress notes, and report outline
```

Raw trip archives, generated results, intermediate caches, and `.venv/` are excluded by `.gitignore`. The root-level `mempalace.yaml` and `entities.json` are local memory-tool configuration, not inputs to the forecasting pipeline.

Each algorithm has its own folder (`rna/` now; `lstm/`, `gradient_boosting/`... later) with an `entrenar.py` that only defines how to train the model. Everything else — test blocks, timing, metrics, and output files — lives in `comun/experimento.py`, so every algorithm is evaluated and saved identically. To add an algorithm, copy `rna/` and change `entrenar()` and its own options.

### Data and preprocessing

#### Sources

- **Trips:** [Madrid Open Data — BiciMAD historical trips](https://datos.madrid.es/dataset/900034-0-bicimad-viajes-estaciones). The prepared trip dataset covers January 2021 through 18 February 2023.
- **Holidays:** Madrid's working-day calendar in `data/raw/calendario_laboral.csv`.
- **Weather:** [Meteostat](https://meteostat.net), Madrid-Retiro station **08222**, redistributing AEMET observations. The weather script documents the source as CC BY-NC 4.0; retain source attribution when reusing the data.

#### How trips become model inputs

1. Read trip CSVs and the earlier JSON archives.
2. Filter CSV trips shorter than 120 seconds, to align with the treatment of failed unlocks in the JSON source.
3. Convert the earlier UTC JSON timestamps to Madrid local time; CSV trip timestamps are treated as local time.
4. Count departures by station and hour.
5. Fill hours with no recorded departures with zero.
6. Add weekday, month, and holiday information.
7. Select stations and write an hourly CSV.

The hourly files use the same schema:

| Column | Meaning |
|---|---|
| `fecha_hora` | Madrid local time, `YYYY-MM-DD HH:00`; identifies the start of an hour |
| `id_estacion` | Station identifier |
| `dia_semana` | ISO weekday: 1 = Monday, 7 = Sunday |
| `mes` | Month, 1–12 |
| `festivo` | 1 for a public holiday, otherwise 0 |
| `salidas` | Number of departures in that hour; the prediction target |

Models derive hour, weekday, and month from `fecha_hora`. They read the holiday flag from the CSV. Weather is joined separately by timestamp from `meteo_horario.csv`, whose columns are `fecha_hora`, `temperatura` (°C), and `precipitacion` (mm during the hour).

The loaders expect each station to have the same complete hourly timeline in chronological order. A custom CSV must preserve this alignment.

The weather script converts UTC to local time, averages temperature and sums rainfall when autumn daylight-saving timestamps coincide, and fills missing values forward, then backward for any leading gaps. The resulting local-time grid does not retain timezone offsets.

### How the 60 stations are selected

The top-50 experiment focuses on the busiest stations. The main experiment broadens the selection into six groups of ten stations:

| Group in the files | Meaning | Selection rule |
|---|---|---|
| `demanda_alta` | High demand | Highest total departures |
| `demanda_baja` | Low demand | Lowest total departures among remaining eligible stations |
| `centrica` | Central | Nearby stations within 500 m of landmarks such as Sol, Prado, and Atocha |
| `afueras` | Outlying | Farthest from Puerta del Sol, at least 1 km apart from one another |
| `facil` | Easier to predict | Highest first-round R² among all remaining eligible stations |
| `dificil` | Harder to predict | Lowest first-round R² among those stations |

Eligibility requires at least **30 departures in every month** of the input period. The project notes report 240 eligible stations out of 294 across 26 months for the original dataset.

Groups are filled in the order above, without reusing stations. For the difficulty groups, the script runs one MLP per station with all features over all remaining candidates (`--modo global` for one shared network) and ranks stations by R², computed **only on test blocks starting before 2022-08-01** (`CORTE_GRUPOS` in `comun/datos.py`). `errores_por_grupo.csv` then also reports each group on the later blocks, which were not used to choose the groups. `--muestra 60` uses a random sample (seed `42`) instead, which is faster. Station coordinates are estimated from median trip unlock GPS positions.

The selection table, `estaciones_seleccionadas.csv`, records station names, groups, reasons, total and mean departures, distance from Sol, coordinates, and first-round R² where available. Selection depends on the supplied data; fixed seeds do not guarantee identical numerical results across all dependency versions and hardware.

### Models and features

#### Neural network studies (`rna/entrenar.py`)

**MLP** means multilayer perceptron: a neural network with fully connected layers. `rna` abbreviates the Spanish *red neuronal artificial*. Each run is one **study**: one concrete configuration.

| Option | Default | Meaning |
|---|---|---|
| `datos` (positional) | `sel60` | `sel60`, `top50`, or the path of any CSV with the same columns |
| `--modo` | `global` | `global` = one network for all stations; `estacion` = one network per station |
| `--variables` | `todas` | Comma-separated features, `todas` (all), or `ninguna` (history only) |
| `--meteo` | `actual` | `actual` = weather of the predicted hour (a perfect forecast); `anterior` = previous hour's weather |
| `--ventana` | 24 | Previous hours used (1–168) |
| `--semanas` | 52 | Training weeks per block (1–52) |
| `--nombre` | CSV name | Study name; repeated names get `_1`, `_2`... |
| `--ocultas` | 64 32 global / 32 16 station | Hidden-layer sizes |
| `--iteraciones` | 200 global / 500 station | Maximum epochs |
| `--jobs` | 2 global / all cores station | Blocks trained in parallel |
| `--hasta` | — | Only test blocks starting before this date |

A **station model** learns from one station's observations. A **global model** learns from all selected stations and receives the station ID as a one-hot feature.

One-hot encoding represents a category as separate 0/1 columns: for example, hour of day uses 24 columns. Calendar features include hour, weekday, month, and holiday status; optional numerical features are temperature and precipitation.

Inputs are standardized with the mean and deviation of each training block, and networks use early stopping and seed `0` (`--semilla`). Early stopping can finish training sooner.

#### Previous algorithm comparison (`antiguo/`)

Until each algorithm has its own folder, `antiguo/compare_models.py` keeps the earlier all-in-one comparison. Its outputs use the old format and are not read by `comparar_algoritmos.py`. It evaluates these model names:

| CLI name | What it does |
|---|---|
| `ref_ayer` | Repeats departures from the same hour yesterday |
| `ref_semana` | Repeats departures from the same hour last week |
| `media_historica` | Uses the training-period mean for that station, hour, and weekday |
| `regresion_lineal` | Ridge regression, a regularized linear model |
| `gradient_boosting` | Histogram-based gradient boosting regression |
| `mlp` | Global MLP with 64- and 32-unit hidden layers |
| `lstm` | A recurrent network that processes the 24 hourly counts, then combines its representation with calendar/weather and station features |

The LSTM uses 32 hidden units and a 64–32-unit prediction head. It trains for up to 10 epochs by default, with early stopping. Learned models in this comparison share a 24-hour input window and 52-week training history.

### How evaluation works

The outer evaluation respects time order using a **sliding window**:

1. Train on the previous 52 weeks by default.
2. Evaluate on the next two weeks (336 hours).
3. Move the test start forward by two weeks.
4. Retrain using the history immediately before the new test period.

Each train/test split is called a **fold** or block. The first block always starts at the same hour (leaving room for the longest allowed window, 1 week, and training history, 52 weeks), so **every run on the same CSV uses the same 29 blocks** and runs can be compared with a paired t-test.

Within a test period, predictions use the actual observations preceding each target hour, including earlier observations from that test period. This evaluates repeated one-hour-ahead predictions with newly available observations; it does not predict an entire two-week block in advance.

`antiguo/compare_models.py` uses every third fold by default (`--folds-cada 3`). Use `--folds-cada 1` for all folds.

Internal early-stopping validation is separate from this outer evaluation. MLP early stopping and the LSTM's validation use random subsets of training observations, rather than a final chronological validation segment.

### Running experiments

#### One study per run

```bash
python src/bicimadness/rna/entrenar.py                       # sel60, global, all features
python src/bicimadness/rna/entrenar.py top50                 # top-50 stations
python src/bicimadness/rna/entrenar.py sel60 --modo estacion # one network per station
python src/bicimadness/rna/entrenar.py sel60 --variables hora,dia_semana,mes,festivo --nombre sel60_calendario
python src/bicimadness/rna/entrenar.py sel60 --meteo anterior
python src/bicimadness/rna/entrenar.py sel60 --ventana 48 --semanas 26
```

Each run creates `results/rna/<date>_<name>/`; earlier runs are never overwritten. For the full list of options:

```bash
python src/bicimadness/rna/entrenar.py --help
```

#### Compare the studies of one algorithm

```bash
python src/bicimadness/comparar_estudios.py rna
python src/bicimadness/comparar_estudios.py rna --datos sel60
```

For each CSV, ranks the runs by MAE and runs a paired t-test of each against the best. `p_vs_mejor >= 0.05` means it is not significantly different from the best (`equivalente_a_la_mejor = sí`). Writes `results/rna/comparacion_estudios.csv`.

#### Compare algorithms

```bash
python src/bicimadness/comparar_algoritmos.py
```

For each CSV, takes the lowest-MAE run from every algorithm folder in `results/`, adds the "same hour yesterday" reference, and compares MAE, RMSE, R², and time, with pairwise t-tests. Writes `results/comparacion_algoritmos.csv` and `results/comparacion_algoritmos_ttest.csv`.

### Understanding the results

#### Metrics

- **MAE (mean absolute error):** average absolute difference between prediction and observed departures. Lower is better. An MAE of 1 means the average absolute error is one departure per station-hour.
- **RMSE (root mean squared error):** also measured in departures, but penalizes large misses more strongly. Lower is better.
- **R²:** measures fit relative to predicting the mean of the evaluated target values. Higher is better; 1 is perfect, 0 matches that mean-based benchmark, and negative values are possible. It is not percentage accuracy.

`resumen.csv` averages metrics across station–block pairs. In particular, its RMSE and R² are averages of individual scores, not scores recomputed from all pooled predictions. Stations receive equal weight in this summary, regardless of trip volume.

#### Generated files

Each run folder `results/<algorithm>/<date>_<name>/` contains:

| File | What to use it for |
|---|---|
| `config.json` | Every option, dates, git commit, and Python version: exactly what was run |
| `resumen.csv` | Overall MAE, RMSE, R², the same for the "same hour yesterday" reference, and timing |
| `resultados_folds.csv` | Metrics for each station and block |
| `tiempos.csv` | Training and prediction seconds for each block |
| `error_por_estacion.csv` | Metrics averaged across blocks for each station |
| `error_por_hora.csv` | MAE and RMSE by hour of day |
| `errores_por_grupo.csv` | Group averages for all blocks and for blocks from 2022-08-01 (only for CSVs with groups, such as `sel60`) |
| `predicciones.csv.gz` | Date, station, block, observed value, and prediction |

The paired t-test compares MAE for matching station–block pairs. A small p-value is evidence under the test's assumptions, not proof of practical importance: station/time dependence and multiple comparisons limit the interpretation.

### Rebuilding the datasets

This is optional when the prepared inputs are already present. Original trip files occupy approximately 2.2 GB and must be downloaded separately from the Madrid Open Data source.

Place the files in this layout, preserving their original names:

```text
data/raw/bicimad_2021/
    202101_movements.zip ... 202105_movements-json.zip  # January–May JSON archives
    trips_21_06_June.csv ... trips_21_12_December.csv   # June–December CSVs
data/raw/bicimad_2022/
    trips_22_01_January.csv ... trips_22_12_December.csv
data/raw/bicimad_2023/
    trips_23_01_January.csv
    trips_23_02_February.csv
```

Keep `data/raw/calendario_laboral.csv` available. Ensure `data/processed/` exists before running the trip-processing scripts.

#### Top-50 dataset

```bash
python src/bicimadness/preprocesamiento/preprocess.py
```

Writes `data/processed/salidas_por_hora_top50.csv` and `data/processed/estaciones_top50.csv`.

#### Selected-60 dataset

```bash
python src/bicimadness/preprocesamiento/count_stations.py
python src/bicimadness/preprocesamiento/select_stations.py
```

The first command generates all-station counts and location information in `data/cache/`. The second performs selection, training the first-round model if needed, and writes:

- `data/processed/estaciones_seleccionadas.csv`
- `data/processed/salidas_por_hora_sel60.csv`
- `results/primera_ronda/` reports and `data/cache/salidas_por_hora_primera_ronda.csv`

The first round trains one network per station for about 200 stations (around 10–20 minutes on 8 cores). A global network over that many stations (`--modo global`) does not fit in 8 GB of RAM.

#### Weather

```bash
python src/bicimadness/preprocesamiento/weather.py
```

Downloads annual compressed weather files to `data/raw/meteo/` if absent, then writes `data/raw/meteo_horario.csv`. Internet access is required for missing downloads. The weather script covers January 2021 through February 2023, extending past the final trip timestamp.

After rebuilding, run the main experiment from the quick start.

### Troubleshooting

| Symptom | What to check |
|---|---|
| `ModuleNotFoundError` | Activate `.venv` and install dependencies using that environment's `python -m pip`. In PyCharm, select the same virtual-environment interpreter. |
| Missing hourly CSV | Verify `data/processed/`; regenerate the relevant dataset if necessary. |
| Missing weather file or weather timestamp `KeyError` | Generate `meteo_horario.csv` and verify that it covers every target timestamp. Custom date ranges require matching weather data. |
| Memory pressure during global training | Use `--jobs 1`. This reduces the number of blocks trained at the same time. |
| Slow comparison | Begin with a subset of `--modelos` or increase `--folds-cada`; include all folds for the final comparison. |
| Old comparison results seem unchanged after changing its settings | `antiguo/compare_models.py` caches folds: use a fresh `--resultados` directory. |
| No test folds for a custom dataset | Provide enough history for the largest input window, training period, and a full 336-hour test block. |
| `ConvergenceWarning` | The optimizer reached its iteration limit. Review the affected experiment before relying on its results; CLI hidden-layer settings do not adjust that limit. |

### Limitations and future work

- **Target-hour weather:** with `--meteo actual` (default) the model uses the observed weather of the predicted hour, which stands for a perfect one-hour-ahead forecast. `--meteo anterior` uses only the previous hour's weather; the difference between both measures the value of a good forecast.
- **Station-selection bias:** easy/difficult groups are chosen with blocks before 2022-08-01; only the `desde_2022-08-01` rows of `errores_por_grupo.csv` evaluate them on unused weeks. The best configuration is chosen and reported on the same 29 blocks, so its error can be slightly optimistic.
- **Dependent observations:** station–fold errors are not fully independent; interpret t-test p-values cautiously.
- **Historical coverage:** the prepared trips end in February 2023. Performance on today's network has not been established by these experiments.
- **Demand versus observed departures:** recorded trips reflect actual usage and available bicycles; they do not directly measure unmet demand at empty stations.
- **Weather and time resolution:** one weather station represents the city, missing weather is imputed, and local-time daylight-saving handling simplifies the timeline.
- **Operational scope:** no arrivals, stock balance, routing, live ingestion, model export, or deployment workflow is implemented.

The project notes propose exploratory charts and maps, event features such as matches and concerts, a complete all-fold model comparison, and a separate 24-hour forecast experiment. Treat these as planned work unless corresponding implementation and outputs are added.

### Further reading

- [Original proposal](docs/propuesta.pdf)
- [Progress notes for 20 November](docs/seguimiento_20nov.md) — Spanish project context and preliminary findings; check the actual run configuration before citing results.
- [Proposed final report outline](docs/indice_memoria.md)

To understand the implementation, start with `preprocesamiento/preprocess.py`, then read `ventana()` and `bloques()` in `comun/datos.py`, `_bloque()` and `guardar()` in `comun/experimento.py`, and `entrenar()` in `rna/entrenar.py`. Continue with `preprocesamiento/select_stations.py` and the two comparison scripts.

---

## Español

[Ir a English](#english) · [Volver al inicio](#bicimadness)

BiciMadness predice **cuántas bicicletas saldrán de una estación de BiciMAD durante la hora siguiente**. Es un proyecto de investigación en Python para la asignatura *Inteligencia Artificial en las Organizaciones* de la UC3M (curso 2026/27), orientado a apoyar la planificación de la redistribución de bicicletas en Madrid.

El proyecto transforma viajes históricos en recuentos horarios, entrena modelos predictivos y los evalúa en periodos posteriores. El experimento principal utiliza **60 estaciones seleccionadas para representar distintos niveles de demanda, ubicaciones y dificultades de predicción**. Un experimento anterior utiliza las 50 estaciones con más salidas.

El repositorio contiene experimentos ejecutables desde la terminal e informes CSV. Actualmente no incluye una aplicación web, una API de predicción en directo, un optimizador de redistribución ni modelos guardados para desplegarlos.

### Índice en español

- [Inicio rápido](#inicio-rápido)
- [Qué predice el modelo](#qué-predice-el-modelo)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Datos y preprocesado](#datos-y-preprocesado)
- [Selección de las 60 estaciones](#selección-de-las-60-estaciones)
- [Modelos y atributos](#modelos-y-atributos)
- [Cómo funciona la evaluación](#cómo-funciona-la-evaluación)
- [Ejecutar experimentos](#ejecutar-experimentos)
- [Interpretar los resultados](#interpretar-los-resultados)
- [Regenerar los datos](#regenerar-los-datos)
- [Solución de problemas](#solución-de-problemas)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
- [Documentación adicional](#documentación-adicional)

### Inicio rápido

Ejecuta los comandos desde la raíz del proyecto: la carpeta que contiene este README y `requirements.txt`. Las dependencias actuales requieren **Python 3.10 o superior**; Python 3.10 o 3.11 es un punto de partida práctico.

#### 1. Crear y activar un entorno virtual

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se instalan NumPy, scikit-learn, SciPy, Joblib, PyTorch y los datos de zonas horarias. Solo la LSTM utiliza PyTorch, aunque el archivo de requisitos lo instala siempre. No hace falta una GPU: la implementación actual de la LSTM se ejecuta en CPU.

#### 2. Comprobar los datos preparados

El espacio de trabajo actual ya contiene:

```text
data/processed/salidas_por_hora_sel60.csv
data/processed/estaciones_seleccionadas.csv
data/processed/salidas_por_hora_top50.csv
data/processed/estaciones_top50.csv
data/raw/meteo_horario.csv
data/raw/calendario_laboral.csv
```

Con estos archivos puedes entrenar sin descargar ni procesar de nuevo los viajes originales. Si faltan, consulta [Regenerar los datos](#regenerar-los-datos).

#### 3. Ejecutar el experimento principal

```bash
python src/bicimadness/rna/entrenar.py
```

Sin argumentos, entrena y evalúa una red global con las 60 estaciones seleccionadas (`sel60`), las salidas de las últimas 24 horas, el calendario y la meteorología, con 52 semanas de entrenamiento.

Cada ejecución se guarda en su propia carpeta con fecha, por ejemplo `results/rna/2026-11-19_1530_sel60/`. Abre `resumen.csv` para consultar las métricas generales y los tiempos, y `errores_por_grupo.csv` para el análisis por grupos. El entrenamiento puede tardar decenas de minutos, según el ordenador. Los mensajes aparecen en la terminal; no hay interfaz en el navegador.

### Qué predice el modelo

Una **salida** es un viaje que comienza en una estación. Para cada estación y cada hora, el objetivo es estimar cuántos viajes comenzarán durante esa hora.

Por ejemplo, para predecir las salidas entre las **09:00 y las 09:59**, un modelo con una ventana de 24 horas recibe:

- Las salidas observadas desde las 09:00 de ayer hasta las 08:00 de hoy.
- Los atributos de calendario de las 09:00 de hoy.
- La temperatura y la precipitación de la hora objetivo, cuando se utilizan atributos meteorológicos.
- El identificador de estación, cuando se utiliza un modelo global.

La salida es una estimación numérica no negativa, por ejemplo `3.7` salidas. Las predicciones negativas se sustituyen por cero, pero no se redondean a bicicletas enteras antes de evaluarlas.

```text
entradas: salidas[t-24], ..., salidas[t-1], atributos[t]
objetivo: salidas[t]
```

La **ventana de entrada** de 24 horas es distinta del **horizonte de predicción**: el sistema actual predice una hora hacia delante. No predice las próximas 24 horas desde un único instante inicial.

### Estructura del proyecto

```text
.
├── README.md
├── requirements.txt
├── src/bicimadness/
│   ├── preprocesamiento/
│   │   ├── preprocess.py        # Viajes originales → datos horarios de las top 50
│   │   ├── count_stations.py    # Viajes → recuentos y coordenadas de todas las estaciones
│   │   ├── select_stations.py   # Selección de 60 estaciones y generación de su CSV horario
│   │   └── weather.py           # Descarga y preparación de meteorología de Madrid-Retiro
│   ├── comun/
│   │   ├── datos.py             # Carga, variables, ventana y bloques de test (compartido)
│   │   ├── experimento.py       # Ejecuta un estudio y guarda sus resultados (compartido)
│   │   └── comparar.py          # Lectura de ejecuciones guardadas y t-test pareado
│   ├── rna/
│   │   └── entrenar.py          # Red neuronal (MLP): un estudio por ejecución
│   ├── comparar_estudios.py     # Compara los estudios de un algoritmo
│   ├── comparar_algoritmos.py   # Compara el mejor estudio de cada algoritmo
│   └── antiguo/                 # Comparación anterior todo-en-uno (Ridge, boosting, LSTM...)
├── data/
│   ├── raw/                   # Viajes originales, calendario y meteorología
│   ├── processed/             # Series horarias y tablas de estaciones
│   └── cache/                 # Recuentos intermedios que se pueden regenerar
├── results/                   # Resultados generados localmente
└── docs/                      # Propuesta, seguimiento e índice de la memoria
```

`.gitignore` excluye los archivos originales de viajes, los resultados generados, las cachés intermedias y `.venv/`. Los archivos raíz `mempalace.yaml` y `entities.json` son configuración de una herramienta de memoria local; no intervienen en la predicción.

Cada algoritmo tiene su carpeta (`rna/` ahora; más adelante `lstm/`, `gradient_boosting/`...) con un `entrenar.py` que solo define cómo se entrena el modelo. Todo lo demás —bloques de test, tiempos, métricas y ficheros de resultados— está en `comun/experimento.py`, así que todos los algoritmos se evalúan y se guardan igual. Para añadir un algoritmo, copia `rna/` y cambia `entrenar()` y sus opciones propias.

### Datos y preprocesado

#### Fuentes

- **Viajes:** [Datos abiertos de Madrid — históricos de BiciMAD](https://datos.madrid.es/dataset/900034-0-bicimad-viajes-estaciones). Los datos preparados abarcan desde enero de 2021 hasta el 18 de febrero de 2023.
- **Festivos:** calendario laboral de Madrid, en `data/raw/calendario_laboral.csv`.
- **Meteorología:** [Meteostat](https://meteostat.net), estación Madrid-Retiro **08222**, que redistribuye observaciones de AEMET. El script meteorológico documenta la licencia de la fuente como CC BY-NC 4.0; conserva la atribución al reutilizar los datos.

#### De viajes a entradas del modelo

1. Leer los CSV de viajes y los archivos JSON anteriores.
2. Filtrar los viajes CSV de menos de 120 segundos para mantener un criterio compatible con el tratamiento de desbloqueos fallidos de la fuente JSON.
3. Convertir las fechas UTC de los primeros JSON a hora local de Madrid; los CSV se interpretan como hora local.
4. Contar las salidas por estación y hora.
5. Rellenar con cero las horas sin salidas registradas.
6. Añadir día de la semana, mes y festivo.
7. Seleccionar estaciones y escribir el CSV horario.

Los archivos horarios comparten este formato:

| Columna | Significado |
|---|---|
| `fecha_hora` | Inicio de la hora, en hora local de Madrid: `YYYY-MM-DD HH:00` |
| `id_estacion` | Identificador de la estación |
| `dia_semana` | Día ISO: 1 = lunes, 7 = domingo |
| `mes` | Mes, de 1 a 12 |
| `festivo` | 1 si es festivo; 0 en caso contrario |
| `salidas` | Número de salidas durante esa hora; variable que se predice |

Los modelos calculan hora, día de la semana y mes a partir de `fecha_hora`, y leen el indicador de festivo del CSV. La meteorología se cruza por fecha y hora con `meteo_horario.csv`, cuyas columnas son `fecha_hora`, `temperatura` (°C) y `precipitacion` (mm durante la hora).

La carga presupone que todas las estaciones tienen la misma serie horaria completa y ordenada cronológicamente. Un CSV personalizado debe mantener esa alineación.

El script meteorológico convierte UTC a hora local. Cuando dos horas coinciden por el cambio de horario de otoño, promedia la temperatura y suma la precipitación. Rellena los valores ausentes con el anterior y, si quedan huecos al principio, con el siguiente. La serie final no conserva el desfase de zona horaria.

### Selección de las 60 estaciones

El experimento de las top 50 se centra en las estaciones con más actividad. El principal amplía la diversidad con seis grupos de diez estaciones:

| Grupo en los archivos | Significado | Criterio de selección |
|---|---|---|
| `demanda_alta` | Demanda alta | Mayor número total de salidas |
| `demanda_baja` | Demanda baja | Menor número de salidas entre las candidatas restantes |
| `centrica` | Céntrica | Estaciones a menos de 500 m de lugares como Sol, Prado y Atocha |
| `afueras` | Afueras | Las más alejadas de Sol, separadas al menos 1 km entre sí |
| `facil` | Más fácil de predecir | Mayor R² en una primera ronda sobre todas las candidatas restantes |
| `dificil` | Más difícil de predecir | Menor R² en esa primera ronda |

Para ser candidata, una estación debe tener al menos **30 salidas en cada mes** del periodo. Las notas del proyecto indican que, en los datos originales, cumplen el criterio 240 de las 294 estaciones durante 26 meses.

Los grupos se completan en el orden de la tabla, sin repetir estaciones. Para los grupos de dificultad se entrena una MLP por estación con todas las variables sobre todas las candidatas restantes (`--modo global` para una red compartida) y se ordenan las estaciones por R², calculado **solo con los bloques de test que empiezan antes del 2022-08-01** (`CORTE_GRUPOS` en `comun/datos.py`). Después, `errores_por_grupo.csv` da también el error de cada grupo en los bloques posteriores, que no se usaron para elegir los grupos. Con `--muestra 60` se usa una muestra aleatoria (semilla `42`), más rápida. Las coordenadas se estiman mediante la mediana de las posiciones GPS de desbloqueo.

`estaciones_seleccionadas.csv` recoge nombres, grupos, motivos, salidas totales y medias, distancia a Sol, coordenadas y R² de primera ronda cuando existe. La selección depende de los datos; fijar semillas no garantiza resultados numéricos idénticos con cualquier versión de dependencias y hardware.

### Modelos y atributos

#### Estudios de red neuronal (`rna/entrenar.py`)

**MLP** significa perceptrón multicapa: una red neuronal con capas completamente conectadas. `rna` abrevia *red neuronal artificial*. Cada ejecución es un **estudio**: una configuración concreta.

| Opción | Por defecto | Significado |
|---|---|---|
| `datos` (posicional) | `sel60` | `sel60`, `top50` o la ruta de cualquier CSV con las mismas columnas |
| `--modo` | `global` | `global` = una red para todas las estaciones; `estacion` = una red por estación |
| `--variables` | `todas` | Variables separadas por comas, `todas` o `ninguna` (solo el historial) |
| `--meteo` | `actual` | `actual` = meteorología de la hora que se predice (pronóstico perfecto); `anterior` = la de la hora anterior |
| `--ventana` | 24 | Horas anteriores utilizadas (1–168) |
| `--semanas` | 52 | Semanas de entrenamiento en cada bloque (1–52) |
| `--nombre` | nombre del CSV | Nombre del estudio; si se repite se añade `_1`, `_2`... |
| `--ocultas` | 64 32 global / 32 16 por estación | Neuronas de cada capa oculta |
| `--iteraciones` | 200 global / 500 por estación | Épocas máximas |
| `--jobs` | 2 global / todos los núcleos por estación | Bloques entrenados en paralelo |
| `--hasta` | — | Solo los bloques de test que empiezan antes de esa fecha |

Un **modelo por estación** aprende de las observaciones de una sola estación. Un **modelo global** aprende de todas las estaciones seleccionadas y recibe el identificador de estación mediante codificación one-hot.

La codificación **one-hot** representa cada categoría en columnas de ceros y unos; por ejemplo, la hora del día ocupa 24 columnas. Los atributos de calendario son hora, día de la semana, mes y festivo. Temperatura y precipitación son atributos numéricos opcionales.

Las entradas se estandarizan con la media y la desviación del entrenamiento de cada bloque; las redes usan parada temprana y semilla `0` (`--semilla`). La parada temprana puede finalizar antes.

#### Comparación anterior de algoritmos (`antiguo/`)

Hasta que cada algoritmo tenga su carpeta, `antiguo/compare_models.py` mantiene la comparación anterior todo-en-uno. Sus resultados usan el formato antiguo y `comparar_algoritmos.py` no los lee. Acepta estos nombres:

| Nombre en el comando | Funcionamiento |
|---|---|
| `ref_ayer` | Repite las salidas de la misma hora de ayer |
| `ref_semana` | Repite las salidas de la misma hora de la semana anterior |
| `media_historica` | Media del periodo de entrenamiento para esa estación, hora y día de la semana |
| `regresion_lineal` | Regresión Ridge: modelo lineal regularizado |
| `gradient_boosting` | Regresión mediante gradient boosting basado en histogramas |
| `mlp` | MLP global con capas ocultas de 64 y 32 neuronas |
| `lstm` | Red recurrente que procesa las 24 salidas horarias y combina su representación con calendario, meteorología e identidad de estación |

La LSTM utiliza 32 unidades ocultas y capas de predicción de 64 y 32 unidades. Por defecto entrena hasta 10 épocas, con parada temprana. Los modelos entrenados de esta comparación comparten una ventana de 24 horas y un historial de 52 semanas.

### Cómo funciona la evaluación

La evaluación externa respeta el orden temporal mediante una **ventana deslizante**:

1. Entrenar con las 52 semanas anteriores, por defecto.
2. Evaluar con las dos semanas siguientes: 336 horas.
3. Desplazar el inicio del test dos semanas.
4. Volver a entrenar con el historial inmediatamente anterior al nuevo periodo de test.

Cada partición de entrenamiento y test se denomina **fold** o bloque. El primer bloque empieza siempre en la misma hora (deja sitio a la ventana y al entrenamiento más largos permitidos: 1 semana y 52 semanas), así que **todas las ejecuciones sobre un mismo CSV usan los mismos 29 bloques** y se pueden comparar con un t-test pareado.

Dentro del test, cada predicción utiliza las observaciones reales anteriores a su hora objetivo, incluidas las de horas anteriores del mismo periodo de test. Por tanto, se evalúan predicciones sucesivas a una hora con observaciones actualizadas; no se predicen dos semanas completas desde el comienzo del bloque.

`antiguo/compare_models.py` utiliza por defecto uno de cada tres folds (`--folds-cada 3`). Para utilizarlos todos, indica `--folds-cada 1`.

La validación interna para la parada temprana es distinta de la evaluación externa. La MLP y la LSTM utilizan subconjuntos aleatorios de las observaciones de entrenamiento, en lugar de reservar el último segmento cronológico para esa validación interna.

### Ejecutar experimentos

#### Un estudio por ejecución

```bash
python src/bicimadness/rna/entrenar.py                       # sel60, global, todas las variables
python src/bicimadness/rna/entrenar.py top50                 # las 50 estaciones con más salidas
python src/bicimadness/rna/entrenar.py sel60 --modo estacion # una red por estación
python src/bicimadness/rna/entrenar.py sel60 --variables hora,dia_semana,mes,festivo --nombre sel60_calendario
python src/bicimadness/rna/entrenar.py sel60 --meteo anterior
python src/bicimadness/rna/entrenar.py sel60 --ventana 48 --semanas 26
```

Cada ejecución crea `results/rna/<fecha>_<nombre>/`; nunca se sobrescriben las anteriores. Para ver todas las opciones:

```bash
python src/bicimadness/rna/entrenar.py --help
```

#### Comparar los estudios de un algoritmo

```bash
python src/bicimadness/comparar_estudios.py rna
python src/bicimadness/comparar_estudios.py rna --datos sel60
```

Para cada CSV, ordena las ejecuciones por MAE y hace un t-test pareado de cada una contra la mejor. `p_vs_mejor >= 0.05` significa que no es significativamente distinta de la mejor (`equivalente_a_la_mejor = sí`). Genera `results/rna/comparacion_estudios.csv`.

#### Comparar algoritmos

```bash
python src/bicimadness/comparar_algoritmos.py
```

Para cada CSV, coge la ejecución con menor MAE de cada carpeta de algoritmo de `results/`, añade la referencia "misma hora de ayer" y compara MAE, RMSE, R² y tiempo, con t-test por pares. Genera `results/comparacion_algoritmos.csv` y `results/comparacion_algoritmos_ttest.csv`.

### Interpretar los resultados

#### Métricas

- **MAE, error absoluto medio:** diferencia absoluta media entre las salidas predichas y las observadas. Cuanto menor, mejor. Un MAE de 1 significa un error absoluto medio de una salida por estación y hora.
- **RMSE, raíz del error cuadrático medio:** también se mide en salidas, pero penaliza más los errores grandes. Cuanto menor, mejor.
- **R²:** mide el ajuste respecto a predecir la media de los valores objetivo evaluados. Cuanto mayor, mejor: 1 es perfecto, 0 iguala esa referencia basada en la media y puede ser negativo. No es un porcentaje de acierto.

`resumen.csv` promedia las métricas de los pares estación–bloque. En particular, RMSE y R² son medias de puntuaciones individuales, no métricas recalculadas sobre todas las predicciones agrupadas. Cada estación tiene el mismo peso en este resumen, independientemente de su volumen de viajes.

#### Archivos generados

Cada carpeta de ejecución `results/<algoritmo>/<fecha>_<nombre>/` contiene:

| Archivo | Para qué sirve |
|---|---|
| `config.json` | Todas las opciones, fechas, commit de git y versión de Python: qué se ejecutó exactamente |
| `resumen.csv` | MAE, RMSE y R² generales, los mismos de la referencia "misma hora de ayer" y los tiempos |
| `resultados_folds.csv` | Métricas de cada estación en cada bloque |
| `tiempos.csv` | Segundos de entrenamiento y de predicción de cada bloque |
| `error_por_estacion.csv` | Métricas de cada estación promediadas entre bloques |
| `error_por_hora.csv` | MAE y RMSE por hora del día |
| `errores_por_grupo.csv` | Medias por grupo en todos los bloques y en los bloques desde el 2022-08-01 (solo en CSV con grupos, como `sel60`) |
| `predicciones.csv.gz` | Fecha, estación, bloque, valor real y predicción |

El t-test pareado compara el MAE de los mismos pares estación–bloque. Un p-valor pequeño aporta evidencia bajo los supuestos del test; no demuestra por sí solo relevancia práctica. La dependencia entre estaciones y periodos, y las comparaciones múltiples, limitan su interpretación.

### Regenerar los datos

Este paso es opcional si ya tienes los datos preparados. Los viajes originales ocupan aproximadamente 2,2 GB y se descargan por separado del portal de datos abiertos de Madrid.

Coloca los archivos siguiendo esta estructura y conserva sus nombres originales:

```text
data/raw/bicimad_2021/
    202101_movements.zip ... 202105_movements-json.zip  # JSON de enero a mayo
    trips_21_06_June.csv ... trips_21_12_December.csv   # CSV de junio a diciembre
data/raw/bicimad_2022/
    trips_22_01_January.csv ... trips_22_12_December.csv
data/raw/bicimad_2023/
    trips_23_01_January.csv
    trips_23_02_February.csv
```

Mantén disponible `data/raw/calendario_laboral.csv`. Comprueba que existe `data/processed/` antes de ejecutar los scripts de procesamiento de viajes.

#### Datos de las top 50

```bash
python src/bicimadness/preprocesamiento/preprocess.py
```

Genera `data/processed/salidas_por_hora_top50.csv` y `data/processed/estaciones_top50.csv`.

#### Datos de las 60 seleccionadas

```bash
python src/bicimadness/preprocesamiento/count_stations.py
python src/bicimadness/preprocesamiento/select_stations.py
```

El primer comando genera recuentos y ubicaciones de todas las estaciones en `data/cache/`. El segundo selecciona estaciones, entrena la primera ronda si hace falta y escribe:

- `data/processed/estaciones_seleccionadas.csv`
- `data/processed/salidas_por_hora_sel60.csv`
- Informes en `results/primera_ronda/` y `data/cache/salidas_por_hora_primera_ronda.csv`

La primera ronda entrena una red por estación para unas 200 estaciones (unos 10–20 minutos con 8 núcleos). Una red global con tantas estaciones (`--modo global`) no cabe en 8 GB de RAM.

#### Meteorología

```bash
python src/bicimadness/preprocesamiento/weather.py
```

Descarga los archivos anuales comprimidos en `data/raw/meteo/` si faltan y genera `data/raw/meteo_horario.csv`. Las descargas requieren acceso a Internet. El script cubre desde enero de 2021 hasta febrero de 2023, más allá de la última fecha de los viajes.

Después de regenerar los datos, ejecuta el experimento principal del inicio rápido.

### Solución de problemas

| Problema | Qué comprobar |
|---|---|
| `ModuleNotFoundError` | Activa `.venv` e instala las dependencias con el `python -m pip` de ese entorno. En PyCharm, selecciona el mismo intérprete. |
| Falta el CSV horario | Comprueba `data/processed/` y regenera el conjunto correspondiente si hace falta. |
| Falta la meteorología o aparece un `KeyError` de fecha meteorológica | Genera `meteo_horario.csv` y comprueba que cubre todas las horas objetivo. Un periodo personalizado necesita meteorología del mismo periodo. |
| Falta memoria durante el entrenamiento global | Usa `--jobs 1`. Reduce los bloques que se entrenan a la vez. |
| La comparación tarda demasiado | Empieza con menos `--modelos` o aumenta `--folds-cada`; utiliza todos los folds para la comparación final. |
| Los resultados antiguos no cambian tras modificar la configuración | `antiguo/compare_models.py` guarda caché de folds: usa una carpeta `--resultados` nueva. |
| Un conjunto personalizado no produce folds de test | Aporta historial suficiente para la ventana máxima, el periodo de entrenamiento y un bloque completo de test de 336 horas. |
| `ConvergenceWarning` | El optimizador ha alcanzado el límite de iteraciones. Revisa el experimento antes de basarte en sus resultados; cambiar las capas ocultas desde el comando no cambia ese límite. |

### Limitaciones y trabajo futuro

- **Meteorología de la hora objetivo:** con `--meteo actual` (por defecto) el modelo usa el tiempo observado de la hora que predice, que equivale a un pronóstico perfecto a una hora vista. `--meteo anterior` usa solo el tiempo de la hora anterior; la diferencia entre ambos mide lo que aporta un buen pronóstico.
- **Sesgo de selección de estaciones:** los grupos fácil/difícil se eligen con los bloques anteriores al 2022-08-01; solo las filas `desde_2022-08-01` de `errores_por_grupo.csv` los evalúan con semanas no usadas. La mejor configuración se elige y se informa con los mismos 29 bloques, así que su error puede ser ligeramente optimista.
- **Observaciones dependientes:** los errores estación–fold no son completamente independientes; interpreta con cautela los p-valores del t-test.
- **Cobertura histórica:** los viajes preparados terminan en febrero de 2023. Estos experimentos no establecen el rendimiento sobre la red actual.
- **Demanda frente a salidas observadas:** los viajes reflejan el uso realizado y la disponibilidad de bicicletas; no miden directamente la demanda insatisfecha cuando una estación está vacía.
- **Meteorología y resolución temporal:** una única estación meteorológica representa toda la ciudad, se imputan valores ausentes y el tratamiento de los cambios de hora simplifica la serie temporal.
- **Alcance operativo:** no se implementan llegadas, balance de bicicletas, rutas, ingestión en directo, exportación de modelos ni despliegue.

Las notas del proyecto proponen gráficos y mapas exploratorios, variables de eventos como partidos y conciertos, una comparación completa con todos los folds y un experimento específico de predicción a 24 horas. Considera estas tareas pendientes mientras no se añadan su implementación y sus resultados.

### Documentación adicional

- [Propuesta original](docs/propuesta.pdf)
- [Seguimiento del 20 de noviembre](docs/seguimiento_20nov.md): contexto y resultados preliminares; comprueba la configuración de cada ejecución antes de citarlos.
- [Índice propuesto de la memoria](docs/indice_memoria.md)

Para entender el código, empieza por `preprocesamiento/preprocess.py`; después lee `ventana()` y `bloques()` en `comun/datos.py`, `_bloque()` y `guardar()` en `comun/experimento.py`, y `entrenar()` en `rna/entrenar.py`. Continúa con `preprocesamiento/select_stations.py` y los dos scripts de comparación.

[Volver al inicio](#bicimadness) · [Read in English](#english)
