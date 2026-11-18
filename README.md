# BiciMadness

Proyecto de la práctica final de **Inteligencia Artificial en las Organizaciones**
(UC3M, curso 2026/27). Estudia la predicción de las salidas de bicicletas por estación
de BiciMAD para anticipar la demanda y ayudar a planificar la redistribución de bicicletas.

El objetivo de cada modelo es estimar cuántas bicicletas saldrán de una estación durante
la hora siguiente. Para ello, se convierten los viajes históricos de la EMT en series
horarias y se entrenan perceptrones multicapa (MLP) con ventanas de las horas anteriores
y variables de calendario. La evaluación respeta el orden temporal mediante validación
de ventana deslizante (*sliding window*) y compara los resultados con una referencia que
repite las salidas de la misma hora del día anterior.

## Estructura

```text
.
├── data/
│   ├── raw/                 # Viajes originales, calendario y meteorología
│   └── processed/           # Series horarias y estaciones seleccionadas
├── docs/                    # Propuesta y presentación del proyecto
├── results/                 # Resultados generados (no versionados)
├── src/bicimadness/
│   ├── preprocess.py        # Viajes brutos → serie horaria de las top 50
│   ├── model_per_station.py # Una red neuronal independiente por estación
│   ├── model_selected60.py  # Modelos para las 60 estaciones seleccionadas
│   └── analyze_groups.py    # Métricas agrupadas de las estaciones seleccionadas
└── requirements.txt
```

Los detalles de la propuesta inicial y la presentación se encuentran en
[docs/propuesta.pdf](docs/propuesta.pdf) y [docs/presentacion.pptx](docs/presentacion.pptx).

## Modelos disponibles

| Ejecución | Datos y enfoque | Variables |
|---|---|---|
| `model_per_station.py` | Top 50 estaciones por volumen de salidas; una red por estación | Salidas recientes, hora, día de la semana, mes y festivo |
| `model_selected60.py` | 60 estaciones seleccionadas; por defecto, una red global compartida entre estaciones | Salidas recientes, calendario, temperatura y precipitación |

El segundo programa también permite comparar redes individuales y globales
(`--modelo estacion global`), distintas ventanas, periodos de entrenamiento y conjuntos de atributos.
Ambos programas calculan MAE, RMSE y R², el test pareado de MAE entre configuraciones y
errores por hora y por estación. El análisis opcional `analyze_groups.py` resume los errores
por grupo de estaciones (demanda, ubicación y dificultad).

## Requisitos e instalación

Se requiere Python 3.10 o superior. Desde la raíz del repositorio:

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecución

Ejecuta los comandos desde la raíz del repositorio. Los datos procesados para ambos
experimentos ya están en `data/processed`, así que no hace falta regenerarlos para entrenar.

```bash
# Top 50: red por estación, ventana de 24 horas y 52 semanas de entrenamiento
python src/bicimadness/model_per_station.py

# Comparar ventanas de 24/48 horas y 26/52 semanas
python src/bicimadness/model_per_station.py --ventanas 24 48 --semanas 26 52

# 60 estaciones: red global con calendario y meteorología
python src/bicimadness/model_selected60.py

# Comparar una red por estación y una red global
python src/bicimadness/model_selected60.py --modelo estacion global

# Resumir por grupos las métricas de la última ejecución de 60 estaciones
python src/bicimadness/analyze_groups.py
```

Los resultados se guardan, respectivamente, en `results/per_station/` y
`results/selected_60/`. Las ejecuciones vuelven a generar los ficheros de salida de su
carpeta. `analyze_groups.py` lee `results/selected_60/error_por_estacion.csv` y escribe
`errores_por_grupo.csv` en esa misma carpeta. Se puede seleccionar otra carpeta con
`--resultados RUTA`; las rutas relativas se interpretan desde la raíz del repositorio.

## Regenerar los datos top 50

El preprocesado combina los ficheros mensuales originales de BiciMAD con el calendario
laboral de Madrid. Los viajes originales se descargan del
[portal de datos abiertos del Ayuntamiento de Madrid](https://datos.madrid.es/dataset/900034-0-bicimad-viajes-estaciones)
y deben colocarse en:

```text
data/raw/bicimad_2021/
data/raw/bicimad_2022/
data/raw/bicimad_2023/
```

Los datos originales ocupan varios gigabytes y no se incluyen en el repositorio. Con las
carpetas anteriores disponibles, ejecuta:

```bash
python src/bicimadness/preprocess.py
```

El script escribe `salidas_por_hora_top50.csv` y `estaciones_top50.csv` en
`data/processed/`. Solo cuenta viajes de al menos dos minutos; convierte a hora local de
Madrid los JSON de enero a mayo de 2021, que están en UTC. El calendario laboral procede
de `data/raw/calendario_laboral.csv`. El periodo disponible por estación abarca del
1 de enero de 2021 al 18 de febrero de 2023.

## Datos

- `data/processed/salidas_por_hora_top50.csv`: salidas horarias de las 50 estaciones con
  más viajes.
- `data/processed/salidas_por_hora_sel60.csv`: salidas horarias de las 60 estaciones
  seleccionadas para el experimento global.
- `data/processed/estaciones_seleccionadas.csv`: selección y grupos usados para el análisis
  de errores.
- `data/raw/meteo_horario.csv`: temperatura y precipitación horarias usadas por el modelo
  de 60 estaciones.
- `data/raw/Clima2021_horaria.csv`, `Clima2022_horaria.csv` y `Clima2023_horaria.csv`:
  ficheros meteorológicos de origen.

El preprocesado incluido regenera la serie top 50. La serie de 60 estaciones y su selección
son entradas preparadas del experimento y no se generan con ese script.
