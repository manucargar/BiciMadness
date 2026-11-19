# BiciMadness

Herramienta de predicción de disponibilidad de bicicletas BiciMad en una estación concreta durante la hora siguiente

## Estructura

```text
.
├── data/
│   ├── raw/                 # Viajes originales, calendario y meteorología
│   └── processed/           # Series horarias y estaciones seleccionadas
├── docs/                    # Propuesta y presentación del proyecto
├── results/                 # Resultados generados (no versionados)
├── src/bicimadness/
│   ├── preprocesado.py      # Viajes brutos → serie horaria de las top 50
│   ├── rna_estaciones.py    # Una red neuronal independiente por estación
│   ├── rna_estaciones_sel60.py # Modelos para las 60 estaciones seleccionadas
│   └── errores_por_grupo.py # Métricas agrupadas de las estaciones seleccionadas
└── requirements.txt
```

Los detalles de la propuesta inicial y la presentación se encuentran en
[docs/propuesta.pdf](docs/propuesta.pdf) y [docs/presentacion.pptx](docs/presentacion.pptx).

### Preparar los datos para el LSTM

La red LSTM de las 60 estaciones usa la misma meteorología que los modelos MLP existentes:
`data/raw/meteo_horario.csv`, con temperatura y precipitación. El preprocesado crea una matriz
horaria de 60 objetivos y añade calendario cíclico y festivo:

```powershell
python src/bicimadness/lstm/preparar_lstm.py --ventana 24 --ventanas 24 48 168 --semanas 26 52
```

Genera `data/processed/lstm_model_data.npz` y su manifiesto JSON. Las secuencias contienen las
salidas de las 60 estaciones en las horas anteriores y las variables exógenas de la hora objetivo.
El manifiesto documenta las ventanas, los entrenamientos de 26/52 semanas y los 29 bloques de test
de dos semanas. Los escaladores se ajustan únicamente con cada intervalo de entrenamiento, antes
de la validación interna y del test.

### Entrenar el LSTM global

El modelo base procesa la ventana de salidas y añade temperatura, precipitación y calendario a la
representación final para predecir las próximas salidas de las 60 estaciones:

```powershell
python src/bicimadness/lstm/modelo_lstm_60.py --ventanas 24 --semanas 52 --folds 0
```

Sin `--folds`, se ejecutan los 29 bloques. Los resultados de cada ejecución se guardan en
`results/lstm_selected_60/`, junto con la configuración, el historial de entrenamiento y las
métricas MAE/RMSE/R² por fold.

Después de guardar al menos una predicción por configuración, puede resumirse la comparación:

```powershell
python src/bicimadness/lstm/analizar_lstm.py --resultados results/lstm_selected_60
```

El análisis genera métricas de referencia y LSTM por estación, hora, grupo de estación
(demanda alta/baja, céntrica/afueras, fácil/difícil) y tipo de día (entre semana/fin de semana),
además de la prueba pareada por fold y un diagnóstico de las estaciones y folds con menor/mayor
MAE.

Para una comparación reducida en tres folds representativos (0, 14 y 28), manteniendo 52 semanas
de entrenamiento, ejecutar:

```powershell
python src/bicimadness/lstm/modelo_lstm_60.py --ventanas 24 48 --ancla-folds 168 --semanas 52 --meteorologia con --folds 0 14 28 --resultados results/lstm_ablaciones_reducida
python src/bicimadness/lstm/modelo_lstm_60.py --ventanas 24 --ancla-folds 168 --semanas 52 --meteorologia sin --folds 0 14 28 --resultados results/lstm_ablaciones_reducida
python src/bicimadness/lstm/analizar_lstm.py --resultados results/lstm_ablaciones_reducida --rna-folds results/mlp_lstm_folds/resultados_folds.csv
```

La opción `sin` elimina temperatura y precipitación, pero conserva las variables de calendario.
La adaptación para comparar los mismos folds está separada; `rna_estaciones_sel60.py` conserva
el contenido original del script de referencia de un compañero.
Para ejecutar la MLP global de referencia sobre los mismos folds:

```powershell
python src/bicimadness/lstm/rna_estaciones_sel60_adaptada.py --ventanas 24 --semanas 52 --ancla-folds 168 --folds 0 14 28 --atributos hora,dia_semana,mes,festivo,temperatura,precipitacion --modelo global --jobs 1 --jobs-estaciones 1 --resultados results/mlp_lstm_folds
```

Las ejecuciones reducidas comparan ventanas 24/48 horas y el efecto de temperatura/precipitación;
la ventana de 168 horas se omite de esta corrida para limitar el tiempo y memoria del entrenamiento
en CPU. Puede añadirse ejecutando el mismo comando LSTM con `--ventanas 168 --meteorologia con`.

## Modelos disponibles

| Ejecución | Datos y enfoque | Variables |
|---|---|---|
| `rna_estaciones.py` | Top 50 estaciones por volumen de salidas; una red por estación | Salidas recientes, hora, día de la semana, mes y festivo |
| `rna_estaciones_sel60.py` | 60 estaciones seleccionadas; por defecto, una red global compartida entre estaciones | Salidas recientes, calendario, temperatura y precipitación |

El segundo programa también permite comparar redes individuales y globales
(`--modelo estacion global`), distintas ventanas, periodos de entrenamiento y conjuntos de atributos.
Ambos programas calculan MAE, RMSE y R², el test pareado de MAE entre configuraciones y
errores por hora y por estación. El análisis opcional `errores_por_grupo.py` resume los errores
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
python src/bicimadness/rna_estaciones.py

# Comparar ventanas de 24/48 horas y 26/52 semanas
python src/bicimadness/rna_estaciones.py --ventanas 24 48 --semanas 26 52

# 60 estaciones: red global con calendario y meteorología
python src/bicimadness/rna_estaciones_sel60.py

# Comparar una red por estación y una red global
python src/bicimadness/rna_estaciones_sel60.py --modelo estacion global

# Resumir por grupos las métricas de la última ejecución de 60 estaciones
python src/bicimadness/errores_por_grupo.py
```

Los resultados se guardan, respectivamente, en `results/per_station/` y
`results/selected_60/`. Las ejecuciones vuelven a generar los ficheros de salida de su
carpeta. `errores_por_grupo.py` lee `results/selected_60/error_por_estacion.csv` y escribe
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
python src/bicimadness/preprocesado.py
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
