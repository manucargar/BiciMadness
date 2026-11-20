# Memoria – Predicción horaria de la demanda por estación en BiciMAD (índice propuesto)

Entrega: 13 de diciembre (memoria + código). Entre corchetes, de dónde salen los datos o resultados.

## 1. Introducción
1.1 Contexto: BiciMAD y el problema de la redistribución de bicicletas
1.2 Objetivo: predecir las salidas de la hora siguiente en cada estación
1.3 Alcance y observaciones del profesor a la propuesta (LSTM, variables exógenas, selección de estaciones)
1.4 Estructura de la memoria

## 2. Estado del arte (breve)
2.1 Predicción de demanda en sistemas de bicicleta compartida
2.2 Series temporales con redes de neuronas: windowing, MLP y LSTM
2.3 Variables exógenas habituales: calendario, meteorología y eventos

## 3. Datos
3.1 Fuentes
- Viajes de la EMT (datos abiertos del Ayuntamiento), enero 2021 – febrero 2023
- Calendario laboral de Madrid (festivos)
- Meteorología horaria de Madrid-Retiro (Meteostat / AEMET)
3.2 Preprocesado [`preprocess.py`]
- Dos formatos: JSON en UTC (enero-mayo 2021) y CSV en hora local
- Filtro de viajes de menos de 2 minutos (desbloqueos fallidos) y comprobación con junio 2021
- Serie horaria por estación, con las horas sin salidas a 0
3.3 Análisis exploratorio
- Volumen por estación, patrón por hora y día de la semana, estacionalidad, efecto de la lluvia
- Porcentaje de horas con 0 salidas según la demanda de la estación
3.4 Limitaciones de los datos
- La EMT no publica viajes por estación después de febrero de 2023
- Una sola estación meteorológica para toda la ciudad

## 4. Selección de estaciones [`count_stations.py`, `select_stations.py`]
4.1 Criterio: conjunto limitado y diverso (indicación del profesor)
4.2 Candidatas: 240 de 294 estaciones con al menos 30 salidas en cada uno de los 26 meses
4.3 Los tres factores
- Demanda alta / baja
- Ubicación: céntrica (lugares emblemáticos) / afueras (lejos de Sol, separadas ≥ 1 km)
- Dificultad predictiva: primera ronda con 60 estaciones al azar → fácil / difícil según el R²
4.4 Las 60 estaciones elegidas (tabla y mapa)

## 5. Metodología
5.1 Planteamiento: entradas (ventana de 24 h + atributos de la hora a predecir) y salida (salidas de t+1)
5.2 Codificación de atributos: hora, día de la semana y mes nominales (one-hot), festivo 0/1, meteo numérica
5.3 Modelos
- Referencias: misma hora del día anterior, misma hora de la semana anterior, media histórica
- MLP por estación (32-16) frente a MLP global (64-32) con la estación como atributo nominal
- Regresión lineal Ridge con las mismas entradas que la MLP, global y por estación (contraste de linealidad) [`ridge/`]
- LSTM: multivariante (ve las 60 estaciones a la vez), global con one-hot y por estación [`lstm/`]
- Otros algoritmos: gradient boosting
5.4 Validación: sliding window (52 semanas de entrenamiento, 2 de test, 29 folds; mismos folds para todos)
5.5 Métricas (MAE, RMSE, R²) y T-test pareado sobre el MAE por (estación, fold)

## 6. Experimentos y resultados
6.1 Ajuste de la ventana y del periodo de entrenamiento: 24 h > 48 h y 6 h; 52 semanas > 26 > 8 [top 50]
6.2 Atributos de calendario: efecto de la hora y del festivo [top 50]
6.3 Red por estación frente a red global: MAE 1,425 → 1,388; sin el atributo estación, 1,409 [top 50]
6.4 Variables exógenas: meteorología (significativa pero pequeña: R² 0,298 → 0,308 en las 60) [sel60_meteo]
6.5 Resultados por grupo de estaciones: demanda, ubicación y dificultad [`analyze_groups.py`]
6.6 Comparación de algoritmos: MLP ≈ LSTM > gradient boosting > regresión > media histórica > referencias, con tiempos de entrenamiento [`compare_models.py`; rehacer con `comparar_algoritmos.py` sobre `rna/`, `ridge/` y `lstm/`]
6.7 Análisis del error por hora del día y por estación

## 7. Discusión
7.1 Qué factor explica más el error: la demanda; la ubicación apenas influye
7.2 La red global aprovecha los datos de todas las estaciones
7.3 La LSTM no mejora a la MLP con una ventana de 24 h y cuesta el doble de entrenar (comprobar con la LSTM multivariante)
7.4 Utilidad para la organización: planificar la redistribución hora a hora

## 8. Limitaciones y trabajo futuro
- Se usa la meteo observada, no la previsión (resultado optimista)
- Fácil/difícil se eligen con el mismo periodo de datos con el que luego se evalúa
- Los folds de una misma estación no son independientes (los p-valores salen algo optimistas)
- Pendiente: eventos (fútbol, conciertos), horizonte de 24 h (predicción a día futuro), ~20 estaciones para una predicción operativa

## 9. Conclusiones

## Referencias

## Anexos
- A. Cómo ejecutar el código (README)
- B. Lista de las 60 estaciones con su grupo y su motivo
- C. Tablas completas de resultados y T-test
