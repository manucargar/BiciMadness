# RNA global de ejemplo: sel60

Un modelo compartido por las 60 estaciones, con capas ocultas de 64 y 32 neuronas.
Recibe las últimas 24 horas de salidas de una estación, el calendario y la meteorología
de la hora objetivo. No incluye el identificador de estación como entrada, siguiendo
el modo global por defecto del proyecto. Las predicciones se limitan a cero como mínimo
y se redondean al entero más cercano (.5 hacia arriba) antes de mostrarlas, guardarlas
y calcular todas las métricas. La red guardada conserva su salida decimal original.

## Modelo entrenado

- Objetivos de entrenamiento: desde **2021-01-02 01:00** hasta **2022-01-02 00:00**,
  ambos incluidos: 365 días, 8.760 horas y 525.600 ejemplos entre las 60 estaciones.
- El primer día disponible se reserva para construir las primeras ventanas de 24 horas.
- Se reutiliza el entrenamiento del proyecto, incluida su validación interna aleatoria
  del 10 % para parada temprana. Esa validación solo utiliza el periodo de entrenamiento.
- Evaluación externa: **2022-01-02 01:00 → 2022-02-02 01:00**, extremo final excluido.
- `modelo.joblib` contiene la red, el escalado y la configuración. `config.json` permite
  consultar los detalles y la duración del entrenamiento.
- `evaluacion_mes/predicciones.csv` contiene las predicciones y valores reales;
  `evaluacion_mes/resumen.json`, MAE, RMSE y R² globales y de la referencia «misma hora de ayer».
  Estas métricas se calculan agrupando las observaciones; RMSE y R² no son las medias
  de métricas por estación/bloque de los estudios anteriores.

El resumen incluye `wape_pct` y `wape_pct_referencia`, en porcentaje (8 significa 8 %).
WAPE = 100 × suma de errores absolutos / suma de salidas reales. Incluye todas las horas,
también aquellas con cero salidas. Solo es indefinido (`null`) cuando el volumen real total
es cero. Se calcula sobre el conjunto completo, sin promediar porcentajes por estación.

Por ejemplo, con salidas reales [10, 0, 10] y predicciones [8, 3, 15], los errores absolutos
son [2, 3, 5]. WAPE = 100 × (2 + 3 + 5) / (10 + 0 + 10) = 50 %.
El error absoluto acumulado equivale a la mitad del volumen real. No significa que se
acierte la mitad de las predicciones; los errores por exceso y por defecto no se cancelan.
Puede superar el 100 %. MAE y RMSE mantienen la interpretación en salidas por estación-hora.

## Consultar una hora

Desde la raíz del proyecto, con el entorno y los datos locales:

```bash
.venv/bin/python modeloejemplo/rna/ejemplo.py predecir --fecha "2022-01-15 18:00"
```

Solo introduces la fecha. El script recupera las 24 horas **reales** anteriores, el
calendario (hora, día de semana, mes y festivo) y la temperatura y precipitación
registradas de la hora objetivo. Muestra las 60 predicciones junto al valor real y
guarda los resultados en `consultas/`, en una carpeta identificada por el intervalo.
Repetir la misma consulta reemplaza sus informes.

Se admiten objetivos desde **2022-01-02 01:00** hasta **2023-02-18 04:00**, siempre
que los ficheros originales sigan disponibles. Se rechazan las horas del entrenamiento.
Las fechas usan las etiquetas horarias de los CSV del proyecto.

## Simular un mes de actualizaciones horarias

```bash
.venv/bin/python modeloejemplo/rna/ejemplo.py predecir --fecha "2022-01-15 18:00" --mes
```

Este ejemplo evalúa hasta el 15 de febrero a las 18:00, excluido. También puedes usar
`--horas 24` o `--horas 720` para una duración concreta. El periodo completo debe estar
dentro de los datos. Al avanzar una hora se utiliza el historial real actualizado,
sin realimentar las predicciones ni reentrenar la red. No es un pronóstico de todo el
mes disponible desde la primera hora: es una simulación de consultas sucesivas.

La meteorología `actual` es la **observada en la hora objetivo**. Esta simulación histórica
presupone conocimiento perfecto del tiempo y no mide los errores de un pronóstico
meteorológico. No consulta servicios externos.

## Volver a entrenar

```bash
.venv/bin/python modeloejemplo/rna/ejemplo.py entrenar
```

Reemplaza el modelo de ejemplo y su evaluación mensual. Requiere las dependencias del
proyecto y los CSV de sel60 y meteorología. Los artefactos generados se conservan localmente
y no se versionan. Los scripts de experimentos y sus resultados permanecen independientes.

## Comprobaciones

```bash
.venv/bin/python -m unittest discover -s modeloejemplo/rna -p 'test_*.py'
```
