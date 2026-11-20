# Seguimiento final (20/11) – Grupo 82, BiciMAD

Revisión de ~4-5 minutos. Se responde a lo que pide la diapositiva del próximo hito.

## 1. Qué hemos corregido desde la propuesta
- **Selección de estaciones** (lo que pidió Elías): ya no usamos las 50 con más salidas, sino 60
  estaciones en 6 grupos de 10, según tres factores: demanda (alta/baja), ubicación
  (céntrica/afueras) y dificultad predictiva (fácil/difícil, elegidas con una primera ronda sobre
  60 estaciones al azar).
- **Variables exógenas**: añadidas la temperatura y la precipitación horarias de Madrid-Retiro
  (Meteostat/AEMET) y el festivo del calendario laboral.
- **Otros algoritmos además de la MLP**: LSTM, gradient boosting, regresión lineal y tres
  referencias, todos evaluados en las mismas condiciones. Todo el trabajo está hecho en Python
  (scikit-learn y PyTorch), no en RapidMiner.

## 2. Qué podemos mostrar (flujo ejecutable)
```
viajes EMT 2021-2023 (2,2 GB) → preprocess / count_stations → select_stations (60 estaciones)
→ model_selected60 (red global + meteo) → analyze_groups → compare_models
```
- Repositorio con README: cualquiera lo ejecuta con `python src/bicimadness/model_selected60.py`.
- Datos preparados: 18.676 horas × 60 estaciones, de enero 2021 a febrero 2023.
- Decisiones justificadas con experimentos: ventana de 24 h, 1 año de entrenamiento, red global
  en vez de una por estación, atributos nominales (one-hot).

## 3. Cómo lo validamos
- Sliding window: entrena con las 52 semanas anteriores y prueba con las 2 siguientes (29 folds,
  enero 2022 – febrero 2023), los mismos para todos los modelos.
- MAE, RMSE y R², con una referencia (misma hora del día anterior) y T-test pareado.

**Resultados preliminares (60 estaciones):**

| Modelo | MAE | R² |
|---|---|---|
| Misma hora del día anterior (referencia) | 1,28 | −0,34 |
| Media histórica | 1,11 | 0,07 |
| Regresión lineal | 1,06 | 0,21 |
| Gradient boosting | 1,01 | 0,28 |
| **MLP global** | **0,97** | **0,30** |
| LSTM | 0,98 | 0,31 |

*Comparación con 10 de los 29 folds. MLP frente a todos los demás: p < 0,01.*

- La MLP global gana a la referencia en todos los grupos. La LSTM empata con la MLP y tarda el
  doble en entrenar.
- La meteo mejora poco pero de forma significativa (R² 0,298 → 0,308, p = 2·10⁻⁶).
- Por grupos, lo que más pesa es la demanda (R² 0,43 en las de mucha demanda frente a 0,17 en
  las de poca). La ubicación apenas influye: las afueras se predicen algo mejor que el centro.

## 4. Limitaciones conocidas
- Usamos la meteo observada, no la previsión: el resultado es algo optimista.
- Los datos por estación acaban en febrero de 2023 (la EMT no publica más).
- Las estaciones fáciles y difíciles se eligen con el mismo periodo con el que se evalúa.
- En estaciones con menos de 1 salida/hora casi todo es ruido (R² bajo).

## 5. Qué falta para la entrega (vídeo 27/11, defensas 2-4-9/12, memoria + código 13/12)
1. Análisis exploratorio con gráficas (patrón horario, mapa de las 60 estaciones, efecto de la lluvia).
2. Ejecutar la comparación de modelos con los 29 folds.
3. Eventos como variable exógena (partidos y conciertos), si da tiempo.
4. Predicción a día futuro: horizonte de 24 h para ~20 estaciones, como sugirió el profesor.
5. Redactar la memoria (índice en `docs/indice_memoria.md`) y preparar el vídeo.
