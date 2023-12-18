# BaubapAIChallenge2_v2

En este trabajo se encuentran los recursos de la participación en el hackaton de Baubap.


La idea central del procesos consiste en 4 pasos:

### Procesamiento de los datos
Se crea la clase [preprocess_data](https://github.com/Enriquesec/BaubapAIChallenge2_v2/blob/main/src/features/build_features.py) para procesar diversos pasos:
1. Imputación de nulos
2. Reducción de variables autocorrelacionadas
3. Transformación de features categoricas.
4. Eliminación de features con poca cardinalidad
5. Segmentación de categorias
6. Eliminación de features con inconsistencias en el conjunto de validación
7. Transformación de embeddings a las variables categoricas: frecuencial

### Autoencoder
Esta parte del proceso se concentra en reduccir la dimensión de las variables y centrarce el la combinación que mejor reduzca la información de las característcas. Para se considera la capa del autoencoder:
![Comparación de las distribuciones](https://github.com/Enriquesec/BaubapAIChallenge2_v2/blob/main/docs/autoencoder.png)

El tamaño de las capas se optimiza considerando optuna, se determina que con 50 features arroja buenos resultados. 

### Fine tunning del modelo
El modelo utilizado es un [Xgboost](https://github.com/Enriquesec/BaubapAIChallenge2_v2/blob/main/notebooks/esc_model_autoencoder_best_model.ipynb), considerando la optimización de hipérparametros considerando el siguiente espacio latente:
1. learning_rate: 0 a 1
2. n_estimators: 100,150, 200, 300,
3. max_depth: 1 a 5,
4. lambda: 1e-8, 1.0, log,
5. alpha1e-8, 1.0, log,
6. subsample 0.2 a 1.0),
7. colsample_bytree: 0.5 a 1.0

El mejor modelo es: [best_model](https://github.com/Enriquesec/BaubapAIChallenge2_v2/blob/main/models/205.pickle).

### Validación de distribuciones: training vs validation
Este paso valida que las distribuciones de los features numericos no sean estadísticamente si son diferentes. Para ello se ocupa las pruebas de [Mann-Whitney U test](https://github.com/Enriquesec/BaubapAIChallenge2_v2/blob/main/notebooks/modelling/esc_eda_model_v2-Copy2.ipynb)

Se encuentra que 30 variables no son consistentes en la distribución:

![Comparación de las distribuciones](https://github.com/Enriquesec/BaubapAIChallenge2_v2/blob/main/docs/histogram_incosistencia_validacion.png)