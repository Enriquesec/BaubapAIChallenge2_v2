import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import sys
sys.path.append("../../")
from BaubapAIChallenge2.src.features.build_features import preprocess_data
import pickle

def predict_final(path_best_model, X_val):
    """
    Realiza predicciones en un conjunto de datos de evaluación utilizando el mejor modelo guardado en un archivo pickle.

    Parámetros:
    path_best_model (str): Ruta al archivo pickle que contiene el mejor modelo entrenado.
    X_val (pandas.DataFrame): Conjunto de datos de evaluación en formato DataFrame.

    Retorna:
    y_val_prediction (numpy.ndarray): Predicciones del modelo en el conjunto de datos de evaluación.
    """
    # Cargar el modelo desde el archivo pickle
    with open(path_best_model, 'rb') as file:
        best_model = pickle.load(file)

    # Cargar el conjunto de datos de entrenamiento
    df = pd.read_csv('../data/raw/training.csv')
    X = df.iloc[:,:-1].copy()
    y = df.Target

    # Dividir el conjunto de entrenamiento en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

    # Preprocesar los datos de entrenamiento y evaluación
    X_train, X_val_transformed = preprocess_data(X_train, X_val, y_train, cat_encoder_strategy='freq')

    # Seleccionar las características necesarias para la predicción
    X_val_transformed = X_val_transformed[best_model.feature_names_in_]

    # Realizar predicciones en el conjunto de datos de evaluación
    y_val_prediction = best_model.predict_proba(X_val_transformed)[:,-1]

    return y_val_prediction
