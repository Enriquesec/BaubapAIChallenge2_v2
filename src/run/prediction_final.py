import pandas as pd

import sys
sys.path.append("../../")

from BaubapAIChallenge2.src.models.predict_model import predict_final

if __name__ == "__main__":
    # Cargar el conjunto de datos de evaluación en un DataFrame
    X_val = pd.read_csv('../data/raw/data_evaluation.csv')
    
    # Ruta al archivo pickle que contiene el mejor modelo entrenado
    best_model_name = "../models/freq_114.pickle"

    # Realizar predicciones en el conjunto de datos de evaluación
    y_val_prediction = predict_final(best_model_name, X_val)
    
    # Crear un DataFrame con las predicciones
    submission = pd.DataFrame({'prediction': y_val_prediction})
    
    # Guardar el DataFrame en un archivo CSV
    submission.to_csv('../data/interim/submission_final.csv', index=False)
