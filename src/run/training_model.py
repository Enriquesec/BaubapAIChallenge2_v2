import pandas as pd
import optuna
from xgboost.sklearn import XGBClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE

import pickle
import sys
sys.path.append("../../")

from BaubapAIChallenge2.src.features.build_features import preprocess_data

if __name__ == "__main__":
    # Cargar los datos desde un archivo CSV
    df = pd.read_csv('../data/raw/training.csv')

    # Separar las características (X) y la variable objetivo (y)
    X = df.iloc[:, :-1].copy()  # X contiene todas las columnas excepto la última
    y = df.Target  # y contiene la columna 'Target' que es la variable objetivo

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

    # Preprocesar los datos de entrenamiento y prueba
    X_transformed, X_val_transformed = preprocess_data(X_train, X_test, y_train, cat_encoder_strategy='freq')
    
    # Define la función objetivo para la optimización de hiperparámetros
    def objective(trial):
        # Define los hiperparámetros que se optimizarán
        param = {
            "verbose": 1,
            "seed":19970808,
            "objective":"binary:logistic",
            "learning_rate":trial.suggest_float("learning_rate", .15, .40, log=True),
            "n_estimators": trial.suggest_categorical("n_estimators", [100,150, 200, 300]),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
        }

        # Crea un modelo de regresión logística con los hiperparámetros definidos
        xgb_model = XGBClassifier(**param)
        steps = [('over', SMOTE(sampling_strategy='minority', random_state=42)), ('model', xgb_model)]
        steps = [('model', xgb_model)]
        pipeline = Pipeline(steps=steps)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict_proba(X_test)[:,-1]
        brier_score = brier_score_loss(y_test, predictions)
        if brier_score<0.085:
            print("../models/freq_4_{}.pickle".format(trial.number))
            with open("../models/freq_4_{}.pickle".format(trial.number), "wb") as fout:
                 pickle.dump(pipeline, fout)
        return brier_score
    
    # Crea un objeto de estudio de Optuna para la optimización
    study = optuna.create_study(direction="minimize")

    # Realiza la optimización de hiperparámetros con un número limitado de iteraciones y tiempo máximo
    study.optimize(objective, n_trials=1000,
        timeout=600,
        show_progress_bar=False)

    # Imprime información sobre los resultados de la optimización
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
