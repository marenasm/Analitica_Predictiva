# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import json
import gzip
import pickle
from typing import Tuple, Dict, Any, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

def load_data(csv_file: str) -> pd.DataFrame:
    """Carga un CSV comprimido en zip."""
    df = pd.read_csv(csv_file, compression="zip")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset:
    - Renombra 'default payment next month' -> 'default'
    - Elimina 'ID'
    - Filtra EDUCATION != 0 y MARRIAGE != 0
    - Agrupa EDUCATION > 4 en 4 (otros)
    """
    df = df.copy()
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns="ID", inplace=True)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return df


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa X (features) e y (target)."""
    X = df.drop(columns="default")
    y = df["default"]
    return X, y



def build_pipeline() -> Pipeline:
    """
    Crea el pipeline de clasificación:
    - OneHotEncoder para categóricas
    - StandardScaler para numéricas
    - SelectKBest
    - PCA
    - MLPClassifier
    """
    cat_features = ["SEX", "EDUCATION", "MARRIAGE"]

    num_features = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore",sparse_output=False), cat_features),
            ("scaler", StandardScaler(with_mean=True, with_std=True), num_features),
        ],
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("pca", PCA()),
            ("classifier", MLPClassifier(max_iter=15000, random_state=17)),
        ]
    )

    return pipeline


def tune_hyperparameters(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "balanced_accuracy",
) -> GridSearchCV:
    """
    Configura y ejecuta GridSearchCV sobre el pipeline.
    """
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "pca__n_components": [None],
            "feature_selection__k": [20],
            "classifier__hidden_layer_sizes": [(50, 30, 40, 60)],
            "classifier__alpha": [0.26],
            "classifier__learning_rate_init": [0.001],
        },
        cv=10,
        scoring=scoring,
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(x_train, y_train)
    return grid_search



def compute_metrics(
    name: str, y_true, y_pred
) -> Dict[str, Any]:
    """Calcula las métricas"""
    precision = round(precision_score(y_true, y_pred), 4)
    balanced_acc = round(balanced_accuracy_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    recall = round(recall_score(y_true, y_pred), 4)

    metrics = {
        "type": "metrics",
        "dataset": name,
        "precision": precision,
        "balanced_accuracy": balanced_acc,
        "recall": recall,
        "f1_score": f1,
    }
    return metrics


def compute_confusion(
    name: str, y_true, y_pred
) -> Dict[str, Any]:
    """Devuelve la matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }



def save_model(estimator, path: str = "files/models/model.pkl.gz") -> None:
    """Guarda el modelo comprimido con gzip."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)


def save_metrics(results: List[Dict[str, Any]], path: str = "files/output/metrics.json") -> None:
    """Guarda métricas y matrices en formato JSON lines."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        for r in results:
            file.write(json.dumps(r) + "\n")



def main() -> None:
    """Ejecuta todo el flujo"""
    os.makedirs("files/output", exist_ok=True)

    # 1. Cargar y limpiar datos
    df_train = clean_data(load_data("files/input/train_data.csv.zip"))
    df_test = clean_data(load_data("files/input/test_data.csv.zip"))

    # 2. Dividir en X, y
    x_train, y_train = split(df_train)
    x_test, y_test = split(df_test)

    # 3. Pipeline
    pipeline = build_pipeline()

    # 4. GridSearch + entrenamiento
    estimator = tune_hyperparameters(pipeline, x_train, y_train, scoring="balanced_accuracy")

    # 5. Predicciones
    y_pred_train = estimator.predict(x_train)
    y_pred_test = estimator.predict(x_test)

    # 6. Métricas
    metrics_train = compute_metrics("train", y_train, y_pred_train)
    metrics_test = compute_metrics("test", y_test, y_pred_test)

    # 7. Matrices de confusión
    cm_train = compute_confusion("train", y_train, y_pred_train)
    cm_test = compute_confusion("test", y_test, y_pred_test)

    # 8. Guardar métricas y modelo
    save_metrics([metrics_train, metrics_test, cm_train, cm_test], "files/output/metrics.json")
    save_model(estimator, "files/models/model.pkl.gz")


if __name__ == "__main__":
    main()