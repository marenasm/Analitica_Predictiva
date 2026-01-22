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
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
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
import zipfile
from typing import Tuple, List, Dict, Any

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:

    file_test = "files/input/test_data.csv.zip"
    file_train = "files/input/train_data.csv.zip"

    with zipfile.ZipFile(file_test, "r") as zipf:
        with zipf.open("test_default_of_credit_card_clients.csv") as f:
            df_test = pd.read_csv(f)

    with zipfile.ZipFile(file_train, "r") as zipf:
        with zipf.open("train_default_of_credit_card_clients.csv") as f:
            df_train = pd.read_csv(f)

    return df_train, df_test


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset según las reglas del enunciado.
    - Renombra la columna objetivo.
    - Elimina ID.
    - Elimina NAs.
    - Filtra EDUCATION y MARRIAGE != 0.
    - Agrupa EDUCATION > 4 en categoría 4 (otros).
    """
    df = df.copy()
    df = df.drop("ID", axis=1)
    df = df.rename(columns={"default payment next month": "default"})
    df = df.dropna()
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    return df


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa X (features) e y (target)."""
    X = df.drop("default", axis=1)
    y = df["default"]
    return X, y


def build_pipeline() -> Pipeline:
    """
    Crea el pipeline:
    - OneHotEncoder para variables categóricas.
    - StandardScaler para numéricas.
    - PCA.
    - SelectKBest.
    - SVC (SVM).
    """
    categories = ["SEX", "EDUCATION", "MARRIAGE"]
    numerics = [
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
            ("cat", OneHotEncoder(handle_unknown="ignore"), categories),
            ("scaler", StandardScaler(), numerics),
        ],
        remainder="passthrough",
    )

    selectkbest = SelectKBest(score_func=f_classif)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("selectkbest", selectkbest),
            ("classifier", SVC(kernel="rbf", random_state=42)),
        ]
    )

    return pipeline


def tune_hyperparameters(
    model: Pipeline,
    n_splits: int,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "balanced_accuracy",
) -> GridSearchCV:
    """
    Ajusta los hiperparámetros con GridSearchCV.
    """
    estimator = GridSearchCV(
        estimator=model,
        param_grid={
            "pca__n_components": [20, 21],
            "selectkbest__k": [12],
            "classifier__kernel": ["rbf"],
            "classifier__gamma": [0.099],
        },
        cv=n_splits,
        refit=True,
        verbose=1,
        return_train_score=False,
        scoring=scoring,
    )

    estimator.fit(x_train, y_train)
    return estimator


def compute_metrics(name: str, y_true, y_pred) -> Dict[str, Any]:
    """Calcula métricas estándar de clasificación."""
    return {
        "type": "metrics",
        "dataset": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_confusion(name: str, y_true, y_pred) -> Dict[str, Any]:
    """
    Calcula la matriz de confusión:
    true_0 / true_1 con predicted_0 / predicted_1.
    """
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": name,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }


def save_model(model: Any, path: str = "files/models/model.pkl.gz") -> None:
    """Guarda el modelo comprimido con gzip."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)


def save_metrics(
    results: List[Dict[str, Any]], path: str = "files/output/metrics.json"
) -> None:
    """Guarda métricas y matrices de confusión en formato JSON lines."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for metric in results:
            json_line = json.dumps(metric)
            f.write(json_line + "\n")


def main() -> None:

    # 1. Cargar datos
    df_train, df_test = load_data()

    # 2. Limpiar
    df_train = clean_data(df_train)
    df_test = clean_data(df_test)

    # 3. Dividir
    x_train, y_train = split(df_train)
    x_test, y_test = split(df_test)

    # 4. Pipeline + GridSearch
    model_pipeline = build_pipeline()
    model_pipeline = tune_hyperparameters(
        model_pipeline,
        n_splits=10,
        x_train=x_train,
        y_train=y_train,
        scoring="balanced_accuracy",
    )

    # 5. Guardar modelo
    save_model(model_pipeline, "files/models/model.pkl.gz")

    # 6. Predicciones
    y_train_pred = model_pipeline.predict(x_train)
    y_test_pred = model_pipeline.predict(x_test)

    # 7. Métricas y matrices
    train_metrics = compute_metrics("train", y_train, y_train_pred)
    test_metrics = compute_metrics("test", y_test, y_test_pred)

    train_matrix = compute_confusion("train", y_train, y_train_pred)
    test_matrix = compute_confusion("test", y_test, y_test_pred)

    # 8. Guardar en metrics.json
    save_metrics(
        [train_metrics, test_metrics, train_matrix, test_matrix],
        "files/output/metrics.json",
    )


if __name__ == "__main__":
    main()