# utils/cost_calculation.py

import logging
from collections import defaultdict
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)

# Definição de constantes
WARM, HOT = 0, 1
HOT_STORAGE_COST, WARM_STORAGE_COST = 0.0230, 0.0125
HOT_OPERATION_COST, WARM_OPERATION_COST = 0.0004, 0.0010
HOT_RETRIEVAL_COST, WARM_RETRIEVAL_COST = 0.0000, 0.0100

def calculate_object_cost(volume_gb, total_access, object_class):
    """Calcula o custo de armazenamento e acesso para um objeto."""
    access_1k = float(total_access) / 1000.0
    if object_class == HOT:
        cost = (
            volume_gb * HOT_STORAGE_COST
            + access_1k * HOT_OPERATION_COST
            + volume_gb * total_access * HOT_RETRIEVAL_COST
        )
    else:  # WARM
        cost = (
            volume_gb * WARM_STORAGE_COST
            + access_1k * WARM_OPERATION_COST
            + volume_gb * total_access * WARM_RETRIEVAL_COST
        )
    return cost

def calculate_threshold_access(volume_gb):
    """Calcula o limite de acesso para decidir entre HOT e WARM."""
    denominator = (
        HOT_OPERATION_COST
        - WARM_OPERATION_COST
        - volume_gb * 1000 * (WARM_RETRIEVAL_COST - HOT_RETRIEVAL_COST)
    )
    if denominator == 0:
        return float('inf')
    else:
        threshold = volume_gb * (WARM_STORAGE_COST - HOT_STORAGE_COST) / denominator
        return max(0, threshold)

def get_optimal_cost(number_of_access, volume_gb, costs):
    """Calcula o custo ótimo para um objeto."""
    access_threshold = calculate_threshold_access(volume_gb)
    if number_of_access > access_threshold:
        costs["opt"] += calculate_object_cost(volume_gb, number_of_access, HOT)
    else:
        costs["opt"] += calculate_object_cost(volume_gb, number_of_access, WARM)

def get_classifier_cost(row, costs, vol_gb, acc_fut):
    """Calcula o custo baseado na predição do classificador."""
    predicted = row["pred"]
    if predicted == HOT:
        costs["prediction"] += calculate_object_cost(vol_gb, acc_fut, HOT)
    else:
        costs["prediction"] += calculate_object_cost(vol_gb, acc_fut, WARM)

def calculate_costs(df, cost_online=None):
    """Calcula os custos para diferentes estratégias."""
    costs = defaultdict(float)
    for _, row in df.iterrows():
        volume_per_gb = float(row["vol_bytes"]) / (1024.0 ** 3)
        total_access = row["acc_fut"]
        if volume_per_gb <= 0 or total_access < 0:
            continue
    return get_cost_of_classifiers(costs, cost_online)

def get_cost_of_classifiers(costs, cost_online):
    """Calcula as métricas de custos relativos."""
    prediction_cost = costs["prediction"]
    optimal_cost = costs["opt"]
    always_hot_cost = costs["always_H"]
    always_warm_cost = costs["always_W"]
    online_cost = cost_online if cost_online is not None else prediction_cost
    if online_cost == 0:
        rcs_ml = rcs_always_hot = rcs_always_warm = rcs_opt = 0
    else:
        rcs_ml = (online_cost - prediction_cost) / online_cost
        rcs_always_hot = (online_cost - always_hot_cost) / online_cost
        rcs_always_warm = (online_cost - always_warm_cost) / online_cost
        rcs_opt = (online_cost - optimal_cost) / online_cost
    return {
        "rcs ml": rcs_ml,
        "rcs always hot": rcs_always_hot,
        "rcs always warm": rcs_always_warm,
        "rcs opt": rcs_opt,
        "cost ml": prediction_cost,
        "cost always hot": always_hot_cost,
        "cost always warm": always_warm_cost,
        "cost opt": optimal_cost,
        "cost online": online_cost,
    }

def generate_results(df, window, log_id, classifier_name, cost_online=None):
    """Gera um DataFrame com os resultados combinados de métricas e custos."""
    metrics_df = calculate_metrics(df, window)
    costs = calculate_costs(df, cost_online=cost_online)
    results = {
        "Dados": log_id,
        "Model": classifier_name,
        "window": window,
        "Qtd obj": df.shape[0],
        **metrics_df.iloc[0].to_dict(),
        **costs,
    }
    return pd.DataFrame([results])

def calculate_metrics(df, window=0):
    """Calcula as métricas de avaliação do modelo."""
    y_true, y_pred = df["label"].values, df["pred"].values
    labels = [0, 1]

    # Computar matriz de confusão com labels fixos
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    total_samples = len(y_true)
    P = fn + tp
    N = fp + tn

    # Evitar divisões por zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0

    # AUC-ROC só é calculado se houver ambas as classes presentes
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        auc_roc = roc_auc_score(y_true, y_pred)
    else:
        auc_roc = 0.0

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "P": P,
        "N": N,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }
    return pd.DataFrame(metrics, index=[window])