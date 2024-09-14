# cost_calculation.py

from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)
import pandas as pd

# region constants
Warm, Hot = 0, 1
hot_storage_cost, warm_storage_cost = 0.0230, 0.0125
hot_operation_cost, warm_operation_cost = 0.0004, 0.0010
hot_retrieval_cost, warm_retrieval_cost = 0.0000, 0.0100
# endregion

def calculate_object_cost(volume_gb, total_access, object_class):
    access_1k = float(total_access) / 1000.0  # acessos por mil
    if object_class == Hot:
        cost = (
            volume_gb * hot_storage_cost
            + access_1k * hot_operation_cost
            + volume_gb * total_access * hot_retrieval_cost
        )
    else:  # Warm
        cost = (
            volume_gb * warm_storage_cost
            + access_1k * warm_operation_cost
            + volume_gb * total_access * warm_retrieval_cost
        )
    return cost

def calculate_threshold_access(volume_gb):
    denominator = (
        hot_operation_cost
        - warm_operation_cost
        - volume_gb * 1000 * (warm_retrieval_cost - hot_retrieval_cost)
    )
    if denominator == 0:
        return float('inf')  # Evitar divisão por zero
    else:
        threshold = volume_gb * (warm_storage_cost - hot_storage_cost) / denominator
        return max(0, threshold)

def get_optimal_cost(number_of_access, volume_gb, costs):
    access_threshold = calculate_threshold_access(volume_gb)
    if number_of_access > access_threshold:  # HOT
        costs["opt"] += calculate_object_cost(volume_gb, number_of_access, Hot)
    else:  # WARM
        costs["opt"] += calculate_object_cost(volume_gb, number_of_access, Warm)

def get_classifier_cost(row, costs, vol_gb, acc_fut):
    label = row["label"]
    predicted = row["pred"]

    if label == Hot and predicted == Hot:
        costs["TP"] += calculate_object_cost(vol_gb, acc_fut, Hot)
    elif label == Hot and predicted == Warm:
        # False Negative: Deveria ser Hot, mas previu Warm
        costs["FN"] += calculate_object_cost(vol_gb, acc_fut, Warm)
    elif label == Warm and predicted == Hot:
        # False Positive: Deveria ser Warm, mas previu Hot
        costs["FP"] += calculate_object_cost(vol_gb, acc_fut, Hot)
    elif label == Warm and predicted == Warm:
        costs["TN"] += calculate_object_cost(vol_gb, acc_fut, Warm)

def get_cost_of_classifiers(costs):
    prediction_cost = costs["TP"] + costs["FN"] + costs["TN"] + costs["FP"]  # Custo total das predições
    optimal_cost = costs["opt"]
    default_cost = costs["always_H"]  # Custo se todos os objetos forem armazenados como Hot

    # Evitar divisão por zero
    if default_cost == 0:
        default_rcs = 0
        prediction_rcs = 0
    else:
        default_rcs = (default_cost - optimal_cost) / default_cost
        prediction_rcs = (default_cost - prediction_cost) / default_cost

    return {
        "rcs ml": prediction_rcs,
        "rcs opt": default_rcs,
        "cost ml": prediction_cost,
        "cost opt": optimal_cost,
        "cost all hot": default_cost,
    }

def calculate_costs(df):
    costs = defaultdict(float)  # default = 0

    for index, row in df.iterrows():
        volume_per_gb = float(row["vol_bytes"]) / (1024.0 ** 3)  # Volume em GB
        total_access = row["acc_fut"]

        # Verificar se volume e acessos são válidos
        if volume_per_gb <= 0 or total_access < 0:
            continue  # Pular objetos com dados inválidos

        # Calcular custo ótimo
        get_optimal_cost(total_access, volume_per_gb, costs)
        # Calcular custo do classificador
        get_classifier_cost(row, costs, volume_per_gb, total_access)

        # Custo se sempre armazenado como Hot
        costs["always_H"] += calculate_object_cost(volume_per_gb, total_access, Hot)

    return get_cost_of_classifiers(costs)

def calculate_metrics(df, window=0, rel=False):
    y_true, y_pred = df["label"].values, df["pred"].values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_samples = len(y_true)

    metrics = {
        "Qtd obj": total_samples,
        "Accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
        "f1_score": f1_score(y_true, y_pred) if tp + fp > 0 and tp + fn > 0 else 0.0,
        "f_beta_2": fbeta_score(y_true, y_pred, beta=2) if tp + fp > 0 and tp + fn > 0 else 0.0,
        "Precision": precision_score(y_true, y_pred) if tp + fp > 0 else 0.0,
        "Recall": recall_score(y_true, y_pred) if tp + fn > 0 else 0.0,
        "P": f"{fn + tp} ({(fn + tp) / total_samples:.2%})" if rel else (fn + tp),
        "N": f"{fp + tn} ({(fp + tn) / total_samples:.2%})" if rel else fp + tn,
        "TP": f"{tp} ({tp / total_samples:.2%})" if rel else tp,
        "FP": f"{fp} ({fp / total_samples:.2%})" if rel else fp,
        "FN": f"{fn} ({fn / total_samples:.2%})" if rel else fn,
        "TN": f"{tn} ({tn / total_samples:.2%})" if rel else tn,
    }
    return pd.DataFrame(metrics, index=[window])

def generate_results(df, window, log_id, classifier_name, FPPenaltyEnabled):
    metrics_df = calculate_metrics(df, window)
    costs = calculate_costs(df)

    results = {
        "Dados": log_id,
        "Model": classifier_name,
        "Qtd obj": df.shape[0],
        **metrics_df.iloc[0].to_dict(),
        **costs,
    }
    return pd.DataFrame(results, index=[window])
