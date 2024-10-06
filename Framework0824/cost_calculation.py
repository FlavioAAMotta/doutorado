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

    if predicted == Hot:
        costs["prediction"] += calculate_object_cost(vol_gb, acc_fut, Hot)
    else:  # Predicted Warm
        costs["prediction"] += calculate_object_cost(vol_gb, acc_fut, Warm)

def get_cost_of_classifiers(costs):
    prediction_cost = costs["prediction"]
    optimal_cost = costs["opt"]
    always_hot_cost = costs["always_H"]
    always_warm_cost = costs["always_W"]

    # Evitar divisão por zero
    if always_hot_cost == 0:
        rcs_ml = 0
        rcs_always_hot = 0
        rcs_always_warm = 0
        rcs_opt = 0
    else:
        rcs_ml = (always_hot_cost - prediction_cost) / always_hot_cost
        rcs_always_hot = 0  # (always_hot_cost - always_hot_cost) / always_hot_cost
        rcs_always_warm = (always_hot_cost - always_warm_cost) / always_hot_cost
        rcs_opt = (always_hot_cost - optimal_cost) / always_hot_cost

    return {
        "rcs ml": rcs_ml,
        "rcs always hot": rcs_always_hot,
        "rcs always warm": rcs_always_warm,
        "rcs opt": rcs_opt,
        "cost ml": prediction_cost,
        "cost always hot": always_hot_cost,
        "cost always warm": always_warm_cost,
        "cost opt": optimal_cost,
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
        # Custo se sempre armazenado como Warm
        costs["always_W"] += calculate_object_cost(volume_per_gb, total_access, Warm)

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
