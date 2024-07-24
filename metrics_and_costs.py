import pandas as pd
from windowCalcs import calculate_time_total
from classifiers import get_default_classifiers

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)

Hot, Warm = [1, 0]
hot_storage_cost, warm_storage_cost = [0.0230, 0.0125]
hot_operation_cost, warm_operation_cost = [0.0004, 0.0010]
hot_retrieval_cost, warm_retrieval_cost = [0.0000, 0.0100]


def initialize_final_result():
    return pd.DataFrame(
        {
            "optimal_cost": [],
            "predicted_cost": [],
            "online_cost": [],
            "always_hot_cost": [],
            "total_objects": [],
        }
    )


def initialize_training_parameters(
    arq_acessos, window_size, steps_to_take, random_state=42, **kwargs
):
    num_weeks = arq_acessos.shape[1]
    clfs_dict = kwargs.get("clfs_dict", get_default_classifiers(True, random_state))
    time_total = calculate_time_total(num_weeks, window_size, steps_to_take)
    return [time_total, clfs_dict]


def collect_metrics_and_costs(
    access_evaluation_label,
    vol_evaluation_label,
    steps_to_take,
    actual_class,
    y_pred,
    arq_acessos,
):
    result = pd.DataFrame()
    result["optimal_cost"] = get_optimal_costs(
        access_evaluation_label, vol_evaluation_label, steps_to_take
    )
    result["always_hot"] = get_always_hot_costs(
        access_evaluation_label, vol_evaluation_label, steps_to_take
    )
    (
        result["predicted_cost"],
        current_accuracy,
        current_precision,
        current_recall,
        current_f1,
        current_fbeta,
        current_tn,
        current_fp,
        current_fn,
        current_tp,
        current_auc_roc_score,
    ) = get_classifier_costs(
        access_evaluation_label,
        vol_evaluation_label,
        steps_to_take,
        actual_class,
        y_pred,
    )
    (
        result["online_cost"],
        online_accuracy,
        online_precision,
        online_recall,
        online_f1,
        online_fbeta,
        online_tn,
        online_fp,
        online_fn,
        online_tp,
        online_auc_roc_score,
    ) = get_online_costs(
        arq_acessos,
        access_evaluation_label,
        vol_evaluation_label,
        steps_to_take,
        actual_class,
    )
    new_row = {
        "optimal_cost": result["optimal_cost"].sum(),
        "predicted_cost": result["predicted_cost"].sum(),
        "online_cost": result["online_cost"].sum(),
        "always_hot_cost": result["always_hot"].sum(),
        "accuracy": current_accuracy,
        "precision": current_precision,
        "recall": current_recall,
        "f1": current_f1,
        "fbeta": current_fbeta,
        "tn": current_tn,
        "fp": current_fp,
        "fn": current_fn,
        "tp": current_tp,
        "auc_roc_score": current_auc_roc_score,
        "online_accuracy": online_accuracy,
        "online_precision": online_precision,
        "online_recall": online_recall,
        "online_f1": online_f1,
        "online_fbeta": online_fbeta,
        "online_tn": online_tn,
        "online_fp": online_fp,
        "online_fn": online_fn,
        "online_tp": online_tp,
        "online_auc_roc_score": online_auc_roc_score,
        "total_objects": result.shape[0],
    }
    return new_row


def get_object_cost(size_in_bytes, number_of_access, storage_type):
    size_in_GB = size_in_bytes / 1024.0 / 1024.0 / 1024.0
    access_as_1k = float(number_of_access) / 1000.0  # acc prop to 1000
    if storage_type == Hot:
        return (
            size_in_GB * hot_storage_cost
            + access_as_1k * hot_operation_cost
            + size_in_GB * number_of_access * hot_retrieval_cost
        )
    elif storage_type == Warm:
        return (
            size_in_GB * warm_storage_cost
            + access_as_1k * warm_operation_cost
            + size_in_GB * number_of_access * warm_retrieval_cost
        )
    else:
        return 0


def calculate_threshold_access(size_in_bytes):
    for i in range(0, 100000):
        if get_object_cost(size_in_bytes, i, Hot) < get_object_cost(
            size_in_bytes, i, Warm
        ):
            return i


def get_optimal_cost(number_of_access, size_in_bytes):
    access_threshold = calculate_threshold_access(size_in_bytes)
    if number_of_access >= access_threshold:  # HOT
        return get_object_cost(size_in_bytes, number_of_access, Hot)
    else:  # WARM
        return get_object_cost(size_in_bytes, number_of_access, Warm)


def get_predicted_cost(number_of_access, size_in_bytes, access_threshold, predicted):
    if predicted == Hot:
        if number_of_access >= access_threshold:  # TP
            # print(f"TP: {get_object_cost(size_in_bytes, number_of_access, Hot)}")
            return get_object_cost(size_in_bytes, number_of_access, Hot)
        else:  # FP
            no_penalty_cost = get_object_cost(size_in_bytes, number_of_access, Hot)
            penalty_cost = no_penalty_cost + get_object_cost(
                size_in_bytes, number_of_access - 1, Warm
            )
            # print(f"FP no_penalty_cost: {no_penalty_cost}, penalty_cost: {penalty_cost}")
            return penalty_cost
    elif predicted == Warm:
        if number_of_access >= access_threshold:  # FN
            # print(f"FN: {get_object_cost(size_in_bytes, number_of_access, Hot) + get_object_cost(size_in_bytes, number_of_access - 1, Warm)}")
            return get_object_cost(size_in_bytes, 1, Hot) + get_object_cost(
                size_in_bytes, number_of_access - 1, Hot
            )  # Penalty
        else:  # TN
            # print(f"TN: {get_object_cost(size_in_bytes, number_of_access, Warm)}")
            return get_object_cost(size_in_bytes, number_of_access, Warm)
    else:
        return 0


def get_always_hot_costs(access, volume, steps_to_take):
    costs = []
    for i in range(len(access)):
        cost = 0.0
        for j in range(steps_to_take):
            access_threshold = calculate_threshold_access(volume.iloc[i, j])
            cost += get_predicted_cost(
                access.iloc[i, j], volume.iloc[i, j], access_threshold, Hot
            )
        costs.append(cost)
    return costs


# Method to iterate over the dataset and check which files are hot and warm
# Returns the correspondent labels for each file (labels are 1 for hot and 0 for warm)
def get_optimal_costs(access, volume, steps_to_take):
    costs = []
    for i in range(len(access)):
        hot_cost = 0.0
        warm_cost = 0.0
        for j in range(steps_to_take):
            # We accumulate the cost of the file in the window, the file is the line in the dataset
            access_threshold = calculate_threshold_access(volume.iloc[i, j])
            hot_cost += get_predicted_cost(
                access.iloc[i, j], volume.iloc[i, j], access_threshold, Hot
            )
            warm_cost += get_predicted_cost(
                access.iloc[i, j], volume.iloc[i, j], access_threshold, Warm
            )
        if hot_cost < warm_cost:
            costs.append(hot_cost)
        else:
            costs.append(warm_cost)
    return costs


def get_classifier_costs(access, volume, steps_to_take, actual, predicted):
    costs = []
    for i in range(len(access)):
        cost = 0.0
        for j in range(steps_to_take):
            access_threshold = calculate_threshold_access(volume.iloc[i, j])
            cost += get_predicted_cost(
                access.iloc[i, j], volume.iloc[i, j], access_threshold, predicted[i]
            )

        costs.append(cost)
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    fbeta = fbeta_score(actual, predicted, beta=0.5)
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    auc_roc_score = roc_auc_score(actual.reshape(-1, 1), predicted.reshape(-1, 1))
    return [
        costs,
        accuracy,
        precision,
        recall,
        f1,
        fbeta,
        tn,
        fp,
        fn,
        tp,
        auc_roc_score,
    ]


def get_online_costs(X_train, access, volume, steps_to_take, actual_class):
    costs = []
    predictions = []
    access_zeroes = turnMinusOneInZeroes(access)
    for i in range(len(access_zeroes)):
        cost = 0.0
        predictions.append(Warm)
        # Check if X_train has any value above 0, if does, then the file is hot
        if access_zeroes.iloc[i, :-1].sum() > 0:
            predictions[i] = Hot
        for j in range(steps_to_take):
            access_threshold = calculate_threshold_access(volume.iloc[i, j])
            cost += get_predicted_cost(
                access_zeroes.iloc[i, j],
                volume.iloc[i, j],
                access_threshold,
                predictions[i],
            )
        costs.append(cost)
    accuracy = accuracy_score(actual_class, predictions)
    precision = precision_score(actual_class, predictions)
    recall = recall_score(actual_class, predictions)
    f1 = f1_score(actual_class, predictions)
    fbeta = fbeta_score(actual_class, predictions, beta=0.5)
    tn, fp, fn, tp = confusion_matrix(actual_class, predictions).ravel()
    auc_roc_score = roc_auc_score(actual_class.reshape(-1, 1), predictions)
    return [
        costs,
        accuracy,
        precision,
        recall,
        f1,
        fbeta,
        tn,
        fp,
        fn,
        tp,
        auc_roc_score,
    ]

def turnMinusOneInZeroes(access):
    access_zeroes = access.copy()
    access_zeroes[access_zeroes == -1] = 0
    return access_zeroes