# All costs calculatinos are done here

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)
from metrics_and_costs import turnMinusOneInZeroes

# region constants
Hot, Warm = [1, 0]
hot_storage_cost, warm_storage_cost = [0.0230, 0.0125]
hot_operation_cost, warm_operation_cost = [0.0004, 0.0010]
hot_retrieval_cost, warm_retrieval_cost = [0.0000, 0.0100]

# endregion

# region functions
def get_object_cost(size_in_bytes, number_of_access, storage_type):
    size_in_GB = size_in_bytes / (1024 ** 3)
    access_as_1k = number_of_access / 1000.0
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
        return ValueError(f"Unknown storage type: {storage_type}")


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
            return get_object_cost(size_in_bytes, number_of_access, Hot)
        else:  # FP
            return get_object_cost(size_in_bytes, number_of_access, Hot)
    elif predicted == Warm:
        if number_of_access >= access_threshold:  # FN
            return get_object_cost(size_in_bytes, 1, Warm) + get_object_cost(
                size_in_bytes, number_of_access - 1, Hot
            )  # Penalty
        else:  # TN
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


# accuracy_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
#     f1_score,
#     fbeta_score,
#     confusion_matrix,
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
        predictions.append(Warm)  # Inicializa a previsão como Warm
        # Verifica se há qualquer valor de acesso acima de 0 no intervalo de treinamento
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
    
    # Calcula as métricas
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



# endregion

# print(get_predicted_cost(0, obj1_size_in_bytes, Warm))
# print(get_predicted_cost(0, obj1_size_in_bytes, Hot))
# print(get_predicted_cost(1, obj1_size_in_bytes, Warm))
# print(get_predicted_cost(1, obj1_size_in_bytes, Hot))
# print(get_predicted_cost(2, obj1_size_in_bytes, Warm))
# print(get_predicted_cost(2, obj1_size_in_bytes, Hot))
# print(get_predicted_cost(3, obj1_size_in_bytes, Warm))
# print(get_predicted_cost(3, obj1_size_in_bytes, Hot))
