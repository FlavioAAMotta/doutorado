import sys
import warnings
from utilities import read_files
import pandas as pd
from classifiers import (
    set_classifier,
    train_and_predict_for_window,
    filter_dataframe,
)
from metrics_and_costs import (
    initialize_final_result,
    collect_metrics_and_costs,
    initialize_training_parameters,
)


def run_train_eval(clf_name, window_size, steps_to_take, pop, threshold, useVolOnTraining, **kwargs):
    arq_acessos, arq_classes, arq_vol_bytes = read_files(pop)
    time_total, classifiers_dictionary = initialize_training_parameters(
        arq_acessos, window_size, steps_to_take, **kwargs
    )

    if clf_name != "GP":
        clf = set_classifier(clf_name, classifiers_dictionary)

    final_result = initialize_final_result()

    for time_window in time_total:
        (
            X_test,
            y_pred,
            actual_class,
            initialEvaluation,
            endEvaluation,
        ) = train_and_predict_for_window(
            time_window,
            clf_name,
            clf,
            window_size,
            steps_to_take,
            arq_acessos,
            arq_classes,
            arq_vol_bytes,
            threshold,
            useVolOnTraining
        )
        access_evaluation_label = X_test
        vol_evaluation_label = filter_dataframe(
            arq_vol_bytes,
            arq_vol_bytes,
            endEvaluation,
            initialEvaluation,
            endEvaluation,
        )
        new_row = collect_metrics_and_costs(
            access_evaluation_label,
            vol_evaluation_label,
            steps_to_take,
            actual_class,
            y_pred,
            arq_acessos,
        )
        new_row_df = pd.DataFrame([new_row])  # Converte new_row para DataFrame
        final_result = pd.concat([final_result, new_row_df], ignore_index=True)


    return final_result


def main():
    warnings.filterwarnings("ignore")
    pop = sys.argv[1]
    clf_name = sys.argv[2]
    window_size = int(sys.argv[3])
    steps_to_take = int(sys.argv[4])
    threshold = float(sys.argv[5])
    useVolOnTraining = sys.argv[6] == "True"

    final_result = run_train_eval(clf_name, window_size, steps_to_take, pop, threshold, useVolOnTraining)
    formatted_result = {
        "optimal_cost": final_result["optimal_cost"].sum(),
        "predicted_cost": final_result["predicted_cost"].sum(),
        "online_cost": final_result["online_cost"].sum(),
        "always_hot_cost": final_result["always_hot_cost"].sum(),
        "accuracy": final_result["accuracy"].mean(),
        "precision": final_result["precision"].mean(),
        "recall": final_result["recall"].mean(),
        "f1": final_result["f1"].mean(),
        "fbeta": final_result["fbeta"].mean(),
        "tn": final_result["tn"].sum(),
        "fp": final_result["fp"].sum(),
        "fn": final_result["fn"].sum(),
        "tp": final_result["tp"].sum(),
        "auc_roc_score": final_result["auc_roc_score"].mean(),
        "online_accuracy": final_result["online_accuracy"].mean(),
        "online_precision": final_result["online_precision"].mean(),
        "online_recall": final_result["online_recall"].mean(),
        "online_f1": final_result["online_f1"].mean(),
        "online_fbeta": final_result["online_fbeta"].mean(),
        "online_tn": final_result["online_tn"].sum(),
        "online_fp": final_result["online_fp"].sum(),
        "online_fn": final_result["online_fn"].sum(),
        "online_tp": final_result["online_tp"].sum(),
        "online_auc_roc_score": final_result["online_auc_roc_score"].mean(),
        "total_objects": final_result["total_objects"].sum(),
    }
    print(formatted_result)


if __name__ == "__main__":
    main()
