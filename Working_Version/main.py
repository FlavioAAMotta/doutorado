# main.py

import logging
import pandas as pd
import os
from datetime import datetime
from utils.data_loader import DataLoader
from utils.utilities import (
    get_all_windows,
    extract_windows_data,
    apply_volume_filter,
    rename_columns,
    classify_by_access,
    prepare_test_data
)
from utils.cost_calculation import calculate_costs, generate_results
from models.classifiers import get_default_classifiers, set_classifier
from models.online_model import get_online_predictions
from utils.logging_config import setup_logging

def compute_overall_metrics(results_list, filename):
    """Computa métricas globais acumuladas e salva em um arquivo CSV."""
    final_results = pd.concat(results_list, ignore_index=True)

    # Agrupar por modelo e somar TP, FP, FN, TN
    grouped = final_results.groupby('Model').agg({
        'TP': 'sum',
        'FP': 'sum',
        'FN': 'sum',
        'TN': 'sum',
        'cost ml': 'sum',
        'Qtd obj': 'sum',
    }).reset_index()

    # Computar métricas globais
    overall_metrics = []
    for _, row in grouped.iterrows():
        model_name = row['Model']
        TP = row['TP']
        FP = row['FP']
        FN = row['FN']
        TN = row['TN']
        total_samples = TP + FP + FN + TN

        # Evitar divisões por zero
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (TP + TN) / total_samples if total_samples > 0 else 0.0

        # Custos
        cost_ml = row['cost ml']

        overall_metrics.append({
            'Model': model_name,
            'Total Samples': total_samples,
            'cost ml': cost_ml,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'f1_score': f1,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
        })

    overall_df = pd.DataFrame(overall_metrics)
    overall_df.to_csv(filename, index=False)
    logging.info(f"Métricas globais salvas em {filename}.")


def main():
    setup_logging()
    logging.info("Iniciando o processamento.")

    # Criar pasta de resultados com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join('results', timestamp)
    os.makedirs(results_folder, exist_ok=True)

    # Carregamento de dados e parâmetros
    data_loader = DataLoader()
    params = data_loader.params
    models_to_run = params['models_to_run']
    pop_name = params['pop_name']

    df = data_loader.load_access_data(pop_name)
    df_vol = data_loader.load_volume_data(pop_name)

    clfs_dict = get_default_classifiers()
    number_of_weeks = len(df.columns) - 1
    last_week_of_training = number_of_weeks - params['window_size'] * 2 

    results_list = []

    for first_week in range(0, last_week_of_training, params['step_size']):
        logging.info(f"Processando janela iniciando na semana {first_week}")
        windows = get_all_windows(first_week, params['window_size'], params['step_size'])
        last_week_of_training = windows['x_train'][1]

        # Extrair e processar dados
        data_dict = extract_windows_data(df, windows)
        filtered_data = apply_volume_filter(data_dict, df_vol, last_week_of_training)
        renamed_data = rename_columns(filtered_data)
        renamed_data['y_train'] = classify_by_access(renamed_data['y_train'])
        renamed_data['y_test'] = classify_by_access(renamed_data['y_test'])

        # Preparação dos dados de teste
        y_test_df = prepare_test_data(renamed_data, df_vol, windows)

        # Processar modelos
        window_results = process_models(models_to_run, clfs_dict, renamed_data, y_test_df, first_week)

        # Coletar resultados
        results_list.extend(window_results)

    # Salvar resultados por janela
    per_window_results_file = os.path.join(results_folder, f'results_per_window_{timestamp}.csv')
    save_results(results_list, per_window_results_file)
    logging.info(f"Resultados por janela salvos em '{per_window_results_file}'.")

    # Calcular e salvar métricas globais
    overall_metrics_file = os.path.join(results_folder, f'overall_metrics_{timestamp}.csv')
    compute_overall_metrics(results_list, overall_metrics_file)
    logging.info(f"Métricas globais salvas em '{overall_metrics_file}'.")

    logging.info(f"Processamento concluído. Resultados disponíveis na pasta '{results_folder}'.")


def process_models(models_to_run, clfs_dict, renamed_data, y_test_df, window):
    """Processa todos os modelos especificados e retorna os resultados."""
    window_results = []

    # Método Online
    y_pred_online = get_online_predictions(renamed_data['x_test'])
    y_test_df_online = y_test_df.copy()
    y_test_df_online['pred'] = y_pred_online

    # Calcular custos e métricas
    cost_online_method = calculate_costs(y_test_df_online)['cost ml']
    results_online = generate_results(
        y_test_df_online,
        window=window,
        log_id='pop',
        classifier_name='Online',
        cost_online=cost_online_method
    )
    window_results.append(results_online)

    # Estratégia "Always Hot"
    y_test_df_always_hot = y_test_df.copy()
    y_test_df_always_hot['pred'] = 1  # Hot = 1
    results_always_hot = generate_results(
        y_test_df_always_hot,
        window=window,
        log_id='pop',
        classifier_name='Always Hot',
        cost_online=cost_online_method
    )
    window_results.append(results_always_hot)

    # Estratégia "Always Warm"
    y_test_df_always_warm = y_test_df.copy()
    y_test_df_always_warm['pred'] = 0  # Warm = 0
    results_always_warm = generate_results(
        y_test_df_always_warm,
        window=window,
        log_id='pop',
        classifier_name='Always Warm',
        cost_online=cost_online_method
    )
    window_results.append(results_always_warm)

    # Iterar sobre todos os modelos
    for model_name in models_to_run:
        if model_name == 'ONL':
            continue  # Já processado acima
        logging.info(f"Processando modelo: {model_name} na janela: {window}")
        clf = set_classifier(model_name, clfs_dict)

        # Treinar o modelo
        clf.fit(renamed_data['x_train'], renamed_data['y_train'])

        # Testar o modelo
        y_pred = clf.predict(renamed_data['x_test'])
        y_prob = clf.predict_proba(renamed_data['x_test'])[:, 1] if hasattr(clf, 'predict_proba') else None

        # Adicionar predições ao DataFrame
        y_test_df_ml = y_test_df.copy()
        y_test_df_ml['pred'] = y_pred

        # Gerar resultados
        results_ml = generate_results(
            y_test_df_ml,
            window=window,
            log_id='pop',
            classifier_name=model_name,
            cost_online=cost_online_method
        )
        window_results.append(results_ml)

    return window_results

def save_results(results_list, filename):
    """Salva os resultados em um arquivo CSV."""
    final_results = pd.concat(results_list, ignore_index=True)
    final_results.to_csv(filename, index=False)
    logging.info(f"Resultados salvos em {filename}.")


if __name__ == '__main__':
    main()
