# exec.py

import pandas as pd
from classifiers import get_default_classifiers, set_classifier, get_online_predictions
from data_loader import load_acc_data, load_vol_data, load_params
from utilities import (
    get_all_windows,
    extract_windows_data,
    apply_volume_filter,
    rename_columns,
    classify_by_access,
    evaluate_model,
)
from cost_calculation import (
    calculate_costs,
    generate_results,
)
import os

# Carregar parâmetros
params = load_params()
# Defina o modelo a ser utilizado ('GP' para Programação Genética ou 'ONL' para o método online)
params['model'] = 'ONL'  # Ou 'ONL' para o método online

# Carregar dados
df = load_acc_data('Pop2')  # Substitua 'Pop2' pelo nome apropriado
df_vol = load_vol_data('Pop2')

clfs_dict = get_default_classifiers()
if params['model'] != "ONL":
    clf = set_classifier(params['model'], clfs_dict)

number_of_weeks = len(df.columns) - 1  # Ignorando a coluna de namespace
last_week_of_training = number_of_weeks - params['window_size'] * 2 

for first_week in range(0, last_week_of_training, params['step_size']):
    windows = get_all_windows(first_week, params['window_size'], params['step_size'])
    last_week_of_training = windows['x_train'][1]
    
    # Extrair os dados das janelas
    data_dict = extract_windows_data(df, df_vol, windows)
    
    # Aplicar o filtro de volume nos dados
    filtered_data = apply_volume_filter(data_dict, df_vol, last_week_of_training)
    
    # Renomear as colunas para serem consistentes
    renamed_data = rename_columns(filtered_data)
    
    # Transformar os dados de rótulo em binário
    renamed_data['y_train'] = classify_by_access(renamed_data['y_train'])
    renamed_data['y_test'] = classify_by_access(renamed_data['y_test'])
    
    # Preparar 'vol_bytes' e 'acc_fut' para dados de teste
    vol_weeks_test = range(windows['x_test'][0] + 1, windows['x_test'][1] + 1)
    vol_bytes_test = df_vol.iloc[renamed_data['x_test'].index, vol_weeks_test].sum(axis=1)
    acc_fut_test = renamed_data['x_test'].sum(axis=1)
    
    # Preparar DataFrame para cálculo de custos
    y_test_df = pd.DataFrame({'label': renamed_data['y_test']})
    y_test_df['vol_bytes'] = vol_bytes_test
    y_test_df['acc_fut'] = acc_fut_test

    if params['model'] != "ONL":
        # Preparar 'vol_bytes' e 'acc_fut' para dados de treinamento
        vol_weeks_train = range(windows['x_train'][0] + 1, windows['x_train'][1] + 1)
        vol_bytes_train = df_vol.iloc[renamed_data['x_train'].index, vol_weeks_train].sum(axis=1)
        acc_fut_train = renamed_data['x_train'].sum(axis=1)
        
        # Treinar o modelo
        clf.fit(renamed_data['x_train'], renamed_data['y_train'], vol_bytes_train, acc_fut_train)
        
        # Testar o modelo
        y_pred = clf.predict(renamed_data['x_test'])
        y_prob = clf.predict_proba(renamed_data['x_test'])[:, 1]  # Probabilidade da classe positiva
        
        # Adicionar predições ao DataFrame
        y_test_df_ml = y_test_df.copy()
        y_test_df_ml['pred'] = y_pred
        
        # Calcular as métricas
        metrics_ml = evaluate_model(y_test_df_ml['label'].values, y_test_df_ml['pred'], y_prob)
        
        # Calcular os custos
        costs_ml = calculate_costs(y_test_df_ml)
        
        # Gerar resultados
        results_ml = generate_results(
            y_test_df_ml,
            window=first_week,
            log_id='pop',  # Substitua por um identificador adequado
            classifier_name=params['model'],
            FPPenaltyEnabled=True
        )
    else:
        # Método Online
        y_pred_online = get_online_predictions(renamed_data['x_test'])
        y_test_df_online = y_test_df.copy()
        y_test_df_online['pred'] = y_pred_online
        
        # Calcular as métricas
        metrics_online = evaluate_model(y_test_df_online['label'].values, y_test_df_online['pred'])
        
        # Calcular os custos
        costs_online = calculate_costs(y_test_df_online)
        
        # Gerar resultados
        results_online = generate_results(
            y_test_df_online,
            window=first_week,
            log_id='pop',
            classifier_name='Online',
            FPPenaltyEnabled=True
        )
    
    # Estratégia "Always Hot"
    y_test_df_always_hot = y_test_df.copy()
    y_test_df_always_hot['pred'] = 1  # Hot = 1
    
    metrics_always_hot = evaluate_model(y_test_df_always_hot['label'].values, y_test_df_always_hot['pred'])
    costs_always_hot = calculate_costs(y_test_df_always_hot)
    results_always_hot = generate_results(
        y_test_df_always_hot,
        window=first_week,
        log_id='pop',
        classifier_name='Always Hot',
        FPPenaltyEnabled=True
    )
    
    # Estratégia "Always Warm"
    y_test_df_always_warm = y_test_df.copy()
    y_test_df_always_warm['pred'] = 0  # Warm = 0
    
    metrics_always_warm = evaluate_model(y_test_df_always_warm['label'].values, y_test_df_always_warm['pred'])
    costs_always_warm = calculate_costs(y_test_df_always_warm)
    results_always_warm = generate_results(
        y_test_df_always_warm,
        window=first_week,
        log_id='pop',
        classifier_name='Always Warm',
        FPPenaltyEnabled=True
    )
    
    # Coletar e imprimir ou salvar os resultados
    # Aqui você pode salvar os resultados em um arquivo ou banco de dados
    # Exemplo:
    print("Resultados para a janela:", first_week)
    if params['model'] != "ONL":
        print(results_ml)
    else:
        print(results_online)
    print(results_always_hot)
    print(results_always_warm)
