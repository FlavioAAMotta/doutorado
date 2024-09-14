# exec.py

import pandas as pd
from classifiers import get_default_classifiers, set_classifier
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
params['model'] = 'GP'  # Defina o modelo a ser utilizado ('GP' para Programação Genética)

# Carregar dados
df = load_acc_data('Pop2')  # Substitua 'pop' pelo nome apropriado
df_vol = load_vol_data('Pop2')

if params['model'] != "ONL":
    clfs_dict = get_default_classifiers()
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
        
        # Preparar 'vol_bytes' e 'acc_fut' para dados de treinamento
        vol_weeks_train = range(windows['x_train'][0] + 1, windows['x_train'][1] + 1)
        vol_bytes_train = df_vol.iloc[renamed_data['x_train'].index, vol_weeks_train].sum(axis=1)
        acc_fut_train = renamed_data['x_train'].sum(axis=1)
        
        # Treinar o modelo
        clf.fit(renamed_data['x_train'], renamed_data['y_train'], vol_bytes_train, acc_fut_train)
        
        # Testar o modelo
        y_pred = clf.predict(renamed_data['x_test'])
        y_prob = clf.predict_proba(renamed_data['x_test'])[:, 1]  # Probabilidade da classe positiva
        
        # Preparar 'vol_bytes' e 'acc_fut' para dados de teste
        vol_weeks_test = range(windows['x_test'][0] + 1, windows['x_test'][1] + 1)
        vol_bytes_test = df_vol.iloc[renamed_data['x_test'].index, vol_weeks_test].sum(axis=1)
        acc_fut_test = renamed_data['x_test'].sum(axis=1)
        
        # Preparar DataFrame para cálculo de custos
        y_test_df = pd.DataFrame({'label': renamed_data['y_test']})
        y_test_df['pred'] = y_pred
        y_test_df['vol_bytes'] = vol_bytes_test
        y_test_df['acc_fut'] = acc_fut_test
        
        # Calcular as métricas
        metrics = evaluate_model(y_test_df['label'].values, y_pred, y_prob)
        
        # Calcular os custos
        costs = calculate_costs(y_test_df)
        
        # Gerar resultados combinando métricas e custos
        results = generate_results(
            y_test_df,
            window=first_week,
            log_id='pop',  # Substitua por um identificador adequado
            classifier_name=params['model'],
            FPPenaltyEnabled=True
        )
        
        # Imprimir ou salvar os resultados
        print(results)
else:
    # Código existente para o modelo "ONL"
    pass
