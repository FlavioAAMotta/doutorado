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
# Defina os modelos a serem utilizados
# models_to_run = ['SVMR', 'SVML', 'SVMS', 'RF', 'KNN', 'DCT', 'LR', 'HV', 'SV', 'SV-Grid', 'ONL']
models_to_run = ['SVMR', 'ONL']

# Carregar dados
df = load_acc_data('Pop2')  # Substitua 'Pop2' pelo nome apropriado
df_vol = load_vol_data('Pop2')

clfs_dict = get_default_classifiers()

number_of_weeks = len(df.columns) - 1  # Ignorando a coluna de namespace
last_week_of_training = number_of_weeks - params['window_size'] * 2 

# Inicializar uma lista para coletar os resultados
results_list = []

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

    # Criar uma lista para armazenar os resultados desta janela
    window_results = []

    # Método Online - Executado primeiro
    y_pred_online = get_online_predictions(renamed_data['x_test'])
    y_test_df_online = y_test_df.copy()
    y_test_df_online['pred'] = y_pred_online

    # Calcular as métricas
    metrics_online = evaluate_model(y_test_df_online['label'].values, y_test_df_online['pred'])

    # Calcular os custos
    costs_online = calculate_costs(y_test_df_online)

    # Salvar o custo do método Online para uso no cálculo do RCS
    cost_online_method = costs_online['cost ml']

    # Gerar resultados
    results_online = generate_results(
        y_test_df_online,
        window=first_week,
        log_id='pop',
        classifier_name='Online',
        FPPenaltyEnabled=True,
        cost_online=cost_online_method  # Include cost_online to calculate RCS
    )
    window_results.append(results_online)

    # Estratégia "Always Hot"
    y_test_df_always_hot = y_test_df.copy()
    y_test_df_always_hot['pred'] = 1  # Hot = 1

    metrics_always_hot = evaluate_model(y_test_df_always_hot['label'].values, y_test_df_always_hot['pred'])
    costs_always_hot = calculate_costs(y_test_df_always_hot, cost_online=cost_online_method)
    results_always_hot = generate_results(
        y_test_df_always_hot,
        window=first_week,
        log_id='pop',
        classifier_name='Always Hot',
        FPPenaltyEnabled=True,
        cost_online=cost_online_method
    )
    window_results.append(results_always_hot)

    # Estratégia "Always Warm"
    y_test_df_always_warm = y_test_df.copy()
    y_test_df_always_warm['pred'] = 0  # Warm = 0

    metrics_always_warm = evaluate_model(y_test_df_always_warm['label'].values, y_test_df_always_warm['pred'])
    costs_always_warm = calculate_costs(y_test_df_always_warm, cost_online=cost_online_method)
    results_always_warm = generate_results(
        y_test_df_always_warm,
        window=first_week,
        log_id='pop',
        classifier_name='Always Warm',
        FPPenaltyEnabled=True,
        cost_online=cost_online_method
    )
    window_results.append(results_always_warm)

    # Iterar sobre todos os modelos
    for model_name in models_to_run:
        if model_name == 'ONL':
            continue  # Já processado acima
        print(f"Processando modelo: {model_name} na janela: {first_week}")
        clf = set_classifier(model_name, clfs_dict)
        # Preparar 'vol_bytes' e 'acc_fut' para dados de treinamento
        vol_weeks_train = range(windows['x_train'][0] + 1, windows['x_train'][1] + 1)
        vol_bytes_train = df_vol.iloc[renamed_data['x_train'].index, vol_weeks_train].sum(axis=1)
        acc_fut_train = renamed_data['x_train'].sum(axis=1)
        
        # Treinar o modelo
        clf.fit(renamed_data['x_train'], renamed_data['y_train'])
        
        # Testar o modelo
        y_pred = clf.predict(renamed_data['x_test'])
        # Verificar se o classificador possui o método predict_proba
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(renamed_data['x_test'])[:, 1]  # Probabilidade da classe positiva
        else:
            y_prob = None  # Ou alguma alternativa
        
        # Adicionar predições ao DataFrame
        y_test_df_ml = y_test_df.copy()
        y_test_df_ml['pred'] = y_pred
        
        # Calcular as métricas
        metrics_ml = evaluate_model(y_test_df_ml['label'].values, y_test_df_ml['pred'], y_prob)
        
        # Calcular os custos
        costs_ml = calculate_costs(y_test_df_ml, cost_online=cost_online_method)
        
        # Gerar resultados
        results_ml = generate_results(
            y_test_df_ml,
            window=first_week,
            log_id='pop',  # Substitua por um identificador adequado
            classifier_name=model_name,
            FPPenaltyEnabled=True,
            cost_online=cost_online_method  # Passar o custo do método Online
        )
        # Adicionar resultados à lista
        window_results.append(results_ml)
    
    # Concatenar os resultados desta janela e adicionar à lista geral
    window_results_df = pd.concat(window_results, ignore_index=True)
    results_list.append(window_results_df)
    
# Após o loop, concatenar todos os resultados e exportar para CSV
final_results = pd.concat(results_list, ignore_index=True)
# Se o arquivo CSV já existe, anexar os novos resultados
if os.path.exists('results.csv'):
    final_results.to_csv('results.csv', mode='a', header=False, index=False)
else:
    final_results.to_csv('results.csv', index=False)

print("Resultados exportados para 'results.csv'.")
