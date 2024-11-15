import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from data_loader import DataLoader
from models.classifiers import get_default_classifiers, set_classifier
import os
import datetime
import json

# Definindo constantes da AWS
WARM, HOT = 0, 1
HOT_STORAGE_COST, WARM_STORAGE_COST = 0.0230, 0.0125
HOT_OPERATION_COST, WARM_OPERATION_COST = 0.0004, 0.0010
HOT_RETRIEVAL_COST, WARM_RETRIEVAL_COST = 0.0000, 0.0100

# Carrega Dados e Parâmetros
data_loader = DataLoader(data_dir='data/', config_path='config/config.yaml')

# Inicializa parâmetros a partir do DataLoader
params = data_loader.params
window_size = params['window_size']
step_size = params['step_size']
pop_name = params['pop_name']

# Criação dos modelos a partir dos parâmetros
models_to_run = params['models_to_run']
classifiers = get_default_classifiers()

# Carrega base de acessos e volume
df_access = data_loader.load_access_data(pop_name)
df_volume = data_loader.load_volume_data(pop_name)

# Ajusta os índices dos DataFrames para o campo de objetos
df_access.set_index('NameSpace', inplace=True)
df_volume.set_index('NameSpace', inplace=True)

# Calculo de Janelas Temporais
def get_time_windows(data, window_size, step_size):
    windows = []
    num_weeks = len(data.columns)
    for start in range(0, num_weeks - window_size * 2, step_size):
        train_start, train_end = start, start + window_size
        label_train_start, label_train_end = train_end, train_end + window_size
        test_start, test_end = label_train_end, label_train_end + window_size
        label_test_start, label_test_end = test_end, test_end + window_size

        windows.append({
            'train': (train_start, train_end),
            'label_train': (label_train_start, label_train_end),
            'test': (test_start, test_end),
            'label_test': (label_test_start, label_test_end)
        })
    return windows

# Obtem as janelas temporais
windows = get_time_windows(df_access, window_size, step_size)

# Extração de Treino/Teste
def extract_data(df, start, end):
    columns = df.columns[start:end]
    return df[columns]

# Processo de Filtragem por Volume
def filter_by_volume(df, df_volume, start_week):
    start_week_column = df_volume.columns[start_week]
    valid_objects = df_volume[df_volume[start_week_column] > 0].index
    return df.loc[df.index.intersection(valid_objects)]

# Função para obter o rótulo binário baseado no somatório
def get_label(data):
    return (data.sum(axis=1) >= 1).astype(int)

# Função para calcular o custo do modelo
def calculate_cost(predictions, actuals, storage_cost, operation_cost, retrieval_cost):
    total_cost = 0
    for pred, actual in zip(predictions, actuals):
        if pred == HOT:
            total_cost += HOT_STORAGE_COST + HOT_OPERATION_COST
        else:
            total_cost += WARM_STORAGE_COST + WARM_OPERATION_COST
            if actual == HOT:
                total_cost += WARM_RETRIEVAL_COST
    return total_cost

# Inicializa acumuladores de resultados
cumulative_results = {}
for model_name in models_to_run:
    cumulative_results[model_name] = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'model_cost': [],
        'oracle_cost': [],
        'confusion_matrix': np.zeros((2, 2), dtype=int)
    }

# Treinamento e avaliação do modelo
for window in windows:
    # Extração das janelas de treinamento e teste
    train_data = extract_data(df_access, *window['train'])
    label_train_data = extract_data(df_access, *window['label_train'])
    test_data = extract_data(df_access, *window['test'])
    label_test_data = extract_data(df_access, *window['label_test'])

    # Filtragem por volume
    try:
        train_data = filter_by_volume(train_data, df_volume, window['train'][0])
        label_train_data = label_train_data.loc[train_data.index]
    except KeyError as e:
        print(e)
        continue

    # Obtendo rótulos binários para as janelas de treinamento e teste
    y_train = get_label(label_train_data)
    y_test = get_label(label_test_data)

    # Verificação de tamanho consistente entre X e y
    if len(train_data) != len(y_train):
        print(f"Tamanho inconsistente entre treino e rótulo: {len(train_data)} vs {len(y_train)}")
        continue

    # Oráculo e balanceamento de classes com SMOTE
    smote = SMOTE()
    X_train = train_data.values
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(test_data.values)

    # Treinamento do modelo para cada modelo especificado
    for model_name in models_to_run:
        if model_name == 'ONL':
            # Modelo Online: prever 1 se houver acesso na janela de leitura, caso contrário prever 0
            y_pred = (test_data.sum(axis=1) >= 1).astype(int).values
        else:
            model = set_classifier(model_name, classifiers)
            model.fit(X_train_scaled, y_train_bal)

            # Previsão
            y_pred = model.predict(X_test_scaled)

        # Avaliação do modelo
        confusion = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Cálculo do custo do modelo e custo do oráculo
        model_cost = calculate_cost(y_pred, y_test, HOT_STORAGE_COST, HOT_OPERATION_COST, HOT_RETRIEVAL_COST)
        oracle_cost = calculate_cost(y_test, y_test, HOT_STORAGE_COST, HOT_OPERATION_COST, HOT_RETRIEVAL_COST)

        # Armazena resultados acumulativos
        cumulative_results[model_name]['accuracy'].append(accuracy)
        cumulative_results[model_name]['precision'].append(precision)
        cumulative_results[model_name]['recall'].append(recall)
        cumulative_results[model_name]['f1'].append(f1)
        cumulative_results[model_name]['model_cost'].append(model_cost)
        cumulative_results[model_name]['oracle_cost'].append(oracle_cost)
        cumulative_results[model_name]['confusion_matrix'] += confusion

        # Impressão dos resultados
        print(f"Modelo: {model_name}")
        print(f"Janela {window}:")
        print(f"Matriz de Confusão:{confusion}")
        print(f"Acurácia: {accuracy:.2f}")
        print(f"Precisão: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Custo do Modelo: {model_cost:.2f}")
        print(f"Custo do Oráculo: {oracle_cost:.2f}")
        print("---------------------------------")

# Cálculo dos resultados finais acumulados
final_results = {}
for model_name in models_to_run:
    final_results[model_name] = {
        'accuracy': np.mean(cumulative_results[model_name]['accuracy']),
        'precision': np.mean(cumulative_results[model_name]['precision']),
        'recall': np.mean(cumulative_results[model_name]['recall']),
        'f1': np.mean(cumulative_results[model_name]['f1']),
        'model_cost': np.sum(cumulative_results[model_name]['model_cost']),
        'oracle_cost': np.sum(cumulative_results[model_name]['oracle_cost']),
        'confusion_matrix': cumulative_results[model_name]['confusion_matrix'].tolist()
    }

# Impressão dos resultados finais
print("\nResultados Finais Acumulativos:")
for model_name, results in final_results.items():
    print(f"Modelo: {model_name}")
    print(f"Acurácia Média: {results['accuracy']:.2f}")
    print(f"Precisão Média: {results['precision']:.2f}")
    print(f"Recall Médio: {results['recall']:.2f}")
    print(f"F1 Score Médio: {results['f1']:.2f}")
    print(f"Custo Total do Modelo: {results['model_cost']:.2f}")
    print(f"Custo Total do Oráculo: {results['oracle_cost']:.2f}")
    print(f"Matriz de Confusão Acumulada: {results['confusion_matrix']}")
    print("---------------------------------")

# Salvando os resultados finais em arquivos
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('results', f'results_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

# Salvando em formato de texto
output_txt_path = os.path.join(output_dir, 'resultados_finais.txt')
with open(output_txt_path, 'w') as file:
    file.write("Resultados Finais Acumulativos:\n")
    for model_name, results in final_results.items():
        file.write(f"Modelo: {model_name}\n")
        file.write(f"Acurácia Média: {results['accuracy']:.2f}\n")
        file.write(f"Precisão Média: {results['precision']:.2f}\n")
        file.write(f"Recall Médio: {results['recall']:.2f}\n")
        file.write(f"F1 Score Médio: {results['f1']:.2f}\n")
        file.write(f"Custo Total do Modelo: {results['model_cost']:.2f}\n")
        file.write(f"Custo Total do Oráculo: {results['oracle_cost']:.2f}\n")
        file.write(f"Matrizes de Confusão: {results['confusion_matrix']}\n")
        file.write("---------------------------------\n")

# Salvando em formato JSON
output_json_path = os.path.join(output_dir, 'resultados_finais.json')
with open(output_json_path, 'w') as file:
    json.dump(final_results, file, indent=4)

print(f"Resultados finais salvos em {output_dir}")
