import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from data_loader import DataLoader
from models.classifiers import get_default_classifiers, set_classifier
import os
import json
from models.user_profiles import UserProfile

# Definindo constantes da AWS
WARM, HOT = 0, 1
HOT_STORAGE_COST, WARM_STORAGE_COST = 0.0230, 0.0125
HOT_OPERATION_COST, WARM_OPERATION_COST = 0.0004, 0.0010
HOT_RETRIEVAL_COST, WARM_RETRIEVAL_COST = 0.0000, 0.0100

# Latency Model Parameters

# Probabilities
P_hit_hot = 0.7   # Probability of cache hit for HOT storage
P_hit_warm = 0.3  # Probability of cache hit for WARM storage

# Times in milliseconds
T_cache_hit = 0.5   # Average time for cache hit (ms)
T_disk_hot = 5.0    # Average disk access time for HOT storage (ms)
T_disk_warm = 20.0  # Average disk access time for WARM storage (ms)

T_proc = 0.5        # Average processing time (ms)

# System Parameters
N = 1             # Number of concurrent requests
C = 1             # System processing capacity (threads or instances)

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

# Load user profile from config
profile_config = data_loader.params.get('profile', {})
cost_weight = profile_config.get('cost_weight', 50)  # Default to 50/50 if not specified
user_profile = UserProfile(cost_weight)

def calculate_latency(predictions, access_counts, volumes):
    """
    Calculate total latency considering object volumes
    :param predictions: Array of predictions (HOT/WARM)
    :param access_counts: Array of access counts per object
    :param volumes: Array of object volumes in bytes
    :return: Total latency in milliseconds
    """
    total_latency = 0.0
    
    # Base latency parameters (in ms/MB)
    BASE_DISK_HOT = 0.1    # Base disk latency per MB for HOT
    BASE_DISK_WARM = 0.4   # Base disk latency per MB for WARM
    
    for pred, access_count, volume_bytes in zip(predictions, access_counts, volumes):
        if access_count == 0:
            continue  # No accesses, so no latency to calculate
            
        # Convert volume to MB for latency calculation
        volume_mb = volume_bytes / (1024 * 1024)
        
        if pred == HOT:
            # For HOT storage
            P_hit = P_hit_hot
            T_disk = T_disk_hot + (BASE_DISK_HOT * volume_mb)
        else:
            # For WARM storage
            P_hit = P_hit_warm
            T_disk = T_disk_warm + (BASE_DISK_WARM * volume_mb)
        
        P_miss = 1 - P_hit

        # Calculate latency components per access
        L_cache = P_hit * T_cache_hit
        L_disk = P_miss * T_disk
        L_processing = T_proc
        L_queue = (N * T_proc) / C  # Adjust N and C as needed

        # Total latency per access
        L_total_per_access = L_cache + L_disk + L_processing + L_queue

        # Total latency for all accesses of the object
        L_total_object = access_count * L_total_per_access

        total_latency += L_total_object
    
    return total_latency



# Calculo de Janelas Temporais
def get_time_windows(data, window_size, step_size):
    windows = []
    num_weeks = len(data.columns)
    total_window_size = window_size * 4  # Total size needed for train, label_train, test, label_test
    for start in range(0, num_weeks - total_window_size + 1, step_size):
        train_start = start
        train_end = train_start + window_size

        label_train_start = train_end
        label_train_end = label_train_start + window_size

        test_start = label_train_end
        test_end = test_start + window_size

        label_test_start = test_end
        label_test_end = label_test_start + window_size

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

# Penalidades
additional_fn_loss = 0.01  # Custo adicional por impacto na latência
fp_penalty = (HOT_STORAGE_COST - WARM_STORAGE_COST) + (HOT_OPERATION_COST - WARM_OPERATION_COST)
fn_penalty = WARM_RETRIEVAL_COST + (HOT_OPERATION_COST - WARM_OPERATION_COST) + additional_fn_loss

# Função para calcular o custo do modelo
def calculate_cost(predictions, actuals):
    total_cost = 0
    for pred, actual in zip(predictions, actuals):
        if pred == HOT:
            total_cost += HOT_STORAGE_COST + HOT_OPERATION_COST
            if actual == WARM:
                # Penalidade por classificar como HOT quando era WARM
                total_cost += fp_penalty
        else:
            total_cost += WARM_STORAGE_COST + WARM_OPERATION_COST
            if actual == HOT:
                # Penalidade por classificar como WARM quando era HOT
                total_cost += fn_penalty
    return total_cost

def run_analysis(window, user_profile):
    """
    Run analysis for a specific window and user profile
    
    Args:
        window: Dictionary containing window indices
        user_profile: UserProfile instance with cost/latency preferences
    
    Returns:
        Dictionary containing results for each model
    """
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
        return {}

    # Preparação dos dados
    y_train = get_label(label_train_data)
    y_test = get_label(label_test_data)
    
    if len(train_data) != len(y_train):
        print(f"Tamanho inconsistente entre treino e rótulo: {len(train_data)} vs {len(y_train)}")
        return {}

    # SMOTE e normalização
    smote = SMOTE()
    X_train = train_data.values
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(test_data.values)

    # Resultados para cada modelo
    results = {}
    
    for model_name in models_to_run:
        if model_name == 'ONL':
            y_pred = (test_data.sum(axis=1) >= 1).astype(int).values
        elif model_name == 'AHL':
            y_pred = np.ones_like(y_test)
        elif model_name == 'AWL':
            y_pred = np.zeros_like(y_test)
        else:
            model = set_classifier(model_name, classifiers)
            model.fit(X_train_scaled, y_train_bal)

            if model_name == "HV":
                y_pred = model.predict(X_test_scaled)
            else:
                y_prob = model.predict_proba(X_test_scaled)
                y_pred = (y_prob[:, 1] >= user_profile.decision_threshold).astype(int)
        
        access_counts = label_test_data.sum(axis=1)
        volumes = test_data.index.map(lambda x: df_volume.loc[x, df_volume.columns[-1]])
        
        # Calculate metrics
        results[model_name] = {
            'cost': calculate_cost(y_pred, y_test),
            'latency': calculate_latency(y_pred, access_counts, volumes),
            'oracle_cost': calculate_cost(y_test, y_test),
            'oracle_latency': calculate_latency(y_test, access_counts, volumes),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return results

# Prepare final results
final_results = {}
for window in windows:
    results = run_analysis(window, user_profile)
    for model_name, result in results.items():
        profile_key = f"{model_name}_{int(user_profile.cost_weight*100)}"
        
        # Get y_true and y_pred from confusion matrix
        cm = result['confusion_matrix']
        y_true = []
        y_pred = []
        for i in range(2):
            for j in range(2):
                y_true.extend([i] * cm[i][j])
                y_pred.extend([j] * cm[i][j])
                
        final_results[profile_key] = {
            'model_name': model_name,
            'cost_weight': user_profile.cost_weight,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'model_cost': result['cost'],
            'oracle_cost': result['oracle_cost'],
            'model_latency': result['latency'],
            'oracle_latency': result['oracle_latency'],
            'confusion_matrix': cm.tolist()
        }

# Save results to CSV with profile information
results_for_csv = []
for profile_key, results in final_results.items():
    model_name, cost_weight = profile_key.split('_')
    cm = results['confusion_matrix']
    result_row = {
        'model_name': model_name,
        'cost_weight': cost_weight,
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1'],
        'model_cost': results['model_cost'],
        'oracle_cost': results['oracle_cost'],
        'model_latency': results['model_latency']/1000,
        'oracle_latency': results['oracle_latency']/1000,
        'tn': cm[0][0],
        'fp': cm[0][1],
        'fn': cm[1][0],
        'tp': cm[1][1]
    }
    results_for_csv.append(result_row)

# Convert the results to a DataFrame
results_df = pd.DataFrame(results_for_csv)

# Save the DataFrame to CSV
output_dir = os.path.join('results', f'results_{pop_name}_{window_size}_{step_size}')
os.makedirs(output_dir, exist_ok=True)

output_csv_path = os.path.join(output_dir, 'resultados_finais.csv')
results_df.to_csv(output_csv_path, index=False)

print(f"Resultados finais salvos em {output_dir}")
print(f"Resultados finais salvos em CSV: {output_csv_path}")
