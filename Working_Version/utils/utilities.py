# utils/utilities.py

import logging
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def get_window(start, size):
    """Calcula o início e o fim da janela."""
    end = start + size
    return start, end

def get_all_windows(time_window, window_size, steps_to_take):
    """Retorna um dicionário com os índices das janelas de treinamento e teste."""
    windows = {}

    # Janela de treinamento
    x_train_start, x_train_end = get_window(time_window, window_size)
    windows['x_train'] = (x_train_start, x_train_end)

    # Rótulos de treinamento
    y_train_start, y_train_end = get_window(x_train_end, steps_to_take)
    windows['y_train'] = (y_train_start, y_train_end)

    # Janela de teste
    x_test_start, x_test_end = get_window(y_train_end, window_size)
    windows['x_test'] = (x_test_start, x_test_end)

    # Rótulos de teste
    y_test_start, y_test_end = get_window(x_test_end, steps_to_take)
    windows['y_test'] = (y_test_start, y_test_end)

    return windows

def extract_windows_data(df, windows):
    """Extrai as janelas de dados de treinamento e teste."""
    x_train = df.iloc[:, windows['x_train'][0] + 1:windows['x_train'][1] + 1]
    y_train = df.iloc[:, windows['y_train'][0] + 1:windows['y_train'][1] + 1]
    x_test = df.iloc[:, windows['x_test'][0] + 1:windows['x_test'][1] + 1]
    y_test = df.iloc[:, windows['y_test'][0] + 1:windows['y_test'][1] + 1]
    
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }

def filter_by_volume(df, df_vol, last_week_of_training):
    """Filtra o DataFrame com base no volume da semana correspondente."""
    volume_filter = df_vol.iloc[:, last_week_of_training] > 0
    filtered_indices = df_vol[volume_filter].index
    return df.loc[filtered_indices]

def apply_volume_filter(data_dict, df_vol, last_week_of_training):
    """Aplica o filtro de volume nos dados de treinamento e teste."""
    filtered_data = {}
    for key in data_dict:
        filtered_data[key] = filter_by_volume(data_dict[key], df_vol, last_week_of_training)
    return filtered_data

def rename_columns(data_dict):
    """Renomeia as colunas dos DataFrames para garantir consistência."""
    renamed_data = {}
    for key in data_dict:
        renamed_data[key] = data_dict[key].copy()
        renamed_data[key].columns = [f'week_{i+1}' for i in range(len(renamed_data[key].columns))]
    return renamed_data

def classify_by_access(df):
    """Classifica cada linha do DataFrame com base no somatório das colunas."""
    return df.sum(axis=1).apply(lambda x: 1 if x >= 1 else 0)

def evaluate_model(y_true, y_pred, y_prob=None):
    """Avalia o modelo utilizando várias métricas."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['auc_roc'] = 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positive'] = tp
    metrics['true_negative'] = tn
    metrics['false_positive'] = fp
    metrics['false_negative'] = fn
    return metrics

def prepare_test_data(renamed_data, df_vol, windows):
    """Prepara os dados de teste para cálculo de custos."""
    vol_weeks_test = range(windows['x_test'][0] + 1, windows['x_test'][1] + 1)
    vol_bytes_test = df_vol.iloc[renamed_data['x_test'].index, vol_weeks_test].sum(axis=1)
    acc_fut_test = renamed_data['x_test'].sum(axis=1)
    y_test_df = pd.DataFrame({'label': renamed_data['y_test']})
    y_test_df['vol_bytes'] = vol_bytes_test
    y_test_df['acc_fut'] = acc_fut_test
    return y_test_df
