# feature_extraction_labeling.py

import numpy as np
import pandas as pd

def get_time_windows(data, window_size, step_size):
    """
    Calcula as janelas temporais para o treino e teste.
    :param data: DataFrame de entrada com as colunas de semanas.
    :param window_size: Tamanho da janela temporal.
    :param step_size: Passo de movimento da janela temporal.
    :return: Lista de dicionários representando as janelas temporais.
    """
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

def extract_data(df, start, end):
    """
    Extrai as colunas do DataFrame de acordo com o índice de início e fim fornecidos.
    :param df: DataFrame com os dados.
    :param start: índice inicial da coluna.
    :param end: índice final da coluna.
    :return: DataFrame contendo as colunas selecionadas.
    """
    columns = df.columns[start:end]
    return df[columns]

def filter_by_volume(df, df_volume, start_week):
    """
    Filtra os objetos que possuem volume maior que zero na primeira semana da janela de treinamento.
    :param df: DataFrame de dados de acesso.
    :param df_volume: DataFrame de dados de volume.
    :param start_week: Semana inicial da janela de treinamento.
    :return: DataFrame filtrado com objetos válidos.
    """
    start_week_column = df_volume.columns[start_week]
    valid_objects = df_volume[df_volume[start_week_column] > 0].index
    return df.loc[df.index.intersection(valid_objects)]

def get_label(data):
    """
    Gera um rótulo binário baseado no somatório das colunas da janela.
    :param data: DataFrame contendo os dados da janela.
    :return: Series com o rótulo binário (1 se o somatório for maior ou igual a 1, caso contrário 0).
    """
    return (data.sum(axis=1) >= 1).astype(int)

# Example usage
if __name__ == "__main__":
    # Testando funções de extração de características e rótulos
    data = pd.DataFrame(np.random.randint(0, 5, size=(10, 20)), columns=[f'week_{i}' for i in range(20)])
    volume = pd.DataFrame(np.random.randint(0, 100, size=(10, 20)), columns=[f'week_{i}' for i in range(20)])

    windows = get_time_windows(data, window_size=4, step_size=4)
    print("Janelas temporais:", windows)

    train_data = extract_data(data, *windows[0]['train'])
    print("Dados de Treinamento:", train_data)

    filtered_data = filter_by_volume(train_data, volume, windows[0]['train'][0])
    print("Dados Filtrados por Volume:", filtered_data)

    labels = get_label(train_data)
    print("Rótulos Binários:", labels)
