from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def get_window(start, size):
    """Calcula o início e o fim da janela."""
    end = start + size
    return start, end

def get_all_windows(time_window, window_size, steps_to_take):
    windows = {}

    # Primeira janela (Treinamento)
    x_train_start, x_train_end = get_window(time_window, window_size)
    windows['x_train'] = (x_train_start, x_train_end)

    # Segunda janela (Rótulo do treinamento)
    y_train_start, y_train_end = get_window(x_train_end, steps_to_take)
    windows['y_train'] = (y_train_start, y_train_end)

    # Terceira janela (Dados para teste)
    x_test_start, x_test_end = get_window(y_train_start, window_size)
    windows['x_test'] = (x_test_start, x_test_end)

    # Quarta janela (Rótulo do teste)
    y_test_start, y_test_end = get_window(x_test_end, steps_to_take)
    windows['y_test'] = (y_test_start, y_test_end)

    return windows

def getPeriodByWindow(time_window, windowSize):
    first_period_week = time_window
    last_period_week = first_period_week + windowSize
    return [first_period_week, last_period_week]

def filter_by_volume(df, df_vol, last_week_of_training):
    """
    Filtra o DataFrame df e df_vol com base no volume da semana correspondente.
    
    Args:
        df (pd.DataFrame): O DataFrame contendo os dados (x ou y).
        df_vol (pd.DataFrame): O DataFrame contendo os volumes.
        last_week_of_training (int): O índice da semana que será usada para filtrar os dados.

    Returns:
        pd.DataFrame: O DataFrame filtrado com base no volume.
    """
    volume_filter = df_vol.iloc[:, last_week_of_training] > 0
    filtered_indices = df_vol[volume_filter].index
    
    return df.loc[filtered_indices]

def extract_windows_data(df, df_vol, windows):
    """
    Extrai as janelas de dados de treinamento e teste com base nos índices das janelas.
    
    Args:
        df (pd.DataFrame): O DataFrame original contendo os dados.
        df_vol (pd.DataFrame): O DataFrame contendo os volumes.
        windows (dict): Um dicionário com os índices das janelas.

    Returns:
        dict: Um dicionário contendo os dados extraídos para x_train, y_train, x_test e y_test.
    """
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

def apply_volume_filter(data_dict, df_vol, last_week_of_training):
    """
    Aplica o filtro de volume nos dados de treinamento e teste.
    
    Args:
        data_dict (dict): Um dicionário contendo os dados para x_train, y_train, x_test e y_test.
        df_vol (pd.DataFrame): O DataFrame contendo os volumes.
        last_week_of_training (int): O índice da semana que será usada para filtrar os dados.

    Returns:
        dict: Um dicionário contendo os dados filtrados para x_train, y_train, x_test e y_test.
    """
    filtered_data = {}
    
    for key in data_dict:
        filtered_data[key] = filter_by_volume(data_dict[key], df_vol, last_week_of_training)
    
    return filtered_data

def process_training_and_test_data(df, df_vol, params):
    """
    Processa os dados de treinamento e teste aplicando as janelas e filtros necessários.
    
    Args:
        df (pd.DataFrame): O DataFrame original contendo os dados.
        df_vol (pd.DataFrame): O DataFrame contendo os volumes.
        params (dict): Parâmetros contendo 'window_size' e 'step_size'.

    """
    number_of_weeks = len(df.columns) - 1  # Ignorando a coluna de namespace
    last_week_of_training = number_of_weeks - params['window_size'] * 2 

    for first_week in range(0, last_week_of_training, params['step_size']):
        windows = get_all_windows(first_week, params['window_size'], params['step_size'])
        last_week_of_training = windows['x_train'][1]
        
        # Extrair os dados das janelas
        data_dict = extract_windows_data(df, df_vol, windows)
        
        # Aplicar o filtro de volume nos dados
        filtered_data = apply_volume_filter(data_dict, df_vol, last_week_of_training)
        
        # Se for o primeiro loop, imprimir x_train filtrado e encerrar
        if first_week == 0:
            print(filtered_data['x_train'].tail())
            break

def rename_columns(data_dict):
    """
    Renomeia as colunas dos DataFrames para garantir consistência entre x_train e x_test.
    
    Args:
        data_dict (dict): Um dicionário contendo os dados para x_train, y_train, x_test e y_test.

    Returns:
        dict: O dicionário com os dados atualizados e colunas renomeadas.
    """
    renamed_data = {}
    for key in data_dict:
        renamed_data[key] = data_dict[key].copy()
        renamed_data[key].columns = [f'week_{i+1}' for i in range(len(renamed_data[key].columns))]
    
    return renamed_data

def classify_by_access(df):
    """
    Classifica cada linha do DataFrame com base no somatório das colunas.
    
    Args:
        df (pd.DataFrame): O DataFrame contendo os dados.

    Returns:
        pd.Series: Uma série com a classificação de cada linha (1 ou 0).
    """
    return df.sum(axis=1).apply(lambda x: 1 if x >= 1 else 0)    

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Avalia o modelo utilizando várias métricas e retorna um dicionário com os resultados.
    
    Args:
        y_true (np.array): Os valores reais dos rótulos.
        y_pred (np.array): Os valores previstos pelo modelo.
        y_prob (np.array, opcional): As probabilidades preditas pelo modelo (necessário para AUC-ROC).

    Returns:
        dict: Um dicionário contendo as principais métricas de avaliação.
    """
    metrics = {}
    
    # Acurácia
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precisão
    metrics['precision'] = precision_score(y_true, y_pred)
    
    # Recall
    metrics['recall'] = recall_score(y_true, y_pred)
    
    # F1 Score
    metrics['f1_score'] = f1_score(y_true, y_pred)
    
    # AUC-ROC (se as probabilidades forem fornecidas)
    if y_prob is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    
    # Matriz de Confusão
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positive'] = tp
    metrics['true_negative'] = tn
    metrics['false_positive'] = fp
    metrics['false_negative'] = fn
    
    return metrics
