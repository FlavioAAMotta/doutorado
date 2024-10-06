# models/online_model.py

def get_online_predictions(x_test):
    """Gera predições usando a estratégia online."""
    # Soma as colunas e retorna 1 se a soma for maior ou igual a 1
    return (x_test.sum(axis=1) >= 1).astype(int)
