from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def get_default_classifiers():
    """Retorna um dicionário com os classificadores padrão."""
    clfs = {
        'SVMR': SVC(kernel='rbf', probability=True),
        'SVML': SVC(kernel='linear', probability=True),
        'RF': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'DCT': DecisionTreeClassifier(),
        'LR': LogisticRegression(),
    }
    return clfs

def set_classifier(model_name, clfs_dict):
    """Configura o classificador com base no nome fornecido."""
    clf = clfs_dict.get(model_name)
    if clf is None:
        raise ValueError(f"Modelo {model_name} não está disponível.")
    return clf
