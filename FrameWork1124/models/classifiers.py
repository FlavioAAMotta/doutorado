from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_default_classifiers():
    """Retorna um dicionário com os classificadores padrão."""
    clfs = {
        'SVMR': SVC(kernel='rbf', probability=True),
        'SVML': SVC(kernel='linear', probability=True),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DCT': DecisionTreeClassifier(random_state=42),
        'LR': LogisticRegression(random_state=42),
        'GBC': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'ABC': AdaBoostClassifier(n_estimators=100, random_state=42),
        'ETC': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'XGB': XGBClassifier(n_estimators=100, random_state=42),
        'LGBM': LGBMClassifier(n_estimators=100, random_state=42)
    }
    return clfs

def set_classifier(model_name, clfs_dict):
    """Configura o classificador com base no nome fornecido."""
    clf = clfs_dict.get(model_name)
    if clf is None:
        raise ValueError(f"Modelo {model_name} não está disponível.")
    return clf
