from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV


def get_default_classifiers(probability=True, random_state=42, n_neighbors=None):
    if n_neighbors is None:
        n_neighbors = 4

    default_clfs = {
        "SVMR": SVC(kernel="rbf", gamma="auto", probability=probability, random_state=random_state),
        "SVML": SVC(kernel="linear", probability=probability, random_state=random_state),
        "SVMS": SVC(kernel="sigmoid", probability=probability, random_state=random_state),
        "RF": RandomForestClassifier(n_jobs=-1, random_state=random_state, n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1),
        "DCT": DecisionTreeClassifier(random_state=random_state),
        "LR": LogisticRegression(n_jobs=-1),
    }

    voting_clf = VotingClassifier(
        estimators=list(default_clfs.items()),
        voting="soft",  # <--- "soft" voting uses predict_proba
        n_jobs=-1
    )

    param_grid = {
        "SVMR__C": [0.1, 1],
        "SVMR__gamma": [1, 0.1],
        "KNN__n_neighbors": [2, 3],
        "RF__n_estimators": [100, 200],
    }

    # Wrap the voting classifier in a GridSearch
    grid = GridSearchCV(voting_clf, param_grid=param_grid, cv=3, n_jobs=-1)

    return {
        **default_clfs,
        "HV": VotingClassifier(list(default_clfs.items()), voting="hard", n_jobs=-1),
        "SV": VotingClassifier(list(default_clfs.items()), voting="soft", n_jobs=-1),
        "SV-Grid": grid,
    }

def set_classifier(model_name, clfs_dict):
    """Configura o classificador com base no nome fornecido."""
    clf = clfs_dict.get(model_name)
    if clf is None:
        raise ValueError(f"Modelo {model_name} não está disponível.")
    return clf
