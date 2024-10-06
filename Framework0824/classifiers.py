# classifiers.py

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV
from genetic_programming_classifier import GeneticProgrammingClassifier

def get_default_classifiers(probability=True, random_state=42, n_neighbors=4):

    default_clfs = {
        "SVMR": svm.SVC(kernel="rbf", gamma="auto", probability=probability, random_state=random_state),
        "SVML": svm.SVC(kernel="linear", probability=probability, random_state=random_state),
        "SVMS": svm.SVC(kernel="sigmoid", probability=probability, random_state=random_state),
        "RF": RandomForestClassifier(n_jobs=-1, random_state=random_state, n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1),
        "DCT": DecisionTreeClassifier(random_state=random_state),
        "GP": GeneticProgrammingClassifier(
            population_size=50,
            generations=100,
            max_depth=5,
            random_state=random_state
        ),
        "LR": LogisticRegression(n_jobs=-1),
    }

    voting_clf = VotingClassifier(list(default_clfs.items()), voting="soft", n_jobs=-1)
    
    param_grid = {
        "SVMR__C": [0.1, 1],
        "SVMR__gamma": [1, 0.1],
        "KNN__n_neighbors": [2, 3],
        "RF__n_estimators": [100, 200],
    }

    grid = GridSearchCV(voting_clf, param_grid=param_grid, cv=3, n_jobs=-1)

    return {
        **default_clfs,
        "HV": VotingClassifier(list(default_clfs.items()), voting="hard", n_jobs=-1),
        "SV": VotingClassifier(list(default_clfs.items()), voting="soft", n_jobs=-1),
        "SV-Grid": grid,
    }

def set_classifier(clf_key, clfs_dict):
    clf = clfs_dict.get(clf_key, None)
    if clf is None:
        raise ValueError("Unknown classifier!")
    return clf

def get_online_predictions(x_test):
    return x_test.sum(axis=1).apply(lambda x: 1 if x >= 1 else 0)
