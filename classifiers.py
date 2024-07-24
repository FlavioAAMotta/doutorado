from utilities import normalize_data, prepare_training_data, scale_volume_bytes, filter_dataframe_with_volume, filter_dataframe
from windowCalcs import getAllWindows
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV


def get_default_classifiers(probability=False, random_state=42, n_neighbors=None):
    if n_neighbors is None:
        n_neighbors = 4

    default_clfs = {
        "SVMR": svm.SVC(kernel="rbf", gamma="auto", probability=probability, random_state=random_state),
        "SVML": svm.SVC(kernel="linear", probability=probability, random_state=random_state),
        "SVMS": svm.SVC(kernel="sigmoid", probability=probability, random_state=random_state),
        "RF": RandomForestClassifier(n_jobs=-1, random_state=random_state, n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1),
        "DCT": DecisionTreeClassifier(random_state=random_state),
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

    # volume_rescaled = normalize_data(volume_at_window)         df = pd.concat([df, volume_at_window[index_of_files_that_exists]], axis=1)
        # df = pd.concat([df, volume_at_window], axis=1)


def train_ML(train, trainLabel, clf, arq_vol_bytes, useVolOnTraining, endEvaluation, X_test):
    x_train, y_train = prepare_training_data(train, trainLabel)
    # Se eu quiser usar o volume rescalado, basta concatenar com o x_train 
    if useVolOnTraining:
        volume_at_window = arq_vol_bytes.iloc[:, endEvaluation]
        volume_normalized = normalize_data(volume_at_window)
        volume_normalized.name = 'normalized_volume'
        df_final = x_train.merge(volume_normalized, left_index=True, right_index=True)
        X_test = X_test.merge(volume_normalized, left_index=True, right_index=True)
    else:
        df_final = x_train
        # x_train = pd.concat([x_train, volume_normalized], axis=1)
    clf.fit(df_final, y_train)
    return clf, X_test

def prepare_data(time_window, window_size, steps_to_take, arq_acessos, arq_classes, arq_vol_bytes):
    window_values = getAllWindows(time_window, window_size, steps_to_take)
    initialTrain, endTrain, initialTrainLabel, endTrainLabel, initialEvaluation, endEvaluation, initialEvaluationLabel, endEvaluationLabel = window_values
    
    X_train = filter_dataframe_with_volume(arq_acessos, arq_vol_bytes,endTrain, initialTrain, endTrain)
    y_train = filter_dataframe_with_volume(arq_acessos, arq_vol_bytes, endTrain, initialTrainLabel, endTrainLabel)
    X_test = filter_dataframe(arq_acessos, arq_vol_bytes, endEvaluation, initialEvaluation, endEvaluation)
    y_test = filter_dataframe(arq_acessos, arq_vol_bytes, endEvaluation, initialEvaluationLabel, endEvaluationLabel)

    for i in range(0, X_test.shape[1]):
        X_train.rename(columns={X_train.columns[i]: i}, inplace=True)
        X_test.rename(columns={X_test.columns[i]: i}, inplace=True)

    return X_train, y_train, X_test, y_test, initialTrain, initialEvaluation, endEvaluation

def train_and_predict_for_window(time_window, clf_name, clf, window_size, steps_to_take, arq_acessos, arq_classes, arq_vol_bytes, threshold, useVolOnTraining):
    X_train, y_train, X_test, y_test, initialTrain, initialEvaluation, endEvaluation = prepare_data(time_window, window_size, steps_to_take, arq_acessos, arq_classes, arq_vol_bytes)
    clf, X_test = train_ML(X_train, y_train, clf, arq_vol_bytes, useVolOnTraining, endEvaluation, X_test)
    y_pred = predict(clf_name, clf, X_test, threshold)
    actual_class = y_test.apply(lambda row: int(row.any()), axis=1).values
    return X_test, y_pred, actual_class, initialEvaluation, endEvaluation

def predict(clf_name, clf, X_test, threshold):
    if clf_name == "HV":
        y_pred = clf.predict(X_test)
    else:
        y_pred_prob = clf.predict_proba(X_test)
        y_pred = np.where(y_pred_prob[:, 1] > threshold, 1, 0)
    return y_pred