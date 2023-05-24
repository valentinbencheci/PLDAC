import svm
import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, classification_report

def preprocess_test_data(dataset):
    X = []
    for data in dataset:
        spectrogram = data[0].numpy()

        X.append(np.mean(spectrogram, axis=1).flatten())

    X = np.array(X)
    y = [data[1] for data in dataset]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def evaluate_svm_model(test_path, model_path, csv="../../dataset/bird_dataset.csv"):
    test_dataset = svm.create_dataset(test_path, csv)

    X_test, y_test = preprocess_test_data(test_dataset)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    return f1, auc