import os
import torch
import joblib
import librosa
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

def preprocess_audio_model(audio_path):
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    audio = audio.astype(np.float32)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram)

    target_width = 150
    if log_spectrogram.shape[1] < target_width:
        pad_width = target_width - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')

    return torch.tensor(log_spectrogram)

def create_dataset(src_path=['../dataset/train/pre_traitement_1/BirdVoxDCASE20k/', '../dataset/train/pre_traitement_1/ff1010bird/', '../dataset/train/pre_traitement_1/warblrb10k/'], csv="../../dataset/bird_dataset.csv"):
    dataset = []
    df = pd.read_csv(csv)

    for i in range(len(src_path)):
        files = os.listdir(src_path[i])

        for file_i in tqdm(range(len(files))):
                spectrogram = preprocess_audio_model(src_path[i] + files[file_i])

                target_item_id = files[file_i].split('.')[0]
                filtered_df = df[df['itemid'] == target_item_id]
                dataset.append((spectrogram.unsqueeze(0), filtered_df['hasbird'].iloc[0]))

    return dataset


dataset = create_dataset(['../dataset/train/pre_traitement_2/BirdVoxDCASE20k/', '../dataset/train/pre_traitement_2/ff1010bird/', '../dataset/train/pre_traitement_2/warblrb10k/'])
desired_shape = (128, 862)


X = []
for data in dataset:
    spectrogram = data[0].numpy()
    print("Original shape:", spectrogram.shape)
    
    if spectrogram.shape[1] < desired_shape[0]:
        spectrogram = np.pad(spectrogram, ((0, 0), (0, desired_shape[0] - spectrogram.shape[1]), (0, 0)))
    elif spectrogram.shape[1] > desired_shape[0]:
        spectrogram = spectrogram[:, :desired_shape[0], :]
    if spectrogram.shape[2] < desired_shape[1]:
        spectrogram = np.pad(spectrogram, ((0, 0), (0, 0), (0, desired_shape[1] - spectrogram.shape[2])))
    elif spectrogram.shape[2] > desired_shape[1]:
        spectrogram = spectrogram[:, :, :desired_shape[1]]
    X.append(np.mean(spectrogram, axis=1).flatten())
    
X = np.array(X)

y = [data[1] for data in dataset]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1],
    'kernel': ['linear', 'sigmoid'],
    'degree': [2, 3],
    'gamma': ['scale']
}

param_combinations = list(itertools.product(*param_grid.values()))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_score = 0
best_params = None

for i, params in enumerate(param_combinations):
    param_dict = dict(zip(param_grid.keys(), params))
    print(f"Training model {i+1}/{len(param_combinations)}: {param_dict}")
    model = svm.SVC(**param_dict)
    
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_score = np.mean(scores)
    print(f"Cross-validation score: {mean_score}")

    if mean_score > best_score:
        best_score = mean_score
        best_params = param_dict

    model.fit(X_train, y_train)
    model_name = f"audio_classifier_model_3_tr2_{params[0]}_{params[1]}_{params[2]}_{params[3]}.pkl"
    joblib.dump(model, model_name)
    print(f"Model saved as {model_name}")

print(f"Best parameters: {best_params}")