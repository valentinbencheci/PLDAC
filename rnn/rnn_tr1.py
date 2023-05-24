import os
import torch
import librosa
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split

def preprocess_audio_model(audio_path):
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    audio = audio.astype(np.float32)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram)

    target_width = 862 
    if log_spectrogram.shape[1] < target_width:
        pad_width = target_width - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    elif log_spectrogram.shape[1] > target_width:
        log_spectrogram = log_spectrogram[:, :target_width]

    return torch.tensor(log_spectrogram)

def create_dataset(src_path=['../../dataset/train/pre_traitement_2/warblrb10k/'], csv="../../dataset/bird_dataset.csv"):
    dataset = []
    df = pd.read_csv(csv)

    for i in range(len(src_path)):
        files = os.listdir(src_path[i])

        for file_i in tqdm(range(len(files))):
            audio_path = os.path.join(src_path[i], files[file_i])
            spectrogram = preprocess_audio_model(audio_path)

            target_item_id = files[file_i].split('.')[0]
            filtered_df = df[df['itemid'] == target_item_id]
            dataset.append((spectrogram.unsqueeze(0), filtered_df['hasbird'].iloc[0]))

    return dataset

# Define the RNN architecture
class AudioClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes=2, hidden_size=64, num_layers=2, dropout=0.3):
        super(AudioClassificationModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Modèle avec une couche de sortie supplémentaire et une activation ReLU    
class AudioClassificationModelDense(nn.Module):
    def __init__(self, input_size, num_classes=2, hidden_size=64, num_layers=2, dropout=0.3):
        super(AudioClassificationModelDense, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  
        x, _ = self.lstm(x)
        x = self.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x

# Modèle avec un GRU au lieu d'un LSTM
class AudioClassificationModelGRU(nn.Module):
    def __init__(self, input_size, num_classes=2, hidden_size=64, num_layers=2, dropout=0.3):
        super(AudioClassificationModelGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x

# Modèle avec une Bidirectional LSTM
class AudioClassificationModelBiLSTM(nn.Module):
    def __init__(self, input_size, num_classes=2, hidden_size=64, num_layers=2, dropout=0.3):
        super(AudioClassificationModelBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  

    def forward(self, x):
        x = x.squeeze(1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

def train(dataset, model_path, modelFlag=0):
    dataset = create_dataset(dataset)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = train_dataset[0][0].shape[-1]
    
    if (modelFlag == 0):
        model = AudioClassificationModel(input_size=input_size).to(device)
    elif (modelFlag == 1):
        model = AudioClassificationModelDense(input_size=input_size).to(device)
    elif (modelFlag == 2):
        model = AudioClassificationModelGRU(input_size=input_size).to(device)
    else:
        model = AudioClassificationModelBiLSTM(input_size=input_size).to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_val_acc = 0.0
    best_model_state_dict = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                val_predictions.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_predictions)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state_dict = model.state_dict()

    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, model_path)
        print(f'Best model saved with validation accuracy: {best_val_acc:.4f}')
    else:
        print('No improvements in validation accuracy. Model not saved.')
