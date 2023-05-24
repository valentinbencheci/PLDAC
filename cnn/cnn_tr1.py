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

    target_width = 150
    if log_spectrogram.shape[1] < target_width:
        pad_width = target_width - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')

    return torch.tensor(log_spectrogram)

def create_dataset(src_path=['../dataset/train/pre_traitement_1/BirdVoxDCASE20k/', '../dataset/train/pre_traitement_1/ff1010bird/', '../dataset/train/pre_traitement_1/warblrb10k/'], csv="../dataset/bird_dataset.csv"):
    dataset = []
    df = pd.read_csv(csv)

    for i in range(len(src_path)):
        files = os.listdir(src_path[i])

        for file_i in tqdm(range(len(files))):
                spectrogram = preprocess_audio_model(src_path[i] + files[file_i])

                target_item_id = files[file_i].split('_')[0]
                filtered_df = df[df['itemid'] == target_item_id]
                dataset.append((spectrogram.unsqueeze(0), filtered_df['hasbird'].iloc[0]))

    return dataset

# CNN architecture
class AudioClassificationModel(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 7), stride=(1, 1), padding=(2, 3))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(1, 1), padding=(2, 3))
        self.fc1 = nn.Linear(64 * 32 * 43, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Modèle avec une couche dense supplémentaire
class AudioClassificationModelDense(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioClassificationModelDense, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 7), stride=(1, 1), padding=(2, 3))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(1, 1), padding=(2, 3))
        self.fc1 = nn.Linear(64 * 32 * 43, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Modèle avec Average Pooling au lieu de Max Pooling
class AudioClassificationModelAvgPool(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioClassificationModelAvgPool, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 7), stride=(1, 1), padding=(2, 3))
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(1, 1), padding=(2, 3))
        self.fc1 = nn.Linear(64 * 32 * 43, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Modèle avec Softmax au lieu de Max Pooling
class AudioClassificationModelSoftmax(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioClassificationModelSoftmax, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 7), stride=(1, 1), padding=(2, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(1, 1), padding=(2, 3))
        self.fc1 = nn.Linear(64 * 32 * 43, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.softmax(self.relu(self.conv1(x)))
        x = self.softmax(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(dataset_path, model_path, modelFlag=0):
    dataset = create_dataset(dataset_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if (modelFlag == 0):
        model = AudioClassificationModel().to(device)
    elif (modelFlag == 1):
        model = AudioClassificationModelDense().to(device)
    elif (modelFlag == 2):
        model = AudioClassificationModelAvgPool().to(device)
    else:
        model = AudioClassificationModelSoftmax().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
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
        
    torch.save(model.state_dict(), model_path)