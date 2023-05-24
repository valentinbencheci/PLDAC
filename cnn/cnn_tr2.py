import os
import torch
import librosa
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
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

def create_dataset(src_path=['../dataset/train/pre_traitement_2/warblrb10k/'], csv="../dataset/bird_dataset.csv"):
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

#CNN architecture
class AudioClassificationModel(nn.Module):
    def __init__(self, input_size=(1, 128, 862), num_classes=2):
        super(AudioClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size[0], out_channels=32, kernel_size=5, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2)

        output_size = self._get_conv_output_size(input_size)
        self.fc1 = nn.Linear(output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def _get_conv_output_size(self, input_size):
        dummy_input = torch.ones(1, *input_size)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        return x.size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(dataset_path, model_path, input_size=(1, 128, 862)):
    dataset = create_dataset(dataset_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioClassificationModel(input_size=input_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_val_acc = 0.0  
    best_epoch = -1  
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
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

    print(f'Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch+1})')
