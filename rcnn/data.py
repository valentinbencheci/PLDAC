import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from spectrogram import extract_log_mel_energy_features


class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, input_channels=1, transform=extract_log_mel_energy_features):
        self.data = pd.read_csv(csv_file, delimiter=",", header=0)
        self.labels = list(set(self.data["hasbird"]))
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_label = {i: label for i, label in enumerate(self.labels)}
        self.audio_dir = audio_dir
        self.transform = transform
        self.input_channels = input_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.loc[idx, "itemid"]
        label = self.data.loc[idx, "hasbird"]
        # Load the audio and extract log mel-band energy features
        audio_path = f"{self.audio_dir}/{filename}.wav"
        features = self.transform(audio_path)
        # Reshape features tensor to have shape (N, Cin, H, W)
        # Note: can be done directly in forward with tensors through x.unsqueeze(1)
        # features = features[np.newaxis, ...]  # add N dimension
        # features = np.repeat(features, self.input_channels, axis=0)  # repeat features along Cin dimension
        # Convert label to index
        label_idx = self.label_to_idx[label]
        return features, label_idx


def load_data(batch_size):
    # Load the data
    dataset = AudioDataset("../data/warblrb10k_public_metadata.csv", "../data/warblrb10k_public/wav")

    # Calculate class weights
    class_counts = [len(dataset.data[dataset.data['hasbird'] == label]) for label in dataset.labels]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

    # Create weighted random sampler
    targets = [dataset.label_to_idx[label] for label in dataset.data['hasbird']]
    sample_weights = [class_weights[class_idx] for class_idx in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, train_loader, val_dataset, val_loader


def load_merged_data(batch_size):
    # Load the data
    warblrb10k = AudioDataset("../data/warblrb10k_public_metadata.csv", "../data/warblrb10k_public/wav")
    ff1010bird = AudioDataset("../data/ff1010bird_metadata.csv", "../data/ff1010bird/wav")

    dataset = ConcatDataset([warblrb10k, ff1010bird])

    # Calculate the class weights based on the balanced dataset:
    class_counts = [len(warblrb10k.data[warblrb10k.data["hasbird"] == label]) + len(ff1010bird.data[ff1010bird.data["hasbird"] == label]) for label in warblrb10k.labels]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

    # Create a list of targets and sample weights for the weighted random sampler:
    targets1 = [warblrb10k.label_to_idx[label] for label in warblrb10k.data["hasbird"]]
    sample_weights1 = [class_weights[class_idx] for class_idx in targets1]

    targets2 = [ff1010bird.label_to_idx[label] for label in ff1010bird.data["hasbird"]]
    sample_weights2 = [class_weights[class_idx] for class_idx in targets2]

    targets = targets1 + targets2
    sample_weights = sample_weights1 + sample_weights2

    # Create the weighted random sampler:
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, train_loader, val_dataset, val_loader


