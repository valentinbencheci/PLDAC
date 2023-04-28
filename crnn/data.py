from spectrogram import extract_log_mel_energy_features
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    ConcatDataset,
    random_split,
)


class AudioDataset(Dataset):
    def __init__(
        self,
        csv_file,
        audio_dir,
        # input_channels=1,
        transform=extract_log_mel_energy_features,
        data_slice=None,
    ):
        self.data = pd.read_csv(csv_file, delimiter=",", header=0)
        if data_slice:
            self.data = self.data.sample(frac=data_slice / 100).reset_index(drop=True)
        self.labels = list(set(self.data["hasbird"]))
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_label = {i: label for i, label in enumerate(self.labels)}
        self.audio_dir = audio_dir
        self.transform = transform
        # self.input_channels = input_channels

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


def load_data(dataset, batch_size, split_size):
    # Calculate class weights
    class_counts = [
        len(dataset.data[dataset.data["hasbird"] == label]) for label in dataset.labels
    ]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

    # Split the dataset into training and validation sets
    train_size = int(split_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create weighted random sampler
    train_indices = train_dataset.indices
    targets = [
        dataset.label_to_idx[label]
        for label in dataset.data.loc[train_indices, "hasbird"]
    ]
    sample_weights = [class_weights[class_idx] for class_idx in targets]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, train_loader, val_dataset, val_loader


def load_merged_data(batch_size, split_size):
    warblrb10k_train, _, warblrb10k_test, _ = load_data(
        AudioDataset(
            csv_file="../data/warblrb10k_public_metadata.csv",
            audio_dir="../data/warblrb10k_public/wav",
            data_slice=50,
        ),
        batch_size,
        split_size,
    )

    ff1010bird_train, _, ff1010bird_test, _ = load_data(
        AudioDataset(
            csv_file="../data/ff1010bird_metadata.csv",
            audio_dir="../data/ff1010bird/wav",
            data_slice=50,
        ),
        batch_size,
        split_size,
    )

    birdvox_train, _, birdvox_test, _ = load_data(
        AudioDataset(
            csv_file="../data/BirdVoxDCASE20k_csvpublic.csv",
            audio_dir="../data/BirdVox-DCASE-20k/wav",
            data_slice=50,
        ),
        batch_size,
        split_size,
    )

    train_loader = DataLoader(
        ConcatDataset([warblrb10k_train, ff1010bird_train, birdvox_train]),
        batch_size=batch_size,
    )
    val_loader = DataLoader(
        ConcatDataset([warblrb10k_test, ff1010bird_test, birdvox_test]),
        batch_size=batch_size,
    )

    return train_loader, val_loader
