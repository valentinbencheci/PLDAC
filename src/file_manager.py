import os
import librosa
import numpy as np
import soundfile as sf
import spectrogram as sp
import torchaudio.transforms as T
import torch
import logging
import datetime
from tqdm import tqdm

log_path = '../log_' + datetime.datetime.now().date().strftime('%d_%m_%Y') + '.log'
logging.basicConfig(level=logging.INFO, filename=log_path, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

def split_signal_with_padding(y):
    spectrogram = sp.compute_spectrogram(y)
    signal_mask = sp.extract_signal_mask(spectrogram, 3)
    indicator_vector = sp.create_indicator_vector(signal_mask)
    y_signal = sp.apply_mask_to_audio(y, indicator_vector)
    return y_signal

def remove_padding(y):
    yAbs = np.abs(y)
    return yAbs[yAbs != 0]

def split_audio_into_chunks(file_path, src_path, dst_path, chunk_duration_ms=2000, sr_cons=44100):
    y, sr = librosa.load(src_path + file_path, sr=None)
    
    # Check empty file
    if (len(y) == 0):
        return
    
    # Calculate the number of samples per chunk
    chunk_size = int(sr_cons * (chunk_duration_ms / 1000))
    # Cut the noise from audio
    y = split_signal_with_padding(y)
    # Remove padding from audio
    y = remove_padding(y)
    # Split the audio into chunks using numpy array_split
    chunks = np.array_split(y, np.arange(chunk_size, len(y), chunk_size))
    chunks[len(chunks) - 1] = np.resize(chunks[len(chunks) - 1], chunk_size)

    file_prefix = file_path.split('.')[0]
    file_extension = file_path.split('.')[1]
    for i in range(len(chunks)):
        sf.write(dst_path + file_prefix + '_' + str(i) + '_' + str(len(chunks)) + '.' + file_extension, chunks[i], sr_cons)

def preprocess_all_original_audio(src_path=['../data/birds/original/', '../data/other/original/'], dst_path=['../data/birds/signal_chunks/', '../data/other/signal_chunks/']):
    for i in range(len(src_path)):
        files = os.listdir(src_path[i])
        for file_i in tqdm(range(len(files))):
            split_audio_into_chunks(files[file_i], src_path[i], dst_path[i])
            logging.info('File {} was splited in chunks'.format(files[file_i]))

def preprocess_audio_model(audio_path):
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    audio = audio.astype(np.float32)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram)

    # Pad the spectrogram to a fixed size
    target_width = 150
    if log_spectrogram.shape[1] < target_width:
        pad_width = target_width - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')

    return torch.tensor(log_spectrogram)


def create_dataset(src_path=['../data/birds/signal_chunks/', '../data/other/signal_chunks/'], testFlag = False):
    dataset = []
    start = 0

    max_chunks = int(min(len(os.listdir(src_path[0])), len(os.listdir(src_path[1]))) * 0.7)

    if (testFlag):
        start = max_chunks
        max_chunks = int(min(len(os.listdir(src_path[0])), len(os.listdir(src_path[1]))) - max_chunks)
    for i in range(len(src_path)):
        
        files = os.listdir(src_path[i])
        for file_i in tqdm(range(max_chunks)):
            
            spectrogram = preprocess_audio_model(src_path[i] + files[start + file_i])
            if (testFlag):
                dataset.append((src_path[i]+files[file_i], i))
            else:
                dataset.append((spectrogram.unsqueeze(0), i))
            logging.info('File {} was added in dataset'.format(files[start + file_i]))

    return dataset