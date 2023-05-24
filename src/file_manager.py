import os
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
import spectrogram as sp
import noisereduce as nr
import scipy.signal as signal
import torchaudio.transforms as T

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
    
    if (len(y) == 0):
        return
    
    chunk_size = int(sr_cons * (chunk_duration_ms / 1000))
    y = split_signal_with_padding(y)
    y = remove_padding(y)
    chunks = np.array_split(y, np.arange(chunk_size, len(y), chunk_size))
    chunks[len(chunks) - 1] = np.resize(chunks[len(chunks) - 1], chunk_size)

    file_prefix = file_path.split('.')[0]
    file_extension = file_path.split('.')[1]
    for i in range(len(chunks)):
        sf.write(dst_path + file_prefix + '_' + str(i) + '_' + str(len(chunks)) + '.' + file_extension, chunks[i], sr_cons)

def preprocess_all_original_audio_tr1(src_path=['../dataset/train/original/BirdVoxDCASE20k/', '../dataset/train/original/ff1010bird/', '../dataset/train/original/warblrb10k/',  '../dataset/test/original/BirdVoxDCASE20k/', '../dataset/test/original/ff1010bird/', '../dataset/test/original/warblrb10k/'], dst_path=['../dataset/train/pre_traitement_1/BirdVoxDCASE20k/', '../dataset/train/pre_traitement_1/ff1010bird/', '../dataset/train/pre_traitement_1/warblrb10k/', '../dataset/test/pre_traitement_1/BirdVoxDCASE20k/', '../dataset/test/pre_traitement_1/ff1010bird/', '../dataset/test/pre_traitement_1/warblrb10k/']):
    for i in range(len(src_path)):
        files = os.listdir(src_path[i])
        for file_i in tqdm(range(len(files))):
            split_audio_into_chunks(files[file_i], src_path[i], dst_path[i])

def reduce_noise(audio_path, output_path, initial_nom):
    data, sample_rate = sf.read(audio_path)
    reduced_noise = nr.reduce_noise(y=data, sr=sample_rate)
    sf.write(output_path + initial_nom, reduced_noise, sample_rate)

def preprocess_all_original_audio_tr2(src_path=['../dataset/train/original/BirdVoxDCASE20k/', '../dataset/train/original/ff1010bird/', '../dataset/train/original/warblrb10k/',  '../dataset/test/original/BirdVoxDCASE20k/', '../dataset/test/original/ff1010bird/', '../dataset/test/original/warblrb10k/'], dst_path=['../dataset/train/pre_traitement_2/BirdVoxDCASE20k/', '../dataset/train/pre_traitement_2/ff1010bird/', '../dataset/train/pre_traitement_2/warblrb10k/', '../dataset/test/pre_traitement_2/BirdVoxDCASE20k/', '../dataset/test/pre_traitement_2/ff1010bird/', '../dataset/test/pre_traitement_2/warblrb10k/']):
    for i in range(len(src_path)):
        files = os.listdir(src_path[i])
        for file_i in tqdm(range(len(files))):
            reduce_noise(src_path[i] + files[file_i], dst_path[i], files[file_i])