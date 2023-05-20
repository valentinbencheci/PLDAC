import librosa
import numpy as np
import pandas as pd


def extract_log_mel_energy_features(audio_file_path):
    # Load audio file
    y, sr = librosa.load(audio_file_path, sr=44100)

    # Define STFT parameters
    n_fft = 1024  # set number of FFT bins
    hop_length = int(sr * 0.04) // 2 # 40 ms, 50 % overlap
    window = 'hamming'

    # Calculate STFT
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)

    # Calculate magnitude spectrum
    mag_spec = np.abs(stft)

    # Define mel filterbank parameters
    n_mels = 40
    fmin = 0
    fmax = sr / 2

    # Calculate mel filterbank
    mel_spec = librosa.feature.melspectrogram(S=mag_spec, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax)

    # Calculate log mel-band energy features
    log_mel_spec = librosa.power_to_db(mel_spec)

    # Ensure output is exactly 10 seconds long
    desired_frames = int(10 * sr / hop_length)
    current_frames = log_mel_spec.shape[1]
    if current_frames < desired_frames:
        # Add frames
        pad_width = desired_frames - current_frames
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    elif current_frames > desired_frames:
        # Remove frames
        log_mel_spec = log_mel_spec[:, :desired_frames]

    # Return features
    return log_mel_spec
