import librosa
import numpy as np
from scipy import ndimage

def compute_spectrogram(y, n_fft=512, hop_length=512*3//4):
    """
    Compute the spectrogram 

    @y: input signal. Multi-channel is supported
    @n_fft: length of the windowed signal after padding with zeros
    @hop_length: number of audio samples between adjacent STFT columns
    @returns: spectrogram calculated with the given parameters
    """
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    amplitude = np.abs(D)
    normalized_amplitude = amplitude / np.max(amplitude)
    return normalized_amplitude

def extract_signal_mask(spectrogram, threshold_factor, row_factor=None, col_factor=None, filter_size=(4, 4)):
    """
    Extract a binary signal mask from the input spectrogram based on the threshold factors and filter size.
    
    @param spectrogram: np.ndarray, 2D input spectrogram to be processed.
    @param threshold_factor: float, factor to be used as the threshold for both row and column factors if not specified.
    @param row_factor: float, optional, factor to be used as the threshold for rows (defaults to threshold_factor).
    @param col_factor: float, optional, factor to be used as the threshold for columns (defaults to threshold_factor).
    @param filter_size: tuple, optional, size of the structuring element for erosion and dilation (defaults to (4, 4)).
    @returns: np.ndarray, binary signal mask extracted from the spectrogram with the same dimensions as the input spectrogram.
    """
    if row_factor is None:
        row_factor = threshold_factor
    if col_factor is None:
        col_factor = threshold_factor
        
    row_median = np.median(spectrogram, axis=1, keepdims=True)
    col_median = np.median(spectrogram, axis=0, keepdims=True)
    signal_mask = np.where((spectrogram > row_median * row_factor) & (spectrogram > col_median * col_factor), 1, 0)
    signal_mask = ndimage.binary_erosion(signal_mask, structure=np.ones(filter_size))
    signal_mask = ndimage.binary_dilation(signal_mask, structure=np.ones(filter_size))
    return signal_mask

def create_indicator_vector(signal_mask, filter_size=(4, 1)):
    """
    Create an indicator vector from the input signal mask based on the specified filter size.
    
    @signal_mask: np.ndarray, binary signal mask to be processed.
    @filter_size: tuple, optional, size of the structuring element for dilation (defaults to (4, 1)).
    @returns: np.ndarray, 1D indicator vector created from the signal mask.
    """
    indicator_vector = (np.sum(signal_mask, axis=0) > 0).astype(int)
    indicator_vector = ndimage.binary_dilation(indicator_vector, structure=np.ones(filter_size[1]))
    indicator_vector = ndimage.binary_dilation(indicator_vector, structure=np.ones(filter_size[1])) 
    return indicator_vector

def apply_mask_to_audio(y, indicator_vector, hop_length=512*3//4):
    """
    Apply the indicator vector as a mask to the input audio signal based on the specified hop length.
    
    @y: np.ndarray, 1D input audio signal to be processed.
    @indicator_vector: np.ndarray, 1D binary indicator vector to be used as a mask.
    @hop_length: int, optional, the number of samples between successive frames (defaults to 512*3//4).
    @returns: np.ndarray, masked audio signal with the same dimensions as the input audio signal.
    """
    audio_mask = np.repeat(indicator_vector, hop_length)
    audio_mask = audio_mask[:len(y)]
    return y * audio_mask
