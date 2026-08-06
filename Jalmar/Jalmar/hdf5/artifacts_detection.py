"""
Artifact detection module for sleep stage analysis.

This module provides functions to detect and mark artifacts in EEG, EMG, and EOG signals.
"""

import numpy as np
from scipy.signal import butter, sosfilt


def find_intervals(bool_array):
    """
    Find start and end indices of consecutive True values.

    Parameters:
        bool_array (numpy.ndarray): Boolean array.

    Returns:
        numpy.ndarray: 2D array of intervals (start, end).
    """
    bool_array = bool_array.reshape((1, len(bool_array)))
    starts = np.where(np.diff(np.concatenate(([False], bool_array[0]), axis=0)).astype(int) > 0)[0]
    ends = np.where(np.diff(np.concatenate((bool_array[0], [False]), axis=0)).astype(int) < 0)[0]
    intervals = np.vstack((starts, ends)).T
    return intervals


def detect_signal_threshold(signal, lower_threshold, upper_threshold):
    """
    Detect artifacts based on signal amplitude thresholds.

    Parameters:
        signal (numpy.ndarray): Input signal.
        lower_threshold (float): Lower threshold.
        upper_threshold (float): Upper threshold.

    Returns:
        numpy.ndarray: Boolean array of artifact locations.
    """
    artifact_mask = (np.abs(signal) < lower_threshold) | (np.abs(signal) > upper_threshold)
    return artifact_mask


def remove_artifacts(signal, fs, thresholds, bandpass_freqs=None):
    """
    Remove artifacts from signal using threshold detection and optional filtering.

    Parameters:
        signal (numpy.ndarray): Input signal.
        fs (float): Sampling frequency.
        thresholds (list): [lower_threshold, upper_threshold] in standard deviations.
        bandpass_freqs (list, optional): [lowcut, highcut] for bandpass filter.

    Returns:
        tuple: (filtered_signal, artifact_indices)
    """
    signal = np.ravel(signal)

    # Apply bandpass filter if specified
    if bandpass_freqs is not None:
        lowcut, highcut = bandpass_freqs
        sos = butter(4, [lowcut, highcut], btype='band', fs=fs, output='sos')
        signal = sosfilt(sos, signal)

    # Detect artifacts based on thresholds (in standard deviations)
    mean_sig = np.nanmean(signal)
    std_sig = np.nanstd(signal)
    lower_thresh = mean_sig - thresholds[0] * std_sig
    upper_thresh = mean_sig + thresholds[1] * std_sig

    artifact_mask = detect_signal_threshold(signal, lower_thresh, upper_thresh)
    artifact_indices = np.where(artifact_mask)[0]

    return signal, artifact_indices


def artifact_epochs(artifact_indices, window_length):
    """
    Convert artifact sample indices to artifact epoch indices.

    Parameters:
        artifact_indices (numpy.ndarray): Indices of artifact samples.
        window_length (int): Samples per epoch.

    Returns:
        numpy.ndarray: Epoch indices containing artifacts.
    """
    if len(artifact_indices) == 0:
        return np.array([])

    artifact_epochs_idx = np.unique(artifact_indices // window_length)
    return artifact_epochs_idx
