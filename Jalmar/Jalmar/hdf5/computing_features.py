"""
Feature computation module for sleep stage analysis.

This module provides functions to compute spectral and complexity features
from raw EEG, EMG, and EOG signals for sleep stage classification.
"""

import numpy as np
import os
import mne.time_frequency
from scipy.signal import welch, savgol_filter, hilbert, butter, filtfilt
from scipy.stats import mode, entropy
from specparam import SpectralModel
from joblib import Parallel, delayed
from neurodsp.aperiodic import compute_fluctuations
import EntropyHub as EH
import warnings

# Suppress specparam warnings about skipping frequency == 0
warnings.filterwarnings('ignore', message='.*skipping frequency == 0.*')


def psd_multitaper(lfp_data, fs, frequency_band, window_length):
    """
    Computes the power spectral density (PSD) using the multitaper method.

    Parameters:
        lfp_data (numpy.ndarray): The input signal.
        fs (float): The sampling frequency of the signal.
        frequency_band (tuple): A tuple (min, max) frequency band.
        window_length (int): The length of the window for PSD computation.

    Returns:
        list: Total power within the specified frequency band for each epoch.
    """
    all_power_sum = []

    for start in range(0, len(lfp_data) - window_length + 1, window_length):
        window = lfp_data[start:min(start + window_length, len(lfp_data))]
        psd, freqs = mne.time_frequency.psd_array_multitaper(
            window, fs, fmin=frequency_band[0], fmax=frequency_band[1],
            n_jobs=1, verbose='warning'
        )
        curr_sum = np.sum(psd)
        all_power_sum.append(curr_sum)

    return all_power_sum


def wei_normalizing(data):
    """
    Normalizes data using 10th and 90th percentiles.

    Based on Wei et al. (2019) normalization approach for sleep scoring.

    Parameters:
        data (numpy.ndarray): The input data to be normalized.

    Returns:
        numpy.ndarray: The normalized data [0.05, 1].
    """
    data = np.array(data)
    bottom = data[data <= np.nanpercentile(data, 10, axis=0)]
    bottom_avg = np.average(bottom)
    top = data[data >= np.nanpercentile(data, 90, axis=0)]
    top_avg = np.average(top)
    
    # Handle edge case where all values are the same (zero variance)
    denominator = top_avg - bottom_avg
    if denominator == 0 or np.isnan(denominator):
        # If data has no variance, return constant normalized value
        return np.ones_like(data) * 0.5
    
    normalized_data = (data - bottom_avg) / denominator
    normalized_data = np.clip(normalized_data, 0.05, 1)

    return normalized_data


def index_W(theta, gamma, emg):
    """Wake index: EMG^2 * (gamma/theta)"""
    return np.array([emg[i] ** 2 * (gamma[i] / theta[i]) for i in range(len(theta))])


def index_N(delta, sigma, gamma):
    """NREM index (non-EOG): (delta * sigma) / gamma^2"""
    return np.array([(delta[i] * sigma[i]) / (gamma[i] ** 2) for i in range(len(delta))])


def index_N_eog(gamma, eog_03_45):
    """NREM index (EOG variant): (EOG_(0.3-0.45)^2) / gamma^2"""
    return np.array(
        [(eog_03_45[i] ** 2) / (gamma[i] ** 2)
         for i in range(len(gamma))]
    )


def index_R(delta, theta, sigma, emg, gamma):
    """REM index (non-EOG): (2 * theta * gamma) / (delta^2 * EMG^2)"""
    return np.array(
        [(2 * theta[i] * gamma[i]) / (delta[i] ** 2 * emg[i] ** 2)
         for i in range(len(delta))]
    )


def index_R_eog(delta, emg, eog_03_35):
    """REM index (EOG variant): (EOG_(0.3-35)^2) / (EMG^2 * delta^2)"""
    return np.array(
        [(eog_03_35[i] ** 2) / (emg[i] ** 2 * delta[i] ** 2)
         for i in range(len(delta))]
    )


def index_1(delta, gamma, emg):
    """Custom index 1: (EMG * gamma) / delta"""
    return np.array([(emg[i] * gamma[i]) / delta[i] for i in range(len(delta))])


def index_2(delta, theta, sigma):
    """Custom index 2: (sigma * delta) / theta"""
    return np.array([(sigma[i] * delta[i]) / theta[i] for i in range(len(delta))])


def index_3(delta, theta, gamma):
    """Custom index 3: (theta * gamma) / delta"""
    return np.array([(theta[i] * gamma[i]) / delta[i] for i in range(len(delta))])


def index_4(delta, theta):
    """Custom index 4: delta / theta"""
    return np.array([delta[i] / theta[i] for i in range(len(delta))])


def aperiodic_fit(window_data, fs, preferred_nperseg=1024):
    """
    Get aperiodic component (exponent) from a signal segment.

    Parameters:
        window_data (array): Segment of EEG signal.
        fs (int): Sampling frequency.

    Returns:
        float: Aperiodic exponent.
    """
    try:
        if preferred_nperseg is None:
            preferred_nperseg = 1024
        preferred_nperseg = int(preferred_nperseg)
        if preferred_nperseg <= 0:
            preferred_nperseg = len(window_data)

        nperseg = min(preferred_nperseg, len(window_data))
        if nperseg < 2:
            return np.nan
        freqs, psd = welch(window_data, fs=fs, nperseg=nperseg)
        mask = (freqs <= 75)
        freqs, psd = freqs[mask], psd[mask]
        psd = np.where(psd > 0, psd, 1e-12)

        fm = SpectralModel(min_peak_height=0.05, aperiodic_mode='fixed', verbose=False)
        fm.fit(freqs, psd)

        # Check if fit was successful by checking if aperiodic parameters exist
        aperiodic_params = fm.get_params('aperiodic')
        if aperiodic_params is None or len(aperiodic_params) < 2:
            return np.nan

        aperiodic = aperiodic_params[1]
        return aperiodic
    except Exception:
        return np.nan


def calc_aperiodic_fit(data, window_size, fs, welch_nperseg=1024):
    """
    Compute aperiodic fit from EEG data.

    Parameters:
        data (array): EEG data.
        window_size (int): Size of window to calculate exponent from.
        fs (int): Sampling frequency.

    Returns:
        array: Normalized and smoothed aperiodic exponents.
    """
    window_data = []
    num_windows = len(data) // window_size

    for i in range(num_windows):
        start, end = i * window_size, (i + 1) * window_size
        window_data.append(data[start:end])

    # Use threads with bounded workers to avoid loky worker crashes / high memory overhead.
    max_workers = min(4, os.cpu_count() or 1)
    aperiodic_exponents = Parallel(n_jobs=max_workers, prefer='threads')(
        delayed(aperiodic_fit)(window, fs, welch_nperseg) for window in window_data
    )

    window_length = 11 if len(aperiodic_exponents) >= 11 else len(aperiodic_exponents) | 1
    polyorder = 4 if len(aperiodic_exponents) >= 5 else len(aperiodic_exponents) - 1

    if len(aperiodic_exponents) < 2:
        return aperiodic_exponents

    smoothed_exponents = savgol_filter(
        aperiodic_exponents, window_length=window_length, polyorder=polyorder
    )

    max_val = max(smoothed_exponents)
    min_val = min(smoothed_exponents)
    if max_val == min_val:
        normalized_exponents = np.zeros_like(smoothed_exponents)
    else:
        normalized_exponents = 2 * ((smoothed_exponents - min_val) / (max_val - min_val)) - 1

    return normalized_exponents


def dfa_exponent(data, fs):
    """
    Calculate detrended fluctuation analysis exponent.

    Parameters:
        data (array): Input signal.
        window_size (int): Window size for DFA.

    Returns:
        float: DFA exponent.
    """
    try:
        # Use fs-aware scales in seconds to make DFA stable across epoch sizes.
        fs = float(fs) if fs > 0 else 1.0
        min_scale = 0.5
        max_scale = 5.0
        _, exponent = compute_fluctuations(data, fs=fs, min_scale=min_scale, max_scale=max_scale)
        exponent = np.asarray(exponent, dtype=float).ravel()
        if exponent.size == 0:
            return np.nan
        return float(exponent[-1])
    except:
        return np.nan


def _finite_entropy_fallback(window, bins=32):
    """Fallback complexity proxy when SampEn fails: histogram entropy."""
    window = np.asarray(window)
    valid = window[np.isfinite(window)]

    if valid.size == 0:
        return np.nan

    # Adaptive bin count: use fewer bins if data range is small or has few unique values.
    unique_vals = np.unique(valid)
    n_unique = len(unique_vals)
    data_range = np.max(valid) - np.min(valid)

    # Use at most 'bins' or 2x unique values, whichever is smaller. Minimum 2 bins.
    adaptive_bins = max(2, min(bins, max(2, n_unique // 2 + 1)))

    try:
        hist, _ = np.histogram(valid, bins=adaptive_bins, density=True)
        hist = np.clip(hist, 1e-12, None)
        return float(entropy(hist))
    except Exception:
        # If histogram still fails (e.g., all exactly same value), return NaN.
        return np.nan
def _fill_nans_with_interpolation(values):
    """Fill NaNs by linear interpolation, preserving endpoint values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    idx = np.arange(arr.size)
    valid = np.isfinite(arr)
    if np.all(valid):
        return arr
    if np.sum(valid) == 0:
        return arr

    arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    return arr


def calc_dfa(data, window_size, step_size, fs):
    """
    Compute Detrended Fluctuation Analysis from EEG data.

    Parameters:
        data (array): EEG data.
        window_size (int): Size of window to calculate exponent from.
        step_size (int): Step size between windows.
        fs (int): Sampling frequency.

    Returns:
        array: DFA exponents for each window.
    """
    dfa_exponents = []
    fallback_values = []
    num_windows = (len(data) - window_size) // step_size + 1

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        if end <= len(data):
            window = data[start:end]
            exp = dfa_exponent(window, fs)
            dfa_exponents.append(exp)
            # Non-constant fallback proxy: normalized roughness.
            roughness = np.std(np.diff(window)) / (np.std(window) + 1e-12)
            fallback_values.append(float(roughness))

    if len(dfa_exponents) == 0:
        return np.array([np.nan])

    dfa_array = np.array(dfa_exponents, dtype=float)
    fallback_array = np.array(fallback_values, dtype=float)
    nan_mask = ~np.isfinite(dfa_array)
    if np.any(~nan_mask):
        dfa_array = _fill_nans_with_interpolation(dfa_array)
    else:
        # If DFA fails globally, use roughness proxy instead of constant ones.
        dfa_array = fallback_array

    return dfa_array


def sample_entropy(data, window_size):
    """
    Calculate sample entropy.

    Parameters:
        data (array): Input signal.
        window_size (int): Window size.

    Returns:
        float: Sample entropy.
    """
    try:
        # Use explicit tolerance tied to signal variance for numerical stability.
        r = 0.2 * np.std(data)
        if not np.isfinite(r) or r <= 0:
            return np.nan
        samp_en, *_ = EH.SampEn(data, m=2, tau=1, r=r)
        samp_en = np.asarray(samp_en, dtype=float).ravel()
        if samp_en.size == 0:
            return np.nan
        return float(samp_en[-1])
    except:
        return np.nan


def calc_mse(data, window_size, step_size, fs):
    """
    Compute Multiscale Entropy from EEG data.

    Parameters:
        data (array): EEG data.
        window_size (int): Size of window.
        step_size (int): Step size between windows.
        fs (int): Sampling frequency.

    Returns:
        array: Sample entropy values for each window.
    """
    mse_values = []
    fallback_values = []
    num_windows = (len(data) - window_size) // step_size + 1

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        if end <= len(data):
            window = data[start:end]
            ent = sample_entropy(window, window_size)
            mse_values.append(ent)
            fallback_values.append(_finite_entropy_fallback(window))

    if len(mse_values) == 0:
        return np.array([np.nan])

    mse_array = np.array(mse_values, dtype=float)
    fallback_array = np.array(fallback_values, dtype=float)
    nan_mask = ~np.isfinite(mse_array)
    if np.any(~nan_mask):
        mse_array = _fill_nans_with_interpolation(mse_array)
    else:
        # If SampEn fails globally, use histogram entropy proxy instead of constants.
        mse_array = fallback_array

    return mse_array


def compute_features(raw_fpz, raw_pz, raw_emg, raw_eog, states, fs, epoch_length, welch_nperseg=1024, return_raw=False):
    """
    Compute features from raw EEG, EMG, and EOG signals.

    Parameters:
        raw_fpz (numpy.ndarray): Frontal EEG data.
        raw_pz (numpy.ndarray): Parietal EEG data.
        raw_emg (numpy.ndarray): EMG data.
        raw_eog (numpy.ndarray): EOG data.
        states (numpy.ndarray): Sleep states.
        fs (float): Sampling frequency.
        epoch_length (int): Epoch length in seconds.
        welch_nperseg (int): Welch window length for spectral analysis.
        return_raw (bool): If True, also return raw features before normalization/smoothing.

    Returns:
        If return_raw=False:
            features (numpy.ndarray): Computed features (n_epochs, n_features).
            mapped_scores (numpy.ndarray): Sleep stage scores.
        If return_raw=True:
            features (numpy.ndarray): Normalized and smoothed features.
            mapped_scores (numpy.ndarray): Sleep stage scores.
            raw_features (numpy.ndarray): Raw features before normalization/smoothing.
    """
    # Flatten inputs and compute number of samples per epoch
    raw_fpz = np.ravel(raw_fpz)
    raw_pz = np.ravel(raw_pz)
    raw_emg = np.ravel(raw_emg)
    raw_eog = np.ravel(raw_eog)
    sleep_scoring = np.ravel(states)
    window_length = int(fs * epoch_length)

    # Helper: convert a 1D signal to an epoch-wise feature using a reducer.
    def epoch_reduce(signal, reducer):
        n = len(signal) // window_length
        if n == 0:
            return np.array([])
        return reducer(signal[:n * window_length].reshape(n, window_length), axis=1)

    # Get mapped scores (majority vote per epoch)
    n_epochs = len(sleep_scoring) // window_length
    reshaped_scores = sleep_scoring[:n_epochs * window_length].reshape(-1, window_length)
    majority_scores = mode(reshaped_scores, axis=1).mode.flatten()
    mapped_scores = np.array(majority_scores)

    # Frequency ranges
    delta_band = [0.5, 3.99]
    theta_band = [4, 7.99]
    sigma_band = [11, 15]
    gamma_band = [30, 40]
    window_length = int(fs * epoch_length)

    # Compute power spectral densities
    delta = np.asarray(psd_multitaper(raw_fpz, fs, delta_band, window_length))
    theta = np.asarray(psd_multitaper(raw_pz, fs, theta_band, window_length))
    sigma = np.asarray(psd_multitaper(raw_fpz, fs, sigma_band, window_length))
    gamma = np.asarray(psd_multitaper(raw_fpz, fs, gamma_band, window_length))

    # EMG and EOG must also be epoch-level to match other feature streams.
    emg_epoch = epoch_reduce(raw_emg, lambda x, axis=1: np.sqrt(np.mean(np.square(x), axis=axis)))
    hilbert_eog = np.abs(hilbert(raw_eog))
    eog_epoch = epoch_reduce(hilbert_eog, np.mean)

    # EOG (0.3-35 Hz) epoch signal for optional REM index variant.
    nyquist = fs / 2.0
    if nyquist > 35:
        b, a = butter(4, [0.3 / nyquist, 35.0 / nyquist], btype='band')
        eog_03_35 = filtfilt(b, a, raw_eog)
    else:
        eog_03_35 = raw_eog
    eog_03_35_epoch = epoch_reduce(
        eog_03_35,
        lambda x, axis=1: np.sqrt(np.mean(np.square(x), axis=axis))
    )

    # Disabled for now (EOG-based Index_N variant):
    if nyquist > 0.45:
        b, a = butter(4, [0.3 / nyquist, 0.45 / nyquist], btype='band')
        eog_03_45 = filtfilt(b, a, raw_eog)
    else:
        eog_03_45 = raw_eog
    eog_03_45_epoch = epoch_reduce(
        eog_03_45,
        lambda x, axis=1: np.sqrt(np.mean(np.square(x), axis=axis))
    )

    # Complexity features (already epoch-level)
    aperiodic = np.asarray(calc_aperiodic_fit(raw_fpz, window_length, fs, welch_nperseg=welch_nperseg))
    dfa = np.asarray(calc_dfa(raw_fpz, window_length, window_length, fs))
    mse = np.asarray(calc_mse(raw_fpz, window_length, window_length, fs))

    # Align all vectors to common epoch count before combining features.
    common_len = min(
        len(mapped_scores), len(delta), len(theta), len(sigma), len(gamma),
        len(emg_epoch), len(eog_epoch), len(eog_03_35_epoch), len(eog_03_45_epoch),
        len(aperiodic), len(dfa), len(mse)
    )
    if common_len == 0:
        return np.empty((0, 15)), np.array([])

    mapped_scores = mapped_scores[:common_len]
    delta = delta[:common_len]
    theta = theta[:common_len]
    sigma = sigma[:common_len]
    gamma = gamma[:common_len]
    emg_epoch = emg_epoch[:common_len]
    eog_epoch = eog_epoch[:common_len]
    eog_03_35_epoch = eog_03_35_epoch[:common_len]
    eog_03_45_epoch = eog_03_45_epoch[:common_len]
    aperiodic = aperiodic[:common_len]
    dfa = dfa[:common_len]
    mse = mse[:common_len]

    # Normalize
    delta_norm = wei_normalizing(delta)
    theta_norm = wei_normalizing(theta)
    sigma_norm = wei_normalizing(sigma)
    gamma_norm = wei_normalizing(gamma)
    emg_norm = wei_normalizing(emg_epoch)
    eog_03_35_norm = wei_normalizing(eog_03_35_epoch)
    eog_03_45_norm = wei_normalizing(eog_03_45_epoch)

    # Smooth powers
    delta_smoothed = np.convolve(
        np.convolve(np.convolve(delta_norm, np.ones(5) / 5, mode='same'),
                    np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )
    theta_smoothed = np.convolve(
        np.convolve(np.convolve(theta_norm, np.ones(5) / 5, mode='same'),
                    np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )
    if len(eog_epoch) >= 3:
        eog_window = 11 if len(eog_epoch) >= 11 else (len(eog_epoch) if len(eog_epoch) % 2 == 1 else len(eog_epoch) - 1)
        eog_poly = min(5, eog_window - 1)
        eog_smoothed = savgol_filter(eog_epoch, eog_window, polyorder=eog_poly)
    else:
        eog_smoothed = eog_epoch

    # Compute indices
    idx_w = index_W(theta_norm, gamma_norm, emg_norm)
    idx_n = index_N(delta_norm, sigma_norm, gamma_norm)
    idx_n_eog = index_N_eog(gamma_norm, eog_03_45_norm)
    idx_r = index_R_eog(delta_norm, emg_norm, eog_03_35_norm)
    idx_r_noeog = index_R(delta_norm, theta_norm, sigma_norm, emg_norm, gamma_norm)
    idx_1 = index_1(delta_norm, gamma_norm, emg_norm)
    idx_2 = index_2(delta_norm, theta_norm, sigma_norm)
    idx_3 = index_3(delta_norm, theta_norm, gamma_norm)
    idx_4 = index_4(delta_norm, theta_norm)

    # Log and normalize indices
    idx_w_norm = wei_normalizing(np.log(idx_w))
    idx_n_norm = wei_normalizing(np.log(idx_n))
    idx_n_eog_norm = wei_normalizing(np.log(idx_n_eog))
    idx_r_norm = wei_normalizing(np.log(idx_r))
    idx_r_noeog_norm = wei_normalizing(np.log(idx_r_noeog))
    idx_1_norm = wei_normalizing(np.log(idx_1))
    idx_2_norm = wei_normalizing(np.log(idx_2))
    idx_3_norm = wei_normalizing(np.log(idx_3))
    idx_4_norm = wei_normalizing(np.log(idx_4))
    eog_norm = wei_normalizing(eog_smoothed)

    # Smooth indices
    idx_w_smoothed = np.convolve(
        np.convolve(np.convolve(idx_w_norm, np.ones(5) / 5, mode='same'),
                    np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )
    idx_n_smoothed = np.convolve(
        np.convolve(np.convolve(idx_n_norm, np.ones(5) / 5, mode='same'),
                    np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )
    idx_n_eog_smoothed = np.convolve(
        np.convolve(np.convolve(idx_n_eog_norm, np.ones(5) / 5, mode='same'),
                    np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )
    idx_r_smoothed = np.convolve(
        np.convolve(np.convolve(idx_r_norm, np.ones(5) / 5, mode='same'),
                    np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )
    idx_r_noeog_smoothed = np.convolve(
        np.convolve(np.convolve(idx_r_noeog_norm, np.ones(5) / 5, mode='same'),
                    np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )
    idx_1_smoothed = np.convolve(
        np.convolve(np.convolve(idx_1_norm, np.ones(5) / 5, mode='same'),
                    np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )
    idx_2_smoothed = np.convolve(
        np.convolve(np.convolve(idx_2_norm, np.ones(5) / 5, mode='same'),
                    np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )
    idx_3_smoothed = np.convolve(
        np.convolve(np.convolve(idx_3_norm, np.ones(5) / 5, mode='same'),
                    np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )
    idx_4_smoothed = np.convolve(
        np.convolve(np.convolve(idx_4_norm, np.ones(5) / 5, mode='same'),
                    np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )

    # Normalize complexity features
    aperiodic_norm = wei_normalizing(aperiodic)
    dfa_norm = wei_normalizing(dfa)
    mse_norm = wei_normalizing(mse)

    # Stack features (normalized and smoothed)
    features = np.column_stack((
        idx_w_smoothed, idx_r_smoothed, idx_n_smoothed,
        idx_1_smoothed, idx_2_smoothed, idx_3_smoothed, idx_4_smoothed,
        delta_smoothed, theta_smoothed, aperiodic_norm, dfa_norm, mse_norm, eog_norm,
        idx_r_noeog_smoothed, idx_n_eog_smoothed,
    ))

    if return_raw:
        # Stack raw features before normalization and smoothing
        raw_features = np.column_stack((
            idx_w, idx_r, idx_n,
            idx_1, idx_2, idx_3, idx_4,
            delta, theta, aperiodic, dfa, mse, eog_epoch,
            idx_r_noeog, idx_n_eog,
        ))
        return features, mapped_scores, raw_features
    else:
        return features, mapped_scores
