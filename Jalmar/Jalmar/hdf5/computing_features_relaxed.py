"""
Relaxed feature computation module for sleep stage analysis.

This variant keeps the same 15-feature output as computing_features.py but uses
less aggressive normalization and smoothing so more local structure survives the
feature construction stage.
"""

import numpy as np
from scipy.signal import savgol_filter, hilbert, butter, filtfilt
from scipy.stats import mode

from computing_features import (
    calc_aperiodic_fit,
    calc_dfa,
    calc_mse,
    index_1,
    index_2,
    index_3,
    index_4,
    index_N,
    index_N_eog,
    index_R,
    index_R_eog,
    index_W,
    psd_multitaper,
)


EPSILON = 1e-6
PERCENTILE_LOW = 1
PERCENTILE_HIGH = 99


def wei_normalizing(data):
    """Percentile-based normalization with a wider dynamic range preserved."""
    data = np.asarray(data)
    if data.size == 0:
        return data

    lower = np.nanpercentile(data, PERCENTILE_LOW)
    upper = np.nanpercentile(data, PERCENTILE_HIGH)

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.ones_like(data) * 0.5

    normalized_data = (data - lower) / (upper - lower)
    # Keep a small floor for log safety, but do not cap at 1.0 here.
    # The later standardization step used for mcRBM training can absorb the wider range.
    return np.clip(normalized_data, EPSILON, None)


def _smooth_1d(values, kernel_size=5):
    """Single-pass moving average to keep more temporal variation."""
    values = np.asarray(values)
    if values.size < 3:
        return values

    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    return np.convolve(values, kernel, mode='same')


def _safe_log1p(values):
    """Log transform that keeps zeros valid and ignores tiny negative noise."""
    values = np.asarray(values)
    return np.log1p(np.clip(values, a_min=0.0, a_max=None))


def _safe_log(values):
    """Safe log with epsilon floor to avoid log(0)."""
    values = np.asarray(values)
    return np.log(np.clip(values, a_min=EPSILON, a_max=None))


def index_W_raw(theta, gamma, emg):
    """Wake index (log-ratio form on raw values): 2*log(EMG) + log(gamma) - log(theta)"""
    return 2.0 * _safe_log(emg) + _safe_log(gamma) - _safe_log(theta)


def index_N_raw(delta, sigma, gamma):
    """NREM index (log-ratio form on raw values): log(delta) + log(sigma) - 2*log(gamma)"""
    return _safe_log(delta) + _safe_log(sigma) - 2.0 * _safe_log(gamma)


def index_N_eog_raw(gamma, eog_03_45):
    """NREM index EOG variant (log-ratio form on raw values): 2*log(eog_03_45) - 2*log(gamma)"""
    return 2.0 * _safe_log(eog_03_45) - 2.0 * _safe_log(gamma)


def index_R_eog_raw(delta, emg, eog_03_35):
    """REM index EOG variant (log-ratio form on raw values): 2*log(eog_03_35) - 2*log(EMG) - 2*log(delta)"""
    return 2.0 * _safe_log(eog_03_35) - 2.0 * _safe_log(emg) - 2.0 * _safe_log(delta)


def index_R_raw(delta, theta, sigma, emg, gamma):
    """REM index (log-ratio form on raw values): log(2) + log(theta) + log(gamma) - 2*log(delta) - 2*log(EMG)"""
    return np.log(2.0) + _safe_log(theta) + _safe_log(gamma) - 2.0 * _safe_log(delta) - 2.0 * _safe_log(emg)


def index_1_raw(delta, gamma, emg):
    """Custom index 1 (log-ratio form on raw values): log(EMG) + log(gamma) - log(delta)"""
    return _safe_log(emg) + _safe_log(gamma) - _safe_log(delta)


def index_2_raw(delta, theta, sigma):
    """Custom index 2 (log-ratio form on raw values): log(sigma) + log(delta) - log(theta)"""
    return _safe_log(sigma) + _safe_log(delta) - _safe_log(theta)


def index_3_raw(delta, theta, gamma):
    """Custom index 3 (log-ratio form on raw values): log(theta) + log(gamma) - log(delta)"""
    return _safe_log(theta) + _safe_log(gamma) - _safe_log(delta)


def index_4_raw(delta, theta):
    """Custom index 4 (log-ratio form on raw values): log(delta) - log(theta)"""
    return _safe_log(delta) - _safe_log(theta)


def compute_features(
    raw_fpz,
    raw_pz,
    raw_emg,
    raw_eog,
    states,
    fs,
    epoch_length,
    welch_nperseg=1024,
    apply_log1p=False,
    return_raw=False,
):
    """
    Compute features from raw EEG, EMG, and EOG signals.
    
    Pipeline:
    1. Extract raw features (delta, theta, sigma, gamma, emg, eog, aperiodic, dfa, mse)
    2. Compute indices from raw features using log-ratio form (numerically stable)
    3. Stack all 15 features (raw + indices)
    4. Normalize each feature with a wide percentile range
    5. Optionally apply log1p for downstream workflows that still want it
    6. If return_raw=True, also return unnormalized features
    """
    raw_fpz = np.ravel(raw_fpz)
    raw_pz = np.ravel(raw_pz)
    raw_emg = np.ravel(raw_emg)
    raw_eog = np.ravel(raw_eog)
    sleep_scoring = np.ravel(states)
    window_length = int(fs * epoch_length)

    def epoch_reduce(signal, reducer):
        n = len(signal) // window_length
        if n == 0:
            return np.array([])
        return reducer(signal[:n * window_length].reshape(n, window_length), axis=1)

    # Get epoch-wise sleep scores
    n_epochs = len(sleep_scoring) // window_length
    reshaped_scores = sleep_scoring[:n_epochs * window_length].reshape(-1, window_length)
    majority_scores = mode(reshaped_scores, axis=1).mode.flatten()
    mapped_scores = np.array(majority_scores)

    # Frequency bands
    delta_band = [0.5, 3.99]
    theta_band = [4, 7.99]
    sigma_band = [11, 15]
    gamma_band = [30, 40]

    # ===== STEP 1: Extract raw features (no normalization yet) =====
    delta = np.asarray(psd_multitaper(raw_fpz, fs, delta_band, window_length))
    theta = np.asarray(psd_multitaper(raw_pz, fs, theta_band, window_length))
    sigma = np.asarray(psd_multitaper(raw_fpz, fs, sigma_band, window_length))
    gamma = np.asarray(psd_multitaper(raw_fpz, fs, gamma_band, window_length))

    emg_epoch = epoch_reduce(raw_emg, lambda x, axis=1: np.sqrt(np.mean(np.square(x), axis=axis)))
    hilbert_eog = np.abs(hilbert(raw_eog))
    eog_epoch = epoch_reduce(hilbert_eog, np.mean)

    # Bandpass-filtered EOG variants
    nyquist = fs / 2.0
    if nyquist > 35:
        b, a = butter(4, [0.3 / nyquist, 35.0 / nyquist], btype='band')
        eog_03_35 = filtfilt(b, a, raw_eog)
    else:
        eog_03_35 = raw_eog

    if nyquist > 0.45:
        b, a = butter(4, [0.3 / nyquist, 0.45 / nyquist], btype='band')
        eog_03_45 = filtfilt(b, a, raw_eog)
    else:
        eog_03_45 = raw_eog

    eog_03_35_epoch = epoch_reduce(
        eog_03_35,
        lambda x, axis=1: np.sqrt(np.mean(np.square(x), axis=axis)),
    )
    eog_03_45_epoch = epoch_reduce(
        eog_03_45,
        lambda x, axis=1: np.sqrt(np.mean(np.square(x), axis=axis)),
    )

    # Complexity features
    aperiodic = np.asarray(calc_aperiodic_fit(raw_fpz, window_length, fs, welch_nperseg=welch_nperseg))
    dfa = np.asarray(calc_dfa(raw_fpz, window_length, window_length, fs))
    mse = np.asarray(calc_mse(raw_fpz, window_length, window_length, fs))

    # Align all to common length
    common_len = min(
        len(mapped_scores), len(delta), len(theta), len(sigma), len(gamma),
        len(emg_epoch), len(eog_epoch), len(eog_03_35_epoch), len(eog_03_45_epoch), 
        len(aperiodic), len(dfa), len(mse),
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

    # ===== STEP 2: Compute indices directly from RAW features using log-ratio form =====
    idx_w = index_W_raw(theta, gamma, emg_epoch)
    idx_r = index_R_eog_raw(delta, emg_epoch, eog_03_35_epoch)
    idx_n = index_N_raw(delta, sigma, gamma)
    idx_1 = index_1_raw(delta, gamma, emg_epoch)
    idx_2 = index_2_raw(delta, theta, sigma)
    idx_3 = index_3_raw(delta, theta, gamma)
    idx_4 = index_4_raw(delta, theta)
    idx_r_noeog = index_R_raw(delta, theta, sigma, emg_epoch, gamma)
    idx_n_eog = index_N_eog_raw(gamma, eog_03_45_epoch)

    # ===== STEP 3: Stack all 15 features (raw base features + indices + complexity) =====
    features = np.column_stack((
        idx_w, idx_r, idx_n,
        idx_1, idx_2, idx_3, idx_4,
        delta, theta, aperiodic, dfa, mse, eog_epoch,
        idx_r_noeog, idx_n_eog,
    ))

    # ===== STEP 4: Normalize each feature with a wide percentile range =====
    for col_idx in range(features.shape[1]):
        features[:, col_idx] = wei_normalizing(features[:, col_idx])

    # ===== STEP 5: Optional log compression =====
    # mcRBM training now standardizes data separately, so skipping log1p preserves more spread.
    if apply_log1p:
        features = np.log1p(np.clip(features, a_min=0.0, a_max=None))

    if return_raw:
        # Return raw (unnormalized) features alongside normalized ones
        raw_features = np.column_stack((
            idx_w, idx_r, idx_n,
            idx_1, idx_2, idx_3, idx_4,
            delta, theta, aperiodic, dfa, mse, eog_epoch,
            idx_r_noeog, idx_n_eog,
        ))
        return features, mapped_scores, raw_features
    else:
        return features, mapped_scores