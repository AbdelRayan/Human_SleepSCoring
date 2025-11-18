from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from fooof import FOOOF
from joblib import Parallel, delayed
from matplotlib import ticker
from neurodsp.aperiodic import compute_irasa, fit_irasa, compute_fluctuations
from scipy.signal import butter, filtfilt, decimate, find_peaks, welch, savgol_filter
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from specparam import SpectralModel
import seaborn as sns
import EntropyHub as EH
import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    """Temporarily suppress all stdout printing."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

Red = '#d13838'
Blue = '#127be3'
DarkBlue = '#09217a'
LightBlue = '#9cd4ff'
Yellow = '#eff250'
Orange = '#faac11'
Purple = '#a170fd'

state_labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

def raw_to_epochs(data, sf, epoch):
    # total amount of samples per epoch
    samples_per_epoch = int(epoch * sf)
    # number of epochs in data
    n_epochs = len(data) // samples_per_epoch
    # cut data to full epochs
    cropped_data = data[:n_epochs * samples_per_epoch]
    # reshape to epochs
    epochs = cropped_data.reshape(n_epochs, samples_per_epoch)
    print("New data shape:", epochs.shape)

    return epochs

def calc_fractal_component(sleep_states, epoched_data, fs, f_range):
    # calculate aperiodic for all epochs per state
    irasa_by_state = defaultdict(list)

    for eeg_epoch, state in zip(epoched_data, sleep_states):
        freqs, psd_aperiodic, _ = compute_irasa(eeg_epoch, fs, f_range=f_range)
        irasa_by_state[state].append([freqs, psd_aperiodic])

    # calculate aperiodic mean per state
    irasa_mean_by_state = {}

    for state, irasa_list in irasa_by_state.items():
        aperiodics = np.array([ap for _, ap in irasa_list])
        freqs = irasa_list[0][0]
        mean_aperiodic = np.mean(aperiodics, axis=0)
        #sem_aperiodic = np.std(aperiodics, axis=0) / np.sqrt(aperiodics.shape[0])
        irasa_mean_by_state[state] = (freqs, mean_aperiodic)
        #irasa_mean_by_state[state] = (freqs, mean_aperiodic, sem_aperiodic)

    # remove index 5, which corresponds to movement
    if 5 in irasa_mean_by_state:
        del irasa_mean_by_state[5]

    return irasa_mean_by_state

def calc_slopes(epoched_data, fs, f_range, states):
    epoch_slopes = []
    valid_states = []

    # calculate slope for each epoch
    for i, epoch in enumerate(epoched_data):
        freqs, psd_aperiodic, _ = compute_irasa(epoch, fs, f_range=f_range)

        # Clean: remove invalid or non-positive values
        valid = np.isfinite(psd_aperiodic) & (psd_aperiodic > 0)
        freqs = freqs[valid]
        psd_aperiodic = psd_aperiodic[valid]

        # Skip if not enough valid data points
        if len(freqs) < 5:
            continue

        try:
            intercept, slope = fit_irasa(freqs, psd_aperiodic)
            epoch_slopes.append(slope)
            valid_states.append(states[i])  # keep corresponding state
        except Exception:
            continue

    if len(epoch_slopes) == 0:
        raise ValueError("No valid slopes were computed — check your data or f_range.")

    # Convert to arrays
    epoch_slopes = np.array(epoch_slopes)
    valid_states = np.array(valid_states)

    # Z-score raw slopes
    raw_slopes = zscore(epoch_slopes)
    min_len = min(len(raw_slopes), len(valid_states))
    raw_slopes = raw_slopes[:min_len]
    valid_states = valid_states[:min_len]

    # Adjust window length if smaller than data length
    window_length = min(101, len(raw_slopes) // 2 * 2 + 1)  # must be odd
    smoothed_slopes = savgol_filter(raw_slopes, window_length, polyorder=5, mode='interp')
    smoothed_slopes = zscore(smoothed_slopes)

    # Mean slope per state
    mean_slope_per_state = {}
    smoothed_mean_slope_per_state = {}

    for state in np.unique(valid_states):
        mask = valid_states == state
        mean_slope_per_state[state] = np.nanmean(raw_slopes[mask])
        smoothed_mean_slope_per_state[state] = np.nanmean(smoothed_slopes[mask])

    return raw_slopes, smoothed_slopes, mean_slope_per_state, smoothed_mean_slope_per_state

def wei_normalizing(data):
    data = np.array(data)

    bottom = data[data <= np.nanpercentile(data, 10, axis=0)]
    top = data[data >= np.nanpercentile(data, 90, axis=0)]

    bottom_avg = np.average(bottom) if len(bottom) > 0 else 0
    top_avg = np.average(top) if len(top) > 0 else 1

    denom = top_avg - bottom_avg if top_avg != bottom_avg else 1
    normalized_data = (data - bottom_avg) / denom
    normalized_data = np.clip(normalized_data, 0.05, 1)

    return normalized_data

def N_feature(EOG1, EOG2, epoch_length, fs):
    c_features = np.array([])
    EOG1 = decimate(EOG1, int(fs/50))
    EOG2 = decimate(EOG2, int(fs/50))
    fs = 50
    b, a = butter(4, [0.3 / (0.5 * fs), 0.45 / (0.5 * fs)], btype='band')
    EOG1 = filtfilt(b, a, EOG1)
    EOG2 = filtfilt(b, a, EOG2)
    cross_cor_val = cross_correlation(EOG1, EOG2, epoch_length, fs, 0)
    auto_corr_val = auto_correlation_slope(EOG1, epoch_length, fs)
    for count, ac in enumerate(auto_corr_val):
        feature = (1-ac) * cross_cor_val[count]
        c_features = np.append(c_features, feature)
    return c_features

def normalized_cross_correlation(epoch1, epoch2):
    # zero-mean
    x = epoch1 - np.mean(epoch1)
    y = epoch2 - np.mean(epoch2)
    denom = np.sqrt(np.sum(x*x) * np.sum(y*y))
    if denom == 0:
        return 0.0
    return np.sum(x * y) / denom

def cross_correlation(EOG1, EOG2, epoch_length, fs, lag=0):
    samples = int(epoch_length * fs)
    out = []
    for i in range(0, len(EOG1) - len(EOG1) % samples, samples):
        e1 = EOG1[i:i + samples]
        e2 = EOG2[i:i + samples]
        # Use normalized cross-correlation (value in [-1,1])
        out.append(normalized_cross_correlation(e1, e2))
    return np.array(out)

def auto_correlation_slope(EOG, epoch_length, fs, min_slope_idx=1):
    samples = int(epoch_length * fs)
    slopes = []
    for i in range(0, len(EOG) - len(EOG) % samples, samples):
        epoch = EOG[i:i + samples]
        ac_full = np.correlate(epoch, epoch, mode='full')
        center = len(ac_full) // 2
        ac = ac_full[center:]  # non-negative lags
        # normalize ac so ac[0] == 1 (helps numeric stability)
        if ac[0] == 0:
            slopes.append(np.nan)
            continue
        ac = ac / ac[0]
        peaks, _ = find_peaks(ac)
        # find first peak with index >= min_slope_idx
        valid_peaks = [p for p in peaks if p >= min_slope_idx]
        if len(valid_peaks) < 1:
            slopes.append(np.nan)
            continue
        first_peak_idx = valid_peaks[0]
        # slope between lag 0 and first_peak_idx
        slope = (ac[first_peak_idx] - ac[0]) / first_peak_idx
        slopes.append(slope)
    return np.array(slopes)

def rem_feature(EOG1, EOG2, epoch_length, fs,
                slope_tol=1e-6, max_inv=1e6, clip_abs=None):
    # filter (check nyquist and signal length outside)
    nyq = 0.5 * fs
    low = 0.3 / nyq
    high = 35.0 / nyq
    if not (0 < low < 1 and 0 < high < 1 and low < high):
        raise ValueError("Filter cutoff frequencies invalid for fs={}".format(fs))
    b, a = butter(4, [low, high], btype='band')
    # be careful: filtfilt requires signal length > padlen (approx 3*max(len(a),len(b)))
    if len(EOG1) < 20 or len(EOG2) < 20:
        raise ValueError("Signals too short for reliable filtering.")
    EOG1_f = filtfilt(b, a, EOG1)
    EOG2_f = filtfilt(b, a, EOG2)

    cross_cor_val = cross_correlation(EOG1_f, EOG2_f, epoch_length, fs, 0)
    auto_slope = auto_correlation_slope(EOG1_f, epoch_length, fs)

    features = []
    for cc, slope in zip(cross_cor_val, auto_slope):
        if np.isnan(slope) or abs(slope) < slope_tol:
            inv = 0.0  # or np.nan depending on whether you want to keep epoch
        else:
            inv = 1.0 / slope
            # guard against absurdly large values
            if not np.isfinite(inv):
                inv = 0.0
            if max_inv is not None:
                if inv > max_inv:
                    inv = max_inv
                elif inv < -max_inv:
                    inv = -max_inv
        # keep sign of cross-corr but use normalized cc in [-1,1]
        feat = np.sign(cc) * inv
        features.append(feat)

    feats = np.array(features)
    if clip_abs is not None:
        feats = np.clip(feats, -clip_abs, clip_abs)
    return feats

def index_N(delta, alpha, EMG, EOG1, EOG2, epoch_length, fs):
    eog_features = wei_normalizing(N_feature(EOG1, EOG2, epoch_length, fs))
    np.convolve(
        np.convolve(
            np.convolve(eog_features, np.ones(5) / 5, mode='same'),
            np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )
    alt_index_n = np.array([])
    for i in range(len(delta)):
        value = (eog_features[i] * delta[i]) / (alpha[i] * EMG[i])
        alt_index_n = np.append(alt_index_n, [value])

    return alt_index_n

def index_R(delta, sigma, EMG, EOG1, EOG2, epoch_length, fs):
    eog_features = wei_normalizing(rem_feature(EOG1, EOG2, epoch_length, fs))
    eog_features = np.convolve(
        np.convolve(
            np.convolve(eog_features, np.ones(5)/5, mode='same'),
            np.ones(5)/5, mode='same'),
        np.ones(5)/5, mode='same'
    )
    alt_index_r = np.array([])
    for i in range(len(delta)):
        value = (eog_features[i] * eog_features[i]) / (EMG[i] * EMG[i] * delta[i] * sigma[i])
        alt_index_r = np.append(alt_index_r, [value])

    return alt_index_r

def index_W(theta, gamma, EMG):
  index_w = np.array([])
  for i in range(len(theta)):
    value = EMG[i]*EMG[i]*((gamma[i])/(theta[i]))
    index_w = np.append(index_w, [value])
  return index_w

def normalised_powers(EMG_norm, noise_norm, delta_norm, theta_norm, sigma_norm, beta_norm, gamma_norm, alpha_norm,
                      hypno_epochs, sleep_scoring, output_dir):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, 1, sharex=True, figsize = (28, 20))
    epochs = np.arange(len(EMG_norm))
    # Plot data on each subplot
    ax1.plot(epochs, EMG_norm, label='EMG', color = 'black')
    ax2.plot(epochs, noise_norm, label='Noise power', color = 'black')
    ax3.plot(epochs, delta_norm, label='Delta power', color = 'black')
    ax4.plot(epochs, theta_norm, label='Theta power', color = 'black')
    ax5.plot(epochs, sigma_norm, label='Sigma power', color = 'black')
    ax6.plot(epochs, beta_norm, label='Beta power', color = 'black')
    ax7.plot(epochs, gamma_norm, label='Gamma power', color = 'black')
    ax8.plot(epochs, alpha_norm, label='Alpha power', color = 'black')
    ax9.plot(hypno_epochs, sleep_scoring, label='Mapped scores', color = 'black')


    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')
    ax4.legend(loc='upper right')
    ax5.legend(loc='upper right')
    ax6.legend(loc='upper right')
    ax7.legend(loc='upper right')
    ax8.legend(loc='upper right')

    ax1.set_ylabel('Normalised amplitude')
    ax2.set_ylabel('Normalised power')
    ax3.set_ylabel('Normalised power')
    ax4.set_ylabel('Normalised power')
    ax5.set_ylabel('Normalised power')
    ax6.set_ylabel('Normalised power')
    ax7.set_ylabel('Normalised power')
    ax9.set_ylabel('States')
    # ax8.set_yticks([0, 1, 2, 3, 4])
    # ax8.set_yticklabels(state_labels)
    ax9.invert_yaxis()
    ax9.set_xlabel('Epochs')
    # Adjust layout to avoid overlap
    plt.tight_layout()
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    ax2.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    ax3.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    ax4.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    ax5.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    ax6.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    ax7.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    ax8.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    ax9.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)

    plt.savefig(f"{output_dir}/Normalised_powers.svg", format='svg', dpi=300)
    # Show the plots
    # plt.show()

def normalised_EMG(EMG_norm, output_dir):
    fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(28, 5))
    epochs = np.arange(len(EMG_norm))
    # Plot data on each subplot
    ax1.plot(epochs, EMG_norm, label='', color='black')

    # Add legends
    ax1.legend()
    ax1.set_title('Normalised EMG')
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Normalised_EMG.svg", format='svg', dpi=300)
    # Show the plots
    # plt.show()

def raw_signals(states, raw_hpc, raw_pfc, pfc_tag, hpc_tag, output_dir, fs, epoch_length):
    upsampled_states = np.repeat(states, fs * epoch_length)

    target_len = len(raw_hpc)

    if len(upsampled_states) < target_len:
        # pad with zeros
        pad_len = target_len - len(upsampled_states)
        upsampled_states = np.concatenate((upsampled_states, np.zeros(pad_len)))
    elif len(upsampled_states) > target_len:
        # trim to match
        upsampled_states = upsampled_states[:target_len]
    x = np.arange(len(raw_hpc)) / 250

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(28, 12))

    # Plot data on each subplot
    ax1.plot(x, raw_pfc, label='', color='black')
    ax2.plot(x, raw_hpc, label='', color='black')
    ax3.plot(x, upsampled_states, label='', color='black')

    # Add legends
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # Add axis titles
    ax1.set_title(f'RAW {pfc_tag} Signal')
    ax2.set_title(f'RAW {hpc_tag} Signal')
    ax3.set_title('U-sleep Scoring')

    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Amplitude')
    ax3.set_ylabel('States')
    ax3.set_xlabel('Time (s)')
    state_labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.set_yticklabels(state_labels)
    ax3.invert_yaxis()
    # Set y-axis limits (example values, adjust as needed)
    ax1.set_ylim(-0.001, 0.001)  # Replace with appropriate limits for your data
    ax2.set_ylim(-0.001, 0.001)  # Replace with appropriate limits for your data
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    ax2.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    ax3.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.savefig(f"{output_dir}/RAW_signals_Hypnogram.svg", format='svg', dpi=300)
    # Show the plots
    # plt.show()

def indices_vs_hypnogram(epochs, hypno_epochs, index_w, index_n, index_r, mapped_scores, output_dir):
    """
    Plot smoothed Wei indices (W, N, R) alongside hypnogram scores.
    Ensures vertical gridlines for every x-tick.
    Requires a pre-defined smooth_and_norm() function.
    """

    # Smooth and normalize all indices using your helper
    index_w_smoothed = smooth_and_norm(index_w)
    index_r_smoothed = smooth_and_norm(index_r)
    index_n_smoothed = smooth_and_norm(index_n)


    # Time axes
    times = np.arange(len(epochs))
    hypno_times = np.arange(len(hypno_epochs))

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(40, 16))
    # Plot indices
    ax1.plot(epochs[times], index_w_smoothed[times], label='Index W', color='black')
    ax1.plot(epochs[times], index_n_smoothed[times], label='Index N', color='blue')
    ax1.plot(epochs[times], index_r_smoothed[times], label='Index R', color='red')

    # Plot hypnogram
    ax2.plot(hypno_epochs[hypno_times], mapped_scores[hypno_times],
             label='Mapped scores', color='gray')

    # Legends
    ax1.legend()
    ax2.legend()

    # Titles and labels
    ax1.set_title('New indices')
    ax2.set_title('Mapped scores')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(state_labels)
    ax2.invert_yaxis()

    # Define evenly spaced x-ticks (increase num for denser grid)
    xticks = np.linspace(epochs[0], epochs[-1], num=40)

    # Apply fixed locator so every tick gets a grid line
    for ax in (ax1, ax2):
        ax.set_xticks(xticks)
        ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
        ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.6)
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)

    # Layout and save
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_new_indices_vs_wei.svg', format='svg')
    plt.close()
    # plt.show()

def smooth_and_norm(x):
    """Log-transform, normalize, and triple-smooth an array."""
    x_log = np.log(x)
    x_norm = wei_normalizing(x_log)
    kernel = np.ones(5) / 5
    for _ in range(3):  # triple smoothing
        x_norm = np.convolve(x_norm, kernel, mode='same')
    return x_norm

def extract_index_values_per_state(index_n, index_r, index_w, mapped_scores):
    """
    Extract mean index values per sleep state for three indices (W, R, N).
    Returns dictionaries ready for plotting.
    """
    # Preprocess (smooth + z-score)
    smoothed = {
        'w': smooth_and_norm(index_w),
        'r': smooth_and_norm(index_r),
        'n': smooth_and_norm(index_n)
    }
    unique_states = np.unique(mapped_scores)
    print("Unique mapped states:", unique_states)
    # Map numerical sleep scores to labels
    score_labels = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    mapped_scores = [score_labels[s] for s in mapped_scores if s in score_labels]

    # Initialize containers
    states = ['Wake', 'N1', 'N2', 'N3', 'REM']
    indices = {state: {'w': [], 'r': [], 'n': []} for state in states}

    # Fill index values per state
    for i, state in enumerate(mapped_scores):
        if state in indices:
            for k in smoothed.keys():
                indices[state][k].append(smoothed[k][i])

    # Compute per-state means (handle empty lists safely)
    means = {
        state: {k: np.mean(v) if len(v) > 0 else np.nan for k, v in vals.items()}
        for state, vals in indices.items()
    }

    return means

def plot_index_barplot(means, output_dir):
    """
    Plot barplot of index means per state.
    """
    states = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # Extract values for plotting
    values_w = [means[s]['w'] for s in states]
    values_r = [means[s]['r'] for s in states]
    values_n = [means[s]['n'] for s in states]

    # Create figure
    plt.figure(figsize=(12, 6))
    fontsize = 5
    x = np.arange(len(states))
    bar_width = 0.25

    plt.bar(x - bar_width, values_w, width=bar_width, label='Index W', color='black')
    plt.bar(x, values_r, width=bar_width, label='Index R', color='red')
    plt.bar(x + bar_width, values_n, width=bar_width, label='Index N', color='blue')

    plt.xlabel('Sleep Stages')
    plt.ylabel('Average Index Values')
    plt.title('Wei Indices per Sleep State')
    plt.xticks(x, states)
    plt.legend(loc='upper center', bbox_to_anchor=(0.3, 1))

    # Add numeric labels
    for i in range(len(states)):
        for offset, vals, color in zip(
            [-bar_width, 0, bar_width],
            [values_w, values_r, values_n],
            ['black', 'red', 'blue']
        ):
            if not np.isnan(vals[i]):
                plt.text(x[i] + offset, vals[i] + 0.02, f"{vals[i]:.2f}",
                         ha='center', va='bottom', fontsize=fontsize, color=color)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_new_indices_vs_wei_bar.svg', format='svg')
    # plt.show()

def index_pca(index_n, mapped_scores_old, index_r, index_w, output_dir):

    index_w_smoothed = smooth_and_norm(index_w)
    index_r_smoothed = smooth_and_norm(index_r)
    index_n_smoothed = smooth_and_norm(index_n)

    difference = len(index_w_smoothed) - len(mapped_scores_old)

    if difference > 0:
        index_w_smoothed = index_w_smoothed[:-difference]
        index_r_smoothed = index_r_smoothed[:-difference]
        index_n_smoothed = index_n_smoothed[:-difference]

    arrays = [
        index_w_smoothed,
        index_r_smoothed,
        index_n_smoothed,
        mapped_scores_old
    ]
    # print(f'epochs: {X.shape}')   # rows = number of samples (dots), columns = features
    stage_map = {
        0.0: "wake",
        1.0: "n1",
        2.0: "n2",
        3.0: "n3",
        4.0: "rem"
    }
    array = np.column_stack(arrays)
    df = pd.DataFrame(array)

    x = df[[0, 1, 2]]
    y = df[3]

    scaler = StandardScaler()
    X = x.to_numpy().astype(float)

    # Replace NaNs with 0 (or another strategy like forward-fill)
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    print(f'PCA data points:{X_pca.shape}')
    # Colorblind-friendly palette
    colors = ['#0072B2', '#E69F00', '#D55E00', '#CC79A7', '#F0E442']

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(y)):
        stage_name = stage_map.get(label, str(label))  # fallback in case of unexpected value
        plt.scatter(
            X_pca[y == label, 0],
            X_pca[y == label, 1],
            label=stage_name,
            color=colors[i % len(colors)],
            alpha=0.8
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Sleep Stages")
    plt.legend(title="Sleep Stage")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Index_PCA_subject_2.svg", format='svg')
    # plt.show()

def prepare_aperiodic_violin_data(valid_states, normalized_exponents):
    """
    Prepare data for violin plot of aperiodic exponents per sleep state.

    Returns:
        df_plot: DataFrame ready for seaborn plotting
        data_for_violin: List of arrays for each state (0..4)
        all_states: List of state indices [0..4]
        labels: Sleep stage labels ['W', 'N1', 'N2', 'N3', 'REM']
    """
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    all_states = list(range(5))  # 0..4

    df = pd.DataFrame({'state': valid_states, 'aperiodic': normalized_exponents})

    # Build data per state for violin plotting
    data_for_violin = [df.loc[df['state'] == s, 'aperiodic'].values for s in all_states]

    counts = [len(d) for d in data_for_violin]
    print("Counts per state (0..4):", counts)
    print("Unique states present:", sorted(df['state'].unique()))

    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    return df_plot, data_for_violin, all_states, labels

def prepare_dfa_violin_data(valid_states, fs, length, lfp_PFC):
    """
    Prepare data for violin plot of dfa per sleep state.

    Returns:
        df_plot: DataFrame ready for seaborn plotting
        data_for_violin: List of arrays for each state (0..4)
        all_states: List of state indices [0..4]
        labels: Sleep stage labels ['W', 'N1', 'N2', 'N3', 'REM']
    """
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    all_states = list(range(5))  # 0..4
    window_size = step_size = fs * length
    num_windows = (len(lfp_PFC) - window_size) // step_size + 1

    dfa_exponents = []
    time_stamps = []

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        segment = lfp_PFC[start:end]

        _, _, exp_window = compute_fluctuations(segment, fs, n_scales=10,
                                                min_scale=0.05, max_scale=4.0)

        dfa_exponents.append(exp_window)
        time_stamps.append((start + end) / 2 / fs)

    dfa_exponents = np.array(dfa_exponents)
    window_length = 11 if len(dfa_exponents) >= 11 else len(dfa_exponents) | 1  # ensure it's odd
    polyorder = 4

    smoothed_dfa = savgol_filter(dfa_exponents, window_length=window_length, polyorder=polyorder)
    normalized_dfa = 2 * ((smoothed_dfa - min(smoothed_dfa)) / (max(smoothed_dfa) - min(smoothed_dfa))) - 1
    min_length = min(len(normalized_dfa), len(valid_states))
    valid_states = valid_states[:min_length]
    normalized_dfa = normalized_dfa[:min_length]
    df = pd.DataFrame({'state': valid_states, 'dfa': normalized_dfa})

    # Build data per state for violin plotting
    data_for_violin = [df.loc[df['state'] == s, 'dfa'].values for s in all_states]

    counts = [len(d) for d in data_for_violin]
    print("Counts per state (0..4):", counts)
    print("Unique states present:", sorted(df['state'].unique()))

    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    return df_plot, data_for_violin, all_states, labels, normalized_dfa

def plot_aperiodic_violin(df_plot, data_for_violin, all_states, labels, output_dir):
    """
    Generate a violin plot of aperiodic exponents per sleep state.

    df_plot: DataFrame from prepare_aperiodic_violin_data
    data_for_violin: List of arrays per state
    all_states: List of state indices [0..4]
    labels: List of labels ['W', 'N1', 'N2', 'N3', 'REM']
    output_dir: Directory to save plot
    """
    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    palette = [colors[s] for s in all_states]

    plt.figure(figsize=(8, 6))

    # Violin plot
    ax = sns.violinplot(
        x='state', y='aperiodic', data=df_plot,
        order=all_states,
        palette=palette,
        cut=0,
        bw='scott',
        inner=None
    )

    # Optional jittered scatter points
    sns.stripplot(
        x='state', y='aperiodic', data=df_plot,
        order=all_states,
        color='k', size=1.5, jitter=0.15, alpha=0.3
    )

    # Overlay medians
    medians = [np.nanmedian(d) if len(d) > 0 else np.nan for d in data_for_violin]
    for i, m in enumerate(medians):
        if not np.isnan(m):
            plt.plot(i, m, marker='o', color='white', markeredgecolor='black',
                     markersize=6, zorder=10)

    # Cosmetics
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State')
    ax.set_ylabel('Normalized mean aperiodic fit')
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('Aperiodic fit per Sleep State (violin)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()

    # plt.savefig(f"{output_dir}/Aperiodic_fit_violin.svg", format="svg")
    # plt.show()

def aperiodic_fit(pfc_data, states, fs, raw_pfc, output_dir):
    # --- Parameters ---
    window_size = 10 * fs
    lfp_PFC = np.ravel(pfc_data)
    sleep_states = states.flatten()
    step_size = window_size

    num_windows = len(lfp_PFC) // window_size
    time_stamps = np.arange(num_windows) * 10  # adjust timing if needed
    window_length = 11
    polyorder = 4

    # --- Prepare windows ---
    window_data = [raw_pfc[i * window_size:(i + 1) * window_size]
                   for i in range(num_windows)]

    # --- Diagnostic safe-fit wrapper ---
    def safe_aperiodic_fit(window, idx, aperiodic_fit_fn):
        try:
            if len(window) == 0 or np.std(window) < 1e-12:
                return np.nan, "flat_or_empty"

            if not np.all(np.isfinite(window)):
                return np.nan, "nonfinite_input"

            exp = aperiodic_fit_fn(window)
            if not np.isfinite(exp):
                return np.nan, "nonfinite_exponent"

            return float(exp), "ok"

        except Exception as e:
            return np.nan, f"exception: {str(e)}"

    # --- Compute exponent ---
    def aperiodic_fit(window_data):
        freqs, psd = welch(window_data, fs=fs, nperseg=1024)
        mask = (freqs <= 100)
        freqs, psd = freqs[mask], psd[mask]
        fm = SpectralModel(min_peak_height=0.05,
                           aperiodic_mode='fixed', verbose=False)
        fm.fit(freqs, psd)
        aperiodic = fm.get_params('aperiodic')[1]
        return aperiodic

    # --- Run fits in parallel ---
    results = Parallel(n_jobs=-1)(
        delayed(safe_aperiodic_fit)(w, i, aperiodic_fit)
        for i, w in enumerate(window_data)
    )

    # Extract exponent values and diagnostic labels
    aperiodic_exponents, statuses = zip(*results)
    aperiodic_exponents = np.array(aperiodic_exponents)

    # --- Ensure states and timestamps match number of windows ---
    if len(aperiodic_exponents) > len(states):
        diff = len(aperiodic_exponents) - len(states)
        print(f"Padded {diff} state values.")
        for _ in range(diff):
            states = np.append(states, states[-1])

    states = states[:len(aperiodic_exponents)]
    time_stamps = time_stamps[:len(aperiodic_exponents)]

    # --- Report problems ---
    problem_windows = [(i, s) for i, s in enumerate(statuses) if s != "ok"]
    print(f"Problematic windows: {len(problem_windows)}")
    print("First 10 problematic windows:", problem_windows[:10])

    # ==============================================================
    #   BETTER SOLUTION: Repair invalid exponent values in place
    # ==============================================================

    # Use pandas for interpolation mechanics
    exp_series = pd.Series(aperiodic_exponents)

    # 1. Replace NaNs via interpolation
    exp_series = exp_series.interpolate(method='linear', limit_direction='both')

    # 2. Fill any remaining NaNs using global median
    global_median = exp_series.median()
    exp_series = exp_series.fillna(global_median)

    # 3. Soft thresholding (clip, do NOT delete indexes)
    threshold_min = np.percentile(exp_series, 2)
    threshold_max = np.percentile(exp_series, 98)
    exp_series = exp_series.clip(lower=threshold_min, upper=threshold_max)

    # Final repaired array
    repaired_exponents = exp_series.to_numpy()

    # --- Smooth + normalize (all indices preserved) ---
    window_length_sg = (
        window_length
        if len(repaired_exponents) >= window_length
        else (len(repaired_exponents) | 1)
    )

    smoothed_exponents = savgol_filter(
        repaired_exponents,
        window_length=window_length_sg,
        polyorder=polyorder
    )

    # Normalization to [-1, 1]
    normalized_exponents = (
        2 * ((smoothed_exponents - smoothed_exponents.min()) /
             (smoothed_exponents.max() - smoothed_exponents.min()))
        - 1
    )

    # --- Plot ---
    plt.figure(figsize=(18, 5))
    plt.plot(time_stamps, normalized_exponents,
             marker='.', linestyle='-', color=DarkBlue)
    plt.xlabel('Time (s)')
    plt.ylabel('Aperiodic Exponent')
    plt.title('Normalized Aperiodic Fit Over Time (Repaired, No Dropping)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Aperiodic_fit.svg", format='svg')

    return normalized_exponents, smoothed_exponents, states, repaired_exponents

def fractal_power_component(states, subject, raw_pfc, output_dir, epoch, sf, f_range=(0.3, 30),
                            fmax = 30, fmin=0.3):

    subject_fractal_data = {}
    subject_slope_data = {}
    subject_states = {}

    subject_states[subject] = states

    # get eeg data in epochs
    eeg_in_epochs = raw_to_epochs(raw_pfc, sf, epoch)

    # calculate fractal component
    subject_fractal_data[subject] = calc_fractal_component(states, eeg_in_epochs, sf, f_range)

    # calculate slopes
    subject_slope_data[subject] = calc_slopes(eeg_in_epochs, sf, f_range, states)

    aperiodic_by_state = defaultdict(list)
    freqs_ref = {}

    for subject, subject_dict in subject_fractal_data.items():
        for state, (freqs, aperiodic) in subject_dict.items():
            aperiodic_by_state[state].append(aperiodic)
            freqs_ref[state] = freqs

    mean_by_state = {}
    sem_by_state = {}

    for state, aperiodic_list in aperiodic_by_state.items():
        arr = np.stack(aperiodic_list, axis=0)
        mean_by_state[state] = (freqs_ref[state], np.mean(arr, axis=0))
        sem_by_state[state] = (freqs_ref[state], np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0]))

    ### slopes

    raw_slope_by_state = defaultdict(list)
    smoothed_slope_by_state = defaultdict(list)

    for subject, (_, _, raw_slopes, smoothed_slopes) in subject_slope_data.items():
        for state, slope in raw_slopes.items():
            raw_slope_by_state[state].append(slope)
        for state, slope in smoothed_slopes.items():
            smoothed_slope_by_state[state].append(slope)

    mean_raw_slope_by_state = {}
    sem_raw_slope_by_state = {}
    mean_smoothed_slope_by_state = {}
    sem_smoothed_slope_by_state = {}

    for state, slope_list in raw_slope_by_state.items():
        slope_array = np.vstack(slope_list)  # shape: (n_subjects, n_epochs)
        mean_raw_slope_by_state[state] = np.mean(slope_array, axis=0)
        sem_raw_slope_by_state[state] = np.std(slope_array, axis=0) / np.sqrt(slope_array.shape[0])

    for state, slope_list in smoothed_slope_by_state.items():
        slope_array = np.vstack(slope_list)
        mean_smoothed_slope_by_state[state] = np.mean(slope_array, axis=0)
        sem_smoothed_slope_by_state[state] = np.std(slope_array, axis=0) / np.sqrt(slope_array.shape[0])
    # Colors and labels
    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    stage_labels = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

    plt.figure(figsize=(10, 6))

    for state in sorted(mean_by_state.keys()):
        freqs, mean_aperiodic = mean_by_state[state]
        _, sem_aperiodic = sem_by_state[state]

        plt.plot(
            freqs, mean_aperiodic,
            label=stage_labels.get(state, f"State {state}"),
            alpha=0.9,
            color=colors.get(state, 'gray')
        )

        # Shaded SEM area
        plt.fill_between(
            freqs,
            mean_aperiodic - sem_aperiodic,
            mean_aperiodic + sem_aperiodic,
            color=colors.get(state, 'gray'),
            alpha=0.15
        )

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-14, 1e-8)  # ✅ Fixed y-axis limits
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Fractal Power Component (IRASA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/fractal_power_component.svg", format='svg')
    # plt.show()
    return eeg_in_epochs, mean_raw_slope_by_state, mean_smoothed_slope_by_state

def slope_per_state(output_dir, mean_raw_slope_by_state, mean_smoothed_slope_by_state,
                    states, eeg_in_epochs, sf=250, f_range=(0.3, 30)):

    epoch_slopes = []
    valid_states = []

    for i, eeg_epoch in enumerate(eeg_in_epochs):
        freqs, psd_aperiodic, _ = compute_irasa(eeg_epoch, sf, f_range=f_range)

        # Remove invalid values
        valid = np.isfinite(psd_aperiodic) & (psd_aperiodic > 0)
        freqs = freqs[valid]
        psd_aperiodic = psd_aperiodic[valid]

        # Skip epochs with too few points
        if len(freqs) < 5:
            continue

        try:
            intercept, slope = fit_irasa(freqs, psd_aperiodic)
            epoch_slopes.append(slope)
            valid_states.append(states[i])
        except Exception:
            continue

    # Convert to arrays
    epoch_slopes = np.array(epoch_slopes)
    valid_states = np.array(valid_states)

    # Compute z-scores
    raw_slopes = zscore(epoch_slopes)
    min_len = min(len(raw_slopes), len(valid_states))
    raw_slopes = raw_slopes[:min_len]
    valid_states = valid_states[:min_len]

    # Adjust window length for savgol_filter
    window_length = min(101, len(raw_slopes) // 2 * 2 + 1)  # must be odd
    smoothed_slopes = savgol_filter(raw_slopes, window_length, polyorder=5, mode='interp')
    smoothed_slopes = zscore(smoothed_slopes)

    # Compute mean slopes per valid state
    mean_slope_per_state = {}
    smoothed_mean_slope_per_state = {}

    for state in np.unique(valid_states):
        mask = valid_states == state
        mean_slope_per_state[state] = np.nanmean(raw_slopes[mask])
        smoothed_mean_slope_per_state[state] = np.nanmean(smoothed_slopes[mask])

    # Exclude any absent states
    stages_sorted = sorted(mean_slope_per_state.keys())

    # Extract means per state (only if present)
    mean_slopes = [
        np.mean(mean_raw_slope_by_state[state])
        for state in stages_sorted if state in mean_raw_slope_by_state
    ]
    smoothed_mean_slopes = [
        np.mean(mean_smoothed_slope_by_state[state])
        for state in stages_sorted if state in mean_smoothed_slope_by_state
    ]

    # Plot only states that exist
    plt.figure(figsize=(7, 6))
    plt.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    plt.axhline(0, color='gray', linewidth=1, alpha=0.5, zorder=0)

    plt.scatter(stages_sorted, mean_slopes, color='black', marker='s', s=60, label='raw slope', zorder=2)
    plt.scatter(stages_sorted, smoothed_mean_slopes, color='green', marker='s', s=30, label='smoothed slope', zorder=2)
    plt.plot(stages_sorted, mean_slopes, color='black', linestyle='--', alpha=0.6, zorder=2)
    plt.plot(stages_sorted, smoothed_mean_slopes, color='green', linestyle='--', alpha=0.6, zorder=2)

    plt.ylabel('Z-normalized slope')
    plt.xlabel('Sleep Stage')
    plt.title('Slope per state')
    plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/slope_per_state.svg", format="svg")

    return smoothed_slopes

def fractal_slope_vs_hypnogram(subject, smoothed_slopes, output_dir, states, epoch_length_sec=10):
    peaks, _ = find_peaks(smoothed_slopes, distance=120, prominence=2)

    valleys, _ = find_peaks(-smoothed_slopes, distance=120, prominence=2)
    subj_states = states
    n_epochs = len(subj_states)
    time_axis = np.arange(n_epochs) * epoch_length_sec / 60  # minutes

    # define sleep stage colors
    # Replace 5 with 0
    # subj_states[subj_states == 5] = 0

    stage_colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    stage_labels = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

    # Parameters
    REM_code = 4
    min_rem_gap = 20  # ignore interruptions <= 1 epoch

    # --- First non-Wake ---
    non_wake_idx = np.where(subj_states != 0)[0]
    first_non_wake = non_wake_idx[0] if len(non_wake_idx) > 0 else None
    last_non_wake = non_wake_idx[-1] if len(non_wake_idx) > 0 else None

    # --- REM clusters ---
    is_rem = (subj_states == REM_code).astype(int)
    diff = np.diff(np.concatenate(([0], is_rem, [0])))
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0] - 1

    # Merge short interruptions
    merged_starts = [run_starts[0]] if len(run_starts) > 0 else []
    merged_ends = []
    for i in range(1, len(run_starts)):
        if run_starts[i] - run_ends[i - 1] - 1 <= min_rem_gap:
            continue
        else:
            merged_ends.append(run_ends[i - 1])
            merged_starts.append(run_starts[i])
    if len(run_ends) > 0:
        merged_ends.append(run_ends[-1])

    # Desired order for y-axis
    desired_order = ['W', 'REM', 'N1', 'N2', 'N3']

    # Map original numeric states to their labels
    stage_labels = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

    # Create a mapping: original numeric code -> new vertical position
    new_y_map = {label: i for i, label in enumerate(desired_order)}
    states_labels = np.array([stage_labels[s] for s in subj_states])
    states_new_y = np.array([new_y_map[s] for s in states_labels])

    # create plot layout (2 plots in 1 figure)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    ### plot 1: colored coded hypnogram
    #######################################################################
    # plot plain black line hypnogram
    # --- Create list of vertical line positions ---
    line_positions = [first_non_wake] + merged_ends if first_non_wake is not None else merged_ends
    # Add the last non-wake if it isn’t already in the list
    if last_non_wake is not None and last_non_wake not in line_positions:
        line_positions.append(last_non_wake)
    line_positions = [0] + line_positions + [len(subj_states) - 1]  # include start and end
    line_times = time_axis[line_positions]

    # --- Plot hypnogram ---
    ax1.step(time_axis, states_new_y, where='post', color='black', linewidth=0.5, zorder=2)

    # Shade regions between lines (alternating) and outside
    colors = ['lightgreen', 'lightblue']
    for i in range(1, len(line_times) - 2):
        start = line_times[i]
        end = line_times[i + 1]
        ax1.axvspan(line_times[i], line_times[i + 1], color=colors[i % 2], alpha=0.5, zorder=1)
        # Add cycle label at the center of the shaded region
        center = (start + end) / 2
        ax1.text(center, 0.5, f'Cycle {i}',
                 ha='center', va='bottom', fontsize=10, color='black')

    # Shade outside areas
    ax1.axvspan(time_axis[0], line_times[1], color='lightgray', alpha=0.3, zorder=0)  # before first non-Wake/cluster
    ax1.axvspan(line_times[-2], time_axis[-1], color='lightgray', alpha=0.3, zorder=0)  # after last cluster

    # Draw vertical lines
    for t in line_times[1:-1]:  # avoid start and end
        ax1.axvline(t, color='red', linestyle='--', alpha=0.7, zorder=3)

    # plot dots colored based on state
    for s_num in np.unique(subj_states):
        ids = np.where(subj_states == s_num)[0]
        ax1.scatter(time_axis[ids], states_new_y[ids],
                    color=stage_colors[s_num], label=stage_labels[s_num], s=10)

    # additional settings
    ax1.set_yticks(range(len(desired_order)))
    ax1.set_yticklabels(desired_order)
    ax1.set_ylabel('Sleep Stage')
    ax1.invert_yaxis()
    ax1.set_xlim(time_axis[0], time_axis[-1])
    ax1.set_title(f'Hypnogram - {subject}')

    difference = len(time_axis) - len(smoothed_slopes)
    ### plot 2: smoothed fractal slope
    ########################################################################
    if difference > 0:
        time_axis = time_axis[:-difference]
    ax2.plot(time_axis, smoothed_slopes, color='black', label='Fractal slope', linewidth=1)
    # color lines based on sleep state
    # Define colors for N3 and REM
    stage_colors = {3: 'green', 4: 'red'}

    # Loop over stages to color the segments
    for stage, color in stage_colors.items():
        # Find contiguous segments of this stage
        mask = (subj_states == stage)
        if not np.any(mask):
            continue  # skip if stage not present

        # Split into contiguous segments
        segments = np.split(time_axis, np.where(np.diff(mask.astype(int)) != 0)[0] + 1)
        slope_segments = np.split(smoothed_slopes, np.where(np.diff(mask.astype(int)) != 0)[0] + 1)

        for t_seg, s_seg, m_seg in zip(segments, slope_segments,
                                       np.split(mask, np.where(np.diff(mask.astype(int)) != 0)[0] + 1)):
            if np.any(m_seg):
                min_len = min(len(t_seg), len(s_seg), len(m_seg))
                ax2.plot(t_seg[:min_len][m_seg[:min_len]], s_seg[:min_len][m_seg[:min_len]], color=color, linewidth=3)

    # Solid line at 0
    ax2.axhline(0, color='gray', linewidth=1, zorder=1)

    # Dashed lines at +1 and -1
    ax2.axhline(1, color='lightgray', linestyle='--', linewidth=1, zorder=1)
    ax2.axhline(-1, color='lightgray', linestyle='--', linewidth=1, zorder=1)

    # additional settings
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Z-normalized fractal slope')
    ax2.set_ylim(-2, 2)
    ax2.set_xlim(time_axis[0], time_axis[-1])
    ax2.set_title(f'Fractal slopes - {subject}')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fractalslope_vs_hypnogram.svg", format='svg')
    # plt.show()

def fooof_report(output_dir, raw_pfc, fs):
    freqs, psd = welch(raw_pfc, fs=fs, nperseg=1024)
    # Initialize and fit FOOOF model
    fm = FOOOF(peak_width_limits=[2, 8], aperiodic_mode='fixed')
    # Generate the report (this creates a matplotlib figure)
    fm.report(freqs, psd, [1, 50], plt_log=True)
    # Save the report plot to a file (e.g., PNG or PDF)
    plt.savefig(f"{output_dir}/fooof_report.png", dpi=300, bbox_inches='tight')
    # plt.show()

def aperiodic_fit_bar(valid_states, normalized_exponents, output_dir):
    colors = ['royalblue', 'teal', 'purple', 'forestgreen', 'firebrick']

    df = pd.DataFrame({'state': valid_states, 'aperiodic': normalized_exponents})
    summary = df.groupby('state')['aperiodic'].agg(['mean', 'sem']).reset_index()
    print(summary)
    plt.figure(figsize=(7, 5))
    plt.bar(summary['state'], summary['mean'], yerr=summary['sem'],
            capsize=5, color=[colors[int(s)] for s in summary['state']], edgecolor='black', zorder=2, alpha=0.6)
    plt.xticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM'])
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Sleep State')
    plt.ylabel('Normalized mean aperiodic fit')
    plt.title('Aperiodic per Sleep State')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Aperiodic_fit_bar.svg", format='svg')
    # plt.show()


def aperiodic_per_state(normalized_exponents, states):
    """
    Compute mean aperiodic exponent per sleep state for a single subject.

    Args:
        normalized_exponents: 1D array of normalized aperiodic exponents per time window
        states: 1D array of corresponding sleep states (numeric, e.g., 0..4)

    Returns:
        mean_per_state: dict[state] = mean exponent for that state
    """
    normalized_exponents = np.array(normalized_exponents)
    states = np.array(states).astype(int)

    # Align lengths
    min_len = min(len(normalized_exponents), len(states))
    normalized_exponents = normalized_exponents[:min_len]
    states = states[:min_len]

    # Remove NaNs
    mask = ~np.isnan(normalized_exponents)
    normalized_exponents = normalized_exponents[mask]
    states = states[mask]

    mean_per_state = {}
    for state in np.unique(states):
        mean_per_state[state] = np.nanmean(normalized_exponents[states == state])

    return mean_per_state

def plot_averaged_aperiodic_violin(aperiodic_violin, output_dir):
    """
    Plot group-level aperiodic violin from concatenated data with SEM bars.
    """
    data_for_violin = aperiodic_violin['data_for_violin']
    all_states = aperiodic_violin['all_states']
    labels = aperiodic_violin['labels']
    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    palette = [colors[s] for s in all_states]

    # Build data list in order of states
    data_list = [data_for_violin[s] for s in all_states]

    plt.figure(figsize=(8, 6))

    # --- Violin plot (background layer) ---
    ax = sns.violinplot(
        data=data_list,
        palette=palette,
        cut=0,
        bw='scott',
        inner=None,
        alpha=0.5,
        zorder=2
    )

    # --- Overlay jittered scatter points ---
    for i, d in enumerate(data_list):
        x = np.random.normal(loc=i, scale=0.15, size=len(d))  # jitter
        ax.scatter(x, d, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    # --- Overlay medians and SEM ---
    for i, d in enumerate(data_list):
        d = np.array(d)
        d = d[~np.isnan(d)]
        if len(d) == 0:
            continue

        mean_val = np.nanmean(d)
        sem_val = np.nanstd(d) / np.sqrt(len(d))

        # Mean point
        plt.plot(i, mean_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=4)

        # SEM bar (vertical line with caps)
        plt.errorbar(
            i,
            mean_val,
            yerr=sem_val,
            color='black',
            capsize=4,
            elinewidth=1.5,
            markeredgewidth=1,
            zorder=3
        )

    # --- Aesthetics ---
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State')
    ax.set_ylabel('Normalized aperiodic exponent')
    ax.set_title('Aperiodic Fit per Sleep State (Mean ± SEM)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/Aperiodic_fit_violin_avg_sem.svg", format="svg")
    plt.show()

def plot_averaged_dfa_violin(dfa_violin, output_dir):
    """
    Plot group-level aperiodic violin from concatenated data with SEM bars.
    """
    data_for_violin = dfa_violin['data_for_violin']
    all_states = dfa_violin['all_states']
    labels = dfa_violin['labels']
    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    palette = [colors[s] for s in all_states]

    # Build data list in order of states
    data_list = [data_for_violin[s] for s in all_states]

    plt.figure(figsize=(8, 6))

    # --- Violin plot (background layer) ---
    ax = sns.violinplot(
        data=data_list,
        palette=palette,
        cut=0,
        bw='scott',
        inner=None,
        alpha=0.5,
        zorder=2
    )

    # --- Overlay jittered scatter points ---
    for i, d in enumerate(data_list):
        x = np.random.normal(loc=i, scale=0.15, size=len(d))  # jitter
        ax.scatter(x, d, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    # --- Overlay medians and SEM ---
    for i, d in enumerate(data_list):
        d = np.array(d)
        d = d[~np.isnan(d)]
        if len(d) == 0:
            continue

        mean_val = np.nanmean(d)
        sem_val = np.nanstd(d) / np.sqrt(len(d))

        # Mean point
        plt.plot(i, mean_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=4)

        # SEM bar (vertical line with caps)
        plt.errorbar(
            i,
            mean_val,
            yerr=sem_val,
            color='black',
            capsize=4,
            elinewidth=1.5,
            markeredgewidth=1,
            zorder=3
        )

    # --- Aesthetics ---
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State')
    ax.set_ylabel('Normalized DFA exponent')
    ax.set_title('DFA per Sleep State (Mean ± SEM)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/dfa_violin_avg_sem.svg", format="svg")
    plt.show()

def plot_averaged_mse_violin(mse_violin, output_dir):
    """
    Plot group-level aperiodic violin from concatenated data with SEM bars.
    """
    data_for_violin = mse_violin['data_for_violin']
    all_states = mse_violin['all_states']
    labels = mse_violin['labels']
    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    palette = [colors[s] for s in all_states]

    # Build data list in order of states
    data_list = [data_for_violin[s] for s in all_states]

    plt.figure(figsize=(8, 6))

    # --- Violin plot (background layer) ---
    ax = sns.violinplot(
        data=data_list,
        palette=palette,
        cut=0,
        bw='scott',
        inner=None,
        alpha=0.5,
        zorder=2
    )

    # --- Overlay jittered scatter points ---
    for i, d in enumerate(data_list):
        x = np.random.normal(loc=i, scale=0.15, size=len(d))  # jitter
        ax.scatter(x, d, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    # --- Overlay medians and SEM ---
    for i, d in enumerate(data_list):
        d = np.array(d)
        d = d[~np.isnan(d)]
        if len(d) == 0:
            continue

        mean_val = np.nanmean(d)
        sem_val = np.nanstd(d) / np.sqrt(len(d))

        # Mean point
        plt.plot(i, mean_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=4)

        # SEM bar (vertical line with caps)
        plt.errorbar(
            i,
            mean_val,
            yerr=sem_val,
            color='black',
            capsize=4,
            elinewidth=1.5,
            markeredgewidth=1,
            zorder=3
        )

    # --- Aesthetics ---
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State')
    ax.set_ylabel('Normalized MSE exponent')
    ax.set_title('MSE per Sleep State (Mean ± SEM)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/mse_violin_avg_sem.svg", format="svg")
    plt.show()

def plot_index_avg_violin_all(results, output_dir):
    """
    Create violin plots for each index (W, N, R, 1, 2, 3, 4)
    x-axis = Sleep states (w, n1, n2, n3, r)
    datapoints = per-night averages from all subjects/nights
    Saves each index plot as SVG and data as NPZ.
    """

    os.makedirs(output_dir, exist_ok=True)

    index_keys = ["W", "N", "R", "1", "2", "3", "4"]
    state_names = ["w", "n1", "n2", "n3", "r"]
    state_labels = ["Wake", "N1", "N2", "N3", "REM"]

    colors = {
        "w":  "royalblue",
        "n1": "teal",
        "n2": "purple",
        "n3": "forestgreen",
        "r":  "firebrick"
    }
    palette = [colors[s] for s in state_names]

    for idx in index_keys:

        # Collect data for each state
        data_for_violin = {sn: [] for sn in state_names}

        for subj, nights in results.items():
            for night, idx_dict in nights.items():
                if idx not in idx_dict:
                    continue
                state_vals = idx_dict[idx]
                for sn in state_names:
                    val = state_vals.get(sn, np.nan)
                    if not np.isnan(val):
                        data_for_violin[sn].append(val)

        # Ordered list for plotting
        data_list = [data_for_violin[sn] for sn in state_names]

        # --- Violin Plot ---
        plt.figure(figsize=(8, 6))
        ax = sns.violinplot(
            data=data_list,
            palette=palette,
            cut=0,
            bw='scott',
            inner=None,
            alpha=0.5,
            zorder=2
        )

        # Scatter jitter
        for i, d in enumerate(data_list):
            x = np.random.normal(loc=i, scale=0.15, size=len(d))
            ax.scatter(x, d, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

        # Means + SEM
        for i, d in enumerate(data_list):
            d = np.array(d)
            if len(d) == 0:
                continue
            mean_val = np.nanmean(d)
            sem_val = np.nanstd(d) / np.sqrt(len(d))

            plt.plot(i, mean_val, 'o', color='white', markeredgecolor='black',
                     markersize=6, zorder=4)
            plt.errorbar(i, mean_val, yerr=sem_val, color='black',
                         capsize=4, elinewidth=1.5, markeredgewidth=1, zorder=3)

        ax.set_xticks(range(len(state_labels)))
        ax.set_xticklabels(state_labels)
        ax.set_xlabel("Sleep State")
        ax.set_ylabel(f"Average Value of Index {idx}")
        ax.set_title(f"Index {idx} per Sleep State (Mean ± SEM)")
        plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
        plt.tight_layout()

        # Save plot
        plot_file = os.path.join(output_dir, f"index_{idx}_violin_avg_sem.svg")
        plt.savefig(plot_file, format="svg")
        plt.show()
        print(f"Saved plot: {plot_file}")

        # Save the underlying data
        data_file = os.path.join(output_dir, f"index_{idx}_data.npz")
        np.savez(data_file, **data_for_violin)
        print(f"Saved data: {data_file}")

def plot_signal_violin_all(results, output_dir, signal_keys=["noise", "theta", "delta"]):
    """
    Create violin plots for each signal (noise, theta, delta)
    across all subjects/nights per sleep state.
    Saves each plot as SVG and underlying data as NPZ.
    """

    os.makedirs(output_dir, exist_ok=True)

    state_names = ["w", "n1", "n2", "n3", "r"]
    state_labels = ["Wake", "N1", "N2", "N3", "REM"]

    colors = {
        "w":  "royalblue",
        "n1": "teal",
        "n2": "purple",
        "n3": "forestgreen",
        "r":  "firebrick"
    }
    palette = [colors[s] for s in state_names]

    for key in signal_keys:

        # Collect per-state values across all subjects/nights
        data_for_violin = {sn: [] for sn in state_names}

        for subj, nights in results.items():
            for night, signal_dict in nights.items():
                if key not in signal_dict:
                    continue
                state_vals = signal_dict[key]  # {"w":..., "n1":..., ...}
                for sn in state_names:
                    val = state_vals.get(sn, np.nan)
                    if not np.isnan(val):
                        data_for_violin[sn].append(val)

        # Ordered list for plotting
        data_list = [data_for_violin[sn] for sn in state_names]

        # --- Violin plot ---
        plt.figure(figsize=(8,6))
        ax = sns.violinplot(
            data=data_list,
            palette=palette,
            cut=0,
            bw='scott',
            inner=None,
            alpha=0.5,
            zorder=2
        )

        # Scatter jitter
        for i, d in enumerate(data_list):
            x = np.random.normal(loc=i, scale=0.15, size=len(d))
            ax.scatter(x, d, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

        # Means + SEM
        for i, d in enumerate(data_list):
            d = np.array(d)
            if len(d) == 0:
                continue
            mean_val = np.nanmean(d)
            sem_val = np.nanstd(d)/np.sqrt(len(d))

            plt.plot(i, mean_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=4)
            plt.errorbar(i, mean_val, yerr=sem_val, color='black', capsize=4,
                         elinewidth=1.5, markeredgewidth=1, zorder=3)

        # Labels & aesthetics
        ax.set_xticks(range(len(state_labels)))
        ax.set_xticklabels(state_labels)
        ax.set_xlabel("Sleep State")
        ax.set_ylabel(f"Average {key.capitalize()} Value")
        ax.set_title(f"{key.capitalize()} per Sleep State (Mean ± SEM)")
        plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
        plt.tight_layout()

        # Save
        plot_file = os.path.join(output_dir, f"{key}_violin_avg_sem.svg")
        plt.savefig(plot_file, format="svg")
        plt.show()
        print(f"Saved plot: {plot_file}")

        # Save underlying data
        data_file = os.path.join(output_dir, f"{key}_data.npz")
        np.savez(data_file, **data_for_violin)
        print(f"Saved data: {data_file}")

def dfa_plot(lfp_PFC, output_dir, length, fs):
    window_size = step_size = fs*length
    num_windows = (len(lfp_PFC) - window_size) // step_size + 1

    dfa_exponents = []
    time_stamps = []

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        segment = lfp_PFC[start:end]

        _, _, exp_window = compute_fluctuations(segment, fs, n_scales=10,
                                                min_scale=0.05, max_scale=4.0)

        dfa_exponents.append(exp_window)
        time_stamps.append((start + end) / 2 / fs)

    dfa_exponents = np.array(dfa_exponents)
    window_length = 11 if len(dfa_exponents) >= 11 else len(dfa_exponents) | 1  # ensure it's odd
    polyorder = 4

    smoothed_dfa = savgol_filter(dfa_exponents, window_length=window_length, polyorder=polyorder)
    normalized_dfa = 2 * ((smoothed_dfa - min(smoothed_dfa)) / (max(smoothed_dfa) - min(smoothed_dfa))) - 1
    plt.figure(figsize=(18, 5))
    plt.plot(time_stamps, normalized_dfa, marker='.', linestyle='-', color=Blue)
    plt.xlabel('Time (s)')
    plt.ylabel('DFA Exponent')
    plt.title('DFA Over Time - RGS14')
    plt.grid()
    plt.savefig(f"{output_dir}/DFA_over_time.svg", format='svg')
    # plt.show()

    return normalized_dfa

def dfa_per_state(normalized_dfa, states, output_dir):
    # --- Convert to numpy arrays ---
    dfa_values = np.array(normalized_dfa)
    valid_states = np.array(states).astype(int)

    # --- Trim to same length ---
    min_len = min(len(dfa_values), len(valid_states))
    dfa_values = dfa_values[:min_len]
    valid_states = valid_states[:min_len]

    # --- Remove NaNs ---
    nan_mask = ~np.isnan(dfa_values)
    dfa_values = dfa_values[nan_mask]
    valid_states = valid_states[nan_mask]

    # --- Compute z-scores ---
    raw_dfa = zscore(dfa_values, nan_policy='omit')

    # --- Smooth ---
    window_length = min(101, len(raw_dfa) // 2 * 2 + 1)  # must be odd
    smoothed_dfa = savgol_filter(raw_dfa, window_length, polyorder=5, mode='interp')
    smoothed_dfa = zscore(smoothed_dfa, nan_policy='omit')

    # --- Compute mean DFA per valid state ---
    mean_dfa_per_state = {}
    smoothed_mean_dfa_per_state = {}

    for state in np.unique(valid_states):
        mask = valid_states == state
        mean_dfa_per_state[state] = np.nanmean(raw_dfa[mask])
        smoothed_mean_dfa_per_state[state] = np.nanmean(smoothed_dfa[mask])

    print("Means per state:", mean_dfa_per_state)

    # --- Exclude any absent states ---
    stages_sorted = sorted(mean_dfa_per_state.keys())

    # --- Compute per-state means (for plotting) ---
    mean_dfas = [mean_dfa_per_state[s] for s in stages_sorted]
    smoothed_mean_dfas = [smoothed_mean_dfa_per_state[s] for s in stages_sorted]

    # --- Map numeric state codes to labels, if applicable ---
    state_labels_map = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    stage_labels = [state_labels_map.get(s, str(s)) for s in stages_sorted]

    # --- Plot only available states ---
    plt.figure(figsize=(7, 6))
    plt.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    plt.axhline(0, color='gray', linewidth=1, alpha=0.5, zorder=0)

    plt.scatter(stages_sorted, mean_dfas, color='black', marker='s', s=60, label='raw DFA', zorder=2)
    plt.scatter(stages_sorted, smoothed_mean_dfas, color='green', marker='s', s=30, label='smoothed DFA', zorder=2)
    plt.plot(stages_sorted, mean_dfas, color='black', linestyle='--', alpha=0.6, zorder=2)
    plt.plot(stages_sorted, smoothed_mean_dfas, color='green', linestyle='--', alpha=0.6, zorder=2)

    plt.xticks(stages_sorted, stage_labels)
    plt.ylabel('Z-normalized DFA exponent')
    plt.xlabel('Sleep Stage')
    plt.title('DFA per State')
    plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dfa_per_state.svg", format="svg")
    # plt.show()

    return smoothed_dfa


def aperiodic_violin_and_bar(aperiodic_fit_values, states, output_dir):
    """
    Plot per-state aperiodic exponents with bar + SEM and violin + jittered points + SEM.

    Parameters
    ----------
    aperiodic_fit_values : np.ndarray
        Array of per-epoch aperiodic exponents.
    states : np.ndarray
        Corresponding sleep stage integers (0=Wake ... 4=REM).
    output_dir : str
        Folder to save plots.
    """
    # Ensure same length
    min_len = min(len(aperiodic_fit_values), len(states))
    aperiodic_fit_values = aperiodic_fit_values[:min_len]
    states = states[:min_len]

    # DataFrame & summary
    df = pd.DataFrame({'state': states, 'aperiodic': aperiodic_fit_values})
    summary = df.groupby('state')['aperiodic'].agg(['mean', 'sem']).reset_index()
    colors_list = ['royalblue', 'teal', 'purple', 'forestgreen', 'firebrick']
    labels = ['W', 'N1', 'N2', 'N3', 'REM']

    # --- Bar plot with SEM ---
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary['state'], summary['mean'],
        yerr=summary['sem'],
        capsize=5,
        color=[colors_list[int(s)] for s in summary['state']],
        edgecolor='black', zorder=2, alpha=0.6
    )
    plt.xticks(range(5), labels)
    plt.xlabel('Sleep State')
    plt.ylabel('Aperiodic Exponent')
    plt.title('Aperiodic Fit per Sleep State (bar + SEM)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aperiodic_bar.svg", format='svg')
    plt.close()

    # --- Violin plot with jitter & SEM ---
    colors_dict = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    all_states = list(range(5))
    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    plt.figure(figsize=(8, 6))
    palette = [colors_dict[s] for s in all_states]

    # Violin
    ax = sns.violinplot(
        x='state', y='aperiodic', data=df_plot,
        order=all_states,
        palette=palette,
        cut=0,
        bw='scott',
        inner=None,
        alpha=0.5,
        zorder=2
    )

    # Jittered points
    for i, s in enumerate(all_states):
        vals = df_plot.loc[df_plot['state'] == s, 'aperiodic'].values
        if len(vals) > 0:
            x = np.random.normal(loc=i, scale=0.15, size=len(vals))
            ax.scatter(x, vals, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    # Medians + SEM
    for i, s in enumerate(all_states):
        vals = df_plot.loc[df_plot['state'] == s, 'aperiodic'].values
        if len(vals) == 0:
            continue
        median_val = np.nanmedian(vals)
        sem_val = np.nanstd(vals) / np.sqrt(len(vals))
        plt.plot(i, median_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=10)
        plt.errorbar(i, median_val, yerr=sem_val, color='black', capsize=4, elinewidth=1.5, markeredgewidth=1, zorder=9)

    # Cosmetics
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State')
    ax.set_ylabel('Aperiodic Exponent')
    ax.set_title('Aperiodic Fit per Sleep State (violin + SEM)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aperiodic_violin.svg", format='svg')
    plt.close()

def dfa_violin_and_bar(normalized_dfa, states, output_dir):
    min_len = min(len(normalized_dfa), len(states))
    normalized_dfa = normalized_dfa[:min_len]
    states = states[:min_len]

    # --- DataFrame & summary ---
    df = pd.DataFrame({'state': states, 'dfa': normalized_dfa})
    summary = df.groupby('state')['dfa'].agg(['mean', 'sem']).reset_index()
    colors_list = ['royalblue', 'teal', 'purple', 'forestgreen', 'firebrick']
    labels = ['W', 'N1', 'N2', 'N3', 'REM']

    # --- Bar plot with SEM ---
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary['state'], summary['mean'],
        yerr=summary['sem'],
        capsize=5,
        color=[colors_list[int(s)] for s in summary['state']],
        edgecolor='black', zorder=2, alpha=0.6
    )
    plt.xticks(range(5), labels)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Sleep State')
    plt.ylabel('Normalized mean DFA')
    plt.title('DFA per Sleep State (bar + SEM)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/DFA_bar.svg", format='svg')
    plt.close()

    # --- Violin plot with jitter & SEM ---
    colors_dict = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    all_states = list(range(5))
    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    plt.figure(figsize=(8, 6))
    palette = [colors_dict[s] for s in all_states]

    # Violin plot
    ax = sns.violinplot(
        x='state', y='dfa', data=df_plot,
        order=all_states,
        palette=palette,
        cut=0,
        bw='scott',
        inner=None,
        alpha=0.5,
        zorder=2
    )

    # Overlay jittered points
    for i, s in enumerate(all_states):
        vals = df_plot.loc[df_plot['state'] == s, 'dfa'].values
        if len(vals) > 0:
            x = np.random.normal(loc=i, scale=0.15, size=len(vals))
            ax.scatter(x, vals, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    # Overlay medians and SEM
    for i, s in enumerate(all_states):
        vals = df_plot.loc[df_plot['state'] == s, 'dfa'].values
        if len(vals) == 0:
            continue
        median_val = np.nanmedian(vals)
        sem_val = np.nanstd(vals) / np.sqrt(len(vals))
        plt.plot(i, median_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=10)
        plt.errorbar(i, median_val, yerr=sem_val, color='black', capsize=4, elinewidth=1.5, markeredgewidth=1, zorder=9)

    # Cosmetics
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State')
    ax.set_ylabel('Normalized mean DFA')
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('DFA per Sleep State (violin + SEM)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/DFA_violin.svg", format='svg')
    plt.close()

def mse_plot(lfp_PFC, output_dir, length, fs):
    Mobj = EH.MSobject('IncrEn', m=2, R=3, Norm=True)
    window_size = step_size = fs*length

    num_windows = (len(lfp_PFC) - window_size) // step_size + 1

    mse_values = []
    time_stamps_mse = []

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        segment = lfp_PFC[start:end]

        MSx, _ = EH.MSEn(segment, Mobj, Scales=2, Methodx='modified')

        mse_values.append(np.mean(MSx))
        time_stamps_mse.append((start + end) / 2 / fs)

    mse_values = np.array(mse_values)
    time_stamps_mse = np.array(time_stamps_mse)
    window_length = 11 if len(mse_values) >= 11 else len(mse_values) | 1  # ensure it's odd
    polyorder = 4

    smoothed_mse = savgol_filter(mse_values, window_length=window_length, polyorder=polyorder)
    normalized_mse = 2 * ((smoothed_mse - min(smoothed_mse)) / (max(smoothed_mse) - min(smoothed_mse))) - 1

    plt.figure(figsize=(18, 5))
    plt.plot(time_stamps_mse, normalized_mse, marker='.', linestyle='-', color=Red)
    plt.xlabel('Time (s)')
    plt.ylabel('MSE')
    plt.title('MSE Over Time (10-sec Windows) - RGS14')
    plt.grid()
    plt.savefig(f"{output_dir}/MSE_10s.svg", format='svg')
    # plt.show()
    return normalized_mse

def prepare_mse_violin_data(valid_states, fs, length, lfp_PFC):
    """
    Prepare data for violin plot of mse per sleep state.

    Returns:
        df_plot: DataFrame ready for seaborn plotting
        data_for_violin: List of arrays for each state (0..4)
        all_states: List of state indices [0..4]
        labels: Sleep stage labels ['W', 'N1', 'N2', 'N3', 'REM']
    """
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    all_states = list(range(5))  # 0..4
    Mobj = EH.MSobject('IncrEn', m=2, R=3, Norm=True)
    window_size = step_size = fs * length

    num_windows = (len(lfp_PFC) - window_size) // step_size + 1

    mse_values = []
    time_stamps_mse = []

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        segment = lfp_PFC[start:end]
        with suppress_stdout():
            MSx, _ = EH.MSEn(segment, Mobj, Scales=2, Methodx='modified')

        mse_values.append(np.mean(MSx))
        time_stamps_mse.append((start + end) / 2 / fs)

    mse_values = np.array(mse_values)
    time_stamps_mse = np.array(time_stamps_mse)
    window_length = 11 if len(mse_values) >= 11 else len(mse_values) | 1  # ensure it's odd
    polyorder = 4

    smoothed_mse = savgol_filter(mse_values, window_length=window_length, polyorder=polyorder)
    normalized_mse = 2 * ((smoothed_mse - min(smoothed_mse)) / (max(smoothed_mse) - min(smoothed_mse))) - 1
    min_length = min(len(normalized_mse), len(valid_states))
    valid_states = valid_states[:min_length]
    normalized_mse = normalized_mse[:min_length]
    df = pd.DataFrame({'state': valid_states, 'mse': normalized_mse})

    # Build data per state for violin plotting
    data_for_violin = [df.loc[df['state'] == s, 'mse'].values for s in all_states]

    counts = [len(d) for d in data_for_violin]
    print("Counts per state (0..4):", counts)
    print("Unique states present:", sorted(df['state'].unique()))

    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    return df_plot, data_for_violin, all_states, labels, normalized_mse

def mse_per_state(normalized_mse, states, output_dir):
    # --- Trim to same length ---
    min_len = min(len(normalized_mse), len(states))
    normalized_mse = normalized_mse[:min_len]
    states = states[:min_len]

    # --- Convert to arrays ---
    mse_values = np.array(normalized_mse)
    valid_states = np.array(states).astype(int)

    # --- Remove NaNs (safety) ---
    nan_mask = ~np.isnan(mse_values)
    mse_values = mse_values[nan_mask]
    valid_states = valid_states[nan_mask]

    # --- Compute z-scores ---
    raw_mse = zscore(mse_values, nan_policy='omit')

    # --- Smooth values ---
    window_length = min(101, len(raw_mse) // 2 * 2 + 1)  # must be odd
    smoothed_mse = savgol_filter(raw_mse, window_length, polyorder=5, mode='interp')
    smoothed_mse = zscore(smoothed_mse, nan_policy='omit')

    # --- Compute mean MSE per valid sleep state ---
    mean_mse_per_state = {}
    smoothed_mean_mse_per_state = {}

    for state in np.unique(valid_states):
        mask = valid_states == state
        mean_mse_per_state[state] = np.nanmean(raw_mse[mask])
        smoothed_mean_mse_per_state[state] = np.nanmean(smoothed_mse[mask])

    print("Mean MSE per state:", mean_mse_per_state)

    # --- Exclude any absent states ---
    stages_sorted = sorted(mean_mse_per_state.keys())

    # --- Compute per-state means for plotting ---
    mean_mses = [mean_mse_per_state[s] for s in stages_sorted]
    smoothed_mean_mses = [smoothed_mean_mse_per_state[s] for s in stages_sorted]

    # --- Map numeric state codes to labels ---
    state_labels_map = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    stage_labels = [state_labels_map.get(s, str(s)) for s in stages_sorted]

    # --- Plot MSE per available sleep state ---
    plt.figure(figsize=(7, 6))
    plt.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    plt.axhline(0, color='gray', linewidth=1, alpha=0.5, zorder=0)

    plt.scatter(stages_sorted, mean_mses, color='black', marker='s', s=60, label='raw MSE', zorder=2)
    plt.scatter(stages_sorted, smoothed_mean_mses, color='purple', marker='s', s=30, label='smoothed MSE', zorder=2)
    plt.plot(stages_sorted, mean_mses, color='black', linestyle='--', alpha=0.6, zorder=2)
    plt.plot(stages_sorted, smoothed_mean_mses, color='purple', linestyle='--', alpha=0.6, zorder=2)

    plt.xticks(stages_sorted, stage_labels)
    plt.ylabel('Z-normalized MSE')
    plt.xlabel('Sleep Stage')
    plt.title('MSE per Sleep State')
    plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mse_per_state.svg", format="svg")
    # plt.show()

    return smoothed_mse

def mse_violin_and_bar(normalized_mse, states, output_dir):
    min_len = min(len(normalized_mse), len(states))
    normalized_mse = normalized_mse[:min_len]
    states = states[:min_len]

    df = pd.DataFrame({'state': states, 'mse': normalized_mse})
    summary = df.groupby('state')['mse'].agg(['mean', 'sem']).reset_index()
    colors = ['royalblue', 'teal', 'purple', 'forestgreen', 'firebrick']
    labels = ['W', 'N1', 'N2', 'N3', 'REM']

    # --- Bar plot with SEM ---
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary['state'], summary['mean'],
        yerr=summary['sem'],
        capsize=5,
        color=[colors[int(s)] for s in summary['state']],
        edgecolor='black', zorder=2, alpha=0.6
    )
    plt.xticks([0, 1, 2, 3, 4], labels)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Sleep State')
    plt.ylabel('Normalized mean MSE')
    plt.title('MSE per Sleep State (bar + SEM)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/MSE_bar.svg", format='svg')
    plt.close()

    # --- Violin plot with SEM ---
    colors_dict = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    all_states = list(range(5))
    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    plt.figure(figsize=(8, 6))
    palette = [colors_dict[s] for s in all_states]

    ax = sns.violinplot(
        x='state', y='mse', data=df_plot,
        order=all_states,
        palette=palette,
        cut=0,
        bw='scott',
        inner=None,
        alpha = 0.5,
        zorder = 2
        )

    # --- Overlay jittered points (fixed) ---
    for i, s in enumerate(all_states):
        vals = df_plot.loc[df_plot['state'] == s, 'mse'].values
        if len(vals) > 0:
            x = np.random.normal(loc=i, scale=0.15, size=len(vals))
            ax.scatter(x, vals, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    # Overlay medians and SEM
    for i, s in enumerate(all_states):
        vals = df_plot.loc[df_plot['state'] == s, 'mse'].values
        if len(vals) == 0:
            continue
        median_val = np.nanmedian(vals)
        sem_val = np.nanstd(vals) / np.sqrt(len(vals))
        # median point
        plt.plot(i, median_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=10)
        # SEM vertical line
        plt.errorbar(i, median_val, yerr=sem_val, color='black', capsize=4, elinewidth=1.5, markeredgewidth=1, zorder=9)

    # Cosmetics
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State')
    ax.set_ylabel('Normalized mean MSE')
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('MSE per Sleep State (violin + SEM)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/MSE_violin.svg", format='svg')
    plt.close()


def Index_1(delta, gamma, EMG):
    index_1 = np.array([])
    for i in range(len(delta)):
        value = (EMG[i] * gamma[i]) / (delta[i])
        index_1 = np.append(index_1, [value])
    return index_1


def Index_2(delta, theta, sigma):
    index_2 = np.array([])
    for i in range(len(delta)):
        value = (sigma[i] * delta[i]) / (theta[i])
        index_2 = np.append(index_2, [value])
    return index_2


def Index_3(delta, theta, gamma):
    index_3 = np.array([])
    for i in range(len(delta)):
        value = (theta[i] * gamma[i]) / (delta[i])
        index_3 = np.append(index_3, [value])
    return index_3


def Index_4(delta, theta):
    index_4 = np.array([])
    for i in range(len(delta)):
        value = delta[i] / theta[i]
        index_4 = np.append(index_4, [value])
    return index_4




