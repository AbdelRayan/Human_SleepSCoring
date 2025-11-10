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
    plt.show()

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
    plt.show()

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
    plt.show()

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
    plt.show()

def smooth_and_norm(x):
    """Log-transform, normalize, and triple-smooth an array."""
    x_log = np.log(x)
    x_norm = wei_normalizing(x_log)
    kernel = np.ones(5) / 5
    for _ in range(3):  # triple smoothing
        x_norm = np.convolve(x_norm, kernel, mode='same')
    return x_norm

def index_barplot(index_n, index_r, index_w, mapped_scores, output_dir):

    # Preprocess indices
    smoothed = {
        'w': smooth_and_norm(index_w),
        'r': smooth_and_norm(index_r),
        'n': smooth_and_norm(index_n)
    }

    # Map score integers to labels
    score_labels = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    mapped_scores = [score_labels[s] for s in mapped_scores]

    # Prepare containers dynamically
    states = ['Wake', 'N1', 'N2', 'N3', 'REM']
    indices = {state: {'w': [], 'r': [], 'n': []} for state in states}

    # Fill index values per state
    for i, state in enumerate(mapped_scores):
        if state in indices:
            for k in smoothed.keys():
                indices[state][k].append(smoothed[k][i])

    # Compute means (safe if list empty)
    means = {
        state: {k: np.mean(v) if v else 0 for k, v in vals.items()}
        for state, vals in indices.items()
    }

    # Prepare data for plotting
    categories = states
    values_w = [means[s]['w'] for s in categories]
    values_r = [means[s]['r'] for s in categories]
    values_n = [means[s]['n'] for s in categories]

    # ---- Plot ----
    plt.figure(figsize=(12, 6))
    fontsize = 5
    x = np.arange(len(categories))
    bar_width = 0.25

    plt.bar(x - bar_width, values_w, width=bar_width, label='Index W', color='black')
    plt.bar(x, values_r, width=bar_width, label='Index R', color='red')
    plt.bar(x + bar_width, values_n, width=bar_width, label='Index N', color='blue')

    plt.xlabel('Sleep Stages')
    plt.ylabel('Average Index Values')
    plt.title('Wei indices')
    plt.xticks(x, categories)
    plt.legend(loc='upper center', bbox_to_anchor=(0.3, 1))

    # Numeric labels above bars
    for i in range(len(categories)):
        plt.text(x[i] - bar_width, values_w[i] + 0.02, f"{values_w[i]:.2f}",
                 ha='center', va='bottom', fontsize=fontsize)
        plt.text(x[i], values_r[i] + 0.02, f"{values_r[i]:.2f}",
                 ha='center', va='bottom', fontsize=fontsize)
        plt.text(x[i] + bar_width, values_n[i] + 0.02, f"{values_n[i]:.2f}",
                 ha='center', va='bottom', fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_new_indices_vs_wei_bar.svg', format='svg')
    plt.show()

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
    plt.show()

def aperiodic_fit(pfc_data, states, fs, raw_pfc, output_dir):
    # --- Parameters ---
    window_size = 10 * fs
    lfp_PFC = np.ravel(pfc_data)  # flatten in case it's 2D
    sleep_states = states.flatten()
    step_size = window_size

    num_windows = len(lfp_PFC) // window_size
    time_stamps = np.arange(num_windows) * 10  # adjust timing scale if needed
    window_length = 11  # for Savitzky-Golay
    polyorder = 4

    # --- Prepare windows ---
    window_data = [raw_pfc[i * window_size:(i + 1) * window_size] for i in range(num_windows)]

    # --- Diagnostic aperiodic fit function ---
    def safe_aperiodic_fit(window, idx, aperiodic_fit_fn):
        try:
            # Skip flat/empty windows
            if len(window) == 0 or np.std(window) < 1e-12:
                return np.nan, "flat_or_empty"
            if not np.all(np.isfinite(window)):
                return np.nan, "nonfinite_input"

            # Compute exponent
            exp = aperiodic_fit_fn(window)
            if not np.isfinite(exp):
                return np.nan, "nonfinite_exponent"

            return float(exp), "ok"

        except Exception as e:
            return np.nan, f"exception: {str(e)}"

    def aperiodic_fit(window_data):
        freqs, psd = welch(window_data, fs=fs, nperseg=1024)
        mask = (freqs <= 100)
        freqs, psd = freqs[mask], psd[mask]
        fm = SpectralModel(min_peak_height=0.05, aperiodic_mode='fixed', verbose=False)
        fm.fit(freqs, psd)
        aperiodic = fm.get_params('aperiodic')[1]
        return aperiodic

    # --- Compute aperiodic exponents in parallel ---
    results = Parallel(n_jobs=-1)(
        delayed(safe_aperiodic_fit)(w, i, aperiodic_fit) for i, w in enumerate(window_data)
    )

    # --- Extract exponents and statuses ---
    aperiodic_exponents, statuses = zip(*results)
    aperiodic_exponents = np.array(aperiodic_exponents)
    if len(aperiodic_exponents) > len(states):
        print(len(aperiodic_exponents) - len(states))
        for n in range(0, len(aperiodic_exponents) - len(states)):
            states = np.append(states, states[len(states) - 1])

    # --- Optional: report problem windows ---
    problem_windows = [(i, s) for i, s in enumerate(statuses) if s != "ok"]
    print(f"Problematic windows: {len(problem_windows)}")
    print("First 10 problematic windows:", problem_windows[:10])

    # --- Replace NaNs with median of valid exponents for plotting ---
    valid_mask = np.isfinite(aperiodic_exponents)
    median_val = np.median(aperiodic_exponents[valid_mask])
    aperiodic_exponents[~valid_mask] = median_val

    # --- Thresholding to remove extreme outliers ---
    threshold_min = np.percentile(aperiodic_exponents, 2)
    threshold_max = np.percentile(aperiodic_exponents, 98)
    valid_indices = (aperiodic_exponents >= threshold_min) & (aperiodic_exponents <= threshold_max)
    filtered_exponents = aperiodic_exponents[valid_indices]
    filtered_timestamps = time_stamps[valid_indices]
    valid_states = states[valid_indices]

    # --- Smooth and normalize ---
    window_length_sg = window_length if len(filtered_exponents) >= window_length else len(filtered_exponents) | 1
    smoothed_exponents = savgol_filter(filtered_exponents, window_length=window_length_sg, polyorder=polyorder)
    normalized_exponents = 2 * ((smoothed_exponents - smoothed_exponents.min()) / (
                smoothed_exponents.max() - smoothed_exponents.min())) - 1

    # --- Plot ---
    plt.figure(figsize=(18, 5))
    plt.plot(filtered_timestamps, normalized_exponents, marker='.', linestyle='-', color=DarkBlue)
    plt.xlabel('Time (s)')
    plt.ylabel('Aperiodic Exponent')
    plt.title('Normalized Aperiodic Fit Over Time (With Threshold)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Aperiodic_fit.svg", format='svg')
    plt.show()

    return normalized_exponents, smoothed_exponents, valid_states, aperiodic_exponents

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
    plt.show()
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

    # Compute mean slopes per state
    mean_slope_per_state = {}
    smoothed_mean_slope_per_state = {}

    for state in np.unique(valid_states):
        mask = valid_states == state
        mean_slope_per_state[state] = np.nanmean(raw_slopes[mask])
        smoothed_mean_slope_per_state[state] = np.nanmean(smoothed_slopes[mask])

    # Extract mean slopes in sorted order
    stages_sorted = sorted(mean_slope_per_state.keys())
    mean_slopes = [mean_slope_per_state[s] for s in stages_sorted]
    smoothed_mean_slopes = [smoothed_mean_slope_per_state[s] for s in stages_sorted]
    stages_sorted = sorted(mean_slope_per_state.keys())
    # Extract means per state
    mean_slopes = [np.mean(mean_raw_slope_by_state[state]) for state in stages_sorted if
                   state in mean_raw_slope_by_state]
    smoothed_mean_slopes = [np.mean(mean_smoothed_slope_by_state[state]) for state in stages_sorted if
                            state in mean_smoothed_slope_by_state]

    plt.figure(figsize=(7, 6))

    # draw grid
    plt.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    plt.axhline(0, color='gray', linewidth=1, alpha=0.5, zorder=0)

    plt.scatter(stages_sorted, mean_slopes, color='black', marker='s', s=60, label='raw slope', zorder=2)
    plt.scatter(stages_sorted, smoothed_mean_slopes, color='green', marker='s', s=30, label='smoothed slope', zorder=2)
    plt.plot(stages_sorted, mean_slopes, color='black', linestyle='--', alpha=0.6, zorder=2)
    plt.plot(stages_sorted, smoothed_mean_slopes, color='green', linestyle='--', alpha=0.6, zorder=2)

    plt.xticks(stages_sorted, ['W', 'N1', 'N2', 'N3', 'REM'])
    plt.ylabel('Z-normalized slope')
    plt.xlabel('Sleep Stage')
    plt.title('Slope per state')
    plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/slope_per_state.svg", format="svg")
    plt.show()
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
            if np.any(m_seg):  # only plot segments where mask is True
                ax2.plot(t_seg[m_seg], s_seg[m_seg], color=color, linewidth=3)

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
    plt.show()

def fooof_report(output_dir, raw_pfc, fs):
    freqs, psd = welch(raw_pfc, fs=fs, nperseg=1024)
    # Initialize and fit FOOOF model
    fm = FOOOF(peak_width_limits=[2, 8], aperiodic_mode='fixed')
    # Generate the report (this creates a matplotlib figure)
    fm.report(freqs, psd, [1, 50], plt_log=True)
    # Save the report plot to a file (e.g., PNG or PDF)
    plt.savefig(f"{output_dir}/fooof_report.png", dpi=300, bbox_inches='tight')
    plt.show()

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
    plt.show()

def aperiodic_fit_violin(valid_states, normalized_exponents, output_dir):
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    all_states = list(range(5))
    df = pd.DataFrame({'state': valid_states, 'aperiodic': normalized_exponents})
    summary = df.groupby('state')['aperiodic'].agg(['mean', 'sem']).reset_index()
    # Build data for ordered states 0..4
    data_for_violin = [df.loc[df['state'] == s, 'aperiodic'].values for s in all_states]
    counts = [len(d) for d in data_for_violin]
    print("Counts per state (0..4):", counts)
    print("Unique states present:", sorted(df['state'].unique()))

    # Prepare dataframe for seaborn
    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    # Violin plot
    plt.figure(figsize=(8, 6))
    palette = [colors[s] for s in all_states]

    ax = sns.violinplot(
        x='state', y='aperiodic', data=df_plot,
        order=all_states,
        palette=palette,
        cut=0,  # no tails beyond data range
        bw='scott',  # kernel bandwidth
        inner=None  # we’ll add medians manually
    )

    # Overlay jittered scatter points (optional)
    sns.stripplot(
        x='state', y='aperiodic', data=df_plot,
        order=all_states,
        color='k', size=1.5, jitter=0.15, alpha=0.3
    )

    # Compute medians and overlay them
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

    # Save + show
    plt.savefig(f"{output_dir}/Aperiodic_fit_violin.svg", format="svg")
    plt.show()

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
    plt.show()

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

    # --- Compute mean DFA per state ---
    mean_dfa_per_state = {}
    smoothed_mean_dfa_per_state = {}

    for state in np.unique(valid_states):
        mask = valid_states == state
        mean_dfa_per_state[state] = np.nanmean(raw_dfa[mask])
        smoothed_mean_dfa_per_state[state] = np.nanmean(smoothed_dfa[mask])

    print("Means per state:", mean_dfa_per_state)

    # --- Prepare for plotting ---
    stages_sorted = sorted(mean_dfa_per_state.keys())
    mean_dfas = [mean_dfa_per_state[s] for s in stages_sorted]
    smoothed_mean_dfas = [smoothed_mean_dfa_per_state[s] for s in stages_sorted]

    # --- Plot ---
    plt.figure(figsize=(7, 6))
    plt.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    plt.axhline(0, color='gray', linewidth=1, alpha=0.5, zorder=0)

    plt.scatter(stages_sorted, mean_dfas, color='black', marker='s', s=60, label='raw DFA', zorder=2)
    plt.scatter(stages_sorted, smoothed_mean_dfas, color='green', marker='s', s=30, label='smoothed DFA', zorder=2)
    plt.plot(stages_sorted, mean_dfas, color='black', linestyle='--', alpha=0.6, zorder=2)
    plt.plot(stages_sorted, smoothed_mean_dfas, color='green', linestyle='--', alpha=0.6, zorder=2)

    plt.xticks(stages_sorted, ['W', 'N1', 'N2', 'N3', 'REM'])
    plt.ylabel('Z-normalized DFA exponent')
    plt.xlabel('Sleep Stage')
    plt.title('DFA per State')
    plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dfa_per_state.svg", format="svg")
    plt.show()

def dfa_violin_and_bar(normalized_dfa, states, output_dir):
    min_len = min(len(normalized_dfa), len(states))
    normalized_dfa = normalized_dfa[:min_len]
    states= states[:min_len]
    colors = ['royalblue', 'teal', 'purple', 'forestgreen', 'firebrick']
    df = pd.DataFrame({'state': states, 'dfa': normalized_dfa})
    summary = df.groupby('state')['dfa'].agg(['mean', 'sem']).reset_index()
    print(summary)
    plt.figure(figsize=(7, 5))
    plt.bar(summary['state'], summary['mean'], yerr=summary['sem'],
            capsize=5, color=[colors[int(s)] for s in summary['state']], edgecolor='black', zorder=2, alpha=0.6)
    plt.xticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM'])
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Sleep State')
    plt.ylabel('Normalized mean DFA')
    plt.title('DFA per Sleep State')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/DFA_bar.svg", format='svg')
    plt.show()
    plt.figure(figsize=(7, 5))
    # Create a boxplot
    # Build data in exact order 0..4
    data_for_box = [df.loc[df['state'] == s, 'dfa'].values for s in range(5)]
    counts = [len(d) for d in data_for_box]

    print("Counts per state (0..4):", counts)
    print("Data types:", df['state'].dtype)

    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    all_states = list(range(5))

    # Build data in exact order 0..4
    data_for_box = [df.loc[df['state'] == s, 'dfa'].values for s in all_states]
    counts = [len(d) for d in data_for_box]
    print("Counts per state (0..4):", counts)
    print("Unique states present:", sorted(df['state'].unique()))

    # Make a copy of df for plotting convenience
    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    plt.figure(figsize=(8, 6))

    # seaborn expects the state column to be categorical strings or ints; we keep ints but pass order
    # Create palette list in the 0..4 order
    palette = [colors[s] for s in all_states]

    # Violinplot - drop NaNs automatically
    ax = sns.violinplot(
        x='state', y='dfa', data=df_plot,
        order=all_states,
        palette=palette,
        cut=0,  # don't show tails beyond data
        bw='scott',  # bandwidth estimator (tune if needed)
        inner=None  # we'll overlay medians ourselves
    )

    # overlay a stripplot (points) for verification (jittered)
    sns.stripplot(x='state', y='dfa', data=df_plot, order=all_states,
                  color='k', size=1.5, jitter=0.15, alpha=0.3)

    # compute medians and plot
    medians = [np.nanmedian(d) if len(d) > 0 else np.nan for d in data_for_box]
    for i, m in enumerate(medians):
        if not np.isnan(m):
            plt.plot(i, m, marker='o', color='white', markeredgecolor='black', markersize=6, zorder=10)

    # cosmetic
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State')
    ax.set_ylabel('Normalized mean DFA')
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('DFA per Sleep State (violin)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dfa_violin.svg", format="svg")
    plt.show()

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
    plt.show()
    return normalized_mse

def mse_per_state(normalized_mse, states, output_dir):
    min_len = min(len(normalized_mse), len(states))
    normalized_mse = normalized_mse[:min_len]
    states= states[:min_len]
    # --- Convert to arrays ---
    mse_values = np.array(normalized_mse)
    valid_states = np.array(states).astype(int)

    # --- Remove any NaNs (safety) ---
    nan_mask = ~np.isnan(mse_values)
    mse_values = mse_values[nan_mask]
    valid_states = valid_states[nan_mask]

    # --- Compute z-scores ---

    raw_mse = zscore(mse_values, nan_policy='omit')

    # --- Smooth values ---
    window_length = min(101, len(raw_mse) // 2 * 2 + 1)  # must be odd
    smoothed_mse = savgol_filter(raw_mse, window_length, polyorder=5, mode='interp')
    smoothed_mse = zscore(smoothed_mse, nan_policy='omit')

    # --- Compute mean MSE per sleep state ---
    mean_mse_per_state = {}
    smoothed_mean_mse_per_state = {}

    for state in np.unique(valid_states):
        mask = valid_states == state
        mean_mse_per_state[state] = np.nanmean(raw_mse[mask])
        smoothed_mean_mse_per_state[state] = np.nanmean(smoothed_mse[mask])

    print("Mean MSE per state:", mean_mse_per_state)

    # --- Prepare for plotting ---
    stages_sorted = sorted(mean_mse_per_state.keys())
    mean_mses = [mean_mse_per_state[s] for s in stages_sorted]
    smoothed_mean_mses = [smoothed_mean_mse_per_state[s] for s in stages_sorted]

    # --- Plot MSE per sleep state ---
    plt.figure(figsize=(7, 6))
    plt.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
    plt.axhline(0, color='gray', linewidth=1, alpha=0.5, zorder=0)

    plt.scatter(stages_sorted, mean_mses, color='black', marker='s', s=60, label='raw MSE', zorder=2)
    plt.scatter(stages_sorted, smoothed_mean_mses, color='purple', marker='s', s=30, label='smoothed MSE', zorder=2)
    plt.plot(stages_sorted, mean_mses, color='black', linestyle='--', alpha=0.6, zorder=2)
    plt.plot(stages_sorted, smoothed_mean_mses, color='purple', linestyle='--', alpha=0.6, zorder=2)

    plt.xticks(stages_sorted, ['W', 'N1', 'N2', 'N3', 'REM'])
    plt.ylabel('Z-normalized MSE')
    plt.xlabel('Sleep Stage')
    plt.title('MSE per Sleep State')
    plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mse_per_state.svg", format="svg")
    plt.show()

def mse_violin_and_bar(normalized_mse, states, output_dir):
    min_len = min(len(normalized_mse), len(states))
    normalized_mse = normalized_mse[:min_len]
    states= states[:min_len]
    df = pd.DataFrame({'state': states, 'mse': normalized_mse})
    summary = df.groupby('state')['mse'].agg(['mean', 'sem']).reset_index()
    colors = ['royalblue', 'teal', 'purple', 'forestgreen', 'firebrick']
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    print(summary)
    plt.figure(figsize=(7, 5))
    plt.bar(summary['state'], summary['mean'], yerr=summary['sem'],
            capsize=5, color=[colors[int(s)] for s in summary['state']], edgecolor='black', zorder=2, alpha=0.6)
    plt.xticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM'])
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Sleep State')
    plt.ylabel('Normalized mean MSE')
    plt.title('MSE per Sleep State')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/MSE_bar.svg", format='svg')
    plt.show()
    plt.figure(figsize=(7, 5))

    # --- Inputs ---
    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    all_states = list(range(5))

    # Build data for each state
    data_for_plot = [df.loc[df['state'] == s, 'mse'].values for s in all_states]
    medians = [np.nanmedian(d) if len(d) > 0 else np.nan for d in data_for_plot]
    print("Counts per state (0..4):", [len(d) for d in data_for_plot])
    print("Unique states present:", sorted(df['state'].unique()))

    # Prepare df copy for plotting
    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    # --- Seaborn violinplot ---
    plt.figure(figsize=(8, 6))
    palette = [colors[s] for s in all_states]

    ax = sns.violinplot(
        x='state', y='mse', data=df_plot,
        order=all_states,
        palette=palette,
        cut=0,  # no tails beyond data
        bw='scott',  # bandwidth estimator
        inner=None  # overlay medians manually
    )

    # Overlay jittered points
    sns.stripplot(x='state', y='mse', data=df_plot, order=all_states,
                  color='k', size=1.5, jitter=0.15, alpha=0.3)

    # Overlay medians
    for i, m in enumerate(medians):
        if not np.isnan(m):
            plt.plot(i, m, marker='o', color='white', markeredgecolor='black', markersize=6, zorder=10)

    # Cosmetics
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State')
    ax.set_ylabel('Normalized mean MSE')
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('MSE per Sleep State (violin)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/MSE_violin.svg", format='svg')
    plt.show()






