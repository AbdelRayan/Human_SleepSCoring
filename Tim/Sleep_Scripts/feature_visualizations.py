import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, decimate, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

state_labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

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

def raw_signals(states, raw_hpc, raw_pfc, pfc_tag, hpc_tag, output_dir):
    upsampled_states = np.repeat(states, 250 * 10)

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

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.savefig(f"{output_dir}/RAW_signals_Hypnogram.svg", format='svg', dpi=300)
    # Show the plots
    plt.show()

def indices_vs_hypnogram(epochs, hypno_epochs, index_w, index_n, index_r, mapped_scores, output_dir):
    """
    Plot smoothed Wei indices (W, N, R) alongside hypnogram scores.
    Requires a pre-defined smooth_and_norm() function.
    """

    # Smooth and normalize all indices using your shared helper
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
    ax2.plot(hypno_epochs[hypno_times], mapped_scores[hypno_times], label='Mapped scores', color='gray')

    # Legends
    ax1.legend()
    ax2.legend()

    # Titles and labels
    ax1.set_title('New indices')
    ax2.set_title('Mapped scores')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(state_labels)
    ax2.invert_yaxis()

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

    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_new_indices_vs_wei_bar.svg', format='svg')
    plt.show()

def index_pca(index_n, mapped_scores_old, index_r, index_w, output_dir):
    index_w_smoothed = smooth_and_norm(index_w)
    index_r_smoothed = smooth_and_norm(index_r)
    index_n_smoothed = smooth_and_norm(index_n)
    difference = len(index_n_smoothed) - len(mapped_scores_old)
    arrays = [
        index_w_smoothed[:-difference],
        index_r_smoothed[:-difference],
        index_n_smoothed[:-difference],
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



