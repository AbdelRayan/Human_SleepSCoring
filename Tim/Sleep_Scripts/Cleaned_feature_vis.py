"""
Several functions for plotting features and LFP data in individual subjects
By: Tim veldema
Date: 20/11/2025
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from scipy.signal import butter, filtfilt, decimate, find_peaks, welch, savgol_filter
from joblib import Parallel, delayed
from specparam import SpectralModel
from neurodsp.aperiodic import compute_irasa, fit_irasa, compute_fluctuations
from collections import defaultdict
from scipy.stats import zscore, sem
from fooof import FOOOF
import seaborn as sns
import EntropyHub as EH
from matplotlib.gridspec import GridSpec

Red = '#d13838'
Blue = '#127be3'
DarkBlue = '#09217a'
LightBlue = '#9cd4ff'
Yellow = '#eff250'
Orange = '#faac11'
Purple = '#a170fd'

def wei_normalizing(data):
    """
    Custom normalization based on top 10% and bottom 10% of data.
    Scales data to range [0.05, 1] after linear normalization.

    Parameters:
        data : array-like
            1D array of numerical values to normalize.

    Returns:
        normalized_data : np.ndarray
            Normalized and clipped data array.
    """
    data = np.array(data)
    bottom = data[data <= np.nanpercentile(data, 10)]
    top = data[data >= np.nanpercentile(data, 90)]

    bottom_avg = np.average(bottom) if len(bottom) > 0 else 0
    top_avg = np.average(top) if len(top) > 0 else 1

    denom = top_avg - bottom_avg if top_avg != bottom_avg else 1
    normalized_data = (data - bottom_avg) / denom
    normalized_data = np.clip(normalized_data, 0.05, 1)

    return normalized_data

def smooth_and_norm(x):
    """
    Apply log-transform, normalize, and triple smooth an array using moving average.

    Parameters:
    - x: np.array, input array to be smoothed

    Returns:
    - x_norm: np.array, smoothed and normalized array
    """
    # Avoid log(0) by clipping very small values
    x_log = np.log(np.clip(x, a_min=1e-12, a_max=None))
    x_norm = wei_normalizing(x_log)  # assumed pre-defined normalization function

    # Triple smoothing with moving average kernel
    kernel = np.ones(5) / 5
    for _ in range(3):
        x_norm = np.convolve(x_norm, kernel, mode='same')

    return x_norm

def normalised_powers(EMG_norm, noise_norm, delta_norm, theta_norm, sigma_norm, beta_norm, gamma_norm, alpha_norm,
                      hypno_epochs, sleep_scoring, output_dir):
    """
    Plot normalized EEG and EMG powers across epochs alongside sleep scoring.

    Parameters:
    - EMG_norm: np.array, normalized EMG signal
    - noise_norm: np.array, normalized noise power
    - delta_norm, theta_norm, sigma_norm, beta_norm, gamma_norm, alpha_norm: np.array, normalized EEG band powers
    - hypno_epochs: np.array, epoch indices corresponding to hypnogram scoring
    - sleep_scoring: np.array, mapped sleep states (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    - output_dir: str, folder path to save the resulting figure

    Saves:
    - SVG figure of all normalized powers and hypnogram.
    """

    # Create 9 vertically stacked subplots sharing the same x-axis
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, 1, sharex=True, figsize=(28, 20))

    epochs = np.arange(len(EMG_norm))  # x-axis for plots

    # Plot each normalized signal
    ax1.plot(epochs, EMG_norm, label='EMG', color='black')
    ax2.plot(epochs, noise_norm, label='Noise power', color='black')
    ax3.plot(epochs, delta_norm, label='Delta power', color='black')
    ax4.plot(epochs, theta_norm, label='Theta power', color='black')
    ax5.plot(epochs, sigma_norm, label='Sigma power', color='black')
    ax6.plot(epochs, beta_norm, label='Beta power', color='black')
    ax7.plot(epochs, gamma_norm, label='Gamma power', color='black')
    ax8.plot(epochs, alpha_norm, label='Alpha power', color='black')
    ax9.plot(hypno_epochs, sleep_scoring, label='Mapped scores', color='black')

    # Add legends to first 8 axes (last axis is self-explanatory)
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.legend(loc='upper right')

    # Set y-axis labels
    ax1.set_ylabel('Normalised amplitude')
    for ax in [ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.set_ylabel('Normalised power')
    ax9.set_ylabel('States')

    # Invert hypnogram y-axis (0 at top)
    ax9.invert_yaxis()
    ax9.set_xlabel('Epochs')

    # Add grid to all axes for better readability
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
        ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/Normalised_powers.svg", format='svg', dpi=300)
    # plt.show()  # Uncomment to display figure interactively


def normalised_powers_paper(noise_norm, delta_norm, theta_norm, sigma_norm, gamma_norm,
                            epoch_length, output_dir):
    """
    Paper-ready plot of selected normalized EEG powers: noise, delta, theta, sigma, gamma.
    X-axis is in minutes.

    Parameters:
    - noise_norm, delta_norm, theta_norm, sigma_norm, gamma_norm: np.array, normalized EEG band powers
    - epoch_length: float/int, duration of one epoch in seconds
    - output_dir: str, folder path to save the resulting figure

    Saves:
    - SVG and PDF figure of selected normalized powers.
    """

    # Create 5 vertically stacked subplots sharing the same x-axis
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})

    # Time vector in minutes
    epochs = np.arange(len(noise_norm))
    x = epochs * epoch_length / 60  # minutes

    # List of signals and labels
    signals = [noise_norm, delta_norm, theta_norm, sigma_norm, gamma_norm]
    labels = ['Noise power', 'Delta power', 'Theta power', 'Sigma power', 'Gamma power']

    for ax, sig, label in zip(axes, signals, labels):
        ax.plot(x, sig, color='black', lw=0.8)
        ax.set_ylabel(label)
        # Auto-scale y-limits with small margin
        margin = 0.05 * (sig.max() - sig.min())
        ax.set_ylim(sig.min() - margin, sig.max() + margin)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)

    axes[-1].set_xlabel('Time (min)')

    # Tight layout for publication
    plt.tight_layout()

    # Save as high-quality SVG and PDF for vector graphics
    plt.savefig(f"{output_dir}/Normalised_powers.svg", format='svg', dpi=300)
    plt.close()

def normalised_EMG(EMG_norm, output_dir):
    """
    Plot a normalized EMG signal across epochs.

    Parameters:
    - EMG_norm: np.array, normalized EMG signal
    - output_dir: str, folder path to save the resulting figure

    Saves:
    - SVG figure of normalized EMG.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(28, 5))
    epochs = np.arange(len(EMG_norm))

    ax1.plot(epochs, EMG_norm, label='', color='black')

    ax1.legend()
    ax1.set_title('Normalised EMG')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/Normalised_EMG.svg", format='svg', dpi=300)
    # plt.show()


def raw_signals(states, raw_hpc, raw_pfc, pfc_tag, hpc_tag, output_dir, fs, epoch_length):
    # Upsample states to match signal length
    upsampled_states = np.repeat(states, int(fs * epoch_length))
    target_len = len(raw_hpc)

    # Pad or trim to match raw signal length
    if len(upsampled_states) < target_len:
        pad_len = target_len - len(upsampled_states)
        upsampled_states = np.concatenate((upsampled_states, np.zeros(pad_len)))
    elif len(upsampled_states) > target_len:
        upsampled_states = upsampled_states[:target_len]

    # Time vector in minutes for clarity in long recordings
    x = np.arange(len(raw_hpc)) / fs / 60  # time in minutes

    # Paper-ready figure
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))  # smaller than original
    plt.rcParams.update({'font.size': 12})

    # Plot raw signals
    ax1.plot(x, raw_pfc, color='black', lw=0.8)
    ax2.plot(x, raw_hpc, color='black', lw=0.8)

    # Titles and labels
    ax1.set_title(f'Raw {pfc_tag} Signal', fontsize=14, fontweight='bold')
    ax2.set_title(f'Raw {hpc_tag} Signal', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Amplitude (V)')
    ax2.set_ylabel('Amplitude (V)')
    ax2.set_xlabel('Time (min)')

    # Auto-scale y-limits with small margin
    for ax, sig in zip([ax1, ax2], [raw_pfc, raw_hpc]):
        margin = 0.05 * (sig.max() - sig.min())
        ax.set_ylim(sig.min() - margin, sig.max() + margin)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)

    # Tight layout for publication
    plt.tight_layout()

    # Save as high-quality SVG for vector graphics
    plt.savefig(f"{output_dir}/RAW_signals.svg", format='svg', dpi=300)
    plt.close()


def combined_raw_and_power_plot(states, raw_hpc, raw_pfc, pfc_tag, hpc_tag,
                                noise_norm, delta_norm, theta_norm, sigma_norm, gamma_norm,
                                fs, epoch_length, output_dir, panel_label=None):
    """
    Creates a paper-ready figure containing:
      - Raw PFC
      - Raw HPC
      - 5 normalized power bands

    Includes a journal-style panel label (A/B/etc.) when provided.
    """

    # ---- Upsample states ----
    upsampled_states = np.repeat(states, int(fs * epoch_length))
    target_len = len(raw_hpc)
    if len(upsampled_states) < target_len:
        upsampled_states = np.concatenate((upsampled_states, np.zeros(target_len - len(upsampled_states))))
    else:
        upsampled_states = upsampled_states[:target_len]

    # Time for raw signals (minutes)
    t = np.arange(len(raw_hpc)) / fs / 60

    # Time for epoch-wise signals (minutes)
    epochs = np.arange(len(noise_norm))
    t_pow = epochs * epoch_length / 60

    # ---- Create figure ----
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(12, 12))
    plt.rcParams.update({'font.size': 12})

    # ---- FIX #2: Add top + left margins to avoid panel label overlap ----
    plt.subplots_adjust(left=0.14, top=0.92)

    # ---- PANEL LABEL (journal style: outside axes, bold, top-left) ----
    if panel_label is not None:
        fig.text(0.02, 0.965, panel_label,
                 fontsize=20, fontweight='bold', va='top', ha='left')

    # ---- RAW SIGNALS ----
    axes[0].plot(t, raw_pfc, color='black', lw=0.8)
    axes[0].set_title(f'Raw {pfc_tag} Signal', fontsize=12)
    axes[0].set_ylabel('Amplitude (V)')
    axes[0].set_ylim(-0.001, 0.001)

    axes[1].plot(t, raw_hpc, color='black', lw=0.8)
    axes[1].set_title(f'Raw {hpc_tag} Signal', fontsize=12)
    axes[1].set_ylabel('Amplitude (V)')
    axes[1].set_ylim(-0.001, 0.001)

    for ax in axes[:2]:
        ax.grid(True, linestyle='--', alpha=0.3)

    # ---- NORMALIZED POWERS ----
    powers = [noise_norm, delta_norm, theta_norm, sigma_norm, gamma_norm]
    labels = ['Noise', 'Delta', 'Theta', 'Sigma', 'Gamma']

    for ax, sig, label in zip(axes[2:], powers, labels):
        ax.plot(t_pow, sig, color='black', lw=0.8)
        ax.set_ylabel(label)

        # Auto-scale with margin
        margin = 0.05 * (sig.max() - sig.min())
        ax.set_ylim(sig.min() - margin, sig.max() + margin)

        ax.grid(True, linestyle='--', alpha=0.3)

    axes[-1].set_xlabel('Time (min)')

    # ---- Export ----
    outA = f"{output_dir}/figure_{panel_label}.svg"
    outB = f"{output_dir}/figure_{panel_label}.pdf"
    fig.savefig(outA, dpi=300)
    fig.savefig(outB, dpi=300)

    plt.close()

def indices_vs_hypnogram(epochs, hypno_epochs, index_w, index_n, index_r, mapped_scores, output_dir, show=False):
    """
    Plot smoothed Wei indices (W, N, R) alongside hypnogram scores.

    Parameters:
    - epochs: np.array, epoch numbers
    - hypno_epochs: np.array, epoch numbers for hypnogram
    - index_w, index_n, index_r: np.array, raw Wei indices for Wake, NREM, REM
    - mapped_scores: np.array, hypnogram scores
    - output_dir: str, folder path to save figure

    Saves:
    - SVG figure showing indices vs hypnogram.
    """
    state_labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # Smooth and normalize indices
    index_w_smoothed = smooth_and_norm(index_w)
    index_n_smoothed = smooth_and_norm(index_n)
    index_r_smoothed = smooth_and_norm(index_r)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(40, 16))

    # Plot smoothed indices
    ax1.plot(epochs, index_w_smoothed, label='Index W', color='black')
    ax1.plot(epochs, index_n_smoothed, label='Index N', color='blue')
    ax1.plot(epochs, index_r_smoothed, label='Index R', color='red')

    # Plot hypnogram
    ax2.plot(hypno_epochs, mapped_scores, label='Mapped scores', color='gray')

    ax1.legend()
    ax2.legend()

    ax1.set_title('New indices')
    ax2.set_title('Mapped scores')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(state_labels)
    ax2.invert_yaxis()

    # Create evenly spaced x-ticks for vertical grid lines
    xticks = np.linspace(epochs[0], epochs[-1], num=40)
    for ax in (ax1, ax2):
        ax.set_xticks(xticks)
        ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
        ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.6)
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if show:
        plt.savefig(f'{output_dir}/all_new_indices_vs_wei.svg', format='svg')
        plt.show()
    else:
        plt.savefig(f'{output_dir}/all_new_indices_vs_wei.svg', format='svg')
        plt.close()

def index_barplot(index_n, index_r, index_w, mapped_scores, output_dir):
    """
    Smooth and normalize Wei indices per sleep stage, then plot a grouped bar chart.

    Parameters
    ----------
    index_n : array-like
        NREM-related index values per epoch.
    index_r : array-like
        REM-related index values per epoch.
    index_w : array-like
        Wake-related index values per epoch.
    mapped_scores : array-like
        Numeric sleep stage labels per epoch (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM).
    output_dir : str
        Directory to save the output plot.

    Returns
    -------
    None
    """
    # --- Smooth and normalize the indices ---
    index_w_smoothed = smooth_and_norm(index_w)
    index_n_smoothed = smooth_and_norm(index_n)
    index_r_smoothed = smooth_and_norm(index_r)

    # --- Map numeric scores to sleep stage names ---
    stage_labels = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    mapped_labels = [stage_labels[s] for s in mapped_scores]

    # --- Organize indices by sleep stage in a dictionary ---
    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    indices_by_stage = {stage: [] for stage in stages}

    for w_val, r_val, n_val, stage in zip(index_w_smoothed, index_r_smoothed, index_n_smoothed, mapped_labels):
        indices_by_stage[stage].append([w_val, r_val, n_val])

    # --- Compute average for each index type per stage ---
    avg_indices = {stage: np.mean(indices_by_stage[stage], axis=0) for stage in stages}

    # --- Prepare data for plotting ---
    values_wake = [avg_indices[stage][0] for stage in stages]
    values_REM = [avg_indices[stage][1] for stage in stages]
    values_NREM = [avg_indices[stage][2] for stage in stages]

    x = np.arange(len(stages))
    bar_width = 0.2

    # --- Plot grouped bar chart ---
    plt.figure(figsize=(12, 6))
    plt.bar(x - bar_width, values_wake, width=bar_width, label='Index W', color='black')
    plt.bar(x, values_REM, width=bar_width, label='Index R', color='red')
    plt.bar(x + bar_width, values_NREM, width=bar_width, label='Index N', color='blue')

    # --- Labels and aesthetics ---
    plt.xlabel('Sleep Stages')
    plt.ylabel('Average Index Values')
    plt.title('Wei indexes')
    plt.xticks(x, stages)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/index_barplots.svg", format='svg', dpi=300)

def index_pca(indices_list, mapped_scores_old, output_dir):
    """
    Perform PCA on multiple indices and plot the first two principal components
    colored by sleep stage.

    Args:
        indices_list (list of array-like): List of index arrays per epoch.
                                           For 7 indices: [index_w, index_r, index_n, index_1, index_2, index_3, index_4]
        mapped_scores_old (array-like): Sleep stage labels per epoch.
        output_dir (str): Directory path to save the PCA plot SVG.
    """
    # Smooth and normalize all indices
    smoothed_indices = [smooth_and_norm(idx) for idx in indices_list]

    # Align lengths with sleep scores if indices are longer
    min_length = len(mapped_scores_old)
    smoothed_indices = [idx[:min_length] for idx in smoothed_indices]

    # Combine indices and sleep labels into a dataframe
    arrays = smoothed_indices + [mapped_scores_old]
    array = np.column_stack(arrays)
    df = pd.DataFrame(array)

    # Features (all indices)
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X, nan=0.0))

    # Apply PCA
    n_components = min(3, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f'PCA data points: {X_pca.shape}')

    # Plot PCA scatter (first 2 PCs)
    stage_map = {0.0: "wake", 1.0: "n1", 2.0: "n2", 3.0: "n3", 4.0: "rem"}
    colors = ['#0072B2', '#E69F00', '#D55E00', '#CC79A7', '#F0E442']

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(y)):
        stage_name = stage_map.get(label, str(label))
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
    plt.savefig(f"{output_dir}/Index_PCA_subject.svg", format='svg')
    plt.close()

def enlarge_ticks(ax, factor=1.25):
    for axis in [ax.xaxis, ax.yaxis]:
        for tick in axis.get_major_ticks():
            # label1 = main label; label2 = opposite-side label
            for lbl in [tick.label1, tick.label2]:
                if lbl:  # some may not exist
                    lbl.set_fontsize(lbl.get_fontsize() * factor)


def smooth_epochs(x, window=20):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')

def eog_vs_hypnogram_paper(EOG1, EOG2, epoch_length, fs, epochs, mapped_scores, output_dir, letters=('A', 'B')):

    N_EOG = N_feature(EOG1, EOG2, epoch_length, fs)
    R_EOG = rem_feature(EOG1, EOG2, epoch_length, fs)

    stage_labels_ordered = ['Wake', 'REM', 'N1', 'N2', 'N3']

    stage_to_num = {label:i for i,label in enumerate(stage_labels_ordered)}

    smoothed_N_EOG = smooth_epochs(smooth_and_norm(N_EOG), window=20)
    smoothed_R_EOG = smooth_epochs(smooth_and_norm(R_EOG), window=20)

    min_len = min(len(mapped_scores), len(smoothed_N_EOG))
    mapped_scores = mapped_scores[:min_len]
    epochs = epochs[:min_len]

    hypno_numeric = np.array([
        stage_to_num[{0:'Wake',1:'N1',2:'N2',3:'N3',4:'REM'}[s]]
        for s in mapped_scores
    ])

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35)

    ax0 = fig.add_subplot(gs[0])

    ax0.plot(epochs, smoothed_N_EOG[:min_len] , label="EOG 0.3-0.45Hz",
             color='blue', linewidth=2, alpha=0.5)

    ax0.plot(epochs, smoothed_R_EOG[:min_len], label='EOG 0.3-35Hz',
             color='red', linewidth=2, alpha=0.5)

    ax0.set_ylabel('EOG feature value', fontsize=14)
    ax0.set_xlabel('Epoch', fontsize=14)
    ax0.set_title('EOG features vs Hypnogram', fontsize=16)
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend(fontsize=12)

    enlarge_ticks(ax0)

    # Hypnogram overlay
    ax0b = ax0.twinx()
    ax0b.step(epochs, hypno_numeric, where='mid',
              color='black', linewidth=2.8, label='Hypnogram')
    ax0b.set_yticks(range(len(stage_labels_ordered)))
    ax0b.set_yticklabels(stage_labels_ordered)
    ax0b.set_ylabel('Sleep Stage', fontsize=14)
    ax0b.invert_yaxis()
    ax0b.legend(loc='upper right', fontsize=12)

    enlarge_ticks(ax0b)

    ax0.text(0.01, 0.95, letters[0], transform=ax0.transAxes,
             fontsize=20, fontweight='bold', va='top')

    # ===========================================================
    #  Plot 2 — Bar Plot (Average EOG Features per Stage)
    # ===========================================================
    ax1 = fig.add_subplot(gs[1])

    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # Feature arrays
    feature_arrays = {
        "EOG_N": smoothed_N_EOG[:min_len],
        "EOG_R": smoothed_R_EOG[:min_len]
    }

    feature_names = ["N-feature (0.3–0.45 Hz)", "R-feature (0.3–35 Hz)"]
    colors = ["blue", "red"]

    # Organize values by stage
    values_by_stage = {stage: {name: [] for name in feature_names} for stage in stages}

    for i, stage_idx in enumerate(hypno_numeric):
        stage = stage_labels_ordered[stage_idx]
        values_by_stage[stage][feature_names[0]].append(feature_arrays["EOG_N"][i])
        values_by_stage[stage][feature_names[1]].append(feature_arrays["EOG_R"][i])

    # Compute mean + SEM
    means = {name: [] for name in feature_names}

    for stage in stages:
        for name, arrlist in values_by_stage[stage].items():
            arr = np.array(arrlist)
            means[name].append(np.nanmean(arr))

    # ---- Plot ----
    x = np.arange(len(stages))
    bar_width = 0.32

    for i, name in enumerate(feature_names):
        ax1.bar(
            x + (i - 0.5) * bar_width,
            means[name],
            width=bar_width,
            color=colors[i],
            edgecolor='black',
            label=name
        )

    # ---- Style ----
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=12)
    ax1.set_ylabel('Average EOG Feature Value', fontsize=14)
    ax1.set_xlabel('Sleep Stage', fontsize=14)
    ax1.set_title('Average EOG Feature Values per Sleep Stage', fontsize=16)

    # Paper-like formatting
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.3)
    ax1.spines['bottom'].set_linewidth(1.3)

    ax1.legend(fontsize=12, frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)

    enlarge_ticks(ax1)

    # Subplot letter
    ax1.text(0.01, 0.95, letters[1], transform=ax1.transAxes,
             fontsize=20, fontweight='bold', va='top')
    # ===========================================================
    # Save figure
    # ===========================================================
    plt.tight_layout()

    plt.savefig(f"{output_dir}/eog_vs_hypnogram.svg", format='svg')
    plt.close()


def combined_indices_figure(epochs, hypno_epochs, mapped_scores,
                            indices_list, index_names,
                            output_dir, letters=('A', 'B'), show=False):

    # Define sleep stage order
    stage_labels_ordered = ['Wake', 'REM', 'N1', 'N2', 'N3']
    stage_to_num = {label:i for i,label in enumerate(stage_labels_ordered)}
    colors = ['black', 'red', 'blue', '#CC79A7', '#F0E442', '#56B4E9', '#009E73']

    # --- Smooth & trim indices ---
    smoothed_indices = [smooth_epochs(smooth_and_norm(idx)) for idx in indices_list]

    min_len = min(len(mapped_scores), *(len(idx) for idx in smoothed_indices))
    smoothed_indices = [idx[:min_len] for idx in smoothed_indices]
    mapped_scores = mapped_scores[:min_len]
    epochs = epochs[:min_len]

    # --- Map hypnogram to desired order ---
    hypno_numeric = np.array([
        stage_to_num[{0:'Wake',1:'N1',2:'N2',3:'N3',4:'REM'}[s]]
        for s in mapped_scores
    ])

    # --- Create figure: now 2 rows, not 3 ---
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35)

    # ===========================================================
    #  Plot 1 — Indices vs Hypnogram
    # ===========================================================
    ax0 = fig.add_subplot(gs[0])

    for i, idx in enumerate(smoothed_indices[:3]):
        ax0.plot(epochs, idx, label=f'{index_names[i]}',
                 color=colors[i], linewidth=2, alpha=0.5)

    ax0.set_ylabel('Index Value', fontsize=14)
    ax0.set_xlabel('Epoch', fontsize=14)
    ax0.set_title('Indices vs Hypnogram', fontsize=16)
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend(fontsize=12)

    enlarge_ticks(ax0)

    # Subplot letter
    ax0.text(0.01, 0.95, letters[0], transform=ax0.transAxes,
             fontsize=20, fontweight='bold', va='top')

    # Hypnogram overlay
    ax0b = ax0.twinx()
    ax0b.step(hypno_epochs[:min_len], hypno_numeric, where='mid',
              color='black', linewidth=2.8, label='Hypnogram')
    ax0b.set_yticks(range(len(stage_labels_ordered)))
    ax0b.set_yticklabels(stage_labels_ordered)
    ax0b.set_ylabel('Sleep Stage', fontsize=14)
    ax0b.invert_yaxis()
    ax0b.legend(loc='upper right', fontsize=12)

    enlarge_ticks(ax0b)

    # ===========================================================
    #  Plot 2 — Bar Plot (Average Indices per Stage)
    # ===========================================================
    ax1 = fig.add_subplot(gs[1])

    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    indices_by_stage = {stage: [] for stage in stages}

    for values, stage in zip(zip(*smoothed_indices[:3]),
                             [stage_labels_ordered[s] for s in hypno_numeric]):
        indices_by_stage[stage].append(values)

    avg_indices = {
        stage: np.mean(indices_by_stage[stage], axis=0)
        for stage in stages
    }

    x = np.arange(len(stages))
    bar_width = 0.25

    for i in range(3):
        ax1.bar(x + i*bar_width - bar_width,
                [avg_indices[s][i] for s in stages],
                width=bar_width,
                color=colors[i],
                edgecolor='black',
                label=index_names[i])

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=12)
    ax1.set_ylabel('Average Index Value', fontsize=14)
    ax1.set_xlabel('Sleep Stage', fontsize=14)
    ax1.set_title('Average Indices per Sleep Stage', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.3)
    ax1.spines['bottom'].set_linewidth(1.3)

    enlarge_ticks(ax1)

    ax1.text(0.01, 0.95, letters[1], transform=ax1.transAxes,
             fontsize=20, fontweight='bold', va='top')
    ax0.set_ylim(0, 1)
    ax1.set_ylim(0, 1)
    # ===========================================================
    # Save figure
    # ===========================================================
    plt.tight_layout()
    save_path = f"{output_dir}/combined_indices_figure.svg"
    plt.savefig(save_path, format='svg', dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved combined figure to: {save_path}")

def aperiodic_fit(pfc_data, states, fs, raw_pfc, output_dir, epoch_length):
    """
    Compute and visualize the aperiodic exponent of PFC LFP signals in epochs.

    This function:
        - Splits the signal into non-overlapping epochs.
        - Computes aperiodic exponents per epoch using a robust "safe" wrapper.
        - Repairs invalid values via interpolation, median filling, and clipping.
        - Smooths and normalizes the resulting exponent series.
        - Plots the normalized aperiodic exponents over time.

    Parameters
    ----------
    pfc_data : array-like
        Preprocessed PFC signal, can be multi-dimensional but flattened internally.
    states : array-like
        Sleep stage labels corresponding to epochs (numeric: 0=W, 1=N1, 2=N2, 3=N3, 4=REM).
    fs : float
        Sampling frequency of the LFP signal (Hz).
    raw_pfc : array-like
        Original raw PFC signal (used for per-window fits).
    output_dir : str
        Directory path to save the aperiodic exponent plot.
    epoch_length : float
        Duration of each epoch in seconds.

    Returns
    -------
    normalized_exponents : np.ndarray
        Smoothed and normalized aperiodic exponents in the range [-1, 1].
    smoothed_exponents : np.ndarray
        Smoothed exponents before normalization.
    states : np.ndarray
        Sleep stage labels aligned to number of windows.
    repaired_exponents : np.ndarray
        Raw exponent values after repair (NaN handling, interpolation, clipping).

    Notes
    -----
    - Invalid or flat epochs are interpolated or replaced with median values.
    - Exponents are smoothed using Savitzky-Golay filter.
    - Normalization rescales values to [-1, 1] for consistent visualization.
    """


    DarkBlue = 'darkblue'  # color for plotting

    # --- Parameters ---
    window_size = int(epoch_length * fs)
    lfp_PFC = np.ravel(pfc_data)
    num_windows = len(lfp_PFC) // window_size
    time_stamps = np.arange(num_windows) * epoch_length  # approximate center of each epoch
    window_length = 11  # Savitzky-Golay filter window length
    polyorder = 4       # SG polynomial order

    # --- Segment raw PFC signal into epochs ---
    window_data = [
        raw_pfc[i * window_size:(i + 1) * window_size]
        for i in range(num_windows)
    ]

    # --- Safe wrapper for per-window aperiodic fitting ---
    def safe_aperiodic_fit(window, idx, aperiodic_fit_fn):
        """
        Compute aperiodic exponent safely with error handling.

        Returns NaN for invalid or flat signals, along with status messages.
        """
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

    # --- Function to compute aperiodic exponent for a window ---
    def aperiodic_fit(window):
        """
        Compute aperiodic exponent using PSD and spectral model fitting.
        """
        freqs, psd = welch(window, fs=fs, nperseg=1024)
        mask = freqs <= 100
        freqs, psd = freqs[mask], psd[mask]

        fm = SpectralModel(min_peak_height=0.05, aperiodic_mode='fixed', verbose=False)
        fm.fit(freqs, psd)
        # Second value of 'aperiodic' tuple is exponent
        return fm.get_params('aperiodic')[1]

    # --- Compute aperiodic exponents in parallel ---
    results = Parallel(n_jobs=-1)(
        delayed(safe_aperiodic_fit)(w, i, aperiodic_fit) for i, w in enumerate(window_data)
    )

    # Extract exponent values and diagnostic statuses
    aperiodic_exponents, statuses = zip(*results)
    aperiodic_exponents = np.array(aperiodic_exponents)

    # --- Align states and timestamps with exponent windows ---
    if len(aperiodic_exponents) > len(states):
        diff = len(aperiodic_exponents) - len(states)
        print(f"Padded {diff} state values to match exponent windows.")
        states = np.append(states, [states[-1]] * diff)

    states = states[:len(aperiodic_exponents)]
    time_stamps = time_stamps[:len(aperiodic_exponents)]

    # --- Report problematic windows ---
    problem_windows = [(i, s) for i, s in enumerate(statuses) if s != "ok"]
    print(f"Problematic windows: {len(problem_windows)}")
    print("First 10 problematic windows:", problem_windows[:10])

    # --- Repair invalid exponent values ---
    exp_series = pd.Series(aperiodic_exponents)
    exp_series = exp_series.interpolate(method='linear', limit_direction='both')  # fill NaNs
    exp_series = exp_series.fillna(exp_series.median())  # fallback median
    # Clip extreme values at 2nd and 98th percentiles
    threshold_min = np.percentile(exp_series, 2)
    threshold_max = np.percentile(exp_series, 98)
    exp_series = exp_series.clip(lower=threshold_min, upper=threshold_max)
    repaired_exponents = exp_series.to_numpy()

    # --- Smooth using Savitzky-Golay filter ---
    window_length_sg = window_length if len(repaired_exponents) >= window_length else (len(repaired_exponents) | 1)
    smoothed_exponents = savgol_filter(repaired_exponents, window_length=window_length_sg, polyorder=polyorder)

    # --- Normalize to [-1, 1] ---
    normalized_exponents = 2 * ((smoothed_exponents - smoothed_exponents.min()) /
                                (smoothed_exponents.max() - smoothed_exponents.min())) - 1

    # --- Plot normalized exponents over time ---
    plt.figure(figsize=(18, 5))
    plt.plot(time_stamps, normalized_exponents, marker='.', linestyle='-', color=DarkBlue)
    plt.xlabel('Time (s)')
    plt.ylabel('Aperiodic Exponent')
    plt.title('Normalized Aperiodic Fit Over Time (Repaired, No Dropping)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Aperiodic_fit.svg", format='svg')
    plt.close()

    return normalized_exponents, smoothed_exponents, states, repaired_exponents

def raw_to_epochs(data, sf, epoch):
    """
    Convert raw continuous data into non-overlapping epochs.

    Parameters:
        data : np.array
            1D array of raw signal samples.
        sf : float
            Sampling frequency of the data in Hz.
        epoch : float
            Desired epoch length in seconds.

    Returns:
        epochs : np.ndarray
            2D array of shape (n_epochs, samples_per_epoch), where each row
            is one epoch.
    """
    samples_per_epoch = int(epoch * sf)  # number of samples in one epoch
    n_epochs = len(data) // samples_per_epoch  # number of full epochs
    cropped_data = data[:n_epochs * samples_per_epoch]  # remove extra samples
    epochs = cropped_data.reshape(n_epochs, samples_per_epoch)  # reshape into epochs
    print("New data shape:", epochs.shape)
    return epochs

def calc_fractal_component(sleep_states, epoched_data, fs, f_range):
    """
    Compute the fractal (aperiodic) component of EEG for each epoch and sleep state
    using IRASA (irregular resampling method).

    Parameters:
        sleep_states : list or array
            Sleep state label for each epoch.
        epoched_data : array-like
            2D array of epochs (n_epochs x n_samples).
        fs : float
            Sampling frequency.
        f_range : tuple or list
            Frequency range for IRASA computation.

    Returns:
        irasa_mean_by_state : dict
            Keys are states, values are tuples of (freqs, mean_aperiodic_psd).
    """
    irasa_by_state = defaultdict(list)

    # Compute IRASA for each epoch
    for eeg_epoch, state in zip(epoched_data, sleep_states):
        freqs, psd_aperiodic, _ = compute_irasa(eeg_epoch, fs, f_range=f_range)
        irasa_by_state[state].append([freqs, psd_aperiodic])

    # Compute mean PSD per state
    irasa_mean_by_state = {}
    for state, irasa_list in irasa_by_state.items():
        aperiodics = np.array([ap for _, ap in irasa_list])
        freqs = irasa_list[0][0]
        mean_aperiodic = np.mean(aperiodics, axis=0)
        irasa_mean_by_state[state] = (freqs, mean_aperiodic)

    # Remove index 5 if it corresponds to movement or artifacts
    if 5 in irasa_mean_by_state:
        del irasa_mean_by_state[5]

    return irasa_mean_by_state

def calc_slopes(epoched_data, fs, f_range, states):
    """
    Calculate slopes of the aperiodic component (from IRASA) for each epoch,
    then smooth and z-score them. Computes mean slope per state.

    Parameters:
        epoched_data : array-like
            2D array of epochs (n_epochs x n_samples).
        fs : float
            Sampling frequency.
        f_range : tuple or list
            Frequency range to compute IRASA.
        states : array-like
            Sleep state label for each epoch.

    Returns:
        raw_slopes : np.ndarray
            Z-scored raw slopes for each epoch.
        smoothed_slopes : np.ndarray
            Savitzky-Golay smoothed and z-scored slopes.
        mean_slope_per_state : dict
            Average raw slope per state.
        smoothed_mean_slope_per_state : dict
            Average smoothed slope per state.
    """
    epoch_slopes = []
    valid_states = []

    for i, epoch in enumerate(epoched_data):
        freqs, psd_aperiodic, _ = compute_irasa(epoch, fs, f_range=f_range)
        # Remove invalid or non-positive PSD values
        valid = np.isfinite(psd_aperiodic) & (psd_aperiodic > 0)
        freqs = freqs[valid]
        psd_aperiodic = psd_aperiodic[valid]

        if len(freqs) < 5:  # skip epochs with insufficient data
            continue

        try:
            intercept, slope = fit_irasa(freqs, psd_aperiodic)
            epoch_slopes.append(slope)
            valid_states.append(states[i])
        except Exception:
            continue

    if len(epoch_slopes) == 0:
        raise ValueError("No valid slopes were computed — check your data or f_range.")

    # Convert to arrays
    epoch_slopes = np.array(epoch_slopes)
    valid_states = np.array(valid_states)

    # Z-score the raw slopes
    raw_slopes = zscore(epoch_slopes)
    min_len = min(len(raw_slopes), len(valid_states))
    raw_slopes = raw_slopes[:min_len]
    valid_states = valid_states[:min_len]

    # Smooth slopes with Savitzky-Golay filter
    window_length = min(101, len(raw_slopes) // 2 * 2 + 1)  # must be odd
    smoothed_slopes = savgol_filter(raw_slopes, window_length, polyorder=5, mode='interp')
    smoothed_slopes = zscore(smoothed_slopes)

    # Compute mean slope per state
    mean_slope_per_state = {}
    smoothed_mean_slope_per_state = {}
    for state in np.unique(valid_states):
        mask = valid_states == state
        mean_slope_per_state[state] = np.nanmean(raw_slopes[mask])
        smoothed_mean_slope_per_state[state] = np.nanmean(smoothed_slopes[mask])

    return raw_slopes, smoothed_slopes, mean_slope_per_state, smoothed_mean_slope_per_state

def fractal_power_component(states, subject, raw_pfc, output_dir, epoch, sf, f_range=(0.3, 30),
                            fmax=30, fmin=0.3):
    """
    Compute and visualize the fractal (aperiodic) component of PFC LFP signals per sleep state.

    This function:
        - Segments raw PFC data into epochs.
        - Calculates the fractal (aperiodic) component per epoch and sleep state using IRASA.
        - Computes slopes (raw and smoothed) for each epoch and state.
        - Aggregates mean and SEM across epochs per sleep state.
        - Generates a log-log plot of fractal power spectra with SEM shading.

    Parameters
    ----------
    states : array-like
        Sleep stage labels per epoch (numeric: 0=W, 1=N1, 2=N2, 3=N3, 4=REM).
    subject : str
        Subject identifier.
    raw_pfc : array-like
        Raw PFC signal to be analyzed.
    output_dir : str
        Directory to save the resulting plot.
    epoch : float
        Length of each epoch in seconds.
    sf : float
        Sampling frequency of the raw PFC signal.
    f_range : tuple, optional
        Frequency range (min, max) for fractal component calculation (default=(0.3, 30)).
    fmax : float, optional
        Maximum frequency for analysis (default=30 Hz).
    fmin : float, optional
        Minimum frequency for analysis (default=0.3 Hz).

    Returns
    -------
    eeg_in_epochs : list of arrays
        Raw PFC signal segmented into epochs.
    mean_raw_slope_by_state : dict
        Mean raw slope per sleep state.
    mean_smoothed_slope_by_state : dict
        Mean smoothed slope per sleep state.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # --- Initialize dictionaries for storing subject-specific data ---
    subject_fractal_data = {}  # stores fractal (aperiodic) spectra
    subject_slope_data = {}    # stores slopes (raw and smoothed)
    subject_states = {}        # stores sleep stage labels
    subject_states[subject] = states

    # --- Segment raw PFC data into epochs ---
    eeg_in_epochs = raw_to_epochs(raw_pfc, sf, epoch)

    # --- Calculate fractal component per sleep state ---
    subject_fractal_data[subject] = calc_fractal_component(states, eeg_in_epochs, sf, f_range)

    # --- Calculate slopes per epoch and sleep state ---
    subject_slope_data[subject] = calc_slopes(eeg_in_epochs, sf, f_range, states)

    # --- Aggregate aperiodic components by sleep state ---
    aperiodic_by_state = defaultdict(list)
    freqs_ref = {}

    for subject, subject_dict in subject_fractal_data.items():
        for state, (freqs, aperiodic) in subject_dict.items():
            aperiodic_by_state[state].append(aperiodic)
            freqs_ref[state] = freqs

    # --- Compute mean and SEM for each state ---
    mean_by_state = {}
    sem_by_state = {}
    for state, aperiodic_list in aperiodic_by_state.items():
        arr = np.stack(aperiodic_list, axis=0)  # shape: (n_epochs, n_freqs)
        mean_by_state[state] = (freqs_ref[state], np.mean(arr, axis=0))
        sem_by_state[state] = (freqs_ref[state], np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0]))

    # --- Aggregate slope data (raw and smoothed) ---
    raw_slope_by_state = defaultdict(list)
    smoothed_slope_by_state = defaultdict(list)

    for subject, (_, _, raw_slopes, smoothed_slopes) in subject_slope_data.items():
        for state, slope in raw_slopes.items():
            raw_slope_by_state[state].append(slope)
        for state, slope in smoothed_slopes.items():
            smoothed_slope_by_state[state].append(slope)

    # --- Compute mean and SEM for slopes ---
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

    # --- Plot fractal power component per sleep state ---
    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    stage_labels = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

    plt.figure(figsize=(10, 6))

    for state in sorted(mean_by_state.keys()):
        freqs, mean_aperiodic = mean_by_state[state]
        _, sem_aperiodic = sem_by_state[state]

        # Plot mean aperiodic power spectrum
        plt.plot(
            freqs, mean_aperiodic,
            label=stage_labels.get(state, f"State {state}"),
            alpha=0.9,
            color=colors.get(state, 'gray')
        )

        # Shaded area representing SEM
        plt.fill_between(
            freqs,
            mean_aperiodic - sem_aperiodic,
            mean_aperiodic + sem_aperiodic,
            color=colors.get(state, 'gray'),
            alpha=0.15
        )

    # --- Plot aesthetics ---
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-14, 1e-8)
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
    """
    Compute slopes per epoch using IRASA and return a full-length, interpolated,
    smoothed slope vector aligned with the hypnogram.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import zscore
    from scipy.signal import savgol_filter

    epoch_slopes = np.full(len(states), np.nan)  # full-length slope container

    # --- Compute slope for each epoch using IRASA ---
    for i, eeg_epoch in enumerate(eeg_in_epochs):

        try:
            freqs, psd_aperiodic, _ = compute_irasa(eeg_epoch, sf, f_range=f_range)
        except Exception:
            continue

        # Filter out invalid values
        valid = np.isfinite(psd_aperiodic) & (psd_aperiodic > 0)
        freqs = freqs[valid]
        psd_aperiodic = psd_aperiodic[valid]

        # Skip epochs with too few points
        if len(freqs) < 5:
            continue

        try:
            intercept, slope = fit_irasa(freqs, psd_aperiodic)
            epoch_slopes[i] = slope
        except Exception:
            continue

    # --- Interpolate missing slope epochs ---
    n = len(epoch_slopes)
    x = np.arange(n)
    mask = np.isfinite(epoch_slopes)

    # If too few valid epochs, avoid failure
    if mask.sum() < 2:
        raise ValueError("Not enough valid IRASA epochs to interpolate slopes.")

    # Fill missing values by linear interpolation
    epoch_slopes_interp = np.interp(x, x[mask], epoch_slopes[mask])

    # --- Z-score normalization ---
    raw_slopes = zscore(epoch_slopes_interp)

    # --- Savitzky-Golay smoothing ---
    window_length = min(101, len(raw_slopes) // 2 * 2 + 1)  # must be odd
    smoothed_slopes = savgol_filter(raw_slopes, window_length, polyorder=5, mode='interp')

    # Final z-score after smoothing (optional but recommended)
    smoothed_slopes = zscore(smoothed_slopes)

    # --- Compute mean slopes per state ---
    mean_slope_per_state = {}
    smoothed_mean_slope_per_state = {}
    for state in np.unique(states):
        mask_state = (states == state)
        mean_slope_per_state[state] = np.nanmean(raw_slopes[mask_state])
        smoothed_mean_slope_per_state[state] = np.nanmean(smoothed_slopes[mask_state])

    # Plotting for sanity check
    stages_sorted = sorted(mean_slope_per_state.keys())
    mean_slopes = [mean_slope_per_state[s] for s in stages_sorted]
    smoothed_means = [smoothed_mean_slope_per_state[s] for s in stages_sorted]

    plt.figure(figsize=(7, 6))
    plt.grid(axis='x', color='lightgray', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='gray', linewidth=1, alpha=0.5)

    plt.scatter(stages_sorted, mean_slopes, color='black', marker='s', s=60, label='raw slope')
    plt.scatter(stages_sorted, smoothed_means, color='green', marker='s', s=30, label='smoothed slope')

    plt.plot(stages_sorted, mean_slopes, color='black', linestyle='--', alpha=0.6)
    plt.plot(stages_sorted, smoothed_means, color='green', linestyle='--', alpha=0.6)

    plt.ylabel('Z-normalized slope')
    plt.xlabel('Sleep Stage')
    plt.title('Slope per state')
    plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/slope_per_state.svg", format="svg")

    return smoothed_slopes

def fractal_slope_vs_hypnogram(subject, smoothed_slopes, output_dir, states, epoch_length_sec=10):
    """
    Plot smoothed fractal slope across time alongside a color-coded hypnogram.

    This function:
        - Detects peaks and valleys in the smoothed slope.
        - Maps numeric sleep stages to a visually meaningful vertical layout.
        - Identifies REM clusters and merges short interruptions.
        - Generates a two-panel figure:
            1. Hypnogram with colored sleep stages and cycle annotations.
            2. Smoothed fractal slope time series, with colored segments for N3 and REM.

    Parameters
    ----------
    subject : str
        Identifier for the subject (used in plot title).
    smoothed_slopes : array-like
        Z-normalized, Savitzky-Golay smoothed fractal slopes per epoch.
    output_dir : str
        Directory where the plot will be saved.
    states : array-like
        Numeric sleep stage labels per epoch (0=W, 1=N1, 2=N2, 3=N3, 4=REM).
    epoch_length_sec : int, optional
        Duration of each epoch in seconds (default=10).

    Returns
    -------
    None
    """

    # --- Detect peaks and valleys in smoothed slope ---
    peaks, _ = find_peaks(smoothed_slopes, distance=120, prominence=2)
    valleys, _ = find_peaks(-smoothed_slopes, distance=120, prominence=2)

    subj_states = states
    n_epochs = len(subj_states)
    time_axis = np.arange(n_epochs) * epoch_length_sec / 60  # convert to minutes

    # --- Define sleep stage colors and labels ---
    stage_colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    stage_labels = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

    REM_code = 4
    min_rem_gap = 20  # merge interruptions ≤ 1 epoch

    # --- Identify first and last non-Wake epochs ---
    non_wake_idx = np.where(subj_states != 0)[0]
    first_non_wake = non_wake_idx[0] if len(non_wake_idx) > 0 else None
    last_non_wake = non_wake_idx[-1] if len(non_wake_idx) > 0 else None

    # --- Identify REM clusters ---
    is_rem = (subj_states == REM_code).astype(int)
    diff = np.diff(np.concatenate(([0], is_rem, [0])))
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0] - 1

    # Merge short interruptions in REM clusters
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

    # --- Map sleep stages to vertical positions ---
    desired_order = ['W', 'REM', 'N1', 'N2', 'N3']
    new_y_map = {label: i for i, label in enumerate(desired_order)}
    states_labels = np.array([stage_labels[s] for s in subj_states])
    states_new_y = np.array([new_y_map[s] for s in states_labels])

    # --- Create two-panel plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    # --- Panel 1: Hypnogram ---
    line_positions = [first_non_wake] + merged_ends if first_non_wake is not None else merged_ends
    if last_non_wake is not None and last_non_wake not in line_positions:
        line_positions.append(last_non_wake)
    line_positions = [0] + line_positions + [len(subj_states) - 1]
    line_times = time_axis[line_positions]

    # Black step line for hypnogram
    ax1.step(time_axis, states_new_y, where='post', color='black', linewidth=0.5, zorder=2)

    # Shade alternating regions and annotate cycles
    colors_shade = ['lightgreen', 'lightblue']
    for i in range(1, len(line_times) - 2):
        ax1.axvspan(line_times[i], line_times[i + 1], color=colors_shade[i % 2], alpha=0.5, zorder=1)
        center = (line_times[i] + line_times[i + 1]) / 2
        ax1.text(center, 0.5, f'Cycle {i}', ha='center', va='bottom', fontsize=10, color='black')

    # Shade areas before first and after last non-Wake/REM clusters
    ax1.axvspan(time_axis[0], line_times[1], color='lightgray', alpha=0.3, zorder=0)
    ax1.axvspan(line_times[-2], time_axis[-1], color='lightgray', alpha=0.3, zorder=0)

    # Draw vertical dashed lines for REM cluster boundaries
    for t in line_times[1:-1]:
        ax1.axvline(t, color='red', linestyle='--', alpha=0.7, zorder=3)

    # Scatter points colored by sleep stage
    for s_num in np.unique(subj_states):
        ids = np.where(subj_states == s_num)[0]
        ax1.scatter(time_axis[ids], states_new_y[ids],
                    color=stage_colors[s_num], label=stage_labels[s_num], s=10)

    ax1.set_yticks(range(len(desired_order)))
    ax1.set_yticklabels(desired_order)
    ax1.set_ylabel('Sleep Stage')
    ax1.invert_yaxis()
    ax1.set_xlim(time_axis[0], time_axis[-1])
    ax1.set_title(f'Hypnogram - {subject}')

    # --- Panel 2: Smoothed fractal slope ---
    difference = len(time_axis) - len(smoothed_slopes)
    if difference > 0:
        time_axis = time_axis[:-difference]

    ax2.plot(time_axis, smoothed_slopes, color='black', label='Fractal slope', linewidth=1)

    # Highlight N3 and REM segments in color
    stage_colors_highlight = {3: 'green', 4: 'red'}
    for stage, color in stage_colors_highlight.items():
        mask = (subj_states == stage)
        if not np.any(mask):
            continue
        segments = np.split(time_axis, np.where(np.diff(mask.astype(int)) != 0)[0] + 1)
        slope_segments = np.split(smoothed_slopes, np.where(np.diff(mask.astype(int)) != 0)[0] + 1)
        for t_seg, s_seg, m_seg in zip(segments, slope_segments,
                                       np.split(mask, np.where(np.diff(mask.astype(int)) != 0)[0] + 1)):
            if np.any(m_seg):
                min_len = min(len(t_seg), len(s_seg), len(m_seg))
                ax2.plot(t_seg[:min_len][m_seg[:min_len]], s_seg[:min_len][m_seg[:min_len]], color=color, linewidth=3)

    # Horizontal reference lines
    ax2.axhline(0, color='gray', linewidth=1, zorder=1)
    ax2.axhline(1, color='lightgray', linestyle='--', linewidth=1, zorder=1)
    ax2.axhline(-1, color='lightgray', linestyle='--', linewidth=1, zorder=1)

    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Z-normalized fractal slope')
    ax2.set_ylim(-3, 3)
    ax2.set_xlim(time_axis[0], time_axis[-1])
    ax2.set_title(f'Fractal slopes - {subject}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fractalslope_vs_hypnogram.svg", format='svg')
    # plt.show()

def fooof_report(output_dir, raw_pfc, fs):
    """
    Generate a FOOOF report for a single channel of PFC data and save the figure.

    This function:
        - Computes the power spectral density (PSD) of the raw signal using Welch's method.
        - Initializes a FOOOF model with a fixed aperiodic component and peak width limits.
        - Fits the FOOOF model to the PSD.
        - Generates a FOOOF report figure.
        - Saves the figure to the specified output directory.

    Parameters
    ----------
    output_dir : str
        Directory where the FOOOF report figure will be saved.
    raw_pfc : array-like
        Raw prefrontal cortex signal to analyze (1D array).
    fs : float
        Sampling frequency of the raw signal in Hz.

    Returns
    -------
    None
    """
    # --- Compute Power Spectral Density ---
    freqs, psd = welch(raw_pfc, fs=fs, nperseg=1024)

    # --- Initialize FOOOF model ---
    # peak_width_limits=[2, 8] sets allowed widths for detected peaks (Hz)
    # aperiodic_mode='fixed' fits a single offset and exponent for the background
    fm = FOOOF(peak_width_limits=[2, 8], aperiodic_mode='fixed')

    # --- Generate FOOOF report ---
    # Frequency range for fitting: 1-50 Hz
    # plt_log=True plots axes in log scale
    fm.report(freqs, psd, [1, 50], plt_log=True)

    # --- Save report to file ---
    plt.savefig(f"{output_dir}/fooof_report.png", dpi=300, bbox_inches='tight')
    # plt.show()  # optionally display figure

def aperiodic_fit_bar(valid_states, normalized_exponents, output_dir):
    """
    Generate a bar plot of normalized aperiodic exponents per sleep state, with SEM error bars.

    This function:
        - Aggregates aperiodic exponents by sleep state.
        - Computes the mean and standard error of the mean (SEM) per state.
        - Creates a bar plot with colors per sleep stage.
        - Saves the plot to the specified output directory.

    Parameters
    ----------
    valid_states : array-like
        Numeric sleep stage labels per epoch (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM).
    normalized_exponents : array-like
        Corresponding normalized aperiodic exponent values.
    output_dir : str
        Directory path where the plot will be saved.

    Returns
    -------
    None
    """

    # Define colors for each sleep stage
    colors = ['royalblue', 'teal', 'purple', 'forestgreen', 'firebrick']

    # --- Prepare DataFrame ---
    df = pd.DataFrame({'state': valid_states, 'aperiodic': normalized_exponents})

    # --- Compute summary statistics per state ---
    summary = df.groupby('state')['aperiodic'].agg(['mean', 'sem']).reset_index()
    print(summary)  # Optional: print table of means and SEMs for verification

    # --- Create bar plot ---
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary['state'],
        summary['mean'],
        yerr=summary['sem'],           # Error bars = SEM
        capsize=5,                      # End caps for error bars
        color=[colors[int(s)] for s in summary['state']],  # Color per state
        edgecolor='black',
        zorder=2,
        alpha=0.6
    )

    # --- Plot formatting ---
    plt.xticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM'])
    plt.ylim(-1.1, 1.1)
    plt.xlabel('Sleep State')
    plt.ylabel('Normalized mean aperiodic fit')
    plt.title('Aperiodic per Sleep State')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()

    # --- Save plot ---
    plt.savefig(f"{output_dir}/Aperiodic_fit_bar.svg", format='svg')
    # plt.show()  # Optionally display figure

def aperiodic_fit_violin(sleep_states, aperiodic_exponents, output_dir):
    """
    Generate a violin plot of aperiodic exponents per sleep stage, with medians,
    individual data points, and SEM bars overlaid.

    Args:
        sleep_states (list or array-like): Numeric sleep stage labels per epoch
            (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM).
        aperiodic_exponents (list or array-like): Corresponding aperiodic exponent values.
        output_dir (str): Directory where the plot will be saved.

    Returns:
        None
    """

    # --- Prepare DataFrame for seaborn ---
    df_plot = pd.DataFrame({'state': sleep_states, 'aperiodic': aperiodic_exponents})

    # --- Define plotting order and labels ---
    all_states = [0, 1, 2, 3, 4]
    labels = ['W', 'N1', 'N2', 'N3', 'REM']

    # --- Extract data for computing medians and SEM ---
    data_for_violin = [df_plot.loc[df_plot['state'] == s, 'aperiodic'].values for s in all_states]

    # --- Define colors per sleep stage ---
    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    palette = [colors[s] for s in all_states]

    plt.figure(figsize=(8, 6))

    # --- Violin plot ---
    ax = sns.violinplot(
        x='state', y='aperiodic', data=df_plot,
        order=all_states,
        palette=palette,
        cut=0,
        bw='scott',
        inner=None
    )

    # --- Overlay individual data points with jitter ---
    sns.stripplot(
        x='state', y='aperiodic', data=df_plot,
        order=all_states,
        color='k', size=1.5, jitter=0.15, alpha=0.3
    )

    # --- Overlay median markers ---
    medians = [np.nanmedian(d) if len(d) > 0 else np.nan for d in data_for_violin]
    for i, m in enumerate(medians):
        if not np.isnan(m):
            plt.plot(i, m, marker='o', color='white', markeredgecolor='black',
                     markersize=6, zorder=10)

    # --- Overlay mean ± SEM bars ---
    for i, d in enumerate(data_for_violin):
        if len(d) > 0:
            mean_val = np.nanmean(d)
            sem_val = sem(d, nan_policy='omit')
            plt.errorbar(i, mean_val, yerr=sem_val, color='black', capsize=4, zorder=11)

    # --- Labels and aesthetics ---
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State')
    ax.set_ylabel('Normalized aperiodic fit')
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('Aperiodic fit per Sleep State (violin + SEM)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Aperiodic_fit_violin.svg", format="svg")
    plt.close()

def index_N(delta, alpha, EMG, EOG1, EOG2, epoch_length, fs, gamma):
    """
    Compute non-REM sleep index per epoch using EOG features.

    Formula: (EOG_feature * delta) / (alpha * EMG)
    """
    eog_features = wei_normalizing(N_feature(EOG1, EOG2, epoch_length, fs))
    eog_features = np.convolve(
        np.convolve(
            np.convolve(eog_features, np.ones(5) / 5, mode='same'),
            np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )

    alt_index_n = np.array([(eog_features[i] * delta[i]) / (gamma[i]**2) for i in range(len(delta))])
    return alt_index_n

def index_R(delta, sigma, EMG, EOG1, EOG2, epoch_length, fs):
    """
    Compute REM sleep index per epoch using EOG features.

    Formula: (EOG_feature^2) / (EMG^2 * delta * sigma)
    """
    eog_features = wei_normalizing(rem_feature(EOG1, EOG2, epoch_length, fs))
    eog_features = np.convolve(
        np.convolve(
            np.convolve(eog_features, np.ones(5) / 5, mode='same'),
            np.ones(5) / 5, mode='same'),
        np.ones(5) / 5, mode='same'
    )

    alt_index_r = np.array([(eog_features[i] ** 2) / (delta[i]*delta[i]*EMG[i]**2) for i in range(len(delta))])
    return alt_index_r

def index_W(theta, gamma, EMG):
    """
    Compute wake index per epoch using EMG and EEG frequency bands.

    Formula: EMG^2 * (gamma / theta)
    """
    index_w = np.array([EMG[i] ** 2 * (gamma[i] / theta[i]) for i in range(len(theta))])
    return index_w

def auto_correlation_slope(EOG, epoch_length, fs, min_slope_idx=1):
    """
    Compute slope of auto-correlation for each epoch of an EOG signal.

    Slope is calculated from lag 0 to the first significant peak (ignoring lag 0).

    Args:
        EOG: EOG signal.
        epoch_length: Epoch length in seconds.
        fs: Sampling frequency.
        min_slope_idx: Minimum lag index to consider for slope.

    Returns:
        np.array: Slopes per epoch.
    """
    samples = int(epoch_length * fs)
    slopes = []

    for i in range(0, len(EOG) - len(EOG) % samples, samples):
        epoch = EOG[i:i + samples]

        # Compute full auto-correlation
        ac_full = np.correlate(epoch, epoch, mode='full')
        center = len(ac_full) // 2
        ac = ac_full[center:]  # only non-negative lags

        # Normalize to 1 at lag 0
        if ac[0] == 0:
            slopes.append(np.nan)
            continue
        ac = ac / ac[0]

        # Identify peaks for slope calculation
        peaks, _ = find_peaks(ac)
        valid_peaks = [p for p in peaks if p >= min_slope_idx]
        if len(valid_peaks) < 1:
            slopes.append(np.nan)
            continue

        first_peak_idx = valid_peaks[0]
        slope = (ac[first_peak_idx] - ac[0]) / first_peak_idx
        slopes.append(slope)

    return np.array(slopes)

def normalized_cross_correlation(epoch1, epoch2):
    """
    Compute normalized cross-correlation between two signals.

    Returns a value in [-1, 1].
    """
    x = epoch1 - np.mean(epoch1)
    y = epoch2 - np.mean(epoch2)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return 0.0
    return np.sum(x * y) / denom

def cross_correlation(EOG1, EOG2, epoch_length, fs, lag=0):
    """
    Compute normalized cross-correlation between two EOG signals epoch-wise.

    Args:
        EOG1, EOG2: EOG signals.
        epoch_length: Epoch duration (seconds).
        fs: Sampling frequency (Hz).
        lag: Not currently used.

    Returns:
        np.array: Cross-correlation values per epoch.
    """
    samples = int(epoch_length * fs)
    out = []

    for i in range(0, len(EOG1) - len(EOG1) % samples, samples):
        e1 = EOG1[i:i + samples]
        e2 = EOG2[i:i + samples]
        out.append(normalized_cross_correlation(e1, e2))

    return np.array(out)

def N_feature(EOG1, EOG2, epoch_length, fs):
    """
    Compute a combined feature from two EOG signals for non-REM sleep detection.

    Steps:
    1. Downsample EOG signals to 50 Hz for computational efficiency.
    2. Bandpass filter between 0.3–0.45 Hz to focus on relevant slow eye movements.
    3. Compute cross-correlation between the two EOG channels for each epoch.
    4. Compute the slope of the auto-correlation of EOG1 for each epoch.
    5. Combine cross-correlation and auto-correlation slope to produce the feature.

    Args:
        EOG1 (np.array): First EOG channel.
        EOG2 (np.array): Second EOG channel.
        epoch_length (float): Epoch duration in seconds.
        fs (float): Original sampling frequency of EOG signals.

    Returns:
        np.array: Feature vector for each epoch.
    """
    c_features = np.array([])

    # Downsample EOG signals to 50 Hz
    EOG1 = decimate(EOG1, int(fs / 50))
    EOG2 = decimate(EOG2, int(fs / 50))
    fs = 50

    # Bandpass filter between 0.3–0.45 Hz
    b, a = butter(4, [0.3 / (0.5 * fs), 0.45 / (0.5 * fs)], btype='band')
    EOG1 = filtfilt(b, a, EOG1)
    EOG2 = filtfilt(b, a, EOG2)

    # Compute cross-correlation between EOG channels
    cross_cor_val = cross_correlation(EOG1, EOG2, epoch_length, fs, 0)

    # Compute auto-correlation slope of EOG1
    auto_corr_val = auto_correlation_slope(EOG1, epoch_length, fs)

    # Combine features
    for count, ac in enumerate(auto_corr_val):
        feature = (1 - ac) * cross_cor_val[count]
        c_features = np.append(c_features, feature)

    return c_features

def rem_feature(EOG1, EOG2, epoch_length, fs, slope_tol=1e-6, max_inv=1e6, clip_abs=None):
    """
    Extract REM-specific features from EOG channels.

    REM feature is calculated as sign(cross-correlation) * inverse auto-correlation slope.

    Args:
        EOG1, EOG2: EOG channels.
        epoch_length: Duration of epochs in seconds.
        fs: Sampling frequency.
        slope_tol: Minimum slope threshold to avoid division by near-zero.
        max_inv: Maximum allowed inverse value.
        clip_abs: Clip final feature values to [-clip_abs, clip_abs].

    Returns:
        np.array: REM features per epoch.
    """
    nyq = 0.5 * fs
    low = 0.3 / nyq
    high = 35.0 / nyq

    if not (0 < low < 1 and 0 < high < 1 and low < high):
        raise ValueError(f"Filter cutoff frequencies invalid for fs={fs}")
    if len(EOG1) < 20 or len(EOG2) < 20:
        raise ValueError("Signals too short for reliable filtering.")

    # Bandpass filter
    b, a = butter(4, [low, high], btype='band')
    EOG1_f = filtfilt(b, a, EOG1)
    EOG2_f = filtfilt(b, a, EOG2)

    cross_cor_val = cross_correlation(EOG1_f, EOG2_f, epoch_length, fs, 0)
    auto_slope = auto_correlation_slope(EOG1_f, epoch_length, fs)

    features = []
    for cc, slope in zip(cross_cor_val, auto_slope):
        if np.isnan(slope) or abs(slope) < slope_tol:
            inv = 0.0
        else:
            inv = 1.0 / slope
            if not np.isfinite(inv):
                inv = 0.0
            if max_inv is not None:
                inv = np.clip(inv, -max_inv, max_inv)

        feat = np.sign(cc) * inv
        features.append(feat)

    feats = np.array(features)
    if clip_abs is not None:
        feats = np.clip(feats, -clip_abs, clip_abs)
    return feats

def dfa_plot(lfp_PFC, output_dir, length, fs):
    """
    Compute detrended fluctuation analysis (DFA) exponent over sliding windows of LFP data,
    smooth and normalize the results, and save a line plot over time.

    Parameters
    ----------
    lfp_PFC : array-like
        Local field potential signal from the prefrontal cortex (PFC).
    output_dir : str
        Directory path to save output figure.
    length : int
        Window length in seconds for DFA computation.
    fs : int or float
        Sampling frequency of the LFP signal (Hz).

    Returns
    -------
    normalized_dfa : np.ndarray
        Smoothed and normalized DFA exponents per window.
    """
    window_size = step_size = fs * length
    num_windows = (len(lfp_PFC) - window_size) // step_size + 1

    dfa_exponents = []
    time_stamps = []

    # --- Compute DFA per window ---
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        segment = lfp_PFC[start:end]

        # compute_fluctuations should return (scale, fluctuation, exponent)
        _, _, exp_window = compute_fluctuations(segment, fs, n_scales=10,
                                                min_scale=0.05, max_scale=4.0)
        dfa_exponents.append(exp_window)
        time_stamps.append((start + end) / 2 / fs)  # timestamp at window center

    dfa_exponents = np.array(dfa_exponents)

    # --- Smooth using Savitzky-Golay filter ---
    # window_length must be odd and smaller than data length
    window_length = 11 if len(dfa_exponents) >= 11 else len(dfa_exponents) | 1
    polyorder = 4
    smoothed_dfa = savgol_filter(dfa_exponents, window_length=window_length, polyorder=polyorder)

    # --- Normalize to [-1, 1] for visualization ---
    normalized_dfa = 2 * ((smoothed_dfa - min(smoothed_dfa)) / (max(smoothed_dfa) - min(smoothed_dfa))) - 1

    # --- Plot DFA over time ---
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
    """
    Compute z-normalized DFA exponents per sleep stage, smooth across epochs, and
    generate a scatter plot showing raw and smoothed values for each sleep stage.

    Parameters
    ----------
    normalized_dfa : array-like
        DFA exponents (smoothed and normalized) per epoch.
    states : array-like
        Numeric sleep stage labels per epoch (0=W, 1=N1, 2=N2, 3=N3, 4=REM).
    output_dir : str
        Directory path to save output figure.

    Returns
    -------
    smoothed_dfa : np.ndarray
        Smoothed DFA exponents per epoch after z-scoring.
    """
    # --- Convert to numpy arrays and align lengths ---
    dfa_values = np.array(normalized_dfa)
    valid_states = np.array(states).astype(int)
    min_len = min(len(dfa_values), len(valid_states))
    dfa_values = dfa_values[:min_len]
    valid_states = valid_states[:min_len]

    # --- Remove NaNs if present ---
    nan_mask = ~np.isnan(dfa_values)
    dfa_values = dfa_values[nan_mask]
    valid_states = valid_states[nan_mask]

    # --- Z-score DFA values ---
    raw_dfa = zscore(dfa_values, nan_policy='omit')

    # --- Smooth using Savitzky-Golay filter ---
    window_length = min(101, len(raw_dfa) // 2 * 2 + 1)  # must be odd
    smoothed_dfa = savgol_filter(raw_dfa, window_length, polyorder=5, mode='interp')
    smoothed_dfa = zscore(smoothed_dfa, nan_policy='omit')

    # --- Compute mean DFA per sleep state ---
    mean_dfa_per_state = {}
    smoothed_mean_dfa_per_state = {}
    for state in np.unique(valid_states):
        mask = valid_states == state
        mean_dfa_per_state[state] = np.nanmean(raw_dfa[mask])
        smoothed_mean_dfa_per_state[state] = np.nanmean(smoothed_dfa[mask])

    print("Means per state:", mean_dfa_per_state)

    # --- Sort states and prepare for plotting ---
    stages_sorted = sorted(mean_dfa_per_state.keys())
    mean_dfas = [mean_dfa_per_state[s] for s in stages_sorted]
    smoothed_mean_dfas = [smoothed_mean_dfa_per_state[s] for s in stages_sorted]

    # Map numeric state codes to labels
    state_labels_map = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    stage_labels = [state_labels_map.get(s, str(s)) for s in stages_sorted]

    # --- Scatter plot with raw and smoothed DFA ---
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
    plt.savefig(f"{output_dir}/dfa_per_state.svg", format='svg')
    # plt.show()

    return smoothed_dfa

def dfa_violin_and_bar(normalized_dfa, states, output_dir):
    """
    Generate bar and violin plots of normalized DFA exponents per sleep state.

    Parameters
    ----------
    normalized_dfa : array-like
        Normalized DFA exponents for each epoch.
    states : array-like
        Numeric sleep stage labels per epoch (0=W, 1=N1, 2=N2, 3=N3, 4=REM).
    output_dir : str
        Directory path to save plots.

    Returns
    -------
    None
    """
    # --- Align lengths of data and state arrays ---
    min_len = min(len(normalized_dfa), len(states))
    normalized_dfa = normalized_dfa[:min_len]
    states = states[:min_len]

    # --- Prepare DataFrame and summary statistics ---
    df = pd.DataFrame({'state': states, 'dfa': normalized_dfa})
    summary = df.groupby('state')['dfa'].agg(['mean', 'sem']).reset_index()

    colors_list = ['royalblue', 'teal', 'purple', 'forestgreen', 'firebrick']
    labels = ['W', 'N1', 'N2', 'N3', 'REM']

    # --- Bar plot with mean ± SEM ---
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary['state'], summary['mean'],
        yerr=summary['sem'],
        capsize=5,
        color=[colors_list[int(s)] for s in summary['state']],
        edgecolor='black',
        zorder=2,
        alpha=0.6
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

    # --- Violin plot with jittered points and SEM overlay ---
    colors_dict = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    all_states = list(range(5))
    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    plt.figure(figsize=(8, 6))
    palette = [colors_dict[s] for s in all_states]

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

    # --- Overlay jittered individual data points per state ---
    for i, s in enumerate(all_states):
        vals = df_plot.loc[df_plot['state'] == s, 'dfa'].values
        if len(vals) > 0:
            x = np.random.normal(loc=i, scale=0.15, size=len(vals))
            ax.scatter(x, vals, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    # --- Overlay medians and SEM per state ---
    for i, s in enumerate(all_states):
        vals = df_plot.loc[df_plot['state'] == s, 'dfa'].values
        if len(vals) == 0:
            continue
        median_val = np.nanmedian(vals)
        sem_val = np.nanstd(vals) / np.sqrt(len(vals))
        plt.plot(i, median_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=10)
        plt.errorbar(i, median_val, yerr=sem_val, color='black', capsize=4, elinewidth=1.5, markeredgewidth=1, zorder=9)

    # --- Final cosmetic adjustments ---
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
    """
    Compute modified multiscale entropy (MSE) for consecutive windows of an LFP signal,
    smooth and normalize the results, and save a line plot.

    Parameters
    ----------
    lfp_PFC : array-like
        Local field potential signal from PFC.
    output_dir : str
        Directory to save the resulting figure.
    length : int
        Window length in seconds.
    fs : int or float
        Sampling frequency of the LFP signal (Hz).

    Returns
    -------
    normalized_mse : np.ndarray
        Smoothed and normalized MSE per window.
    """
    # --- Initialize MSE object from EH toolbox ---
    Mobj = EH.MSobject('IncrEn', m=2, R=3, Norm=True)

    window_size = step_size = fs * length
    num_windows = (len(lfp_PFC) - window_size) // step_size + 1

    mse_values = []
    time_stamps_mse = []

    # --- Compute MSE per window ---
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        segment = lfp_PFC[start:end]

        MSx, _ = EH.MSEn(segment, Mobj, Scales=2, Methodx='modified')
        mse_values.append(np.mean(MSx))
        time_stamps_mse.append((start + end) / 2 / fs)

    mse_values = np.array(mse_values)
    time_stamps_mse = np.array(time_stamps_mse)

    # --- Smooth using Savitzky-Golay filter ---
    window_length = 11 if len(mse_values) >= 11 else len(mse_values) | 1  # must be odd
    polyorder = 4
    smoothed_mse = savgol_filter(mse_values, window_length=window_length, polyorder=polyorder)

    # --- Normalize between -1 and 1 ---
    normalized_mse = 2 * ((smoothed_mse - min(smoothed_mse)) / (max(smoothed_mse) - min(smoothed_mse))) - 1

    # --- Plot ---
    plt.figure(figsize=(18, 5))
    plt.plot(time_stamps_mse, normalized_mse, marker='.', linestyle='-', color=Red)
    plt.xlabel('Time (s)')
    plt.ylabel('MSE')
    plt.title('MSE Over Time (10-sec Windows) - RGS14')
    plt.grid()
    plt.savefig(f"{output_dir}/MSE_10s.svg", format='svg')

    return normalized_mse

def mse_per_state(normalized_mse, states, output_dir):
    """
    Compute mean and smoothed MSE per sleep state and generate a scatter/line plot.

    Parameters
    ----------
    normalized_mse : array-like
        Normalized mean-squared error values per epoch.
    states : array-like
        Numeric sleep state labels per epoch (e.g., 0=W, 1=N1, 2=N2, 3=N3, 4=REM).
    output_dir : str
        Directory to save the resulting figure.

    Returns
    -------
    smoothed_mse : np.ndarray
        Smoothed and z-scored MSE values for each epoch.
    """
    # --- Align lengths ---
    min_len = min(len(normalized_mse), len(states))
    normalized_mse = normalized_mse[:min_len]
    states = states[:min_len]

    # --- Convert to arrays ---
    mse_values = np.array(normalized_mse)
    valid_states = np.array(states).astype(int)

    # --- Remove NaNs for safety ---
    nan_mask = ~np.isnan(mse_values)
    mse_values = mse_values[nan_mask]
    valid_states = valid_states[nan_mask]

    # --- Z-score normalization ---
    raw_mse = zscore(mse_values, nan_policy='omit')

    # --- Smooth MSE using Savitzky-Golay filter ---
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

    # --- Map numeric codes to human-readable labels ---
    stages_sorted = sorted(mean_mse_per_state.keys())
    state_labels_map = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    stage_labels = [state_labels_map.get(s, str(s)) for s in stages_sorted]

    # --- Prepare mean values for plotting ---
    mean_mses = [mean_mse_per_state[s] for s in stages_sorted]
    smoothed_mean_mses = [smoothed_mean_mse_per_state[s] for s in stages_sorted]

    # --- Plot ---
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

    return smoothed_mse

def mse_violin_and_bar(normalized_mse, states, output_dir):
    """
    Generate a bar plot and violin plot of MSE per sleep state, including SEM and median markers.

    Parameters
    ----------
    normalized_mse : array-like
        Normalized MSE values per epoch.
    states : array-like
        Numeric sleep stage labels per epoch.
    output_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """
    # --- Align lengths ---
    min_len = min(len(normalized_mse), len(states))
    normalized_mse = normalized_mse[:min_len]
    states = states[:min_len]

    # --- Create dataframe ---
    df = pd.DataFrame({'state': states, 'mse': normalized_mse})
    summary = df.groupby('state')['mse'].agg(['mean', 'sem']).reset_index()

    # --- Define colors and labels ---
    colors = ['royalblue', 'teal', 'purple', 'forestgreen', 'firebrick']
    labels = ['W', 'N1', 'N2', 'N3', 'REM']

    # --- Bar plot ---
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

    # --- Violin plot ---
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
        alpha=0.5,
        zorder=2
    )

    # --- Overlay jittered points per state ---
    for i, s in enumerate(all_states):
        vals = df_plot.loc[df_plot['state'] == s, 'mse'].values
        if len(vals) > 0:
            x = np.random.normal(loc=i, scale=0.15, size=len(vals))
            ax.scatter(x, vals, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    # --- Overlay medians and SEM ---
    for i, s in enumerate(all_states):
        vals = df_plot.loc[df_plot['state'] == s, 'mse'].values
        if len(vals) == 0:
            continue
        median_val = np.nanmedian(vals)
        sem_val = np.nanstd(vals) / np.sqrt(len(vals))
        plt.plot(i, median_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=10)
        plt.errorbar(i, median_val, yerr=sem_val, color='black', capsize=4, elinewidth=1.5, markeredgewidth=1, zorder=9)

    # --- Cosmetics ---
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