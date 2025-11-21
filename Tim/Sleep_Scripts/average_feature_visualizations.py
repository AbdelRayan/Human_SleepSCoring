import os
import sys
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from neurodsp.aperiodic import compute_irasa, fit_irasa, compute_fluctuations
import seaborn as sns
import EntropyHub as EH
import Human_SleepSCoring.Tim.Sleep_Scripts.Cleaned_feature_vis as C

@contextmanager
def suppress_stdout():
    """
    Context manager to temporarily suppress all printing to stdout.

    Usage:
        with suppress_stdout():
            # code that prints will not appear
            do_something()
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def extract_index_values_per_state(index_n, index_r, index_w, mapped_scores):
    """
    Extract mean index values per sleep state for three indices (W, R, N).

    Args:
        index_n (array-like): Index N values per epoch.
        index_r (array-like): Index R values per epoch.
        index_w (array-like): Index W values per epoch.
        mapped_scores (array-like): Sleep stage labels per epoch (numeric 0-4).

    Returns:
        dict: Dictionary with mean index values per sleep stage, ready for plotting.
              Structure: {state: {'w': mean_w, 'r': mean_r, 'n': mean_n}, ...}
    """
    # Smooth and normalize indices
    smoothed = {
        'w': C.smooth_and_norm(index_w),
        'r': C.smooth_and_norm(index_r),
        'n': C.smooth_and_norm(index_n)
    }

    # Print unique states for verification
    unique_states = np.unique(mapped_scores)
    print("Unique mapped states:", unique_states)

    # Map numerical sleep scores to human-readable labels
    score_labels = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    mapped_scores = [score_labels[s] for s in mapped_scores if s in score_labels]

    # Initialize containers for each state
    states = ['Wake', 'N1', 'N2', 'N3', 'REM']
    indices = {state: {'w': [], 'r': [], 'n': []} for state in states}

    # Populate indices for each state
    for i, state in enumerate(mapped_scores):
        if state in indices:
            for k in smoothed.keys():
                indices[state][k].append(smoothed[k][i])

    # Compute mean index values per state, safely handling empty lists
    means = {
        state: {k: np.mean(v) if len(v) > 0 else np.nan for k, v in vals.items()}
        for state, vals in indices.items()
    }

    return means

def plot_index_barplot(means, output_dir):
    """
    Plot a barplot of mean indices per sleep state.

    Args:
        means (dict): Mean index values per state, from extract_index_values_per_state.
        output_dir (str): Directory path to save the plot SVG.
    """
    states = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # Extract index values for plotting
    values_w = [means[s]['w'] for s in states]
    values_r = [means[s]['r'] for s in states]
    values_n = [means[s]['n'] for s in states]

    plt.figure(figsize=(12, 6))
    fontsize = 5
    x = np.arange(len(states))
    bar_width = 0.25

    # Plot bars for each index
    plt.bar(x - bar_width, values_w, width=bar_width, label='Index W', color='black')
    plt.bar(x, values_r, width=bar_width, label='Index R', color='red')
    plt.bar(x + bar_width, values_n, width=bar_width, label='Index N', color='blue')

    plt.xlabel('Sleep Stages')
    plt.ylabel('Average Index Values')
    plt.title('Wei Indices per Sleep State')
    plt.xticks(x, states)
    plt.legend(loc='upper center', bbox_to_anchor=(0.3, 1))

    # Add numeric values on top of bars
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

def prepare_aperiodic_violin_data(valid_states, normalized_exponents):
    """
    Prepare data for violin plot of aperiodic exponents per sleep state.

    Args:
        valid_states (array-like): Sleep stage labels per epoch.
        normalized_exponents (array-like): Normalized aperiodic exponents per epoch.

    Returns:
        tuple: (df_plot, data_for_violin, all_states, labels)
            df_plot: DataFrame for seaborn plotting.
            data_for_violin: List of arrays per sleep stage.
            all_states: List of stage indices [0..4].
            labels: Human-readable labels ['W', 'N1', 'N2', 'N3', 'REM'].
    """
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    all_states = list(range(5))
    df = pd.DataFrame({'state': valid_states, 'aperiodic': normalized_exponents})
    data_for_violin = [df.loc[df['state'] == s, 'aperiodic'].values for s in all_states]

    counts = [len(d) for d in data_for_violin]
    print("Counts per state (0..4):", counts)
    print("Unique states present:", sorted(df['state'].unique()))

    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)
    return df_plot, data_for_violin, all_states, labels

def prepare_dfa_violin_data(valid_states, fs, length, lfp_PFC):
    """
    Prepare data for violin plot of DFA exponents per sleep state.

    Args:
        valid_states (array-like): Sleep stage labels per epoch.
        fs (float): Sampling frequency of LFP.
        length (int): Window length in seconds.
        lfp_PFC (array-like): LFP signal from PFC.

    Returns:
        tuple: (df_plot, data_for_violin, all_states, labels, normalized_dfa)
    """
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    all_states = list(range(5))
    window_size = step_size = fs * length
    num_windows = (len(lfp_PFC) - window_size) // step_size + 1

    dfa_exponents = []
    time_stamps = []

    # Compute DFA for each segment
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        segment = lfp_PFC[start:end]
        _, _, exp_window = compute_fluctuations(segment, fs, n_scales=10,
                                                min_scale=0.05, max_scale=4.0)
        dfa_exponents.append(exp_window)
        time_stamps.append((start + end) / 2 / fs)

    # Smooth DFA and normalize between -1 and 1
    dfa_exponents = np.array(dfa_exponents)
    window_length = 11 if len(dfa_exponents) >= 11 else len(dfa_exponents) | 1
    smoothed_dfa = savgol_filter(dfa_exponents, window_length=window_length, polyorder=4)
    normalized_dfa = 2 * ((smoothed_dfa - min(smoothed_dfa)) / (max(smoothed_dfa) - min(smoothed_dfa))) - 1

    # Match lengths with valid states
    min_length = min(len(normalized_dfa), len(valid_states))
    valid_states = valid_states[:min_length]
    normalized_dfa = normalized_dfa[:min_length]
    df = pd.DataFrame({'state': valid_states, 'dfa': normalized_dfa})
    data_for_violin = [df.loc[df['state'] == s, 'dfa'].values for s in all_states]

    counts = [len(d) for d in data_for_violin]
    print("Counts per state (0..4):", counts)
    print("Unique states present:", sorted(df['state'].unique()))

    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    return df_plot, data_for_violin, all_states, labels, normalized_dfa

def plot_aperiodic_violin(df_plot, data_for_violin, all_states, labels, output_dir):
    """
    Generate violin plot of aperiodic exponents per sleep stage.

    Args:
        df_plot: DataFrame from prepare_aperiodic_violin_data.
        data_for_violin: List of arrays for each sleep state.
        all_states: List of state indices [0..4].
        labels: List of labels ['W', 'N1', 'N2', 'N3', 'REM'].
        output_dir: Directory to save plot.
    """
    # Define colors per sleep stage
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

    # Overlay individual points with slight jitter
    sns.stripplot(
        x='state', y='aperiodic', data=df_plot,
        order=all_states,
        color='k', size=1.5, jitter=0.15, alpha=0.3
    )

    # Overlay median markers
    medians = [np.nanmedian(d) if len(d) > 0 else np.nan for d in data_for_violin]
    for i, m in enumerate(medians):
        if not np.isnan(m):
            plt.plot(i, m, marker='o', color='white', markeredgecolor='black',
                     markersize=6, zorder=10)

    # Set labels and aesthetics
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State')
    ax.set_ylabel('Normalized mean aperiodic fit')
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('Aperiodic fit per Sleep State (violin)')
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Aperiodic_fit_violin.svg", format="svg")

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
    Plot a group-level violin plot of normalized aperiodic exponents per sleep stage,
    with overlaid mean ± SEM markers.

    Parameters
    ----------
    aperiodic_violin : dict
        Dictionary containing:
            - 'data_for_violin': list or dict of values per sleep state
            - 'all_states': list of numeric sleep states to plot
            - 'labels': list of labels for x-axis
    output_dir : str
        Directory to save the resulting plot.

    Returns
    -------
    None
    """
    data_for_violin = aperiodic_violin['data_for_violin']
    all_states = aperiodic_violin['all_states']
    labels = aperiodic_violin['labels']

    # Define color palette per sleep state
    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    palette = [colors[s] for s in all_states]

    # Build ordered data list for violin plot
    data_list = [data_for_violin[s] for s in all_states]

    plt.figure(figsize=(8, 6))

    # --- Violin plot (background) ---
    ax = sns.violinplot(
        data=data_list,
        palette=palette,
        cut=0,
        bw='scott',
        inner=None,
        alpha=0.5,
        zorder=2
    )

    # --- Overlay jittered points for individual values ---
    for i, d in enumerate(data_list):
        x = np.random.normal(loc=i, scale=0.15, size=len(d))
        ax.scatter(x, d, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    # --- Overlay mean and SEM per state ---
    for i, d in enumerate(data_list):
        d = np.array(d)
        d = d[~np.isnan(d)]
        if len(d) == 0:
            continue

        mean_val = np.nanmean(d)
        sem_val = np.nanstd(d) / np.sqrt(len(d))

        # Mean marker
        plt.plot(i, mean_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=4)

        # SEM vertical line
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
    Plot a group-level violin plot of normalized DFA exponents per sleep stage,
    with overlaid mean ± SEM markers.

    Parameters
    ----------
    dfa_violin : dict
        Dictionary containing:
            - 'data_for_violin': list or dict of values per sleep state
            - 'all_states': list of numeric sleep states to plot
            - 'labels': list of labels for x-axis
    output_dir : str
        Directory to save the resulting plot.

    Returns
    -------
    None
    """
    data_for_violin = dfa_violin['data_for_violin']
    all_states = dfa_violin['all_states']
    labels = dfa_violin['labels']

    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    palette = [colors[s] for s in all_states]

    data_list = [data_for_violin[s] for s in all_states]

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

    for i, d in enumerate(data_list):
        x = np.random.normal(loc=i, scale=0.15, size=len(d))
        ax.scatter(x, d, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    for i, d in enumerate(data_list):
        d = np.array(d)
        d = d[~np.isnan(d)]
        if len(d) == 0:
            continue

        mean_val = np.nanmean(d)
        sem_val = np.nanstd(d) / np.sqrt(len(d))

        plt.plot(i, mean_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=4)
        plt.errorbar(i, mean_val, yerr=sem_val, color='black', capsize=4, elinewidth=1.5, markeredgewidth=1, zorder=3)

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
    Plot a group-level violin plot of normalized MSE exponents per sleep stage,
    with overlaid mean ± SEM markers.

    Parameters
    ----------
    mse_violin : dict
        Dictionary containing:
            - 'data_for_violin': list or dict of values per sleep state
            - 'all_states': list of numeric sleep states to plot
            - 'labels': list of labels for x-axis
    output_dir : str
        Directory to save the resulting plot.

    Returns
    -------
    None
    """
    data_for_violin = mse_violin['data_for_violin']
    all_states = mse_violin['all_states']
    labels = mse_violin['labels']

    colors = {0: 'royalblue', 1: 'teal', 2: 'purple', 3: 'forestgreen', 4: 'firebrick'}
    palette = [colors[s] for s in all_states]

    data_list = [data_for_violin[s] for s in all_states]

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

    for i, d in enumerate(data_list):
        x = np.random.normal(loc=i, scale=0.15, size=len(d))
        ax.scatter(x, d, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    for i, d in enumerate(data_list):
        d = np.array(d)
        d = d[~np.isnan(d)]
        if len(d) == 0:
            continue

        mean_val = np.nanmean(d)
        sem_val = np.nanstd(d) / np.sqrt(len(d))

        plt.plot(i, mean_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=4)
        plt.errorbar(i, mean_val, yerr=sem_val, color='black', capsize=4, elinewidth=1.5, markeredgewidth=1, zorder=3)

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
    Create violin plots for each index across sleep states (W, N1, N2, N3, REM).

    For each index:
        - x-axis: Sleep states (Wake, N1, N2, N3, REM)
        - y-axis: Per-night averages from all subjects/nights
        - Overlays mean ± SEM markers and jittered individual points
        - Saves the plot as SVG and the underlying data as NPZ

    Parameters
    ----------
    results : dict
        Nested dictionary of subjects -> nights -> indices -> per-state values
        e.g., results[subject][night][index][state] = value
    output_dir : str
        Directory to save plots and data.
    """
    os.makedirs(output_dir, exist_ok=True)

    index_keys = ["W", "N", "R", "1", "2", "3", "4"]
    state_names = ["w", "n1", "n2", "n3", "r"]
    state_labels = ["Wake", "N1", "N2", "N3", "REM"]

    colors = {
        "w": "royalblue",
        "n1": "teal",
        "n2": "purple",
        "n3": "forestgreen",
        "r": "firebrick"
    }
    palette = [colors[s] for s in state_names]

    for idx in index_keys:
        # Collect per-state data across subjects/nights
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

        data_list = [data_for_violin[sn] for sn in state_names]

        # --- Violin plot ---
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

        # Overlay jittered points
        for i, d in enumerate(data_list):
            x = np.random.normal(loc=i, scale=0.15, size=len(d))
            ax.scatter(x, d, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

        # Overlay mean ± SEM
        for i, d in enumerate(data_list):
            d = np.array(d)
            if len(d) == 0:
                continue
            mean_val = np.nanmean(d)
            sem_val = np.nanstd(d) / np.sqrt(len(d))

            plt.plot(i, mean_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=4)
            plt.errorbar(i, mean_val, yerr=sem_val, color='black',
                         capsize=4, elinewidth=1.5, markeredgewidth=1, zorder=3)

        # Labels and aesthetics
        ax.set_xticks(range(len(state_labels)))
        ax.set_xticklabels(state_labels)
        ax.set_xlabel("Sleep State")
        ax.set_ylabel(f"Average Value of Index {idx}")
        ax.set_title(f"Index {idx} per Sleep State (Mean ± SEM)")
        plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
        plt.tight_layout()

        # Save plot and data
        plot_file = os.path.join(output_dir, f"index_{idx}_violin_avg_sem.svg")
        plt.savefig(plot_file, format="svg")
        plt.show()
        print(f"Saved plot: {plot_file}")

        data_file = os.path.join(output_dir, f"index_{idx}_data.npz")
        np.savez(data_file, **data_for_violin)
        print(f"Saved data: {data_file}")

def plot_signal_violin_all(results, output_dir, signal_keys=["noise", "theta", "delta"]):
    """
    Create violin plots for each signal (noise, theta, delta) across sleep states.

    For each signal:
        - x-axis: Sleep states (Wake, N1, N2, N3, REM)
        - y-axis: Values across all subjects/nights
        - Overlays mean ± SEM markers and jittered individual points
        - Saves plot as SVG and underlying data as NPZ

    Parameters
    ----------
    results : dict
        Nested dictionary of subjects -> nights -> signals -> per-state values
        e.g., results[subject][night][signal][state] = value
    output_dir : str
        Directory to save plots and data.
    signal_keys : list of str, optional
        List of signal names to plot, default is ["noise", "theta", "delta"].
    """
    os.makedirs(output_dir, exist_ok=True)

    state_names = ["w", "n1", "n2", "n3", "r"]
    state_labels = ["Wake", "N1", "N2", "N3", "REM"]

    colors = {
        "w": "royalblue",
        "n1": "teal",
        "n2": "purple",
        "n3": "forestgreen",
        "r": "firebrick"
    }
    palette = [colors[s] for s in state_names]

    for key in signal_keys:
        # Collect per-state data
        data_for_violin = {sn: [] for sn in state_names}

        for subj, nights in results.items():
            for night, signal_dict in nights.items():
                if key not in signal_dict:
                    continue
                state_vals = signal_dict[key]
                for sn in state_names:
                    val = state_vals.get(sn, np.nan)
                    if not np.isnan(val):
                        data_for_violin[sn].append(val)

        data_list = [data_for_violin[sn] for sn in state_names]

        # --- Violin plot ---
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

        # Overlay mean ± SEM
        for i, d in enumerate(data_list):
            d = np.array(d)
            if len(d) == 0:
                continue
            mean_val = np.nanmean(d)
            sem_val = np.nanstd(d) / np.sqrt(len(d))

            plt.plot(i, mean_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=4)
            plt.errorbar(i, mean_val, yerr=sem_val, color='black', capsize=4,
                         elinewidth=1.5, markeredgewidth=1, zorder=3)

        # Labels and aesthetics
        ax.set_xticks(range(len(state_labels)))
        ax.set_xticklabels(state_labels)
        ax.set_xlabel("Sleep State")
        ax.set_ylabel(f"Average {key.capitalize()} Value")
        ax.set_title(f"{key.capitalize()} per Sleep State (Mean ± SEM)")
        plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)
        plt.tight_layout()

        # Save plot and data
        plot_file = os.path.join(output_dir, f"{key}_violin_avg_sem.svg")
        plt.savefig(plot_file, format="svg")
        plt.show()
        print(f"Saved plot: {plot_file}")

        data_file = os.path.join(output_dir, f"{key}_data.npz")
        np.savez(data_file, **data_for_violin)
        print(f"Saved data: {data_file}")

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

def prepare_mse_violin_data(valid_states, fs, length, lfp_PFC):
    """
    Prepare multi-scale entropy (MSE) data per sleep state for violin plotting.

    Steps:
    1. Segment LFP signal into overlapping windows.
    2. Compute MSE for each window.
    3. Smooth MSE using Savitzky-Golay filter.
    4. Normalize MSE values to [-1, 1].
    5. Align MSE values with corresponding sleep states.
    6. Build data structures for plotting violin plots per state.

    Parameters
    ----------
    valid_states : array-like
        Sleep stage indices for each time point, typically integers 0..4.
    fs : int
        Sampling frequency of the LFP signal.
    length : float
        Window length in seconds for MSE computation.
    lfp_PFC : array-like
        Raw LFP signal from prefrontal cortex.

    Returns
    -------
    df_plot : pd.DataFrame
        DataFrame with columns 'state' and 'mse', ready for seaborn plotting.
    data_for_violin : list of np.ndarray
        List containing arrays of MSE values per state (0..4), suitable for violin plots.
    all_states : list of int
        Sleep stage indices [0,1,2,3,4].
    labels : list of str
        Sleep stage labels ['W', 'N1', 'N2', 'N3', 'REM'].
    normalized_mse : np.ndarray
        Smoothed and normalized MSE values aligned to valid_states.
    """
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    all_states = list(range(5))  # indices for sleep stages 0..4

    # Initialize MSE object using EntropyHub
    Mobj = EH.MSobject('IncrEn', m=2, R=3, Norm=True)
    window_size = step_size = fs * length  # number of samples per window

    # Number of windows to process
    num_windows = (len(lfp_PFC) - window_size) // step_size + 1

    mse_values = []
    time_stamps_mse = []

    # Compute MSE for each window
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        segment = lfp_PFC[start:end]
        # Suppress verbose output from EntropyHub
        with suppress_stdout():
            MSx, _ = EH.MSEn(segment, Mobj, Scales=2, Methodx='modified')

        # Store mean MSE for the segment
        mse_values.append(np.mean(MSx))
        time_stamps_mse.append((start + end) / 2 / fs)  # timestamp in seconds

    mse_values = np.array(mse_values)
    time_stamps_mse = np.array(time_stamps_mse)

    # Smooth MSE using Savitzky-Golay filter
    window_length = 11 if len(mse_values) >= 11 else len(mse_values) | 1  # must be odd
    polyorder = 4
    smoothed_mse = savgol_filter(mse_values, window_length=window_length, polyorder=polyorder)

    # Normalize to [-1, 1]
    normalized_mse = 2 * ((smoothed_mse - min(smoothed_mse)) / (max(smoothed_mse) - min(smoothed_mse))) - 1

    # Ensure alignment with valid_states
    min_length = min(len(normalized_mse), len(valid_states))
    valid_states = valid_states[:min_length]
    normalized_mse = normalized_mse[:min_length]

    # Build DataFrame
    df = pd.DataFrame({'state': valid_states, 'mse': normalized_mse})

    # Prepare data per state for violin plotting
    data_for_violin = [df.loc[df['state'] == s, 'mse'].values for s in all_states]

    # Debug info: counts per state and unique states
    counts = [len(d) for d in data_for_violin]
    print("Counts per state (0..4):", counts)
    print("Unique states present:", sorted(df['state'].unique()))

    df_plot = df.copy()
    df_plot['state'] = df_plot['state'].astype(int)

    return df_plot, data_for_violin, all_states, labels, normalized_mse

def Index_1(delta, gamma, EMG):
    """
    Compute Index 1 per epoch: (EMG * gamma) / delta

    Parameters
    ----------
    delta, gamma, EMG : array-like
        Power or amplitude per epoch for each component.

    Returns
    -------
    index_1 : np.ndarray
        Computed Index 1 for each epoch.
    """
    index_1 = (EMG * gamma) / delta
    return index_1

def Index_2(delta, theta, sigma):
    """
    Compute Index 2 per epoch: (sigma * delta) / theta
    """
    index_2 = (sigma * delta) / theta
    return index_2

def Index_3(delta, theta, gamma):
    """
    Compute Index 3 per epoch: (theta * gamma) / delta
    """
    index_3 = (theta * gamma) / delta
    return index_3

def Index_4(delta, theta):
    """
    Compute Index 4 per epoch: delta / theta
    """
    index_4 = delta / theta
    return index_4




