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
from scipy.stats import sem
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, friedmanchisquare, shapiro, wilcoxon

from statsmodels.stats.multitest import multipletests
from itertools import combinations

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


def prepare_data_for_2way_anova(collected_means):
    """
    Convert collected_means into long-form DataFrame suitable for repeated-measures two-way ANOVA.

    Parameters
    ----------
    collected_means : dict
        Structure: collected_means[state][index] = list of subject values

    Returns
    -------
    df_long : pd.DataFrame
        Columns: 'subject', 'stage', 'index_type', 'value'
    """
    states = ['Wake', 'N1', 'N2', 'N3', 'REM']
    indices = ['w', 'r', 'n']

    # Determine number of subjects (assumes same number per state/index)
    n_subjects = min(len(collected_means[state][idx]) for state in states for idx in indices)

    data = []
    for subj in range(n_subjects):
        for idx in indices:
            for state in states:
                value = collected_means[state][idx][subj]
                data.append({
                    'subject': subj,
                    'stage': state,
                    'index_type': idx.upper(),
                    'value': value
                })

    df_long = pd.DataFrame(data)
    return df_long




def test_index_discrimination(collected_means):
    """
    Test whether W, R, N indices differ across sleep stages.
    Performs repeated measures ANOVA (parametric) or Friedman test (non-parametric)
    with post-hoc pairwise comparisons and multiple testing correction.
    Additionally tests:
      - W and R: target stage vs all others (per-bar)
      - N: NREM (N1+N2+N3) vs non-NREM, and intra-NREM (N3 vs N1+N2)

    Parameters
    ----------
    collected_means : dict
        Structure: collected_means[state][index] = list of subject values

    Returns
    -------
    results : dict
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import f_oneway, friedmanchisquare, shapiro, wilcoxon
    from itertools import combinations
    from statsmodels.stats.multitest import multipletests

    states = ['Wake', 'N1', 'N2', 'N3', 'REM']
    indices = ['w', 'r', 'n']
    results = {}

    for idx in indices:
        # Gather data per stage
        data_by_stage = [np.array(collected_means[state][idx]) for state in states]
        min_len = min(len(d) for d in data_by_stage)
        data_by_stage_equal = [d[:min_len] for d in data_by_stage]
        df = pd.DataFrame({state: data_by_stage_equal[i] for i, state in enumerate(states)})

        # Normality check
        normality = [shapiro(df[state])[1] > 0.05 for state in states]
        if all(normality):
            f_stat, p_val = f_oneway(*[df[state] for state in states])
            test_type = 'ANOVA'
        else:
            f_stat, p_val = friedmanchisquare(*[df[state] for state in states])
            test_type = 'Friedman'

        results[idx] = {'test': test_type, 'stat': f_stat, 'p_uncorrected': p_val}

        # Post-hoc pairwise comparisons
        pairwise_p = {}
        for s1, s2 in combinations(states, 2):
            try:
                stat, p = wilcoxon(df[s1], df[s2])
            except ValueError:
                p = np.nan
            pairwise_p[f'{s1} vs {s2}'] = p
        corrected = multipletests(list(pairwise_p.values()), method='bonferroni')[1]
        results[idx]['posthoc_corrected'] = {k: v for k, v in zip(pairwise_p.keys(), corrected)}

        # --- Target stage tests ---
        target_stage_test = {}

        if idx in ['w', 'r']:
            # W or R: target stage vs all others individually
            target_stage = 'Wake' if idx == 'w' else 'REM'
            stage_values = np.array(collected_means[target_stage][idx])
            other_stages = [s for s in states if s != target_stage]
            other_values = np.concatenate([np.array(collected_means[s][idx]) for s in other_stages])
            min_len_target = min(len(stage_values), len(other_values))
            try:
                stat, p_target = wilcoxon(stage_values[:min_len_target], other_values[:min_len_target])
            except ValueError:
                p_target = np.nan
            target_stage_test[target_stage] = {'p_raw': p_target, 'significant': p_target < 0.05 if not np.isnan(p_target) else False}

        elif idx == 'n':
            # N: NREM vs non-NREM
            nrem_stages = ['N1', 'N2', 'N3']
            non_nrem_stages = ['Wake', 'REM']
            nrem_values = np.concatenate([np.array(collected_means[s][idx]) for s in nrem_stages])
            non_nrem_values = np.concatenate([np.array(collected_means[s][idx]) for s in non_nrem_stages])
            min_len_target = min(len(nrem_values), len(non_nrem_values))
            try:
                stat, p_nrem = wilcoxon(nrem_values[:min_len_target], non_nrem_values[:min_len_target])
            except ValueError:
                p_nrem = np.nan
            target_stage_test['NREM'] = {'p_raw': p_nrem, 'significant': p_nrem < 0.05 if not np.isnan(p_nrem) else False}

            # Intra-NREM: N3 vs N1+N2
            n3_values = np.array(collected_means['N3'][idx])
            n1n2_values = np.concatenate([np.array(collected_means[s][idx]) for s in ['N1', 'N2']])
            min_len_intra = min(len(n3_values), len(n1n2_values))
            try:
                stat, p_n3 = wilcoxon(n3_values[:min_len_intra], n1n2_values[:min_len_intra])
            except ValueError:
                p_n3 = np.nan
            target_stage_test['N3_vs_N1N2'] = {'p_raw': p_n3, 'significant': p_n3 < 0.05 if not np.isnan(p_n3) else False}

        results[idx]['target_stage_test'] = target_stage_test

    return results

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


def plot_index_barplot(collected_means, output_dir):
    """
    Plot barplot of mean ± SEM Wei indices across sleep states.

    Parameters
    ----------
    collected_means : dict
        Structure:
        collected_means[state][index] = list of subject values
    output_dir : str
        Directory where the SVG is saved.
    """

    states = ['Wake', 'N1', 'N2', 'N3', 'REM']
    indices = ['w', 'r', 'n']  # must match your keys

    # Colors for paper-friendly plot
    colors = {
        'w': 'black',
        'r': 'red',
        'n': 'blue'
    }

    # Paper-quality figure
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 24,
        'axes.titlesize': 26,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 18
    })

    x = np.arange(len(states))
    bar_width = 0.25

    # Offsets for the 3 bars
    offsets = {
        'w': -bar_width,
        'r': 0,
        'n': bar_width
    }

    # Prepare a list to store values for saving
    saved_data = []

    # Plot each index as a bar group
    for idx in indices:
        means = [np.nanmean(collected_means[state][idx]) for state in states]
        sems = [sem(collected_means[state][idx], nan_policy='omit')
                if len(collected_means[state][idx]) > 1 else 0
                for state in states]

        # Append data for saving
        for state, mean_val, sem_val in zip(states, means, sems):
            saved_data.append({
                'State': state,
                'Index': idx,
                'Mean': mean_val,
                'SEM': sem_val
            })

        plt.bar(
            x + offsets[idx],
            means,
            width=bar_width,
            label=f"Index {idx.upper()}",
            color=colors[idx],
            edgecolor='black',
            linewidth=1.0
        )

        # Error bars (SEM)
        plt.errorbar(
            x + offsets[idx],
            means,
            yerr=sems,
            fmt='none',
            ecolor='dimgray',
            elinewidth=2,
            capsize=5
        )

    # Labels & title
    plt.xlabel('Sleep Stages')
    plt.ylabel('Mean Index Value ± SEM')
    plt.title('Wei Indices Across Sleep Stages')
    plt.ylim(0, 1)
    plt.xticks(x, states)
    plt.legend(frameon=True, loc='upper right')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save figure
    plt.savefig(f'{output_dir}/wei_indices_barplot.svg', format='svg')
    plt.close()

    # Save the data to CSV
    df = pd.DataFrame(saved_data)
    df.to_csv(os.path.join("D:","EEG_Data_stage","stat_plotting", "values", f'wei_indices_values.csv'), index=False)

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
    df = pd.DataFrame({'state': valid_states, 'aperiodic': wei_normalizing(normalized_exponents)})
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
    normalized_dfa = wei_normalizing(normalized_dfa[:min_length])
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

def wei_normalizing(data):
    """
    Perform robust normalization of a 1D array based on 10th and 90th percentiles.

    Parameters
    ----------
    data : array-like
        Input data to normalize.

    Returns
    -------
    normalized_data : np.ndarray
        Normalized array scaled between 0.05 and 1. Values below the 10th percentile
        are mapped to 0.05, and above the 90th percentile to 1.

    Notes
    -----
    This normalization reduces the effect of extreme outliers by ignoring
    values outside the 10th–90th percentile range.
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

def enlarge_ticks(ax, factor=1.5):
    """Scale tick label font sizes for x and y axes."""
    for tick in ax.get_xticklabels():
        tick.set_fontsize(tick.get_fontsize() * factor)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(tick.get_fontsize() * factor)


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

    # --- Overlay jittered points ---
    for i, d in enumerate(data_list):
        x = np.random.normal(loc=i, scale=0.15, size=len(d))
        ax.scatter(x, d, color=palette[i], alpha=0.5, marker='D', s=10, zorder=1)

    # --- Overlay mean ± SEM ---
    for i, d in enumerate(data_list):
        d = np.array(d)
        d = d[~np.isnan(d)]
        if len(d) == 0:
            continue
        mean_val = np.nanmean(d)
        sem_val = np.nanstd(d) / np.sqrt(len(d))
        plt.plot(i, mean_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=4)
        plt.errorbar(i, mean_val, yerr=sem_val, color='black', capsize=4, elinewidth=1.5, markeredgewidth=1, zorder=3)

    # --- Labels & ticks ---
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sleep State', fontsize=20)
    ax.set_ylabel('Normalized MSE exponent', fontsize=20)
    ax.set_title('MSE per Sleep State (Mean ± SEM)', fontsize=20)

    # Grid
    plt.grid(axis='y', color='lightgray', linestyle='--', alpha=0.6, zorder=0)

    # Enlarge ticks
    enlarge_ticks(ax, factor=1.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/mse_violin_avg_sem.svg", format="svg")
    plt.show()

from pathlib import Path

def save_violin_inputs_for_stats(
    aperiodic_violin,
    dfa_violin,
    mse_violin,
    output_dir,
    filename="sleep_metrics_longform.csv"
):
    records = []

    def unpack(metric_name, data_dict):
        df_plot = data_dict["df_plot"]  # retrieve subject and night info
        for _, row in df_plot.iterrows():
            value = row[metric_name.capitalize()] if metric_name != "mse" else row["MSE"]
            if pd.isna(value):
                continue
            records.append({
                "metric": metric_name,
                "sleep_state": row["State"],
                "value": float(value),
                "subject": row["Subject"],
                "night": row["Night"],
                "subject_night": f"{row['Subject']}_{row['Night']}"
            })

    unpack("aperiodic", aperiodic_violin)
    unpack("dfa", dfa_violin)
    unpack("mse", mse_violin)

    df = pd.DataFrame.from_records(records)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    df.to_csv(out_path, index=False)

    return df

def plot_averaged_sleep_boxplots(aperiodic_violin, dfa_violin, mse_violin, output_dir):
    """
    Plot group-level boxplots for Aperiodic, DFA, and MSE metrics
    across sleep stages, with overlaid mean ± SEM.
    Ensures true reordering: Wake, N1, N2, N3, REM.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    df_stats = save_violin_inputs_for_stats(
        aperiodic_violin,
        dfa_violin,
        mse_violin,
        output_dir=r"D:\EEG_Data_stage\stat_plotting\values"
    )
    # Professional style
    sns.set(style='whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'figure.dpi': 300
    })

    # Correct color mapping
    colors = {
        'W'  : '#76B7B2',   # teal
        'N1' : '#4E79A7',   # blue
        'N2' : '#59A14F',   # green
        'N3' : '#F28E2B',   # orange
        'REM': '#E15759'    # red
    }

    # Desired plotting order
    x_order_labels = ['W', 'N1', 'N2', 'N3', 'REM']

    # Setup figure with 3 horizontal subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)

    plot_data = [
        (aperiodic_violin, 'Normalized aperiodic exponent', 'Aperiodic Fit per Sleep State'),
        (dfa_violin, 'Normalized DFA exponent', 'DFA per Sleep State'),
        (mse_violin, 'Normalized MSE exponent', 'MSE per Sleep State')
    ]

    for ax, (data_dict, ylabel, title) in zip(axes, plot_data):
        dfv = data_dict['data_for_violin']
        labels = data_dict['labels']

        # -----------------------
        # DEBUG PRINT
        # -----------------------
        print("\n================ DEBUG ================")
        print("Title:", title)
        print("Labels (data_dict['labels']):", labels)
        print("Type of data_for_violin:", type(dfv))
        print("Length of data_for_violin:", len(dfv))
        for i, arr in enumerate(dfv):
            print(f"  Index {i}: label={labels[i]}, n={len(arr)}, preview={arr[:5]}")
        print("=======================================\n")

        # -----------------------
        # Map label -> data and reorder
        # -----------------------
        dfv_map = {labels[i]: dfv[i] for i in range(len(labels))}
        data_list = [dfv_map[label] for label in x_order_labels]
        palette = [colors[label] for label in x_order_labels]

        # -----------------------
        # Boxplot
        # -----------------------
        bp = ax.boxplot(
            data_list,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2),
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(color='black', linewidth=1.5),
            capprops=dict(color='black', linewidth=1.5)
        )

        # Color boxes
        for patch, c in zip(bp['boxes'], palette):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        # Overlay mean ± SEM
        for i, d in enumerate(data_list):
            d = np.array(d)
            d = d[~np.isnan(d)]
            if len(d) == 0:
                continue
            mean_val = np.mean(d)
            sem_val = np.std(d)/np.sqrt(len(d))
            x_pos = i + 1.08  # slight offset for visibility
            ax.plot(x_pos, mean_val, 'o', color='white', markeredgecolor='black', markersize=10, zorder=4)
            ax.errorbar(x_pos, mean_val, yerr=sem_val, color='black', capsize=6, elinewidth=2.5, zorder=3)

        # Overlay jittered points
        for i, d in enumerate(data_list):
            x = np.random.normal(loc=i+1, scale=0.07, size=len(d))
            ax.scatter(x, d, color='black', alpha=0.7, s=15, zorder=2, marker='D')

        # Axis formatting
        ax.set_xticks(range(1, len(x_order_labels)+1))
        ax.set_xticklabels(x_order_labels, fontsize=22)
        ax.set_xlabel('Sleep State', fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.set_ylim(0,1)
        ax.grid(axis='y', linestyle='--', color='gray', alpha=0.3, zorder=0)
        ax.tick_params(axis='y', labelsize=22)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sleep_boxplots_reordered_correctcolors.svg", format="svg")
    plt.show()

def plot_subject_pca(loaded_dicts, output_dir):
    """
    Generate PCA plots:
        - Per subject/night
        - Global PCA across all subjects

    Features used:
        - index_vals (all keys)
        - aperiodic_fit
        - dfa
        - mse
        - noise
        - theta
        - delta

    Points colored by sleep state.

    Parameters
    ----------
    loaded_dicts : list of dict
        Each dict contains features and epoch-level data for a subject/night.
    output_dir : str
        Directory to save PCA plots.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    sns.set(style='whitegrid')

    # ------------------------
    # State colors
    # ------------------------
    state_colors = {
        0: '#76B7B2',  # W → teal
        1: '#4E79A7',  # N1 → blue
        2: '#59A14F',  # N2 → green
        3: '#F28E2B',  # N3 → orange
        4: '#E15759'   # REM → red
    }
    state_labels = ['W', 'N1', 'N2', 'N3', 'REM']

    # ------------------------
    # Store global data
    # ------------------------
    all_X = []
    all_states = []

    # ------------------------
    # PCA per subject
    # ------------------------
    for subj_dict in loaded_dicts:
        subject = subj_dict['subject']
        night = subj_dict['night']

        # Collect features per epoch
        feature_list = []

        # index_vals
        for key, arr in subj_dict['index_vals'].items():
            feature_list.append(np.array(arr).flatten())

        # Other features
        for feat_key in ['aperiodic_fit', 'dfa', 'mse', 'noise', 'theta', 'delta']:
            feature_list.append(np.array(subj_dict[feat_key]).flatten())

        # Stack features: shape (n_epochs, n_features)
        X = np.column_stack(feature_list)
        states = subj_dict["dfa_violin"]["df_plot"].iloc[:,0].to_numpy()

        # Add to global
        all_X.append(X)
        all_states.append(states)

        # Run PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Plot PCA per subject
        plt.figure(figsize=(8,6))
        for state in np.unique(states):
            mask = states == state
            plt.scatter(
                X_pca[mask,0], X_pca[mask,1],
                color=state_colors[state],
                alpha=0.7,
                s=25,
                label=state_labels[state]
            )
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=24)
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=24)
        plt.legend(title="Sleep State", fontsize=18)
        plt.grid(alpha=0.3)
        plt.tick_params(axis='x', labelsize=22)
        plt.tick_params(axis='y', labelsize=22)
        plt.tight_layout()

        out_path = os.path.join(output_dir, 'PCA', f"PCA_Subject{subject}_Night{night}.svg")
        plt.savefig(out_path, format="svg", dpi=300)
        plt.close()
        print(f"Saved PCA plot: {out_path}")

    # ------------------------
    # Global PCA across all subjects
    # ------------------------
    all_X_combined = np.vstack(all_X)
    all_states_combined = np.concatenate(all_states)

    pca_global = PCA(n_components=2)
    X_global_pca = pca_global.fit_transform(all_X_combined)

    plt.figure(figsize=(10,8))
    for state in np.unique(all_states_combined):
        mask = all_states_combined == state
        plt.scatter(
            X_global_pca[mask,0], X_global_pca[mask,1],
            color=state_colors[state],
            alpha=0.5,
            s=20,
            label=state_labels[state]
        )
    plt.xlabel(f"PC1 ({pca_global.explained_variance_ratio_[0]*100:.1f}%)", fontsize=24)
    plt.ylabel(f"PC2 ({pca_global.explained_variance_ratio_[1]*100:.1f}%)", fontsize=24)
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.legend(title="Sleep State", fontsize=18)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path_global = os.path.join(output_dir, 'PCA', "PCA_Global_AllSubjects.svg")
    plt.savefig(out_path_global, format="svg", dpi=300)
    plt.close()
    print(f"Saved global PCA plot: {out_path_global}")

def plot_subject_pca_rodent(loaded_dicts, output_dir):
    """
    Generate PCA plots:
        - Per subject/night
        - Global PCA across all subjects

    Features used:
        - index_vals (all keys)
        - aperiodic_fit
        - dfa
        - mse
        - noise
        - theta
        - delta

    Points colored by sleep state.

    Parameters
    ----------
    loaded_dicts : list of dict
        Each dict contains features and epoch-level data for a subject/night.
    output_dir : str
        Directory to save PCA plots.
    """


    os.makedirs(output_dir, exist_ok=True)
    sns.set(style='whitegrid')

    # ------------------------
    # State colors
    # ------------------------
    state_colors = {
        0: '#76B7B2',  # W → teal
        1: '#4E79A7',  # N1 → blue
        2: '#59A14F',  # N2 → green
        3: '#F28E2B',  # N3 → orange
        4: '#E15759'   # REM → red
    }
    state_labels = ['W', 'N1', 'N2', 'N3', 'REM']

    # ------------------------
    # Store global data
    # ------------------------
    all_X = []
    all_states = []

    # ------------------------
    # PCA per subject
    # ------------------------
    for subj_dict in loaded_dicts:
        subject = subj_dict['subject']
        night = subj_dict['night']

        # Collect features per epoch
        feature_list = []

        # index_vals
        for key, arr in subj_dict['index_vals'].items():
            feature_list.append(np.array(arr).flatten())

        # Other features
        for feat_key in ['noise', 'theta', 'delta']:
            feature_list.append(np.array(subj_dict[feat_key]).flatten())

        # Stack features: shape (n_epochs, n_features)
        X = np.column_stack(feature_list)
        states = subj_dict['states']

        # Add to global
        all_X.append(X)
        all_states.append(states)

        # Run PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Plot PCA per subject
        plt.figure(figsize=(8,6))
        for state in np.unique(states):
            mask = states == state
            plt.scatter(
                X_pca[mask,0], X_pca[mask,1],
                color=state_colors[state],
                alpha=0.7,
                s=25,
                label=state_labels[state]
            )
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=24)
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=24)
        plt.tick_params(axis='x', labelsize=22)
        plt.tick_params(axis='y', labelsize=22)

        plt.legend(title="Sleep State", fontsize=18)
        plt.grid(alpha=0.3)
        plt.ylim(-1.5, 1.5)
        plt.xlim(-1.5, 1.5)
        plt.tight_layout()

        out_path = os.path.join(output_dir, 'PCA', f"PCA_Subject{subject}_Night{night}.svg")
        plt.savefig(out_path, format="svg", dpi=300)
        plt.close()
        print(f"Saved PCA plot: {out_path}")

    # ------------------------
    # Global PCA across all subjects
    # ------------------------
    all_X_combined = np.vstack(all_X)
    all_states_combined = np.concatenate(all_states)

    pca_global = PCA(n_components=2)
    X_global_pca = pca_global.fit_transform(all_X_combined)

    plt.figure(figsize=(10,8))
    for state in np.unique(all_states_combined):
        mask = all_states_combined == state
        plt.scatter(
            X_global_pca[mask,0], X_global_pca[mask,1],
            color=state_colors[state],
            alpha=0.5,
            s=20,
            label=state_labels[state]
        )
    plt.xlabel(f"PC1 ({pca_global.explained_variance_ratio_[0]*100:.1f}%)", fontsize=24)
    plt.ylabel(f"PC2 ({pca_global.explained_variance_ratio_[1]*100:.1f}%)", fontsize=24)
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.legend(title="Sleep State", fontsize=18)
    plt.grid(alpha=0.3)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.tight_layout()

    out_path_global = os.path.join(output_dir, 'PCA', "PCA_Global_AllSubjects.svg")
    plt.savefig(out_path_global, format="svg", dpi=300)
    plt.close()
    print(f"Saved global PCA plot: {out_path_global}")

def plot_index_avg_boxplot_combined(results, output_dir,
                                    index_keys_to_plot=["W", "N", "R"]):
    """
    Create a single figure containing professional-quality boxplots
    for selected indices across sleep states (W, N1, N2, N3, REM).

    Matching style of: plot_averaged_sleep_boxplots()

    - One subplot per index (e.g., W, N, R)
    - Jittered points
    - Mean ± SEM overlay
    - Publication-ready styling
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Professional styling settings
    # -----------------------------
    sns.set(style='whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'figure.dpi': 300
    })

    # Sleep states
    state_names = ["w", "n1", "n2", "n3", "r"]
    state_labels = ["W", "N1", "N2", "N3", "REM"]

    # Updated color palette (match plot_averaged_sleep_boxplots)
    state_colors = {
        "w": "#76B7B2",    # teal
        "n1": "#4E79A7",   # blue
        "n2": "#59A14F",   # green
        "n3": "#F28E2B",   # orange
        "r": "#E15759"     # red
    }
    palette = [state_colors[s] for s in state_names]

    n_plots = len(index_keys_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 7), sharey=True)
    if n_plots == 1:
        axes = [axes]

    # -----------------------------
    # Loop over selected indices
    # -----------------------------
    for ax, idx in zip(axes, index_keys_to_plot):

        # Collect per-state data
        data_for_box = {sn: [] for sn in state_names}

        for subj, nights in results.items():
            for night, idx_dict in nights.items():
                if idx not in idx_dict:
                    continue
                state_vals = idx_dict[idx]
                for sn in state_names:
                    v = state_vals.get(sn, np.nan)
                    if not np.isnan(v):
                        data_for_box[sn].append(v)

        data_list = [data_for_box[sn] for sn in state_names]

        # ---------------------------------
        # Boxplot (professional style)
        # ---------------------------------
        bp = ax.boxplot(
            data_list,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2),
            boxprops=dict(linewidth=1.6),
            whiskerprops=dict(color='black', linewidth=1.6),
            capprops=dict(color='black', linewidth=1.6)
        )

        # Fill boxes with updated palette
        for patch, col in zip(bp["boxes"], palette):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)

        # ---------------------------------
        # Jittered points + Mean ± SEM
        # ---------------------------------
        for i, vals in enumerate(data_list):
            vals = np.array(vals)
            if len(vals) == 0:
                continue

            # Jitter
            jitter = np.random.normal(i + 1, 0.07, len(vals))
            ax.scatter(jitter, vals, color='black', s=20,
                       alpha=0.7, marker='D', zorder=2)

            # Mean ± SEM
            m = np.mean(vals)
            s = np.std(vals) / np.sqrt(len(vals))
            x_pos = i + 1.08
            ax.plot(x_pos, m, 'o', color='white',
                    markeredgecolor='black', markersize=10, zorder=4)
            ax.errorbar(x_pos, m, yerr=s, color='black',
                        capsize=6, elinewidth=2.5, zorder=3)

        # Aesthetics
        ax.set_xticks(range(1, len(state_labels)+1))
        ax.set_xticklabels(state_labels, fontsize=22)
        ax.tick_params(axis='y', labelsize=22)
        ax.set_xlabel("Sleep State", fontsize=24)
        ax.set_title(f"Index {idx}", fontsize=26)
        ax.grid(axis='y', linestyle='--', color='gray', alpha=0.35)
        ax.set_axisbelow(True)
        ax.set_ylim(0,1)

        # Save per-index NPZ
        np.savez(os.path.join(output_dir, f"index_{idx}_data.npz"),
                 **data_for_box)

    axes[0].set_ylabel("Value", fontsize=24)

    # Final layout + save
    plt.tight_layout()
    out_path = os.path.join(output_dir, "index_avg_boxplots_combined.svg")
    plt.savefig(out_path, format="svg")
    plt.show()

    print(f"Saved combined figure: {out_path}")


def plot_index_avg_violin_WNR(means, output_dir, letters=('A', 'B', 'C'), show=False):
    """
        Create a single figure with W, N, R indices as horizontal subplots.
        Each subplot shows violin plots of the index across sleep states
        with mean ± SEM and jittered individual points.

        Parameters
        ----------
        means : dict
            Nested dictionary: means[state][index] = array-like values across subjects
            States: 'Wake', 'N1', 'N2', 'N3', 'REM'
            Indices: 'w', 'n', 'r'
        output_dir : str
            Directory to save the figure.
        letters : tuple
            Letters to label subplots for reference (A, B, C).
        show : bool
            Whether to display the plot interactively.
        """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(output_dir, exist_ok=True)

    states = ['Wake', 'N1', 'N2', 'N3', 'REM']
    index_keys = ['w', 'n', 'r']
    index_labels = ['W', 'N', 'R']
    colors = ['royalblue', 'forestgreen', 'firebrick']

    n_states = len(states)
    fontsize_labels = 16
    fontsize_title = 18
    fontsize_ticks = 14

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for i, (idx, ax) in enumerate(zip(index_keys, axes)):
        # Collect data per state
        data_list = [means[state][idx] for state in states]

        # --- Violin plot ---
        parts = ax.violinplot(
            data_list,
            positions=np.arange(n_states),
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        # Color the violins
        for pc, color in zip(parts['bodies'], [colors[i]] * n_states):
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)

        # Overlay jittered points + mean ± SEM
        for j, vals in enumerate(data_list):
            vals = np.array(vals)
            if len(vals) == 0:
                continue
            mean_val = np.nanmean(vals)
            sem_val = np.nanstd(vals) / np.sqrt(len(vals))

            # Jittered points
            x_jit = np.random.normal(loc=j, scale=0.08, size=len(vals))
            ax.scatter(x_jit, vals, color=colors[i], alpha=0.5, s=12, zorder=1)

            # Mean marker
            ax.plot(j, mean_val, 'o', color='white', markeredgecolor='black', markersize=6, zorder=4)
            # SEM
            ax.errorbar(j, mean_val, yerr=sem_val, color='black',
                        capsize=4, elinewidth=1.8, markeredgewidth=1.2, zorder=3)

        # Labels and aesthetics
        ax.set_xticks(np.arange(n_states))
        ax.set_xticklabels(states, fontsize=fontsize_ticks)
        ax.set_xlabel("Sleep State", fontsize=fontsize_labels)
        if i == 0:
            ax.set_ylabel(f"Index Value", fontsize=fontsize_labels)
        ax.set_title(f"Index {index_labels[i]}", fontsize=fontsize_title)
        ax.grid(axis='y', color='lightgray', linestyle='--', alpha=0.5, zorder=0)

        # Subplot letter
        ax.text(0.01, 0.95, letters[i], transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='top')

        # Optional: set y-limits if indices are normalized between 0-1
        ax.set_ylim(0, 1)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "indices_violin_combined.svg")
    plt.savefig(save_path, format='svg', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Saved combined violin plot: {save_path}")

import os
from pathlib import Path

def save_signal_boxplot_inputs_for_stats(results, output_dir, signal_keys=["noise", "theta", "delta"],
                                         filename="sleep_metrics_longform_signals.csv"):
    """
    Save underlying data used for boxplots to a long-form CSV suitable for stats in R.

    Parameters
    ----------
    results : dict
        Nested dict: results[subj][night][signal][state] = value
    output_dir : str or Path
        Directory to save CSV
    signal_keys : list
        List of signals to extract
    filename : str
        CSV filename
    """
    records = []
    state_map = {"w": "W", "n1": "N1", "n2": "N2", "n3": "N3", "r": "REM"}

    for subj, nights in results.items():
        for night, signal_dict in nights.items():
            for signal_key in signal_keys:
                if signal_key not in signal_dict:
                    continue
                state_vals = signal_dict[signal_key]
                for state_raw, value in state_vals.items():
                    if value is None or np.isnan(value):
                        continue
                    state_std = state_map.get(state_raw.lower())
                    if state_std is None:
                        continue
                    records.append({
                        "metric": signal_key,
                        "sleep_state": state_std,
                        "value": float(value),
                        "subject": subj,
                        "night": night,
                        "subject_night": f"{subj}_{night}"
                    })

    df = pd.DataFrame.from_records(records)

    output_dir = r"D:\EEG_Data_stage\stat_plotting\values"
    # output_dir.mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, index=False)

    print(f"Saved long-form CSV for stats: {out_path}")
    return df


def plot_signal_boxplot_all_combined(results, output_dir,
                                     signal_keys=["noise", "theta", "delta"],
                                     letters=("A", "B", "C"),
                                     show=False):
    """
    Create a single figure containing boxplots for multiple signals
    (e.g., noise, theta, delta), with one subplot per signal.

    Styles match `plot_averaged_sleep_boxplots`:
        - Reordered states: Wake, N1, N2, N3, REM
        - Custom colors per state
        - Overlay mean ± SEM
        - Jittered individual points
        - Publication-ready sizing and fonts

    Parameters
    ----------
    results : dict
        Nested dictionary: results[subj][night][signal][state] = value
    output_dir : str
        Directory where the figure and NPZ files will be saved.
    signal_keys : list
        List of signals to plot as subplots.
    letters : tuple
        Subplot labels (e.g., A, B, C).
    show : bool
        Whether to display the figure interactively.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    df_stats = save_signal_boxplot_inputs_for_stats(results, output_dir)
    sns.set(style='whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'figure.dpi': 300
    })

    # -----------------------
    # Sleep state order & colors
    # -----------------------
    state_map = {"w": "W", "n1": "N1", "n2": "N2", "n3": "N3", "r": "REM"}
    x_order_labels = ["W", "N1", "N2", "N3", "REM"]
    state_colors = {
        "W": "#76B7B2",    # teal
        "N1": "#4E79A7",   # blue
        "N2": "#59A14F",   # green
        "N3": "#F28E2B",   # orange
        "REM": "#E15759"   # red
    }

    n_signals = len(signal_keys)
    fig, axes = plt.subplots(1, n_signals, figsize=(7*n_signals, 6), sharey=True)
    if n_signals == 1:
        axes = [axes]

    # -----------------------
    # Fonts
    # -----------------------
    fontsize_labels = 24
    fontsize_title = 26
    fontsize_ticks = 22

    for ax_i, (signal_key, ax) in enumerate(zip(signal_keys, axes)):

        # ---- Collect per-state data ----
        data_for_boxplot = {lbl: [] for lbl in x_order_labels}

        for subj, nights in results.items():
            for night, signal_dict in nights.items():
                if signal_key not in signal_dict:
                    continue
                state_vals = signal_dict[signal_key]
                for sn, val in state_vals.items():
                    if val is None or np.isnan(val):
                        continue
                    std_label = state_map.get(sn.lower())
                    if std_label is not None:
                        data_for_boxplot[std_label].append(val)

        # Convert to list-of-lists
        data_list = [data_for_boxplot[label] for label in x_order_labels]
        palette = [state_colors[label] for label in x_order_labels]

        # ---- Boxplot ----
        bp = ax.boxplot(
            data_list,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2),
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(color='black', linewidth=1.5),
            capprops=dict(color='black', linewidth=1.5)
        )

        # Color boxes
        for patch, c in zip(bp['boxes'], palette):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        # ---- Overlay mean ± SEM and jittered points ----
        for i, vals in enumerate(data_list):
            vals = np.array(vals)
            if len(vals) == 0:
                continue

            # Jittered individual points
            x_jitter = np.random.normal(i + 1, 0.07, size=len(vals))
            ax.scatter(x_jitter, vals, color='black', alpha=0.7, s=15, zorder=2, marker='D')

            # Mean ± SEM
            mean_val = np.mean(vals)
            sem_val = np.std(vals)/np.sqrt(len(vals))
            x_pos = i + 1.08
            ax.plot(x_pos, mean_val, 'o', color='white', markeredgecolor='black', markersize=10, zorder=4)
            ax.errorbar(x_pos, mean_val, yerr=sem_val, color='black', capsize=6, elinewidth=2.5, zorder=3)

        # ---- Axis formatting ----
        ax.set_xticks(range(1, len(x_order_labels)+1))
        ax.set_xticklabels(x_order_labels, fontsize=fontsize_ticks)
        ax.set_xlabel("Sleep State", fontsize=fontsize_labels)
        ax.set_title(signal_key.capitalize(), fontsize=fontsize_title)
        ax.tick_params(axis='y', labelsize=22)
        ax.grid(axis="y", linestyle="--", color="gray", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 1)

        # Save underlying data
        np.savez(os.path.join(output_dir, f"{signal_key}_data.npz"), **data_for_boxplot)

    axes[0].set_ylabel("Value", fontsize=fontsize_labels)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "signals_boxplot_combined.svg")
    plt.savefig(fig_path, format="svg", dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved combined figure: {fig_path}")


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
    normalized_mse = wei_normalizing(normalized_mse[:min_length])

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




