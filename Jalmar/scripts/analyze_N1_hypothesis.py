"""
Analyze N1 hypothesis: N1 is more related to Wake and/or REM than to other NREM stages.

This script creates visualizations to test whether N1 latent activations are more similar to
Wake/REM than to N2/N3, which would support the hypothesis that N1 is a transitional stage
between sleep and wakefulness.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler
import scipy.io as sio

def load_h5_labels(h5_path):
    """Load sleep stage labels from HDF5 file.
    
    The H5 file has hierarchical structure: subjects (SC400, SC401, ...) > features/scores.
    Concatenates scores from all subjects in sorted order to match NPZ row ordering.
    """
    import h5py
    labels_list = []
    with h5py.File(h5_path, 'r') as f:
        for subject_key in sorted(f.keys()):
            subject = f[subject_key]
            if 'scores' in subject:
                scores = subject['scores'][()]
                labels_list.append(scores)
    
    if not labels_list:
        raise ValueError(f"No scores found in H5 file")
    
    return np.concatenate(labels_list)

def load_mcrbm_activations(analysis_dir):
    """Load mcRBM activations and state info."""
    activations_file = analysis_dir / 'activations.npz'
    state_info_file = analysis_dir / 'state_info.npy'
    
    activations = np.load(activations_file)
    p_all = activations['p_all']  # Shape: (n_samples, n_hidden_units)
    
    state_info = np.load(state_info_file)  # Shape: (n_states, 27)
    
    return p_all, state_info

def compute_stage_centroids(p_all, stage_labels):
    """Compute centroid of activations for each sleep stage."""
    stage_names = {0: 'Awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Movement', 6: 'Unknown'}
    centroids = {}
    stage_sizes = {}
    
    for stage_id in np.unique(stage_labels):
        if stage_id >= 0:  # Skip invalid labels
            mask = stage_labels == stage_id
            centroids[stage_id] = p_all[mask].mean(axis=0)
            stage_sizes[stage_id] = np.sum(mask)
    
    return centroids, stage_sizes

def compute_pairwise_distances(centroids):
    """Compute distances between stage centroids."""
    stage_ids = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[sid] for sid in stage_ids])
    
    # Compute pairwise Euclidean distances
    distances = squareform(pdist(centroid_matrix, metric='euclidean'))
    
    return pd.DataFrame(distances, index=stage_ids, columns=stage_ids)

def compute_stage_similarities(p_all, stage_labels, metric='cosine'):
    """
    Compute mean similarity between N1 and other stages.
    
    Computes the mean pairwise distance from N1 samples to samples of each other stage.
    """
    stage_names = {0: 'Awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Movement'}
    
    # Get N1 samples
    n1_mask = stage_labels == 1
    if not np.any(n1_mask):
        print("Warning: No N1 samples found!")
        return None
    
    n1_samples = p_all[n1_mask]
    
    similarities = {}
    for stage_id in np.unique(stage_labels):
        if stage_id < 0:
            continue
        stage_mask = stage_labels == stage_id
        stage_samples = p_all[stage_mask]
        
        # Compute mean distance from N1 to this stage
        distances = cdist(n1_samples, stage_samples, metric=metric)
        mean_dist = distances.mean()
        similarities[stage_id] = mean_dist
    
    return similarities


def compute_full_n1_distance_tables(p_all, stage_labels, metric='euclidean'):
    """Compute complete N1-to-all-stage distance tables."""
    stage_ids = sorted(np.unique(stage_labels).tolist())
    stage_names = {0: 'Awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Movement', 6: 'Unknown'}

    centroids = {stage_id: p_all[stage_labels == stage_id].mean(axis=0) for stage_id in stage_ids}
    n1_samples = p_all[stage_labels == 1]

    centroid_distances = {}
    mean_sample_distances = {}
    for stage_id in stage_ids:
        centroid_distances[stage_id] = float(np.linalg.norm(centroids[1] - centroids[stage_id]))
        stage_samples = p_all[stage_labels == stage_id]
        mean_sample_distances[stage_id] = float(cdist(n1_samples, stage_samples, metric=metric).mean())

    return stage_ids, stage_names, centroid_distances, mean_sample_distances


def save_n1_distance_tables(output_dir, stage_ids, stage_names, centroid_distances, mean_sample_distances):
    """Save N1 distance tables to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    centroid_rows = []
    mean_rows = []
    for stage_id in stage_ids:
        centroid_rows.append({
            'stage_id': int(stage_id),
            'stage_name': stage_names.get(stage_id, f'Stage{stage_id}'),
            'n1_centroid_distance': centroid_distances[stage_id],
        })
        mean_rows.append({
            'stage_id': int(stage_id),
            'stage_name': stage_names.get(stage_id, f'Stage{stage_id}'),
            'n1_mean_sample_distance': mean_sample_distances[stage_id],
        })

    pd.DataFrame(centroid_rows).to_csv(output_dir / 'n1_centroid_distances.csv', index=False)
    pd.DataFrame(mean_rows).to_csv(output_dir / 'n1_mean_sample_distances.csv', index=False)


def run_n1_analysis(p_all, stage_labels, output_dir):
    """Run the N1 hypothesis analysis and save the outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ANALYSIS 1: Stage Centroid Distances")
    print("="*70)
    centroids, stage_sizes = compute_stage_centroids(p_all, stage_labels)
    dist_df = compute_pairwise_distances(centroids)
    print(dist_df.to_string())

    print("\n" + "="*70)
    print("ANALYSIS 2: N1 Similarity to Other Stages")
    print("="*70)
    similarities = compute_stage_similarities(p_all, stage_labels)
    for stage_id in sorted(similarities.keys()):
        stage_name = {0: 'Awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Movement', 6: 'Unknown'}.get(stage_id, f'Stage{stage_id}')
        print(f"  N1 → {stage_name:15s}: {similarities[stage_id]:.4f}")

    n2_n3_dist = np.mean([similarities[2], similarities[3]])
    wake_rem_dist = np.mean([similarities[0], similarities[4]])
    print(f"\nMean distance N1 to N2/N3: {n2_n3_dist:.4f}")
    print(f"Mean distance N1 to Awake/REM: {wake_rem_dist:.4f}")
    if wake_rem_dist < n2_n3_dist:
        print(f"✓ HYPOTHESIS SUPPORTED: N1 is {100*(n2_n3_dist - wake_rem_dist)/n2_n3_dist:.1f}% closer to Wake/REM than to N2/N3")
    else:
        print(f"✗ Hypothesis not supported: N1 is {100*(wake_rem_dist - n2_n3_dist)/wake_rem_dist:.1f}% farther from Wake/REM")

    print("\n" + "="*70)
    print("ANALYSIS 3: Full N1 Distance Tables")
    print("="*70)
    stage_ids, stage_names, centroid_distances, mean_sample_distances = compute_full_n1_distance_tables(p_all, stage_labels)
    print("Centroid distances from N1:")
    for stage_id in stage_ids:
        print(f"  N1 -> {stage_names.get(stage_id, f'Stage{stage_id}'):15s}: {centroid_distances[stage_id]:.6f}")
    print("\nMean sample-wise distances from N1:")
    for stage_id in stage_ids:
        print(f"  N1 -> {stage_names.get(stage_id, f'Stage{stage_id}'):15s}: {mean_sample_distances[stage_id]:.6f}")

    save_n1_distance_tables(output_dir, stage_ids, stage_names, centroid_distances, mean_sample_distances)
    print(f"\nSaved N1 distance tables to: {output_dir}")

    print("\nGenerating visualizations...")
    create_stage_distance_heatmap(p_all, stage_labels, output_dir / 'stage_distance_heatmap.png')
    print("  ✓ stage_distance_heatmap.png")
    create_stage_correlation_heatmap(p_all, stage_labels, output_dir / 'stage_correlation_heatmap.png')
    print("  ✓ stage_correlation_heatmap.png")
    create_n1_similarity_barplot(p_all, stage_labels, output_dir / 'n1_similarity_barplot.png')
    print("  ✓ n1_similarity_barplot.png")
    create_activation_distribution_violin(p_all, stage_labels, output_dir / 'activation_distributions.png')
    print("  ✓ activation_distributions.png")

    return {
        'centroid_distances': centroid_distances,
        'mean_sample_distances': mean_sample_distances,
        'stage_ids': stage_ids,
    }

def create_stage_correlation_heatmap(p_all, stage_labels, output_file):
    """Create heatmap of correlations between stage activations."""
    stage_names = {0: 'Awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Movement'}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute centroid correlations
    centroids, stage_sizes = compute_stage_centroids(p_all, stage_labels)
    
    stage_ids = sorted([s for s in centroids.keys() if s >= 0])
    centroid_matrix = np.array([centroids[sid] for sid in stage_ids])
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(centroid_matrix)
    
    labels = [stage_names.get(sid, f'Stage{sid}') for sid in stage_ids]
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=labels, yticklabels=labels, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation of mcRBM Latent Activations by Sleep Stage')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return corr_matrix, stage_ids

def create_stage_distance_heatmap(p_all, stage_labels, output_file):
    """Create heatmap of distances between stage centroids."""
    stage_names = {0: 'Awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Movement'}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    centroids, stage_sizes = compute_stage_centroids(p_all, stage_labels)
    stage_ids = sorted([s for s in centroids.keys() if s >= 0])
    
    centroid_matrix = np.array([centroids[sid] for sid in stage_ids])
    dist_matrix = squareform(pdist(centroid_matrix, metric='euclidean'))
    
    labels = [stage_names.get(sid, f'Stage{sid}') for sid in stage_ids]
    sns.heatmap(dist_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('Euclidean Distance of mcRBM Latent Centroids by Sleep Stage')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return dist_matrix, stage_ids

def create_n1_similarity_barplot(p_all, stage_labels, output_file):
    """
    Create bar plot showing mean distance from N1 to each other stage.
    Closer = more similar.
    """
    stage_names = {0: 'Awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Movement'}
    
    similarities = compute_stage_similarities(p_all, stage_labels, metric='euclidean')
    
    if similarities is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by similarity (ascending = closer/more similar)
    stage_ids = sorted(similarities.keys())
    distances = [similarities[sid] for sid in stage_ids]
    labels = [stage_names.get(sid, f'Stage{sid}') for sid in stage_ids]
    
    colors = ['#ff7f0e' if sid == 1 else '#2ca02c' if sid in [2, 3] else '#d62728' if sid == 4 else '#1f77b4' 
              for sid in stage_ids]
    
    ax.bar(labels, distances, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Euclidean Distance from N1')
    ax.set_xlabel('Sleep Stage')
    ax.set_title('N1 Similarity to Other Stages (Lower = More Similar)')
    ax.axhline(y=np.mean(distances), color='k', linestyle='--', alpha=0.5, label='Mean')
    
    # Highlight hypothesis prediction
    n2_n3_dist = np.mean([distances[stage_ids.index(2)], distances[stage_ids.index(3)]])
    wake_rem_dist = np.mean([distances[stage_ids.index(0)], distances[stage_ids.index(4)]])
    
    textstr = f'N1-to-N2/N3 mean: {n2_n3_dist:.3f}\nN1-to-Awake/REM mean: {wake_rem_dist:.3f}'
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return similarities

def create_activation_distribution_violin(p_all, stage_labels, output_file):
    """
    Create violin plots of the first few principal components of latent activations
    for each sleep stage.
    """
    from sklearn.decomposition import PCA
    
    stage_names = {0: 'Awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Movement'}
    
    # Apply PCA to activations
    pca = PCA(n_components=3)
    activations_pca = pca.fit_transform(p_all)
    
    # Create DataFrame for plotting
    data_list = []
    for i, stage_id in enumerate(np.unique(stage_labels)):
        if stage_id < 0:
            continue
        mask = stage_labels == stage_id
        for pc in range(3):
            for val in activations_pca[mask, pc]:
                data_list.append({
                    'Stage': stage_names.get(stage_id, f'Stage{stage_id}'),
                    'PC': f'PC{pc+1}',
                    'Value': val
                })
    
    df = pd.DataFrame(data_list)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for pc in range(3):
        ax = axes[pc]
        pc_data = df[df['PC'] == f'PC{pc+1}']
        sns.violinplot(data=pc_data, x='Stage', y='Value', ax=ax)
        ax.set_title(f'PC{pc+1} (var: {pca.explained_variance_ratio_[pc]:.1%})')
        ax.set_xlabel('')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def main():
    base_dir = Path('plots/N1_V2.3_test')
    h5_path = r'C:\Users\jalma\OneDrive - HAN\stage_donders\features\N1_selection_v2\sleep_features_N1_selection_test.h5'
    
    analysis_dir = base_dir / 'mcrbm' / 'analysis'
    output_dir = base_dir / 'hypothesis_analysis'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    # Load sleep stage labels
    stage_labels = load_h5_labels(h5_path)
    
    # Load mcRBM activations
    p_all, state_info = load_mcrbm_activations(analysis_dir)
    
    print(f"Data shape: {p_all.shape}")
    print(f"Stage distribution:")
    unique, counts = np.unique(stage_labels, return_counts=True)
    stage_names = {0: 'Awake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Movement'}
    for u, c in zip(unique, counts):
        print(f"  {stage_names.get(u, f'Unknown({u})')}: {c} ({100*c/len(stage_labels):.1f}%)")
    
    run_n1_analysis(p_all, stage_labels, output_dir)
    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == '__main__':
    main()
