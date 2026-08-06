"""
Simple visualization for NPZ/HDF5 feature sets and mcRBM latent outputs.

Usage examples:

python scripts/visualize_npz.py \
  --npz "C:/.../sleep_features_N1_selection_test.npz" \
  --h5 "C:/.../sleep_features_N1_selection_test.h5" \
  --output_dir "plots"

Optional: provide `--latent` pointing to mcRBM latent vectors (n_samples x n_latent)
to plot latent space colored by true labels.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import importlib.util
import sys
import os
from pathlib import PurePath
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

STAGE_NAMES = {
    0: "Awake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "Movement time",
    6: "unknown_stage",
}

def _cluster_label_names(n_clusters: int) -> dict[int, str]:
    return {idx: f"Cluster {idx}" for idx in range(n_clusters)}


def _label_name(label: int, label_names: dict[int, str] | None = None) -> str:
    if label_names is None:
        label_names = STAGE_NAMES
    return label_names.get(int(label), str(int(label)))


def load_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if 'd' in data.files:
        return data['d']
    # fallback: first array-like
    first = data[data.files[0]]
    return first


def load_h5_labels(h5path: Path) -> tuple[np.ndarray, list[str]]:
    import h5py
    features_list = []
    labels_list = []
    names = []
    with h5py.File(h5path, 'r') as f:
        for subject in sorted(f.keys()):
            g = f[subject]
            if 'features' in g and 'scores' in g:
                feats = np.asarray(g['features'][:])
                scores = np.asarray(g['scores'][:]).astype(int)
                # features may be per-subject selected features; append all
                features_list.append(feats)
                labels_list.append(scores)
            if 'description_features' in g.attrs and not names:
                desc = g.attrs.get('description_features')
                if isinstance(desc, bytes):
                    desc = desc.decode('utf-8', errors='ignore')
                names = [s.strip() for s in str(desc).split(',') if s.strip()]
    if not features_list:
        raise RuntimeError('No features/scores found in HDF5')
    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    return X, y, names


def plot_pca(
    X: np.ndarray,
    y: np.ndarray | None,
    out_path: Path,
    title: str = 'PCA projection',
    label_names: dict[int, str] | None = None,
):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    if y is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.6)
    else:
        cmap = plt.get_cmap('tab10')
        classes = np.unique(y)
        for c in classes:
            mask = y == c
            ax.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.7, label=f"{c}:{_label_name(int(c), label_names)}",
                       color=cmap(int(c) % 10))
        ax.legend(markerscale=2, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pca_annotation_and_kmeans(
    X: np.ndarray,
    true_labels: np.ndarray,
    kmeans_labels: np.ndarray,
    out_path: Path,
    title: str = 'PCA annotated vs KMeans',
    annotation_names: dict[int, str] | None = None,
    cluster_names: dict[int, str] | None = None,
):
    """Plot the same PCA embedding twice: once colored by annotation and once by KMeans cluster."""
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    cmap = plt.get_cmap('tab10')

    # Left: annotated sleep stages
    ax = axes[0]
    for c in np.unique(true_labels):
        mask = true_labels == c
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=10,
            alpha=0.7,
            label=f"{int(c)}:{_label_name(int(c), annotation_names)}",
            color=cmap(int(c) % 10),
        )
    ax.set_title('Annotated sleep stage')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(markerscale=2, fontsize=8)

    # Right: KMeans clusters
    ax = axes[1]
    for c in np.unique(kmeans_labels):
        mask = kmeans_labels == c
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=10,
            alpha=0.7,
            label=f"{int(c)}:{_label_name(int(c), cluster_names)}",
            color=cmap(int(c) % 10),
        )
    ax.set_title('KMeans clusters')
    ax.set_xlabel('PC1')
    ax.legend(markerscale=2, fontsize=8)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_latent_vs_labels(
    latent: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str = 'Latent PCA colored by labels',
    label_names: dict[int, str] | None = None,
):
    # reduce latent to 2D for plotting
    pca = PCA(n_components=2)
    Z = pca.fit_transform(latent)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap('tab10')
    classes = np.unique(labels)
    for c in classes:
        mask = labels == c
        ax.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.7, label=f"{c}:{_label_name(int(c), label_names)}", color=cmap(int(c) % 10))
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_umap_or_tsne(
    X: np.ndarray,
    out_path: Path,
    labels: np.ndarray | None = None,
    title: str = 'UMAP/TSNE projection',
    label_names: dict[int, str] | None = None,
):
    """Compute 2D UMAP if available, otherwise TSNE, and save a scatter plot."""
    try:
        import umap.umap_ as umap
        reducer = umap.UMAP(n_components=2, random_state=0)
        Z = reducer.fit_transform(X)
        method = 'UMAP'
    except Exception:
        tsne = TSNE(n_components=2, random_state=0, init='pca')
        Z = tsne.fit_transform(X)
        method = 'TSNE'

    fig, ax = plt.subplots(figsize=(7, 6))
    if labels is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.6)
    else:
        cmap = plt.get_cmap('tab10')
        classes = np.unique(labels)
        for c in classes:
            mask = labels == c
            ax.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.7, label=f"{c}:{_label_name(int(c), label_names)}", color=cmap(int(c) % 10))
        ax.legend(markerscale=2, fontsize=8)
    ax.set_title(f"{title} ({method})")
    ax.set_xlabel('Dim1')
    ax.set_ylabel('Dim2')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def filter_out_stages(X: np.ndarray, labels: np.ndarray, exclude: list[int]) -> tuple[np.ndarray, np.ndarray]:
    mask = ~np.isin(labels, exclude)
    return X[mask], labels[mask]


def plot_per_stage_pca(
    X: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    prefix: str = 'per_stage',
    label_names: dict[int, str] | None = None,
):
    """Create a PCA scatter per sleep stage and save into out_dir/prefix/"""
    out_dir = Path(out_dir) / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    classes = np.unique(labels)
    cmap = plt.get_cmap('tab10')
    for c in classes:
        mask = labels == c
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.7, color=cmap(int(c) % 10))
        ax.set_title(f'PCA: Stage {c} ({_label_name(int(c), label_names)})')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.tight_layout()
        fig.savefig(out_dir / f'stage_{int(c)}_{_label_name(int(c), label_names).replace("/", "_")}.png', dpi=200)
        plt.close(fig)


def plot_pca_3d(X: np.ndarray, labels: np.ndarray | None, out_path: Path, title: str, label_names: dict[int, str] | None = None):
    pca = PCA(n_components=3)
    Z = pca.fit_transform(X)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    if labels is None:
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=5, alpha=0.5)
    else:
        cmap = plt.get_cmap('tab10')
        for c in np.unique(labels):
            mask = labels == c
            ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], s=8, alpha=0.7, color=cmap(int(c) % 10), label=f"{c}:{_label_name(int(c), label_names)}")
        ax.legend(markerscale=2, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_feature_matrix(X: np.ndarray, labels: np.ndarray | None, out_path: Path, title: str, feature_names: list[str] | None = None):
    """Plot pairwise feature scatter matrix for low-dimensional inputs."""
    n_features = X.shape[1]
    fig, axes = plt.subplots(n_features, n_features, figsize=(3.5 * n_features, 3.5 * n_features))
    if n_features == 1:
        axes = np.array([[axes]])
    cmap = plt.get_cmap('tab10')
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            if i == j:
                if labels is None:
                    ax.hist(X[:, j], bins=40, color='steelblue', alpha=0.8)
                else:
                    for c in np.unique(labels):
                        mask = labels == c
                        ax.hist(X[mask, j], bins=35, alpha=0.45, color=cmap(int(c) % 10), density=True)
                ax.set_ylabel('Density')
            else:
                if labels is None:
                    ax.scatter(X[:, j], X[:, i], s=3, alpha=0.35, color='steelblue')
                else:
                    for c in np.unique(labels):
                        mask = labels == c
                        ax.scatter(X[mask, j], X[mask, i], s=3, alpha=0.35, color=cmap(int(c) % 10))
            if i == n_features - 1:
                if feature_names and j < len(feature_names):
                    ax.set_xlabel(feature_names[j])
                else:
                    ax.set_xlabel(f'F{j+1}')
            else:
                ax.set_xticklabels([])
            if j == 0:
                if feature_names and i < len(feature_names):
                    ax.set_ylabel(feature_names[i])
                else:
                    ax.set_ylabel(f'F{i+1}')
            else:
                ax.set_yticklabels([])
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def try_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute silhouette score if valid, otherwise return None."""
    try:
        # silhouette requires at least 2 labels and less samples than n_samples
        if len(np.unique(labels)) > 1 and X.shape[0] > len(np.unique(labels)):
            score = silhouette_score(X, labels)
            return float(score)
    except Exception:
        pass
    return None


def pca_variance_summary(X: np.ndarray, n_components: int = 3) -> np.ndarray:
    n_components = min(n_components, X.shape[1], X.shape[0])
    if n_components < 1:
        return np.array([])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca.explained_variance_ratio_


def kmeans_silhouette_summary(X: np.ndarray, n_clusters: int = 6) -> tuple[float | None, np.ndarray | None, float | None]:
    try:
        if X.shape[0] <= n_clusters:
            return None, None, None
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(X)
        score = silhouette_score(X, km.labels_) if len(np.unique(km.labels_)) > 1 else None
        return (float(score) if score is not None else None, km.labels_, float(km.inertia_))
    except Exception:
        return None, None, None


def class_centroid_distances(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    centroids = np.vstack([X[labels == c].mean(axis=0) for c in classes])
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return classes, dist


def save_summary_csv(out_path: Path, rows: list[dict[str, object]]):
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_distance_matrix_csv(out_path: Path, classes: np.ndarray, dist: np.ndarray):
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['label'] + [str(int(c)) for c in classes])
        for idx, c in enumerate(classes):
            writer.writerow([int(c)] + [f'{value:.6f}' for value in dist[idx]])


def compute_latent_state_ids(latent_probs: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample latent state ids and the unique binary state patterns.

    The state id is the index of the first occurrence of each unique binary pattern
    in lexicographic order as returned by np.unique.
    """
    binary = (latent_probs >= threshold).astype(np.uint8)
    state_keys = np.array([''.join(map(str, row)) for row in binary])
    _, unique_indices, inverse_indices = np.unique(state_keys, return_index=True, return_inverse=True)
    unique_states = binary[unique_indices]
    return inverse_indices.astype(int), unique_states


def save_state_annotation_csv(
    out_path: Path,
    state_ids: np.ndarray,
    labels: np.ndarray,
    label_names: dict[int, str] | None = None,
):
    rows: list[dict[str, object]] = []
    for state_id in np.unique(state_ids):
        mask = state_ids == state_id
        counts = np.bincount(labels[mask].astype(int))
        total = int(mask.sum())
        dominant_label = int(np.argmax(counts)) if counts.size else -1
        dominant_name = _label_name(dominant_label, label_names) if dominant_label >= 0 else ''
        for label in np.unique(labels):
            label = int(label)
            count = int(np.sum(labels[mask] == label))
            rows.append({
                'state_id': int(state_id),
                'label_id': label,
                'label_name': _label_name(label, label_names),
                'count': count,
                'row_total': total,
                'row_fraction': (count / total) if total else 0.0,
                'dominant_label_id': dominant_label,
                'dominant_label_name': dominant_name,
            })
    save_summary_csv(out_path, rows)


def save_state_purity_csv(
    out_path: Path,
    state_ids: np.ndarray,
    labels: np.ndarray,
    label_names: dict[int, str] | None = None,
):
    rows: list[dict[str, object]] = []
    state_ids = np.asarray(state_ids).astype(int)
    labels = np.asarray(labels).astype(int)
    for state_id in np.unique(state_ids):
        mask = state_ids == state_id
        label_counts = np.bincount(labels[mask])
        total = int(mask.sum())
        dominant_label = int(np.argmax(label_counts)) if label_counts.size else -1
        dominant_name = _label_name(dominant_label, label_names) if dominant_label >= 0 else ''
        dominant_count = int(label_counts[dominant_label]) if dominant_label >= 0 else 0
        probs = label_counts / label_counts.sum() if label_counts.sum() > 0 else np.array([])
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12))) if probs.size else 0.0
        purity = (dominant_count / total) if total else 0.0
        rows.append({
            'state_id': int(state_id),
            'count': total,
            'dominant_label_id': dominant_label,
            'dominant_label_name': dominant_name,
            'dominant_count': dominant_count,
            'purity': purity,
            'entropy_bits': entropy,
        })
    save_summary_csv(out_path, rows)


def save_state_merge_map_csv(
    out_path: Path,
    merge_rows: list[dict[str, object]],
):
    save_summary_csv(out_path, merge_rows)


def summarize_latent_states(
    state_ids: np.ndarray,
    labels: np.ndarray,
    label_names: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    state_ids = np.asarray(state_ids).astype(int)
    labels = np.asarray(labels).astype(int)
    for state_id in np.unique(state_ids):
        mask = state_ids == state_id
        label_counts = np.bincount(labels[mask])
        total = int(mask.sum())
        dominant_label = int(np.argmax(label_counts)) if label_counts.size else -1
        dominant_name = _label_name(dominant_label, label_names) if dominant_label >= 0 else ''
        dominant_count = int(label_counts[dominant_label]) if dominant_label >= 0 else 0
        probs = label_counts / label_counts.sum() if label_counts.sum() > 0 else np.array([])
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12))) if probs.size else 0.0
        purity = (dominant_count / total) if total else 0.0
        rows.append({
            'state_id': int(state_id),
            'count': total,
            'dominant_label_id': dominant_label,
            'dominant_label_name': dominant_name,
            'dominant_count': dominant_count,
            'purity': purity,
            'entropy_bits': entropy,
        })
    return rows


def merge_latent_states(
    state_ids: np.ndarray,
    unique_states: np.ndarray,
    labels: np.ndarray,
    min_count: int = 250,
    min_purity: float = 0.55,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Merge rare or mixed latent states into the nearest stable state.

    Stable states are those meeting both the count and purity thresholds. All
    other states are assigned to the nearest stable state by Hamming distance on
    the binary latent pattern. If no state is stable, the original ids are kept.
    """
    state_ids = np.asarray(state_ids).astype(int)
    labels = np.asarray(labels).astype(int)
    state_order = np.unique(state_ids)
    stats = summarize_latent_states(state_ids, labels)
    stats_by_state = {int(row['state_id']): row for row in stats}
    state_to_idx = {int(state_id): idx for idx, state_id in enumerate(state_order)}

    stable_states = [int(row['state_id']) for row in stats if row['count'] >= min_count and row['purity'] >= min_purity]
    if not stable_states:
        stable_states = [int(state_id) for state_id in state_order]

    stable_indices = [state_to_idx[state_id] for state_id in stable_states]
    merged_ids = state_ids.copy()
    merge_rows: list[dict[str, object]] = []

    for state_id in state_order:
        state_id = int(state_id)
        row = stats_by_state[state_id]
        if state_id in stable_states:
            target_state = state_id
            hamming_distance = 0
            merged_reason = 'stable'
        else:
            candidate = unique_states[state_to_idx[state_id]].astype(int)
            best_target = stable_states[0]
            best_distance = None
            for stable_state, stable_idx in zip(stable_states, stable_indices):
                stable_pattern = unique_states[stable_idx].astype(int)
                distance = int(np.sum(candidate != stable_pattern))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_target = int(stable_state)
            target_state = int(best_target)
            hamming_distance = int(best_distance if best_distance is not None else 0)
            merged_reason = 'merged_to_nearest_stable'

        merged_ids[state_ids == state_id] = target_state
        merge_rows.append({
            'original_state_id': state_id,
            'original_count': int(row['count']),
            'original_purity': float(row['purity']),
            'dominant_label_id': int(row['dominant_label_id']),
            'dominant_label_name': row['dominant_label_name'],
            'merged_state_id': int(target_state),
            'merge_reason': merged_reason,
            'hamming_distance_to_target': hamming_distance,
        })

    # Renumber merged ids to contiguous integers for easier plotting and comparison.
    unique_merged = np.unique(merged_ids)
    remap = {int(old_id): new_id for new_id, old_id in enumerate(unique_merged)}
    merged_ids = np.array([remap[int(state_id)] for state_id in merged_ids], dtype=int)
    for row in merge_rows:
        row['merged_state_id'] = int(remap[int(row['merged_state_id'])])

    return merged_ids, merge_rows


def plot_state_transition_heatmap(
    state_ids: np.ndarray,
    out_path: Path,
    title: str = 'Latent state transition heatmap',
    normalize_rows: bool = True,
):
    state_ids = np.asarray(state_ids).astype(int)
    state_order = np.unique(state_ids)
    index = {int(state_id): idx for idx, state_id in enumerate(state_order)}
    counts = np.zeros((len(state_order), len(state_order)), dtype=float)
    for left, right in zip(state_ids[:-1], state_ids[1:]):
        counts[index[int(left)], index[int(right)]] += 1.0

    display = counts.copy()
    if normalize_rows:
        row_sums = display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display = display / row_sums

    fig, ax = plt.subplots(figsize=(0.32 * max(10, len(state_order)), 0.32 * max(10, len(state_order))))
    if normalize_rows:
        im = ax.imshow(display, aspect='auto', cmap='magma', interpolation='nearest', norm=Normalize(vmin=0.0, vmax=1.0))
    else:
        im = ax.imshow(display, aspect='auto', cmap='magma', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Next latent state id')
    ax.set_ylabel('Current latent state id')
    ax.set_xticks(np.arange(len(state_order)))
    ax.set_xticklabels([str(int(s)) for s in state_order], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(state_order)))
    ax.set_yticklabels([str(int(s)) for s in state_order], fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Row-normalized fraction' if normalize_rows else 'Transition count')

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_state_timeline(
    state_ids: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str = 'Annotation vs latent state timeline',
    label_names: dict[int, str] | None = None,
):
    state_ids = np.asarray(state_ids).astype(int)
    labels = np.asarray(labels).astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(16, 4), sharex=True)

    state_image = state_ids[np.newaxis, :]
    label_image = labels[np.newaxis, :]

    state_cmap = plt.get_cmap('tab20')
    label_cmap = plt.get_cmap('tab10')
    axes[0].imshow(state_image, aspect='auto', interpolation='nearest', cmap=state_cmap)
    axes[0].set_ylabel('State')
    axes[0].set_title('Latent state id over time')
    axes[0].set_yticks([])

    axes[1].imshow(label_image, aspect='auto', interpolation='nearest', cmap=label_cmap)
    axes[1].set_ylabel('Label')
    axes[1].set_title('Annotation over time')
    axes[1].set_yticks([])
    axes[1].set_xlabel('Fragment index')

    # Add a compact legend-like label summary on the right side.
    unique_labels = np.unique(labels)
    legend_text = '\n'.join([f'{int(l)}: {_label_name(int(l), label_names)}' for l in unique_labels])
    fig.text(0.985, 0.5, legend_text, va='center', ha='right', fontsize=8)
    fig.suptitle(title, y=1.03)
    fig.tight_layout(rect=[0.0, 0.0, 0.94, 1.0])
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_latent_state_annotation_heatmap(
    state_ids: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str = 'Latent state vs annotation heatmap',
    label_names: dict[int, str] | None = None,
    normalize_rows: bool = False,
):
    """Plot a heatmap of latent states (rows) against annotation labels (columns)."""
    state_ids = np.asarray(state_ids).astype(int)
    labels = np.asarray(labels).astype(int)

    state_order = np.unique(state_ids)
    label_order = np.unique(labels)

    counts = np.zeros((len(state_order), len(label_order)), dtype=float)
    for row_idx, state_id in enumerate(state_order):
        mask = state_ids == state_id
        label_counts = np.bincount(labels[mask], minlength=int(label_order.max()) + 1 if len(label_order) else 0)
        for col_idx, label_id in enumerate(label_order):
            counts[row_idx, col_idx] = float(label_counts[int(label_id)])

    display = counts.copy()
    if normalize_rows:
        row_sums = display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display = display / row_sums

    fig, ax = plt.subplots(figsize=(1.2 * max(6, len(label_order)), 0.35 * max(8, len(state_order))))
    if normalize_rows:
        im = ax.imshow(display, aspect='auto', cmap='viridis', interpolation='nearest', norm=Normalize(vmin=0.0, vmax=1.0))
    else:
        im = ax.imshow(display, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Annotation label')
    ax.set_ylabel('Latent state id')

    ax.set_xticks(np.arange(len(label_order)))
    ax.set_xticklabels([f"{int(label)}: {_label_name(int(label), label_names)}" for label in label_order], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(state_order)))
    ax.set_yticklabels([str(int(state_id)) for state_id in state_order])

    # Annotate non-zero counts for readability.
    if counts.size > 0:
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                value = int(counts[i, j])
                if value <= 0:
                    continue
                rgba = im.cmap(im.norm(display[i, j]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = 'black' if luminance > 0.5 else 'white'
                ax.text(j, i, str(value), ha='center', va='center', color=text_color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Row-normalized fraction' if normalize_rows else 'Count')

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def maybe_sample_for_plots(X: np.ndarray, labels: np.ndarray | None, max_samples: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    if max_samples <= 0 or X.shape[0] <= max_samples:
        return X, labels
    rng = np.random.default_rng(seed)
    indices = rng.choice(X.shape[0], size=max_samples, replace=False)
    indices.sort()
    X_sampled = X[indices]
    labels_sampled = labels[indices] if labels is not None else None
    return X_sampled, labels_sampled


def run_mcrbm_inference(X: np.ndarray, out_dir: Path, model_dir: Path, model_file: str = 'ws_final.mat', use_gpu: bool = False):
    """Prepare visData and call the repo mcRBM inference implementation.

    Saves outputs into out_dir/analysis and returns path to activations.npz if produced.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_path = out_dir / 'visData.npz'
    # obsKeys: placeholder zeros; mcRBMInference may map to obs keys if provided
    obs_keys = np.zeros(X.shape[0], dtype=int)
    np.savez(vis_path, data=X.astype(np.float32), obsKeys=obs_keys)

    # Import the inference module by path
    infer_path = Path('Jalmar') / 'mcRBM' / 'scripts' / 'infer_states.py'
    if not infer_path.exists():
        raise FileNotFoundError(f'mcRBM inference script not found at {infer_path}')

    spec = importlib.util.spec_from_file_location('mcrbm_infer', str(infer_path))
    m = importlib.util.module_from_spec(spec)
    # Ensure the mcRBM scripts directory is on sys.path so local imports like array_backend work
    scripts_dir = str(infer_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    # Ensure stdout encoding supports unicode checkmarks used by the backend
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    sys.modules['mcrbm_infer'] = m
    spec.loader.exec_module(m)
    # cleanup inserted path (optional)
    try:
        if sys.path[0] == scripts_dir:
            del sys.path[0]
    except Exception:
        pass

    # Instantiate inference, using out_dir as expDir so outputs land there
    original_cwd = Path.cwd()
    inf = m.mcRBMInference(str(Path(model_dir)), str(out_dir), modelFile=model_file, use_gpu=use_gpu)
    # loadData expects statesFilePath and statesFile; pass empty/unused
    try:
        inf.loadData('.', '')
    except Exception:
        # loadData may have already loaded visData via cwd; ignore errors
        pass

    try:
        # Attempt computeStates; if shapes mismatch due to FH orientation, try transposing FH and retry once.
        try:
            results = inf.computeStates(saveProbabilities=True, saveBinary=True)
        except Exception as e:
            err = str(e)
            if 'shapes' in err or 'not aligned' in err:
                if hasattr(inf, 'FH'):
                    inf.FH = inf.FH.T
                    results = inf.computeStates(saveProbabilities=True, saveBinary=True)
                else:
                    raise
            else:
                raise
    finally:
        try:
            os.chdir(original_cwd)
        except Exception:
            pass

    # analysis/activations.npz will exist in out_dir
    activations_path = out_dir / 'analysis' / 'activations.npz'
    if activations_path.exists():
        return activations_path
    return None


def run_full_analysis(X: np.ndarray, labels: np.ndarray | None, output_dir: Path, title_prefix: str, exclude_movement: bool = False, feature_names: list[str] | None = None) -> list[dict[str, object]]:
    """
    Run a complete analysis suite on data: PCA, 3D, feature matrix, per-stage, UMAP, silhouette, KMeans, no-movement variants.
    
    Returns list of summary rows for CSV.
    """
    summary_rows: list[dict[str, object]] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Core metrics
    pca_var = pca_variance_summary(X)
    kmeans_score, kmeans_labels_data, kmeans_inertia = kmeans_silhouette_summary(X)
    stage_silhouette = None
    if labels is not None:
        stage_silhouette = try_silhouette(X, labels)
    
    summary_rows.append({
        'dataset': title_prefix,
        'samples': X.shape[0],
        'features': X.shape[1],
        'labels_present': labels is not None,
        'pca_var_1': float(pca_var[0]) if len(pca_var) > 0 else '',
        'pca_var_2': float(pca_var[1]) if len(pca_var) > 1 else '',
        'pca_var_3': float(pca_var[2]) if len(pca_var) > 2 else '',
        'kmeans_silhouette': '' if kmeans_score is None else float(kmeans_score),
        'kmeans_inertia': '' if kmeans_inertia is None else float(kmeans_inertia),
        'stage_silhouette': '' if stage_silhouette is None else float(stage_silhouette),
    })
    
    # PCA plots
    try:
        plot_pca(X, labels, output_dir / f'{title_prefix}_pca.png', title=f'{title_prefix} PCA')
        print(f'Saved {output_dir / f"{title_prefix}_pca.png"}')
    except Exception as e:
        print(f'{title_prefix} PCA failed:', e)
    
    # 3D PCA
    try:
        plot_pca_3d(X, labels, output_dir / f'{title_prefix}_pca_3d.png', title=f'{title_prefix} PCA 3D')
        print(f'Saved {output_dir / f"{title_prefix}_pca_3d.png"}')
    except Exception as e:
        print(f'{title_prefix} 3D PCA failed:', e)
    
    # Feature matrix
    try:
        plot_feature_matrix(X, labels, output_dir / f'{title_prefix}_feature_matrix.png', title=f'{title_prefix} feature matrix', feature_names=feature_names)
        print(f'Saved {output_dir / f"{title_prefix}_feature_matrix.png"}')
    except Exception as e:
        print(f'{title_prefix} feature matrix failed:', e)
    
    # Per-stage plots and centroid distances
    if labels is not None:
        try:
            plot_per_stage_pca(X, labels, output_dir, prefix='per_stage')
            print(f'Saved per-stage PCA plots to {output_dir / "per_stage"}')
            
            classes, dist = class_centroid_distances(X, labels)
            save_distance_matrix_csv(output_dir / 'stage_centroid_distances.csv', classes, dist)
            print(f'Saved {output_dir / "stage_centroid_distances.csv"}')
        except Exception as e:
            print(f'{title_prefix} per-stage analysis failed:', e)
    
    # UMAP/TSNE
    try:
        compute_umap_or_tsne(X, output_dir / f'{title_prefix}_umap.png', labels=labels, title=f'{title_prefix} features')
        print(f'Saved {output_dir / f"{title_prefix}_umap.png"}')
    except Exception as e:
        print(f'{title_prefix} UMAP/TSNE failed:', e)
    
    # KMeans comparison on PCA
    try:
        if labels is not None and kmeans_labels_data is not None:
            n_clusters = int(np.max(kmeans_labels_data)) + 1 if kmeans_labels_data.size else 0
            plot_pca_annotation_and_kmeans(
                X,
                labels,
                kmeans_labels_data,
                output_dir / f'{title_prefix}_pca_kmeans.png',
                title=f'{title_prefix} PCA: annotation vs KMeans',
                annotation_names=STAGE_NAMES,
                cluster_names=_cluster_label_names(n_clusters) if n_clusters else None,
            )
        elif labels is not None:
            plot_pca(X, labels, output_dir / f'{title_prefix}_pca_kmeans.png', title=f'{title_prefix} PCA colored by annotation', label_names=STAGE_NAMES)
        else:
            plot_pca(X, kmeans_labels_data, output_dir / f'{title_prefix}_pca_kmeans.png', title=f'{title_prefix} PCA colored by KMeans', label_names=_cluster_label_names(int(np.max(kmeans_labels_data)) + 1) if kmeans_labels_data is not None and kmeans_labels_data.size else None)
        print(f'Saved {output_dir / f"{title_prefix}_pca_kmeans.png"}')
    except Exception as e:
        print(f'{title_prefix} KMeans overlay failed:', e)
    
    # No-movement variants
    if labels is not None and exclude_movement:
        try:
            X_nom, labels_nom = filter_out_stages(X, labels, [5])
            score_nom = try_silhouette(X_nom, labels_nom)
            if score_nom is not None:
                print(f"{title_prefix} silhouette (no movement): {score_nom:.3f}")
            pca_nom = pca_variance_summary(X_nom)
            summary_rows.append({
                'dataset': f'{title_prefix} [no movement]',
                'samples': X_nom.shape[0],
                'features': X_nom.shape[1],
                'labels_present': True,
                'pca_var_1': float(pca_nom[0]) if len(pca_nom) > 0 else '',
                'pca_var_2': float(pca_nom[1]) if len(pca_nom) > 1 else '',
                'pca_var_3': float(pca_nom[2]) if len(pca_nom) > 2 else '',
                'kmeans_silhouette': '',
                'kmeans_inertia': '',
                'stage_silhouette': '' if score_nom is None else float(score_nom),
            })
            plot_pca(X_nom, labels_nom, output_dir / f'{title_prefix}_pca_no_movement.png', title=f'{title_prefix} PCA (no movement)')
            compute_umap_or_tsne(X_nom, output_dir / f'{title_prefix}_umap_no_movement.png', labels=labels_nom, title=f'{title_prefix} (no movement)')
            plot_pca_3d(X_nom, labels_nom, output_dir / f'{title_prefix}_pca_3d_no_movement.png', title=f'{title_prefix} PCA 3D (no movement)')
            plot_feature_matrix(X_nom, labels_nom, output_dir / f'{title_prefix}_feature_matrix_no_movement.png', title=f'{title_prefix} feature matrix (no movement)', feature_names=feature_names)
            print(f'Saved {title_prefix} plots without movement')
        except Exception as e:
            print(f'{title_prefix} no-movement analysis failed:', e)
    
    return summary_rows


def run_latent_state_annotation_analysis(
    latent_probs: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    title_prefix: str,
    label_names: dict[int, str] | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_ids, unique_states = compute_latent_state_ids(latent_probs)
    plot_latent_state_annotation_heatmap(
        state_ids,
        labels,
        output_dir / f'{title_prefix}_state_annotation_heatmap.png',
        title=f'{title_prefix} latent state vs annotation heatmap',
        label_names=label_names,
        normalize_rows=False,
    )
    save_state_annotation_csv(
        output_dir / f'{title_prefix}_state_annotation_counts.csv',
        state_ids,
        labels,
        label_names=label_names,
    )
    save_state_purity_csv(
        output_dir / f'{title_prefix}_state_purity.csv',
        state_ids,
        labels,
        label_names=label_names,
    )
    plot_state_transition_heatmap(
        state_ids,
        output_dir / f'{title_prefix}_state_transition_heatmap.png',
        title=f'{title_prefix} latent state transitions',
        normalize_rows=True,
    )
    plot_state_timeline(
        state_ids,
        labels,
        output_dir / f'{title_prefix}_state_timeline.png',
        title=f'{title_prefix} latent state vs annotation timeline',
        label_names=label_names,
    )
    np.save(output_dir / f'{title_prefix}_state_ids.npy', state_ids)
    np.save(output_dir / f'{title_prefix}_unique_states.npy', unique_states)
    print(f'Saved {output_dir / f"{title_prefix}_state_annotation_heatmap.png"}')
    print(f'Saved {output_dir / f"{title_prefix}_state_annotation_counts.csv"}')
    print(f'Saved {output_dir / f"{title_prefix}_state_purity.csv"}')
    print(f'Saved {output_dir / f"{title_prefix}_state_transition_heatmap.png"}')
    print(f'Saved {output_dir / f"{title_prefix}_state_timeline.png"}')
    return state_ids, unique_states


def run_latent_state_merge_analysis(
    latent_probs: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    title_prefix: str,
    label_names: dict[int, str] | None = None,
    merge_min_count: int = 250,
    merge_min_purity: float = 0.55,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_state_ids, unique_states = compute_latent_state_ids(latent_probs)
    pre_prefix = f'{title_prefix}_pre_merge'
    post_prefix = f'{title_prefix}_post_merge'

    # Save the raw state mapping first so the pre-merge state space is preserved.
    run_latent_state_annotation_analysis(
        latent_probs,
        labels,
        output_dir,
        pre_prefix,
        label_names=label_names,
    )

    merged_state_ids, merge_rows = merge_latent_states(
        pre_state_ids,
        unique_states,
        labels,
        min_count=merge_min_count,
        min_purity=merge_min_purity,
    )

    plot_latent_state_annotation_heatmap(
        merged_state_ids,
        labels,
        output_dir / f'{post_prefix}_state_annotation_heatmap.png',
        title=f'{title_prefix} merged latent state vs annotation heatmap',
        label_names=label_names,
        normalize_rows=False,
    )
    save_state_annotation_csv(
        output_dir / f'{post_prefix}_state_annotation_counts.csv',
        merged_state_ids,
        labels,
        label_names=label_names,
    )
    save_state_purity_csv(
        output_dir / f'{post_prefix}_state_purity.csv',
        merged_state_ids,
        labels,
        label_names=label_names,
    )
    plot_state_transition_heatmap(
        merged_state_ids,
        output_dir / f'{post_prefix}_state_transition_heatmap.png',
        title=f'{title_prefix} merged latent state transitions',
        normalize_rows=True,
    )
    plot_state_timeline(
        merged_state_ids,
        labels,
        output_dir / f'{post_prefix}_state_timeline.png',
        title=f'{title_prefix} merged latent state vs annotation timeline',
        label_names=label_names,
    )
    np.save(output_dir / f'{post_prefix}_state_ids.npy', merged_state_ids)
    save_state_merge_map_csv(output_dir / f'{title_prefix}_state_merge_map.csv', merge_rows)

    print(f'Saved {output_dir / f"{post_prefix}_state_annotation_heatmap.png"}')
    print(f'Saved {output_dir / f"{post_prefix}_state_annotation_counts.csv"}')
    print(f'Saved {output_dir / f"{post_prefix}_state_purity.csv"}')
    print(f'Saved {output_dir / f"{post_prefix}_state_transition_heatmap.png"}')
    print(f'Saved {output_dir / f"{post_prefix}_state_timeline.png"}')
    print(f'Saved {output_dir / f"{title_prefix}_state_merge_map.csv"}')

    return pre_state_ids, merged_state_ids, unique_states, merge_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--h5', required=False)
    p.add_argument('--latent', required=False, help='Path to latent vectors (n x d) npy or npz')
    p.add_argument('--mcrbm_run', dest='mcrbm_run', action='store_true', help='Run mcRBM inference using model weights (default: on)')
    p.add_argument('--no-mcrbm-run', dest='mcrbm_run', action='store_false', help='Skip mcRBM inference')
    p.add_argument('--mcrbm_model_dir', required=False, help='Directory containing mcRBM weights (.mat)')
    p.add_argument('--mcrbm_model_file', required=False, default='ws_final.mat', help='mcRBM weights filename')
    p.add_argument('--use_gpu', action='store_true', help='Use GPU backend for mcRBM inference')
    p.add_argument('--exclude_movement', action='store_true', help='Exclude movement/artifact (stage 5) from specific plots and scoring')
    p.add_argument('--plot_sample_size', type=int, default=0, help='If >0, randomly subsample this many rows for expensive plots and mcRBM inference')
    p.add_argument('--run_n1_analysis', action='store_true', help='Run the N1 hypothesis analysis after mcRBM inference')
    p.add_argument('--n1_analysis_output_dir', default='hypothesis_analysis', help='Subdirectory name for the N1 analysis outputs')
    p.add_argument('--merge_min_count', type=int, default=250, help='Merge latent states with fewer samples than this into nearby stable states')
    p.add_argument('--merge_min_purity', type=float, default=0.55, help='Merge latent states with purity below this threshold')
    p.add_argument('--output_dir', default='plots')
    p.set_defaults(mcrbm_run=True)
    args = p.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = out_dir / 'npz'
    mcrbm_dir = out_dir / 'mcrbm'
    npz_dir.mkdir(parents=True, exist_ok=True)
    mcrbm_dir.mkdir(parents=True, exist_ok=True)

    X = load_npz(npz_path)
    print('Loaded NPZ', npz_path, 'shape', X.shape)

    labels = None
    feature_names = None
    if args.h5:
        h5_path = Path(args.h5)
        X_h5, y_h5, names = load_h5_labels(h5_path)
        print('Loaded HDF5 features', X_h5.shape, 'labels', y_h5.shape)
        # Only color directly when the row counts align.
        if X_h5.shape[0] == X.shape[0]:
            labels = y_h5
            feature_names = names
        else:
            print('Warning: HDF5 label count does not match NPZ row count; stage-colored plots will be skipped for this run.')

    plot_X, plot_labels = maybe_sample_for_plots(X, labels, args.plot_sample_size)
    if plot_X.shape[0] != X.shape[0]:
        print(f'Using sampled subset for plots and mcRBM: {plot_X.shape[0]} / {X.shape[0]} rows')

    # Run full NPZ analysis
    summary_rows = run_full_analysis(plot_X, plot_labels, npz_dir, 'npz', exclude_movement=args.exclude_movement, feature_names=feature_names)

    # Run mcRBM inference and analysis
    if args.mcrbm_run:
        model_dir = Path(args.mcrbm_model_dir) if args.mcrbm_model_dir else Path('experiments') / 'mcrbm_N1_test' / 'weights'
        model_file = args.mcrbm_model_file
        print('\nRunning mcRBM inference using model:', model_dir / model_file)
        try:
            activations = run_mcrbm_inference(plot_X, mcrbm_dir, model_dir, model_file=model_file, use_gpu=args.use_gpu)
            if activations is not None and activations.exists():
                print('mcRBM activations saved at', activations)
                container = np.load(activations)
                p_all = container['p_all']
                if plot_labels is not None:
                    run_latent_state_merge_analysis(
                        p_all,
                        plot_labels,
                        mcrbm_dir,
                        'mcrbm',
                        label_names=STAGE_NAMES,
                        merge_min_count=args.merge_min_count,
                        merge_min_purity=args.merge_min_purity,
                    )
                print('Running full analysis on mcRBM activations...')
                mcrbm_summary_rows = run_full_analysis(p_all, plot_labels, mcrbm_dir, 'mcrbm', exclude_movement=args.exclude_movement, feature_names=None)
                summary_rows.extend(mcrbm_summary_rows)

                if args.run_n1_analysis:
                    if not args.h5 or labels is None:
                        print('N1 hypothesis analysis skipped: --h5 is required to load stage labels.')
                    else:
                        try:
                            import analyze_N1_hypothesis as n1_analysis
                            n1_output_dir = out_dir / args.n1_analysis_output_dir
                            print('\nRunning N1 hypothesis analysis...')
                            n1_analysis.run_n1_analysis(p_all, plot_labels, n1_output_dir)
                        except Exception as e:
                            print('N1 hypothesis analysis failed:', e)
        except Exception as e:
            print('mcRBM inference failed:', e)

    save_summary_csv(out_dir / 'cluster_summary.csv', summary_rows)
    print('Saved', out_dir / 'cluster_summary.csv')

if __name__ == '__main__':
    main()
