"""
Preprocess raw features HDF5 for mcRBM training.

This script:
- Reads a raw features HDF5 (per-subject groups with 'features' datasets).
- Computes and prints/saves diagnostics before and after each preprocessing step.
- Steps: global winsorization (percentile clipping), optional log-transform for skewed positive features,
  robust or standard scaling, optional PCA (with whitening).
- Saves final per-subject processed `features` into an output HDF5 and stores per-step stats.

Usage example:
python preprocess_raw_for_mcRBM.py \
  --input C:\\path\\to\\sleep_features_raw.h5 \
  --output C:\\path\\to\\sleep_features_preprocessed.h5 \
  --winsor 1 99 --scaler robust --pca-var 0.99 --whiten

"""

from __future__ import annotations
import argparse, json, os
from collections import OrderedDict
import h5py
import numpy as np
from scipy import stats

try:
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.decomposition import PCA
except Exception as e:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")


def compute_stats(X: np.ndarray, low_clip=0.05, high_clip=0.9999):
    d = {}
    d['shape'] = list(X.shape)
    d['mean'] = X.mean(axis=0).tolist()
    d['std'] = X.std(axis=0).tolist()
    d['min'] = X.min(axis=0).tolist()
    d['max'] = X.max(axis=0).tolist()
    d['p1'] = np.percentile(X, 1, axis=0).tolist()
    d['p99'] = np.percentile(X, 99, axis=0).tolist()
    try:
        skew = stats.skew(X, axis=0, nan_policy='omit')
        kurt = stats.kurtosis(X, axis=0, nan_policy='omit')
        d['skew'] = skew.tolist()
        d['kurtosis'] = kurt.tolist()
    except Exception:
        d['skew'] = None
        d['kurtosis'] = None
    # clipping counts
    d['clip_low_counts'] = (X <= low_clip).sum(axis=0).tolist()
    d['clip_high_counts'] = (X >= high_clip).sum(axis=0).tolist()
    # correlations
    try:
        corr = np.corrcoef(X.T)
        off = corr.copy()
        np.fill_diagonal(off, 0)
        d['mean_abs_corr'] = float(np.mean(np.abs(off)))
        d['max_abs_corr'] = float(np.max(np.abs(off)))
    except Exception:
        d['mean_abs_corr'] = None
        d['max_abs_corr'] = None
    return d


def load_raw_features(input_h5: str):
    subs = OrderedDict()
    scores = OrderedDict()
    attrs = OrderedDict()
    with h5py.File(input_h5, 'r') as f:
        keys = sorted([k for k in f.keys()])
        for k in keys:
            grp = f[k]
            if 'features' in grp:
                X = grp['features'][:].astype(np.float32)
                subs[k] = X
                if 'scores' in grp:
                    scores[k] = grp['scores'][:]
                else:
                    scores[k] = None
                # copy relevant attributes if present
                subattrs = {}
                for a_key, a_val in grp.attrs.items():
                    try:
                        # convert bytes to str for JSON-friendly storage
                        if isinstance(a_val, bytes):
                            subattrs[a_key] = a_val.decode('utf-8', errors='ignore')
                        else:
                            subattrs[a_key] = a_val
                    except Exception:
                        subattrs[a_key] = str(a_val)
                attrs[k] = subattrs
    return subs, scores, attrs


def stack_subjects(subs: dict[str, np.ndarray]):
    feats = list(subs.values())
    X_all = np.vstack(feats) if feats else np.zeros((0, 0))
    return X_all


def apply_winsorization(X_all: np.ndarray, low_perc: float, high_perc: float):
    lows = np.percentile(X_all, low_perc, axis=0)
    highs = np.percentile(X_all, high_perc, axis=0)
    X_clip = np.clip(X_all, lows[None, :], highs[None, :])
    return X_clip, lows, highs


def apply_per_subject_clip(subs: dict[str, np.ndarray], lows: np.ndarray, highs: np.ndarray):
    res = {}
    for k, X in subs.items():
        res[k] = np.clip(X, lows[None, :], highs[None, :])
    return res


def auto_log_transform(X_all: np.ndarray, min_pos_threshold=1e-8, skew_thresh=1.0):
    # Decide which columns to log-transform: positive-only and skew > threshold
    mins = X_all.min(axis=0)
    skews = stats.skew(X_all, axis=0, nan_policy='omit')
    to_log = []
    for i, (mn, sk) in enumerate(zip(mins, skews)):
        if mn > min_pos_threshold and sk > skew_thresh:
            to_log.append(i)
    # apply log1p per chosen column
    X_new = X_all.copy()
    if to_log:
        X_new[:, to_log] = np.log1p(X_new[:, to_log])
    return X_new, to_log


def apply_scaler(X_all: np.ndarray, scaler_name: str):
    if scaler_name == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    return X_scaled, scaler


def apply_pca_whiten(X_all: np.ndarray, var_threshold: float | None = None, n_components: int | None = None, whiten: bool = True):
    if var_threshold is not None:
        pca = PCA(n_components=var_threshold, svd_solver='full', whiten=whiten)
    elif n_components is not None:
        pca = PCA(n_components=n_components, whiten=whiten)
    else:
        raise ValueError('Either var_threshold or n_components must be provided')
    X_pca = pca.fit_transform(X_all)
    return X_pca, pca


def write_output_h5(outfile: str, subs_processed: dict[str, np.ndarray], subs_scores: dict[str, np.ndarray] | None = None, subs_attrs: dict | None = None):
    """Write a final HDF5 file matching the raw structure: per-subject groups with
    'features' (required) and 'scores' (if available). Do not include /stats.
    """
    with h5py.File(outfile, 'w') as f:
        for k, X in subs_processed.items():
            grp = f.create_group(k)
            grp.create_dataset('features', data=X.astype(np.float32), compression='gzip')
            if subs_scores is not None and k in subs_scores and subs_scores[k] is not None:
                grp.create_dataset('scores', data=subs_scores[k])
            # restore attributes when available (keep original description_features, etc.)
            if subs_attrs is not None and k in subs_attrs and isinstance(subs_attrs[k], dict):
                for a_key, a_val in subs_attrs[k].items():
                    try:
                        grp.attrs[a_key] = a_val
                    except Exception:
                        try:
                            grp.attrs[a_key] = str(a_val)
                        except Exception:
                            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Raw features HDF5 input')
    p.add_argument('--output', required=True, help='Output HDF5 path for processed features')
    p.add_argument('--winsor', nargs=2, type=float, default=[1.0, 99.0], help='Winsorize percentiles low high (default 1 99)')
    p.add_argument('--no-log-auto', dest='log_auto', action='store_false', help='Disable automatic log-transform')
    p.add_argument('--scaler', choices=['robust', 'standard'], default='robust')
    p.add_argument('--pca-var', type=float, default=None, help='If set, PCA var threshold (0<x<=1) to keep that fraction of variance')
    p.add_argument('--pca-n', type=int, default=None, help='If set, number of PCA components')
    p.add_argument('--whiten', action='store_true', help='Apply whitening in PCA')
    p.add_argument('--save-intermediate', action='store_true', help='Save intermediate per-step datasets into output HDF5')

    args = p.parse_args()

    inp = args.input
    out = args.output
    winsor_low, winsor_high = float(args.winsor[0]), float(args.winsor[1])

    print(f"Loading raw features from {inp}...")
    subs, subs_scores, subs_attrs = load_raw_features(inp)
    print(f"Loaded {len(subs)} subjects.")

    X_all = stack_subjects(subs)
    print('Computing baseline stats...')
    stats_all = {}
    stats_all['baseline'] = compute_stats(X_all)
    print(json.dumps({'baseline': {k: stats_all['baseline'][k] for k in ['shape','mean','std','p1','p99','mean_abs_corr']}}, indent=2))

    intermediates = {} if args.save_intermediate else None
    if intermediates is not None:
        intermediates['baseline'] = subs

    # Winsorization
    print(f"Applying winsorization: {winsor_low}th -> {winsor_high}th percentiles (global)...")
    X_clip, lows, highs = apply_winsorization(X_all, winsor_low, winsor_high)
    stats_all['winsor'] = compute_stats(X_clip)
    print('Winsorization stats:')
    print(json.dumps({'winsor': {k: stats_all['winsor'][k] for k in ['mean','std','p1','p99']}}, indent=2))
    subs_wins = apply_per_subject_clip(subs, lows, highs)
    if intermediates is not None:
        intermediates['winsor'] = subs_wins

    # Log-transform auto
    subs_after_log = subs_wins
    X_for_log = stack_subjects(subs_after_log)
    to_log = []
    if args.log_auto:
        print('Detecting positive, highly skewed features for log-transform...')
        X_logged, to_log = auto_log_transform(X_for_log)
        if to_log:
            print(f'Auto log-transform will apply to feature indices: {to_log}')
            # apply per-subject
            subs_after_log = {k: v.copy() for k, v in subs_after_log.items()}
            for s, X in subs_after_log.items():
                X_local = X.copy()
                X_local[:, to_log] = np.log1p(X_local[:, to_log])
                subs_after_log[s] = X_local
            stats_all['log'] = compute_stats(stack_subjects(subs_after_log))
            print('Post-log stats (sample):')
            print(json.dumps({'log': {k: stats_all['log'][k] for k in ['mean','std','skew']}}, indent=2))
            if intermediates is not None:
                intermediates['log'] = subs_after_log
        else:
            print('No features met auto log-transform criteria.')
            stats_all['log'] = stats_all['winsor']
    else:
        print('Auto log-transform disabled.')
        stats_all['log'] = stats_all['winsor']

    # Scaling
    print(f'Applying scaler: {args.scaler} (fit on global data after previous steps)...')
    X_after_log = stack_subjects(subs_after_log)
    X_scaled_all, scaler = apply_scaler(X_after_log, args.scaler)
    stats_all['scaled'] = compute_stats(X_scaled_all)
    print('Scaled stats (mean/std):')
    print(json.dumps({'scaled': {'mean': stats_all['scaled']['mean'], 'std': stats_all['scaled']['std']}}, indent=2))
    # Apply per-subject transform
    subs_scaled = {}
    # We must transform per-subject using the fitted scaler
    if args.scaler == 'robust':
        trans = RobustScaler().fit(X_after_log)
        # but we already fit above; reuse scaler object instead
        trans = scaler
    else:
        trans = scaler
    for s, X in subs_after_log.items():
        subs_scaled[s] = trans.transform(X)
    if intermediates is not None:
        intermediates['scaled'] = subs_scaled

    # PCA / Whitening
    subs_final = subs_scaled
    if args.pca_var is not None or args.pca_n is not None:
        print('Applying PCA...')
        X_scaled_all = stack_subjects(subs_scaled)
        if args.pca_var is not None:
            X_pca_all, pca = apply_pca_whiten(X_scaled_all, var_threshold=args.pca_var, whiten=args.whiten)
        else:
            X_pca_all, pca = apply_pca_whiten(X_scaled_all, n_components=args.pca_n, whiten=args.whiten)
        stats_all['pca'] = compute_stats(X_pca_all)
        print('PCA stats:')
        print(json.dumps({'pca': {k: stats_all['pca'][k] for k in ['shape','mean','std']}}, indent=2))
        # split back per subject
        subs_final = {}
        idx = 0
        for s, X in subs_scaled.items():
            n = X.shape[0]
            subs_final[s] = X_pca_all[idx:idx+n]
            idx += n
        if intermediates is not None:
            intermediates['pca'] = subs_final
    else:
        print('Skipping PCA; no PCA args provided.')

    # Final write
    print(f'Writing final processed HDF5 to {out} ...')
    write_output_h5(out, subs_final, subs_scores, subs_attrs)
    print('Done. Final HDF5 written (no /stats group).')


if __name__ == '__main__':
    main()
