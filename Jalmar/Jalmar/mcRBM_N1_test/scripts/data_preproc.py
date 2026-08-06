"""
Optional preprocessing helpers for mcRBM.

This module is only needed if you want mcRBM-local scaling, PCA, whitening, or
batch trimming before training on features.
"""

import numpy as np
from sklearn.decomposition import PCA
import pickle
import os
import argparse
from scipy.io import loadmat, savemat


# Configuration defaults for CLI usage.
CONFIG = {
    'PATHS': {
        'input_npz': r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\N1_selection_npz\sleep_features_N1_selection_train.npz",
        'output_npz': r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\N1_selection_npz\sleep_features_N1_selection_train_processed.npz",
        'save_dir': r"./experiments",
        'stats_file': r"scaler_stats.pkl",
    }
}


class DataPreproc:
    """
    Class for preprocessing electrophysiological data before mcRBM training.
    Handles scaling, normalization, log transforms, PCA, and whitening.
    """

    def __init__(self):
        self.name = 'mcrbm_dpp'

    def trimForGPU(self, d, obsKeys, epochTime, batch_size):
        """
        Extract subset of data matrix that divides evenly into batches.
        
        Args:
            d: data matrix (epochs × features)
            obsKeys: epoch IDs and labels
            epochTime: epoch timestamps
            batch_size: desired batch size
            
        Returns:
            Trimmed versions of d, obsKeys, epochTime
        """
        totnumcases = d.shape[0]
        trim_idx = int(np.floor(totnumcases / batch_size) * batch_size)
        
        return (
            d[:trim_idx, :].copy(),
            obsKeys[:trim_idx].copy() if obsKeys is not None else None,
            epochTime[:trim_idx, :].copy() if epochTime is not None else None
        )

    def preprocAndScaleData(
        self, d, obsKeys, logFlag, meanSubtractionFlag, scalingFlag, scaling,
        pcaFlag, whitenFlag, rescalingFlag, rescaling, minmaxFile, saveDir
    ):
        """
        Apply preprocessing and scaling to data matrix.
        
        Args:
            d: data matrix
            obsKeys: epoch labels/IDs
            logFlag: apply log transform
            meanSubtractionFlag: subtract mean
            scalingFlag: apply scaling
            scaling: scaling method ('standard', 'minmax', 'robust')
            pcaFlag: apply PCA
            whitenFlag: apply whitening
            rescalingFlag: rescale after preprocessing
            rescaling: rescaling method
            minmaxFile: file to store statistics
            saveDir: directory for output
            
        Returns:
            Preprocessed data matrix, obsKeys, statistics dict
        """
        if not os.path.exists(saveDir + '/dataDetails/'):
            os.makedirs(saveDir + '/dataDetails/')

        prep_log = []

        # Log transform (for positive-only features)
        if logFlag:
            print("Applying log transform to non-negative features...")
            for feat in range(d.shape[1]):
                if d[:, feat].min() >= 0:
                    d[:, feat] = np.log(d[:, feat] + np.finfo(float).eps)
            d = d.astype(np.float32)
            prep_log.append("Log transform applied to non-negative features.")

        # Compute statistics
        dMean = np.mean(d, axis=0)
        dStd = np.std(d, axis=0)
        dMin = np.min(d, axis=0)
        dMax = np.max(d, axis=0)

        # Mean subtraction
        if meanSubtractionFlag:
            print("Subtracting mean...")
            d = d - dMean
            prep_log.append("Mean subtraction applied.")

        # Scaling
        if scalingFlag:
            print(f"Scaling with method: {scaling}")
            if scaling == 'standard':
                d = (d - dMean) / (dStd + np.finfo(float).eps)
            elif scaling == 'minmax':
                dRange = dMax - dMin
                d = (d - dMin) / (dRange + np.finfo(float).eps)
            elif scaling == 'robust':
                q25 = np.percentile(d, 25, axis=0)
                q75 = np.percentile(d, 75, axis=0)
                iqr = q75 - q25
                d = (d - np.median(d, axis=0)) / (iqr + np.finfo(float).eps)
            d = d.astype(np.float32)
            prep_log.append(f"Scaling applied ({scaling}).")

        # PCA
        if pcaFlag:
            print("Applying PCA...")
            pca = PCA(n_components=0.95)  # Keep 95% variance
            d = pca.fit_transform(d).astype(np.float32)
            prep_log.append(f"PCA applied. Components: {pca.n_components_}")

        # Whitening (ZCA-like)
        if whitenFlag:
            print("Applying whitening...")
            cov_matrix = np.cov(d.T)
            U, S, Vt = np.linalg.svd(cov_matrix)
            d = d @ U @ np.diag(1.0 / np.sqrt(S + 1e-5)) @ U.T
            d = d.astype(np.float32)
            prep_log.append("Whitening applied.")

        # Rescaling
        if rescalingFlag:
            print(f"Rescaling with method: {rescaling}")
            if rescaling == 'standard':
                dMean_new = np.mean(d, axis=0)
                dStd_new = np.std(d, axis=0)
                d = (d - dMean_new) / (dStd_new + np.finfo(float).eps)
            elif rescaling == 'minmax':
                dMin_new = np.min(d, axis=0)
                dMax_new = np.max(d, axis=0)
                dRange_new = dMax_new - dMin_new
                d = (d - dMin_new) / (dRange_new + np.finfo(float).eps)
            d = d.astype(np.float32)
            prep_log.append(f"Rescaling applied ({rescaling}).")

        # Save statistics
        stats = {
            'original_mean': dMean,
            'original_std': dStd,
            'original_min': dMin,
            'original_max': dMax,
        }

        if minmaxFile:
            with open(saveDir + '/dataDetails/' + minmaxFile, 'wb') as f:
                pickle.dump(stats, f)

        # Write preprocessing log
        with open(saveDir + '/dataDetails/preprocDetails.txt', 'w') as f:
            f.write("Preprocessing details:\n")
            for log_entry in prep_log:
                f.write(f"  - {log_entry}\n")
            f.write(f"\nFinal data shape: {d.shape}\n")
            f.write(f"Final data dtype: {d.dtype}\n")
            f.write(f"Final data range: [{d.min():.6f}, {d.max():.6f}]\n")

        return d, obsKeys, stats

    def normalizeData(self, d, norm_type='l2'):
        """
        Normalize data by rows or columns.
        
        Args:
            d: data matrix
            norm_type: 'l2' for L2 norm, 'l1' for L1 norm, 'max' for max norm
            
        Returns:
            Normalized data
        """
        if norm_type == 'l2':
            norms = np.linalg.norm(d, axis=1, keepdims=True)
            return d / (norms + np.finfo(float).eps)
        elif norm_type == 'l1':
            norms = np.linalg.norm(d, axis=1, ord=1, keepdims=True)
            return d / (norms + np.finfo(float).eps)
        elif norm_type == 'max':
            maxs = np.max(np.abs(d), axis=1, keepdims=True)
            return d / (maxs + np.finfo(float).eps)
        else:
            return d


def _load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if 'd' not in data:
        raise ValueError(f"Input NPZ does not contain required key 'd': {npz_path}")
    d = data['d']
    obs = data['epochsLinked'] if 'epochsLinked' in data else None
    epoch_time = data['epochTime'] if 'epochTime' in data else None
    return d, obs, epoch_time


def _save_npz(npz_path, d, obs, epoch_time):
    out = {'d': d.astype(np.float32)}
    if obs is not None:
        out['epochsLinked'] = obs
    if epoch_time is not None:
        out['epochTime'] = epoch_time
    np.savez_compressed(npz_path, **out)


def _apply_stats_only(d, stats, scaling, mean_subtraction):
    d_out = d.astype(np.float64, copy=True)
    eps = np.finfo(float).eps

    if mean_subtraction:
        if 'original_mean' not in stats:
            raise ValueError("Stats file missing 'original_mean' for mean subtraction.")
        d_out = d_out - stats['original_mean']

    if scaling == 'standard':
        if 'original_mean' not in stats or 'original_std' not in stats:
            raise ValueError("Stats file missing standard-scaling fields.")
        d_out = (d_out - stats['original_mean']) / (stats['original_std'] + eps)
    elif scaling == 'minmax':
        if 'original_min' not in stats or 'original_max' not in stats:
            raise ValueError("Stats file missing minmax-scaling fields.")
        d_range = stats['original_max'] - stats['original_min']
        d_out = (d_out - stats['original_min']) / (d_range + eps)
    elif scaling == 'robust':
        raise ValueError(
            "Transform mode does not support robust scaling with current saved stats. "
            "Use fit mode on the target dataset or extend saved stats to include robust parameters."
        )
    elif scaling == 'none':
        pass
    else:
        raise ValueError(f"Unknown scaling method: {scaling}")

    return d_out.astype(np.float32)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Preprocess NPZ data for mcRBM (fit or transform modes).'
    )
    parser.add_argument(
        '--mode',
        choices=['fit', 'transform'],
        default='fit',
        help='fit: compute stats and preprocess input; transform: apply existing stats.'
    )
    parser.add_argument(
        '--input-npz',
        default=CONFIG['PATHS']['input_npz'],
        help='Path to input NPZ (must contain key d). Defaults to CONFIG[PATHS][input_npz].'
    )
    parser.add_argument(
        '--output-npz',
        default=CONFIG['PATHS']['output_npz'],
        help='Path to output NPZ. Defaults to CONFIG[PATHS][output_npz].'
    )
    parser.add_argument(
        '--save-dir',
        default=CONFIG['PATHS']['save_dir'],
        help='Output directory for dataDetails logs/stats in fit mode.'
    )
    parser.add_argument(
        '--stats-file',
        default=CONFIG['PATHS']['stats_file'],
        help='Stats filename used in fit mode and read in transform mode.'
    )

    parser.add_argument('--log', action='store_true', help='Apply log transform to non-negative features.')
    parser.add_argument(
        '--mean-subtraction',
        action='store_true',
        help='Subtract original feature means before scaling.'
    )
    parser.add_argument(
        '--scaling',
        choices=['none', 'standard', 'minmax', 'robust'],
        default='standard',
        help='Scaling method.'
    )
    parser.add_argument('--pca', action='store_true', help='Apply PCA (95%% explained variance).')
    parser.add_argument('--whiten', action='store_true', help='Apply whitening.')
    parser.add_argument('--rescale', action='store_true', help='Apply post-processing rescaling.')
    parser.add_argument(
        '--rescaling',
        choices=['standard', 'minmax'],
        default='standard',
        help='Rescaling method when --rescale is enabled.'
    )
    return parser


def main():
    args = build_parser().parse_args()
    preproc = DataPreproc()

    if not args.input_npz or not args.output_npz:
        raise ValueError(
            "Missing input/output NPZ path. Set CONFIG['PATHS'] values or pass "
            "--input-npz and --output-npz."
        )

    d, obs, epoch_time = _load_npz(args.input_npz)
    print(f"Loaded input: {args.input_npz}")
    print(f"Input shape: {d.shape}, dtype: {d.dtype}")

    if args.mode == 'fit':
        scaling_flag = args.scaling != 'none'
        d_proc, obs_proc, _stats = preproc.preprocAndScaleData(
            d=d.astype(np.float64),
            obsKeys=obs,
            logFlag=args.log,
            meanSubtractionFlag=args.mean_subtraction,
            scalingFlag=scaling_flag,
            scaling=args.scaling,
            pcaFlag=args.pca,
            whitenFlag=args.whiten,
            rescalingFlag=args.rescale,
            rescaling=args.rescaling,
            minmaxFile=args.stats_file,
            saveDir=args.save_dir,
        )
    else:
        stats_path = os.path.join(args.save_dir, 'dataDetails', args.stats_file)
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found for transform mode: {stats_path}")
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        d_proc = _apply_stats_only(
            d,
            stats,
            scaling=args.scaling,
            mean_subtraction=args.mean_subtraction,
        )
        obs_proc = obs

    _save_npz(args.output_npz, d_proc, obs_proc, epoch_time)
    print(f"Saved output: {args.output_npz}")
    print(f"Output shape: {d_proc.shape}, dtype: {d_proc.dtype}")


if __name__ == '__main__':
    main()
