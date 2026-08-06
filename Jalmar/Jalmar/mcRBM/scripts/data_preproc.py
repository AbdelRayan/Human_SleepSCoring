"""
Optional preprocessing helpers for mcRBM.

This module is only needed if you want mcRBM-local scaling, PCA, whitening, or
batch trimming before training on features.
"""

import numpy as np
from sklearn.decomposition import PCA
import pickle
import os
from scipy.io import loadmat, savemat


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
