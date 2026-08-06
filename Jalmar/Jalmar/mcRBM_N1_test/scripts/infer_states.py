"""
Inference and latent-state extraction for mcRBM.

This module loads trained mcRBM weights and analyzes the latent states found
in HDF5-derived sleep feature space.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import pickle
from array_backend import get_backend


class mcRBMInference:
    """
    Compute and analyze latent states from trained mcRBM model.
    """

    def __init__(self, modelDir, expDir, modelFile='ws_final.mat', use_gpu=False, gpu_id=0):
        """
        Initialize inference engine.
        
        Args:
            modelDir: Directory containing trained weights
            expDir: Experiment directory
            modelFile: Filename of trained model weights
            use_gpu: Whether to use GPU (CuPy)
            gpu_id: GPU device ID if using CuPy
        """
        self.modelDir = modelDir
        self.expDir = expDir
        self.modelFile = modelFile
        self.saveDir = expDir
        
        # Initialize backend
        self.backend = get_backend(use_gpu=use_gpu, gpu_id=gpu_id, verbose=True)
        self.xp = self.backend.xp
        
        self.loadModel()

    def loadModel(self):
        """Load trained mcRBM weights."""
        model_path = os.path.join(self.modelDir, self.modelFile)
        print(f"Loading model from: {model_path}")
        
        model_data = loadmat(model_path)
        # Load as NumPy first (scipy returns NumPy), then move to backend
        self.VF = self.xp.asarray(model_data['VF'].astype(np.float32))
        self.FH = self.xp.asarray(model_data['FH'].astype(np.float32))
        self.bias_cov = self.xp.asarray(model_data['bias_cov'].astype(np.float32).reshape(-1, 1))
        self.bias_vis = self.xp.asarray(model_data['bias_vis'].astype(np.float32).reshape(-1, 1))
        self.w_mean = self.xp.asarray(model_data['w_mean'].astype(np.float32))
        self.bias_mean = self.xp.asarray(model_data['bias_mean'].astype(np.float32).reshape(-1, 1))
        
        print(f"Model loaded. VF shape: {self.VF.shape}, FH shape: {self.FH.shape}")

    def loadData(self, statesFilePath, statesFile):
        """
        Load visible data and pre-computed states.
        
        Args:
            statesFilePath: Path to states directory
            statesFile: States filename
        """
        os.chdir(self.saveDir)
        print(f"Working directory: {os.getcwd()}")
        
        # Load visible data (NumPy first, then move to backend)
        visData = np.load('visData.npz')
        self.d = self.xp.asarray(visData['data'].astype(np.float32))
        self.obsKeys = visData['obsKeys'].astype(int)
        
        # Load states if they exist
        try:
            states_data = loadmat(os.path.join(statesFilePath, statesFile))
            self.states = states_data.get('states', None)
            print(f"States loaded. Shape: {self.states.shape if self.states is not None else 'None'}")
        except:
            print("No pre-computed states found.")
            self.states = None

    def hidden_activation(self):
        """
        Compute hidden unit activations for all visible data.
        
        Returns:
            p_hc: Covariance hidden activations (samples × hidden_cov)
            p_hm: Mean hidden activations (samples × hidden_mean)
        """
        xp = self.xp
        num_vis = self.d.shape[0]
        num_samples = self.d.shape[0]
        
        # Normalize data
        lengthsq = (self.d ** 2).sum(axis=1, keepdims=True) / num_vis + 0.5
        length = xp.sqrt(lengthsq)
        normcoeff = 1.0 / length
        normdata = self.d * normcoeff
        
        # Covariance hidden units
        feat = xp.dot(normdata, self.VF)  # samples × factors
        featsq = feat ** 2
        t1 = xp.dot(featsq, self.FH.T) * (-0.5)  # samples × hidden_cov
        t1 = t1 + self.bias_cov.T  # broadcast
        p_hc = 1.0 / (1.0 + xp.exp(-t1))  # Sigmoid
        
        # Mean hidden units
        feat_mean = xp.dot(self.d, self.w_mean) + self.bias_mean.T  # samples × hidden_mean
        p_hm = 1.0 / (1.0 + xp.exp(-feat_mean))  # Sigmoid
        
        return p_hc, p_hm

    def computeStates(self, saveProbabilities=True, saveBinary=True):
        """
        Compute and save latent states.
        
        Args:
            saveProbabilities: Save probability activations as images
            saveBinary: Save binarized states (threshold 0.5)
            
        Returns:
            Dictionary with state information
        """
        print("Computing latent states...")
        p_hc, p_hm = self.hidden_activation()
        self.p_all = self.xp.concatenate((p_hc, p_hm), axis=1)
        
        # Setup directories
        for dirname in ['analysis', 'analysis/activations', 'analysis/binary']:
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
        
        # Save activations (convert to NumPy if on GPU)
        p_hc_np = self.backend.to_numpy(p_hc)
        p_hm_np = self.backend.to_numpy(p_hm)
        p_all_np = self.backend.to_numpy(self.p_all)
        
        np.savez('analysis/activations.npz', p_hc=p_hc_np, p_hm=p_hm_np, p_all=p_all_np)
        
        # Binarize at 0.5 threshold (work on NumPy for CPU efficiency)
        self.binary_latentActivation = (p_all_np >= 0.5).astype(int)
        
        # Save probability images
        if saveProbabilities:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            axes[0].imshow(p_hc.T, aspect='auto', cmap='viridis')
            axes[0].set_title('Covariance Hidden Unit Activations')
            axes[0].set_xlabel('Sample')
            axes[0].set_ylabel('Hidden Unit')
            
            axes[1].imshow(p_hm.T, aspect='auto', cmap='viridis')
            axes[1].set_title('Mean Hidden Unit Activations')
            axes[1].set_xlabel('Sample')
            axes[1].set_ylabel('Hidden Unit')
            
            plt.tight_layout()
            plt.savefig('analysis/hidden_activations.png', dpi=150)
            plt.close()
        
        # Save binary activations
        if saveBinary:
            fig = plt.figure(figsize=(15, 20))
            plt.imshow(self.binary_latentActivation.T, cmap='binary', aspect='auto')
            plt.title('Binary Latent Activations (threshold=0.5)')
            plt.xlabel('Sample')
            plt.ylabel('Hidden Unit')
            plt.tight_layout()
            plt.savefig('analysis/binary_activations.png', dpi=150)
            plt.close()
        
        # Find unique binary patterns
        str_repr = np.array([''.join(map(str, row)) for row in self.binary_latentActivation])
        unique_bin, unique_indices, inverse_indices = np.unique(str_repr, return_index=True, return_inverse=True)
        
        unique_states = self.binary_latentActivation[unique_indices]
        unique_counts = np.array([np.sum(inverse_indices == i) for i in range(len(unique_indices))])
        p_unique = self.p_all[unique_indices]
        
        # Map states to obs labels if available
        state_info = np.zeros((len(unique_states), len(unique_states[0]) + 3))
        state_info[:, 0] = np.arange(1, len(unique_states) + 1)
        state_info[:, 1] = unique_counts
        state_info[:, 2] = 0  # Placeholder for modal obs label
        state_info[:, 3:] = unique_states
        
        # Determine most common obs label for each state
        if len(self.obsKeys) >= len(self.binary_latentActivation):
            for i in range(len(unique_states)):
                row_indices = np.where((self.binary_latentActivation == unique_states[i]).all(axis=1))[0]
                if len(row_indices) > 0:
                    labels_in_state = self.obsKeys[row_indices]
                    if len(labels_in_state) > 0:
                        state_info[i, 2] = np.argmax(np.bincount(labels_in_state.astype(int)))
        
        # Save state info
        self.state_info = state_info
        np.save('analysis/state_info.npy', state_info)
        savemat('analysis/state_info.mat', {'state_info': state_info})
        
        print(f"Found {len(unique_states)} unique latent states")
        
        return {
            'p_hc': p_hc,
            'p_hm': p_hm,
            'p_all': self.p_all,
            'binary': self.binary_latentActivation,
            'unique_states': unique_states,
            'unique_counts': unique_counts,
            'state_info': state_info
        }

    def analyzeStates(self, saveResults=True):
        """
        Analyze latent states and their characteristics.
        
        Args:
            saveResults: Save analysis results
            
        Returns:
            Dictionary with analysis results
        """
        print("Analyzing latent states...")
        
        # Compute statistics per state
        analysis_results = {}
        
        # Find unique states
        str_repr = np.array([''.join(map(str, row)) for row in self.binary_latentActivation])
        unique_bin, unique_indices, inverse_indices = np.unique(str_repr, return_index=True, return_inverse=True)
        
        num_unique_states = len(unique_indices)
        
        # Analyze features per state
        state_feature_means = np.zeros((num_unique_states, self.d.shape[1]))
        state_feature_stds = np.zeros_like(state_feature_means)
        
        for state_id in range(num_unique_states):
            mask = (inverse_indices == state_id)
            state_data = self.d[mask, :]
            state_feature_means[state_id, :] = state_data.mean(axis=0)
            state_feature_stds[state_id, :] = state_data.std(axis=0)
        
        analysis_results['state_feature_means'] = state_feature_means
        analysis_results['state_feature_stds'] = state_feature_stds
        
        # Plot state-specific features
        fig, axes = plt.subplots(num_unique_states, 1, figsize=(12, 3 * num_unique_states))
        if num_unique_states == 1:
            axes = [axes]
        
        for state_id in range(num_unique_states):
            axes[state_id].bar(range(self.d.shape[1]), state_feature_means[state_id, :],
                             yerr=state_feature_stds[state_id, :], capsize=5)
            axes[state_id].set_title(f'State {state_id}: Feature Means')
            axes[state_id].set_ylabel('Feature Value')
        
        if saveResults:
            plt.tight_layout()
            plt.savefig('analysis/state_feature_analysis.png', dpi=150)
            plt.close()
            
            np.save('analysis/state_feature_means.npy', state_feature_means)
            np.save('analysis/state_feature_stds.npy', state_feature_stds)
        
        analysis_results['num_states'] = num_unique_states
        
        return analysis_results

    def computeTransitionMatrix(self):
        """
        Compute transition matrix between latent states (if temporal structure exists).
        
        Returns:
            Transition matrix (num_states × num_states)
        """
        print("Computing state transitions...")
        
        # Find unique states
        str_repr = np.array([''.join(map(str, row)) for row in self.binary_latentActivation])
        unique_bin, unique_indices, inverse_indices = np.unique(str_repr, return_index=True, return_inverse=True)
        
        num_unique_states = len(unique_indices)
        transition_matrix = np.zeros((num_unique_states, num_unique_states))
        
        # Count transitions
        for i in range(len(inverse_indices) - 1):
            from_state = inverse_indices[i]
            to_state = inverse_indices[i + 1]
            transition_matrix[from_state, to_state] += 1
        
        # Normalize by row sums
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix

    def visualizeStateSequence(self, maxSamples=5000, saveResults=True):
        """
        Visualize the sequence of latent states over time.
        
        Args:
            maxSamples: Maximum number of samples to plot
            saveResults: Save visualization
        """
        print("Visualizing state sequence...")
        
        # Find unique states
        str_repr = np.array([''.join(map(str, row)) for row in self.binary_latentActivation])
        unique_bin, unique_indices, inverse_indices = np.unique(str_repr, return_index=True, return_inverse=True)
        
        plot_indices = np.arange(min(maxSamples, len(inverse_indices)))
        
        fig = plt.figure(figsize=(15, 3))
        plt.scatter(plot_indices, inverse_indices[plot_indices], alpha=0.5, s=1)
        plt.xlabel('Sample Index')
        plt.ylabel('Latent State ID')
        plt.title('Latent State Sequence Over Time')
        
        if saveResults:
            plt.tight_layout()
            plt.savefig('analysis/state_sequence.png', dpi=150)
            plt.close()
        
        return inverse_indices
