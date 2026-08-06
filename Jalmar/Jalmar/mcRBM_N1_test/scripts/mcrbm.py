"""
Mean-Covariance Restricted Boltzmann Machine (mcRBM) implementation.
Based on "Modeling Pixel Means and Covariances Using Factorized Third-Order Boltzmann Machines"
by Marc'Aurelio Ranzato.

This module trains mcRBM on the HDF5-derived feature sets and supports
optional NumPy/CuPy execution via the backend abstraction.
"""

import sys
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from numpy.random import RandomState
from scipy.io import loadmat, savemat
from configparser import ConfigParser
from datetime import datetime
from data_preproc import DataPreproc
from array_backend import get_backend


class mcRBM:
    """
    Mean-Covariance Restricted Boltzmann Machine (mcRBM).
    Trained with Hybrid Monte Carlo (HMC) sampling.
    """

    def __init__(self, refDir, expConfigFilename, modelConfigFilename):
        """
        Initialize mcRBM with configuration files.
        
        Args:
            refDir: Reference directory containing config files
            expConfigFilename: Experiment config filename
            modelConfigFilename: Model config filename
        """
        self.refDir = refDir
        self.expConfigFilename = os.path.join(refDir, expConfigFilename)
        self.modelConfigFilename = os.path.join(refDir, modelConfigFilename)
        self.dpp = DataPreproc()
        
        self.loadExpConfig()
        self.loadModelConfig()

    def loadExpConfig(self):
        """Load experiment configuration."""
        config = ConfigParser()
        config.read(self.expConfigFilename)

        self.npRandSeed = config.getint('PARAMETERS', 'npRandSeed')
        self.npRandState = config.getint('PARAMETERS', 'npRandState')

        self.dataDir = config.get('EXP_DETAILS', 'dsetDir')
        self.expsDir = config.get('EXP_DETAILS', 'expsDir')
        self.expName = config.get('EXP_DETAILS', 'expID')
        self.dSetName = config.get('EXP_DETAILS', 'dSetName')

        self.logFlag = config.getboolean('EXP_DETAILS', 'logFlag')
        self.meanSubtructionFlag = config.getboolean('EXP_DETAILS', 'meanSubtructionFlag')
        self.scaleFlag = config.getboolean('EXP_DETAILS', 'scaleFlag')
        self.scaling = config.get('EXP_DETAILS', 'scaling')
        self.doPCA = config.getboolean('EXP_DETAILS', 'doPCA')
        self.whitenFlag = config.getboolean('EXP_DETAILS', 'whitenFlag')
        self.rescaleFlag = config.getboolean('EXP_DETAILS', 'rescaleFlag')
        self.rescaling = config.get('EXP_DETAILS', 'rescaling')

        self.dataFilename = os.path.join(self.dataDir, self.dSetName)
        self.saveDir = os.path.join(self.expsDir, self.expName)

        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)

    def loadModelConfig(self):
        """Load model configuration."""
        config = ConfigParser()
        config.read(self.modelConfigFilename)

        self.verbose = config.getint('VERBOSITY', 'verbose')

        self.use_gpu = config.getboolean('COMPUTE_BACKEND', 'use_gpu')
        self.gpu_id = config.getint('COMPUTE_BACKEND', 'gpu_id')
        
        # Initialize backend (NumPy or CuPy)
        self.backend = get_backend(use_gpu=self.use_gpu, gpu_id=self.gpu_id, verbose=True)
        self.xp = self.backend.xp

        self.num_epochs = config.getint('MAIN_PARAMETER_SETTING', 'num_epochs')
        self.batch_size = config.getint('MAIN_PARAMETER_SETTING', 'batch_size')
        self.startFH = config.getint('MAIN_PARAMETER_SETTING', 'startFH')
        self.startwd = config.getint('MAIN_PARAMETER_SETTING', 'startwd')
        self.doPCD = config.getint('MAIN_PARAMETER_SETTING', 'doPCD')

        self.num_fac = config.getint('MODEL_PARAMETER_SETTING', 'num_fac')
        self.num_hid_cov = config.getint('MODEL_PARAMETER_SETTING', 'num_hid_cov')
        self.num_hid_mean = config.getint('MODEL_PARAMETER_SETTING', 'num_hid_mean')
        self.apply_mask = config.getint('MODEL_PARAMETER_SETTING', 'apply_mask')
        
        self.epsilon = config.getfloat('OPTIMIZER_PARAMETERS', 'epsilon')
        self.weightcost_final = config.getfloat('OPTIMIZER_PARAMETERS', 'weightcost_final')
        # Optional multipliers let us tune per-parameter learning rates without editing code.
        self.epsilonVF_mult = config.getfloat('OPTIMIZER_PARAMETERS', 'epsilonVF_mult', fallback=2.0)
        self.epsilonFH_mult = config.getfloat('OPTIMIZER_PARAMETERS', 'epsilonFH_mult', fallback=0.02)
        self.epsilonb_mult = config.getfloat('OPTIMIZER_PARAMETERS', 'epsilonb_mult', fallback=0.02)
        self.epsilonw_mean_mult = config.getfloat('OPTIMIZER_PARAMETERS', 'epsilonw_mean_mult', fallback=0.2)
        self.epsilonb_mean_mult = config.getfloat('OPTIMIZER_PARAMETERS', 'epsilonb_mean_mult', fallback=0.1)
        self.hmc_step_nr = config.getint('HMC_PARAMETERS', 'hmc_step_nr')
        self.hmc_target_ave_rej = config.getfloat('HMC_PARAMETERS', 'hmc_target_ave_rej')

    def loadData(self):
        """Load training data from .npz or .mat file."""
        if not os.path.exists(os.path.join(self.saveDir, 'dataDetails')):
            os.makedirs(os.path.join(self.saveDir, 'dataDetails'))

        if self.dataFilename.endswith('.npz'):
            dLoad = np.load(self.dataFilename)
            self.d = dLoad['d']
            self.obsKeys = dLoad.get('epochsLinked', np.zeros(self.d.shape[0]))
            self.epochTime = dLoad.get('epochTime', np.zeros((self.d.shape[0], 1)))
        elif self.dataFilename.endswith('.mat'):
            dLoad = loadmat(self.dataFilename)
            self.d = dLoad['d']
            self.obsKeys = dLoad.get('epochsLinked', np.zeros(self.d.shape[0]))
            self.epochTime = dLoad.get('epochTime', np.zeros((self.d.shape[0], 1)))
        else:
            raise ValueError("Unrecognized data file format. Must be .npz or .mat")

        self.d = np.array(self.d, dtype=np.float32)
        self.obsKeys = np.array(self.obsKeys, dtype=np.float32).flatten()
        self.epochTime = np.array(self.epochTime, dtype=np.float32)

        print(f"Initial data shape: {self.d.shape}")

        # Save initial data statistics
        with open(os.path.join(self.saveDir, 'dataDetails', 'initialData.txt'), 'w') as f:
            f.write(f"\nModeling: {self.dataFilename}\n")
            f.write(f"Dataset size: {self.d.shape}\n")
            f.write(f"Dataset dtype: {self.d.dtype}\n")
            f.write(f"d_min: {np.min(self.d, axis=0)}\n")
            f.write(f"d_max: {np.max(self.d, axis=0)}\n")
            f.write(f"d_mean: {np.mean(self.d, axis=0)}\n")
            f.write(f"d_std: {np.std(self.d, axis=0)}\n")

    def compute_energy_mcRBM(self, data, normdata, vel, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, small=0.5):
        """
        Compute energy function of mcRBM.
        
        Args:
            data: visible data (features × samples)
            normdata: normalized data
            vel: velocity for HMC
            VF: factor weights (features × factors)
            FH: factor-to-hidden mapping (factors × hidden_cov)
            bias_cov: hidden covariance biases
            bias_vis: visible biases
            w_mean: mean weights (features × hidden_mean)
            bias_mean: mean biases
            small: small constant to prevent division by zero
            
        Returns:
            Energy per sample (shape: samples,)
        """
        xp = self.xp
        num_vis = data.shape[0]
        
        # Normalize data
        t6 = data ** 2
        lengthsq = t6.sum(axis=0) / num_vis + small
        length = xp.sqrt(lengthsq)
        normcoeff = 1.0 / length
        
        energy = 0.5 * (data ** 2).sum(axis=0)
        
        # Covariance contribution
        feat = xp.dot(VF.T, normdata)
        featsq = feat ** 2
        t1 = xp.dot(FH.T, featsq) * (-0.5)
        t1 = t1 + bias_cov
        t2 = xp.log1p(xp.exp(t1))
        energy = energy + t2.sum(axis=0)
        
        # Mean contribution
        feat_mean = xp.dot(w_mean.T, data) + bias_mean
        feat_mean = -xp.log1p(xp.exp(feat_mean))
        energy = energy + feat_mean.sum(axis=0)
        
        # Visible bias
        energy = energy - (data * bias_vis).sum(axis=0)
        
        # Kinetic energy
        energy = energy + 0.5 * (vel ** 2).sum(axis=0)
        
        return energy

    def compute_gradient_mcRBM(self, data, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, small=0.5):
        """
        Compute gradient of free energy with respect to visible data.
        
        Args:
            data: visible data (features × samples)
            normdata: normalized data
            VF: factor weights
            FH: factor-to-hidden mapping
            bias_cov: hidden covariance biases
            bias_vis: visible biases
            w_mean: mean weights
            bias_mean: mean biases
            small: small constant
            
        Returns:
            Gradient w.r.t. data (features × samples)
        """
        xp = self.xp
        num_vis = data.shape[0]
        num_samples = data.shape[1]
        
        # Normalize data
        t6 = data ** 2
        lengthsq = t6.sum(axis=0) / num_vis + small
        length = xp.sqrt(lengthsq)
        normcoeff = 1.0 / length
        normdata = data * normcoeff
        
        # Forward pass - covariance part
        feat = xp.dot(VF.T, normdata)
        featsq = feat ** 2
        t1 = xp.dot(FH.T, featsq) * (-0.5) + bias_cov
        t2 = 1.0 / (1.0 + xp.exp(-t1))  # sigmoid
        
        t3 = xp.dot(FH, t2) * feat
        normgradient = xp.dot(VF, t3)
        
        # Backprop through normalization
        normcoeff2 = length * lengthsq
        normcoeff2 = 1.0 / (normcoeff2 + small)
        
        gradient = normgradient * data
        t4 = -gradient.sum(axis=0) / num_vis
        gradient = gradient + data * t4
        gradient = gradient * lengthsq
        gradient = gradient * normcoeff2
        
        # Add quadratic term
        gradient = gradient + data
        
        # Add visible bias term
        gradient = gradient - bias_vis
        
        # Add mean contribution
        feat_mean = xp.dot(w_mean.T, data) + bias_mean
        feat_mean = 1.0 / (1.0 + xp.exp(-feat_mean))
        gradient = gradient - xp.dot(w_mean, feat_mean)
        
        return gradient

    def draw_HMC_samples(self, data, negdata, vel, gradient, VF, FH, bias_cov, bias_vis, w_mean, bias_mean,
                        hmc_step, hmc_step_nr, hmc_ave_rej, hmc_target_ave_rej, small=0.5):
        """
        Hybrid Monte Carlo sampler for generating negative samples.
        
        Args:
            data: positive data samples
            negdata: current negative samples
            vel: velocity
            gradient: gradient storage
            VF, FH, bias_cov, bias_vis, w_mean, bias_mean: model parameters
            hmc_step: HMC step size
            hmc_step_nr: number of leapfrog steps
            hmc_ave_rej: average rejection rate
            hmc_target_ave_rej: target rejection rate
            small: small constant
            
        Returns:
            Updated negdata, hmc_step, hmc_ave_rej
        """
        xp = self.xp
        num_vis = data.shape[0]
        batch_size = data.shape[1]
        
        # Initialize velocity
        vel = xp.random.randn(*vel.shape).astype(xp.float32)
        # Start HMC from the provided negative samples (supports PCD chains).
        negdata_copy = negdata.copy()
        
        # Compute initial energy and gradient
        normdata = negdata_copy / (xp.linalg.norm(negdata_copy, axis=0, keepdims=True) + small)
        old_energy = self.compute_energy_mcRBM(negdata_copy, normdata, vel, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, small)
        gradient = self.compute_gradient_mcRBM(negdata_copy, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, small)
        
        # HMC leapfrog integrator
        vel = -0.5 * hmc_step * gradient
        negdata_copy = negdata_copy + hmc_step * vel
        
        for ss in range(hmc_step_nr - 1):
            gradient = self.compute_gradient_mcRBM(negdata_copy, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, small)
            vel = vel - hmc_step * gradient
            negdata_copy = negdata_copy + hmc_step * vel
        
        # Final half-step
        gradient = self.compute_gradient_mcRBM(negdata_copy, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, small)
        vel = vel - 0.5 * hmc_step * gradient
        
        # Compute new energy
        normdata = negdata_copy / (xp.linalg.norm(negdata_copy, axis=0, keepdims=True) + small)
        new_energy = self.compute_energy_mcRBM(negdata_copy, normdata, vel, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, small)
        
        # Metropolis-Hastings acceptance
        energy_diff = old_energy - new_energy
        accept_prob = xp.exp(xp.clip(energy_diff, -500, 500))  # Clip to prevent overflow
        accept = xp.random.rand(batch_size) < accept_prob
        
        rej = float(xp.mean(~accept))
        hmc_ave_rej = 0.9 * hmc_ave_rej + 0.1 * rej
        
        # Update negdata only where accepted
        negdata = xp.where(accept, negdata_copy, negdata)
        
        # Adapt step size
        if hmc_ave_rej < hmc_target_ave_rej:
            hmc_step = min(hmc_step * 1.01, 0.25)
        else:
            hmc_step = max(hmc_step * 0.99, 0.001)
        
        return negdata, hmc_step, hmc_ave_rej

    def train(self):
        """Main training function."""
        xp = self.xp
        print("Loading data...")
        self.loadData()
        
        # Setup
        os.chdir(self.saveDir)
        np.random.seed(self.npRandSeed)
        prng = RandomState(self.npRandState)
        
        for dirname in ['plots', 'plots/energy', 'weights']:
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
        
        d = self.d.astype(np.float32)
        num_vis = d.shape[1]
        totnumcases = d.shape[0]
        
        # Permute data
        permIdx = prng.permutation(totnumcases)
        d = d[permIdx, :]
        
        # Save data info
        np.savez('visData.npz', data=d, obsKeys=self.obsKeys, epochTime=self.epochTime)
        with open('visData.txt', 'w') as f:
            f.write(f"\nDataset: {self.dataFilename}\n")
            f.write(f"visData size: {d.shape}\n")
            f.write(f"visData type: {d.dtype}\n")
            f.write(f"visData range: {np.max(d, axis=0) - np.min(d, axis=0)}\n")
        
        del self.d, self.obsKeys, self.epochTime
        
        # Batch processing
        num_batches = int(totnumcases / self.batch_size)
        print(f"Number of batches: {num_batches}")
        
        dev_dat = xp.array(d.T, dtype=xp.float32)  # VxP
        
        # Training parameters
        epsilon = self.epsilon
        epsilonVF = self.epsilonVF_mult * epsilon
        epsilonFH = self.epsilonFH_mult * epsilon
        epsilonb = self.epsilonb_mult * epsilon
        epsilonw_mean = self.epsilonw_mean_mult * epsilon
        epsilonb_mean = self.epsilonb_mean_mult * epsilon
        weightcost_final = self.weightcost_final
        
        # HMC settings
        hmc_step_nr = self.hmc_step_nr
        hmc_step = 0.01
        hmc_target_ave_rej = self.hmc_target_ave_rej
        hmc_ave_rej = hmc_target_ave_rej
        
        # Initialize weights
        VF = xp.array(0.02 * prng.randn(num_vis, self.num_fac), dtype=xp.float32)
        FH = xp.eye(self.num_fac, self.num_hid_cov, dtype=xp.float32)
        bias_cov = 2.0 * xp.ones((self.num_hid_cov, 1), dtype=xp.float32)
        bias_vis = xp.zeros((num_vis, 1), dtype=xp.float32)
        w_mean = xp.array(0.05 * prng.randn(num_vis, self.num_hid_mean), dtype=xp.float32)
        bias_mean = -2.0 * xp.ones((self.num_hid_mean, 1), dtype=xp.float32)
        
        # Initialize increments
        VFinc = xp.zeros_like(VF)
        FHinc = xp.zeros_like(FH)
        bias_covinc = xp.zeros_like(bias_cov)
        bias_visinc = xp.zeros_like(bias_vis)
        w_meaninc = xp.zeros_like(w_mean)
        bias_meaninc = xp.zeros_like(bias_mean)
        
        # Dropout settings
        use_hidden_dropout = 1
        dropout_cov = 0.3
        dropout_mean = 0.15
        
        small = 0.5
        normVF = 1.0
        
        # Energy tracking
        meanEnergy = xp.zeros(self.num_epochs)
        minEnergy = xp.zeros(self.num_epochs)
        maxEnergy = xp.zeros(self.num_epochs)
        
        # Persistent negative chain for PCD.
        persistent_negdata = None
        
        # Figure for plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # Training loop
        print("Starting training...")
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}/{self.num_epochs}")
            
            # Learning rate annealing
            epsilonVFc = epsilonVF / max(1, epoch / 20)
            epsilonFHc = epsilonFH / max(1, epoch / 20)
            epsilonbc = epsilonb / max(1, epoch / 20)
            epsilonw_meanc = epsilonw_mean / max(1, epoch / 20)
            epsilonb_meanc = epsilonb_mean / max(1, epoch / 20)
            
            weightcost = weightcost_final
            
            if epoch <= self.startFH:
                epsilonFHc = 0
            if epoch <= self.startwd:
                weightcost = 0
            
            epoch_energy = []
            
            for batch in range(num_batches):
                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                data = dev_dat[:, start:end].copy()
                
                # Reset batch-local increments (avoid cross-batch accumulation bugs).
                VFinc = xp.zeros_like(VF)
                FHinc = xp.zeros_like(FH)
                bias_covinc = xp.zeros_like(bias_cov)
                bias_visinc = xp.zeros_like(bias_vis)
                w_meaninc = xp.zeros_like(w_mean)
                bias_meaninc = xp.zeros_like(bias_mean)

                # Dropout
                if use_hidden_dropout:
                    keep_cov = 1.0 - dropout_cov
                    keep_mean = 1.0 - dropout_mean
                    cov_drop = xp.random.binomial(1, keep_cov, size=(self.num_hid_cov, self.batch_size)).astype(xp.float32) / keep_cov
                    mean_drop = xp.random.binomial(1, keep_mean, size=(self.num_hid_mean, self.batch_size)).astype(xp.float32) / keep_mean
                else:
                    cov_drop = 1.0
                    mean_drop = 1.0
                
                # Normalize data
                lengthsq = (data ** 2).sum(axis=0) / num_vis + small
                length = xp.sqrt(lengthsq)
                normcoeff = 1.0 / length
                normdata = data * normcoeff
                
                # Positive phase
                feat = xp.dot(VF.T, normdata)
                featsq = feat ** 2
                t1 = xp.dot(FH.T, featsq) * (-0.5) + bias_cov
                t2 = 1.0 / (1.0 + xp.exp(-t1))
                t2 = t2 * cov_drop
                
                FHinc = xp.dot(featsq, t2.T)
                t3 = xp.dot(FH, t2) * feat
                VFinc = VFinc + xp.dot(normdata, t3.T)
                
                bias_covinc = -1 * t2.sum(axis=1, keepdims=True)
                bias_visinc = -1 * data.sum(axis=1, keepdims=True)
                
                # Mean part
                feat_mean = xp.dot(w_mean.T, data) + bias_mean
                feat_mean = 1.0 / (1.0 + xp.exp(-feat_mean))
                feat_mean = feat_mean * mean_drop * (-1)
                
                w_meaninc = xp.dot(data, feat_mean.T)
                bias_meaninc = feat_mean.sum(axis=1, keepdims=True)
                
                # HMC sampling
                if self.doPCD:
                    if persistent_negdata is None:
                        persistent_negdata = xp.random.randn(num_vis, self.batch_size).astype(xp.float32)
                    negdata = persistent_negdata.copy()
                else:
                    # CD: start each chain from current minibatch data.
                    negdata = data.copy()
                vel = xp.random.randn(num_vis, self.batch_size).astype(xp.float32)
                gradient = xp.zeros_like(data)
                
                negdata, hmc_step, hmc_ave_rej = self.draw_HMC_samples(
                    data, negdata, vel, gradient, VF, FH, bias_cov, bias_vis, w_mean, bias_mean,
                    hmc_step, hmc_step_nr, hmc_ave_rej, hmc_target_ave_rej, small
                )
                if self.doPCD:
                    persistent_negdata = negdata.copy()
                
                # Negative phase
                lengthsq = (negdata ** 2).sum(axis=0) / num_vis + small
                normcoeff = 1.0 / xp.sqrt(lengthsq)
                normdata_neg = negdata * normcoeff
                
                feat = xp.dot(VF.T, normdata_neg)
                featsq = feat ** 2
                t1 = xp.dot(FH.T, featsq) * (-0.5) + bias_cov
                t2 = 1.0 / (1.0 + xp.exp(-t1))
                t2 = t2 * cov_drop
                
                FHinc = FHinc - xp.dot(featsq, t2.T) * 0.5
                t3 = xp.dot(FH, t2) * feat
                VFinc = VFinc - xp.dot(normdata_neg, t3.T)
                
                bias_covinc = bias_covinc + t2.sum(axis=1, keepdims=True)
                bias_visinc = bias_visinc + negdata.sum(axis=1, keepdims=True)
                
                # Mean part
                feat_mean = xp.dot(w_mean.T, negdata) + bias_mean
                feat_mean = 1.0 / (1.0 + xp.exp(-feat_mean))
                feat_mean = feat_mean * mean_drop
                
                w_meaninc = w_meaninc + xp.dot(negdata, feat_mean.T)
                bias_meaninc = bias_meaninc + feat_mean.sum(axis=1, keepdims=True)
                
                # Parameter updates
                VFinc = VFinc + xp.sign(VF) * weightcost
                VF = VF - (epsilonVFc / self.batch_size) * VFinc
                
                # Normalize VF
                vf_norm = xp.linalg.norm(VF, axis=0, keepdims=True)
                normVF = 0.95 * normVF + 0.05 / self.num_fac * vf_norm.sum()
                VF = VF * (1.0 / (vf_norm + small)) * normVF
                
                bias_cov = bias_cov - (epsilonbc / self.batch_size) * bias_covinc
                bias_vis = bias_vis - (epsilonbc / self.batch_size) * bias_visinc
                
                if epoch > self.startFH:
                    FHinc = FHinc + xp.sign(FH) * weightcost
                    FH = FH - (epsilonFHc / self.batch_size) * FHinc
                    FH[FH < 0] = 0  # Ensure non-negative
                    FH = FH / (FH.sum(axis=0, keepdims=True) + small)  # Normalize
                
                w_meaninc = w_meaninc + xp.sign(w_mean) * weightcost
                w_mean = w_mean - (epsilonw_meanc / self.batch_size) * w_meaninc
                bias_mean = bias_mean - (epsilonb_meanc / self.batch_size) * bias_meaninc
                
                # Track energy
                normdata = data * (1.0 / (xp.sqrt((data ** 2).sum(axis=0) / num_vis + small)))
                energy = self.compute_energy_mcRBM(data, normdata, vel, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, small)
                epoch_energy.extend(energy)
            
            epoch_energy = xp.array(epoch_energy)
            meanEnergy[epoch] = xp.mean(epoch_energy)
            minEnergy[epoch] = xp.min(epoch_energy)
            maxEnergy[epoch] = xp.max(epoch_energy)
            
            if self.verbose == 1:
                print(f"Mean Energy: {meanEnergy[epoch]:.4f}, Min: {minEnergy[epoch]:.4f}, Max: {maxEnergy[epoch]:.4f}")
            
            # Synchronize GPU if using GPU backend
            self.backend.synchronize()
            
            # Save weights and plots
            if epoch % 100 == 0 or epoch == self.num_epochs - 1:
                # Convert weights to NumPy for saving (if on GPU)
                weights_to_save = {
                    'VF': self.backend.to_numpy(VF),
                    'FH': self.backend.to_numpy(FH),
                    'bias_cov': self.backend.to_numpy(bias_cov),
                    'bias_vis': self.backend.to_numpy(bias_vis),
                    'w_mean': self.backend.to_numpy(w_mean),
                    'bias_mean': self.backend.to_numpy(bias_mean),
                    'epoch': epoch
                }
                savemat(f"./weights/ws_epoch{epoch}.mat", weights_to_save)
                
                ax.cla()
                ax.plot(self.backend.to_numpy(meanEnergy[:epoch+1]), label='Mean Energy', marker='o')
                ax.plot(self.backend.to_numpy(maxEnergy[:epoch+1]), label='Max Energy', marker='s')
                ax.plot(self.backend.to_numpy(minEnergy[:epoch+1]), label='Min Energy', marker='^')
                ax.legend()
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Energy')
                fig.savefig(f'./plots/energy/energy_epoch_{epoch}.png')
            
            # Check for stop file
            if os.path.exists('stop_now'):
                print("Stop file detected. Halting training.")
                break
        
        # Final save
        weights_to_save = {
            'VF': self.backend.to_numpy(VF),
            'FH': self.backend.to_numpy(FH),
            'bias_cov': self.backend.to_numpy(bias_cov),
            'bias_vis': self.backend.to_numpy(bias_vis),
            'w_mean': self.backend.to_numpy(w_mean),
            'bias_mean': self.backend.to_numpy(bias_mean),
            'epoch': self.num_epochs - 1
        }
        savemat("./weights/ws_final.mat", weights_to_save)
        
        energy_to_save = {
            'meanEnergy': self.backend.to_numpy(meanEnergy),
            'maxEnergy': self.backend.to_numpy(maxEnergy),
            'minEnergy': self.backend.to_numpy(minEnergy)
        }
        savemat(f"training_energy_f{self.num_fac}_c{self.num_hid_cov}_m{self.num_hid_mean}.mat", energy_to_save)
        
        with open('done', 'w') as f:
            f.write(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
        
        print("Training complete.")
        plt.close(fig)
