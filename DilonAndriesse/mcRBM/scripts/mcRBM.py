"""
@:description
Class implementing the mean-covariance Restricted Boltzmann Machine (mcRBM) by Marc'Aurelio Ranzato. It is based on the
original code with minor modifications according to the needs of our experiments.

@:refer
"Modeling Pixel Means and Covariances Using Factorized Third-Order Boltzmann Machines"

@:original_code
http://www.cs.toronto.edu/~ranzato/publications/mcRBM/code/mcRBM_04May2010.zip
"""

import sys
import numpy as np
import os
import cupy as cp
import pickle
import matplotlib.pyplot as plt
import shutil

from numpy.random import RandomState
from scipy.io import loadmat, savemat
from configparser import *
from datetime import datetime
from data_preproc import DataPreproc



class mcRBM:
    def __init__(self, refDir, expConfigFilename, modelConfigFilename, gpuId=0):
        # directory containing all the configuration files for the experiment
        self.refDir = refDir
        # file with configuration details for the launched experiment
        self.expConfigFilename = refDir + '/' + expConfigFilename
        # file with configuration details for the model to be trained
        self.modelConfigFilename = refDir + '/' + modelConfigFilename
        # data pre-processing object
        self.dpp = DataPreproc()
        # loading details from configuration files
        self.loadExpConfig()
        self.loadModelConfig()
        # id of the GPU which will be used for computation
        self.gpuId = int(gpuId)

    def loadExpConfig(self):
        """
        Function loading the configuration details for the experiment & data pre-processing flags
        """
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

        self.dataFilename = self.dataDir + self.dSetName
        self.saveDir = self.expsDir + self.expName

        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
        # shutil.copy2(self.expConfigFilename, self.saveDir)
        # shutil.copy2(self.modelConfigFilename, self.saveDir)

    def loadModelConfig(self):
        """
        Function loading the configuration details for the model to be trained
        """
        config = ConfigParser()
        config.read(self.modelConfigFilename)

        self.verbose = config.getint('VERBOSITY', 'verbose')

        self.num_epochs = config.getint('MAIN_PARAMETER_SETTING', 'num_epochs')
        self.batch_size = config.getint('MAIN_PARAMETER_SETTING', 'batch_size')
        self.startFH = config.getint('MAIN_PARAMETER_SETTING', 'startFH')
        self.startwd = config.getint('MAIN_PARAMETER_SETTING', 'startwd')
        self.doPCD = config.getint('MAIN_PARAMETER_SETTING', 'doPCD')

        # model parameters
        self.num_fac = config.getint('MODEL_PARAMETER_SETTING', 'num_fac')
        self.num_hid_cov = config.getint('MODEL_PARAMETER_SETTING', 'num_hid_cov')
        self.num_hid_mean = config.getint('MODEL_PARAMETER_SETTING', 'num_hid_mean')
        self.apply_mask = config.getint('MODEL_PARAMETER_SETTING', 'apply_mask')
        self.epsilon = config.getfloat('OPTIMIZER_PARAMETERS', 'epsilon')
        self.weightcost_final = config.getfloat('OPTIMIZER_PARAMETERS', 'weightcost_final')
        self.hmc_step_nr = config.getint('HMC_PARAMETERS', 'hmc_step_nr')
        self.hmc_target_ave_rej = config.getfloat('HMC_PARAMETERS', 'hmc_target_ave_rej')

    def loadData(self):
        """
        Function loading the data
        """
        if not os.path.exists(self.saveDir + '/dataDetails/'):
            os.makedirs(self.saveDir + '/dataDetails/')

        # load data file:
        if self.dataFilename.split('.')[1] == 'npz':
            dLoad = np.load(self.dataFilename)
        elif self.dataFilename.split('.') == 'mat':
            dLoad = loadmat(self.dataFilename)
        else:
            print("error! Unrecognized data file")
        
        self.d = dLoad['d']
        self.obsKeys = dLoad['epochsLinked']
        self.epochTime = dLoad['epochTime']
        
        
        """
        If you want to keep only EEG features, uncomment next line.
		"""

        # self.d = self.d[:, :self.d.shape[1]-1]

        self.d = np.array(self.d, dtype=np.float32)
        self.obsKeys = np.array(self.obsKeys, dtype=np.float32)
        print(("initial size: ", self.d.shape))
        # print("FrameIDs : ", self.obsKeys, "of shape : ", self.obsKeys.shape)

        with open(self.saveDir + '/dataDetails/' + 'initialData.txt', 'w') as f:
            f.write("\n Modeling: %s " % self.dataFilename)
            f.write("\n Dataset size: %s " % str(self.d.shape))
            f.write("\n Dataset type: %s " % str(self.d.dtype))
            f.write("\n \n d_min: %s " % str(np.min(self.d, axis=0)))
            f.write("\n \n d_max: %s " % str(np.max(self.d, axis=0)))
            f.write("\n \n d_mean: %s " % str(np.mean(self.d, axis=0)))
            f.write("\n \n d_std: %s " % str(np.std(self.d, axis=0)))
            f.close()

    # Function taken from original code
    def compute_energy_mcRBM(self, data, normdata, vel, energy, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2,
                             t6, feat, featsq, feat_mean, length, lengthsq, normcoeff, small, num_vis):
        # normalize input data vectors
        t6 = data ** 2  # DxP
        lengthsq = t6.sum(axis=0)  # 1xP
        energy = 0.5 * lengthsq  # energy of quadratic regularization term
        lengthsq = lengthsq / num_vis  # normalize by number of components (like std)
        lengthsq = lengthsq + small  # small prevents division by 0
        length = cp.sqrt(lengthsq)
        normcoeff = 1.0 / length  # 1xP
        normdata = data * normcoeff  # normalized data

        # covariance contribution
        feat = cp.dot(VF.T, normdata)  # HxP
        featsq = feat ** 2  # HxP
        t1 = cp.dot(FH.T, featsq)  # OxP
        t1 = -0.5 * t1
        t1 = t1 + bias_cov[:, cp.newaxis]  # add column vector
        t2 = cp.log1p(cp.exp(t1))  # log(1 + exp(t1))
        energy = energy + t2.sum(axis=0)

        # mean contribution
        feat_mean = cp.dot(w_mean.T, data)  # HxP
        feat_mean = feat_mean + bias_mean[:, cp.newaxis]  # add column vector
        feat_mean = -cp.log1p(cp.exp(feat_mean))  # -log(1 + exp(feat_mean))
        energy = energy + feat_mean.sum(axis=0)

        # visible bias term
        t6 = -data * bias_vis[:, cp.newaxis]  # DxP
        energy = energy + t6.sum(axis=0)

        # kinetic energy
        energy = energy + 0.5 * (vel ** 2).sum(axis=0)

    # same as the previous function. Needed only if the energy has to be computed
    # and stored to check the training process
    def compute_energy_mcRBM_visual(self, data, normdata, energy, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2,
                                    t6, feat, featsq, feat_mean, length, lengthsq, normcoeff, small, num_vis):
        # normalize input data vectors
        t6 = data ** 2  # DxP
        lengthsq = t6.sum(axis=0)  # 1xP
        energy = 0.5 * lengthsq  # energy of quadratic regularization term
        lengthsq = lengthsq / num_vis  # normalize by number of components (like std)
        lengthsq = lengthsq + small  # small prevents division by 0
        length = cp.sqrt(lengthsq)
        normcoeff = 1.0 / length  # 1xP
        normdata = data * normcoeff  # normalized data

        # covariance contribution
        feat = cp.dot(VF.T, normdata)  # HxP
        featsq = feat ** 2  # HxP
        t1 = cp.dot(FH.T, featsq)  # OxP
        t1 = -0.5 * t1
        t1 = t1 + bias_cov[:, cp.newaxis]  # broadcast column vector
        t2 = cp.log1p(cp.exp(t1))  # log(1 + exp(t1))
        energy = energy + (-t2).sum(axis=0)  # add negative log

        # mean contribution
        feat_mean = cp.dot(w_mean.T, data)  # HxP
        feat_mean = feat_mean + bias_mean[:, cp.newaxis]  # add column vector
        feat_mean = -cp.log1p(cp.exp(feat_mean))  # -log(1 + exp(feat_mean))
        energy = energy + feat_mean.sum(axis=0)

        # visible bias term
        t6 = -data * bias_vis[:, cp.newaxis]  # DxP
        energy = energy + t6.sum(axis=0)

        # kinetic energy
        t6 = 0.5 * (data ** 2)  # DxP
        energy = energy + t6.sum(axis=0)

    # Function taken from original code
    #################################################################
    # compute the derivative if the free energy at a given input
    def compute_gradient_mcRBM(self, data, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2, t3, t4, t6,
                               feat, featsq, feat_mean, gradient, normgradient, length, lengthsq, normcoeff, small,
                               num_vis):
        # normalize input data
        t6 = data ** 2  # DxP
        lengthsq = t6.sum(axis=0)  # 1xP
        lengthsq = lengthsq / num_vis  # normalize by number of components
        lengthsq = lengthsq + small
        length = cp.sqrt(lengthsq)
        normcoeff = 1.0 / length  # 1xP
        normdata = data * normcoeff  # normalized data

        # forward pass
        feat = cp.dot(VF.T, normdata)  # HxP
        featsq = feat ** 2  # HxP
        t1 = cp.dot(FH.T, featsq)  # OxP
        t1 = -0.5 * t1 + bias_cov[:, cp.newaxis]  # add column vector with broadcast
        t2 = 1 / (1 + cp.exp(-t1))  # sigmoid

        t3 = cp.dot(FH, t2)  # HxP
        t3 = t3 * feat
        normgradient = cp.dot(VF, t3)  # VxP

        # backprop through normalization
        normcoeff2 = length * lengthsq
        normcoeff2 = 1.0 / normcoeff2  # 1xP
        gradient = normgradient * data  # VxP

        t4 = -gradient.sum(axis=0) / num_vis  # 1xP
        gradient = gradient + data * t4  # broadcast row-wise
        gradient = gradient * lengthsq  # broadcast
        gradient = gradient * normcoeff2  # broadcast

        # add quadratic term gradient
        gradient = gradient + data

        # add visible bias term
        gradient = gradient - bias_vis[:, cp.newaxis]

        # add MEAN contribution to gradient
        feat_mean = cp.dot(w_mean.T, data) + bias_mean[:, cp.newaxis]  # HxP
        feat_mean = 1 / (1 + cp.exp(-feat_mean))  # sigmoid
        gradient = gradient - cp.dot(w_mean, feat_mean)  # VxP

    # Function taken from original code
    ############################################################3
    # Hybrid Monte Carlo sampler
    def draw_HMC_samples(self, data, negdata, normdata, vel, gradient, normgradient, new_energy, old_energy, VF, FH,
                         bias_cov, bias_vis, w_mean, bias_mean, hmc_step, hmc_step_nr, hmc_ave_rej, hmc_target_ave_rej,
                         t1, t2, t3, t4, t5, t6, t7, thresh, feat, featsq, batch_size, feat_mean, length, lengthsq,
                         normcoeff, small, num_vis):
        vel = cp.random.randn(*vel.shape, dtype=cp.float32)
        #vel.fill_with_randn()
        negdata = data.copy()
        self.compute_energy_mcRBM(negdata, normdata, vel, old_energy, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1,
                                  t2, t6, feat, featsq, feat_mean, length, lengthsq, normcoeff, small, num_vis)
        self.compute_gradient_mcRBM(negdata, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2, t3, t4,
                                    t6, feat, featsq, feat_mean, gradient, normgradient, length, lengthsq, normcoeff,
                                    small, num_vis)
        # half step
        vel = -0.5 * hmc_step * gradient
        negdata = hmc_step * vel
        # full leap-frog steps
        for ss in range(hmc_step_nr - 1):
            ## re-evaluate the gradient
            self.compute_gradient_mcRBM(negdata, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2, t3,
                                        t4, t6, feat, featsq, feat_mean, gradient, normgradient, length, lengthsq,
                                        normcoeff, small, num_vis)
            # update variables
            vel += -hmc_step * gradient
            negdata += hmc_step * vel
        # final half-step
        self.compute_gradient_mcRBM(negdata, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2, t3, t4,
                                    t6, feat, featsq, feat_mean, gradient, normgradient, length, lengthsq, normcoeff,
                                    small, num_vis)
        vel += -0.5 * hmc_step * gradient
        # compute new energy
        self.compute_energy_mcRBM(negdata, normdata, vel, new_energy, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1,
                                  t2, t6, feat, featsq, feat_mean, length, lengthsq, normcoeff, small, num_vis)
        # rejecton
        thresh = old_energy - new_energy
        cp.exp(thresh)
        t4 = cp.random.randn(*t4.shape, dtype=cp.float32)
        print("t4:", t4.shape)
        #t4.fill_with_rand()
        cp.less(t4, thresh)
        print("t4:", t4.shape)
        #    update negdata and rejection rate
        t4 *= -1
        t4 += 1  # now 1's detect rejections
        t5 = t4.sum(axis=1, keepdims=True)
        print("t5:", t5.shape)
        t5.get()
        rej = t5[0, 0] / batch_size

        repeats = data.shape[1] // t4.shape[1]  # number of repeats to match columns
        t4_expanded = cp.repeat(t4, repeats=repeats, axis=1)  # expand t4 to match data
        # Multiply and store in target
        t6[:] = data * t4_expanded
        #t6[:] = data * t4[cp.newaxis, :]
                # Assuming block_size = t4.shape[1]
        repeats = negdata.shape[1] // t4.shape[1]  # number of repeats to match columns
        t4_expanded = cp.repeat(t4, repeats=repeats, axis=1)  # expand t4 to match data
        # Multiply and store in target
        t7[:] = negdata * t4_expanded
        #t7[:] = negdata * t4[cp.newaxis, :]
        negdata -= t7
        negdata += t6
        hmc_ave_rej = 0.9 * hmc_ave_rej + 0.1 * rej
        if hmc_ave_rej < hmc_target_ave_rej:
            hmc_step = min(hmc_step * 1.01, 0.25)
        else:
            hmc_step = max(hmc_step * 0.99, .001)
        return hmc_step, hmc_ave_rej

    def saveLsq(self):
        '''
        Function saving the sum of the square of the data
        (needed for training as well as for post-analysis)
        '''
        d = self.d.astype(np.float32)

        dsq = np.square(d)
        lsq = np.sum(dsq, axis=0)
        with open(self.refDir + 'lsqComplete.pkl', 'wb') as pklFile:
            pickle.dump(lsq, pklFile)

    def train(self):
        """
        @:description:
        Main train function; modified version of the original train function.
        @:addition
        GPU selection (useful for multi-GPU machines) Saving the sum of the square of the data for post-processing
        - Visible data are saved
        - Data samples are permuted for training
        - Weights are saved every 100 training epochs
        - Training energy is visualized every 100 training epochs

        NOTE : anneal learning rate used in the initial code, is NOT used here!
        """

        # plt.ion()

        f1 = plt.figure()
        ax1 = f1.add_subplot(111)
        # ax2 = f1.add_subplot(122)
        # plt.show()

        cp.cuda.Device(0).use()
        # cp.cublas_init()
        cp.random.rand(1).astype(cp.float32)

        np.random.seed(self.npRandSeed)
        prng = RandomState(self.npRandState)

        ################################################################
        ##################### CHANGE PATH ##############################
        # Move to current experiment path:
        os.chdir(self.saveDir)
        # Get current path:
        os.getcwd()

        self.plotsDir = 'plots'
        # self.probabilitiesDir = 'p_all'
        if not os.path.isdir(self.plotsDir):
            os.makedirs(self.plotsDir)
        if not os.path.isdir(self.plotsDir + '/energy'):
            os.makedirs(self.plotsDir + '/energy')
        # if not os.path.isdir(self.probabilitiesDir):
        #	os.makedirs(self.probabilitiesDir)
        if not os.path.isdir('weights'):
            os.makedirs('weights')

        d = self.d.astype(np.float32)
        print(("visible size: ", d.shape))

        dsq = np.square(d)
        lsq = np.sum(dsq, axis=0)
        with open('lsqComplete.pkl', 'wb') as pklFile:
            pickle.dump(lsq, pklFile)

        del dsq, lsq

        # Save visible data :
        visData = d
        np.savez('visData.npz', data=d, obsKeys=self.obsKeys, epochTime=self.epochTime)

        with open('visData.txt', 'w') as f:
            f.write("\n Dataset : %s" % (self.dataFilename))
            f.write("\n visData size: %s " % str(visData.shape))
            f.write("\n visData type: %s " % str(visData.dtype))
            f.write("\n \n visData Range: %s " % str(np.max(visData, axis=0) - np.min(visData, axis=0)))
            f.write("\n \n visData min: %s " % str(np.min(visData, axis=0)))
            f.write("\n \n visData max: %s " % str(np.max(visData, axis=0)))
            f.write("\n \n visData mean: %s " % str(np.mean(visData, axis=0)))
            f.write("\n \n visData std: %s " % str(np.std(visData, axis=0)))
            f.close()

        del visData  # if not needed for computing the latent states

        permIdx = prng.permutation(d.shape[0])

        d = d[permIdx, :]

        # subsetting train and test datasets
        # trainPerc = 0.7
        # trainSampNum = int(np.ceil(trainPerc*d.shape[0]))
        # trainSampNum = int(np.floor(trainSampNum/self.batch_size)*self.batch_size)
        # testSampNum = int(d.shape[0]-trainSampNum-1)

        # The test dataset is not used at the moment, it can be used as
        # a validation set to check for overfitting. To use it, uncomment
        # all the variables with 'test' in their name

        # ~ d_test = d[trainSampNum+1:,:]
        # d = d[:trainSampNum,:]
        # obsKeys = self.obsKeys[:trainSampNum]

        totnumcases = d.shape[0]
        num_vis = d.shape[1]

        num_batches = int(totnumcases / self.batch_size)
        print(("num_batches: ", num_batches))
        dev_dat = cp.array(d.T)  # VxP
        # ~ test_dat = cp.array(d_test.T)

        del d, self.d, self.epochTime, self.obsKeys

        # training parameters (as in the original code by Ranzato)
        epsilon = self.epsilon
        epsilonVF = 2 * epsilon
        epsilonFH = 0.02 * epsilon
        epsilonb = 0.02 * epsilon
        epsilonw_mean = 0.2 * epsilon
        epsilonb_mean = 0.1 * epsilon
        weightcost_final = self.weightcost_final

        # HMC setting
        hmc_step_nr = self.hmc_step_nr
        hmc_step = 0.01
        hmc_target_ave_rej = self.hmc_target_ave_rej
        hmc_ave_rej = hmc_target_ave_rej

        # initialize weights
        VF = cp.array(np.array(0.02 * prng.randn(num_vis, self.num_fac), dtype=np.float32, order='F'))  # VxH
        if self.apply_mask == 0:
            FH = cp.array(np.array(np.eye(self.num_fac, self.num_hid_cov), dtype=np.float32, order='F'))  # HxO
        else:
            dd = loadmat(
                'your_FHinit_mask_file.mat')  # see CVPR2010paper_material/topo2D_3x3_stride2_576filt.mat for an example
            FH = cp.array(np.array(dd["FH"], dtype=np.float32, order='F'))
        bias_cov = cp.array(np.array(2.0 * np.ones((self.num_hid_cov, 1)), dtype=np.float32, order='F'))
        bias_vis = cp.array(np.array(np.zeros((num_vis, 1)), dtype=np.float32, order='F'))
        w_mean = cp.array(
            np.array(0.05 * prng.randn(num_vis, self.num_hid_mean), dtype=np.float32, order='F'))  # VxH
        bias_mean = cp.array(np.array(-2.0 * np.ones((self.num_hid_mean, 1)), dtype=np.float32, order='F'))

        # initialize variables to store derivatives
        VFinc = cp.array(np.array(np.zeros((num_vis, self.num_fac)), dtype=np.float32, order='F'))
        FHinc = cp.array(np.array(np.zeros((self.num_fac, self.num_hid_cov)), dtype=np.float32, order='F'))
        bias_covinc = cp.array(np.array(np.zeros((self.num_hid_cov, 1)), dtype=np.float32, order='F'))
        bias_visinc = cp.array(np.array(np.zeros((num_vis, 1)), dtype=np.float32, order='F'))
        w_meaninc = cp.array(np.array(np.zeros((num_vis, self.num_hid_mean)), dtype=np.float32, order='F'))
        bias_meaninc = cp.array(np.array(np.zeros((self.num_hid_mean, 1)), dtype=np.float32, order='F'))

        # initialize temporary storage
        data = cp.array(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))  # VxP
        normdata = cp.array(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))  # VxP
        negdataini = cp.array(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))  # VxP
        feat = cp.array(np.array(np.empty((self.num_fac, self.batch_size)), dtype=np.float32, order='F'))
        featsq = cp.array(np.array(np.empty((self.num_fac, self.batch_size)), dtype=np.float32, order='F'))
        negdata = cp.array(np.array(prng.randn(num_vis, self.batch_size), dtype=np.float32, order='F'))
        old_energy = cp.array(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))
        new_energy = cp.array(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))
        energy = cp.array(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))
        gradient = cp.array(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))  # VxP
        normgradient = cp.array(
            np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))  # VxP
        thresh = cp.array(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))
        feat_mean = cp.array(
            np.array(np.empty((self.num_hid_mean, self.batch_size)), dtype=np.float32, order='F'))
        vel = cp.array(np.array(prng.randn(num_vis, self.batch_size), dtype=np.float32, order='F'))
        length = cp.array(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))  # 1xP
        lengthsq = cp.array(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))  # 1xP
        normcoeff = cp.array(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))  # 1xP

        # commented to avoid computing the energy on test data
        # ~ data_test = cp.array( np.array(np.empty((num_vis, testSampNum)), dtype=np.float32, order='F')) # Vxtest_batch
        # ~ normdata_test = cp.array( np.array(np.empty((num_vis, testSampNum)), dtype=np.float32, order='F')) # Vxtest_batch
        # ~ length_test = cp.array( np.array(np.zeros((1, testSampNum)), dtype=np.float32, order='F')) # 1xtest_batch
        # ~ lengthsq_test = cp.array( np.array(np.zeros((1, testSampNum)), dtype=np.float32, order='F')) # 1xtest_batch
        # ~ normcoeff_test = cp.array( np.array(np.zeros((1, testSampNum)), dtype=np.float32, order='F')) # 1xtest_batch
        # ~ vel_test = cp.array( np.array(prng.randn(num_vis, testSampNum), dtype=np.float32, order='F'))
        # ~ feat_test = cp.array( np.array(np.empty((self.num_fac, testSampNum)), dtype=np.float32, order='F'))
        # ~ featsq_test = cp.array( np.array(np.empty((self.num_fac, testSampNum)), dtype=np.float32, order='F'))
        # ~ feat_mean_test = cp.array( np.array(np.empty((self.num_hid_mean, testSampNum)), dtype=np.float32, order='F'))
        # ~ energy_test = cp.array( np.array(np.zeros((1, testSampNum)), dtype=np.float32, order='F'))

        if self.apply_mask == 1:  # this used to constrain very large FH matrices only allowing to change values in a neighborhood
            dd = loadmat('your_FHinit_mask_file.mat')
            mask = cp.array(np.array(dd["mask"], dtype=np.float32, order='F'))
        normVF = 1
        small = 0.5

        # other temporary vars
        t1 = cp.array(np.array(np.empty((self.num_hid_cov, self.batch_size)), dtype=np.float32, order='F'))
        t2 = cp.array(np.array(np.empty((self.num_hid_cov, self.batch_size)), dtype=np.float32, order='F'))
        t3 = cp.array(np.array(np.empty((self.num_fac, self.batch_size)), dtype=np.float32, order='F'))
        t4 = cp.array(np.array(np.empty((1, self.batch_size)), dtype=np.float32, order='F'))
        t5 = cp.array(np.array(np.empty((1, 1)), dtype=np.float32, order='F'))
        t6 = cp.array(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))
        t7 = cp.array(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))
        t8 = cp.array(np.array(np.empty((num_vis, self.num_fac)), dtype=np.float32, order='F'))
        t9 = cp.array(np.array(np.zeros((self.num_fac, self.num_hid_cov)), dtype=np.float32, order='F'))
        t10 = cp.array(np.array(np.empty((1, self.num_fac)), dtype=np.float32, order='F'))
        t11 = cp.array(np.array(np.empty((1, self.num_hid_cov)), dtype=np.float32, order='F'))

        # commented to avoid computing the energy on test data
        # ~ t1_test = cp.array( np.array(np.empty((self.num_hid_cov, testSampNum)), dtype=np.float32, order='F'))
        # ~ t2_test = cp.array( np.array(np.empty((self.num_hid_cov, testSampNum)), dtype=np.float32, order='F'))
        # ~ t3_test = cp.array( np.array(np.empty((self.num_fac, testSampNum)), dtype=np.float32, order='F'))
        # ~ t4_test = cp.array( np.array(np.empty((1,testSampNum)), dtype=np.float32, order='F'))
        # ~ t5_test = cp.array( np.array(np.empty((1,1)), dtype=np.float32, order='F'))
        # ~ t6_test = cp.array( np.array(np.empty((num_vis, testSampNum)), dtype=np.float32, order='F'))

        meanEnergy = np.zeros(self.num_epochs)
        minEnergy = np.zeros(self.num_epochs)
        maxEnergy = np.zeros(self.num_epochs)
        # ~ meanEnergy_test = np.zeros(self.num_epochs)
        # ~ minEnergy_test = np.zeros(self.num_epochs)
        # ~ maxEnergy_test = np.zeros(self.num_epochs)

        # start training
        for epoch in range(self.num_epochs):

            print("Epoch " + str(epoch))

            # anneal learning rates as found in the original code -
            # uncomment if you wish to use annealing!
            # ~ epsilonVFc    = epsilonVF/max(1,epoch/20)
            # ~ epsilonFHc    = epsilonFH/max(1,epoch/20)
            # ~ epsilonbc    = epsilonb/max(1,epoch/20)
            # ~ epsilonw_meanc = epsilonw_mean/max(1,epoch/20)
            # ~ epsilonb_meanc = epsilonb_mean/max(1,epoch/20)

            # no annealing is used in our experiments because learning
            # was stopping too early
            epsilonVFc = epsilonVF
            epsilonFHc = epsilonFH
            epsilonbc = epsilonb
            epsilonw_meanc = epsilonw_mean
            epsilonb_meanc = epsilonb_mean

            weightcost = weightcost_final

            if epoch <= self.startFH:
                epsilonFHc = 0
            if epoch <= self.startwd:
                weightcost = 0

            # commented to avoid computing the energy on test data
            # ~ data_test = test_dat

            # ~ data_test.mult(data_test, target = t6_test) # DxP
            # ~ t6_test.sum(axis = 0, target = lengthsq_test) # 1xP
            # ~ lengthsq_test.mult(1./num_vis) # normalize by number of components (like std)
            # ~ lengthsq_test.add(small) # small avoids division by 0
            # ~ cp.sqrt(lengthsq_test, target = length_test)
            # ~ length_test.reciprocal(target = normcoeff_test) # 1xP
            # ~ data_test.mult_by_row(normcoeff_test, target = normdata_test) # normalized data

            for batch in range(num_batches):

                # get current minibatch
                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                data = dev_dat[start:end, :]
                print("data:", data.shape)
                # data = dev_dat.slice(batch * self.batch_size,
                #                      (batch + 1) * self.batch_size)  # DxP (nr dims x nr samples)

                # normalize input data
                t6 = data ** 2 # DxP
                lengthsq = t6.sum(axis=0)  # 1xP
                lengthsq = lengthsq / num_vis # normalize by number of components (like std)
                lengthsq = lengthsq + small  # small avoids division by 0
                length = cp.sqrt(lengthsq)
                normcoeff = 1.0/length  # 1xP
                normdata = data * normcoeff  # normalized data
                print("normdata:", normdata.shape)
                ## compute positive sample derivatives
                # covariance part
                feat = cp.dot(VF.T, normdata)  # HxP (nr facs x nr samples)
                print("feat:", feat.shape)
                featsq = feat ** 2  # HxP
                print("featsq:", featsq.shape)
                t1 = cp.dot(FH.T, featsq)  # OxP (nr cov hiddens x nr samples)
                print("t1:", t1.shape)
                t1 = t1 * (-0.5)
                print("bias_cov:", bias_cov.shape)
                bias_cov = bias_cov.ravel()
                t1 = t1 + bias_cov[:, cp.newaxis]  # OxP
                print("t1:", t1.shape)
                t2 = 1 / (1 + cp.exp(-t1)) # OxP
                print("t2:", t2.shape)
                t2 = cp.squeeze(t2)
                print("t2 after squeenze:", t2.shape)
                FHinc = cp.dot(featsq, t2.T)  # HxO
                t3 = cp.dot(FH, t2)  # HxP
                print("t3:", t3.shape)
                t3 = t3 * feat
                VFinc += cp.dot(normdata, t3.T)  # VxH
                print("VFinc:", VFinc.shape)
                bias_covinc = -1 * t2.sum(axis=1)
                # visible bias
                bias_visinc = -1 * data.sum(axis=1)
                # mean part
                print("w_mean:", w_mean.shape)
                print("data:", data.shape)
                feat_mean = cp.dot(w_mean.T, data)  # HxP (nr mean hiddens x nr samples)
                print("feat_mean:", feat_mean.shape)
                bias_mean = bias_mean.ravel()
                feat_mean = feat_mean + bias_mean[:, cp.newaxis]  # HxP
                feat_mean = 1 / (1 + cp.exp(-feat_mean)) # HxP
                feat_mean = feat_mean * (-1)
                print("feat_mean_2:", feat_mean.shape)

                w_meaninc = cp.dot(data, feat_mean.T)
                bias_meaninc = feat_mean.sum(axis=1)

                # HMC sampling: draw an approximate sample from the model
                if self.doPCD == 0:  # CD-1 (set negative data to current training samples)
                    hmc_step, hmc_ave_rej = self.draw_HMC_samples(data, negdata, normdata, vel, gradient, normgradient,
                                                                  new_energy, old_energy, VF, FH, bias_cov, bias_vis,
                                                                  w_mean, bias_mean, hmc_step, hmc_step_nr, hmc_ave_rej,
                                                                  hmc_target_ave_rej, t1, t2, t3, t4, t5, t6, t7,
                                                                  thresh, feat, featsq, self.batch_size, feat_mean,
                                                                  length, lengthsq, normcoeff, small, num_vis)
                else:  # PCD-1 (use previous negative data as starting point for chain)
                    negdataini = negdata.copy()
                    hmc_step, hmc_ave_rej = self.draw_HMC_samples(negdataini, negdata, normdata, vel, gradient,
                                                                  normgradient, new_energy, old_energy, VF, FH,
                                                                  bias_cov, bias_vis, w_mean, bias_mean, hmc_step,
                                                                  hmc_step_nr, hmc_ave_rej, hmc_target_ave_rej, t1, t2,
                                                                  t3, t4, t5, t6, t7, thresh, feat, featsq,
                                                                  self.batch_size, feat_mean, length, lengthsq,
                                                                  normcoeff, small, num_vis)

                # --- normalize negative input data ---
                t6 = negdata ** 2                        # DxP
                lengthsq = t6.sum(axis=0) / num_vis      # 1xP
                lengthsq = lengthsq + small
                length = cp.sqrt(lengthsq)
                normcoeff = 1.0 / length
                normdata = negdata * normcoeff           # broadcasting

                # --- covariance part ---
                feat = cp.dot(VF.T, normdata)            # HxP
                featsq = feat ** 2
                t1 = cp.dot(FH.T, featsq) * -0.5         # OxP
                t1 = t1 + bias_cov[:, cp.newaxis]        # add bias
                t2 = 1 / (1 + cp.exp(-t1))               # sigmoid
                FHinc -= cp.dot(featsq, t2.T)            # subtract update
                FHinc *= 0.5

                t3 = cp.dot(FH, t2) * feat               # HxP
                VFinc -= cp.dot(normdata, t3.T)          # subtract update

                bias_covinc += t2.sum(axis=1)            # Ox1
                bias_visinc += negdata.sum(axis=1)       # Vx1

                # --- mean part ---
                feat_mean = cp.dot(w_mean.T, negdata)    # HxP
                feat_mean = 1 / (1 + cp.exp(-(feat_mean + bias_mean[:, cp.newaxis])))
                w_meaninc += cp.dot(negdata, feat_mean.T)
                bias_meaninc += feat_mean.sum(axis=1)

                # --- update parameters ---
                VFinc += VF.sign() * weightcost
                VF -= (epsilonVFc / self.batch_size) * VFinc

                # normalize columns of VF (running average)
                t8 = VF ** 2
                t10 = cp.sqrt(t8.sum(axis=0))
                normVF = 0.95 * normVF + 0.05 / self.num_fac * t10.sum()
                VF = VF * (1.0 / t10) * normVF  # normalize columns

                bias_cov -= (epsilonbc / self.batch_size) * bias_covinc
                bias_vis -= (epsilonbc / self.batch_size) * bias_visinc

                if epoch > self.startFH:
                    FHinc += FH.sign() * weightcost
                    FH -= (epsilonFHc / self.batch_size) * FHinc

                    # zero out negative entries
                    FH = FH * (FH > 0)

                    if self.apply_mask == 1:
                        FH *= mask

                    # normalize columns of FH (L1 norm)
                    col_sum = FH.sum(axis=0)
                    FH = FH / col_sum

                w_meaninc += w_mean.sign() * weightcost
                w_mean -= (epsilonw_meanc / self.batch_size) * w_meaninc
                bias_mean -= (epsilonb_meanc / self.batch_size) * bias_meaninc

            if self.verbose == 1:
                print("VF: " + '%3.2e' % VF.euclid_norm() + ", DVF: " + '%3.2e' % (VFinc.euclid_norm() * (
                        epsilonVFc / self.batch_size)) + ", FH: " + '%3.2e' % FH.euclid_norm() + ", DFH: " + '%3.2e' % (
                              FHinc.euclid_norm() * (
                              epsilonFHc / self.batch_size)) + ", bias_cov: " + '%3.2e' % bias_cov.euclid_norm() + ", Dbias_cov: " + '%3.2e' % (
                              bias_covinc.euclid_norm() * (
                              epsilonbc / self.batch_size)) + ", bias_vis: " + '%3.2e' % bias_vis.euclid_norm() + ", Dbias_vis: " + '%3.2e' % (
                              bias_visinc.euclid_norm() * (
                              epsilonbc / self.batch_size)) + ", wm: " + '%3.2e' % w_mean.euclid_norm() + ", Dwm: " + '%3.2e' % (
                              w_meaninc.euclid_norm() * (
                              epsilonw_meanc / self.batch_size)) + ", bm: " + '%3.2e' % bias_mean.euclid_norm() + ", Dbm: " + '%3.2e' % (
                              bias_meaninc.euclid_norm() * (
                              epsilonb_meanc / self.batch_size)) + ", step: " + '%3.2e' % hmc_step + ", rej: " + '%3.2e' % hmc_ave_rej)
                with open('terminal.txt', 'a') as f:
                    f.write('\n' + "epoch: %s" % str(
                        epoch) + ", VF: " + '%3.2e' % VF.euclid_norm() + ", DVF: " + '%3.2e' % (VFinc.euclid_norm() * (
                            epsilonVFc / self.batch_size)) + ", FH: " + '%3.2e' % FH.euclid_norm() + ", DFH: " + '%3.2e' % (
                                    FHinc.euclid_norm() * (
                                    epsilonFHc / self.batch_size)) + ", bias_cov: " + '%3.2e' % bias_cov.euclid_norm() + ", Dbias_cov: " + '%3.2e' % (
                                    bias_covinc.euclid_norm() * (
                                    epsilonbc / self.batch_size)) + ", bias_vis: " + '%3.2e' % bias_vis.euclid_norm() + ", Dbias_vis: " + '%3.2e' % (
                                    bias_visinc.euclid_norm() * (
                                    epsilonbc / self.batch_size)) + ", wm: " + '%3.2e' % w_mean.euclid_norm() + ", Dwm: " + '%3.2e' % (
                                    w_meaninc.euclid_norm() * (
                                    epsilonw_meanc / self.batch_size)) + ", bm: " + '%3.2e' % bias_mean.euclid_norm() + ", Dbm: " + '%3.2e' % (
                                    bias_meaninc.euclid_norm() * (
                                    epsilonb_meanc / self.batch_size)) + ", step: " + '%3.2e' % hmc_step + ", rej: " + '%3.2e' % hmc_ave_rej)
                sys.stdout.flush()

            # commented to avoid computing the energy on trainig data
            self.compute_energy_mcRBM_visual(data, normdata, energy, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1,
                                             t2, t6, feat, featsq, feat_mean, length, lengthsq, normcoeff, small,
                                             num_vis)
            energy.get()
            meanEnergy[epoch] = np.mean(energy)
            minEnergy[epoch] = np.min(energy)
            maxEnergy[epoch] = np.max(energy)

            # commented to avoid computing the energy on test data
            # ~ self.compute_energy_mcRBM_visual(data_test,normdata_test,energy_test,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,t1_test,t2_test,t6_test,feat_test,featsq_test,feat_mean_test,length_test,lengthsq_test,normcoeff_test,small,num_vis)
            # ~ energy_test.copy_to_host()
            # ~ meanEnergy_test[epoch] = np.mean(energy_test.numpy_array)
            # ~ minEnergy_test[epoch] = np.min(energy_test.numpy_array)
            # ~ maxEnergy_test[epoch] = np.max(energy_test.numpy_array)

            ax1.cla()
            ax1.plot(list(range(epoch)), meanEnergy[0:epoch])
            ax1.plot(list(range(epoch)), maxEnergy[0:epoch])
            ax1.plot(list(range(epoch)), minEnergy[0:epoch])

            if np.mod(epoch, 100) == 0:
                # f1.savefig(output_folder + str(epoch)+'_'+'fig.png')
                f1.savefig(self.plotsDir + '/energy/energyAt_%s.png' % str(epoch))

            # back-up every once in a while
            if np.mod(epoch, 100) == 0:
                VF.get()
                FH.get()
                bias_cov.get()
                w_mean.get()
                bias_mean.get()
                bias_vis.get()
                savemat("./weights/ws_temp%s" % str(epoch),
                        {'VF': VF, 'FH': FH, 'bias_cov': bias_cov,
                         'bias_vis': bias_vis, 'w_mean': w_mean,
                         'bias_mean': bias_mean, 'epoch': epoch})

                # uncomment if computing the energy in order to store its evolution throghout training
                # ~ savemat(self.refDir + '/' + "training_energy_" + str(self.num_fac) + "_cov" + str(self.num_hid_cov) + "_mean" + str(self.num_hid_mean), {'meanEnergy':meanEnergy,'meanEnergy_test':meanEnergy_test,'maxEnergy': maxEnergy, 'maxEnergy_test': maxEnergy_test, 'minEnergy': minEnergy, 'minEnergy_test': minEnergy_test, 'epoch':epoch})
                # savemat("training_energy_" + str(self.num_fac) + "_cov" + str(self.num_hid_cov) + "_mean" + str(self.num_hid_mean), {'meanEnergy':meanEnergy, 'maxEnergy': maxEnergy, 'minEnergy': minEnergy, 'epoch':epoch})

            # in order to stop the training gracefully, create an empty file
            # named 'stop_now' in the folder containing the experiment
            # configuration file
            if os.path.isfile('stop_now'):
                break

        # final back-up
        VF.get()
        FH.get()
        bias_cov.get()
        bias_vis.get()
        w_mean.get()
        bias_mean.get()
        savemat("ws_fac%s" % str(self.num_fac) + "_cov%s" % str(self.num_hid_cov) + "_mean%s" % str(self.num_hid_mean),
                {'VF': VF, 'FH': FH, 'bias_cov': bias_cov,
                 'bias_vis': bias_vis, 'w_mean': w_mean, 'bias_mean': bias_mean,
                 'epoch': epoch})

        # uncomment if computing the energy in order to store its evolution throghout training
        # ~ savemat(self.refDir + '/' + "training_energy_" + str(self.num_fac) + "_cov" + str(self.num_hid_cov) + "_mean" + str(self.num_hid_mean), {'meanEnergy':meanEnergy,'meanEnergy_test':meanEnergy_test,'maxEnergy': maxEnergy, 'maxEnergy_test': maxEnergy_test, 'minEnergy': minEnergy, 'minEnergy_test': minEnergy_test, 'epoch':epoch})
        savemat(
            "training_energy_" + str(self.num_fac) + "_cov" + str(self.num_hid_cov) + "_mean" + str(self.num_hid_mean),
            {'meanEnergy': meanEnergy, 'maxEnergy': maxEnergy, 'minEnergy': minEnergy, 'epoch': epoch})

        # Compute states if desired:
        # normalise data for covariance hidden:
        # dsq = np.square(visData)
        # lsq = np.sum(dsq, axis=0)
        # lsq /= visData.shape[1]
        # lsq += np.spacing(1)
        # l = np.sqrt(lsq)
        # normD = visData/l

        # logisticArg_c = (-0.5*np.dot(FH.numpy_array.T, np.square(np.dot(VF.numpy_array.T, normD.T))) + bias_cov.numpy_array).T
        # p_hc = logisticFunc(logisticArg_c)

        # logisticArg_m = np.dot(visData, w_mean.numpy_array) + bias_mean.numpy_array.T
        # p_hm = logisticFunc(logisticArg_m)

        # p_all = np.concatenate((p_hc, p_hm), axis=1)
        # savemat(self.probabilitiesDir + '/pAll_%i.mat' % epoch, mdict={'p_all':p_all})

        with open('done', 'w') as doneFile:
            doneFile.write(datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
        # doneFile.closed
