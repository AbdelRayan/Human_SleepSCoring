import sys
print(sys.executable)
print(sys.version)

import os
if "MPLBACKEND" in os.environ:
    del os.environ["MPLBACKEND"]
import sys
import numpy as np
# import pandas as pd
from numpy.random import RandomState
from scipy.io import loadmat, savemat
from configparser import ConfigParser
import PIL.Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
def inspect_npz(filepath):
    print(f"Inspecting: {filepath}\n")

    data = np.load(filepath)
    print("Keys:", data.files)

    for key in data.files:
        arr = data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

class GetStates(object):
    def __init__(self, refDir, expDoneFlag, modelDir, finalModel):
        self.refDir = refDir
        self.expDoneFlag = expDoneFlag
        print(f"finalModel: {finalModel}")
        self.model = finalModel
        self.modelDir = modelDir

        np.random.seed(124)
        self.prng = RandomState(123)

        self.saveDir = self.refDir



    def loadData(self, statesFilePath, statesFile):
        os.chdir(self.saveDir)
        print("Analysing experiment : ", os.getcwd())

        visData = 'visData.npz'
        inspect_npz(visData)
        dataFile = np.load(visData)
        self.d = dataFile['data']
        self.obsKeys = dataFile['obsKeys'].astype(int)
        self.states = loadmat(f'{statesFilePath}{statesFile}')
        print("dataFile zeik:", dataFile["data"].shape)
        print(f"STATES: ", self.states['states'][0].shape)
        self.states['downsampledStates'] = self.states.pop('states')

    def computeStates(self):
        p_hc, p_hm = self.hidden_activation()
        self.p_all = np.concatenate((p_hc, p_hm), axis=1)
        # print("p_hc :", p_hc)
        # np.savez("p_hc", p_hc)
        # np.savez("p_hm", p_hm)
        if not os.path.isdir('analysis'):
            os.makedirs('analysis')
        os.chdir('analysis')


        if not os.path.isdir('epoch%d' % self.epochID):
            os.makedirs('epoch%d' % self.epochID)

        os.chdir('epoch%d' % self.epochID)
        print("Storing in...", os.getcwd())

        if not os.path.isdir('hcActivation'):
            os.makedirs('hcActivation')
        if not os.path.isdir('hmActivation'):
            os.makedirs('hmActivation')
        if not os.path.isdir('binaryActivation'):
            os.makedirs('binaryActivation')

        image = PIL.Image.fromarray(np.uint8(p_hc * 255.))
        resized_image = image.resize((1200, 1200))
        image.save('./hcActivation/%i.png' % self.epochID)

        image = PIL.Image.fromarray(np.uint8(p_hm * 255.))
        resized_image = image.resize((1200, 1200))
        resized_image.save('./hmActivation/%i.png' % self.epochID)

        self.binary_latentActivation = (self.p_all >= 0.5).astype(int)
        plt.figure(figsize=(10, 50))
        plt.imshow(self.binary_latentActivation, cmap='gray')
        plt.title('Binary Latent Activations')
        plt.xlabel('Hidden units')
        plt.ylabel('Epoch')
        plt.savefig('./binaryActivation/%i.png' % self.epochID)

        str_repr = np.array([''.join(map(str, row)) for row in self.binary_latentActivation])

        unique_bin, uniqueFramesID, ic = np.unique(str_repr, return_index=True, return_inverse=True)
        uniqueAct = self.binary_latentActivation[uniqueFramesID]
        uniqueCount = np.array([np.sum(ic == i) for i in range(len(uniqueFramesID))])
        p_unique = self.p_all[uniqueFramesID]
        uniqueStates = np.zeros((len(uniqueAct), len(uniqueAct[0]) + 2))
        inferredStates = np.column_stack((
            np.zeros(len(self.binary_latentActivation)),
            self.states['downsampledStates'].astype(int).flatten()[:len(self.binary_latentActivation)]))

        for i in range(len(uniqueAct)):
            uniqueStates[i, 0] = i + 1
            uniqueStates[i, 1] = uniqueCount[i]
            uniqueStates[i, 2:] = uniqueAct[i]

            row_indices = np.where((self.binary_latentActivation == uniqueAct[i]).all(axis=1))[0]

            inferredStates[row_indices, 0] = i + 1

        np.savez_compressed('latentStates.npz',
                            probabilities=self.p_all,
                            binary=self.binary_latentActivation,
                            inferredStates=inferredStates,
                            uniqueStates=uniqueStates)

    def computeUniqueStates(self):
        uniqueAct, p_unique = self.compute_uniques(self.binary_latentActivation, self.p_all)
        print("np.shape(uniqueAct) : ", np.shape(uniqueAct))
        print("np.unique(uniqueAct) :", np.unique(uniqueAct))
        print("uniqueAct", uniqueAct)
        del self.p_all

        print("Checking if there are hidden_units that are always off..")
        print("The sum of the unique latent activations' columns is : ", np.sum(uniqueAct, axis=0))

        with open('latentStatesInfo.txt', 'w') as f:
            f.write("\n The number of the unique latent activations is : %s" % uniqueAct.shape[0])
            f.write("\n The sum of the unique latent activations' columns is : %s" % np.sum(uniqueAct, axis=0))
            f.close()

        uniqueAct2 = np.insert(uniqueAct, 0, 0, axis=1)
        uniqueAct2 = np.insert(uniqueAct2, 0, 0, axis=1)

        self.obsKeys = np.insert(self.obsKeys, 1, 0, axis=1)
        for i in range(uniqueAct.shape[0]):
            temp_idx = np.where(np.all(self.binary_latentActivation == uniqueAct[i, :], axis=1))[0]

            uniqueAct2[i, 0] = i
            uniqueAct2[i, 1] = len(temp_idx)

            self.obsKeys[temp_idx, 1] = i

        np.savez_compressed('uniqueStates.npz', uniqueStates=uniqueAct2, probabilities=p_unique)
        np.savez('obsKeys.npz', obsKeys=self.obsKeys)

    def compute_uniques(self, p_h_bin, p_h):
        tmpUnique = np.unique(p_h_bin.view(np.dtype((np.void, p_h_bin.dtype.itemsize * p_h_bin.shape[1]))),
                              return_index=True, return_counts=True)
        uniqueAct = tmpUnique[0].view(p_h_bin.dtype).reshape(-1, p_h_bin.shape[1])
        uniqueFramesID = tmpUnique[1]
        uniqueFramesID = uniqueFramesID.reshape(-1, 1)
        uniqueCount = tmpUnique[2]
        p_unique = p_h[uniqueFramesID[:, 0], :]

        print("The number of the unique latent activations is :", uniqueAct.shape[0])

        return uniqueAct, p_unique

    def logisticFunc(self, x):
        return 1. / (1. + np.exp(-x))

    def hidden_activation(self):
        # Load model
        if self.expDoneFlag == 'True':
            print(self.modelDir + self.model)
            ws_temp = loadmat(self.modelDir + self.model)
        else:
            temp_model = input("Please enter the training epoch you want to analyze: ")
            ws_temp = loadmat('./weights/ws_temp%d.mat' % int(temp_model))

        # Extract model params
        w_mean = ws_temp['w_mean']  # expected shape: (n_vis, n_mean)
        FH = ws_temp['FH']  # expected shape: (n_fac, n_cov)
        VF = ws_temp['VF']  # expected shape: (n_vis, n_fac)
        bias_cov = ws_temp['bias_cov']  # may be (n_cov,1) or (1,n_cov) or (n_cov,)
        bias_mean = ws_temp['bias_mean']  # may be (n_mean,1) or (1,n_mean) or (n_mean,)
        self.epochID = ws_temp['epoch']

        # --- Normalise data ---
        # We want self.d to have shape (n_vis, n_samples)
        # Model's expected n_vis is w_mean.shape[0] or VF.shape[0]
        n_vis_model = w_mean.shape[0] if w_mean is not None else VF.shape[0]

        # Ensure self.d shape matches (n_vis_model, n_samples)
        d_shape = self.d.shape
        if d_shape[0] == n_vis_model:
            d_mat = self.d  # already (n_vis, n_samples)
        elif d_shape[1] == n_vis_model:
            # transpose if data is (n_samples, n_vis)
            d_mat = self.d.T
            print("Transposed self.d to match model visible dimension.")
        else:
            raise ValueError(
                f"Data dimension mismatch: model expects {n_vis_model} visibles but data shape is {d_shape}.")

        # compute per-sample lengths and normalized data
        dsq = np.square(d_mat)  # (n_vis, n_samples)
        lsq = np.sum(dsq, axis=0)  # (n_samples,)
        lsq = lsq / d_mat.shape[0]  # normalize by n_vis (like original)
        lsq = lsq + np.spacing(1)
        l = np.sqrt(lsq)  # (n_samples,)
        normD = d_mat / l  # broadcasting -> (n_vis, n_samples)

        # --- Covariance pathway ---
        # shapes:
        # VF.T @ normD     -> (n_fac, n_samples)
        # square -> same
        # FH.T @ featsq    -> (n_cov, n_samples)
        feats = VF.T.dot(normD)  # (n_fac, n_samples)
        featsq = np.square(feats)  # (n_fac, n_samples)
        pre_cov = -0.5 * (FH.T.dot(featsq))  # (n_cov, n_samples)

        # normalize bias_cov shape to (n_cov, 1) so broadcast along samples is correct
        bias_cov = np.array(bias_cov).reshape(-1)  # (n_cov,)
        bias_cov = bias_cov[:, np.newaxis]  # (n_cov, 1)

        logisticArg_c = (pre_cov + bias_cov).T  # transpose -> (n_samples, n_cov)

        # Debug prints for covariance pre-activation
        print("logisticArg_c shape:", logisticArg_c.shape)
        print("logisticArg_c min:", np.min(logisticArg_c))
        print("logisticArg_c max:", np.max(logisticArg_c))
        print("logisticArg_c mean:", np.mean(logisticArg_c))
        print("all finite:", np.isfinite(logisticArg_c).all())

        p_hc = self.logisticFunc(logisticArg_c)  # (n_samples, n_cov)

        # --- Mean pathway ---
        # Ensure w_mean shape (n_vis, n_mean)
        # We want dot result shape (n_samples, n_mean)
        # If d_mat is (n_vis, n_samples), do (d_mat.T @ w_mean) -> (n_samples, n_mean)
        dot_mean = d_mat.T.dot(w_mean)  # (n_samples, n_mean)

        # normalize bias_mean to (1, n_mean) for row-wise addition
        bias_mean = np.array(bias_mean).reshape(-1)  # (n_mean,)
        bias_mean_row = bias_mean[np.newaxis, :]  # (1, n_mean)

        logisticArg_m = dot_mean + bias_mean_row  # (n_samples, n_mean)

        # Debug prints for mean pre-activation
        print("logisticArg_m shape:", logisticArg_m.shape)
        print("logisticArg_m min:", np.min(logisticArg_m))
        print("logisticArg_m max:", np.max(logisticArg_m))
        print("logisticArg_m mean:", np.mean(logisticArg_m))
        print("all finite:", np.isfinite(logisticArg_m).all())

        p_hm = self.logisticFunc(logisticArg_m)  # (n_samples, n_mean)

        return p_hc, p_hm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='Experiment path', default='/teamspace/studios/this_studio/mouse-sleep-analysis/sample_data/experiments')
    parser.add_argument('-done', help='Experiment done flag', default=False)
    parser.add_argument('-input', help = 'Input (dataset + manual scoring) path', default='/teamspace/studios/this_studio/mouse-sleep-analysis/sample_data/input')
    parser.add_argument('-md', help = 'model Dir')
    parser.add_argument('-m', help='Saved model name')
    parser.add_argument('-s', help='Saved states file name')
    args = parser.parse_args()
    print('Initialization...')
    print("args.f :", args.f)
    print("args.done :" , args.done)
    print("args.md", args.md)
    print("args.input", args.input)
    print("args.m :", args.m)
    print("args.s :", args.s)
    model = GetStates(args.f, args.done, args.md, args.m)


    print('Loading data...')
    model.loadData(args.input, args.s)

    print('Computing latent states...')
    model.computeStates()

    print('Computing the unique binary latent states...')
    model.computeUniqueStates()
