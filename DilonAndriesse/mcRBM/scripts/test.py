from scipy.io import loadmat
import numpy as np

data = loadmat(r"C:\Users\andri\school\bio-informatics\internship\donders\vsc\Human_SleepSCoring\DilonAndriesse\mcRBM\sample_data\input\states_sleep-edf.mat")

print(np.unique(data['states']))