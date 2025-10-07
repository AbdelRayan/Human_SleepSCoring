import mne
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, correlate

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def autocorr_slope(ac):
    # Find peaks around zero-lag
    mid = len(ac) // 2
    left_peak = np.argmax(ac[:mid])
    right_peak = np.argmax(ac[mid+1:]) + mid + 1
    # Compute slope between peaks (simple linear slope)
    return abs(ac[right_peak] - ac[left_peak]) / (right_peak - left_peak)

def crosscorr_peak(x, y):
    cc = correlate(x, y, mode='full')
    mid = len(cc) // 2
    cc /= np.max(np.abs(cc))  # normalize
    return cc[mid]


def eog_features(EOG1, EOG2, fs=250):
    samples_per_epoch = int(fs * 10)  # make sure this is defined!
    n_epochs = len(EOG1) // samples_per_epoch
    feats = {"EOG1": [], "EOG2": [], "EOG3": []}

    for i in range(n_epochs):
        s = i * samples_per_epoch
        e = s + samples_per_epoch
        eog1 = EOG1[s:e]
        eog2 = EOG2[s:e]

        # --- Filters ---
        eog_03_35_1 = bandpass_filter(eog1, 0.3, 35, fs)
        eog_03_35_2 = bandpass_filter(eog2, 0.3, 35, fs)
        eog_01_045_1 = bandpass_filter(eog1, 0.1, 0.45, fs)
        eog_01_045_2 = bandpass_filter(eog2, 0.1, 0.45, fs)
        eog_03_045_1 = bandpass_filter(eog1, 0.3, 0.45, fs)
        eog_03_045_2 = bandpass_filter(eog2, 0.3, 0.45, fs)

        # --- Correlations ---
        ac1_03_35 = correlate(eog_03_35_1, eog_03_35_1, mode='full')
        ac2_03_35 = correlate(eog_03_35_2, eog_03_35_2, mode='full')
        mAC_03_35 = (autocorr_slope(ac1_03_35) + autocorr_slope(ac2_03_35)) / 2
        CCpeak_03_35 = crosscorr_peak(eog_03_35_1, eog_03_35_2)

        ac1_01_045 = correlate(eog_01_045_1, eog_01_045_1, mode='full')
        ac2_01_045 = correlate(eog_01_045_2, eog_01_045_2, mode='full')
        meanAC_01_045 = (np.mean(np.abs(ac1_01_045)) + np.mean(np.abs(ac2_01_045))) / 2
        CCpeak_01_045 = crosscorr_peak(eog_01_045_1, eog_01_045_2)

        ac1_03_045 = correlate(eog_03_045_1, eog_03_045_1, mode='full')
        ac2_03_045 = correlate(eog_03_045_2, eog_03_045_2, mode='full')
        mAC_03_045 = (autocorr_slope(ac1_03_045) + autocorr_slope(ac2_03_045)) / 2
        CCpeak_03_045 = crosscorr_peak(eog_03_045_1, eog_03_045_2)

        # --- Feature formulas ---
        feat_EOG1 = (1 / mAC_03_35) * np.sign(CCpeak_03_35)
        feat_EOG2 = meanAC_01_045 * CCpeak_01_045
        feat_EOG3 = (1 - mAC_03_045) * CCpeak_03_045

        feats["EOG1"].append(feat_EOG1)
        feats["EOG2"].append(feat_EOG2)
        feats["EOG3"].append(feat_EOG3)

    return feats

fs = 250  # EEG sampling frequency
window_length = 10 * fs
epoch_length = int(window_length / fs)
samples_per_epoch = int(fs * window_length)

files = [
    "C:/EEG_Data_stage/2/iEEG/converted_uni_and_bi/2_night1_01.vhdr",
    "C:/EEG_Data_stage/2/iEEG/converted_uni_and_bi/2_night1_02.vhdr",
    "C:/EEG_Data_stage/2/iEEG/converted_uni_and_bi/2_night1_03.vhdr"
]

raw_list = [mne.io.read_raw_brainvision(f, preload=True) for f in files]
raw = mne.concatenate_raws(raw_list)
hpc_data = raw.get_data(picks='Oz-Cz')[0]
hpc_tag = 'Oz-Cz'
raw_hpc = np.ravel(hpc_data)
pfc_data = raw.get_data(picks='C3-Cz')[0]
pfc_tag = 'C3-Cz'
raw_pfc = np.ravel(pfc_data)

scores_files = [
    "C:/EEG_Data_stage/2/iEEG/edf/2_night1_01_hypnogram.npy",
    "C:/EEG_Data_stage/2/iEEG/edf/2_night1_02_hypnogram.npy",
    "C:/EEG_Data_stage/2/iEEG/edf/2_night1_03_hypnogram.npy"
]

score_list = [np.load(f) for f in scores_files]
states = np.concatenate(score_list)
# states = np.append(states, [0,0,0])
print(f"Raw length: {len(raw_hpc)}")
print(f"Raw length / 250: {len(raw_hpc) / 250}")
print(f"Raw time in hours: {len(raw_hpc) / 250 / 3600}")
print(f"Raw epoch count: {len(raw_hpc) / 250 / 10}")
print(f"Sleep state amount: {len(states)}")

EMG = raw.get_data(picks='EMG1-EMG2')[0]
EMG = EMG[:len(EMG)// (epoch_length * fs) * (epoch_length * fs)]
EMG = EMG.reshape(-1, (epoch_length * fs))
EMG = EMG.sum(axis=1)

EOG1, EOG2 = np.ravel(raw.get_data(picks='EOG1')[0]), np.ravel(raw.get_data(picks='EOG2')[0])
print(f"length of EOG1: {len(EOG1)}", f"length of EOG1: {len(EOG2)}")

features = eog_features(EOG1, EOG2, fs=250)

epochs = np.arange(len(features["EOG1"]))
time_sec = (epochs / (250 * 10))  # adjust to your epoch duration / sampling

fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(28, 16))

# Plot EOG features
ax1.plot(time_sec, features["EOG1"], label='EOG1', color='black')
# ax1.plot(time_sec, features["EOG2"], label='EOG2', color='blue')
# ax1.plot(time_sec, features["EOG3"], label='EOG3', color='red')
ax1.set_title('EOG features')
ax1.legend()

# Plot sleep stages
min_len = min(len(time_sec), len(states))
ax3.plot(time_sec[:min_len], states[:min_len], label='Mapped scores')
ax3.set_title('Mapped scores')
ax3.set_yticks([0, 1, 2, 3, 4])
ax3.invert_yaxis()
ax3.legend()

plt.tight_layout()
plt.show()