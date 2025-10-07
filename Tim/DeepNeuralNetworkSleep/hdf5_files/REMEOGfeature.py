import mne
import numpy as np

fs = 250  # EEG sampling frequency
window_length = 10 * fs
epoch_length = int(window_length / fs)

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

EOG1, EOG2 = raw.get_data(picks='EOG1')[0], raw.get_data(picks='EOG2')[0]
print(f"length of EOG1: {len(EOG1)}", f"length of EOG1: {len(EOG2)}")