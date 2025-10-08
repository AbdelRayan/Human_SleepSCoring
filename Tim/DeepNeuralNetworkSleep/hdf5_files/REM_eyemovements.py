import mne
import numpy as np
from matplotlib import pyplot as plt

fs = 250
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

scores_files = [
    "C:/EEG_Data_stage/2/iEEG/edf/2_night1_01_hypnogram.npy",
    "C:/EEG_Data_stage/2/iEEG/edf/2_night1_02_hypnogram.npy",
    "C:/EEG_Data_stage/2/iEEG/edf/2_night1_03_hypnogram.npy"
]

score_list = [np.load(f) for f in scores_files]
states = np.concatenate(score_list)

picks = ['EOG1', 'EOG2', 'EOG1-EOG2']
raw.pick_channels(picks)

rem_indices = np.where(states == 4)[0]

rem_mask = np.zeros(len(raw.times), dtype=bool)
for idx in rem_indices:
    start = idx * samples_per_epoch
    stop = start + samples_per_epoch
    if stop > len(rem_mask):
        stop = len(rem_mask)
    rem_mask[start:stop] = True

rem_raw = raw.copy().crop(tmin=0, tmax=len(rem_mask)/fs)
rem_data = rem_raw.get_data()
rem_data = rem_data[:, rem_mask]
rem_raw = mne.io.RawArray(rem_data, rem_raw.info)

print(rem_raw)
rem_raw.plot(scalings='auto', title='REM segments: EOG1, EOG2, EOG1-EOG2')