import mne
import matplotlib.pyplot as plt
import yasa
import edfio
import numpy as np
from mne.export import export_raw
import pandas as pd

# files = [
#     "D:/converted_sleep_data/2/con_full/2_night1_01.vhdr",
#     "D:/converted_sleep_data/2/con_full/2_night1_02.vhdr",
#     "D:/converted_sleep_data/2/con_full/2_night1_03.vhdr"
# ]
#
# raw_list = [mne.io.read_raw_brainvision(f, preload=True) for f in files]
#
# raw = mne.concatenate_raws(raw_list)

raw = mne.io.read_raw_brainvision("D:/converted_sleep_data/2/con_full/2_night1_02.vhdr")
# raw_full.save("night1_full-raw.fif", overwrite=True)
#
# out_dir = "D:/converted_sleep_data/2/con_full/edf/"
# duration = raw_full.times[-1]  # total recording duration in seconds
# chunk_length = 1800  # 30 min chunks
#
# for i, start in enumerate(range(0, int(duration), chunk_length)):
#     stop = min(start + chunk_length, int(duration))
#     raw_chunk = raw_full.copy().crop(tmin=start, tmax=stop)
#     fname = f"{out_dir}/night1_full_part{i+1}.edf"
#     raw_chunk.export(fname, fmt="edf")
#     print(f"Saved {fname}")
#
#
# data = "D:/converted_sleep_data/2/2_night1_02.vhdr"
#
# raw = mne.io.read_raw_brainvision(data)
# print(raw.get_channel_types())
# print(raw.ch_names)
#

#
#
# scores_files = [
#     "D:/converted_sleep_data/2/stages/2_night1_01_hypnogram.npy",
#     "D:/converted_sleep_data/2/stages/2_night1_02_hypnogram.npy",
#     "D:/converted_sleep_data/2/stages/2_night1_03_hypnogram.npy"
# ]
#
# score_list = [np.load(f) for f in scores_files]
# full_score = np.concatenate(score_list)
#

# np.save("D:/converted_sleep_data/2/stages/night1_full_score.npy", full_score)

raw.set_channel_types({
    'TBAR1': 'eeg',
    'TBAR2': 'eeg',
    'TBAR3': 'eeg',
    'TBAR4': 'eeg',
    'TBPR1': 'eeg',
    'TBPR2': 'eeg',
    'TBPR3': 'eeg',
    'TBPR4': 'eeg',
    'TLR01': 'eeg',
    'TLR02': 'eeg',
    'TLR03': 'eeg',
    'TLR04': 'eeg',
    'TLR05': 'eeg',
    'TLR06': 'eeg',
    'TBAL1': 'eeg',
    'TBAL2': 'eeg',
    'TBAL3': 'eeg',
    'TBAL4': 'eeg',
    'TBPL1': 'eeg',
    'TBPL2': 'eeg',
    'TBPL3': 'eeg',
    'TBPL4': 'eeg',
    'TLL01': 'eeg',
    'TLL02': 'eeg',
    'TLL03': 'eeg',
    'TLL04': 'eeg',
    'TLL05': 'eeg',
    'TLL06': 'eeg',
    'TR01': 'eeg',
    'TR02': 'eeg',
    'TR03': 'eeg',
    'TR04': 'eeg',
    'TR05': 'eeg',
    'TR06': 'eeg',
    'TR07': 'eeg',
    'TR08': 'eeg',
    'TR09': 'eeg',
    'TR10': 'eeg',
    'TL01':'eeg',
    'TL02':'eeg',
    'TL03':'eeg',
    'TL04':'eeg',
    'TL05':'eeg',
    'TL06':'eeg',
    'TL07':'eeg',
    'TL08':'eeg',
    'TL09':'eeg',
    'TL10':'eeg',
    'EKG1':'ecg',
    'EKG2':'ecg',
    'T5':'eeg',
    'T6':'eeg',
    'C3':'eeg',
    'C4':'eeg',
    'Cz':'eeg',
    'Oz':'eeg',
    'EOG1':'eog',
    'EOG2':'eog',
    'EMG1':'emg',
    'EMG2':'emg'
})



# sls = yasa.SleepStaging(raw, eeg_name="Cz")
#
# hypno_pred = sls.predict()  # Predict the sleep stages
#
# hypno_pred = yasa.hypno_str_to_int(hypno_pred)  # Convert "W" to 0, "N1" to 1, etc
#
# yasa.plot_hypnogram(hypno_pred)
# plt.show(block=True)

# raw.export("D:/converted_sleep_data/2/edf/2_night1_02.edf", fmt='edf', physical_range=(-200, 200))

# raw.plot(block=True, scalings=dict(eeg=1e-4))

# raw = mne.io.read_raw_edf("D:/converted_sleep_data/2/edf/night1_full.edf")
#
# raw.set_channel_types({
#     'Cz':'eeg',
#     'EOG1':'eog'
# })
# Sampling frequency
# print("Sampling frequency:", raw.info["sfreq"], "Hz")

# Number of samples (time points)
# print("Samples:", raw.n_times)
scores = "D:/converted_sleep_data/2/stages/2_night1_02_hypnogram.npy"
#
hypno = np.load(scores)
# chan = raw.ch_names
# sf = raw.info["sfreq"]
# data = raw.get_data(picks="eeg", units="uV")
# print(sf)
# print(len(hypno))
# print(np.unique(hypno))
# print(chan)
hypno_up = yasa.hypno_upsample_to_data(hypno, sf_hypno=4/30, data=raw)
# yasa.plot_spectrogram(data[chan.index("Cz")], sf, hypno_up)
# plt.show(block=True)
# yasa.plot_hypnogram(hypno)
# plt.show(block=True)

# yasa.bandpower(raw).to_csv("D:/converted_sleep_data/2/con_full/excel/2_night1_02.csv", index=True)
# bandpower = yasa.bandpower(raw, hypno=hypno_up, include=(2, 3, 4))
# bandpower.to_csv("D:/converted_sleep_data/2/con_full/excel/2_night1_02_stages.csv", index=True)
#
# delta_power = bandpower.xs(3)["Gamma"]
#
# delta_power.plot(kind="bar", figsize=(12, 4))
# plt.ylabel("Delta power (µV²/Hz)")
# plt.xlabel("Channels")
# plt.show()