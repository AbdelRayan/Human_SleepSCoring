import mne
import matplotlib.pyplot as plt
import yasa
import edfio
import numpy as np
from mne.export import export_raw
import pandas as pd
import Sleep_Scripts.Practical_scripts as P

files = [
    "D:/EEG_Data_stage/135/iEEG/converted_test_electrodes/135_night1_02.vhdr"
]

raw_list = [mne.io.read_raw_brainvision(f, preload=True) for f in files]

raw = mne.concatenate_raws(raw_list)
print(raw.ch_names)

# scores_files = [
#     "D:/EEG_Data_stage/2/iEEG/edf/2_night1_02_hypnogram.npy",
#     "D:/EEG_Data_stage/2/iEEG/edf/2_night1_03_hypnogram.npy"
# ]
#
# score_list = [np.load(f) for f in scores_files]
# hypno = np.concatenate(score_list)

# raw = mne.io.read_raw_brainvision("D:/Intercranial_sleep_data/2/iEEG/converted/2_night1_02.vhdr")

# raw = mne.io.read_raw_edf("2_night1_03.edf")
# hypno = np.load("D:/converted_sleep_data/2/combined_nights/edf_files/night1_hypnogram_30s.npy")
# hypno = hypno.astype(int)
# eeg_duration = raw.n_times / raw.info['sfreq']
# n_epochs = int(eeg_duration // (10))  # assuming 30s epochs
# hypno = hypno[:n_epochs]            # crop to EEG length if needed
# stage_mapping = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
# description = [stage_mapping[s] for s in hypno]
# onset = np.arange(len(hypno)) * 10  # 30s per epoch
# duration = np.repeat(10, len(hypno))
# #
# annotations = mne.Annotations(onset, duration, description)
# raw.set_annotations(annotations)
#
# raw = P.convert_channel_types(raw)
#
#
# fig = raw.plot(duration=120,
#     scalings=dict(eeg=1e-4)             # interactive plotting
# )
# fig.subplots_adjust(top=0.9)
# plt.show(block=True)
# raw = mne.io.read_raw_brainvision("D:/converted_sleep_data/2/con_full/2_night1_02.vhdr")
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

#

# np.save("D:/converted_sleep_data/2/stages/night1_full_score.npy", full_score)
# raw.set_channel_types({
#     'C3-Cz':'eeg',
#     'Oz-Cz':'eeg',
#     'EOG1-EOG2':'eog',
#     'EMG1':'emg'
# })
# raw = P.convert_channel_types(raw.drop_channels(['Cb', 'T', 'C']))
#
# sls = yasa.SleepStaging(raw, eeg_name="Oz-Cz", eog_name="EOG1-EOG2", emg_name="EMG1-EMG2")
# #
# hypno_pred = sls.predict()  # Predict the sleep stages
# #
# hypno_pred = yasa.hypno_str_to_int(hypno_pred)  # Convert "W" to 0, "N1" to 1, etc
# #
# # yasa.plot_hypnogram(hypno_pred)
# # plt.show(block=True)
#
# # Sampling frequency
# sf = raw.info["sfreq"]
#
# # Each epoch is 30 seconds long by default
# epoch_length = 30
#
# # Create onset times for each epoch
# onset = np.arange(0, len(hypno_pred) * epoch_length, epoch_length)
#
# # Map stage integers back to readable strings for annotations
# stage_map = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
# descriptions = [stage_map[int(stage)] for stage in hypno_pred]
#
# # Create MNE Annotations
# annotations = mne.Annotations(
#     onset=onset,
#     duration=[epoch_length] * len(hypno_pred),
#     description=descriptions
# )
#
# # Add annotations to raw
# raw.set_annotations(annotations)
########## fragment vis ###############################
# Define the window of interest
# Fragment window
# tmin = 0
# tmax = tmin + 60 # 30-second fragment
# sfreq = raw.info['sfreq']
# data, times = raw[:, int(tmin * sfreq):int(tmax * sfreq)]
#
# # Shift time to start at 0
# times = times - tmin
#
# # Scale to µV
# data = data * 1e6
#
# plt.figure(figsize=(12, 6))
#
# # Vertical spacing between channels
# offset = 100
# yticks = []
# scale = 0.5
# data_scaled = data * scale  # data is in µV
# for idx, ch_name in enumerate(raw.info['ch_names']):
#     ch_offset = idx * offset
#     plt.plot(
#         times,
#         data_scaled[idx] + ch_offset,
#         linewidth=0.4,
#         color="black"
#     )
#     yticks.append(ch_offset)
#
# # Label y-axis with channel names
# plt.yticks(yticks, raw.info['ch_names'])
# plt.xlabel("Time (s)")
# plt.ylabel("Channels")
# plt.title(f"EEG fragment 0-30 s, U-Sleep = n2 ")
#
# # --- Add scale bars (50 µV) ---
# scalebar_height = 50 * scale  # µV
# scalebar_x = 1  # 1 s into the fragment
#
# for ch_offset in yticks:  # one bar per channel
#     plt.plot([scalebar_x, scalebar_x],
#              [ch_offset, ch_offset + scalebar_height],
#              color="red", linewidth=2)
#     plt.text(scalebar_x + 0.2,
#              ch_offset + scalebar_height / 2,
#              f"{scalebar_height} µV",
#              va="center", ha="left", color="red")
#
# # Top channel
# scalebar_y_top = yticks[-1]
# plt.plot([scalebar_x, scalebar_x], [scalebar_y_top, scalebar_y_top + scalebar_height],
#          color="red", linewidth=2)
# plt.text(scalebar_x + 0.2, scalebar_y_top + scalebar_height / 2,
#          f"{scalebar_height} µV", va="center", ha="left", color="red")
#
# plt.xlim(0, tmax - tmin)  # set x-axis from 0 to 30
# plt.tight_layout()
# plt.savefig("possible_error_frag_100s.svg", dpi=300, format="svg")
# plt.show(block=True)
# plt.close()

##########################################################################
# #
# fig = raw.plot(start=3570, duration=30,
#     scalings=dict(eeg=1e-4)             # interactive plotting
# )
# fig.subplots_adjust(top=0.9)
# plt.show(block=True)


# raw.export("D:/converted_sleep_data/2/edf/2_night1_02.edf", fmt='edf', physical_range=(-200, 200))



# raw = mne.io.read_raw_edf("D:/converted_sleep_data/2/edf/night1_full.edf")
#

# Sampling frequency
# print("Sampling frequency:", raw.info["sfreq"], "Hz")

# Number of samples (time points)
# print("Samples:", raw.n_times)


# hypno = hypno_pred
# chan = raw.ch_names
# sf = raw.info["sfreq"]
# data = raw.get_data(picks="eeg", units="uV")
# print(sf)
# print(len(hypno))
# print(np.unique(hypno))
# print(chan)
# hypno_up = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=raw)
# yasa.plot_spectrogram(data[chan.index("C3-Cz")], sf, hypno_up)
# # plt.savefig("YASA_hypno_spectro.svg", format="svg")
# plt.show(block=True)
# yasa.plot_hypnogram(hypno)
# plt.show(block=True)

yasa.bandpower(raw).to_csv("135_night1.csv", index=True)
# bandpower = yasa.bandpower(raw, hypno=hypno_up, include=(2, 3, 4))
# bandpower.to_csv("D:/converted_sleep_data/2/con_full/excel/2_night1_02_stages.csv", index=True)
#
# delta_power = bandpower.xs(3)["Gamma"]
#
# delta_power.plot(kind="bar", figsize=(12, 4))
# plt.ylabel("Delta power (µV²/Hz)")
# plt.xlabel("Channels")
# plt.show()
# raw.pick_channels(['Cz', 'Oz', 'T5', 'C3', 'C4', 'T6'])
# raw.plot(duration=300, scalings=dict(eeg=1e-4), block=True)
# bandpower = yasa.bandpower(raw, hypno=hypno_up, include=(2, 3, 4))
# fig = yasa.topoplot(bandpower.xs(3)["Theta"])
# plt.show(block=True)