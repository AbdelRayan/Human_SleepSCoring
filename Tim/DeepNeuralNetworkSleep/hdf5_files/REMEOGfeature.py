import mne
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt

def wei_normalizing(data):
    data = np.array(data)

    bottom = data[data <= np.nanpercentile(data, 10, axis=0)]
    top = data[data >= np.nanpercentile(data, 90, axis=0)]

    bottom_avg = np.average(bottom) if len(bottom) > 0 else 0
    top_avg = np.average(top) if len(top) > 0 else 1

    denom = top_avg - bottom_avg if top_avg != bottom_avg else 1
    normalized_data = (data - bottom_avg) / denom
    normalized_data = np.clip(normalized_data, 0.05, 1)

    return normalized_data

def cross_correlation(EOG1, EOG2, epoch_length, fs, lag=0):
    samples = int(epoch_length * fs)
    cross_cor_epochs = []

    for i in range(0, len(EOG1) - len(EOG1) % samples, samples):
        epoch1 = EOG1[i:i + samples]
        epoch2 = EOG2[i:i + samples]
        cc_full = np.correlate(epoch1, epoch2, mode='full')
        center = len(cc_full) // 2
        cross_cor_epochs.append(cc_full[center + lag])

    return cross_cor_epochs

def auto_correlation(EOG, epoch_length, fs):
    samples = int(epoch_length * fs)
    slopes = []

    for i in range(0, len(EOG) - len(EOG) % samples, samples):
        epoch = EOG[i:i + samples]
        ac = np.correlate(epoch, epoch, mode='full')
        center = len(ac) // 2
        ac = ac[center:]
        peaks, _ = find_peaks(ac)
        if len(peaks) < 1:
            slopes.append(0)
            continue
        first_peak_idx = peaks[0]
        slope = (ac[first_peak_idx] - ac[0]) / first_peak_idx
        slopes.append(slope)

    return slopes

def rem_feature(EOG1, EOG2, epoch_length, fs):
    rem_features = np.array([])
    b, a = butter(4, [0.3 / (0.5 * fs), 35 / (0.5 * fs)], btype='band')
    EOG1 = filtfilt(b, a, EOG1)
    EOG2 = filtfilt(b, a, EOG2)
    cross_cor_val = cross_correlation(EOG1, EOG2, epoch_length, fs, 0)
    auto_slope = auto_correlation(EOG1, epoch_length, fs)
    for count, slope in enumerate(auto_slope):
        eog_feature = (1/slope)*np.sign(cross_cor_val[count])
        rem_features = np.append(rem_features, eog_feature)
    return rem_features



if __name__ == "__main__":
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
    EMG = EMG[:len(EMG) // (epoch_length * fs) * (epoch_length * fs)]
    EMG = EMG.reshape(-1, (epoch_length * fs))
    EMG = EMG.sum(axis=1)

    EOG1, EOG2 = np.ravel(raw.get_data(picks='EOG1')[0]), np.ravel(raw.get_data(picks='EOG2')[0])
    print(f"length of EOG1: {len(EOG1)}", f"length of EOG1: {len(EOG2)}")

    rem_features = rem_feature(EOG1, EOG2, epoch_length, fs)

    print(len(rem_features))

    eog_epochs = np.arange(len(rem_features))
    hypno_epochs = np.arange(len(states))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(28, 10))

    # Colorblind-friendly colors: Blue and Orange
    ax1.plot(eog_epochs, rem_features, color='#0072B2', linewidth=2, label='REM Feature')
    ax2.plot(hypno_epochs, states, color='#E69F00', linewidth=2, label='Sleep Stage')

    # Titles and labels
    ax1.set_ylabel('REM Feature Value', fontsize=14)
    ax2.set_ylabel('Sleep Stage', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=14)

    remapped_states = np.array([0 if s == 0 else 1 if s == 4 else s + 1 for s in states])
    ax2.clear()
    ax2.plot(hypno_epochs, remapped_states, color='#E69F00', linewidth=2, label='Sleep Stage')

    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['Wake', 'REM', 'N1', 'N2', 'N3'])
    ax2.invert_yaxis()  # Keep standard hypnogram convention (deepest sleep at bottom)

    # Gridlines
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Legends
    ax1.legend(fontsize=12)
    ax2.legend(fontsize=12)

    # Overall title
    fig.suptitle("REM EOG Feature Visualization (REM Under Wake)", fontsize=18, y=0.95)

    # Tight layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    fig.savefig('rem_feature_reordered.svg', format="svg")

    plt.show()