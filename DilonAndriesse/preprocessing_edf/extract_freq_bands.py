import matplotlib.pyplot as plt
import numpy as np
import mne


psg_file = "C:/Users/andri/OneDrive - HAN/Desktop/Internship Donders/VSC/data/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf"
anno_file = "C:/Users/andri/OneDrive - HAN/Desktop/Internship Donders/VSC/data/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf"


#Frequency ranges
freq_bands = {
    "noise_band": [0,0.5],
    "delta_band": [0.5,5],
    "theta_band": [6,10],
    "sigma_band": [11,17],
    "beta_band": [22,30],
    "gamma_band": [35,45],
    "total_band": [0,30]
}


def extract_raw(file, loc):
    # read raw edf file
    raw = mne.io.read_raw_edf(
        file,
        # Tries to infer channel types from names
        infer_types=True,
        # loads data into memory for data manipulation and indexing
        preload=True,
    )
    # saves data of EEG location in array
    eeg_loc_data = raw.get_data(picks=loc)

    return eeg_loc_data


def extract_anno(anno_file):
    # extract annotations and create a dataframe
    anno = mne.read_annotations(anno_file, sfreq=100).to_data_frame()


def calc_psd(data, window_len, fs, freq_band):
    all_power_sum = []
    # determine start of each epoch
    for start in range(0, len(data) - window_len + 1, window_len):
        # collects data from array in epoch length
        window = data[start:min(start + window_len, len(data))]
        # calculate psd of frequency band
        psd, _ = mne.time_frequency.psd_array_multitaper(
            window, 
            fs, 
            fmin=freq_band[0], 
            fmax = freq_band[1], 
            n_jobs=-1, 
            verbose = 'warning'
            )
        # get the sum of each epoch
        curr_sum = np.sum(psd)
        # append all sums to a list
        all_power_sum.append(curr_sum)

    return all_power_sum


if __name__ == '__main__':
    extract_anno(anno_file)

    for eeg_loc in ["Fpz-Cz", "Pz-Oz"]:
        eeg_loc_data = extract_raw(psg_file, eeg_loc)
        for key, value in freq_bands.items():
            all_power_sum = calc_psd(
                np.ravel(eeg_loc_data), 
                window_len=1000, 
                fs=100.0, 
                freq_band=value
                )
            np.save(f"test_data/psd_sum/{eeg_loc}_{key}_psd_sum", all_power_sum)