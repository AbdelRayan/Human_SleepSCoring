import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import yasa

import mne

# load in data files
psg_file = "C:/Users/andri/OneDrive - HAN/Desktop/Internship Donders/VSC/data/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf"
anno_file = "C:/Users/andri/OneDrive - HAN/Desktop/Internship Donders/VSC/data/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf"


def read_raw_psg(psg_file):
    """ 
    Read the raw input PSG.edf file and extract the channels to be used.

    Parameters:
    psg_file (str): The path to the PSG.edf file containing the raw data
    
    Returns:
    raw (obj): Object containing the raw measurement data.
    """
    raw = mne.io.read_raw_edf(
        psg_file,
        # Tries to infer channel types from names
        infer_types=True,
        # load data in memory for data manipulation and faster indexing
        preload=True,
        # exclude unused channels
        exclude=["Event marker", "oro-nasal", "rectal"]
    )

    return raw


def extract_annotation(anno_file, raw):
    """
    Extract annotations from hypnogram.edf annotation file.
    Done by following the MNE 'Sleep stage classification from -
    polysomnography (PSG) data' tutorial
    
    Parameters:
    anno_file (str): Path to the hypnogram.edf file containing annotations.
    
    Returns:
    anno (obj): Object containing annotations.
    """
    # extract annotation from file
    anno = mne.read_annotations(anno_file)

    raw.set_annotations(anno, emit_warning=False)

    return anno


def crop_data(anno, raw):
    """
    """
    # crop data for 30 minutes before and after being awake
    cropped_raw = raw.crop(
        anno[1]["onset"] - 30 * 60, 
        anno[-2]["onset"] + 30 * 60
        )
    cropped_anno = anno.crop(
        anno[1]["onset"] - 30 * 60, 
        anno[-2]["onset"] + 30 * 60
        )
    
    # # annotate the data with the changed annotations
    # cropped_raw.set_annotations(anno, emit_warning=False)

    return cropped_raw, cropped_anno


def plot_data(cropped_raw):
    """
    Plot the raw signal data to an interactive plot with colored 
    annotations sections.
    """
    cropped_raw.plot(
        start=0,
        # window size
        duration=300,
        # stops the program from closing if plot is still opened
        block=True,
        scalings=dict(eeg=1e-4, eog=5e-4, emg=1e-6)
    )


def create_sleep_events(raw, epoch):
    """
    """
    # dictionary of the current sleep stages and set ids
    annotation_desc_2_event_id = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    }

    # creates nested list with epoch number and sleep stage
    sleep_events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=epoch
    )

    return sleep_events


def plot_events(sleep_events, raw_data):
    """
    """
    # dictionary with the proper sleep stages and ids
    event_id = {
        "Sleep stage W": 0,
        "Sleep stage N1": 1,
        "Sleep stage N2": 2,
        "Sleep stage N3": 3,
        "Sleep stage R": 4,
    }

    mne.viz.plot_events(
        sleep_events,
        event_id=event_id,
        sfreq=raw_data.info["sfreq"],
        first_samp=sleep_events[0, 0]
    )


def hypnogram_vis(cropped_raw, sleep_events):
    """
    """
    raw_array = []
    cropped_raw.drop_channels(["horizontal", "submental"])
    data = cropped_raw.get_data(units="uV")
    sf = cropped_raw.info["sfreq"]
    chan = cropped_raw.ch_names

    # create array from sleep events with only sleep stages
    for event in sleep_events:
        raw_array.append(int(event[-1]))

    # plot hypnogram
    yasa.plot_hypnogram(raw_array, sf_hypno=float(1/30))


    hypno_up = yasa.hypno_upsample_to_data(
        raw_array, 
        sf_hypno=float(1/30), 
        data=cropped_raw
        )
    yasa.plot_spectrogram(data[chan.index("Fpz-Cz")], sf, hypno_up)

    # stop the program from closing until plots are closed
    #plt.show(block=True)

    return hypno_up


def calc_bandpower(cropped_raw, hypno_up):
    """
    """
    bandpower = yasa.bandpower(cropped_raw, hypno=hypno_up, include=(0, 1, 2, 3, 4))
    yasa.topoplot(bandpower.xs(3)["Delta"])
    yasa.topoplot(bandpower.xs(2)["Theta"])
    yasa.topoplot(bandpower.xs(0)["Gamma"])

    plt.show(block=True)


if __name__ == '__main__':
    epoch = 30

    # visualize the data of the different channels in dataset.
    raw_data = read_raw_psg(psg_file)
    anno_data = extract_annotation(anno_file, raw_data)
    cropped_raw_data, cropped_anno_data = crop_data(anno_data, raw_data)
    #plot_data(cropped_raw_data)

    # visualize the distribution of epochs spend in certain sleep state
    sleep_events = create_sleep_events(cropped_raw_data, epoch)
    #plot_events(sleep_events, raw_data)

    # yasa visualization of hypnogram and spectrogram plot
    hypno_up = hypnogram_vis(raw_data, sleep_events)
    calc_bandpower(cropped_raw_data, hypno_up)

