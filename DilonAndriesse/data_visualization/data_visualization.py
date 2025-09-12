import matplotlib.pyplot as plt
import yasa
import mne


def read_raw_psg(psg_file):
    """ 
    Read the raw input PSG.edf file and extract the channels to be used.

    Parameters:
    psg_file (str): The path to the PSG.edf file containing the raw data.
    
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
    anno_file (str): Path to the hypnogram.edf file containing 
    annotations.
    
    Returns:
    anno (obj): Object containing annotations.
    """
    # extract annotation from file
    anno = mne.read_annotations(anno_file)

    raw.set_annotations(anno, emit_warning=False)

    return anno


def crop_data(anno, raw):
    """ 
    Crops the data to remove most of the measurement from being awake.

    Parameters:
    anno (obj): Object containing annotations.
    raw (obj): Object containing the raw measurement data.

    Returns:
    cropped_anno (obj): Object containing annotations.
    cropped_raw (obj): Object containing the raw measurement data.
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

    return cropped_raw, cropped_anno


def plot_data(cropped_raw):
    """
    Plot the raw signal data to an interactive plot with colored 
    annotations sections.

    Parameters:
    cropped_raw (obj): Object containing the raw measurement data.
    """
    cropped_raw.plot(
        start=0,
        # window size
        duration=300,
        # custom scalings
        scalings=dict(eeg=1e-4, eog=5e-4, emg=1e-6)
    )


def create_sleep_events(cropped_raw, epoch):
    """
    Get sleep events from the data with the sleep stages we are 
    interested in.

    Parameters:
    cropped_raw (obj): Object containing the raw measurement data.
    epoch (int): Integer representing epoch length.

    Returns:
    sleep_events (2d-list): A nested list with epoch index, 
    and sleep stage id.
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
        cropped_raw, event_id=annotation_desc_2_event_id, chunk_duration=epoch
    )

    return sleep_events


def plot_events(sleep_events, cropped_raw):
    """
    Creates a plot to give an overview of the distribution of 
    sleep stages.

    Parameters:
    sleep_events (2d-list): A nested list with epoch index, 
    and sleep stage id.
    cropped_raw (obj): Object containing the raw measurement data.
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
        sfreq=cropped_raw.info["sfreq"],
        first_samp=sleep_events[0, 0]
    )


def hypnogram_vis(cropped_raw, sleep_events):
    """
    Plot a hypnogram and corresponding spectrogram.

    Parameters:
    cropped_raw (obj): Object containing the raw measurement data.
    sleep_events (2d-list): A nested list with epoch index, 
    and sleep stage id.

    Returns:
    hypno_up (list): List with the upscaled annotation
    """
    raw_array = []
    # remove non-EEG channels
    cropped_raw.drop_channels(["horizontal", "submental"])
    # units to use
    data = cropped_raw.get_data(units="uV")
    # sampling frequency
    sf = cropped_raw.info["sfreq"]
    # channel names
    chan = cropped_raw.ch_names

    # create array from sleep events with only sleep stages
    for event in sleep_events:
        raw_array.append(int(event[-1]))

    # plot hypnogram
    yasa.plot_hypnogram(raw_array, sf_hypno=float(1/30))

    # upsample the annotation to the data
    hypno_up = yasa.hypno_upsample_to_data(
        raw_array, 
        sf_hypno=float(1/30), 
        data=cropped_raw
        )
    # plot spectogram
    yasa.plot_spectrogram(data[chan.index("Fpz-Cz")], sf, hypno_up)

    return hypno_up


def calc_bandpower(cropped_raw, hypno_up):
    """
    Calculate and plot the band powers to the two EEG channels.

    Parameters:
    cropped_raw (obj): Object containing the raw measurement data.
    hypno_up (list): List with the upscaled annotation
    """
    bandpower = yasa.bandpower(cropped_raw, hypno=hypno_up, include=(0, 1, 2, 3, 4))
    yasa.topoplot(bandpower.xs(3)["Delta"])
    yasa.topoplot(bandpower.xs(2)["Theta"])
    yasa.topoplot(bandpower.xs(0)["Gamma"])
