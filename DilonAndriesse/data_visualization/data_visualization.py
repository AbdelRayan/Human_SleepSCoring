import yasa
import mne


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
