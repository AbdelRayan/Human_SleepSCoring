import scipy
import mne
import numpy as np


def read_raw_psg(psg_file):
    """ 
    Loads the PSG.edf file.

    Parameters:
    psg_file (str): The path to the PSG.edf file.
    
    Returns:
    mne_obj (obj): Object containing the PSG measurement data.
    """
    mne_obj = mne.io.read_raw_edf(
        psg_file,
        # Tries to infer channel types from names
        infer_types=True,
        # load data in memory for data manipulation and faster indexing
        # can be set to True if you're working with smaller files
        preload=False
    )

    return mne_obj


def add_annotation(anno_file, mne_obj):
    """
    Extract annotations from hypnogram.edf annotation file and adds 
    annotation to the raw object.
    Done by following the MNE 'Sleep stage classification from -
    polysomnography (PSG) data' tutorial
    
    Parameters:
    anno_file (str): Path to the hypnogram.edf file containing 
    annotations.
    mne_obj (obj): Object containing measurement data
    
    Returns:
    anno (obj): Object containing annotations.
    """
    # extract annotation from file
    anno = mne.read_annotations(anno_file)

    # annotate raw
    mne_obj.set_annotations(anno, emit_warning=False)

    return anno


def crop_data(mne_obj, wake_time):
    """ 
    Crop the raw object data to keep only the given amount of 
    wake time (e.g. 30 min) before the first sleep and after the 
    last sleep.

    Parameters:
    mne_obj (obj): Object containing the measurement data.
    wake_time (int): Integer containing the amount of seconds to use 
    before and after first sleep

    Returns:
    cropped_mne_obj (obj): Object containing cropped raw measurement data.
    """
    # get annotations from annotated raw object
    anno = mne_obj.annotations

    # get list of descriptions (sleep stages) from the annotation
    desc_list = anno.description

    # create list of booleans for when a sleep stage is 
    # considered not asleep (True when sleep, False when not)
    sleep_bool = np.array([
        d != "Sleep stage W" and 
        d != "Sleep stage ?" 
        for d in desc_list
        ]
    )

    # create list with indexes
    sleep_idx = np.where(sleep_bool)[0]

    # get index of first sleep stage
    first_sleep = sleep_idx[0]
    # get index of last sleep stage
    last_sleep  = sleep_idx[-1]

    # get starting time of first sleep stage
    first_sleep_start = anno.onset[first_sleep]
    # get ending time of last sleep stage
    last_sleep_end = anno.onset[last_sleep] + anno.duration[last_sleep]

    print(f"First sleep starts at: {first_sleep_start}")
    print(f"Last sleep ends at: {last_sleep_end}")

    # get the maximum value between wake time before first sleep stage 
    # and start time 
    # (to avoid ValueErrors where tmax is larger than max time)
    crop_start = max(0, first_sleep_start - wake_time)

    # get the minimum value between wake time after last sleep stage
    # and the end time of raw object
    crop_end = min(mne_obj.times[-1], last_sleep_end + wake_time)

    print(f"Cropping raw: {crop_start} - {crop_end}")

    # crop data for wake time before first sleep and after last sleep
    try:
        cropped_mne_obj = mne_obj.crop(
            crop_start, 
            crop_end
            )
        print("Cropping finished.")
    except ValueError as e:
        print(f"An error has occured: {e}")
        print("Could not crop data.")
    
    return cropped_mne_obj


def create_sleep_events(cropped_mne_obj, chunk_duration):
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
    # this works for Sleep-EDF dataset
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
        cropped_mne_obj, event_id=annotation_desc_2_event_id, chunk_duration=chunk_duration
    )

    return sleep_events


def preprocess_egi(mne_obj, highpass, lowpass):
    """
    Basic pre-processing of the raw data to remove slowdrift and 
    low frequency noise as well as high frequencies, 
    because we are interested in frequency bands up to 45 Hz.

    Parameters:
    raw (obj): raw mne object with EGI data
    highpass (int): Value for lower cutoff bandpass
    lowpass (int): Value for higher cutoff bandpass
    """
    # load data into memory (otherwise you can't apply filters)
    mne_obj.load_data()

    # apply detrend to all channels in raw object
    # the use of lambda was suggested by ChatGPT after running into 
    # errors before
    mne_obj.apply_function(lambda x: mne.filter.detrend(x, axis=0, order=1), picks="all")

    # apply bandpass filter to all channels in data
    mne_obj.filter(l_freq=highpass, h_freq=lowpass, picks="all")


def create_mat(output_path, subject, ch_pb, data):
    """
    Create a .mat file from channel data with channel specific name.
    
    Parameters:
        output_path (str): Filepath to output directory.
        subject (str): Subject name.
        ch_pb (str): addition variable name.
        data (array): data to save to .mat file.
    """
    import os
    
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Validate data
    data = np.asarray(data)
    
    # Check for empty or all-NaN data
    if data.size == 0:
        print(f"Warning: Empty data for {subject}_{ch_pb}")
        return
    
    data_flat = np.ravel(data)
    if np.all(np.isnan(data_flat)):
        print(f"Warning: All NaN data for {subject}_{ch_pb}")
    
    # try to reshape 
    try:
        scipy.io.savemat(
                f"{output_path}/{subject}_{ch_pb}.mat", 
                {ch_pb: data.reshape((-1,1))}
                )
    except (AttributeError, ValueError) as e:
        # For 1D arrays or special cases like sleep stages
        print(f"Saving {subject}_{ch_pb} without reshape")
        scipy.io.savemat(
                f"{output_path}/{subject}_{ch_pb}.mat", 
                {ch_pb: data}
                )


def get_stages(raw, stage_id):
    """
    Get sleep states from EEG object and convert to numeric sleep state system.
    
    Parameters:
        raw (obj): raw mne object with EGI data.
        stage_id (dict): Dictionary with sleep state (str) and 
                         corresponding sleep state (int).

    Returns:
        array: Sleep states as numeric values.
    """
    fs = raw.info["sfreq"]
    sleep_states = []
    
    n_samples = raw.n_times

    # iterate annotations
    for anno in raw.annotations:
        stage = str(anno["description"])
        duration_sample = int(anno["duration"] * fs)

        # add sleep states (int) to list
        if stage in stage_id:
            sleep_states.extend([stage_id[stage]] * duration_sample)
        # unknown states are set to 5 for artefact
        else:
            sleep_states.extend([5] * duration_sample)
    
    # Validate length matches signal length
    if len(sleep_states) != n_samples:
        print(f"Warning: Sleep states length ({len(sleep_states)}) != signal length ({n_samples})")
        # Pad or truncate to match
        if len(sleep_states) < n_samples:
            sleep_states.extend([5] * (n_samples - len(sleep_states)))
        else:
            sleep_states = sleep_states[:n_samples]
        print(f"  Adjusted sleep states to {len(sleep_states)} samples")

    return sleep_states
