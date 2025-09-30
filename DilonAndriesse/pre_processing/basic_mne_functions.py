import mne
import numpy as np


def read_raw_psg(psg_file, channels):
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
        include=channels
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


def crop_data(raw, wake_time):
    """ 
    Crops the data to remove most of the measurement from being awake.

    Parameters:
    raw (obj): Object containing the raw measurement data.
    wake_time (int): Integer containing the amount of seconds to use 
    before and after first sleep

    Returns:
    cropped_raw (obj): Object containing the raw measurement data.
    """
    # get annotations from annotated raw object
    anno = raw.annotations

    # get list of descriptions (sleep stages) from the annotation
    desc_list = anno.description

    # create list of booleans for when a sleep stage is 
    # considered not asleep (True when sleep, False when not)
    sleep_bool = np.array([d != "Sleep stage W" and d != "Sleep stage ?" for d in desc_list])

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

    # get the maximum value between 30 mins before first sleep stage 
    # and start time 
    # (to avoid ValueErrors where tmax is larger than max time)
    crop_start = max(0, first_sleep_start - wake_time)

    # get the minimum value between 30 mins after last sleep stage
    # and the end time of raw object
    crop_end = min(raw.times[-1], last_sleep_end + wake_time)

    print(f"Cropping raw: {crop_start} - {crop_end}")

    # crop data for 30 minutes before first sleep and after last sleep
    try:
        cropped_raw = raw.crop(
            crop_start, 
            crop_end
            )
        print("Cropping finished.")
    except ValueError as e:
        print(f"An error has occured: {e}")
    
    return cropped_raw


def create_sleep_events(cropped_raw, chunk_duration):
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
        cropped_raw, event_id=annotation_desc_2_event_id, chunk_duration=chunk_duration
    )

    return sleep_events
