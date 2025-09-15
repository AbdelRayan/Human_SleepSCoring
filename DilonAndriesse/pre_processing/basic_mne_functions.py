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


def crop_data(data, anno):
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
    cropped_data = data.crop(
        anno[1]["onset"] - 30 * 60, 
        anno[-2]["onset"] + 30 * 60
        )
    
    return cropped_data


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