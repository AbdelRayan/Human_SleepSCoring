import os
import re
import mne
import numpy as np
import scipy.io
import warnings
import shared_processing_functions as spf


def find_edf_files(directory):
    """Return list of .edf files in directory."""
    return [f for f in os.listdir(directory) if f.endswith('.edf') and not f.startswith('.')]


def extract_subject_id(filename, pattern):
    """Extract subject ID using regex pattern."""
    match = pattern.search(filename)
    if not match:
        return None
    subj = match.group(1)
    return subj[:-1] + "_" + subj[-1] if "_" not in subj else subj


def subject_already_processed(subject, output_dir):
    """Check if subject's data already exists in output directory."""
    if not os.path.exists(output_dir):
        return False
    return any(subject in fname for fname in os.listdir(output_dir))


def load_raw_data(edf_path, channels):
    """
    Load raw EDF data using MNE, suppressing filter metadata warnings.
    Always preload for downstream processing.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Channels contain different highpass filters. Highest filter setting will be stored."
        )
        warnings.filterwarnings(
            "ignore",
            message="Channels contain different lowpass filters. Lowest filter setting will be stored."
        )
        warnings.filterwarnings(
            "ignore",
            message="Highpass cutoff frequency .* is greater than lowpass cutoff frequency .*"
        )
        raw = mne.io.read_raw_edf(edf_path, infer_types=True, preload=True, verbose='ERROR')
        raw.pick(channels)
    return raw


def get_annotation_file(subject, anno_dir):
    """Find annotation file for subject in annotation directory."""
    for fname in os.listdir(anno_dir):
        if subject in fname:
            return os.path.join(anno_dir, fname)
    return None


def process_subject(
    edf_file, main_data_path, output_path, sleep_edf_anno, pattern,
    channels, sleep_edf_stage_id, wake_time, edf_sf, emg_bandpass, eog_bandpass
):
    subject = extract_subject_id(edf_file, pattern)
    if not subject:
        print(f"Could not extract subject from {edf_file}")
        return

    if subject_already_processed(subject, output_path):
        print(f"Skipping {subject}: already processed.")
        return

    print(f"Processing subject: {subject}")

    edf_path = os.path.join(main_data_path, edf_file)
    raw = load_raw_data(edf_path, channels)

    # Find annotation file
    anno_file = get_annotation_file(subject.replace("_", ""), sleep_edf_anno)
    if not anno_file:
        print(f"Annotation file not found for {subject}")
        return

    # Add annotation
    spf.add_annotation(anno_file, raw)

    # Crop to wake_time using crop_data
    raw = spf.crop_data(raw, wake_time)
    raw.load_data()

    # Save sleep states
    sleep_states = spf.get_stages(raw, sleep_edf_stage_id)
    spf.create_mat(output_path, subject, "states", sleep_states)

    # Filter EEG channels
    raw.filter(l_freq=0, h_freq=49, picks=["Fpz-Cz", "Pz-Oz"])

    # Process and save each channel
    for channel in ['horizontal', 'submental', 'Fpz-Cz', 'Pz-Oz']:
        raw_c = raw.copy().pick([channel])
        if channel == "horizontal":
            raw_c.filter(
                l_freq=eog_bandpass[0], h_freq=eog_bandpass[1],
                method="iir", iir_params=dict(order=4, ftype="butter"),
                picks=['horizontal'], verbose='ERROR'
            )
            raw_c.rename_channels({"horizontal": "EOG"})
            channel_name = "EOG"
        elif channel == "submental":
            raw_c.rename_channels({"submental": "EMG"})
            channel_name = "EMG"
        else:
            channel_name = channel
        
        # Get data and validate before saving
        data = raw_c.get_data()
        
        # Check data quality
        std_val = np.nanstd(data)
        if std_val < 1e-10:
            print(f"  Warning: {channel_name} has very low variance (std={std_val:.2e})")
        
        spf.create_mat(output_path, subject, channel_name, data)


def preprocess_sleep_edf(
    main_data_path, output_path, sleep_edf_anno,
    channels, sleep_edf_stage_id, wake_time, edf_sf, emg_bandpass, eog_bandpass
):
    # Validate input directories
    if not os.path.isdir(main_data_path):
        print(f"Error: Input data directory not found: {main_data_path}")
        return
    
    if not os.path.isdir(sleep_edf_anno):
        print(f"Error: Annotation directory not found: {sleep_edf_anno}")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")
    
    # Find and process EDF files
    pattern = re.compile(r'(SC4\d{3}|ST7\d{3}|S\d{2,3}_\d)')
    edf_files = find_edf_files(main_data_path)
    
    if not edf_files:
        print(f"Error: No EDF files found in {main_data_path}")
        return
    
    print(f"Found {len(edf_files)} EDF files.")

    processed = 0
    failed = 0
    
    for edf_file in edf_files:
        try:
            process_subject(
                edf_file, main_data_path, output_path, sleep_edf_anno, pattern,
                channels, sleep_edf_stage_id, wake_time, edf_sf, emg_bandpass, eog_bandpass
            )
            processed += 1
        except Exception as e:
            print(f"Error processing {edf_file}: {str(e)}")
            failed += 1
    
    print(f"\nPreprocessing complete: {processed} successful, {failed} failed")


if __name__ == "__main__":
    # Set parameters
    main_data_path = r'C:\Users\jalma\OneDrive - HAN\stage_donders\edf'
    output_path = r'C:\Users\jalma\OneDrive - HAN\stage_donders\output'
    sleep_edf_anno = r'C:\Users\jalma\OneDrive - HAN\stage_donders\edf_annotation'
    channels = ["Fpz-Cz", "Pz-Oz", "horizontal", "submental"]
    sleep_edf_stage_id = {
        "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
        "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
        "Movement time": 5, "Sleep stage ?": 6
    }
    wake_time = 1800
    edf_sf = 100
    emg_bandpass = [5, 120]
    eog_bandpass = [0.5, 35]

    preprocess_sleep_edf(
        main_data_path, output_path, sleep_edf_anno,
        channels, sleep_edf_stage_id, wake_time, edf_sf, emg_bandpass, eog_bandpass
    )