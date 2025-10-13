import scipy

def create_mat(output_path, subject, ch_pb, data):
    try:
        scipy.io.savemat(
                f"{output_path}/{subject}_{ch_pb}.mat", 
                {ch_pb: data.reshape((-1,1))}
                )
    except AttributeError as e:
        print("Sleep stages variable can't be reshaped.\nSaving the sleep stages to .mat file.")
        scipy.io.savemat(
                f"{output_path}/{subject}_{ch_pb}.mat", 
                {ch_pb: data}
                )


def get_stages(raw, stage_id):
    fs = raw.info["sfreq"]
    sleep_states = []

    for anno in raw.annotations:
        stage = str(anno["description"])
        duration_sample = int(anno["duration"] * fs)

        if stage in stage_id:
            sleep_states.extend([stage_id[stage]] * duration_sample)
        else:
            sleep_states.extend([5] * duration_sample)

    return sleep_states


def psd():
    pass