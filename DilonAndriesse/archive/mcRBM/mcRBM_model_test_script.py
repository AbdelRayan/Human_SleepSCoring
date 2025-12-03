import h5py
import numpy as np

def wei_normalizing(data):
  """
  Normalizes the input data based on the 10th and 90th percentiles.

  Parameters:
      data (numpy.ndarray): The input data to be normalized.

  Returns:
      numpy.ndarray: The normalized data.

  Note:
      This function is based on the normalization used in 
      Wei, TY., Young, CP., Liu, YT. et al. 
      Development of a rule-based automatic five-sleep-stage scoring method for rats. 
      BioMed Eng OnLine 18, 92 (2019). https://doi.org/10.1186/s12938-019-0712-8
      
      This function first calculates the 10th and 90th percentiles of the input data. 
      It then computes the average of the data below the 10th percentile (bottom_avg) and above the 90th percentile (top_avg). 
      The data is normalized such that bottom_avg maps to 0 and top_avg maps to 1. 
      Finally, all values below 0.05 are set to 0.05 and all values above 1 are set to 1.
  """
  data = np.array(data)
  bottom = data[data <= np.nanpercentile(data, 10, axis=0) ]
  bottom_avg = np.average(bottom)
  top = data[data >= np.nanpercentile(data, 90, axis=0) ]
  top_avg = np.average(top)
  normalized_data = (data - bottom_avg) / (top_avg - bottom_avg)  # Normalise with [min,max] -> [0,1]
  normalized_data = np.clip(normalized_data, 0.05, 1) # set to 0.05 all negative values, set to 1 all values greater than 1

  return normalized_data


if __name__ == "__main__":     
    input_file = r"D:\dilon_data\hdf5_collection\main\partial_static_dataset_v2.h5"
    output_file = r"D:\dilon_data\hdf5_collection\main\partial_static_dataset_normalized_sleep-edf.h5"

    with h5py.File(input_file, "r") as f_in, h5py.File(output_file, "w") as f_out:
        removed_subjects = []
        for subject_key in f_in.keys():
            print(subject_key)
            if "SC" in subject_key:
                subj_group = f_in[subject_key]
                features = subj_group["Features"][:]
                sleep_states = subj_group["Mapped_scores"][:]

                if np.isnan(features).any():
                    removed_subjects.append(subject_key)
                    print(f"Removing subject {subject_key} (contains NaNs)")
                    continue
                else:
                    aperiodic = features[:, 10].flatten()
                    aperiodic_norm = wei_normalizing(aperiodic)
                    dfa = features[:, 11].flatten()
                    dfa_norm = wei_normalizing(dfa)
                    mse = features[:, 12].flatten()
                    mse_norm = wei_normalizing(mse)
                    eog = features[:, 13].flatten()
                    eog_norm = wei_normalizing(eog)
                    new_features = np.column_stack([aperiodic_norm, dfa_norm, mse_norm, eog_norm])
                    features[:, 10:14] = new_features
                    # Create a new group for the subject in the output file
                    new_group = f_out.create_group(subject_key)
                    new_group.attrs['Description features'] = '[index_w_smoothed, index_r_smoothed, index_n_smoothed, index_1_smoothed, index_2_smoothed, index_3_smoothed, index_4_smoothed, noise_smoothed, theta_smoothed, delta_smoothed, aperiodic, dfa, mse, eog_smoothed]'
                    new_group.attrs['Description Mapped_scores'] = '[0: Wake, 1: N1, 2: N2, 3: N3, 4: REM, 5: Movement]'
                    new_group.create_dataset("Features", data=features)
                    new_group.create_dataset("Mapped_scores", data=sleep_states)

print(f"Done! Removed {len(removed_subjects)} subjects with NaNs.")

# from scipy.io import loadmat
# import numpy as np
# import pandas as pd
# file = r"C:\Users\andri\school\bio-informatics\internship\donders\vsc\Human_SleepSCoring\DilonAndriesse\mcRBM\sample_data\experiments\SPN_eps1e3_b4096_e10k\weights\ws_temp0.mat"
# file2 = r"C:\Users\andri\school\bio-informatics\internship\donders\vsc\Human_SleepSCoring\DilonAndriesse\mcRBM\sample_data\experiments\SPN_eps1e3_b4096_e10k\weights\ws_temp9999.mat"
# #latent_states_file = r"C:\Users\andri\school\bio-informatics\internship\donders\vsc\Human_SleepSCoring\DilonAndriesse\mcRBM\sample_data\experiments\rodent_test\analysis\epoch999\all_latent_states.npy"
# latent_states_file = r"C:\Users\andri\school\bio-informatics\internship\donders\vsc\Human_SleepSCoring\DilonAndriesse\mcRBM\sample_data\experiments\partial_static_current\analysis\epoch999\latentStates.npz"
# # mat = loadmat(file)
# # mat2 = loadmat(file2)
# # VF = mat["VF"]
# # FH = mat["FH"]
# # bias_cov = mat["bias_cov"]
# # bias_vis = mat["bias_vis"]
# # w_mean = mat["w_mean"]
# # bias_mean = mat["bias_mean"]
# # VF2 = mat2["VF"]
# # FH2 = mat2["FH"]
# # bias_cov2 = mat2["bias_cov"]
# # bias_vis2 = mat2["bias_vis"]
# # w_mean2 = mat2["w_mean"]
# # bias_mean2 = mat2["bias_mean"]


# # # for i in range(len(VF)):
# # print(f"VF0: {np.max(VF)} - VF9999: {np.max(VF2)}")
# # print(f"FH0: {np.max(FH)} - FH9999: {np.max(FH2)}")
# # print(f"bias_cov0: {np.max(bias_cov)} - bias_cov9999: {np.max(bias_cov2)}")
# # print(f"bias_vis0: {np.max(bias_vis)} - bias_vis9999: {np.max(bias_vis2)}")
# # print(f"w_mean0: {np.max(w_mean)} - w_mean9999: {np.max(w_mean2)}")
# # print(f"bias_mean0: {np.max(bias_mean)} - bias_mean9999: {np.max(bias_mean2)}")

# # print("-" * 50)
# # print("bias_mean2", bias_mean2[-1, :])
# # print("w_mean2", w_mean2[-1, :])
# # print("VF2", VF2[-1, :])
# # print("bias_cov2", bias_cov2[-1, :])
# np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=6)

# data = np.load(latent_states_file)
# print(data.keys())
# #print(data["probabilities"])
# print(data["binary"])
# #print(data["inferredStates"])
# # for i, latent_state in enumerate(data):
# #     print(f"Latent state {i} 14:", latent_state[14], f"latent state {i} 13:", latent_state[13])

# # df = pd.DataFrame(data)
# # df.columns = [f"latent_{i}" for i in range(df.shape[1])]
# # print(df)
# # binary_mask = (df >= 0.5).astype(int)
# #print(df['latent_14'])
# # print(binary_mask)