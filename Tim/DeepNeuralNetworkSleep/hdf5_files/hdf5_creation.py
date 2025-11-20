import numpy as np
import h5py
from scipy.stats import mode
from scipy.io import loadmat
import os


from EMG_buzsakiMethod import compute_emg_buzsakiMethod
from computing_features import psd_multitaper, wei_normalizing, index_W, index_N, index_R, Index_1, Index_2, Index_3, Index_4
from Artefacts_Detection import removeArtefacts, artefact_epochs

def getNewFeatures(recording_dict, states, fs, epoch_length):
  """
  Computes new features from raw data.

  Parameters:
      raw_hpc (numpy.ndarray): The data from the hippocampus.
      raw_pfc (numpy.ndarray): The data from the prefrontal cortex.
      states (numpy.ndarray): The sleep states.
      fs (float) : the sampling frequency.
      epoch_length (int) : the length of an epoch in seconds.

  Returns:
      new_features (numpy.ndarray): The computed features.
      mapped_scores (numpy.ndarray): The mapped sleep scores.

  Notes:
      This function first computes the EMG from the raw data. 
      It then computes the power spectral density (PSD) in different frequency bands for the raw data. 
      The PSDs are normalized and smoothed to be used as features. 
      The function also computes several indices from the PSDs, normalizes and smooths them, and uses them as features. 
      The function returns a matrix of these features along with the mapped sleep scores.
  """

  # Get mapped scores
  mapped_scores = np.ravel(states)

  # Get smoothed powers as feature
  noise_smoothed = recording_dict["noise"]
  theta_smoothed = recording_dict["theta"]
  delta_smoothed = recording_dict["delta"]

  indices = recording_dict["index_vals"]

  index_w_smoothed = indices["W"]
  index_n_smoothed = indices["N"]
  index_r_smoothed = indices["R"]
  index_1_smoothed = indices["1"]
  index_2_smoothed = indices["2"]
  index_3_smoothed = indices["3"]
  index_4_smoothed = indices["4"]

  aperiodic_fit = recording_dict["aperiodic_fit"]
  dfa = recording_dict["dfa"]
  mse = recording_dict["mse"]


  # Create matrix
  new_features = np.column_stack((index_w_smoothed, index_r_smoothed, index_n_smoothed,
                                  index_1_smoothed, index_2_smoothed, index_3_smoothed, index_4_smoothed,
                                  noise_smoothed, theta_smoothed, delta_smoothed,
                                  aperiodic_fit, dfa, mse))
  return new_features, mapped_scores

def prepare_for_hdf5(recording, fs, files_path, epoch_length):
  """
  Prepares data for HDF5 format.
  
  Parameters:
      recording (str): Name of the recording
      fs (float) : The sampling frequency.
      files_path (str) : The path that leads to the files.
      epoch_length (int) : the length of an epoch in seconds.
  
  Returns:
      Features (numpy.ndarray): The computed features from HPC and PFC data. It consists of a list of 10 indices for each epoch.
      Mapped_scores (numpy.ndarray): The mapped sleep scores after adding artefact epochs.
      recording_name (str): The name of the recording group.
  """
  # Get the right data
  recording_dict = np.load(os.path.join(files_path, recording), allow_pickle=True).item()
  states = recording_dict["dfa_violin"]["df_plot"].iloc[:,0].to_numpy()
  recording_name = recording.split("features_")[1].split(".npy")[0]
  sleep_scoring = states

  # Create matrix for specific set of recordings
  a = getNewFeatures(recording_dict, states, fs, epoch_length)
  Features = a[0]
  Mapped_scores = a[1]

  # Add the artefact epochs to mapped scores

  return(Features, Mapped_scores, recording_name)

def update_hdf5(a, path_to_hdf5):
  """
  Updates an HDF5 file with the data of a recording (features and mapped scores)/
  
  Parameters:  
      a (tuple): A tuple containing the features, mapped scores, and the recording name.
      path_to_hdf5 (str): The path to the HDF5 file to be updated.
  
  Notes:
      This function opens the HDF5 file at the given path in append mode.
      It creates a new group in the file with the name of the recording.
      The group is given two attributes: 'Description features' and 'Description Mapped_scores', 
      which describe the features and mapped scores respectively.
      Two datasets, 'Features' and 'Mapped_scores', are created in the group using the data from the input tuple.
  """
  # Add the data to the hdf5 file
  with h5py.File(path_to_hdf5, 'a')  as database:
  # Create group and 2 datasets
    group = database.create_group(a[2])
    group.attrs['Description features'] = ('[index_w_smoothed, index_r_smoothed, index_n_smoothed, '
                                           'index_1_smoothed, index_2_smoothed, index_3_smoothed, index_4_smoothed, '
                                           'noise_smoothed, theta_smoothed, delta_smoothed, '
                                           'aperiodic_fit, dfa, mse]')
    group.attrs['Description Mapped_scores'] = '[0: Wake, 1: N1, 2: N2, 3: N3, 4: REM]'
    group.create_dataset('Features', data = a[0])
    group.create_dataset('Mapped_scores', data = a[1])
