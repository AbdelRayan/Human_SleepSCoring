"""
@:description
Original script from the employed rodent model. 
Adjusted to make it work using human data.

@:refer
not sure

@:original_code
#post git link here?
"""

import numpy as np
import h5py
from scipy.stats import mode
from scipy.io import loadmat
import os

#temp
import multiprocessing as mp
from joblib import Parallel, delayed


from hdf5_files.computing_features import psd_multitaper, wei_normalizing, index_W, index_N, index_R, Index_1, Index_2, Index_3, Index_4
from hdf5_files.Artefacts_Detection import removeArtefacts, artefact_epochs

def getNewFeatures(raw_fpz, raw_pz, raw_emg, raw_eog, states, fs, epoch_length):
  """
  Computes new features from raw data.

  Parameters:
      raw_fpz (numpy.ndarray): The data from the frontal EEG.
      raw_pz (numpy.ndarray): The data from the parenial EEG.
      raw_emg (numpy.ndarray): The data from the EMG.
      raw_eog (numpy.ndarray): The data from the EOG.
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
  sleep_scoring = np.ravel(states)
  # print(sleep_scoring)
  # print(len(sleep_scoring))
  reshaped_scores = sleep_scoring[:len(sleep_scoring) // epoch_length * epoch_length].reshape(-1, epoch_length)
  # reshaped_scores = sleep_scoring
  # print(reshaped_scores)
  # print(len(reshaped_scores))
  majority_scores = mode(reshaped_scores, axis=1).mode.flatten()
  # majority_scores = mode(sleep_scoring).mode.flatten()
  mapped_scores = np.array(majority_scores)
  # mapped_scores = np.array(sleep_scoring)

  #Frequency ranges
  noise_band = [0,0.5]
  delta_band = [0.5,5]
  theta_band = [6,10]
  sigma_band = [11,17]
  beta_band = [22,30]
  gamma_band = [35,45]
  total_band = [0,30]

  # proper data window length based on epoch and sampling frequency
  window_length = fs*epoch_length

  # Get powers
  noise = psd_multitaper(np.ravel(raw_pz), fs, noise_band, window_length)
  delta = psd_multitaper(np.ravel(raw_fpz), fs, delta_band, window_length)
  theta = psd_multitaper(np.ravel(raw_pz), fs, theta_band, window_length)
  sigma = psd_multitaper(np.ravel(raw_fpz), fs, sigma_band, window_length)
  gamma = psd_multitaper(np.ravel(raw_fpz), fs, gamma_band, window_length)

  # Normalize all these powers
  noise_norm = wei_normalizing(noise)
  delta_norm = wei_normalizing(delta)
  theta_norm = wei_normalizing(theta)
  sigma_norm = wei_normalizing(sigma)
  gamma_norm = wei_normalizing(gamma)
  emg_norm = wei_normalizing(raw_emg)
  eog_norm = wei_normalizing(raw_eog)

  # Get smoothed powers as feature
  noise_smoothed = np.convolve(np.convolve(np.convolve(noise_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  theta_smoothed = np.convolve(np.convolve(np.convolve(theta_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  delta_smoothed = np.convolve(np.convolve(np.convolve(delta_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  #emg_smoothed = np.convolve(np.convolve(np.convolve(emg_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  #eog_smoothed = np.convolve(np.convolve(np.convolve(eog_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')

  # Compute new indices
  index_w = index_W(theta_norm, gamma_norm, emg_norm)
  index_n = index_N(delta_norm, sigma_norm, gamma_norm)
  index_r = index_R(delta_norm, theta_norm, sigma_norm, emg_norm, gamma_norm)
  index_1 = Index_1(delta_norm, gamma_norm, emg_norm)
  index_2 = Index_2(delta_norm, theta_norm, sigma_norm)
  index_3 = Index_3(delta_norm, theta_norm, gamma_norm)
  index_4 = Index_4(delta_norm, theta_norm)

  # Take log of indices
  index_w_log = np.log(index_w)
  index_n_log = np.log(index_n)
  index_r_log = np.log(index_r)
  index_1_log = np.log(index_1)
  index_2_log = np.log(index_2)
  index_3_log = np.log(index_3)
  index_4_log = np.log(index_4)

  # Normalise indices
  index_w_norm = wei_normalizing(index_w_log)
  index_n_norm = wei_normalizing(index_n_log)
  index_r_norm = wei_normalizing(index_r_log)
  index_1_norm = wei_normalizing(index_1_log)
  index_2_norm = wei_normalizing(index_2_log)
  index_3_norm = wei_normalizing(index_3_log)
  index_4_norm = wei_normalizing(index_4_log)

  # Smooth indices
  index_w_smoothed = np.convolve(np.convolve(np.convolve(index_w_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_n_smoothed = np.convolve(np.convolve(np.convolve(index_n_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_r_smoothed = np.convolve(np.convolve(np.convolve(index_r_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_1_smoothed = np.convolve(np.convolve(np.convolve(index_1_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_2_smoothed = np.convolve(np.convolve(np.convolve(index_2_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_3_smoothed = np.convolve(np.convolve(np.convolve(index_3_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')
  index_4_smoothed = np.convolve(np.convolve(np.convolve(index_4_norm, np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same'), np.ones(5)/5, mode='same')

  # Create matrix
  new_features = np.column_stack((index_w_smoothed, index_r_smoothed, index_n_smoothed, index_1_smoothed, index_2_smoothed, index_3_smoothed, index_4_smoothed, noise_smoothed, theta_smoothed, delta_smoothed))
  return new_features, mapped_scores

def prepare_for_hdf5(recording, fs, files_path, epoch_length):
  """
  Prepares data for HDF5 format.
  
  Parameters:
      recording (list): The list containing names of the fpz, pz, and states data files.
      fs (float) : The sampling frequency.
      files_path (str) : The path that leads to the files.
      epoch_length (int) : the length of an epoch in seconds.
  
  Returns:
      Features (numpy.ndarray): The computed features from fpz and Pz data. It consists of a list of 10 indices for each epoch.
      Mapped_scores (numpy.ndarray): The mapped sleep scores after adding artefact epochs.
      recording_name (str): The name of the recording group.
  """
  # Get the right data
  for i in recording:
    if 'Fpz-Cz' in i:
      fpz = i
    elif 'Pz-Oz' in i:
      pz = i
    elif 'EMG' in i:
      emg = i
    elif 'EOG' in i:
      eog = i
    elif 'states' in i:
      states = i

  #print(recording)
  recording_name = str(fpz[:-4].split("_")[0])     # Group name 
  print(f"Subject name: {recording_name}")

  # Fpz-Cz
  path_to_fpz = files_path + "/" + fpz
  # Prepare data (Load + artefact removal)
  fpz_data = loadmat(path_to_fpz)
  fpz_data = fpz_data['Fpz-Cz']
  fpz_data = fpz_data[8*fs:]
  sigfpz = removeArtefacts(fpz_data, fs, [9,8], [0.2,0.1])
  fpz_filt = np.ravel(sigfpz[0])
  fpz_artefact_indexes = np.ravel(sigfpz[1])                 # Get the indexes of epochs containing artefacts
  
  # Pz-Oz
  pz_data = loadmat(os.path.join(files_path, pz))
  pz_data = pz_data['Pz-Oz']
  pz_data = pz_data[8*fs:]
  sigpz = removeArtefacts(pz_data, fs, [9,8], [0.2,0.1])
  pz_filt = np.ravel(sigpz[0])
  pz_artefact_indexes = np.ravel(sigpz[1])

  # EMG
  emg_data = loadmat(os.path.join(files_path, emg))
  emg_sampling = 1
  emg_data = emg_data['EMG']
  emg_data = emg_data[8:]
  emg_data = emg_data[:len(emg_data) // (epoch_length * emg_sampling) * (epoch_length*emg_sampling)]
  emg_data = emg_data.reshape(-1, (epoch_length * emg_sampling))
  emg_data = emg_data.sum(axis=1)

  # EOG
  eog_data = loadmat(os.path.join(files_path, eog))
  eog_data = eog_data['EOG']
  eog_data = eog_data[8*fs:]


  sleep_scoring = loadmat(os.path.join(files_path, states))
  states = sleep_scoring['States'][0][7:]
  print(f"Size of raw fpz-data: {len(fpz_data)}")
  print(f"Size of raw states: {len(states)}")


  # Create matrix for specific set of recordings
  a = getNewFeatures(fpz_filt, pz_filt, emg_data, eog_data, states, fs, epoch_length)
  Features = a[0]
  Mapped_scores = a[1]
  print(f"Size of epoched features: {len(Features)}")
  print(Features)
  print(f"Size of epoched states: {len(Mapped_scores)}")

  # Add the artefact epochs to mapped scores
  window_length = fs * epoch_length
  fpz_arte_epochs = artefact_epochs(fpz_artefact_indexes, window_length)
  #print(f"Last epoch index of artefacts: {fpz_arte_epochs[-1]}")
  pz_arte_epochs = artefact_epochs(pz_artefact_indexes, window_length)
  artefact_indices = np.unique(np.concatenate((fpz_arte_epochs, pz_arte_epochs)))
  artefact_indices = artefact_indices.astype(int)
  #print(f"Total amount of artefacts indices: {len(artefact_indices)}")
  Mapped_scores[artefact_indices] = 0 
  return (Features, Mapped_scores, recording_name)


# def remove_artifacts(files_path, channel, fs):
#   """
#   """
#   # get specific file path
#   path_to_chan = files_path + "/" + channel

#   # Prepare data (Load + artefact removal)
#   chan_data = loadmat(path_to_chan)
#   chan_data = chan_data[channel]
#   chan_data = chan_data[8*fs:]
#   sig_chan = removeArtefacts(chan_data, fs, [9,8], [0.2,0.1])
#   chan_filt = np.ravel(sig_chan[0])
#   fpz_artefact_indexes = np.ravel(sig_chan[1])                 # Get the indexes of epochs containing artefacts

#   return fpz_artefact_indexes


def update_hdf5(result, path_to_hdf5):
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
    print(path_to_hdf5)
    print(result[2])
    group = database.create_group(str(result[2]))
    group.attrs['Description features'] = '[index_w_smoothed, index_r_smoothed, index_n_smoothed, index_1_smoothed, index_2_smoothed, index_3_smoothed, index_4_smoothed, noise_smoothed, theta_smoothed, delta_smoothed]'
    group.attrs['Description Mapped_scores'] = '[0: Artefact, 1: Wake, 3: NREM, 4: Intermediate, 5: REM]'
    group.create_dataset('Features', data = result[0])
    group.create_dataset('Mapped_scores', data = result[1])

if __name__ == "__main__":
  fs = 100 # EEG sampling frequency
  epoch_length = 5 #in seconds

  path_to_pt5 = 'C:/Users/andri/school/bio-informatics/internship/donders/data/human_test_data/mat_files/'   # Input folder path
  hdf5_path = 'C:/Users/andri/school/bio-informatics/internship/donders/data/human_test_data/hdf5/physionet_test5.h5'   #Name of the new hdf5 file to create

  Database = h5py.File(hdf5_path, 'w')  # Output directory path

  files = np.ravel(os.listdir(path_to_pt5))
  # Create recording quintiplets (Pfz-Cz, Pz-Cz, EMG, EOG states)
  files = files[:len(files) // 5 * 5].reshape(-1, 5)
  num_processes = mp.cpu_count()
  print('Number of processes :', num_processes)
  # files = [
  #   'SC4042E0_EMG.mat', 
  #   'SC4042E0_EOG.mat', 
  #   'SC4042E0_Fpz-Cz.mat',
  #   'SC4042E0_Pz-Oz.mat',
  #   'SC4042E0_sleep_states.mat'
  # ]

  results = Parallel(n_jobs=min(num_processes, len(files)), verbose = 0)(delayed(prepare_for_hdf5)(recording, fs, path_to_pt5, epoch_length) for recording in files)
  #results = prepare_for_hdf5(files[0], fs, path_to_pt5, epoch_length)

  for result in results:
    update_hdf5(result, hdf5_path)