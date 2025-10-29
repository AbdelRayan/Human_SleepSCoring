import numpy as np
import mne.time_frequency
from scipy.signal import welch
from specparam import SpectralModel
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
import EntropyHub as EH
from neurodsp.aperiodic import compute_fluctuations

def psd_multitaper(lfp_data, fs, frequency_band, window_length):
    """
    Computes the power spectral density (PSD) of a signal using the multitaper method.

    Parameters:
        lfp_data (numpy.ndarray): The input signal for which the PSD is to be computed.
        fs (float): The sampling frequency of the signal.
        frequency_band (tuple): A tuple of two elements representing the frequency band (min, max) within which the PSD is to be computed.
        window_length (int): The length of the window for which the PSD is computed.

    Returns:
        list: A list of total power within the specified frequency band for each epoch of the input signal.

    Note:
        This function divides the input signal into epochs of a specified length (window_length). 
        For each epoch, it computes the PSD using the multitaper method and then sums the power within the specified frequency band. 
        The function returns a list of these total power values for each epoch.
    """
    all_power_sum = []

    # loop through each epoch
    for start in range(0, len(lfp_data) - window_length + 1, window_length):
        window = lfp_data[start:min(start + window_length, len(lfp_data))]

        # compute power spectral density using multitaper method
        psd, freqs = mne.time_frequency.psd_array_multitaper(window, fs, fmin=frequency_band[0], fmax=frequency_band[1], n_jobs=1, verbose = 'warning')

        # compute total power within frequency band
        freq_indices = (freqs >= frequency_band[0]) & (freqs <= frequency_band[1])
        curr_sum = np.sum(psd)
        all_power_sum.append(curr_sum)

    return all_power_sum

def Index_1(delta, gamma, EMG):
  index_1 = np.array([])
  for i in range(len(delta)):
    value = (EMG[i]*gamma[i])/(delta[i])
    index_1 = np.append(index_1, [value])
  return index_1
  
def Index_2(delta, theta, sigma):
  index_2 = np.array([])
  for i in range(len(delta)):
    value = (sigma[i]*delta[i])/(theta[i])
    index_2 = np.append(index_2, [value])
  return index_2
  
def Index_3(delta, theta, gamma):
  index_3 = np.array([])
  for i in range(len(delta)):
    value = (theta[i]*gamma[i])/(delta[i])
    index_3 = np.append(index_3, [value])
  return index_3
  
def Index_4(delta, theta):
  index_4 = np.array([])
  for i in range(len(delta)):
    value = delta[i]/theta[i]
    index_4 = np.append(index_4, [value])
  return index_4
  
def index_W(theta, gamma, EMG):
  index_w = np.array([])
  for i in range(len(theta)):
    value = EMG[i]*EMG[i]*((gamma[i])/(theta[i]))
    index_w = np.append(index_w, [value])
  return index_w

def index_R(delta, theta, sigma, EMG, gamma):
  index_r = np.array([])
  for i in range(len(delta)):
    value = ((theta[i]*2)*gamma[i])/(delta[i]*delta[i]*EMG[i]**2)
    index_r = np.append(index_r, [value])
  return index_r

def index_N(delta, sigma, gamma):
  index_n = np.array([])
  for i in range(len(delta)):
    value = (sigma[i]*delta[i])/(gamma[i]**2)
    index_n = np.append(index_n, [value])
  return index_n

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


def aperiodic_fit(window_data, fs):
    freqs, psd = welch(window_data, fs=fs, nperseg=4096)

    mask = (freqs <= 75)
    freqs, psd = freqs[mask], psd[mask]

    fm = SpectralModel(min_peak_height=0.05, aperiodic_mode='knee', verbose=False)
    fm.fit(freqs, psd)
    aperiodic = fm.get_params('aperiodic')[1]

    return aperiodic


def calc_aperiodic_fit(data, window_size, epoch_length, fs):
  window_data = []

  num_windows = len(data) // window_size
  time_stamps = np.arange(num_windows) * epoch_length

  # calculate windows
  for i in range(num_windows):
      start, end = i * window_size, (i + 1) * window_size
      window_data.append(data[start:end])

  aperiodic_exponents = Parallel(n_jobs=-1)(delayed(aperiodic_fit)(window, fs) for window in window_data)

  # Thresholding to remove outliers
  threshold_min = np.percentile(aperiodic_exponents, 2)
  threshold_max = np.percentile(aperiodic_exponents, 98)

  valid_indices = (aperiodic_exponents >= threshold_min) & (aperiodic_exponents <= threshold_max)

  filtered_exponents = aperiodic_exponents[valid_indices]
  filtered_timestamps = time_stamps[valid_indices]

  #filtered_exponents_z = (filtered_exponents - np.mean(filtered_exponents)) / np.std(filtered_exponents)
  window_length = 11 if len(filtered_exponents) >= 11 else len(filtered_exponents) | 1  # ensure it's odd
  polyorder = 4

  smoothed_exponents = savgol_filter(filtered_exponents, window_length=window_length, polyorder=polyorder)

  normalized_exponents = 2 * ((smoothed_exponents - min(smoothed_exponents)) /(max(smoothed_exponents) - min(smoothed_exponents))) - 1

  return valid_indices, normalized_exponents


def calc_dfa(data, window_size, step_size, fs):
  num_windows = (len(data) - window_size) // step_size + 1

  dfa_exponents = []
  time_stamps = []

  for i in range(num_windows):
      start = i * step_size
      end = start + window_size
      segment = data[start:end]

      _, _, exp_window = compute_fluctuations(segment, fs, n_scales=10,
                                              min_scale=0.05, max_scale=4.0)

      dfa_exponents.append(exp_window)
      time_stamps.append((start + end) / 2 / fs)

  dfa_exponents = np.array(dfa_exponents)
  window_length = 11 if len(dfa_exponents) >= 11 else len(dfa_exponents) | 1  # ensure it's odd
  polyorder = 4

  smoothed_dfa = savgol_filter(dfa_exponents, window_length=window_length, polyorder=polyorder)
  normalized_dfa = 2 * ((smoothed_dfa - min(smoothed_dfa)) /(max(smoothed_dfa) - min(smoothed_dfa))) - 1

  return normalized_dfa


def calc_mse(data, window_size, step_size, fs):
  Mobj = EH.MSobject('IncrEn', m=2, R=3, Norm=True)

  num_windows = (len(data) - window_size) // step_size + 1

  mse_values = []
  time_stamps_mse = []

  for i in range(num_windows):
      start = i * step_size
      end = start + window_size
      segment = data[start:end]

      MSx, _ = EH.MSEn(segment, Mobj, Scales=2, Methodx='modified')

      mse_values.append(np.mean(MSx))
      time_stamps_mse.append((start + end) / 2 / fs)

  mse_values = np.array(mse_values)
  time_stamps_mse = np.array(time_stamps_mse)
  window_length = 11 if len(mse_values) >= 11 else len(mse_values) | 1  # ensure it's odd
  polyorder = 4

  smoothed_mse = savgol_filter(mse_values, window_length=window_length, polyorder=polyorder)
  normalized_mse = 2 * ((smoothed_mse - min(smoothed_mse)) /(max(smoothed_mse) - min(smoothed_mse))) - 1

  return normalized_mse