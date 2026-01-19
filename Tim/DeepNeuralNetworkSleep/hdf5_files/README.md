# Introduction
This folder allows us to collect features and manual scoring to feed the mcRBM algorithm. 

It is divided in 2 steps :
- step 1 : Creation of the hdf5 file;
- step 2 : Extraction of a dataset or a training dataset from the hdf5 file.

# File structure
## Pre Processing
The raw data (HPC and PFC) first go through `Artefacts_Detection.py`. This function detects the artifacts and sets them to zero.
Epochs where artifacts are detected will have their manual score set to 0.

## Features
To obtain the desired features, we first compute the mean spectral power of the following frequency ranges on the epochs:
- $`EEG_{lo}`$ : [0; 0.5] Hz;
- $`\delta`$ : [0.5; 4] Hz;
- $`\theta`$ : [4; 8] Hz;
- $`\alpha`$ : [8; 13] Hz;
- $`\sigma`$ : [12; 14] Hz;
- $`\gamma`$ : [30; 90] Hz;
  
as well as:
- $`EMG`$ : mean value of the EMG.
- $`EOG_{0.3-0.45}`$ : value of the 0.3-0.45 EOG feature described by Gunnarsdottir et al. (2020)
- $`EOG_{0.3-35}`$ : value of the 0.3-35 EOG feature described by Gunnarsdottir et al. (2020)

From these values, we compute the following features for each 2-second epoch:
 - Index_W: $\mathrm{EMG}^2 \cdot \frac{\gamma}{\theta}$
 - Index_N : $\frac{\mathrm{EOG}_{0.3-0.45}^2}{\gamma^2}$
 - Index_R : $\frac{\mathrm{EOG}_{0.3-35}^2}{\mathrm{EMG}^2 \cdot \delta^2}$
 - Index_1 : $\frac{E M G \cdot \gamma}{\delta}$
 - Index_2 : $\frac{\sigma \cdot \delta}{\theta}$
 - Index_3 : $\frac{\theta \cdot \gamma}{\delta}$
 - Index_4 : $\frac{\delta}{\theta}$
 - Theta
 - Delta
 - $`EEG_{lo}`$ (Noise)

### Complexity analysis
Additionally, features looking at the temporal structure are calculated from the cortex signal.
- $`Aperiodic Fit`$
- $`Detrended Fluctuation Analysis`$
- $`Multi Scale Entropy`$

# References
Gunnarsdottir, K. M., Gamaldo, C., Salas, R. M., Ewen, J. B., Allen, R. P., Hu, K., & Sarma, S. V. (2020). A novel sleep stage scoring system: Combining expert-based features with the generalized linear model. Journal of sleep research, 29(5), e12991. https://doi.org/10.1111/jsr.12991