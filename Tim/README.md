# Creating a Training Dataset

This guide describes the workflow for transforming raw iEEG recordings format into a clean, standardized dataset ready for downstream modelling. Each step corresponds to a Jupyter notebook provided in this repository.

---

## 1. Convert BrainVision ASCII Files (Optional)

**Notebook:**  
`Sleep_staging.ipynb`

Required only if your recordings are in BrainVision ASCII format.  
This notebook allows you to:

- Convert ASCII files into standard continuous data arrays  
- Optionally filter and retain only relevant electrodes  
- Downsample the recordings to reduce file size and processing time  

---

## 2. Export to EDF and Score with U-Sleep (Optional)

**Notebooks:**  
`Sleep_staging.ipynb` (EDF conversion)  
`U-sleep_API.ipynb` (automatic scoring via API)

If your dataset has not yet been scored, you can convert the processed signals to EDF and perform sleep staging with **U-Sleep** (Perslev et al., 2021).  
You may score manually using the web interface, or, if you have API access, automate scoring using the API notebook.

---

## 3. Generate Feature Files

**Notebook:**  
`Averaged_features.ipynb`

This notebook computes all required features per subject, including:

- Sleep state indices
- Spectral powers
- Aperiodic exponent (Specparam)  
- Detrended Fluctuation Analysis (DFA)  
- Multiscale Entropy (MSE)  
- Additional averaged or per-epoch features  

The output is a set of feature files for all subjects/nights.

---

## 4. Create HDF5 Storage Files

**Notebook:**  
`DeepNeuralNetworkSleep/hdf5_files/create_hdf5.ipynb`

Converts all previously generated features into a unified **HDF5 dataset**

---

## 5. Build the Training Dataset

**Notebook:**  
`dataset_creation.ipynb`

This step assembles the final dataset used for model training.

---

## 6. Prepare mcRBM Input Features

**Notebook:**  
`DeepNeuralNetworkSleep/mcRBM/sample_data/mcRBM_input_features.ipynb`

If you plan to use the mcRBM model included in the repository, run this notebook.

---

# Summary Workflow

```
Raw Data
   → (Optional) Convert ASCII → Filter → Downsample
   → (Optional) Convert to EDF → Score with U-Sleep
   → Extract Features
   → Build HDF5 Files
   → Create Training Dataset
   → Prepare mcRBM Input Data
```

## References
Perslev, M., Darkner, S., Kempfner, L. et al. U-Sleep: resilient high-frequency sleep staging. npj Digit. Med. 4, 72 (2021). https://doi.org/10.1038/s41746-021-00440-5 