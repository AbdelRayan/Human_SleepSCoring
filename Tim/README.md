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

This step sets up the dataset for the mcRBM

---

## 6. Prepare mcRBM Input Features

**Notebook:**  
`DeepNeuralNetworkSleep/mcRBM/sample_data/mcRBM_input_features.ipynb`

This step separates creates a training and test dataset for training and using the model.

---

## 7. Train the mcRBM

**Notebook:**  
`DeepNeuralNetworkSleep/mcRBM/run_model/train_model.ipynb`

This scripts allows for the training of the network, by default it's set up to run using Google CoLab.
Use 
`DeepNeuralNetworkSleep/mcRBM/configuration_files/exp_details`
and
`DeepNeuralNetworkSleep/mcRBM/configuration_files/input_configuration`
to configure the training process

---

## 8. Infer states using a trained model

**Notebook:**  
`DeepNeuralNetworkSleep/mcRBM/run_model/infer_states_test.ipynb`

After training, this notebook can be used to infer latent states in the test dataset.

Use
`DeepNeuralNetworkSleep/mcRBM/configuration_files/exp_details_test`
to configure the inference.

---

## 9. Analyse inferred latent states

**Notebook:**  
`DeepNeuralNetworkSleep/mcRBM/run_model/latent_states_analysis.ipynb`

After inference, this notebook can be used to analyse the latent states as the final step.


---


# Summary Workflow

```
→ Convert ASCII → Filter → Downsample
→ Convert to EDF → Score with U-Sleep
→ Extract Features
→ Build HDF5 Files
→ Create Training Dataset
→ Prepare mcRBM Input Data
→ Train the network
→ Infer latent states
→ Analyse latent states
```

## References
Perslev, M., Darkner, S., Kempfner, L. et al. U-Sleep: resilient high-frequency sleep staging. npj Digit. Med. 4, 72 (2021). https://doi.org/10.1038/s41746-021-00440-5 