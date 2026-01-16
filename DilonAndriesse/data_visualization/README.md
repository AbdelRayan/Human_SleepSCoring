# Data visualization
This directory is used to visualize the input data and features, to manually check wether these feature properly represent the different sleep states.

## _"data_config.yaml"_
Config file for the _"visualize_data.ipynb"_ Notebook.

Parameters:
- **epoch** <- epoch length
- **wake_time** <- time of wake to save
- **channels** <- channels to use for visualization
- **subject** <- subject name 

Paths:
- **data** <- general datapath where all following paths can be found
- **raw** <- path to the raw EDF data
- **anno** <- path to the raw hypnogram data
- **bandpower** <- path to bandpower output to
- **data_vis** <- path to save visualizations to 

## _"visualize_data.ipynb"_
Notebook to visualize some of the input data. Run the "visualize_data.ipynb" script after adjusting the parameters.

## _"feature_config.yaml"_
Config file for both _"adjusted_newFeatures.ipynb"_ and _"Complexity_Analysis.ipynb"_ Notebooks.

Parameters:
- **subject** <- subject name
- **category** <- specific category, subdirectory where visualizations will be saved to
- **fs** <- sampling frequency
- **epoch** <-epoch length
- **upper_mask_freq** <- low-pass value
- **lower_mask_freq** <- high-pass value
- **wake_time** <- time of wake to save before and after sleep
- **sleep_labels** <- sleep ids and their corresponding labels

Paths:
- **data** <-
- **mat** <-
- **feature_vis** <-

## _"adjusted_newFeatures.ipynb"_
To visualize the features originated form the rodent research run the "adjusted_NewFeatures.ipynb" script. The resulting images are saved to a custom path which you can set in the config file. Subject must be specified. Images are saved per subject. The structure may look something like this:

    feature_visualization/
        |
        ├──Category_name (i.e. "10s-epochs")
            ├──S35_1
            |    ├──normalized_emg.svg
            |    ├──wei_all-indices.svg
            |    ├──wei-indices_vs_new-indices.svg
            |    ├──Etc...
            ├──S35_2
            |    ├──normalized_emg.svg
            |    ├──wei_all-indices.svg
            |    ├──wei-indices_vs_new-indices.svg
            |    ├──Etc...
            ├──Etc...

## _"Complexity_Analysis.ipynb"_
To visualize the newly introduced features run the "Complexity_analysis.ipynb" script. For this script the EEG channel .mat file path and state file path must be manually changed in the Notebook at the following two variables: _"fpz"_ and _"states"_.

## _"feature_quality_control"_
This Notebook has its own built-in paths and variables and does not use the configuration files.

Parameters:
- **epoch_length** <- length of epochs
- **fs** <- sampling frequency
- **feature_indices** <- list of features indices (0 through 13)
- **file_id** <- additional file name identifier
- **title_id** <- additional title identifier

Paths:
- **h5_file** <- path to HDF5 file to visualize
- **output** <- path to save visualizations to

Main goal of this script is to visualize data of the complete dataset.