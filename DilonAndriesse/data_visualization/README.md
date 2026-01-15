# Data visualization
This directory is used to visualize the input data and features, to manually check wether these feature properly represent the different sleep states.

## Input data
To visualize the input data run the "visualize_data.ipynb" script. Different parameters can be found and adjusted in the file called "visualization_config.yaml".

## Rodent features
To visualize the features originated form the rodent research run the "adjusted_NewFeatures.ipynb" script. The resulting images are saved to a custom path which you can set in the "visualization_config.yaml" file. Within this config file specific parameters can also be adjsuted. Specific subject must be specified. Images are saved per subject. The structure may look something like this:

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

## New features
To visualize the newly introduced features run the "Complexity_analysis.ipynb" script. Specific subject can be set in the "visualization_config.yaml" file.