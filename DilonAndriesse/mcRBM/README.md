# mcRBM (machine learning)
The mcRBM scripts were adjusted from previous rodent research, now specifically built to work on our current set of 14 features, manual adjustment is required if you use different features.

## Configuration
All configuration files can be found in the "mcRBM/configuration_files" directory.

mcRBM parameters can be adjusted in the "input_configuration.txt" file.

Additional parameters for training, such as location of input files and model name can be adjusted in the "exp_details.txt" file.

Additional parameters for testing, such as location of input files, model name (must me the same as the one in the "exp_details.txt" file) and actual model location can be adjusted in the "exp_details_test.txt" file.

## Dataset
To create the dataset that will be used for training the model you must run the "mcRBM_input_features.ipynb" script found in the "mcRBM/sample_data/" directory, which will create a features.npz file required within the same folder as where the training data is located.

## Training
To start training the model you must run the "train_model.ipynb" script in the "mcRBM/run_model/" directory. Within the scripts there are mutliple paths to different directories within the "mcRBM" directory that must be manually adjusted. After running a the current model can be found under "Sample_data/experiments/{model_name}". Here you can find the weights, energy plots and additional information.

## Analysis
Required scripts can be found in the "mcRBM/run_model/" directory.

To start the analysis, you must first infer the latent states based on test data. This can be done by running the "infer_states_test.ipynb" script. WIthin this script some custom paths must be manually replaced aswell.

After inferring the latents states, some analysis can be done by running the "latent_states_analysis.ipynb" script. Within this script some custom paths must be manually replaced aswell. Depending on the number of epoch you used to train the model, an additional parameter in the following line must be replaced, by the number corresponding to your epoch count - 1: 

    model = StatesAnalysis(
            "C:/Users/andri/school/bio-informatics/internship/
             donders/vsc/Human_SleepSCoring/DilonAndriesse/mcRBM/",
            expFile, 
            {epoch_count-1}, 
            1, 
            False, 
            "L2", 
            "ratios", 
            ["Extracranial Human"]
    )
