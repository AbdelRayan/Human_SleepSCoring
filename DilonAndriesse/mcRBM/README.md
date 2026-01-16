# mcRBM (machine learning)
The mcRBM scripts were adjusted from previous rodent research, now specifically built to work on our current set of 14 features, manual adjustment is required if you use different features.

## _mcRBM_input_features.ipynb_
This script is run to create the final version of the dataset in a specific format required to train the mcRBM model. 

Default parameters:
- **feature_npy_file_name** <- name of the feature file created in the previous step
- **feature_npz_file_name** <- name of the feature file that will be created in this script
- **states_file_name** <- name of the states file created in the previous step

Paths:
- **data_path** <- Path to the training or testing dataset created in the previous step (e.g. _"mcRBM/sample_data/input/training/"_)

Requirements:
- **Followed previous step to have data in proper structure**

From the _"mcRBM/sample_data"_ directory, run the _"mcRBM_input_features.ipynb"_ script to create the data required for mcRBM machine learning. After adjusting the paths you can press run all to run the script. This needs to be done for both the training and testing subdirectory.

## _Configuration files_
Config files for the mcRBM model can be found under _"mcRBM/configuration_files/"_.

Files:
- **input_configuration.txt** <- File with input parameters for the mcRBM model
- **exp_details.txt** <- File with parameters and paths required for training the model
- **exp_details_test.txt** <- File with parameters and paths required for infering latent states

Within _"exp_details"_ adjust the _"dsetDir"_ path to your input training subdirecty, adjust the _"expsDir"_ path to the _"mcRBM/sample_data/experiments"_ subdirectory. Finally set _"expID"_ to a model name as wanted.

Change these same settings within the _"exp_details_test"_, but change _"dsetDir"_ to your testing dataset instead of training. Set the _"modelDirName"_ to the model name and _"modelName"_ to the correct weight file.

## _train_model.ipynb_
To Actually train the model the Cupy package is required. For this research Cupy for CUDA 12 was used. This package needs to manually be installed on a PC with an NVIDIA graphics card that supports CUDA.

Change the paths from the top cells (_"sys.path.append({path})"_ and _"os.chdir"_) to your complete path to the _"mcRBM/scripts"_ subdirectory.

After adjusting all config files you can run the _"train_model.
ipynb"_ script from the _"mcRBM/run_model/"_ subdirectory to start training the model. Depending on the amount of datapoints in your dataset training could take a while (e.g. model with 800.000 datapoints took 30+ hours).

## _infer_states_test.ipynb_
After training has finished you must run the _"infer_state_test.ipynb"_ script.

Change the paths from the top cells (_"os.chdir"_) to your complete path to the _"mcRBM/scripts"_ subdirectory.

Paths:
- **root_dir** <- path to the configuration files
- **base_dir** <- path to the model directory
- **model_dir** <- path to the specific weights

Variables:
- **command** <- specific statement to run in subprocess

In general it is important to change all paths to their corresponding ones on your PC.

## _latent_states_analysis.ipynb_
To do some analysis on the inferred latent states the _"latent_states_analysis.ipynb"_ can be run.

Same as before it is important to change the filepaths within the Notebook.

In this file one of the parameters is very important. A variable called model is created as followed: _"model = StatesAnalysis("C:/Users/andri/school/bio-informatics/internship/donders/vsc/Human_SleepSCoring/DilonAndriesse/mcRBM/", expFile, 9999, 1, False, "L2", "ratios", ["Extracranial Human"])"_. The variable 9999 refers to the weights file that is used, ensure that this number is the same as the number in the weights file from the _"exp_details_test"_.

The results from the analysis can be found under _"mcRBM/sample_data/experiments/{model_name}/analysis/"_.