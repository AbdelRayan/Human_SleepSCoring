# mcRBM (Mean-Covariance Restricted Boltzmann Machine)

A pure Python implementation of the mean-covariance Restricted Boltzmann Machine (mcRBM) for unsupervised learning of latent sleep states and features.

Based on the original implementation by Marc'Aurelio Ranzato and the sleep-focused adaptation by DilonAndriesse.

This is the latest Jalmar-created mcRBM variant and can be used for feature sets with any feature count, making analysis of subsets work seamlessly.

## Features

- **CPU & GPU support**: Uses NumPy (CPU) or CuPy (GPU, optional) for flexible deployment
- **Hybrid Monte Carlo sampling**: Accurate MCMC sampling for generative modeling
- **Hidden unit dropout**: Regularization during training
- **Flexible preprocessing**: Log transform, scaling, PCA, whitening
- **Latent state analysis**: Tools for analyzing discovered latent states
- **Visualization**: Automatic plotting of activations, transitions, and features

## Installation

### CPU-Only (Default)
```bash
cd Jalmar/mcRBM
pip install numpy scipy scikit-learn matplotlib pillow
```

### GPU Support (Optional)
To enable GPU acceleration via CuPy:

```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x

# CUDA 13.x
pip install cupy-cuda13x
```

For CUDA compatibility and installation details, see: https://docs.cupy.dev/en/stable/install.html

## Quick Start

### 1. Configure your experiment

Edit `configuration_files/exp_config.ini`:
```ini
[EXP_DETAILS]
dsetDir = /path/to/your/data/
expsDir = ./experiments/
expID = my_experiment
dSetName = data.npz
```

Edit `configuration_files/model_config.ini`:
```ini
[COMPUTE_BACKEND]
use_gpu = False          # Change to True for GPU (CuPy) acceleration
gpu_id = 0               # GPU device ID (0, 1, 2, ...). Ignored if use_gpu=False

[MODEL_PARAMETER_SETTING]
num_fac = 3            # Number of factors (optimized for 4-feature dataset)
num_hid_cov = 16       # Covariance hidden units
num_hid_mean = 8       # Mean hidden units
num_epochs = 200       # Training epochs
batch_size = 128       # Batch size
```

**Note**: This N1_test package is configured for a 4-feature dataset (Index_R_noEOG, Index_N_noEOG, Index_W, EOG).
The parameters above are optimized for this dimensionality.

**Backend Selection:**
- `use_gpu = False` (default): Uses NumPy on CPU
- `use_gpu = True`: Uses CuPy on GPU (requires CuPy installation)
- If CuPy unavailable with `use_gpu=True`: Automatically falls back to NumPy with warning

## N1_test Workflow (4-Feature Dataset)

This package includes a specialized data extractor for the 4-feature N1 test dataset:

### Step 1: Extract selected features from HDF5

Use `extract_selected_features.py` to extract only the 4 target features from the full HDF5 feature file:

```bash
cd scripts
# Edit extract_selected_features.py to set:
#   INPUT_HDF5_PATH = path/to/full/sleep_features.h5
#   OUTPUT_HDF5_PATH = path/to/sleep_features_N1_selection.h5
#   TARGET_FEATURES = ["Index_R_noEOG", "Index_N_noEOG", "Index_W", "EOG"]

python extract_selected_features.py
```

This creates a new HDF5 file containing only the 4 selected features, preserving subject structure and metadata.

### Step 2: Convert HDF5 to NPZ for mcRBM training

Use `example_usage.py` to load the extracted HDF5 into an NPZ file suitable for mcRBM:

```bash
python example_usage.py
# This reads the extracted HDF5 and creates mcrbm_input.npz
```

### Step 2a (Optional): Strip sleep stage labels

If you want purely unsupervised learning without ground truth labels, use `strip_labels.py`:

```bash
python strip_labels.py \
  --input mcrbm_input.npz \
  --output mcrbm_input_no_labels.npz \
  --keep-epoch-time
```

This removes the `epochsLinked` field (sleep stage labels) while preserving the feature data and optional epoch times. Useful for ensuring the model learns patterns without label information.

### Step 2b (Optional): Create balanced train/test split

To evaluate generalization, create a train/test split directly from HDF5 with built-in balance checking:

**Option 1: Split from HDF5 (recommended for N1 test workflow)**

Use `train_test_split_hdf5.py` to split directly from the HDF5 file:

```bash
python train_test_split_hdf5.py \
  --input sleep_features_N1_selection.h5 \
  --output-dir ./ \
  --test-ratio 0.2 \
  --seed 42 \
  --tolerance 5.0
```

Edit the script's config section to set:
```python
INPUT_HDF5_PATH = r"path/to/sleep_features_N1_selection.h5"
OUTPUT_DIR = r"path/to/output"
TEST_RATIO = 0.2
RANDOM_SEED = 42
BALANCE_TOLERANCE = 5.0
```

This creates:
- `sleep_features_N1_selection_train.npz`: Training set (80% of data)
- `sleep_features_N1_selection_test.npz`: Test set (20% of data)
- `sleep_features_N1_selection_split_report.txt`: Balance check report

**Option 2: Split from NPZ (if data already in NPZ format)**

```bash
python train_test_split.py \
  --input mcrbm_input.npz \
  --output-dir ./ \
  --test-ratio 0.2 \
  --seed 42 \
  --tolerance 5.0
```

This also creates:
- `mcrbm_input_train.npz`: Training set (80% of data)
- `mcrbm_input_test.npz`: Test set (20% of data)
- `mcrbm_input_split_report.txt`: Balance check report

**Features**:
- **Stratified splitting**: Maintains label distribution across train/test
- **Balance checking**: Verifies that both sets have similar label distributions
- **Tolerance control**: Allows up to `--tolerance` percentage points difference between train/test label distributions

The script will warn if the split is imbalanced, which may indicate class imbalance in the original data.

### Step 3: Train and analyze

Follow the standard training and inference steps below.

## Generic Quick Start (For custom datasets)

### 1. Prepare your data

Create an `.npz` file with:
- `d`: Data matrix (epochs × features) as float32
- `epochsLinked`: Epoch labels (optional)
- `epochTime`: Epoch times (optional)

```python
import numpy as np

data = np.random.randn(10000, 15).astype(np.float32)
obsKeys = np.random.randint(0, 6, 10000).astype(np.float32)  # 0-5 for sleep stages

np.savez('data.npz', d=data, epochsLinked=obsKeys, epochTime=np.zeros((10000, 1)))
```

### 2. Train the model

```bash
cd scripts
python train_mcrbm.py \
  --config-dir ../configuration_files \
  --exp-config exp_config.ini \
  --model-config model_config.ini
```

Training will create:
- `./weights/`: Model checkpoints
- `./plots/energy/`: Energy evolution plots
- `visData.npz`: Training data snapshot

### 3. Extract latent states

```bash
python infer_mcrbm.py \
  --model-dir ../experiments/my_experiment/weights \
  --exp-dir ../experiments/my_experiment \
  --analyze
```

This generates:
- `./analysis/hidden_activations.png`: Hidden unit activations
- `./analysis/binary_activations.png`: Binarized states
- `./analysis/state_feature_analysis.png`: Per-state feature means
- `./analysis/state_sequence.png`: State transitions over time

## Data Format

### Input Data (.npz)

```python
{
    'd': array (N_samples × N_features) - normalized to float32
    'epochsLinked': array (N_samples,) - sleep stage labels (optional)
    'epochTime': array (N_samples × 2) - time info (optional)
}
```

Typical feature dimensions (from HDF5 pipeline):
- 13-15 features: [Index_W, Index_R, Index_N, Index_1-4, Delta, Theta, Aperiodic, DFA, MSE, EOG, ...]

### Training Output

**Weights** (`ws_final.mat`):
- `VF`: Factor weights (N_features × N_factors)
- `FH`: Factor-to-hidden mapping (N_factors × N_hid_cov)
- `bias_cov`: Covariance hidden biases
- `bias_vis`: Visible biases
- `w_mean`: Mean weights (N_features × N_hid_mean)
- `bias_mean`: Mean biases

**Analysis** (`analysis/`):
- `activations.npz`: Hidden unit probabilities
- `state_info.npy`: Unique states and counts
- Visualization PNGs

## Utility Scripts
### train_test_split_hdf5.py

Split HDF5 dataset directly into balanced train and test NPZ sets.

**Purpose**: Create holdout test sets for evaluation directly from HDF5 files while ensuring both sets have similar label distributions. Outputs are saved as NPZ for mcRBM training.

**Configuration** (edit script defaults):
```python
INPUT_HDF5_PATH = r"path/to/sleep_features_N1_selection.h5"
OUTPUT_DIR = r"path/to/output"
TEST_RATIO = 0.2
RANDOM_SEED = 42
BALANCE_TOLERANCE = 5.0
```

**Command-line Usage**:
```bash
python train_test_split_hdf5.py \
  --input path/to/file.h5 \
  --output-dir ./ \
  --test-ratio 0.2 \
  --seed 42 \
  --tolerance 5.0
```

**Options**:
- `--input`: Input .h5/.hdf5 or .npz file (default: config INPUT_HDF5_PATH)
- `--output-dir`: Output directory (default: config OUTPUT_DIR)
- `--test-ratio`: Test set fraction, 0-1 (default: config TEST_RATIO)
- `--seed`: Random seed for reproducibility (default: config RANDOM_SEED)
- `--tolerance`: Max allowed percentage point difference in label distributions (default: config BALANCE_TOLERANCE)

**Input Format** (HDF5):
HDF5 file with subject groups containing:
- `features`: Feature matrix (samples × features)
- Optional: labels, epochTime attributes/datasets

**Output**:
- `{input}_train.npz`: Training set (NPZ format)
- `{input}_test.npz`: Test set (NPZ format)
- `{input}_split_report.txt`: Balance check report with per-label statistics

**Features**:
- **HDF5 support**: Automatically aggregates features from all subject groups
- **Stratified splitting**: Maintains label distribution across train/test
- **Balance checking**: Verifies both sets have similar label distributions
- **Config-driven**: Default paths can be set at top of script



### train_test_split.py

Split a dataset into balanced train and test sets.

**Purpose**: Create holdout test sets for evaluation while ensuring both sets have similar label distributions.

**Usage**:
```bash
python train_test_split.py \
  --input data.npz \
  --output-dir ./ \
  --test-ratio 0.2 \
  --seed 42 \
  --tolerance 5.0
```

**Options**:
- `--input`: Input .npz file
- `--output-dir`: Output directory (default: same as input)
- `--test-ratio`: Test set fraction, 0-1 (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)
- `--tolerance`: Max allowed percentage point difference in label distributions (default: 5.0)

**Output**:
- `{input}_train.npz`: Training set
- `{input}_test.npz`: Test set
- `{input}_split_report.txt`: Balance check report with per-label statistics

**Stratification**: Automatically stratifies by label if present, ensuring both train and test have similar distributions.

### strip_labels.py

Remove sleep stage labels from a dataset for purely unsupervised learning.

**Purpose**: Create label-free datasets to ensure the model learns patterns without ground truth information.

**Usage**:
```bash
python strip_labels.py \
  --input data.npz \
  --output data_no_labels.npz \
  --keep-epoch-time
```

**Options**:
- `--input`: Input .npz file
- `--output`: Output .npz file (default: `{input}_stripped.npz`)
- `--keep-epoch-time`: Keep epochTime field (default: True)
- `--no-keep-epoch-time`: Remove epochTime field as well

**Output**:
- `.npz` file with only the `d` (data) field
- `epochTime` optionally preserved if present
- `epochsLinked` (labels) removed

## Model Architecture

The mcRBM has two types of hidden units:

1. **Covariance hidden units**: Model feature correlations through normalized data
   - Compute: `h_cov = σ(-0.5 * (FH)^T (VF)^T x_norm)^2 + b_cov)`
   
2. **Mean hidden units**: Model feature means
   - Compute: `h_mean = σ(w_mean^T x + b_mean)`

Joint distribution energy:
```
E(x, h_cov, h_mean) = 0.5 ||x||^2 
                      + Σ log(1 + exp(t_cov))
                      - Σ log(1 + exp(t_mean))
                      - (b_vis)^T x
```

## Training Details

### Hyperparameters

- **epsilon**: Base learning rate (default: 0.001)
- **num_epochs**: Training epochs (default: 200)
- **batch_size**: Minibatch size (default: 128)
- **hmc_step_nr**: HMC leapfrog steps (default: 20)
- **dropout_cov**: Covariance hidden dropout (default: 0.3)
- **dropout_mean**: Mean hidden dropout (default: 0.15)

### Learning Rates (per parameter)
- VF: `2 * epsilon`
- FH: `0.02 * epsilon` (after `startFH` epochs)
- Biases: `0.02 * epsilon`
- w_mean: `0.2 * epsilon`
- bias_mean: `0.1 * epsilon`

### Sampling

- **Contrastive Divergence** (doPCD=0): Use training data as negative samples
- **Persistent CD** (doPCD=1): Maintain chain across minibatches (more stable)

## Interpreting Results

### Hidden Activations

- **p_hc**: Covariance hidden probabilities (N_samples × N_hid_cov)
  - Indicates which factor combinations are active
  - Values closer to 1 = strong activation
  
- **p_hm**: Mean hidden probabilities (N_samples × N_hid_mean)
  - Indicates which feature mean patterns are active

### Latent States

Binary states = (p_hc | p_hm) ≥ 0.5

Each unique binary pattern represents a discovered latent state. Analyze:
1. **State frequency**: How often each state occurs
2. **Feature profile**: Mean features for each state
3. **Transitions**: State transition probabilities
4. **Stage alignment**: Correspondence with sleep labels

### Quality Metrics

Monitor during training:
- **Energy**: Should decrease and stabilize
- **Gradient norms**: Should decrease over time
- **HMC rejection rate**: Target ~0.9

## Advanced Usage

### Custom Preprocessing

Edit `exp_config.ini`:
```ini
logFlag = True              # Log-transform positive features
scaleFlag = True            # Normalize features
scaling = standard          # 'standard', 'minmax', 'robust'
doPCA = False               # Reduce dimensionality
whitenFlag = False          # ZCA whitening
```

### Fine-tuning Model Architecture

For sleep data, recommended settings:
- **num_fac**: 32-128 (more = more expressive)
- **num_hid_cov**: Equal to num_fac
- **num_hid_mean**: ~0.5 * num_hid_cov
- **batch_size**: 64-256 (larger = more stable)

### Stopping Training Early

Create a file named `stop_now` in the experiment directory:
```bash
touch ./experiments/my_experiment/stop_now
```

### Loading Checkpoints

```python
from scipy.io import loadmat

weights = loadmat('./weights/ws_epoch100.mat')
VF = weights['VF']
FH = weights['FH']
# ... etc
```

## GPU Acceleration

### Enabling GPU Support

1. **Install CuPy** (if not already installed):
   ```bash
   # For CUDA 11.x (most common)
   pip install cupy-cuda11x
   
   # For CUDA 12.x
   pip install cupy-cuda12x
   
   # Check your CUDA version: nvidia-smi
   ```

2. **Configure for GPU** in `model_config.ini`:
   ```ini
   [COMPUTE_BACKEND]
   use_gpu = True          # Enable GPU
   gpu_id = 0              # Which GPU (0, 1, 2, ...)
   ```

3. **Run training** (same as CPU):
   ```bash
   python train_mcrbm.py --config-dir ../configuration_files --exp-config exp_config.ini --model-config model_config.ini
   ```

### Performance Expectations

- **CPU (NumPy)**: Baseline performance
- **GPU (CuPy)**: 5-20x speedup depending on:
  - Model size (num_fac, hidden units)
  - Batch size (larger → better GPU utilization)
  - Data size (more samples → amortizes overhead)

### GPU Memory Management

Monitor GPU memory during training:
```python
from scripts.array_backend import get_backend

backend = get_backend(use_gpu=True, gpu_id=0, verbose=True)
print(backend.get_memory_info())
# Output: {'used_mb': 1234.5, 'total_mb': 24576.0}
```

Reduce GPU memory usage:
- Decrease `batch_size` (128 → 64)
- Reduce `num_fac` or hidden units
- Enable gradient checkpointing (if implemented)

### Troubleshooting GPU

**"CuPy not available" warning:**
- CuPy not installed: Run `pip install cupy-cuda11x`
- CUDA drivers outdated: Update NVIDIA drivers
- GPU compute capability too old: Use CPU instead
- **Solution:** Code automatically falls back to NumPy

**GPU out of memory:**
- Reduce batch size: `batch_size = 64` (from 128)
- Reduce model size: `num_fac = 32` (from 64)
- Use CPU instead (set `use_gpu = False`)

**Slow GPU training:**
- GPU may not be utilized: Check batch size (should be ≥32)
- System might be CPU-bottlenecked: Check data loading
- Verify CUDA/GPU in use: See initialization messages

## Troubleshooting

### OutOfMemory errors
- Reduce `batch_size`
- Reduce `num_fac` or hidden units
- Use CPU instead of GPU

### Energy not decreasing
- Reduce learning rate (`epsilon`)
- Increase `num_fac`
- Check data normalization

### Poor latent state discovery
- Ensure data is properly normalized
- Increase `num_epochs`
- Try different `num_fac` (64, 128)
- Check data quality and range

## References

1. Ranzato, M., Boureau, Y. L., & LeCun, Y. (2013). Sparse feature learning for deep belief networks. In Advances in neural information processing systems (pp. 1185-1192).

2. Dilon & Andriesse's original sleep-focused implementation for additional context.

3. CuPy installation documentation for GPU support: https://docs.cupy.dev/en/stable/install.html

## License

Based on academic research code. See DilonAndriesse/mcRBM for original references.

## Notes

- This CPU implementation is slower than GPU but more portable
- For large datasets (>100K samples), consider GPU acceleration
- Recommended to normalize features to ~[-1, 1] or [0, 1] range
- Results depend strongly on hyperparameter selection and data quality
