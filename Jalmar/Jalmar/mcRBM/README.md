# mcRBM (Mean-Covariance Restricted Boltzmann Machine)

A pure Python implementation of the mean-covariance Restricted Boltzmann Machine (mcRBM) for unsupervised learning of latent sleep states and features.

Based on the original implementation by Marc'Aurelio Ranzato and the sleep-focused adaptation by DilonAndriesse.

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
num_fac = 64           # Number of factors
num_hid_cov = 64       # Covariance hidden units
num_hid_mean = 32      # Mean hidden units
num_epochs = 200       # Training epochs
batch_size = 128       # Batch size
```

**Backend Selection:**
- `use_gpu = False` (default): Uses NumPy on CPU
- `use_gpu = True`: Uses CuPy on GPU (requires CuPy installation)
- If CuPy unavailable with `use_gpu=True`: Automatically falls back to NumPy with warning

### 2. Prepare your data

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

### 3. Train the model

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

### 4. Extract latent states

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
