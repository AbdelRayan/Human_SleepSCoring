"""
mcRBM - Mean-Covariance Restricted Boltzmann Machine
CPU-based (NumPy) implementation for sleep analysis
"""

from .mcrbm import mcRBM
from .infer_states import mcRBMInference
from .data_preproc import DataPreproc

__version__ = '1.0.0'
__all__ = ['mcRBM', 'mcRBMInference', 'DataPreproc']
