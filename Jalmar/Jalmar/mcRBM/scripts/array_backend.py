"""
Array backend abstraction for NumPy/CuPy compatibility.
Provides unified interface for CPU and GPU computation.
"""

import numpy as np
import warnings


class ArrayBackend:
    """
    Manages array library selection (NumPy or CuPy).
    Provides unified interface and memory management.
    """
    
    def __init__(self, use_gpu=False, gpu_id=0, verbose=True):
        """
        Initialize backend.
        
        Args:
            use_gpu: Whether to use GPU (CuPy)
            gpu_id: GPU device ID if using CuPy
            verbose: Print backend information
        """
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.verbose = verbose
        self.backend_name = "unknown"
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize NumPy or CuPy based on use_gpu flag."""
        if self.use_gpu:
            try:
                import cupy as cp
                
                # Set GPU device
                cp.cuda.Device(self.gpu_id).use()
                
                self.xp = cp
                self.backend_name = f"CuPy (GPU {self.gpu_id})"
                
                if self.verbose:
                    print(f"✓ Backend: {self.backend_name}")
                    print(f"  GPU: {cp.cuda.Device().get_device_id()}")
                    print(f"  GPU Memory: {cp.cuda.Device().mem_info}")
                
            except ImportError:
                warnings.warn("CuPy not available. Falling back to NumPy.", UserWarning)
                self.xp = np
                self.backend_name = "NumPy (CPU, fallback)"
                self.use_gpu = False
                
                if self.verbose:
                    print(f"✓ Backend: {self.backend_name}")
                    print("  Install CuPy for GPU support: pip install cupy-cuda11x")
            
            except RuntimeError as e:
                warnings.warn(f"GPU initialization failed: {e}. Falling back to NumPy.", UserWarning)
                self.xp = np
                self.backend_name = "NumPy (CPU, fallback)"
                self.use_gpu = False
                
                if self.verbose:
                    print(f"✓ Backend: {self.backend_name}")
        
        else:
            self.xp = np
            self.backend_name = "NumPy (CPU)"
            
            if self.verbose:
                print(f"✓ Backend: {self.backend_name}")
    
    def array(self, *args, **kwargs):
        """Create array on current backend."""
        return self.xp.array(*args, **kwargs)
    
    def zeros(self, *args, **kwargs):
        """Create zero array on current backend."""
        return self.xp.zeros(*args, **kwargs)
    
    def ones(self, *args, **kwargs):
        """Create ones array on current backend."""
        return self.xp.ones(*args, **kwargs)
    
    def empty(self, *args, **kwargs):
        """Create empty array on current backend."""
        return self.xp.empty(*args, **kwargs)
    
    def randn(self, *args, **kwargs):
        """Create random normal array on current backend."""
        if self.use_gpu:
            import cupy as cp
            return cp.random.randn(*args, **kwargs)
        else:
            return np.random.randn(*args, **kwargs)
    
    def to_numpy(self, arr):
        """Transfer array to NumPy (CPU)."""
        if self.use_gpu:
            return arr.get()  # CuPy to NumPy
        else:
            return arr  # Already NumPy
    
    def to_gpu(self, arr):
        """Transfer array to GPU (CuPy)."""
        if self.use_gpu and isinstance(arr, np.ndarray):
            import cupy as cp
            return cp.asarray(arr)
        else:
            return arr  # Already on GPU or GPU disabled
    
    def synchronize(self):
        """Synchronize GPU operations (if GPU is used)."""
        if self.use_gpu:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
    
    def get_memory_info(self):
        """Get memory usage information."""
        if self.use_gpu:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            used_mb = mempool.used_bytes() / 1024 / 1024
            total_mb = cp.cuda.Device().mem_info[1] / 1024 / 1024
            return {'used_mb': used_mb, 'total_mb': total_mb}
        else:
            return {'used_mb': 'N/A', 'total_mb': 'N/A', 'note': 'CPU memory not tracked'}
    
    def print_memory_info(self):
        """Print current memory usage."""
        info = self.get_memory_info()
        if self.use_gpu:
            print(f"GPU Memory: {info['used_mb']:.1f} MB / {info['total_mb']:.1f} MB")
        else:
            print("Memory tracking: CPU (not tracked)")


def get_backend(use_gpu=False, gpu_id=0, verbose=True):
    """
    Factory function to create array backend.
    
    Args:
        use_gpu: Use GPU if available
        gpu_id: GPU device ID
        verbose: Print backend info
        
    Returns:
        ArrayBackend instance with .xp attribute
    """
    backend = ArrayBackend(use_gpu=use_gpu, gpu_id=gpu_id, verbose=verbose)
    return backend
