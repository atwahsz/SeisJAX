"""
Coherence attributes for seismic interpretation.

This module provides JAX-accelerated implementations of coherence attributes
commonly used in seismic interpretation and structural geology.
"""

from typing import Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from .utils import _jax_gradient


def _sliding_window_3d(x: jnp.ndarray, window_shape: Tuple[int, int, int]) -> jnp.ndarray:
    """
    Create sliding windows for 3D coherence calculation.
    
    Args:
        x: Input 3D array
        window_shape: Shape of the sliding window (nz, ny, nx)
        
    Returns:
        Windowed array with shape (..., nz, ny, nx)
    """
    # Use JAX's sliding window approach for efficient windowing
    strides = (1, 1, 1)
    
    # Create patches using lax.conv_general_dilated with appropriate dimensions
    patches = []
    nz, ny, nx = window_shape
    
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                rolled = jnp.roll(x, shift=(-i+nz//2, -j+ny//2, -k+nx//2), axis=(0, 1, 2))
                patches.append(rolled)
    
    return jnp.stack(patches, axis=-1)


@jax.jit
def semblance(
    x: jnp.ndarray,
    window_shape: Tuple[int, int, int] = (5, 5, 5),
    axis: int = -1
) -> jnp.ndarray:
    """
    Compute semblance coherence attribute.
    
    Semblance is a normalized measure of coherence that ranges from 0 to 1,
    where 1 indicates perfect coherence and 0 indicates no coherence.
    
    Args:
        x: Input seismic data (3D array)
        window_shape: Shape of the analysis window (nz, ny, nx)
        axis: Axis along which to compute semblance (default: -1)
        
    Returns:
        Semblance coherence volume
    """
    # Ensure input is 3D
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    nz, ny, nx = window_shape
    
    # Create padded version for boundary handling
    pad_width = ((nz//2, nz//2), (ny//2, ny//2), (nx//2, nx//2))
    x_padded = jnp.pad(x, pad_width, mode='reflect')
    
    # Initialize output
    output = jnp.zeros_like(x)
    
    # Vectorized semblance computation
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                # Extract local window
                z_start, z_end = i, i + x.shape[0]
                y_start, y_end = j, j + x.shape[1]
                x_start, x_end = k, k + x.shape[2]
                
                window = x_padded[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Compute semblance
                trace_sum = jnp.sum(window, axis=(1, 2), keepdims=True)
                trace_sum_sq = jnp.sum(trace_sum**2, axis=0)
                
                individual_sq = jnp.sum(window**2, axis=(1, 2))
                individual_sum_sq = jnp.sum(individual_sq, axis=0)
                
                # Avoid division by zero
                numerator = trace_sum_sq
                denominator = individual_sum_sq * (ny * nx)
                
                semb = jnp.where(denominator > 1e-10, numerator / denominator, 0.0)
                
                if i == nz//2 and j == ny//2 and k == nx//2:
                    output = semb.squeeze()
    
    return output


@jax.jit
def eigenstructure_coherence(
    x: jnp.ndarray,
    window_shape: Tuple[int, int, int] = (5, 5, 5)
) -> jnp.ndarray:
    """
    Compute eigenstructure coherence attribute.
    
    This coherence measure is based on the eigenvalues of the structure tensor
    and provides a measure of linear coherence in the data.
    
    Args:
        x: Input seismic data (3D array)
        window_shape: Shape of the analysis window (nz, ny, nx)
        
    Returns:
        Eigenstructure coherence volume
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    # Compute gradients
    dz = _jax_gradient(x, axis=0)
    dy = _jax_gradient(x, axis=1)
    dx = _jax_gradient(x, axis=2)
    
    # Compute structure tensor components
    nz, ny, nx = window_shape
    pad_width = ((nz//2, nz//2), (ny//2, ny//2), (nx//2, nx//2))
    
    # Structure tensor components
    Jxx = jnp.pad(dx * dx, pad_width, mode='reflect')
    Jyy = jnp.pad(dy * dy, pad_width, mode='reflect')
    Jzz = jnp.pad(dz * dz, pad_width, mode='reflect')
    Jxy = jnp.pad(dx * dy, pad_width, mode='reflect')
    Jxz = jnp.pad(dx * dz, pad_width, mode='reflect')
    Jyz = jnp.pad(dy * dz, pad_width, mode='reflect')
    
    # Smooth structure tensor with local averaging
    kernel = jnp.ones(window_shape) / jnp.prod(jnp.array(window_shape))
    
    # Use convolution for smoothing
    from jax.scipy.signal import convolve
    
    Jxx_smooth = convolve(Jxx, kernel, mode='same')[nz//2:-nz//2, ny//2:-ny//2, nx//2:-nx//2]
    Jyy_smooth = convolve(Jyy, kernel, mode='same')[nz//2:-nz//2, ny//2:-ny//2, nx//2:-nx//2]
    Jzz_smooth = convolve(Jzz, kernel, mode='same')[nz//2:-nz//2, ny//2:-ny//2, nx//2:-nx//2]
    Jxy_smooth = convolve(Jxy, kernel, mode='same')[nz//2:-nz//2, ny//2:-ny//2, nx//2:-nx//2]
    Jxz_smooth = convolve(Jxz, kernel, mode='same')[nz//2:-nz//2, ny//2:-ny//2, nx//2:-nx//2]
    Jyz_smooth = convolve(Jyz, kernel, mode='same')[nz//2:-nz//2, ny//2:-ny//2, nx//2:-nx//2]
    
    # Compute eigenvalues for each point
    def compute_eigenvalues(point_idx):
        i, j, k = point_idx
        
        # Build structure tensor matrix
        J = jnp.array([
            [Jxx_smooth[i, j, k], Jxy_smooth[i, j, k], Jxz_smooth[i, j, k]],
            [Jxy_smooth[i, j, k], Jyy_smooth[i, j, k], Jyz_smooth[i, j, k]],
            [Jxz_smooth[i, j, k], Jyz_smooth[i, j, k], Jzz_smooth[i, j, k]]
        ])
        
        # Compute eigenvalues
        eigenvals = jnp.linalg.eigvals(J)
        eigenvals = jnp.sort(eigenvals)[::-1]  # Sort in descending order
        
        # Coherence measure: (λ1 - λ2) / (λ1 + λ2 + λ3)
        coherence = (eigenvals[0] - eigenvals[1]) / (jnp.sum(eigenvals) + 1e-10)
        return coherence
    
    # Vectorized computation
    indices = jnp.mgrid[0:x.shape[0], 0:x.shape[1], 0:x.shape[2]]
    indices = indices.reshape(3, -1).T
    
    coherence_values = jax.vmap(compute_eigenvalues)(indices)
    coherence_volume = coherence_values.reshape(x.shape)
    
    return coherence_volume


@jax.jit
def energy_ratio_coherence(
    x: jnp.ndarray,
    window_shape: Tuple[int, int, int] = (5, 5, 5)
) -> jnp.ndarray:
    """
    Compute energy ratio coherence attribute.
    
    This coherence measure compares the energy of the central trace
    to the energy of surrounding traces in a local window.
    
    Args:
        x: Input seismic data (3D array)
        window_shape: Shape of the analysis window (nz, ny, nx)
        
    Returns:
        Energy ratio coherence volume
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    nz, ny, nx = window_shape
    pad_width = ((nz//2, nz//2), (ny//2, ny//2), (nx//2, nx//2))
    x_padded = jnp.pad(x, pad_width, mode='reflect')
    
    # Initialize output
    output = jnp.zeros_like(x)
    
    # Compute energy ratio for each point
    def compute_energy_ratio(center_idx):
        i, j, k = center_idx
        
        # Extract local window
        window = x_padded[i:i+nz, j:j+ny, k:k+nx]
        
        # Central trace
        center_trace = window[nz//2, ny//2, nx//2]
        
        # Energy of central trace
        center_energy = jnp.sum(center_trace**2)
        
        # Energy of all traces in window
        total_energy = jnp.sum(window**2)
        
        # Energy ratio
        ratio = center_energy / (total_energy + 1e-10)
        
        return ratio
    
    # Vectorized computation
    indices = jnp.mgrid[0:x.shape[0], 0:x.shape[1], 0:x.shape[2]]
    indices = indices.reshape(3, -1).T
    
    energy_values = jax.vmap(compute_energy_ratio)(indices)
    energy_volume = energy_values.reshape(x.shape)
    
    return energy_volume


@jax.jit
def c3_coherence(
    x: jnp.ndarray,
    window_shape: Tuple[int, int, int] = (5, 5, 5)
) -> jnp.ndarray:
    """
    Compute C3 coherence attribute.
    
    C3 coherence is a robust measure that uses the covariance matrix
    of the complex trace attributes within a local window.
    
    Args:
        x: Input seismic data (3D array)
        window_shape: Shape of the analysis window (nz, ny, nx)
        
    Returns:
        C3 coherence volume
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    # Import hilbert from attributes module
    from .attributes import analytic_signal
    
    # Compute analytic signal
    analytic = analytic_signal(x, axis=0)
    
    nz, ny, nx = window_shape
    pad_width = ((nz//2, nz//2), (ny//2, ny//2), (nx//2, nx//2))
    analytic_padded = jnp.pad(analytic, pad_width, mode='reflect')
    
    def compute_c3(center_idx):
        i, j, k = center_idx
        
        # Extract local window
        window = analytic_padded[i:i+nz, j:j+ny, k:k+nx]
        
        # Flatten spatial dimensions
        traces = window.reshape(nz, -1)
        
        # Compute covariance matrix
        traces_centered = traces - jnp.mean(traces, axis=1, keepdims=True)
        cov_matrix = jnp.dot(traces_centered, traces_centered.conj().T) / (traces.shape[1] - 1)
        
        # Compute eigenvalues
        eigenvals = jnp.linalg.eigvals(cov_matrix)
        eigenvals = jnp.real(eigenvals)
        eigenvals = jnp.sort(eigenvals)[::-1]
        
        # C3 coherence
        c3 = eigenvals[0] / (jnp.sum(eigenvals) + 1e-10)
        
        return c3
    
    # Vectorized computation
    indices = jnp.mgrid[0:x.shape[0], 0:x.shape[1], 0:x.shape[2]]
    indices = indices.reshape(3, -1).T
    
    c3_values = jax.vmap(compute_c3)(indices)
    c3_volume = c3_values.reshape(x.shape)
    
    return c3_volume


@jax.jit
def variance_coherence(
    x: jnp.ndarray,
    window_shape: Tuple[int, int, int] = (5, 5, 5)
) -> jnp.ndarray:
    """
    Compute variance-based coherence attribute.
    
    This coherence measure is based on the variance of amplitude values
    within a local window, normalized by the mean amplitude.
    
    Args:
        x: Input seismic data (3D array)
        window_shape: Shape of the analysis window (nz, ny, nx)
        
    Returns:
        Variance coherence volume
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    nz, ny, nx = window_shape
    pad_width = ((nz//2, nz//2), (ny//2, ny//2), (nx//2, nx//2))
    x_padded = jnp.pad(x, pad_width, mode='reflect')
    
    def compute_variance_coherence(center_idx):
        i, j, k = center_idx
        
        # Extract local window
        window = x_padded[i:i+nz, j:j+ny, k:k+nx]
        
        # Compute variance and mean
        window_flat = window.flatten()
        mean_val = jnp.mean(window_flat)
        var_val = jnp.var(window_flat)
        
        # Normalized variance (coefficient of variation)
        coherence = 1.0 / (1.0 + var_val / (jnp.abs(mean_val) + 1e-10))
        
        return coherence
    
    # Vectorized computation
    indices = jnp.mgrid[0:x.shape[0], 0:x.shape[1], 0:x.shape[2]]
    indices = indices.reshape(3, -1).T
    
    variance_values = jax.vmap(compute_variance_coherence)(indices)
    variance_volume = variance_values.reshape(x.shape)
    
    return variance_volume 