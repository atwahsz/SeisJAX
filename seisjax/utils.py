"""
Utility functions for seismic data preprocessing and manipulation.

This module provides common utilities for working with seismic data,
including normalization, filtering, and geometric transformations.
"""

from typing import Dict, Union, Optional, Tuple, Any

import jax
import jax.numpy as jnp
from jax import lax


def _jax_gradient(f: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    JAX implementation of gradient function using central differences.
    """
    # Normalize axis
    if axis < 0:
        axis = f.ndim + axis
        
    # Use central differences for interior points and forward/backward for edges
    # Move axis to last position for easier processing
    if axis != f.ndim - 1:
        axes = list(range(f.ndim))
        axes[axis], axes[-1] = axes[-1], axes[axis]
        f = jnp.transpose(f, axes)
    
    # Get the size along the gradient axis
    n = f.shape[-1]
    
    if n < 2:
        return jnp.zeros_like(f)
    elif n == 2:
        # Only two points, use simple difference
        return jnp.diff(f, axis=-1)
    else:
        # Use central differences for interior, forward/backward for edges
        # Forward difference for first point
        forward = f[..., 1] - f[..., 0]
        
        # Central differences for interior points
        central = (f[..., 2:] - f[..., :-2]) / 2.0
        
        # Backward difference for last point  
        backward = f[..., -1] - f[..., -2]
        
        # Combine
        result = jnp.concatenate([
            jnp.expand_dims(forward, -1),
            central,
            jnp.expand_dims(backward, -1)
        ], axis=-1)
    
    # Move axis back to original position
    if axis != f.ndim - 1:
        inv_axes = list(range(f.ndim))
        inv_axes[axis], inv_axes[-1] = inv_axes[-1], inv_axes[axis]
        result = jnp.transpose(result, inv_axes)
    
    return result


def rescale_volume(
    data: jnp.ndarray, 
    percentiles: Tuple[float, float] = (2.0, 98.0)
) -> jnp.ndarray:
    """
    Rescale seismic volume to specified percentile range.
    
    Args:
        data: Input seismic data
        percentiles: Tuple of (lower_percentile, upper_percentile) for rescaling
        
    Returns:
        Rescaled data with values between 0 and 1
    """
    lower, upper = jnp.percentile(data, jnp.array(percentiles))
    return jnp.clip((data - lower) / (upper - lower), 0.0, 1.0)


def normalize_volume(
    data: jnp.ndarray, 
    method: str = "minmax",
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> jnp.ndarray:
    """
    Normalize seismic volume using various methods.
    
    Args:
        data: Input seismic data
        method: Normalization method ('minmax', 'zscore', 'robust')
        axis: Axis or axes along which to normalize. If None, normalize globally.
        
    Returns:
        Normalized data
    """
    if method == "minmax":
        data_min = jnp.min(data, axis=axis, keepdims=True)
        data_max = jnp.max(data, axis=axis, keepdims=True)
        return (data - data_min) / (data_max - data_min + 1e-8)
    
    elif method == "zscore":
        data_mean = jnp.mean(data, axis=axis, keepdims=True)
        data_std = jnp.std(data, axis=axis, keepdims=True)
        return (data - data_mean) / (data_std + 1e-8)
    
    elif method == "robust":
        data_median = jnp.median(data, axis=axis, keepdims=True)
        data_mad = jnp.median(jnp.abs(data - data_median), axis=axis, keepdims=True)
        return (data - data_median) / (1.4826 * data_mad + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_slice(
    data: jnp.ndarray,
    slice_type: str,
    index: int,
    axis: Optional[int] = None
) -> jnp.ndarray:
    """
    Extract a 2D slice from 3D seismic data.
    
    Args:
        data: 3D seismic data (time, crossline, inline) or (axis0, axis1, axis2)
        slice_type: Type of slice ('inline', 'crossline', 'timeslice', 'axis0', 'axis1', 'axis2')
        index: Index along the specified axis
        axis: Specific axis to slice along (overrides slice_type)
        
    Returns:
        2D slice of the data
    """
    if axis is not None:
        return jnp.take(data, index, axis=axis)
    
    if slice_type in ['inline', 'axis2']:
        return data[:, :, index]
    elif slice_type in ['crossline', 'axis1']:
        return data[:, index, :]
    elif slice_type in ['timeslice', 'axis0']:
        return data[index, :, :]
    else:
        raise ValueError(f"Unknown slice type: {slice_type}")


def calculate_statistics(data: jnp.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for seismic data.
    
    Args:
        data: Input seismic data
        
    Returns:
        Dictionary containing various statistics
    """
    stats = {
        'mean': float(jnp.mean(data)),
        'std': float(jnp.std(data)),
        'min': float(jnp.min(data)),
        'max': float(jnp.max(data)),
        'median': float(jnp.median(data)),
        'p25': float(jnp.percentile(data, 25)),
        'p75': float(jnp.percentile(data, 75)),
        'rms': float(jnp.sqrt(jnp.mean(data**2))),
        'mad': float(jnp.median(jnp.abs(data - jnp.median(data)))),
    }
    return stats


def apply_agc(
    data: jnp.ndarray,
    window_length: int,
    axis: int = 0,
    method: str = "rms"
) -> jnp.ndarray:
    """
    Apply Automatic Gain Control (AGC) to seismic data.
    
    AGC normalizes the amplitude by dividing by a moving window statistic
    to balance amplitudes across the trace.
    
    Args:
        data: Input seismic data
        window_length: Length of the AGC window
        axis: Axis along which to apply AGC (typically time axis)
        method: AGC method ('rms', 'mean', 'median')
        
    Returns:
        AGC-corrected data
    """
    # Move axis to last position for easier processing
    if axis == -1 or axis == data.ndim - 1:
        # Already at last position
        pass
    elif axis == 0:
        # Move first axis to last
        data = jnp.moveaxis(data, 0, -1)
    else:
        # General case - move specified axis to last
        data = jnp.moveaxis(data, axis, -1)
    
    # Apply AGC along the last axis
    def agc_single_trace(trace):
        if method == "rms":
            # Compute RMS in sliding windows
            trace_squared = trace**2
            windowed_power = jnp.convolve(
                trace_squared, 
                jnp.ones(window_length) / window_length, 
                mode='same'
            )
            gain = jnp.sqrt(windowed_power + 1e-12)
        elif method == "mean":
            # Compute mean absolute value in sliding windows
            trace_abs = jnp.abs(trace)
            gain = jnp.convolve(
                trace_abs,
                jnp.ones(window_length) / window_length,
                mode='same'
            ) + 1e-12
        elif method == "median":
            # Use a simple moving average approximation of median
            gain = jnp.convolve(
                jnp.abs(trace),
                jnp.ones(window_length) / window_length,
                mode='same'
            ) + 1e-12
        else:
            raise ValueError(f"Unknown AGC method: {method}")
            
        return trace / gain
    
    # Apply AGC to each trace
    result = jax.vmap(agc_single_trace)(data.reshape(-1, data.shape[-1]))
    result = result.reshape(data.shape)
    
    # Move axis back to original position
    if axis == -1 or axis == data.ndim - 1:
        return result
    elif axis == 0:
        return jnp.moveaxis(result, -1, 0)
    else:
        return jnp.moveaxis(result, -1, axis)


def bandpass_filter(
    data: jnp.ndarray,
    low_freq: float,
    high_freq: float,
    fs: float,
    axis: int = 0,
    filter_type: str = "butterworth"
) -> jnp.ndarray:
    """
    Apply a bandpass filter to seismic data using frequency domain methods.
    
    Args:
        data: Input seismic data
        low_freq: Low cutoff frequency (Hz)
        high_freq: High cutoff frequency (Hz) 
        fs: Sampling frequency (Hz)
        axis: Axis along which to apply filter
        filter_type: Type of filter ('butterworth', 'gaussian')
        
    Returns:
        Filtered data
    """
    # Move axis to last position for easier processing
    if axis == -1 or axis == data.ndim - 1:
        # Already at last position
        work_data = data
    elif axis == 0:
        # Move first axis to last
        work_data = jnp.moveaxis(data, 0, -1)
    else:
        # General case - move specified axis to last
        work_data = jnp.moveaxis(data, axis, -1)
    
    # Get dimensions
    n_samples = work_data.shape[-1]
    
    # Create frequency axis
    freqs = jnp.fft.fftfreq(n_samples, 1.0 / fs)
    freqs = jnp.abs(freqs)
    
    # Create filter
    if filter_type == "butterworth":
        # Simple frequency domain butterworth-like filter
        low_mask = freqs >= low_freq
        high_mask = freqs <= high_freq
        filter_mask = low_mask & high_mask
        filter_response = filter_mask.astype(jnp.float32)
    elif filter_type == "gaussian":
        # Gaussian bandpass filter
        center_freq = (low_freq + high_freq) / 2
        bandwidth = high_freq - low_freq
        filter_response = jnp.exp(-((freqs - center_freq) / (bandwidth / 4))**2)
        # Zero out frequencies outside the band
        filter_response = jnp.where(
            (freqs >= low_freq) & (freqs <= high_freq),
            filter_response, 0.0
        )
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply filter in frequency domain
    def filter_trace(trace):
        fft_trace = jnp.fft.fft(trace)
        filtered_fft = fft_trace * filter_response
        return jnp.real(jnp.fft.ifft(filtered_fft))
    
    # Apply to each trace
    result = jax.vmap(filter_trace)(work_data.reshape(-1, n_samples))
    result = result.reshape(work_data.shape)
    
    # Move axis back to original position
    if axis == -1 or axis == data.ndim - 1:
        return result
    elif axis == 0:
        return jnp.moveaxis(result, -1, 0)
    else:
        return jnp.moveaxis(result, -1, axis) 