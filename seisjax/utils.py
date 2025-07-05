"""
Utility functions for seismic data loading and preprocessing.

This module provides helper functions for loading seismic data from various formats,
preprocessing, and visualization utilities.
"""

from typing import Tuple, Union, Optional
import jax.numpy as jnp
import numpy as np


def rescale_volume(seismic: jnp.ndarray, low: float = 0, high: float = 100) -> jnp.ndarray:
    """
    Rescale 3D seismic volumes to 0-255 range, clipping values between low and high percentiles.
    
    Args:
        seismic: Input seismic volume
        low: Lower percentile for clipping
        high: Upper percentile for clipping
        
    Returns:
        Rescaled seismic volume
    """
    minval = jnp.percentile(seismic, low)
    maxval = jnp.percentile(seismic, high)
    seismic = jnp.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255
    return seismic


def normalize_volume(seismic: jnp.ndarray, method: str = 'minmax') -> jnp.ndarray:
    """
    Normalize seismic volume using different methods.
    
    Args:
        seismic: Input seismic volume
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized seismic volume
    """
    if method == 'minmax':
        min_val = jnp.min(seismic)
        max_val = jnp.max(seismic)
        return (seismic - min_val) / (max_val - min_val + 1e-10)
    elif method == 'zscore':
        mean_val = jnp.mean(seismic)
        std_val = jnp.std(seismic)
        return (seismic - mean_val) / (std_val + 1e-10)
    elif method == 'robust':
        median_val = jnp.median(seismic)
        mad_val = jnp.median(jnp.abs(seismic - median_val))
        return (seismic - median_val) / (mad_val + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_slice(
    volume: jnp.ndarray,
    slice_type: str,
    slice_index: int,
    axis_labels: Tuple[str, str, str] = ('inline', 'crossline', 'time')
) -> jnp.ndarray:
    """
    Extract a 2D slice from a 3D seismic volume.
    
    Args:
        volume: 3D seismic volume
        slice_type: Type of slice ('inline', 'crossline', 'time')
        slice_index: Index of the slice to extract
        axis_labels: Labels for the three axes
        
    Returns:
        2D slice
    """
    if slice_type == axis_labels[0]:  # inline
        return volume[slice_index, :, :]
    elif slice_type == axis_labels[1]:  # crossline
        return volume[:, slice_index, :]
    elif slice_type == axis_labels[2]:  # time
        return volume[:, :, slice_index]
    else:
        raise ValueError(f"Unknown slice type: {slice_type}")


def calculate_statistics(volume: jnp.ndarray) -> dict:
    """
    Calculate basic statistics for a seismic volume.
    
    Args:
        volume: Input seismic volume
        
    Returns:
        Dictionary with statistics
    """
    return {
        'min': float(jnp.min(volume)),
        'max': float(jnp.max(volume)),
        'mean': float(jnp.mean(volume)),
        'std': float(jnp.std(volume)),
        'median': float(jnp.median(volume)),
        'shape': volume.shape,
        'dtype': str(volume.dtype)
    }


def create_coordinate_arrays(
    shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create coordinate arrays for seismic data.
    
    Args:
        shape: Shape of the seismic volume (nz, ny, nx)
        spacing: Spacing in each dimension
        origin: Origin coordinates
        
    Returns:
        Tuple of coordinate arrays (z, y, x)
    """
    nz, ny, nx = shape
    dz, dy, dx = spacing
    z0, y0, x0 = origin
    
    z = jnp.arange(nz) * dz + z0
    y = jnp.arange(ny) * dy + y0
    x = jnp.arange(nx) * dx + x0
    
    return z, y, x


def apply_agc(
    data: jnp.ndarray,
    window_length: int = 100,
    axis: int = -1
) -> jnp.ndarray:
    """
    Apply Automatic Gain Control (AGC) to seismic data.
    
    Args:
        data: Input seismic data
        window_length: Length of the AGC window
        axis: Axis along which to apply AGC
        
    Returns:
        AGC-processed data
    """
    # Move the specified axis to the last position
    data = jnp.moveaxis(data, axis, -1)
    
    # Apply AGC along the last axis
    def agc_trace(trace):
        # Calculate running RMS
        trace_squared = trace ** 2
        rms = jnp.sqrt(jnp.convolve(trace_squared, jnp.ones(window_length) / window_length, mode='same'))
        
        # Apply gain
        gain = 1.0 / (rms + 1e-10)
        return trace * gain
    
    # Apply to all traces
    if data.ndim == 1:
        result = agc_trace(data)
    else:
        # Flatten all dimensions except the last one
        original_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])
        result_flat = jnp.array([agc_trace(trace) for trace in data_flat])
        result = result_flat.reshape(original_shape)
    
    # Move the axis back to its original position
    result = jnp.moveaxis(result, -1, axis)
    
    return result


def bandpass_filter(
    data: jnp.ndarray,
    low_freq: float,
    high_freq: float,
    fs: float,
    axis: int = -1
) -> jnp.ndarray:
    """
    Apply bandpass filter to seismic data.
    
    Args:
        data: Input seismic data
        low_freq: Low cutoff frequency
        high_freq: High cutoff frequency
        fs: Sampling frequency
        axis: Axis along which to apply filter
        
    Returns:
        Filtered data
    """
    # Move the specified axis to the last position
    data = jnp.moveaxis(data, axis, -1)
    
    # Apply filter in frequency domain
    def filter_trace(trace):
        # FFT
        fft_data = jnp.fft.fft(trace)
        freqs = jnp.fft.fftfreq(len(trace), 1/fs)
        
        # Create filter mask
        mask = (jnp.abs(freqs) >= low_freq) & (jnp.abs(freqs) <= high_freq)
        
        # Apply filter
        fft_filtered = fft_data * mask
        
        # IFFT
        return jnp.real(jnp.fft.ifft(fft_filtered))
    
    # Apply to all traces
    if data.ndim == 1:
        result = filter_trace(data)
    else:
        # Flatten all dimensions except the last one
        original_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])
        result_flat = jnp.array([filter_trace(trace) for trace in data_flat])
        result = result_flat.reshape(original_shape)
    
    # Move the axis back to its original position
    result = jnp.moveaxis(result, -1, axis)
    
    return result 