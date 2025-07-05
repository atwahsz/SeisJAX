"""
Geometric attributes for seismic structural analysis.

This module provides JAX-accelerated implementations of geometric attributes
used for analyzing seismic structures and geological features.
"""

from typing import Tuple, Union, Optional

import jax
import jax.numpy as jnp
from jax import lax
from .utils import _jax_gradient


@jax.jit
def dip_magnitude(
    x: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    axis_order: str = "txy"
) -> jnp.ndarray:
    """
    Calculate the magnitude of dip in seismic data.
    
    Args:
        x: Input seismic data (can be 2D or 3D)
        dx: Spatial sampling in x-direction
        dy: Spatial sampling in y-direction  
        axis_order: Order of axes ("txy", "tyx", "xyt", etc.)
        
    Returns:
        Dip magnitude
    """
    if x.ndim == 2:
        # 2D case
        # Compute gradients
        if axis_order.startswith("t"):
            grad_x = _jax_gradient(x, axis=1) / dx
            grad_y = _jax_gradient(x, axis=0) / dy
        else:
            grad_x = _jax_gradient(x, axis=1) / dx
            grad_y = _jax_gradient(x, axis=0) / dy
    else:
        # 3D case
        if axis_order == "txy":
            grad_x = _jax_gradient(x, axis=2) / dx
            grad_y = _jax_gradient(x, axis=1) / dy
        elif axis_order == "tyx":
            grad_x = _jax_gradient(x, axis=2) / dx
            grad_y = _jax_gradient(x, axis=0) / dy
        else:
            grad_x = _jax_gradient(x, axis=1) / dx
            grad_y = _jax_gradient(x, axis=0) / dy
    
    # Calculate dip magnitude
    return jnp.sqrt(grad_x**2 + grad_y**2)


@jax.jit
def dip_azimuth(
    x: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    axis_order: str = "txy"
) -> jnp.ndarray:
    """
    Calculate the azimuth of dip in seismic data.
    
    Args:
        x: Input seismic data (can be 2D or 3D)
        dx: Spatial sampling in x-direction
        dy: Spatial sampling in y-direction
        axis_order: Order of axes ("txy", "tyx", "xyt", etc.)
        
    Returns:
        Dip azimuth in radians
    """
    if x.ndim == 2:
        # 2D case
        # Compute gradients
        if axis_order.startswith("t"):
            grad_x = _jax_gradient(x, axis=1) / dx
            grad_y = _jax_gradient(x, axis=0) / dy
        else:
            grad_x = _jax_gradient(x, axis=1) / dx
            grad_y = _jax_gradient(x, axis=0) / dy
    else:
        # 3D case
        if axis_order == "txy":
            grad_x = _jax_gradient(x, axis=2) / dx
            grad_y = _jax_gradient(x, axis=1) / dy
        elif axis_order == "tyx":
            grad_x = _jax_gradient(x, axis=2) / dx
            grad_y = _jax_gradient(x, axis=0) / dy
        else:
            grad_x = _jax_gradient(x, axis=1) / dx
            grad_y = _jax_gradient(x, axis=0) / dy
    
    # Calculate dip azimuth
    return jnp.arctan2(grad_y, grad_x)


@jax.jit
def strike_angle(
    x: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    axis_order: str = "txy"
) -> jnp.ndarray:
    """
    Calculate the strike angle from dip azimuth.
    
    Strike is perpendicular to dip direction.
    
    Args:
        x: Input seismic data
        dx: Spatial sampling in x-direction
        dy: Spatial sampling in y-direction
        axis_order: Order of axes
        
    Returns:
        Strike angle in radians
    """
    dip_az = dip_azimuth(x, dx, dy, axis_order)
    return dip_az + jnp.pi/2


@jax.jit
def true_dip(
    x: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    dt: float = 1.0,
    axis_order: str = "txy"
) -> jnp.ndarray:
    """
    Calculate true dip angle from seismic data.
    
    Args:
        x: Input seismic data
        dx: Spatial sampling in x-direction
        dy: Spatial sampling in y-direction
        dt: Time sampling
        axis_order: Order of axes
        
    Returns:
        True dip angle in radians
    """
    if x.ndim == 2:
        # 2D case - assume first axis is time
        grad_t = _jax_gradient(x, axis=0) / dt
        grad_x = _jax_gradient(x, axis=1) / dx
        
        # True dip
        return jnp.arctan(grad_x / (grad_t + 1e-10))
    else:
        # 3D case
        if axis_order == "txy":
            grad_t = _jax_gradient(x, axis=0) / dt
            grad_x = _jax_gradient(x, axis=2) / dx
            grad_y = _jax_gradient(x, axis=1) / dy
        elif axis_order == "tyx":
            grad_t = _jax_gradient(x, axis=1) / dt
            grad_x = _jax_gradient(x, axis=2) / dx
            grad_y = _jax_gradient(x, axis=0) / dy
        else:
            grad_t = _jax_gradient(x, axis=2) / dt
            grad_x = _jax_gradient(x, axis=1) / dx
            grad_y = _jax_gradient(x, axis=0) / dy
        
        # True dip magnitude
        spatial_grad = jnp.sqrt(grad_x**2 + grad_y**2)
        return jnp.arctan(spatial_grad / (grad_t + 1e-10))


@jax.jit
def apparent_dip(
    x: jnp.ndarray,
    direction: str = "x",
    dx: float = 1.0,
    dy: float = 1.0,
    axis_order: str = "txy"
) -> jnp.ndarray:
    """
    Calculate apparent dip in a specific direction.
    
    Args:
        x: Input seismic data
        direction: Direction for apparent dip ("x" or "y")
        dx: Spatial sampling in x-direction
        dy: Spatial sampling in y-direction
        axis_order: Order of axes
        
    Returns:
        Apparent dip in specified direction
    """
    if direction == "x":
        if axis_order == "txy":
            grad_dir = _jax_gradient(x, axis=2) / dx
        elif axis_order == "tyx":
            grad_dir = _jax_gradient(x, axis=2) / dx
        else:
            grad_dir = _jax_gradient(x, axis=1) / dx
    else:  # direction == "y"
        if axis_order == "txy":
            grad_dir = _jax_gradient(x, axis=1) / dy
        elif axis_order == "tyx":
            grad_dir = _jax_gradient(x, axis=0) / dy
        else:
            grad_dir = _jax_gradient(x, axis=0) / dy
    
    return grad_dir


@jax.jit
def reflection_intensity(
    x: jnp.ndarray,
    axis: int = -1
) -> jnp.ndarray:
    """
    Calculate reflection intensity using the envelope of the analytic signal.
    
    Args:
        x: Input seismic data
        axis: Axis along which to calculate (typically time axis)
        
    Returns:
        Reflection intensity
    """
    # Calculate envelope (from complex trace attributes)
    from .attributes import envelope
    return envelope(x, axis=axis)


@jax.jit
def relative_acoustic_impedance(
    x: jnp.ndarray,
    axis: int = -1
) -> jnp.ndarray:
    """
    Calculate relative acoustic impedance using integration.
    
    Args:
        x: Input seismic data (assumed to be reflectivity)
        axis: Axis along which to integrate (typically time axis)
        
    Returns:
        Relative acoustic impedance
    """
    # Simple integration approximation
    return jnp.cumsum(x, axis=axis)


@jax.jit
def convergence(
    x: jnp.ndarray,
    window_size: int = 5,
    axis_order: str = "txy"
) -> jnp.ndarray:
    """
    Calculate convergence attribute (opposite of divergence).
    
    Args:
        x: Input seismic data
        window_size: Size of analysis window
        axis_order: Order of axes
        
    Returns:
        Convergence attribute
    """
    if x.ndim == 2:
        # 2D case
        dip_x = _jax_gradient(x, axis=1)
        dip_y = _jax_gradient(x, axis=0)
    else:
        # 3D case
        if axis_order == "txy":
            dip_x = _jax_gradient(x, axis=2)
            dip_y = _jax_gradient(x, axis=1)
        else:
            dip_x = _jax_gradient(x, axis=1)
            dip_y = _jax_gradient(x, axis=0)
    
    # Calculate divergence (negative convergence)
    divergence = _jax_gradient(dip_x, axis=-1) + _jax_gradient(dip_y, axis=-2)
    
    # Return negative divergence (convergence)
    return -divergence


@jax.jit
def parallelism(
    x: jnp.ndarray,
    window_size: int = 5,
    axis_order: str = "txy"
) -> jnp.ndarray:
    """
    Calculate parallelism of seismic events.
    
    Args:
        x: Input seismic data
        window_size: Size of analysis window
        axis_order: Order of axes
        
    Returns:
        Parallelism measure
    """
    # Calculate local dip
    dip_mag = dip_magnitude(x, axis_order=axis_order)
    
    # Calculate variance of dip in local window
    # Approximate with a simple smoothing operation
    kernel = jnp.ones((window_size, window_size)) / (window_size * window_size)
    
    if x.ndim == 2:
        # 2D case
        mean_dip = jnp.convolve(dip_mag.flatten(), kernel.flatten(), mode='same')
        mean_dip = mean_dip.reshape(dip_mag.shape)
        variance = jnp.convolve((dip_mag - mean_dip).flatten()**2, kernel.flatten(), mode='same')
        variance = variance.reshape(dip_mag.shape)
    else:
        # 3D case - use 2D convolution on each time slice
        mean_dip = jnp.zeros_like(dip_mag)
        variance = jnp.zeros_like(dip_mag)
        
        for i in range(dip_mag.shape[0]):
            slice_data = dip_mag[i, :, :]
            mean_slice = jnp.convolve(slice_data.flatten(), kernel.flatten(), mode='same')
            mean_dip = mean_dip.at[i, :, :].set(mean_slice.reshape(slice_data.shape))
            
            var_slice = jnp.convolve((slice_data - mean_slice.reshape(slice_data.shape))**2, 
                                   kernel.flatten(), mode='same')
            variance = variance.at[i, :, :].set(var_slice.reshape(slice_data.shape))
    
    # Parallelism is inverse of variance
    return 1.0 / (variance + 1e-10)


@jax.jit  
def continuity(
    x: jnp.ndarray,
    window_size: int = 5,
    axis_order: str = "txy"
) -> jnp.ndarray:
    """
    Calculate continuity of seismic events.
    
    Args:
        x: Input seismic data
        window_size: Size of analysis window
        axis_order: Order of axes
        
    Returns:
        Continuity measure
    """
    # Apply a small smoothing window
    if x.ndim == 2:
        # Create 2D smoothing kernel
        kernel = jnp.ones((window_size, window_size)) / (window_size * window_size)
        
        # Pad the data for convolution
        padded_x = jnp.pad(x, ((window_size//2, window_size//2), (window_size//2, window_size//2)), mode='edge')
        
        # Apply convolution (simplified)
        smoothed = jnp.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                window = padded_x[i:i+window_size, j:j+window_size]
                smoothed = smoothed.at[i, j].set(jnp.mean(window))
    else:
        # 3D case
        smoothed = jnp.zeros_like(x)
        for t in range(x.shape[0]):
            slice_data = x[t, :, :]
            padded_slice = jnp.pad(slice_data, ((window_size//2, window_size//2), 
                                              (window_size//2, window_size//2)), mode='edge')
            
            for i in range(slice_data.shape[0]):
                for j in range(slice_data.shape[1]):
                    window = padded_slice[i:i+window_size, j:j+window_size]
                    smoothed = smoothed.at[t, i, j].set(jnp.mean(window))
    
    # Compute gradients in spatial directions
    if x.ndim == 2:
        grad_y = _jax_gradient(smoothed, axis=1)
        grad_x = _jax_gradient(smoothed, axis=0)
    else:
        grad_y = _jax_gradient(smoothed, axis=1)
        grad_x = _jax_gradient(smoothed, axis=2)
    
    # Continuity as inverse of gradient magnitude
    grad_magnitude = jnp.sqrt(grad_x**2 + grad_y**2)
    return 1.0 / (grad_magnitude + 1e-10)


@jax.jit
def fault_likelihood(
    x: jnp.ndarray,
    axis_order: str = "txy"
) -> jnp.ndarray:
    """
    Calculate fault likelihood using eigenvalue analysis.
    
    Args:
        x: Input seismic data
        axis_order: Order of axes
        
    Returns:
        Fault likelihood
    """
    # Compute gradients
    if x.ndim == 2:
        grad_z = _jax_gradient(x, axis=0)
        grad_y = _jax_gradient(x, axis=1)
        grad_x = jnp.zeros_like(grad_z)  # No x-gradient in 2D
    else:
        grad_z = _jax_gradient(x, axis=0)
        grad_y = _jax_gradient(x, axis=1)
        grad_x = _jax_gradient(x, axis=2)
    
    # Compute gradient magnitude
    grad_magnitude = jnp.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    # Simple fault likelihood approximation
    # High gradient magnitude indicates potential faults
    return grad_magnitude / (jnp.max(grad_magnitude) + 1e-10)


@jax.jit
def edge_detection(
    x: jnp.ndarray,
    axis_order: str = "txy"
) -> jnp.ndarray:
    """
    Detect edges in seismic data using gradient-based methods.
    
    Args:
        x: Input seismic data
        axis_order: Order of axes
        
    Returns:
        Edge detection result
    """
    if x.ndim == 2:
        # 2D edge detection
        grad_x = _jax_gradient(x, axis=1)
        grad_y = _jax_gradient(x, axis=0)
    else:
        # 3D edge detection
        if axis_order == "txy":
            grad_x = _jax_gradient(x, axis=2)
            grad_y = _jax_gradient(x, axis=1)
            grad_z = _jax_gradient(x, axis=0)
        else:
            grad_x = _jax_gradient(x, axis=2)
            grad_y = _jax_gradient(x, axis=1)
            grad_z = _jax_gradient(x, axis=0)
    
    # Compute gradient magnitude
    if x.ndim == 2:
        return jnp.sqrt(grad_x**2 + grad_y**2)
    else:
        # For 3D, use Laplacian for better edge detection
        laplace_x = _jax_gradient(_jax_gradient(x, axis=2), axis=2)
        laplace_y = _jax_gradient(_jax_gradient(x, axis=1), axis=1)
        laplace_z = _jax_gradient(_jax_gradient(x, axis=0), axis=0)
        
        return jnp.abs(laplace_x + laplace_y + laplace_z)


@jax.jit
def texture_energy(
    x: jnp.ndarray,
    window_size: int = 5
) -> jnp.ndarray:
    """
    Calculate texture energy of seismic data.
    
    Args:
        x: Input seismic data
        window_size: Size of analysis window
        
    Returns:
        Texture energy
    """
    # Calculate local variance as a texture measure
    # This is a simplified version
    kernel = jnp.ones((window_size, window_size)) / (window_size * window_size)
    
    if x.ndim == 2:
        # 2D case
        mean_val = jnp.convolve(x.flatten(), kernel.flatten(), mode='same')
        mean_val = mean_val.reshape(x.shape)
        variance = jnp.convolve((x - mean_val)**2, kernel.flatten(), mode='same')
        return variance.reshape(x.shape)
    else:
        # 3D case - process each time slice
        result = jnp.zeros_like(x)
        for t in range(x.shape[0]):
            slice_data = x[t, :, :]
            mean_val = jnp.convolve(slice_data.flatten(), kernel.flatten(), mode='same')
            mean_val = mean_val.reshape(slice_data.shape)
            variance = jnp.convolve((slice_data - mean_val)**2, kernel.flatten(), mode='same')
            result = result.at[t, :, :].set(variance.reshape(slice_data.shape))
        
        return result


@jax.jit
def local_structural_entropy(
    x: jnp.ndarray,
    window_size: int = 5
) -> jnp.ndarray:
    """
    Calculate local structural entropy of seismic data.
    
    Args:
        x: Input seismic data
        window_size: Size of analysis window
        
    Returns:
        Local structural entropy
    """
    # Calculate local variance as a proxy for structural complexity
    # This is a simplified implementation
    return texture_energy(x, window_size) 