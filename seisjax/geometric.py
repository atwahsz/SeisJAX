"""
Geometric attributes for seismic structural analysis.

This module provides JAX-accelerated implementations of geometric attributes
used for analyzing seismic structures and geological features.
"""

from typing import Tuple, Union, Optional

import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def dip_magnitude(
    x: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    axis: int = 0
) -> jnp.ndarray:
    """
    Compute the dip magnitude from seismic data.
    
    Dip magnitude represents the steepness of the reflector.
    
    Args:
        x: Input seismic data (3D array)
        dx: Spacing in x direction
        dy: Spacing in y direction
        axis: Time axis (default: 0)
        
    Returns:
        Dip magnitude array
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    # Compute gradients
    if axis == 0:
        grad_x = jnp.gradient(x, dx, axis=2)
        grad_y = jnp.gradient(x, dy, axis=1)
    elif axis == 1:
        grad_x = jnp.gradient(x, dx, axis=2)
        grad_y = jnp.gradient(x, dy, axis=0)
    else:  # axis == 2
        grad_x = jnp.gradient(x, dx, axis=1)
        grad_y = jnp.gradient(x, dy, axis=0)
    
    # Dip magnitude
    dip_mag = jnp.sqrt(grad_x**2 + grad_y**2)
    
    return dip_mag


@jax.jit
def dip_azimuth(
    x: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    axis: int = 0
) -> jnp.ndarray:
    """
    Compute the dip azimuth from seismic data.
    
    Dip azimuth represents the direction of the steepest dip.
    
    Args:
        x: Input seismic data (3D array)
        dx: Spacing in x direction
        dy: Spacing in y direction
        axis: Time axis (default: 0)
        
    Returns:
        Dip azimuth array (in radians)
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    # Compute gradients
    if axis == 0:
        grad_x = jnp.gradient(x, dx, axis=2)
        grad_y = jnp.gradient(x, dy, axis=1)
    elif axis == 1:
        grad_x = jnp.gradient(x, dx, axis=2)
        grad_y = jnp.gradient(x, dy, axis=0)
    else:  # axis == 2
        grad_x = jnp.gradient(x, dx, axis=1)
        grad_y = jnp.gradient(x, dy, axis=0)
    
    # Dip azimuth
    azimuth = jnp.arctan2(grad_y, grad_x)
    
    return azimuth


@jax.jit
def strike_angle(
    x: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    axis: int = 0
) -> jnp.ndarray:
    """
    Compute the strike angle from seismic data.
    
    Strike angle is perpendicular to the dip direction.
    
    Args:
        x: Input seismic data (3D array)
        dx: Spacing in x direction
        dy: Spacing in y direction
        axis: Time axis (default: 0)
        
    Returns:
        Strike angle array (in radians)
    """
    azimuth = dip_azimuth(x, dx, dy, axis)
    strike = azimuth + jnp.pi / 2
    
    # Normalize to [0, 2π]
    strike = jnp.mod(strike, 2 * jnp.pi)
    
    return strike


@jax.jit
def true_dip(
    x: jnp.ndarray,
    dt: float = 1.0,
    dx: float = 1.0,
    dy: float = 1.0,
    axis: int = 0
) -> jnp.ndarray:
    """
    Compute the true dip from seismic data.
    
    True dip accounts for the time-to-depth conversion.
    
    Args:
        x: Input seismic data (3D array)
        dt: Sampling interval in time
        dx: Spacing in x direction
        dy: Spacing in y direction
        axis: Time axis (default: 0)
        
    Returns:
        True dip array
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    # Compute gradients
    if axis == 0:
        grad_t = jnp.gradient(x, dt, axis=0)
        grad_x = jnp.gradient(x, dx, axis=2)
        grad_y = jnp.gradient(x, dy, axis=1)
    elif axis == 1:
        grad_t = jnp.gradient(x, dt, axis=1)
        grad_x = jnp.gradient(x, dx, axis=2)
        grad_y = jnp.gradient(x, dy, axis=0)
    else:  # axis == 2
        grad_t = jnp.gradient(x, dt, axis=2)
        grad_x = jnp.gradient(x, dx, axis=1)
        grad_y = jnp.gradient(x, dy, axis=0)
    
    # True dip calculation
    dip_x = -grad_x / (grad_t + 1e-10)
    dip_y = -grad_y / (grad_t + 1e-10)
    
    true_dip_val = jnp.sqrt(dip_x**2 + dip_y**2)
    
    return true_dip_val


@jax.jit
def apparent_dip(
    x: jnp.ndarray,
    dx: float = 1.0,
    axis: int = 0,
    direction: str = 'inline'
) -> jnp.ndarray:
    """
    Compute the apparent dip in a specific direction.
    
    Args:
        x: Input seismic data (3D array)
        dx: Spacing in the specified direction
        axis: Time axis (default: 0)
        direction: Direction for apparent dip ('inline' or 'crossline')
        
    Returns:
        Apparent dip array
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    # Compute gradient in specified direction
    if direction == 'inline':
        if axis == 0:
            grad_dir = jnp.gradient(x, dx, axis=2)
        elif axis == 1:
            grad_dir = jnp.gradient(x, dx, axis=2)
        else:  # axis == 2
            grad_dir = jnp.gradient(x, dx, axis=1)
    else:  # crossline
        if axis == 0:
            grad_dir = jnp.gradient(x, dx, axis=1)
        elif axis == 1:
            grad_dir = jnp.gradient(x, dx, axis=0)
        else:  # axis == 2
            grad_dir = jnp.gradient(x, dx, axis=0)
    
    return jnp.abs(grad_dir)


@jax.jit
def reflection_intensity(
    x: jnp.ndarray,
    axis: int = -1
) -> jnp.ndarray:
    """
    Compute reflection intensity (envelope of the analytic signal).
    
    Args:
        x: Input seismic data
        axis: Axis along which to compute reflection intensity
        
    Returns:
        Reflection intensity array
    """
    from .attributes import envelope
    return envelope(x, axis=axis)


@jax.jit
def relative_acoustic_impedance(
    x: jnp.ndarray,
    axis: int = -1
) -> jnp.ndarray:
    """
    Compute relative acoustic impedance.
    
    This is computed as the integral of the seismic trace.
    
    Args:
        x: Input seismic data
        axis: Axis along which to compute the impedance
        
    Returns:
        Relative acoustic impedance array
    """
    return jnp.cumsum(x, axis=axis)


@jax.jit
def convergence(
    x: jnp.ndarray,
    window_shape: Tuple[int, int, int] = (5, 5, 5)
) -> jnp.ndarray:
    """
    Compute convergence attribute.
    
    Convergence measures how traces converge or diverge in a local window.
    
    Args:
        x: Input seismic data (3D array)
        window_shape: Shape of the analysis window (nz, ny, nx)
        
    Returns:
        Convergence array
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    nz, ny, nx = window_shape
    pad_width = ((nz//2, nz//2), (ny//2, ny//2), (nx//2, nx//2))
    x_padded = jnp.pad(x, pad_width, mode='reflect')
    
    def compute_convergence(center_idx):
        i, j, k = center_idx
        
        # Extract local window
        window = x_padded[i:i+nz, j:j+ny, k:k+nx]
        
        # Central trace
        center_trace = window[nz//2, ny//2, :]
        
        # Compute correlation with surrounding traces
        correlations = []
        for ii in range(ny):
            for jj in range(nx):
                if ii == ny//2 and jj == nx//2:
                    continue
                trace = window[nz//2, ii, jj]
                corr = jnp.corrcoef(center_trace, trace)[0, 1]
                correlations.append(corr)
        
        # Convergence as mean correlation
        convergence_val = jnp.mean(jnp.array(correlations))
        
        return convergence_val
    
    # Vectorized computation
    indices = jnp.mgrid[0:x.shape[0], 0:x.shape[1], 0:x.shape[2]]
    indices = indices.reshape(3, -1).T
    
    convergence_values = jax.vmap(compute_convergence)(indices)
    convergence_volume = convergence_values.reshape(x.shape)
    
    return convergence_volume


@jax.jit
def parallelism(
    x: jnp.ndarray,
    window_shape: Tuple[int, int, int] = (5, 5, 5)
) -> jnp.ndarray:
    """
    Compute parallelism attribute.
    
    Parallelism measures how parallel the reflectors are in a local window.
    
    Args:
        x: Input seismic data (3D array)
        window_shape: Shape of the analysis window (nz, ny, nx)
        
    Returns:
        Parallelism array
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    # Compute dip in different directions
    dip_x = jnp.gradient(x, axis=2)
    dip_y = jnp.gradient(x, axis=1)
    
    nz, ny, nx = window_shape
    pad_width = ((nz//2, nz//2), (ny//2, ny//2), (nx//2, nx//2))
    
    dip_x_padded = jnp.pad(dip_x, pad_width, mode='reflect')
    dip_y_padded = jnp.pad(dip_y, pad_width, mode='reflect')
    
    def compute_parallelism(center_idx):
        i, j, k = center_idx
        
        # Extract local windows
        window_x = dip_x_padded[i:i+nz, j:j+ny, k:k+nx]
        window_y = dip_y_padded[i:i+nz, j:j+ny, k:k+nx]
        
        # Compute variance of dips in window
        var_x = jnp.var(window_x)
        var_y = jnp.var(window_y)
        
        # Parallelism as inverse of variance
        parallelism_val = 1.0 / (1.0 + var_x + var_y)
        
        return parallelism_val
    
    # Vectorized computation
    indices = jnp.mgrid[0:x.shape[0], 0:x.shape[1], 0:x.shape[2]]
    indices = indices.reshape(3, -1).T
    
    parallelism_values = jax.vmap(compute_parallelism)(indices)
    parallelism_volume = parallelism_values.reshape(x.shape)
    
    return parallelism_volume


@jax.jit
def continuity(
    x: jnp.ndarray,
    window_shape: Tuple[int, int, int] = (5, 5, 5)
) -> jnp.ndarray:
    """
    Compute continuity attribute.
    
    Continuity measures how continuous the reflectors are in a local window.
    
    Args:
        x: Input seismic data (3D array)
        window_shape: Shape of the analysis window (nz, ny, nx)
        
    Returns:
        Continuity array
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    nz, ny, nx = window_shape
    pad_width = ((nz//2, nz//2), (ny//2, ny//2), (nx//2, nx//2))
    x_padded = jnp.pad(x, pad_width, mode='reflect')
    
    def compute_continuity(center_idx):
        i, j, k = center_idx
        
        # Extract local window
        window = x_padded[i:i+nz, j:j+ny, k:k+nx]
        
        # Compute gradients in spatial directions
        grad_y = jnp.gradient(window, axis=1)
        grad_x = jnp.gradient(window, axis=2)
        
        # Continuity as inverse of gradient magnitude
        grad_mag = jnp.sqrt(grad_x**2 + grad_y**2)
        continuity_val = jnp.mean(1.0 / (1.0 + grad_mag))
        
        return continuity_val
    
    # Vectorized computation
    indices = jnp.mgrid[0:x.shape[0], 0:x.shape[1], 0:x.shape[2]]
    indices = indices.reshape(3, -1).T
    
    continuity_values = jax.vmap(compute_continuity)(indices)
    continuity_volume = continuity_values.reshape(x.shape)
    
    return continuity_volume


@jax.jit
def fault_likelihood(
    x: jnp.ndarray,
    sigma: float = 1.0
) -> jnp.ndarray:
    """
    Compute fault likelihood attribute.
    
    Fault likelihood estimates the probability of fault presence
    based on discontinuities in the seismic data.
    
    Args:
        x: Input seismic data (3D array)
        sigma: Standard deviation for Gaussian smoothing
        
    Returns:
        Fault likelihood array
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    # Compute gradients
    grad_z = jnp.gradient(x, axis=0)
    grad_y = jnp.gradient(x, axis=1)
    grad_x = jnp.gradient(x, axis=2)
    
    # Compute gradient magnitude
    grad_mag = jnp.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    # Apply Gaussian smoothing
    from jax.scipy.ndimage import gaussian_filter
    grad_mag_smooth = gaussian_filter(grad_mag, sigma=sigma)
    
    # Normalize to [0, 1]
    fault_likelihood_val = grad_mag_smooth / (jnp.max(grad_mag_smooth) + 1e-10)
    
    return fault_likelihood_val


@jax.jit
def edge_detection(
    x: jnp.ndarray,
    method: str = 'sobel'
) -> jnp.ndarray:
    """
    Compute edge detection attribute.
    
    Args:
        x: Input seismic data (2D or 3D array)
        method: Edge detection method ('sobel', 'prewitt', 'laplacian')
        
    Returns:
        Edge detection array
    """
    if method == 'sobel':
        if x.ndim == 2:
            # 2D Sobel operator
            grad_x = jnp.gradient(x, axis=1)
            grad_y = jnp.gradient(x, axis=0)
            edges = jnp.sqrt(grad_x**2 + grad_y**2)
        else:  # 3D
            grad_x = jnp.gradient(x, axis=2)
            grad_y = jnp.gradient(x, axis=1)
            grad_z = jnp.gradient(x, axis=0)
            edges = jnp.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    elif method == 'prewitt':
        # Similar to Sobel but with different kernel weights
        if x.ndim == 2:
            grad_x = jnp.gradient(x, axis=1)
            grad_y = jnp.gradient(x, axis=0)
            edges = jnp.sqrt(grad_x**2 + grad_y**2)
        else:  # 3D
            grad_x = jnp.gradient(x, axis=2)
            grad_y = jnp.gradient(x, axis=1)
            grad_z = jnp.gradient(x, axis=0)
            edges = jnp.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    elif method == 'laplacian':
        # Laplacian operator (second derivatives)
        if x.ndim == 2:
            laplace_x = jnp.gradient(jnp.gradient(x, axis=1), axis=1)
            laplace_y = jnp.gradient(jnp.gradient(x, axis=0), axis=0)
            edges = jnp.abs(laplace_x + laplace_y)
        else:  # 3D
            laplace_x = jnp.gradient(jnp.gradient(x, axis=2), axis=2)
            laplace_y = jnp.gradient(jnp.gradient(x, axis=1), axis=1)
            laplace_z = jnp.gradient(jnp.gradient(x, axis=0), axis=0)
            edges = jnp.abs(laplace_x + laplace_y + laplace_z)
    
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    
    return edges


@jax.jit
def texture_energy(
    x: jnp.ndarray,
    window_shape: Tuple[int, int, int] = (5, 5, 5)
) -> jnp.ndarray:
    """
    Compute texture energy attribute.
    
    Texture energy measures the local energy content of the seismic data.
    
    Args:
        x: Input seismic data (3D array)
        window_shape: Shape of the analysis window (nz, ny, nx)
        
    Returns:
        Texture energy array
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    nz, ny, nx = window_shape
    pad_width = ((nz//2, nz//2), (ny//2, ny//2), (nx//2, nx//2))
    x_padded = jnp.pad(x, pad_width, mode='reflect')
    
    def compute_texture_energy(center_idx):
        i, j, k = center_idx
        
        # Extract local window
        window = x_padded[i:i+nz, j:j+ny, k:k+nx]
        
        # Compute energy as sum of squared amplitudes
        energy = jnp.sum(window**2)
        
        return energy
    
    # Vectorized computation
    indices = jnp.mgrid[0:x.shape[0], 0:x.shape[1], 0:x.shape[2]]
    indices = indices.reshape(3, -1).T
    
    energy_values = jax.vmap(compute_texture_energy)(indices)
    energy_volume = energy_values.reshape(x.shape)
    
    return energy_volume


@jax.jit
def local_structural_entropy(
    x: jnp.ndarray,
    window_shape: Tuple[int, int, int] = (5, 5, 5)
) -> jnp.ndarray:
    """
    Compute local structural entropy attribute.
    
    Structural entropy measures the complexity of the local structure.
    
    Args:
        x: Input seismic data (3D array)
        window_shape: Shape of the analysis window (nz, ny, nx)
        
    Returns:
        Local structural entropy array
    """
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    nz, ny, nx = window_shape
    pad_width = ((nz//2, nz//2), (ny//2, ny//2), (nx//2, nx//2))
    x_padded = jnp.pad(x, pad_width, mode='reflect')
    
    def compute_entropy(center_idx):
        i, j, k = center_idx
        
        # Extract local window
        window = x_padded[i:i+nz, j:j+ny, k:k+nx]
        
        # Compute histogram
        hist, _ = jnp.histogram(window.flatten(), bins=32, density=True)
        
        # Add small epsilon to avoid log(0)
        hist = hist + 1e-10
        
        # Compute entropy
        entropy = -jnp.sum(hist * jnp.log(hist))
        
        return entropy
    
    # Vectorized computation
    indices = jnp.mgrid[0:x.shape[0], 0:x.shape[1], 0:x.shape[2]]
    indices = indices.reshape(3, -1).T
    
    entropy_values = jax.vmap(compute_entropy)(indices)
    entropy_volume = entropy_values.reshape(x.shape)
    
    return entropy_volume 