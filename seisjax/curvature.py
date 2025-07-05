"""
Curvature attributes for seismic horizon analysis.

This module provides JAX-accelerated implementations of curvature attributes
used for analyzing seismic horizons and structural features.
"""

from typing import Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax


def _compute_surface_derivatives(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute first and second derivatives of a surface for curvature calculation.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Tuple of (dz_dx, dz_dy, d2z_dx2, d2z_dy2, d2z_dxdy)
    """
    # First derivatives
    dz_dx = jnp.gradient(surface, dx, axis=1)
    dz_dy = jnp.gradient(surface, dy, axis=0)
    
    # Second derivatives
    d2z_dx2 = jnp.gradient(dz_dx, dx, axis=1)
    d2z_dy2 = jnp.gradient(dz_dy, dy, axis=0)
    d2z_dxdy = jnp.gradient(dz_dx, dy, axis=0)
    
    return dz_dx, dz_dy, d2z_dx2, d2z_dy2, d2z_dxdy


@jax.jit
def mean_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute mean curvature of a surface.
    
    Mean curvature is the average of the principal curvatures and provides
    a measure of the overall bending of the surface.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Mean curvature array
    """
    if surface.ndim != 2:
        raise ValueError("Surface must be a 2D array")
    
    # Compute derivatives
    dz_dx, dz_dy, d2z_dx2, d2z_dy2, d2z_dxdy = _compute_surface_derivatives(surface, dx, dy)
    
    # Mean curvature formula
    numerator = (1 + dz_dy**2) * d2z_dx2 - 2 * dz_dx * dz_dy * d2z_dxdy + (1 + dz_dx**2) * d2z_dy2
    denominator = 2 * (1 + dz_dx**2 + dz_dy**2)**(3/2)
    
    mean_curv = numerator / (denominator + 1e-10)
    
    return mean_curv


@jax.jit
def gaussian_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute Gaussian curvature of a surface.
    
    Gaussian curvature is the product of the principal curvatures and
    characterizes the intrinsic curvature of the surface.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Gaussian curvature array
    """
    if surface.ndim != 2:
        raise ValueError("Surface must be a 2D array")
    
    # Compute derivatives
    dz_dx, dz_dy, d2z_dx2, d2z_dy2, d2z_dxdy = _compute_surface_derivatives(surface, dx, dy)
    
    # Gaussian curvature formula
    numerator = d2z_dx2 * d2z_dy2 - d2z_dxdy**2
    denominator = (1 + dz_dx**2 + dz_dy**2)**2
    
    gaussian_curv = numerator / (denominator + 1e-10)
    
    return gaussian_curv


@jax.jit
def principal_curvatures(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute principal curvatures of a surface.
    
    Principal curvatures are the maximum and minimum curvatures at each point.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Tuple of (maximum_curvature, minimum_curvature)
    """
    if surface.ndim != 2:
        raise ValueError("Surface must be a 2D array")
    
    # Compute mean and Gaussian curvatures
    H = mean_curvature(surface, dx, dy)
    K = gaussian_curvature(surface, dx, dy)
    
    # Principal curvatures from mean and Gaussian curvatures
    discriminant = H**2 - K
    discriminant = jnp.where(discriminant >= 0, discriminant, 0)
    
    k1 = H + jnp.sqrt(discriminant)  # Maximum curvature
    k2 = H - jnp.sqrt(discriminant)  # Minimum curvature
    
    return k1, k2


@jax.jit
def maximum_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute maximum principal curvature of a surface.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Maximum curvature array
    """
    k1, k2 = principal_curvatures(surface, dx, dy)
    return k1


@jax.jit
def minimum_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute minimum principal curvature of a surface.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Minimum curvature array
    """
    k1, k2 = principal_curvatures(surface, dx, dy)
    return k2


@jax.jit
def curvedness(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute curvedness of a surface.
    
    Curvedness is a measure of how much a surface deviates from being flat,
    regardless of the shape type.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Curvedness array
    """
    k1, k2 = principal_curvatures(surface, dx, dy)
    curvedness_val = jnp.sqrt((k1**2 + k2**2) / 2)
    return curvedness_val


@jax.jit
def shape_index(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute shape index of a surface.
    
    Shape index characterizes the shape of the surface at each point,
    ranging from -1 (spherical cup) to +1 (spherical cap).
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Shape index array
    """
    k1, k2 = principal_curvatures(surface, dx, dy)
    
    # Avoid division by zero
    denominator = k1 - k2
    shape_idx = jnp.where(
        jnp.abs(denominator) > 1e-10,
        -2.0 / jnp.pi * jnp.arctan((k1 + k2) / denominator),
        0.0
    )
    
    return shape_idx


@jax.jit
def dip_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute dip curvature of a surface.
    
    Dip curvature measures the curvature in the direction of maximum dip.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Dip curvature array
    """
    if surface.ndim != 2:
        raise ValueError("Surface must be a 2D array")
    
    # Compute derivatives
    dz_dx, dz_dy, d2z_dx2, d2z_dy2, d2z_dxdy = _compute_surface_derivatives(surface, dx, dy)
    
    # Dip direction (gradient magnitude)
    grad_mag = jnp.sqrt(dz_dx**2 + dz_dy**2)
    
    # Dip curvature formula
    numerator = dz_dx**2 * d2z_dx2 + 2 * dz_dx * dz_dy * d2z_dxdy + dz_dy**2 * d2z_dy2
    denominator = (grad_mag**2) * (1 + grad_mag**2)**(1/2)
    
    dip_curv = jnp.where(
        grad_mag > 1e-10,
        numerator / denominator,
        0.0
    )
    
    return dip_curv


@jax.jit
def strike_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute strike curvature of a surface.
    
    Strike curvature measures the curvature perpendicular to the direction
    of maximum dip.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Strike curvature array
    """
    if surface.ndim != 2:
        raise ValueError("Surface must be a 2D array")
    
    # Compute derivatives
    dz_dx, dz_dy, d2z_dx2, d2z_dy2, d2z_dxdy = _compute_surface_derivatives(surface, dx, dy)
    
    # Strike curvature formula
    numerator = dz_dy**2 * d2z_dx2 - 2 * dz_dx * dz_dy * d2z_dxdy + dz_dx**2 * d2z_dy2
    denominator = (dz_dx**2 + dz_dy**2) * (1 + dz_dx**2 + dz_dy**2)**(1/2)
    
    strike_curv = jnp.where(
        dz_dx**2 + dz_dy**2 > 1e-10,
        numerator / denominator,
        0.0
    )
    
    return strike_curv


@jax.jit
def contour_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute contour curvature of a surface.
    
    Contour curvature measures the curvature of contour lines on the surface.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Contour curvature array
    """
    # Contour curvature is essentially the same as strike curvature
    return strike_curvature(surface, dx, dy)


@jax.jit
def profile_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute profile curvature of a surface.
    
    Profile curvature measures the curvature of the surface in the direction
    of steepest descent.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Profile curvature array
    """
    # Profile curvature is essentially the same as dip curvature
    return dip_curvature(surface, dx, dy)


@jax.jit
def planform_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute planform curvature of a surface.
    
    Planform curvature measures the curvature of the surface perpendicular
    to the direction of steepest descent.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Planform curvature array
    """
    # Planform curvature is essentially the same as contour curvature
    return contour_curvature(surface, dx, dy)


@jax.jit
def most_positive_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute most positive curvature of a surface.
    
    This is the maximum of the two principal curvatures.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Most positive curvature array
    """
    k1, k2 = principal_curvatures(surface, dx, dy)
    return jnp.maximum(k1, k2)


@jax.jit
def most_negative_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute most negative curvature of a surface.
    
    This is the minimum of the two principal curvatures.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Most negative curvature array
    """
    k1, k2 = principal_curvatures(surface, dx, dy)
    return jnp.minimum(k1, k2)


@jax.jit
def total_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute total curvature of a surface.
    
    Total curvature is the sum of the absolute values of the principal curvatures.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Total curvature array
    """
    k1, k2 = principal_curvatures(surface, dx, dy)
    return jnp.abs(k1) + jnp.abs(k2)


@jax.jit
def tangential_curvature(
    surface: jnp.ndarray,
    dx: float = 1.0,
    dy: float = 1.0
) -> jnp.ndarray:
    """
    Compute tangential curvature of a surface.
    
    Tangential curvature measures the curvature of the surface in the direction
    tangent to the contour lines.
    
    Args:
        surface: 2D surface (horizon) data
        dx: Spacing in x direction
        dy: Spacing in y direction
        
    Returns:
        Tangential curvature array
    """
    if surface.ndim != 2:
        raise ValueError("Surface must be a 2D array")
    
    # Compute derivatives
    dz_dx, dz_dy, d2z_dx2, d2z_dy2, d2z_dxdy = _compute_surface_derivatives(surface, dx, dy)
    
    # Tangential curvature formula
    numerator = dz_dy**2 * d2z_dx2 - 2 * dz_dx * dz_dy * d2z_dxdy + dz_dx**2 * d2z_dy2
    denominator = (dz_dx**2 + dz_dy**2)**(3/2)
    
    tangential_curv = jnp.where(
        dz_dx**2 + dz_dy**2 > 1e-10,
        numerator / denominator,
        0.0
    )
    
    return tangential_curv 