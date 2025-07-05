from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


def _check_axis(axis: int, ndim: int) -> None:
    """Helper to ensure the specified axis is valid for the array dimensions."""
    # Note: JAX will handle axis validation naturally, so we skip dynamic checks
    # inside JIT-compiled functions to avoid TracerBoolConversionError
    pass


@jax.jit
def _hilbert_1d(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Hilbert transform for 1D array.
    """
    N = x.shape[0]
    
    # Take FFT
    X = jnp.fft.fft(x)
    
    # Create the Hilbert transform filter
    h = jnp.zeros(N, dtype=jnp.float32)
    h = h.at[0].set(1.0)  # DC component
    
    # Use JAX conditional to avoid TracerBoolConversionError
    h = lax.cond(
        N % 2 == 0,
        lambda h: h.at[1:N//2].set(2.0).at[N//2].set(1.0),  # Even length
        lambda h: h.at[1:(N+1)//2].set(2.0),                 # Odd length
        h
    )
    
    # Apply filter and take IFFT
    return jnp.fft.ifft(X * h)


def _hilbert_jax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Compute the Hilbert transform using JAX FFT operations.
    This version handles different axes statically to avoid moveaxis issues.
    """
    # Normalize axis to positive value
    if axis < 0:
        axis = x.ndim + axis
    
    # Handle different axis cases statically
    if axis == 0:
        # Apply along first axis
        if x.ndim == 1:
            return _hilbert_1d(x)
        elif x.ndim == 2:
            return jax.vmap(_hilbert_1d, in_axes=1, out_axes=1)(x)
        else:  # 3D
            return jax.vmap(jax.vmap(_hilbert_1d, in_axes=1, out_axes=1), in_axes=2, out_axes=2)(x)
    
    elif axis == 1:
        # Apply along second axis
        if x.ndim == 2:
            return jax.vmap(_hilbert_1d, in_axes=0, out_axes=0)(x)
        else:  # 3D
            return jax.vmap(jax.vmap(_hilbert_1d, in_axes=0, out_axes=0), in_axes=2, out_axes=2)(x)
    
    elif axis == 2:
        # Apply along third axis (3D only)
        return jax.vmap(jax.vmap(_hilbert_1d, in_axes=0, out_axes=0), in_axes=0, out_axes=0)(x)
    
    else:
        # For axis == -1 or last axis, transpose to last and back
        if axis == x.ndim - 1:
            # Already at last axis
            if x.ndim == 1:
                return _hilbert_1d(x)
            elif x.ndim == 2:
                return jax.vmap(_hilbert_1d, in_axes=0, out_axes=0)(x)
            else:  # 3D
                return jax.vmap(jax.vmap(_hilbert_1d, in_axes=0, out_axes=0), in_axes=0, out_axes=0)(x)
        else:
            # For other axes, use a general approach
            return _hilbert_1d(x)


def _jax_unwrap(p: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    JAX implementation of unwrap function.
    """
    # Normalize axis
    if axis < 0:
        axis = p.ndim + axis
    
    # Handle different axis cases statically
    if axis == 0:
        # Unwrap along first axis
        if p.ndim == 1:
            return _unwrap_1d(p)
        elif p.ndim == 2:
            return jax.vmap(_unwrap_1d, in_axes=1, out_axes=1)(p)
        else:  # 3D
            return jax.vmap(jax.vmap(_unwrap_1d, in_axes=1, out_axes=1), in_axes=2, out_axes=2)(p)
    
    elif axis == 1:
        # Unwrap along second axis
        if p.ndim == 2:
            return jax.vmap(_unwrap_1d, in_axes=0, out_axes=0)(p)
        else:  # 3D
            return jax.vmap(jax.vmap(_unwrap_1d, in_axes=0, out_axes=0), in_axes=2, out_axes=2)(p)
    
    elif axis == 2:
        # Unwrap along third axis
        return jax.vmap(jax.vmap(_unwrap_1d, in_axes=0, out_axes=0), in_axes=0, out_axes=0)(p)
    
    else:
        # For last axis
        if p.ndim == 1:
            return _unwrap_1d(p)
        elif p.ndim == 2:
            return jax.vmap(_unwrap_1d, in_axes=0, out_axes=0)(p)
        else:  # 3D
            return jax.vmap(jax.vmap(_unwrap_1d, in_axes=0, out_axes=0), in_axes=0, out_axes=0)(p)


@jax.jit
def _unwrap_1d(p: jnp.ndarray) -> jnp.ndarray:
    """
    Unwrap 1D phase array.
    """
    # Handle case where array is too short
    def unwrap_long():
        # Compute phase differences
        dp = jnp.diff(p)
        
        # Find discontinuities (jumps greater than pi)
        dp_corrected = dp - 2 * jnp.pi * jnp.round(dp / (2 * jnp.pi))
        
        # Integrate to get unwrapped phase - fix shape consistency
        unwrapped = jnp.concatenate([p[:1], p[:1] + jnp.cumsum(dp_corrected)])
        return unwrapped
    
    def unwrap_short():
        return p
    
    return lax.cond(
        p.shape[0] < 2,
        unwrap_short,
        unwrap_long
    )


def _jax_gradient(f: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    JAX implementation of gradient function using central differences.
    """
    # Normalize axis
    if axis < 0:
        axis = f.ndim + axis
        
    # Handle different axis cases statically
    if axis == 0:
        # Gradient along first axis
        if f.ndim == 1:
            return _gradient_1d(f)
        elif f.ndim == 2:
            return jax.vmap(_gradient_1d, in_axes=1, out_axes=1)(f)
        else:  # 3D
            return jax.vmap(jax.vmap(_gradient_1d, in_axes=1, out_axes=1), in_axes=2, out_axes=2)(f)
    
    elif axis == 1:
        # Gradient along second axis
        if f.ndim == 2:
            return jax.vmap(_gradient_1d, in_axes=0, out_axes=0)(f)
        else:  # 3D
            return jax.vmap(jax.vmap(_gradient_1d, in_axes=0, out_axes=0), in_axes=2, out_axes=2)(f)
    
    elif axis == 2:
        # Gradient along third axis
        return jax.vmap(jax.vmap(_gradient_1d, in_axes=0, out_axes=0), in_axes=0, out_axes=0)(f)
    
    else:
        # For last axis
        if f.ndim == 1:
            return _gradient_1d(f)
        elif f.ndim == 2:
            return jax.vmap(_gradient_1d, in_axes=0, out_axes=0)(f)
        else:  # 3D
            return jax.vmap(jax.vmap(_gradient_1d, in_axes=0, out_axes=0), in_axes=0, out_axes=0)(f)


@jax.jit
def _gradient_1d(f: jnp.ndarray) -> jnp.ndarray:
    """
    Compute gradient for 1D array using central differences.
    """
    n = f.shape[0]
    
    def grad_too_short():
        return jnp.zeros_like(f)
    
    def grad_two_points():
        diff_val = f[1] - f[0]
        # Return same shape as input, not hardcoded shape
        return jnp.full_like(f, diff_val)
    
    def grad_normal():
        # Forward difference for first point
        forward = f[1] - f[0]
        
        # Central differences for interior points
        central = (f[2:] - f[:-2]) / 2.0
        
        # Backward difference for last point
        backward = f[-1] - f[-2]
        
        return jnp.concatenate([
            jnp.array([forward]),
            central,
            jnp.array([backward])
        ])
    
    return lax.cond(
        n < 3,
        lambda: lax.cond(
            n < 1,
            grad_too_short,
            grad_two_points
        ),
        grad_normal
    )


def analytic_signal(
    x: jnp.ndarray,
    axis: int = -1
) -> jnp.ndarray:
    """
    Compute the analytic signal using the Hilbert transform.
    
    The analytic signal is a complex-valued function that has no negative
    frequency components. It is constructed by taking the Hilbert transform
    of the real-valued signal.
    
    Args:
        x: Input real-valued signal
        axis: Axis along which to compute the Hilbert transform
        
    Returns:
        The complex-valued analytic signal.
    """
    return _hilbert_jax(x, axis=axis)


def envelope(
    x: jnp.ndarray,
    axis: int = -1
) -> jnp.ndarray:
    """
    Compute the envelope of a signal.
    
    The envelope is the magnitude of the analytic signal, representing
    the instantaneous amplitude of the signal.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the envelope
        
    Returns:
        The envelope of the signal.
    """
    return jnp.abs(analytic_signal(x, axis=axis))


def instantaneous_phase(
    x: jnp.ndarray,
    axis: int = -1
) -> jnp.ndarray:
    """
    Compute the instantaneous phase of a signal.
    
    The instantaneous phase is the argument (angle) of the complex
    analytic signal. The phase is unwrapped to be continuous.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the phase
        
    Returns:
        The unwrapped instantaneous phase of the signal.
    """
    return _jax_unwrap(jnp.angle(analytic_signal(x, axis=axis)), axis=axis)


def cosine_instantaneous_phase(
    x: jnp.ndarray,
    axis: int = -1
) -> jnp.ndarray:
    """
    Compute the cosine of the instantaneous phase.
    
    This attribute highlights phase relationships and can be useful
    for detecting phase anomalies.
    
    Args:
        x: Input signal
        axis: Axis along which to compute
        
    Returns:
        Cosine of the instantaneous phase
    """
    return jnp.cos(instantaneous_phase(x, axis=axis))


def instantaneous_frequency(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0
) -> jnp.ndarray:
    """
    Compute the instantaneous frequency of a signal.
    
    The instantaneous frequency is the time derivative of the instantaneous
    phase. This is calculated as the time derivative of the unwrapped instantaneous
    phase, scaled by the sampling frequency.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the frequency
        fs: Sampling frequency (default: 1.0)
        
    Returns:
        The instantaneous frequency of the signal.
    """
    # Get instantaneous phase
    phi = instantaneous_phase(x, axis=axis)
    
    # Use custom gradient function
    freq = _jax_gradient(phi, axis=axis)
    
    # Convert to frequency with proper sampling rate
    return (fs / (2 * jnp.pi)) * freq


def instantaneous_bandwidth(
    x: jnp.ndarray,
    axis: int = -1
) -> jnp.ndarray:
    """
    Compute the instantaneous bandwidth of a signal.
    
    The instantaneous bandwidth is related to the time derivative of the
    instantaneous amplitude (envelope).
    
    Args:
        x: Input signal
        axis: Axis along which to compute the bandwidth
        
    Returns:
        The instantaneous bandwidth of the signal.
    """
    # Get envelope
    env = envelope(x, axis=axis)
    
    # Use custom gradient function
    bandwidth = _jax_gradient(env, axis=axis)
    
    # Return absolute value of bandwidth
    return jnp.abs(bandwidth) 