from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


def _check_axis(axis: int, ndim: int) -> None:
    """Helper to ensure the specified axis is valid for the array dimensions."""
    # Note: JAX will handle axis validation naturally, so we skip dynamic checks
    # inside JIT-compiled functions to avoid TracerBoolConversionError
    pass


def _jax_unwrap(p: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    JAX implementation of unwrap function.
    """
    # Normalize axis
    if axis < 0:
        axis = p.ndim + axis
    
    # Move the specified axis to the last position for easier processing
    if axis != p.ndim - 1:
        # Create permutation that moves axis to the end
        axes = list(range(p.ndim))
        axes[axis], axes[-1] = axes[-1], axes[axis]
        p = jnp.transpose(p, axes)
        
    # Compute differences along the last axis
    diff = jnp.diff(p, axis=-1)
    
    # Find discontinuities (where abs(diff) > pi)
    discontinuities = jnp.abs(diff) > jnp.pi
    
    # Compute correction values
    corrections = jnp.where(discontinuities,
                           -2 * jnp.pi * jnp.sign(diff),
                           0.0)
    
    # Apply cumulative corrections
    cumulative_corrections = jnp.cumsum(corrections, axis=-1)
    
    # Pad with zeros for the first element along the axis
    pad_shape = [(0, 0)] * p.ndim
    pad_shape[-1] = (1, 0)
    cumulative_corrections = jnp.pad(cumulative_corrections, pad_shape)
    
    # Apply corrections
    result = p + cumulative_corrections
    
    # Move axis back to original position
    if axis != p.ndim - 1:
        # Create inverse permutation
        inv_axes = list(range(p.ndim))
        inv_axes[axis], inv_axes[-1] = inv_axes[-1], inv_axes[axis]
        result = jnp.transpose(result, inv_axes)
    
    return result


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


def _hilbert_jax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Compute the Hilbert transform using JAX FFT operations.
    
    Args:
        x: Real-valued input array
        axis: Axis along which to compute the Hilbert transform
        
    Returns:
        Complex-valued analytic signal
    """
    # For specific axis values, use direct computation to avoid dynamic axis issues
    if axis == -1 or axis == x.ndim - 1:
        # Process along last axis (most common case)
        N = x.shape[-1]
        X = jnp.fft.fft(x, axis=-1)
        
        # Create the Hilbert transform filter
        h = jnp.zeros(N)
        h = lax.cond(
            N % 2 == 0,
            lambda: h.at[0].set(1).at[1:N//2].set(2).at[N//2].set(1),  # Even length
            lambda: h.at[0].set(1).at[1:(N+1)//2].set(2)                # Odd length
        )
        
        # Apply the filter
        X_filtered = X * h
        
        # Compute inverse FFT
        return jnp.fft.ifft(X_filtered, axis=-1)
        
    elif axis == 0:
        # Process along first axis
        N = x.shape[0]
        X = jnp.fft.fft(x, axis=0)
        
        h = jnp.zeros(N)
        h = lax.cond(
            N % 2 == 0,
            lambda: h.at[0].set(1).at[1:N//2].set(2).at[N//2].set(1),
            lambda: h.at[0].set(1).at[1:(N+1)//2].set(2)
        )
        
        # Reshape h for broadcasting
        shape = [1] * x.ndim
        shape[0] = N
        h = h.reshape(shape)
        
        X_filtered = X * h
        return jnp.fft.ifft(X_filtered, axis=0)
        
    else:
        # For other axes, fall back to moving to last axis and back
        # This will not be JIT-compiled but will work
        x_moved = jnp.moveaxis(x, axis, -1)
        result = _hilbert_jax(x_moved, axis=-1)
        return jnp.moveaxis(result, -1, axis)


@jax.jit
def analytic_signal(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Computes the analytic signal of a real-valued input array using the
    Hilbert transform.

    The analytic signal `xa` of a signal `x` is defined as:
    xa = x + i * h(x)
    where `h(x)` is the Hilbert transform of `x`.

    Args:
        x: The real-valued input array.
        axis: The axis along which to compute the Hilbert transform.

    Returns:
        The complex-valued analytic signal.
    """
    return _hilbert_jax(x, axis=axis)


@jax.jit
def envelope(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Computes the envelope of the signal, which is the magnitude of the
    analytic signal.

    Args:
        x: The real-valued input array.
        axis: The axis along which to compute the analytic signal.

    Returns:
        The envelope of the signal.
    """
    return jnp.abs(analytic_signal(x, axis=axis))


@jax.jit
def instantaneous_phase(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Computes the instantaneous phase of the signal, which is the angle of the
    analytic signal. The phase is unwrapped to be continuous.

    Args:
        x: The real-valued input array.
        axis: The axis along which to compute the analytic signal.

    Returns:
        The unwrapped instantaneous phase of the signal.
    """
    return _jax_unwrap(jnp.angle(analytic_signal(x, axis=axis)), axis=axis)


@jax.jit
def cosine_instantaneous_phase(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Computes the cosine of the instantaneous phase.

    This attribute is useful for highlighting the continuity of seismic reflectors.

    Args:
        x: The real-valued input array.
        axis: The axis along which to compute the instantaneous phase.

    Returns:
        The cosine of the instantaneous phase.
    """
    return jnp.cos(instantaneous_phase(x, axis=axis))


@jax.jit
def instantaneous_frequency(
    x: jnp.ndarray, axis: int = -1, fs: float = 1.0
) -> jnp.ndarray:
    """
    Computes the instantaneous frequency of the signal.

    This is calculated as the time derivative of the unwrapped instantaneous
    phase. A centered finite difference is used for the differentiation.

    Args:
        x: The real-valued input array.
        axis: The axis along which to compute the attribute.
        fs: The sampling frequency of the signal (in Hz). Defaults to 1.0.

    Returns:
        The instantaneous frequency of the signal.
    """
    phi = instantaneous_phase(x, axis=axis)
    # Use custom gradient function
    freq = _jax_gradient(phi, axis=axis)
    return (fs / (2 * jnp.pi)) * freq


@jax.jit
def instantaneous_bandwidth(
    x: jnp.ndarray, axis: int = -1, fs: float = 1.0
) -> jnp.ndarray:
    """
    Computes the instantaneous bandwidth of the signal.

    This is calculated as the time derivative of the signal's envelope.
    A centered finite difference is used for the differentiation.

    Args:
        x: The real-valued input array.
        axis: The axis along which to compute the attribute.
        fs: The sampling frequency of the signal (in Hz). Defaults to 1.0.

    Returns:
        The instantaneous bandwidth of the signal.
    """
    env = envelope(x, axis=axis)
    # Use custom gradient function
    bandwidth = _jax_gradient(env, axis=axis)
    return (fs / (2 * jnp.pi)) * bandwidth 