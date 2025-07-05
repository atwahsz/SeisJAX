from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


def _check_axis(axis: int, ndim: int) -> None:
    """Helper to ensure the specified axis is valid for the array dimensions."""
    # Note: JAX will handle axis validation naturally, so we skip dynamic checks
    # inside JIT-compiled functions to avoid TracerBoolConversionError
    pass


def _hilbert_jax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Compute the Hilbert transform using JAX FFT operations.
    
    Args:
        x: Real-valued input array
        axis: Axis along which to compute the Hilbert transform
        
    Returns:
        Complex-valued analytic signal
    """
    # Move the specified axis to the last position
    x = jnp.moveaxis(x, axis, -1)
    
    # Get the length of the transform axis
    N = x.shape[-1]
    
    # Compute FFT
    X = jnp.fft.fft(x, axis=-1)
    
    # Create the Hilbert transform filter
    h = jnp.zeros(N)
    # Use JAX conditional to avoid TracerBoolConversionError
    h = h.at[0].set(1)
    h = lax.cond(
        N % 2 == 0,
        lambda: h.at[1:N//2].set(2).at[N//2].set(1),  # Even length
        lambda: h.at[1:(N+1)//2].set(2)                # Odd length
    )
    
    # Apply the filter
    X_filtered = X * h
    
    # Compute inverse FFT
    analytic = jnp.fft.ifft(X_filtered, axis=-1)
    
    # Move the axis back to its original position
    analytic = jnp.moveaxis(analytic, -1, axis)
    
    return analytic


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
    return jnp.unwrap(jnp.angle(analytic_signal(x, axis=axis)), axis=axis)


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
    # Use centered finite differences for the derivative
    freq = jnp.gradient(phi, axis=axis)
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
    # Use centered finite differences for the derivative
    bandwidth = jnp.gradient(env, axis=axis)
    return (fs / (2 * jnp.pi)) * bandwidth 