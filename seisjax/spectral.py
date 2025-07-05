"""
Spectral attributes for seismic interpretation.

This module provides JAX-accelerated implementations of spectral attributes
commonly used in seismic interpretation and reservoir characterization.
"""

from typing import Tuple, Union, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy import signal


def _create_window(window_type: str, nperseg: int) -> jnp.ndarray:
    """
    Create window function since JAX doesn't have all window types.
    Note: This should be called before JIT compilation to avoid string comparison issues.
    
    Args:
        window_type: Type of window ('hann', 'hamming', 'blackman', 'bartlett')
        nperseg: Window length
        
    Returns:
        Window array
    """
    n = jnp.arange(nperseg)
    
    if window_type == 'hann':
        return 0.5 * (1 - jnp.cos(2 * jnp.pi * n / (nperseg - 1)))
    elif window_type == 'hamming':
        return 0.54 - 0.46 * jnp.cos(2 * jnp.pi * n / (nperseg - 1))
    elif window_type == 'blackman':
        return (0.42 - 0.5 * jnp.cos(2 * jnp.pi * n / (nperseg - 1)) + 
                0.08 * jnp.cos(4 * jnp.pi * n / (nperseg - 1)))
    elif window_type == 'bartlett':
        return 1 - 2 * jnp.abs(n - (nperseg - 1) / 2) / (nperseg - 1)
    else:
        return jnp.ones(nperseg)  # rectangular window


def _power_spectrum_jit(
    x: jnp.ndarray,
    window: jnp.ndarray,
    nperseg: int,
    noverlap: int,
    fs: float,
    axis: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX-compatible power spectrum computation using Welch's method.
    """
    # Get input shape
    n_samples = x.shape[axis]
    
    # Calculate number of segments
    step = nperseg - noverlap
    n_segments = (n_samples - noverlap) // step
    
    # Ensure we have at least one segment
    if n_segments < 1:
        n_segments = 1
    
    # Create frequency array
    freqs = jnp.fft.fftfreq(nperseg, 1.0 / fs)[:nperseg // 2 + 1]
    
    # Pad the input array to handle boundary conditions
    if axis == 0:
        x_padded = jnp.pad(x, ((0, nperseg), *(((0, 0),) * (x.ndim - 1))))
    elif axis == 1:
        x_padded = jnp.pad(x, ((0, 0), (0, nperseg), *(((0, 0),) * (x.ndim - 2))))
    else:  # axis == -1 or axis == 2
        if x.ndim == 2:
            x_padded = jnp.pad(x, ((0, 0), (0, nperseg)))
        else:  # 3D case
            x_padded = jnp.pad(x, ((0, 0), (0, 0), (0, nperseg)))
    
    # Function to process each segment using JAX-compatible slicing
    def process_segment(i):
        start_idx = i * step
        
        # Use lax.dynamic_slice for JAX-compatible slicing with traced indices
        if axis == 0:
            # Extract segment along first axis
            segment = lax.dynamic_slice(x_padded, (start_idx,), (nperseg,))
            windowed = segment * window
        elif axis == 1:
            # Extract segment along second axis
            segment = lax.dynamic_slice(x_padded, (0, start_idx), (x.shape[0], nperseg))
            windowed = segment * window[None, :]
        else:  # axis == -1 or axis == 2
            # Extract segment along last axis
            if x.ndim == 2:
                segment = lax.dynamic_slice(x_padded, (0, start_idx), (x.shape[0], nperseg))
                windowed = segment * window
            else:  # 3D case
                segment = lax.dynamic_slice(x_padded, (0, 0, start_idx), (x.shape[0], x.shape[1], nperseg))
                windowed = segment * window
        
        # Take FFT along the windowed axis
        fft_result = jnp.fft.fft(windowed, axis=axis)
        
        # Get one-sided spectrum
        if axis == 0:
            spectrum = fft_result[:nperseg // 2 + 1]
        elif axis == 1:
            spectrum = fft_result[:, :nperseg // 2 + 1]
        else:  # axis == -1 or axis == 2
            spectrum = fft_result[..., :nperseg // 2 + 1]
        
        # Compute power spectral density
        psd = jnp.abs(spectrum) ** 2
        
        # Scale for one-sided spectrum (avoid first and last bins for 2x scaling)
        scaling_start = 1
        scaling_end = nperseg // 2
        if nperseg % 2 == 0:
            scaling_end = nperseg // 2  # Don't scale Nyquist frequency for even nperseg
        else:
            scaling_end = (nperseg + 1) // 2  # Scale all the way for odd nperseg
        
        if axis == 0:
            psd = psd.at[scaling_start:scaling_end].multiply(2)
        elif axis == 1:
            psd = psd.at[:, scaling_start:scaling_end].multiply(2)
        else:  # axis == -1 or axis == 2
            psd = psd.at[..., scaling_start:scaling_end].multiply(2)
        
        return psd
    
    # Process all segments
    if n_segments > 1:
        segments = jax.vmap(process_segment)(jnp.arange(n_segments))
        # Average across segments
        psd = jnp.mean(segments, axis=0)
    else:
        psd = process_segment(0)
    
    # Normalize by sampling frequency and window power
    window_power = jnp.sum(window ** 2)
    psd = psd / (fs * window_power)
    
    return freqs, psd


def power_spectrum(
    x: jnp.ndarray,
    axis: int = -1,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    fs: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the power spectral density using Welch's method.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the spectrum
        nperseg: Length of each segment
        noverlap: Number of points to overlap between segments
        window: Window function type
        fs: Sampling frequency
        
    Returns:
        Tuple of (frequencies, power_spectrum)
    """
    if nperseg is None:
        nperseg = min(256, x.shape[axis])
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Create window outside of JIT
    window_array = _create_window(window, nperseg)
    
    # Use JAX-compatible power spectrum computation
    freqs, psd = _power_spectrum_jit(x, window_array, nperseg, noverlap, fs, axis)
    
    return freqs, psd


def dominant_frequency(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute the dominant frequency of the signal.
    
    The dominant frequency is the frequency with the maximum power
    in the power spectrum.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the dominant frequency
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        
    Returns:
        Dominant frequency array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    # Compute power spectrum
    freqs, psd = power_spectrum(x, axis=axis, nperseg=window_length, fs=fs)
    
    # Find dominant frequency using JAX-compatible method
    dom_freq_idx = jnp.argmax(psd, axis=axis)
    
    # Use jnp.take to avoid TracerIntegerConversionError
    dom_freq = jnp.take(freqs, dom_freq_idx, axis=0)
    
    return dom_freq


def spectral_centroid(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute the spectral centroid (center of mass of the spectrum).
    
    The spectral centroid indicates where the "center of mass" of the
    spectrum is located. It is the weighted mean of the frequencies
    present in the signal.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the spectral centroid
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        
    Returns:
        Spectral centroid array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    # Compute power spectrum
    freqs, psd = power_spectrum(x, axis=axis, nperseg=window_length, fs=fs)
    
    # Compute spectral centroid
    centroid = jnp.sum(freqs * psd, axis=axis) / (jnp.sum(psd, axis=axis) + 1e-10)
    
    return centroid


def spectral_bandwidth(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute the spectral bandwidth (spread of the spectrum).
    
    The spectral bandwidth is the width of the spectrum and is computed
    as the square root of the second moment of the power spectrum.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the spectral bandwidth
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        
    Returns:
        Spectral bandwidth array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    # Compute power spectrum
    freqs, psd = power_spectrum(x, axis=axis, nperseg=window_length, fs=fs)
    
    # Compute spectral centroid
    centroid = jnp.sum(freqs * psd, axis=axis) / (jnp.sum(psd, axis=axis) + 1e-10)
    
    # Compute spectral bandwidth
    bandwidth = jnp.sqrt(
        jnp.sum(((freqs - centroid[..., None])**2) * psd, axis=axis) / 
        (jnp.sum(psd, axis=axis) + 1e-10)
    )
    
    return bandwidth


def spectral_slope(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute the spectral slope (rate of change of spectrum with frequency).
    
    The spectral slope is computed as the slope of the linear regression
    line fitted to the log-power spectrum.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the spectral slope
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        
    Returns:
        Spectral slope array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    # Compute power spectrum
    freqs, psd = power_spectrum(x, axis=axis, nperseg=window_length, fs=fs)
    
    # Avoid log(0) by adding small epsilon
    log_psd = jnp.log(psd + 1e-10)
    
    # Compute spectral slope using linear regression
    # slope = (N * sum(f * log_psd) - sum(f) * sum(log_psd)) / (N * sum(f^2) - sum(f)^2)
    N = freqs.shape[0]
    sum_f = jnp.sum(freqs)
    sum_f2 = jnp.sum(freqs**2)
    sum_log_psd = jnp.sum(log_psd, axis=axis)
    sum_f_log_psd = jnp.sum(freqs * log_psd, axis=axis)
    
    slope = (N * sum_f_log_psd - sum_f * sum_log_psd) / (N * sum_f2 - sum_f**2 + 1e-10)
    
    return slope


def spectral_rolloff(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None,
    rolloff_percent: float = 0.85
) -> jnp.ndarray:
    """
    Compute the spectral rolloff frequency.
    
    The spectral rolloff is the frequency below which a specified percentage
    of the total spectral energy is contained.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the spectral rolloff
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        rolloff_percent: Percentage of total energy for rolloff calculation
        
    Returns:
        Spectral rolloff frequency array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    # Compute power spectrum
    freqs, psd = power_spectrum(x, axis=axis, nperseg=window_length, fs=fs)
    
    # Compute cumulative sum of power
    cumsum_psd = jnp.cumsum(psd, axis=axis)
    total_energy = jnp.sum(psd, axis=axis, keepdims=True)
    
    # Find rolloff frequency using JAX-compatible method
    rolloff_energy = rolloff_percent * total_energy
    rolloff_idx = jnp.argmax(cumsum_psd >= rolloff_energy, axis=axis)
    
    # Use jnp.take to avoid TracerIntegerConversionError
    rolloff_freq = jnp.take(freqs, rolloff_idx, axis=0)
    
    return rolloff_freq


def spectral_flux(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None,
    hop_length: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute the spectral flux (rate of change of spectrum over time).
    
    The spectral flux measures how quickly the power spectrum changes
    over time. It is computed as the sum of positive differences between
    consecutive power spectra.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the spectral flux
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        hop_length: Hop length for sliding window
        
    Returns:
        Spectral flux array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    if hop_length is None:
        hop_length = window_length // 2
    
    # Compute power spectrum
    freqs, psd = power_spectrum(x, axis=axis, nperseg=window_length, fs=fs)
    
    # Compute spectral flux as the sum of positive differences
    # between consecutive frames
    diff_psd = jnp.diff(psd, axis=axis)
    flux = jnp.sum(jnp.maximum(diff_psd, 0), axis=axis)
    
    return flux


def spectral_flatness(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute the spectral flatness (measure of how noise-like the spectrum is).
    
    The spectral flatness is the ratio of the geometric mean to the
    arithmetic mean of the power spectrum. It ranges from 0 (tonal)
    to 1 (noise-like).
    
    Args:
        x: Input signal
        axis: Axis along which to compute the spectral flatness
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        
    Returns:
        Spectral flatness array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    # Compute power spectrum
    freqs, psd = power_spectrum(x, axis=axis, nperseg=window_length, fs=fs)
    
    # Compute spectral flatness
    # Geometric mean / Arithmetic mean
    geometric_mean = jnp.exp(jnp.mean(jnp.log(psd + 1e-10), axis=axis))
    arithmetic_mean = jnp.mean(psd, axis=axis)
    
    flatness = geometric_mean / (arithmetic_mean + 1e-10)
    
    return flatness


def spectral_contrast(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None,
    n_bands: int = 6
) -> jnp.ndarray:
    """
    Compute the spectral contrast (difference between peaks and valleys).
    
    The spectral contrast measures the difference in dB between
    peaks and valleys in the spectrum across different frequency bands.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the spectral contrast
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        n_bands: Number of frequency bands
        
    Returns:
        Spectral contrast array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    # Compute power spectrum
    freqs, psd = power_spectrum(x, axis=axis, nperseg=window_length, fs=fs)
    
    # Divide spectrum into bands
    n_bins = psd.shape[axis]
    band_size = n_bins // n_bands
    
    contrasts = []
    for i in range(n_bands):
        start_idx = i * band_size
        end_idx = min((i + 1) * band_size, n_bins)
        
        # Extract band
        if axis == 0:
            band_psd = psd[start_idx:end_idx]
        elif axis == 1:
            band_psd = psd[:, start_idx:end_idx]
        else:  # axis == -1 or axis == 2
            band_psd = psd[..., start_idx:end_idx]
        
        # Compute contrast for this band
        peak = jnp.max(band_psd, axis=axis)
        valley = jnp.mean(band_psd, axis=axis)
        contrast = 20 * jnp.log10((peak + 1e-10) / (valley + 1e-10))
        contrasts.append(contrast)
    
    return jnp.stack(contrasts, axis=axis)


def zero_crossing_rate(
    x: jnp.ndarray,
    axis: int = -1,
    frame_length: int = 1024,
    hop_length: int = 512
) -> jnp.ndarray:
    """
    Compute the zero-crossing rate.
    
    The zero-crossing rate is the rate at which the signal changes sign.
    It is commonly used to distinguish between voiced and unvoiced speech.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the zero-crossing rate
        frame_length: Length of each frame
        hop_length: Hop length between frames
        
    Returns:
        Zero-crossing rate array
    """
    # Compute sign changes
    signs = jnp.sign(x)
    sign_changes = jnp.abs(jnp.diff(signs, axis=axis))
    
    # Count zero crossings in each frame
    n_samples = x.shape[axis]
    n_frames = (n_samples - frame_length) // hop_length + 1
    
    zcr = []
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        
        if axis == 0:
            frame_changes = sign_changes[start:end]
        elif axis == 1:
            frame_changes = sign_changes[:, start:end]
        else:  # axis == -1 or axis == 2
            frame_changes = sign_changes[..., start:end]
        
        frame_zcr = jnp.sum(frame_changes, axis=axis) / frame_length
        zcr.append(frame_zcr)
    
    return jnp.stack(zcr, axis=axis)


def spectral_energy(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None,
    freq_range: Optional[Tuple[float, float]] = None
) -> jnp.ndarray:
    """
    Compute the spectral energy within a frequency range.
    
    The spectral energy is the sum of the power spectrum within
    a specified frequency range.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the spectral energy
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        freq_range: Frequency range (min_freq, max_freq) in Hz
        
    Returns:
        Spectral energy array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    # Compute power spectrum
    freqs, psd = power_spectrum(x, axis=axis, nperseg=window_length, fs=fs)
    
    # Apply frequency range filter if specified
    if freq_range is not None:
        min_freq, max_freq = freq_range
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        
        if axis == 0:
            psd = psd[freq_mask]
        elif axis == 1:
            psd = psd[:, freq_mask]
        else:  # axis == -1 or axis == 2
            psd = psd[..., freq_mask]
    
    # Compute spectral energy
    energy = jnp.sum(psd, axis=axis)
    
    return energy


def peak_frequency(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute the peak frequency of the signal.
    
    The peak frequency is the frequency bin with the maximum power
    in the power spectrum.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the peak frequency
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        
    Returns:
        Peak frequency array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    # Compute power spectrum
    freqs, psd = power_spectrum(x, axis=axis, nperseg=window_length, fs=fs)
    
    # Find peak frequency using JAX-compatible method
    peak_idx = jnp.argmax(psd, axis=axis)
    
    # Use jnp.take to avoid TracerIntegerConversionError
    peak_freq = jnp.take(freqs, peak_idx, axis=0)
    
    return peak_freq


def _spectral_decomposition_jit(
    x: jnp.ndarray,
    window: jnp.ndarray,
    fs: float,
    hop_length: int,
    axis: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Core spectral decomposition function (removed JIT for axis compatibility).
    """
    # Handle specific axis cases to avoid moveaxis in JIT
    if axis == -1 or axis == x.ndim - 1:
        # Already at last axis
        work_data = x
    elif axis == 0:
        # Move first axis to last using transpose
        axes = list(range(x.ndim))
        axes = axes[1:] + [0]
        work_data = jnp.transpose(x, axes)
    elif axis == 1 and x.ndim == 3:
        # Move second axis to last for 3D data
        work_data = jnp.transpose(x, (0, 2, 1))
    else:
        # For other cases, use a safe approach
        work_data = x
    
    # Get parameters
    nperseg = len(window)
    n_times = (work_data.shape[-1] - nperseg) // hop_length + 1
    n_freqs = nperseg // 2 + 1
    
    # Perform STFT for each trace using vectorized operations
    def stft_trace(trace):
        """Compute STFT for a single trace using vectorized operations."""
        # Create all windowed segments at once
        def extract_segment(i):
            start_idx = i * hop_length
            end_idx = start_idx + nperseg
            # Use lax.dynamic_slice for JAX-compatible slicing
            segment = lax.dynamic_slice(
                jnp.pad(trace, (0, nperseg)), 
                (start_idx,), 
                (nperseg,)
            ) * window
            # Compute FFT and take positive frequencies
            fft_result = jnp.fft.fft(segment)
            return fft_result[:n_freqs]
        
        # Vectorize over time indices
        time_indices = jnp.arange(n_times)
        spectogram = jax.vmap(extract_segment)(time_indices)
        return spectogram.T  # Shape: (n_freqs, n_times)
    
    # Apply STFT to each trace
    original_shape = work_data.shape
    traces = work_data.reshape(-1, work_data.shape[-1])
    stft_results = jax.vmap(stft_trace)(traces)
    
    # Reshape back to original spatial dimensions + frequency/time
    result_shape = original_shape[:-1] + (n_freqs, n_times)
    stft_result = stft_results.reshape(result_shape)
    
    # Create frequency and time arrays
    frequencies = jnp.fft.fftfreq(nperseg, 1.0 / fs)[:n_freqs]
    times = jnp.arange(n_times) * hop_length / fs
    
    # Handle axis restoration for output
    if axis == -1 or axis == x.ndim - 1:
        # Already correct
        pass
    elif axis == 0:
        # Move time axis back to first position
        axes = [stft_result.ndim - 1] + list(range(stft_result.ndim - 1))
        stft_result = jnp.transpose(stft_result, axes)
    elif axis == 1 and x.ndim == 3:
        # Restore original axis order for 3D data
        stft_result = jnp.transpose(stft_result, (0, 2, 1, 3, 4))
    
    return frequencies, times, stft_result


def spectral_decomposition(
    x: jnp.ndarray,
    fs: float = 250.0,
    window_length: float = 0.2,
    hop_length: int = 1,
    window_type: str = 'hann',
    axis: int = -1
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Perform spectral decomposition on seismic data using Short-Time Fourier Transform (STFT).
    
    This function computes the time-frequency representation of seismic data,
    allowing analysis of frequency content that varies with time.
    
    Args:
        x: Input seismic data (can be 1D, 2D, or 3D)
        fs: Sampling frequency in Hz
        window_length: Length of the STFT window in seconds
        hop_length: Step size for the STFT window
        window_type: Type of window function ('hann', 'hamming', 'blackman', 'bartlett')
        axis: Axis along which to compute the STFT (time axis)
        
    Returns:
        Tuple of (frequencies, times, spectrogram)
        - frequencies: Frequency array
        - times: Time array
        - spectrogram: Complex-valued STFT coefficients
    """
    # Calculate window size in samples
    nperseg = int(fs * window_length)
    
    # Create window function outside JIT compilation
    window = _create_window(window_type, nperseg)
    
    # Call JIT-compiled function
    return _spectral_decomposition_jit(x, window, fs, hop_length, axis)


def instantaneous_amplitude_spectrum(
    x: jnp.ndarray,
    fs: float = 250.0,
    window_length: float = 0.2,
    axis: int = -1
) -> jnp.ndarray:
    """
    Compute instantaneous amplitude spectrum from spectral decomposition.
    
    Args:
        x: Input seismic data
        fs: Sampling frequency in Hz
        window_length: Length of the STFT window in seconds
        axis: Axis along which to compute the spectrum
        
    Returns:
        Instantaneous amplitude spectrum
    """
    frequencies, times, stft = spectral_decomposition(x, fs, window_length, axis=axis)
    
    # Return magnitude of complex STFT
    return jnp.abs(stft)


def time_frequency_decomposition(
    x: jnp.ndarray,
    fs: float = 250.0,
    frequency_range: Tuple[float, float] = (10.0, 100.0),
    n_frequencies: int = 20,
    axis: int = -1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Decompose seismic data into specific frequency bands.
    
    Args:
        x: Input seismic data
        fs: Sampling frequency in Hz
        frequency_range: Tuple of (min_freq, max_freq) in Hz
        n_frequencies: Number of frequency bands to extract
        axis: Axis along which to compute the decomposition
        
    Returns:
        Tuple of (frequencies, frequency_volumes)
        - frequencies: Array of center frequencies for each band
        - frequency_volumes: Array of filtered data for each frequency band
    """
    # Create frequency bands
    min_freq, max_freq = frequency_range
    frequencies = jnp.linspace(min_freq, max_freq, n_frequencies)
    
    # Compute spectral decomposition
    freqs, times, stft = spectral_decomposition(x, fs, axis=axis)
    
    # Extract amplitude for each frequency band using vectorized operations
    def extract_frequency_band(target_freq):
        # Find closest frequency bin
        freq_idx = jnp.argmin(jnp.abs(freqs - target_freq))
        # Extract amplitude at this frequency
        return jnp.abs(stft[..., freq_idx, :])
    
    # Vectorize over all target frequencies
    frequency_volumes = jax.vmap(extract_frequency_band)(frequencies)
    
    # Transpose to get the right shape
    frequency_volumes = jnp.transpose(frequency_volumes, (1, 2, 0))
    
    return frequencies, frequency_volumes


def rgb_frequency_blend(
    x: jnp.ndarray,
    fs: float = 250.0,
    freq_red: float = 15.0,
    freq_green: float = 35.0,
    freq_blue: float = 60.0,
    axis: int = -1
) -> jnp.ndarray:
    """
    Create RGB frequency blend for seismic visualization.
    
    This function extracts three frequency components and combines them
    as RGB channels for enhanced visualization of frequency content.
    
    Args:
        x: Input seismic data
        fs: Sampling frequency in Hz
        freq_red: Center frequency for red channel
        freq_green: Center frequency for green channel
        freq_blue: Center frequency for blue channel
        axis: Axis along which to compute the decomposition
        
    Returns:
        RGB frequency blend array with shape (..., 3)
    """
    # Get frequency decomposition
    freqs, times, stft = spectral_decomposition(x, fs, axis=axis)
    
    # Find closest frequency bins
    red_idx = jnp.argmin(jnp.abs(freqs - freq_red))
    green_idx = jnp.argmin(jnp.abs(freqs - freq_green))
    blue_idx = jnp.argmin(jnp.abs(freqs - freq_blue))
    
    # Extract amplitudes
    red_channel = jnp.abs(stft[..., red_idx, :])
    green_channel = jnp.abs(stft[..., green_idx, :])
    blue_channel = jnp.abs(stft[..., blue_idx, :])
    
    # Normalize each channel
    red_channel = (red_channel - jnp.min(red_channel)) / (jnp.max(red_channel) - jnp.min(red_channel) + 1e-10)
    green_channel = (green_channel - jnp.min(green_channel)) / (jnp.max(green_channel) - jnp.min(green_channel) + 1e-10)
    blue_channel = (blue_channel - jnp.min(blue_channel)) / (jnp.max(blue_channel) - jnp.min(blue_channel) + 1e-10)
    
    # Stack as RGB
    rgb_blend = jnp.stack([red_channel, green_channel, blue_channel], axis=-1)
    
    return rgb_blend 