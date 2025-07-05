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


@jax.jit
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
    
    # Use JAX's signal processing for power spectrum
    freqs, psd = signal.welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=axis
    )
    
    return freqs, psd


@jax.jit
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
    
    # Find dominant frequency
    dom_freq_idx = jnp.argmax(psd, axis=axis)
    dom_freq = freqs[dom_freq_idx]
    
    return dom_freq


@jax.jit
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


@jax.jit
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


@jax.jit
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


@jax.jit
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
    
    # Find rolloff frequency
    rolloff_energy = rolloff_percent * total_energy
    rolloff_idx = jnp.argmax(cumsum_psd >= rolloff_energy, axis=axis)
    rolloff_freq = freqs[rolloff_idx]
    
    return rolloff_freq


@jax.jit
def spectral_flux(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None,
    hop_length: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute the spectral flux (rate of change of spectrum over time).
    
    Spectral flux measures the rate of change of the power spectrum
    and is useful for detecting onset events.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the spectral flux
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        hop_length: Hop length between successive windows
        
    Returns:
        Spectral flux array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    if hop_length is None:
        hop_length = window_length // 4
    
    # Compute short-time Fourier transform
    freqs, times, stft = signal.stft(
        x,
        fs=fs,
        window='hann',
        nperseg=window_length,
        noverlap=window_length - hop_length,
        axis=axis
    )
    
    # Compute power spectrum
    power = jnp.abs(stft)**2
    
    # Compute spectral flux
    diff_power = jnp.diff(power, axis=-1)
    flux = jnp.sum(jnp.maximum(diff_power, 0), axis=-2)
    
    return flux


@jax.jit
def spectral_flatness(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute the spectral flatness (Wiener entropy).
    
    Spectral flatness is a measure of how noise-like vs. tonal a signal is.
    A higher spectral flatness indicates more noise-like characteristics.
    
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
    
    # Avoid log(0) by adding small epsilon
    psd_eps = psd + 1e-10
    
    # Compute spectral flatness
    geometric_mean = jnp.exp(jnp.mean(jnp.log(psd_eps), axis=axis))
    arithmetic_mean = jnp.mean(psd, axis=axis)
    
    flatness = geometric_mean / (arithmetic_mean + 1e-10)
    
    return flatness


@jax.jit
def spectral_contrast(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None,
    n_bands: int = 6
) -> jnp.ndarray:
    """
    Compute the spectral contrast.
    
    Spectral contrast considers the spectral peak, spectral valley, and
    their difference in each frequency subband.
    
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
    
    # Create frequency bands
    n_freqs = freqs.shape[0]
    band_size = n_freqs // n_bands
    
    contrast = []
    for i in range(n_bands):
        start_idx = i * band_size
        end_idx = (i + 1) * band_size if i < n_bands - 1 else n_freqs
        
        band_psd = psd[start_idx:end_idx]
        
        # Compute contrast for this band
        sorted_band = jnp.sort(band_psd, axis=0)
        peak = jnp.mean(sorted_band[-band_size//10:], axis=0)  # Top 10%
        valley = jnp.mean(sorted_band[:band_size//10], axis=0)  # Bottom 10%
        
        band_contrast = jnp.log(peak / (valley + 1e-10) + 1e-10)
        contrast.append(band_contrast)
    
    return jnp.stack(contrast, axis=0)


@jax.jit
def zero_crossing_rate(
    x: jnp.ndarray,
    axis: int = -1,
    frame_length: int = 1024,
    hop_length: int = 512
) -> jnp.ndarray:
    """
    Compute the zero-crossing rate of the signal.
    
    The zero-crossing rate is the rate at which the signal changes sign.
    It is a simple measure of the noisiness of the signal.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the zero-crossing rate
        frame_length: Length of the frame for analysis
        hop_length: Hop length between successive frames
        
    Returns:
        Zero-crossing rate array
    """
    # Compute sign changes
    sign_changes = jnp.diff(jnp.sign(x), axis=axis)
    
    # Count zero crossings in frames
    n_frames = (x.shape[axis] - frame_length) // hop_length + 1
    zcr = jnp.zeros(x.shape[:-1] + (n_frames,))
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame_changes = sign_changes[..., start:end-1]
        zcr = zcr.at[..., i].set(jnp.sum(jnp.abs(frame_changes), axis=axis) / (2 * frame_length))
    
    return zcr


@jax.jit
def spectral_energy(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None,
    freq_range: Optional[Tuple[float, float]] = None
) -> jnp.ndarray:
    """
    Compute the spectral energy in a specific frequency range.
    
    Args:
        x: Input signal
        axis: Axis along which to compute the spectral energy
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        freq_range: Tuple of (low_freq, high_freq) for energy calculation
        
    Returns:
        Spectral energy array
    """
    if window_length is None:
        window_length = min(64, x.shape[axis])
    
    # Compute power spectrum
    freqs, psd = power_spectrum(x, axis=axis, nperseg=window_length, fs=fs)
    
    if freq_range is not None:
        low_freq, high_freq = freq_range
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
        psd_masked = jnp.where(freq_mask, psd, 0)
    else:
        psd_masked = psd
    
    # Compute total energy
    energy = jnp.sum(psd_masked, axis=axis)
    
    return energy


@jax.jit
def peak_frequency(
    x: jnp.ndarray,
    axis: int = -1,
    fs: float = 1.0,
    window_length: Optional[int] = None
) -> jnp.ndarray:
    """
    Compute the peak frequency (frequency with maximum amplitude).
    
    Args:
        x: Input signal
        axis: Axis along which to compute the peak frequency
        fs: Sampling frequency
        window_length: Length of the window for spectral analysis
        
    Returns:
        Peak frequency array
    """
    # This is essentially the same as dominant_frequency
    return dominant_frequency(x, axis=axis, fs=fs, window_length=window_length)


@jax.jit
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
    
    # Create window function
    if window_type == 'hann':
        window = jnp.hanning(nperseg)
    elif window_type == 'hamming':
        window = jnp.hamming(nperseg)  
    elif window_type == 'blackman':
        window = jnp.blackman(nperseg)
    elif window_type == 'bartlett':
        window = jnp.bartlett(nperseg)
    else:
        window = jnp.ones(nperseg)  # rectangular window
    
    # Move time axis to the last dimension for easier processing
    x = jnp.moveaxis(x, axis, -1)
    original_shape = x.shape
    
    # Flatten all dimensions except time
    x_flat = x.reshape(-1, x.shape[-1])
    n_traces, n_samples = x_flat.shape
    
    # Calculate output dimensions
    n_freqs = nperseg // 2 + 1
    n_times = (n_samples - nperseg) // hop_length + 1
    
    # Initialize output
    stft_result = jnp.zeros((n_traces, n_freqs, n_times), dtype=jnp.complex64)
    
    # Perform STFT for each trace
    def stft_trace(trace):
        """Compute STFT for a single trace."""
        spectogram = jnp.zeros((n_freqs, n_times), dtype=jnp.complex64)
        
        for i in range(n_times):
            start_idx = i * hop_length
            end_idx = start_idx + nperseg
            
            if end_idx <= len(trace):
                # Extract windowed segment
                segment = trace[start_idx:end_idx] * window
                
                # Compute FFT
                fft_result = jnp.fft.fft(segment)
                
                # Take only positive frequencies
                spectogram = spectogram.at[:, i].set(fft_result[:n_freqs])
        
        return spectogram
    
    # Apply STFT to all traces
    stft_result = jax.vmap(stft_trace)(x_flat)
    
    # Reshape back to original spatial dimensions
    new_shape = original_shape[:-1] + (n_freqs, n_times)
    stft_result = stft_result.reshape(new_shape)
    
    # Move time axis back to original position
    stft_result = jnp.moveaxis(stft_result, -1, axis)
    stft_result = jnp.moveaxis(stft_result, -1, axis)
    
    # Create frequency and time arrays
    frequencies = jnp.fft.fftfreq(nperseg, 1/fs)[:n_freqs]
    times = jnp.arange(n_times) * hop_length / fs
    
    return frequencies, times, stft_result


@jax.jit
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


@jax.jit
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
    
    # Extract amplitude for each frequency band
    frequency_volumes = []
    
    for target_freq in frequencies:
        # Find closest frequency bin
        freq_idx = jnp.argmin(jnp.abs(freqs - target_freq))
        
        # Extract amplitude at this frequency
        freq_volume = jnp.abs(stft[..., freq_idx, :])
        frequency_volumes.append(freq_volume)
    
    return frequencies, jnp.stack(frequency_volumes, axis=-1)


@jax.jit
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