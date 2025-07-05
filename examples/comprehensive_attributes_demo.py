#!/usr/bin/env python3
"""
Comprehensive demonstration of SeisJAX attributes.

This script demonstrates the JAX-accelerated seismic attributes available in SeisJAX,
showcasing the performance benefits compared to traditional CPU implementations.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Import SeisJAX attributes
import seisjax


def generate_synthetic_data() -> Dict[str, jnp.ndarray]:
    """
    Generate synthetic seismic data for demonstration.
    
    Returns:
        Dictionary containing various synthetic datasets
    """
    print("Generating synthetic seismic data...")
    
    # 1D trace for complex trace attributes
    time = jnp.linspace(-1, 1, 201)
    ricker = (1 - 2 * (jnp.pi**2) * (25**2) * (time**2)) * jnp.exp(-(jnp.pi**2) * (25**2) * (time**2))
    
    # Add some noise
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, ricker.shape) * 0.1
    ricker_noisy = ricker + noise
    
    # 2D horizon surface for curvature attributes
    x = jnp.linspace(-5, 5, 100)
    y = jnp.linspace(-5, 5, 100)
    X, Y = jnp.meshgrid(x, y)
    
    # Create a synthetic horizon with some structures
    horizon = (
        2 * jnp.exp(-(X**2 + Y**2) / 4) +  # Central dome
        jnp.sin(X) * jnp.cos(Y) * 0.5 +    # Regular undulations
        jnp.random.normal(jax.random.PRNGKey(123), X.shape) * 0.1  # Noise
    )
    
    # 3D seismic volume for coherence and geometric attributes
    nz, ny, nx = 64, 64, 64
    key = jax.random.PRNGKey(456)
    
    # Create layered structure with some faults
    z = jnp.arange(nz)[:, None, None]
    y_3d = jnp.arange(ny)[None, :, None]
    x_3d = jnp.arange(nx)[None, None, :]
    
    # Base layered model
    seismic_3d = jnp.sin(z * 0.3) + 0.2 * jnp.sin(z * 0.8)
    
    # Add lateral variations
    seismic_3d += 0.3 * jnp.sin(x_3d * 0.2) * jnp.cos(y_3d * 0.15)
    
    # Add some faults (discontinuities)
    fault_mask = (x_3d > 30) & (x_3d < 35)
    seismic_3d = jnp.where(fault_mask, seismic_3d + 1.5, seismic_3d)
    
    # Add noise
    noise_3d = jax.random.normal(jax.random.PRNGKey(789), seismic_3d.shape) * 0.1
    seismic_3d += noise_3d
    
    return {
        'trace_1d': ricker_noisy,
        'time': time,
        'horizon_2d': horizon,
        'x_2d': X,
        'y_2d': Y,
        'seismic_3d': seismic_3d
    }


def demo_complex_trace_attributes(data: Dict[str, jnp.ndarray]) -> None:
    """Demonstrate complex trace attributes."""
    print("\n=== Complex Trace Attributes Demo ===")
    
    trace = data['trace_1d']
    
    # Time the computations
    start_time = time.time()
    
    # Compute various complex trace attributes
    analytic = seisjax.analytic_signal(trace)
    envelope = seisjax.envelope(trace)
    inst_phase = seisjax.instantaneous_phase(trace)
    cos_phase = seisjax.cosine_instantaneous_phase(trace)
    inst_freq = seisjax.instantaneous_frequency(trace, fs=100.0)
    inst_bandwidth = seisjax.instantaneous_bandwidth(trace, fs=100.0)
    
    end_time = time.time()
    print(f"Complex trace attributes computed in {end_time - start_time:.4f} seconds")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    time_axis = data['time']
    
    axes[0, 0].plot(time_axis, trace)
    axes[0, 0].set_title('Original Trace')
    axes[0, 0].set_xlabel('Time (s)')
    
    axes[0, 1].plot(time_axis, envelope)
    axes[0, 1].set_title('Envelope')
    axes[0, 1].set_xlabel('Time (s)')
    
    axes[0, 2].plot(time_axis, inst_phase)
    axes[0, 2].set_title('Instantaneous Phase')
    axes[0, 2].set_xlabel('Time (s)')
    
    axes[1, 0].plot(time_axis, cos_phase)
    axes[1, 0].set_title('Cosine Instantaneous Phase')
    axes[1, 0].set_xlabel('Time (s)')
    
    axes[1, 1].plot(time_axis, inst_freq)
    axes[1, 1].set_title('Instantaneous Frequency')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    
    axes[1, 2].plot(time_axis, inst_bandwidth)
    axes[1, 2].set_title('Instantaneous Bandwidth')
    axes[1, 2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('complex_trace_attributes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Complex trace attributes visualization saved as 'complex_trace_attributes.png'")


def demo_curvature_attributes(data: Dict[str, jnp.ndarray]) -> None:
    """Demonstrate curvature attributes."""
    print("\n=== Curvature Attributes Demo ===")
    
    horizon = data['horizon_2d']
    
    start_time = time.time()
    
    # Compute various curvature attributes
    mean_curv = seisjax.mean_curvature(horizon)
    gaussian_curv = seisjax.gaussian_curvature(horizon)
    max_curv = seisjax.maximum_curvature(horizon)
    min_curv = seisjax.minimum_curvature(horizon)
    curvedness = seisjax.curvedness(horizon)
    shape_idx = seisjax.shape_index(horizon)
    
    end_time = time.time()
    print(f"Curvature attributes computed in {end_time - start_time:.4f} seconds")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    X, Y = data['x_2d'], data['y_2d']
    
    im1 = axes[0, 0].contourf(X, Y, horizon, levels=20, cmap='terrain')
    axes[0, 0].set_title('Original Horizon')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].contourf(X, Y, mean_curv, levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('Mean Curvature')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].contourf(X, Y, gaussian_curv, levels=20, cmap='RdBu_r')
    axes[0, 2].set_title('Gaussian Curvature')
    plt.colorbar(im3, ax=axes[0, 2])
    
    im4 = axes[1, 0].contourf(X, Y, max_curv, levels=20, cmap='Reds')
    axes[1, 0].set_title('Maximum Curvature')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].contourf(X, Y, curvedness, levels=20, cmap='viridis')
    axes[1, 1].set_title('Curvedness')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].contourf(X, Y, shape_idx, levels=20, cmap='coolwarm')
    axes[1, 2].set_title('Shape Index')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('curvature_attributes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Curvature attributes visualization saved as 'curvature_attributes.png'")


def demo_coherence_attributes(data: Dict[str, jnp.ndarray]) -> None:
    """Demonstrate coherence attributes."""
    print("\n=== Coherence Attributes Demo ===")
    
    seismic_3d = data['seismic_3d']
    
    start_time = time.time()
    
    # Compute coherence attributes (using smaller window for speed)
    window_shape = (3, 3, 3)
    
    # Note: These are computationally intensive, so we'll use a subset
    semblance = seisjax.semblance(seismic_3d, window_shape=window_shape)
    energy_ratio = seisjax.energy_ratio_coherence(seismic_3d, window_shape=window_shape)
    variance_coh = seisjax.variance_coherence(seismic_3d, window_shape=window_shape)
    
    end_time = time.time()
    print(f"Coherence attributes computed in {end_time - start_time:.4f} seconds")
    
    # Visualize time slices
    mid_slice = seismic_3d.shape[0] // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    im1 = axes[0, 0].imshow(seismic_3d[mid_slice], cmap='seismic', aspect='auto')
    axes[0, 0].set_title('Original Seismic (Time Slice)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(semblance[mid_slice], cmap='hot', aspect='auto')
    axes[0, 1].set_title('Semblance Coherence')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[1, 0].imshow(energy_ratio[mid_slice], cmap='plasma', aspect='auto')
    axes[1, 0].set_title('Energy Ratio Coherence')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(variance_coh[mid_slice], cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Variance Coherence')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('coherence_attributes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Coherence attributes visualization saved as 'coherence_attributes.png'")


def demo_spectral_attributes(data: Dict[str, jnp.ndarray]) -> None:
    """Demonstrate spectral attributes."""
    print("\n=== Spectral Attributes Demo ===")
    
    trace = data['trace_1d']
    
    start_time = time.time()
    
    # Compute spectral attributes
    fs = 100.0  # Sampling frequency
    dom_freq = seisjax.dominant_frequency(trace, fs=fs)
    spec_centroid = seisjax.spectral_centroid(trace, fs=fs)
    spec_bandwidth = seisjax.spectral_bandwidth(trace, fs=fs)
    spec_slope = seisjax.spectral_slope(trace, fs=fs)
    spec_rolloff = seisjax.spectral_rolloff(trace, fs=fs)
    spec_flatness = seisjax.spectral_flatness(trace, fs=fs)
    
    # Get power spectrum for visualization
    freqs, psd = seisjax.power_spectrum(trace, fs=fs)
    
    end_time = time.time()
    print(f"Spectral attributes computed in {end_time - start_time:.4f} seconds")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(freqs, psd)
    axes[0, 0].axvline(dom_freq, color='red', linestyle='--', label=f'Dominant: {dom_freq:.1f} Hz')
    axes[0, 0].axvline(spec_centroid, color='green', linestyle='--', label=f'Centroid: {spec_centroid:.1f} Hz')
    axes[0, 0].set_title('Power Spectrum')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Power')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Bar plot of spectral attributes
    attributes = ['Dominant Freq', 'Centroid', 'Bandwidth', 'Rolloff']
    values = [dom_freq, spec_centroid, spec_bandwidth, spec_rolloff]
    
    axes[0, 1].bar(attributes, values, color=['red', 'green', 'blue', 'orange'])
    axes[0, 1].set_title('Spectral Attributes')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Additional attributes
    attrs_2 = ['Slope', 'Flatness']
    values_2 = [spec_slope, spec_flatness]
    
    axes[1, 0].bar(attrs_2, values_2, color=['purple', 'brown'])
    axes[1, 0].set_title('Additional Spectral Attributes')
    axes[1, 0].set_ylabel('Value')
    
    # Spectrogram (for demonstration)
    axes[1, 1].specgram(np.array(trace), Fs=fs, cmap='viridis')
    axes[1, 1].set_title('Spectrogram')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig('spectral_attributes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Spectral attributes visualization saved as 'spectral_attributes.png'")
    
    # Print summary
    print(f"\nSpectral Attribute Summary:")
    print(f"  Dominant Frequency: {dom_freq:.2f} Hz")
    print(f"  Spectral Centroid: {spec_centroid:.2f} Hz")
    print(f"  Spectral Bandwidth: {spec_bandwidth:.2f} Hz")
    print(f"  Spectral Slope: {spec_slope:.4f}")
    print(f"  Spectral Rolloff: {spec_rolloff:.2f} Hz")
    print(f"  Spectral Flatness: {spec_flatness:.4f}")


def demo_geometric_attributes(data: Dict[str, jnp.ndarray]) -> None:
    """Demonstrate geometric attributes."""
    print("\n=== Geometric Attributes Demo ===")
    
    seismic_3d = data['seismic_3d']
    
    start_time = time.time()
    
    # Compute geometric attributes
    dip_mag = seisjax.dip_magnitude(seismic_3d)
    dip_azim = seisjax.dip_azimuth(seismic_3d)
    true_dip = seisjax.true_dip(seismic_3d)
    fault_like = seisjax.fault_likelihood(seismic_3d, sigma=1.0)
    edges = seisjax.edge_detection(seismic_3d[:, :, 32])  # 2D slice for edge detection
    
    end_time = time.time()
    print(f"Geometric attributes computed in {end_time - start_time:.4f} seconds")
    
    # Visualize
    mid_slice = seismic_3d.shape[0] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    im1 = axes[0, 0].imshow(seismic_3d[mid_slice], cmap='seismic', aspect='auto')
    axes[0, 0].set_title('Original Seismic')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(dip_mag[mid_slice], cmap='hot', aspect='auto')
    axes[0, 1].set_title('Dip Magnitude')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(dip_azim[mid_slice], cmap='hsv', aspect='auto')
    axes[0, 2].set_title('Dip Azimuth')
    plt.colorbar(im3, ax=axes[0, 2])
    
    im4 = axes[1, 0].imshow(true_dip[mid_slice], cmap='plasma', aspect='auto')
    axes[1, 0].set_title('True Dip')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(fault_like[mid_slice], cmap='Reds', aspect='auto')
    axes[1, 1].set_title('Fault Likelihood')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(edges, cmap='gray', aspect='auto')
    axes[1, 2].set_title('Edge Detection')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('geometric_attributes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Geometric attributes visualization saved as 'geometric_attributes.png'")


def benchmark_performance() -> None:
    """Benchmark JAX vs NumPy performance."""
    print("\n=== Performance Benchmark ===")
    
    # Generate test data
    key = jax.random.PRNGKey(42)
    test_data = jax.random.normal(key, (1000,))
    
    # JAX version (compiled)
    jax_envelope = jax.jit(seisjax.envelope)
    
    # Warm up JIT compilation
    _ = jax_envelope(test_data)
    
    # Benchmark JAX
    start_time = time.time()
    for _ in range(100):
        result_jax = jax_envelope(test_data)
    jax_time = time.time() - start_time
    
    # NumPy version for comparison
    def numpy_envelope(x):
        from scipy.signal import hilbert
        return np.abs(hilbert(np.array(x)))
    
    # Benchmark NumPy
    start_time = time.time()
    for _ in range(100):
        result_numpy = numpy_envelope(test_data)
    numpy_time = time.time() - start_time
    
    speedup = numpy_time / jax_time
    
    print(f"NumPy time (100 iterations): {numpy_time:.4f} seconds")
    print(f"JAX time (100 iterations): {jax_time:.4f} seconds")
    print(f"Speedup: {speedup:.2f}x faster with JAX")
    
    # Verify results are similar
    error = np.mean(np.abs(np.array(result_jax) - result_numpy))
    print(f"Mean absolute error: {error:.2e}")


def main():
    """Main demonstration function."""
    print("SeisJAX Comprehensive Attributes Demonstration")
    print("=" * 50)
    
    # Check JAX setup
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    # Generate synthetic data
    data = generate_synthetic_data()
    
    # Run all demonstrations
    demo_complex_trace_attributes(data)
    demo_curvature_attributes(data)
    demo_coherence_attributes(data)
    demo_spectral_attributes(data)
    demo_geometric_attributes(data)
    
    # Performance benchmark
    benchmark_performance()
    
    print("\n" + "=" * 50)
    print("All demonstrations completed successfully!")
    print("Check the generated PNG files for visualizations.")
    print("\nSeisJAX provides JAX-accelerated implementations of:")
    print("  • Complex trace attributes")
    print("  • Coherence attributes") 
    print("  • Curvature attributes")
    print("  • Spectral attributes")
    print("  • Geometric attributes")
    print("\nConversion from d2geo framework complete!")


if __name__ == "__main__":
    main() 