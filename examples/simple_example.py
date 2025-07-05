#!/usr/bin/env python3
"""
Simple example demonstrating basic SeisJAX usage.

This script shows how to use SeisJAX for computing seismic attributes
with JAX acceleration, converting from d2geo framework functionality.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seisjax


def main():
    """Simple demonstration of SeisJAX attributes."""
    print("SeisJAX Simple Example - JAX-accelerated seismic attributes")
    print("Based on d2geo framework: https://github.com/dudley-fitzgerald/d2geo")
    print("=" * 60)
    
    # Generate a simple Ricker wavelet
    time = jnp.linspace(-0.1, 0.1, 201)
    freq = 25  # Hz
    ricker = (1 - 2 * (jnp.pi * freq * time)**2) * jnp.exp(-(jnp.pi * freq * time)**2)
    
    print(f"Computing attributes for Ricker wavelet (frequency: {freq} Hz)")
    
    # Complex trace attributes
    envelope = seisjax.envelope(ricker)
    inst_phase = seisjax.instantaneous_phase(ricker)
    inst_freq = seisjax.instantaneous_frequency(ricker, fs=2000.0)
    
    # Spectral attributes
    dom_freq = seisjax.dominant_frequency(ricker, fs=2000.0)
    spec_centroid = seisjax.spectral_centroid(ricker, fs=2000.0)
    spec_bandwidth = seisjax.spectral_bandwidth(ricker, fs=2000.0)
    
    print(f"\nComputed attributes:")
    print(f"  Dominant frequency: {dom_freq:.1f} Hz")
    print(f"  Spectral centroid: {spec_centroid:.1f} Hz")
    print(f"  Spectral bandwidth: {spec_bandwidth:.1f} Hz")
    print(f"  Max envelope: {jnp.max(envelope):.3f}")
    
    # Create a simple 2D horizon for curvature demo
    x = jnp.linspace(-2, 2, 50)
    y = jnp.linspace(-2, 2, 50)
    X, Y = jnp.meshgrid(x, y)
    horizon = jnp.exp(-(X**2 + Y**2)) + 0.3 * jnp.sin(3*X) * jnp.cos(3*Y)
    
    # Curvature attributes
    mean_curv = seisjax.mean_curvature(horizon)
    gaussian_curv = seisjax.gaussian_curvature(horizon)
    
    print(f"\nCurvature analysis:")
    print(f"  Mean curvature range: [{jnp.min(mean_curv):.3f}, {jnp.max(mean_curv):.3f}]")
    print(f"  Gaussian curvature range: [{jnp.min(gaussian_curv):.3f}, {jnp.max(gaussian_curv):.3f}]")
    
    # Create a simple 3D volume for coherence demo
    seismic_3d = jnp.zeros((20, 20, 20))
    
    # Add some layered structure
    for i in range(20):
        seismic_3d = seismic_3d.at[i, :, :].set(jnp.sin(i * 0.5))
    
    # Add noise
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, seismic_3d.shape) * 0.1
    seismic_3d += noise
    
    # Coherence attributes (small window for speed)
    semblance = seisjax.semblance(seismic_3d, window_shape=(3, 3, 3))
    variance_coh = seisjax.variance_coherence(seismic_3d, window_shape=(3, 3, 3))
    
    print(f"\nCoherence analysis:")
    print(f"  Semblance range: [{jnp.min(semblance):.3f}, {jnp.max(semblance):.3f}]")
    print(f"  Variance coherence range: [{jnp.min(variance_coh):.3f}, {jnp.max(variance_coh):.3f}]")
    
    # Geometric attributes
    dip_mag = seisjax.dip_magnitude(seismic_3d)
    dip_azim = seisjax.dip_azimuth(seismic_3d)
    
    print(f"\nGeometric analysis:")
    print(f"  Dip magnitude range: [{jnp.min(dip_mag):.3f}, {jnp.max(dip_mag):.3f}]")
    print(f"  Dip azimuth range: [{jnp.min(dip_azim):.3f}, {jnp.max(dip_azim):.3f}] rad")
    
    # Simple visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot Ricker wavelet and envelope
    axes[0, 0].plot(time, ricker, label='Ricker wavelet', linewidth=2)
    axes[0, 0].plot(time, envelope, label='Envelope', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('Complex Trace Attributes')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot instantaneous frequency
    axes[0, 1].plot(time, inst_freq)
    axes[0, 1].set_title('Instantaneous Frequency')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].grid(True)
    
    # Plot horizon and mean curvature
    im1 = axes[1, 0].contourf(X, Y, horizon, levels=15, cmap='terrain')
    axes[1, 0].set_title('Synthetic Horizon')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Plot mean curvature
    im2 = axes[1, 1].contourf(X, Y, mean_curv, levels=15, cmap='RdBu_r')
    axes[1, 1].set_title('Mean Curvature')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('simple_seisjax_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved as 'simple_seisjax_example.png'")
    print(f"\nSeisJAX provides JAX-accelerated versions of seismic attributes from d2geo!")
    print(f"For more examples, run: python examples/comprehensive_attributes_demo.py")


if __name__ == "__main__":
    main() 