<div align="center">
  <img src="LOGO.png" alt="SeisJAX Logo" width="300"/>
  <p>
    <b>High-Performance Seismic Attribute Computation with JAX</b>
  </p>
</div>

---

## About SeisJAX

**SeisJAX** is a Python library that leverages Google's JAX to provide a high-performance, hardware-agnostic framework for computing seismic attributes. Inspired by the research on accelerating scientific computing with machine learning compilers, SeisJAX offers a simple, NumPy-like API for complex seismic analysis that can run seamlessly on CPUs, GPUs, and TPUs.

This project is a JAX-based re-implementation of the concepts from the [d2geo](https://github.com/dudley-fitzgerald/d2geo) framework, optimized for speed and portability. The [d2geo](https://github.com/dudley-fitzgerald/d2geo) framework provides a comprehensive set of seismic attributes for prospect identification and reservoir characterization. As shown in the paper *Accelerating Seismic Attribute Computation with Machine Learning Compilers*, using ML compilers like JAX can result in speedups of over 100x compared to traditional single-threaded CPU implementations.

## Key Features

- **JIT Compilation**: Functions are just-in-time compiled to native machine code for maximum performance.
- **Hardware Acceleration**: Run the same code on CPU, GPU, or TPU without any changes.
- **NumPy-like API**: If you know NumPy, you already know how to use SeisJAX.
- **Comprehensive Seismic Attributes**: Includes essential attributes like envelope, instantaneous phase, frequency, coherence, curvature, and more.
- **d2geo Compatible**: Provides JAX-accelerated versions of attributes from the [d2geo](https://github.com/dudley-fitzgerald/d2geo) framework.

## Installation

You can install SeisJAX directly from the source or via pip once it's published.

First, ensure you have a version of JAX that matches your hardware (CPU/GPU/TPU). See the [official JAX installation guide](https://github.com/google/jax#installation) for details.

Then, install SeisJAX:

```bash
pip install .
```

## Quick Start

Here's a quick example of how to compute the envelope and instantaneous phase for a random seismic trace.

```python
import jax.numpy as jnp
from seisjax import envelope, instantaneous_phase

# Generate a sample seismic trace (e.g., a Ricker wavelet)
# In a real scenario, you would load your seismic data here.
time = jnp.linspace(-1, 1, 201)
ricker = (1 - 2 * (jnp.pi**2) * (25**2) * (time**2)) * jnp.exp(-(jnp.pi**2) * (25**2) * (time**2))

# Compute attributes
trace_envelope = envelope(ricker)
trace_phase = instantaneous_phase(ricker)

print("Envelope:", trace_envelope)
print("Instantaneous Phase:", trace_phase)
```

## Available Attributes

### Complex Trace Attributes
- `analytic_signal`
- `envelope`
- `instantaneous_phase`
- `cosine_instantaneous_phase`
- `instantaneous_frequency`
- `instantaneous_bandwidth`

### Coherence Attributes
- `semblance`
- `eigenstructure_coherence`
- `energy_ratio_coherence`

### Curvature Attributes  
- `mean_curvature`
- `gaussian_curvature`
- `maximum_curvature`
- `minimum_curvature`

### Spectral Attributes
- `dominant_frequency`
- `spectral_slope`
- `spectral_bandwidth`
- `spectral_decompostion`

### Geometric Attributes
- `dip`
- `azimuth`
- `continuity`
- `convergence`

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This project builds upon the concepts from the [d2geo](https://github.com/dudley-fitzgerald/d2geo) framework for computing seismic attributes. We thank the original authors for their foundational work in seismic attribute computation.
