#!/usr/bin/env python3
"""
Comprehensive test suite for all SeisJAX functions.
This script tests every function in every module to ensure they work correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os
import traceback

# Add the seisjax directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    import seisjax
    print("✅ SeisJAX imported successfully")
except ImportError as e:
    print(f"❌ Failed to import SeisJAX: {e}")
    sys.exit(1)

def create_test_data():
    """Create various test data shapes for comprehensive testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Create synthetic seismic-like data
    t = jnp.linspace(0, 1, 100)
    frequency = 25.0  # Hz
    
    # 1D signal
    signal_1d = jnp.sin(2 * jnp.pi * frequency * t) + 0.1 * jnp.sin(2 * jnp.pi * frequency * 3 * t)
    
    # 2D seismic section (traces x time)
    signal_2d = jnp.array([
        signal_1d + 0.1 * jnp.sin(2 * jnp.pi * frequency * 0.5 * t + i * 0.1) 
        for i in range(50)
    ])
    
    # 3D seismic volume (inline x crossline x time)
    signal_3d = jnp.array([
        [
            signal_1d + 0.1 * jnp.sin(2 * jnp.pi * frequency * 0.5 * t + i * 0.1 + j * 0.05) 
            for j in range(30)
        ] 
        for i in range(20)
    ])
    
    return {
        '1d': signal_1d,
        '2d': signal_2d, 
        '3d': signal_3d,
        'fs': 250.0,  # sampling frequency
        't': t
    }

def test_attributes_module():
    """Test all functions in the attributes module."""
    print("\n🧪 Testing Attributes Module...")
    test_data = create_test_data()
    
    functions_to_test = [
        ('analytic_signal', {'axis': -1}),
        ('envelope', {'axis': -1}),
        ('instantaneous_phase', {'axis': -1}),
        ('cosine_instantaneous_phase', {'axis': -1}),
        ('instantaneous_frequency', {'axis': -1, 'fs': test_data['fs']}),
        ('instantaneous_bandwidth', {'axis': -1})
    ]
    
    passed = 0
    total = len(functions_to_test) * 3  # 3 data shapes
    
    for func_name, kwargs in functions_to_test:
        func = getattr(seisjax, func_name)
        
        for data_type in ['1d', '2d', '3d']:
            try:
                data = test_data[data_type]
                result = func(data, **kwargs)
                
                # Check that result has the right shape
                assert result.shape == data.shape, f"Shape mismatch for {func_name} on {data_type}"
                assert not jnp.any(jnp.isnan(result)), f"NaN values in {func_name} on {data_type}"
                
                print(f"    ✅ {func_name} on {data_type}: {result.shape}")
                passed += 1
                
            except Exception as e:
                print(f"    ❌ {func_name} on {data_type}: {e}")
    
    print(f"  Attributes Module: {passed}/{total} tests passed")
    return passed == total

def test_spectral_module():
    """Test all functions in the spectral module."""
    print("\n🧪 Testing Spectral Module...")
    test_data = create_test_data()
    
    # Functions that return same shape as input
    same_shape_functions = [
        ('spectral_flatness', {'axis': -1, 'fs': test_data['fs']}),
        ('zero_crossing_rate', {'axis': -1}),
        ('spectral_energy', {'axis': -1, 'fs': test_data['fs']})
    ]
    
    # Functions that return reduced shape (along specified axis)
    reduced_shape_functions = [
        ('dominant_frequency', {'axis': -1, 'fs': test_data['fs']}),
        ('spectral_centroid', {'axis': -1, 'fs': test_data['fs']}),
        ('spectral_bandwidth', {'axis': -1, 'fs': test_data['fs']}),
        ('spectral_slope', {'axis': -1, 'fs': test_data['fs']}),
        ('spectral_rolloff', {'axis': -1, 'fs': test_data['fs']}),
        ('peak_frequency', {'axis': -1, 'fs': test_data['fs']})
    ]
    
    # Functions that return tuple (frequencies, spectrum)
    tuple_functions = [
        ('power_spectrum', {'axis': -1, 'fs': test_data['fs']}),
        ('spectral_decomposition', {'axis': -1, 'fs': test_data['fs']})
    ]
    
    # Special functions
    special_functions = [
        ('spectral_flux', {'axis': -1, 'fs': test_data['fs']}),
        ('spectral_contrast', {'axis': -1, 'fs': test_data['fs']}),
        ('rgb_frequency_blend', {'axis': -1, 'fs': test_data['fs'], 'freq_red': 10, 'freq_green': 30, 'freq_blue': 60})
    ]
    
    passed = 0
    total = 0
    
    # Test same shape functions
    for func_name, kwargs in same_shape_functions:
        func = getattr(seisjax, func_name)
        
        for data_type in ['1d', '2d', '3d']:
            total += 1
            try:
                data = test_data[data_type]
                result = func(data, **kwargs)
                assert result.shape == data.shape, f"Shape mismatch for {func_name}"
                print(f"    ✅ {func_name} on {data_type}: {result.shape}")
                passed += 1
            except Exception as e:
                print(f"    ❌ {func_name} on {data_type}: {e}")
    
    # Test reduced shape functions  
    for func_name, kwargs in reduced_shape_functions:
        func = getattr(seisjax, func_name)
        
        for data_type in ['1d', '2d', '3d']:
            total += 1
            try:
                data = test_data[data_type]
                result = func(data, **kwargs)
                expected_shape = data.shape[:-1]  # Remove last dimension
                assert result.shape == expected_shape, f"Shape mismatch for {func_name}"
                print(f"    ✅ {func_name} on {data_type}: {result.shape}")
                passed += 1
            except Exception as e:
                print(f"    ❌ {func_name} on {data_type}: {e}")
    
    # Test tuple functions
    for func_name, kwargs in tuple_functions:
        func = getattr(seisjax, func_name)
        
        for data_type in ['1d', '2d', '3d']:
            total += 1
            try:
                data = test_data[data_type]
                freqs, spectrum = func(data, **kwargs)
                assert len(freqs.shape) == 1, f"Frequency array should be 1D for {func_name}"
                print(f"    ✅ {func_name} on {data_type}: freqs {freqs.shape}, spectrum {spectrum.shape}")
                passed += 1
            except Exception as e:
                print(f"    ❌ {func_name} on {data_type}: {e}")
    
    # Test special functions
    for func_name, kwargs in special_functions:
        func = getattr(seisjax, func_name)
        
        for data_type in ['1d', '2d', '3d']:
            total += 1
            try:
                data = test_data[data_type]
                result = func(data, **kwargs)
                print(f"    ✅ {func_name} on {data_type}: {result.shape}")
                passed += 1
            except Exception as e:
                print(f"    ❌ {func_name} on {data_type}: {e}")
    
    print(f"  Spectral Module: {passed}/{total} tests passed")
    return passed == total

def test_geometric_module():
    """Test all functions in the geometric module."""
    print("\n🧪 Testing Geometric Module...")
    test_data = create_test_data()
    
    # Most geometric functions work on 2D/3D data and require spatial derivatives
    data_2d = test_data['2d']
    data_3d = test_data['3d']
    
    functions_to_test = [
        'dip_magnitude', 'dip_azimuth', 'strike_angle', 'true_dip', 'apparent_dip',
        'reflection_intensity', 'relative_acoustic_impedance', 'convergence',
        'parallelism', 'continuity', 'fault_likelihood', 'edge_detection',
        'texture_energy', 'local_structural_entropy'
    ]
    
    passed = 0
    total = len(functions_to_test) * 2  # 2D and 3D data
    
    for func_name in functions_to_test:
        func = getattr(seisjax, func_name)
        
        # Test on 2D data
        try:
            result_2d = func(data_2d)
            assert not jnp.any(jnp.isnan(result_2d)), f"NaN values in {func_name} on 2D"
            print(f"    ✅ {func_name} on 2D: {result_2d.shape}")
            passed += 1
        except Exception as e:
            print(f"    ❌ {func_name} on 2D: {e}")
        
        # Test on 3D data
        try:
            result_3d = func(data_3d)
            assert not jnp.any(jnp.isnan(result_3d)), f"NaN values in {func_name} on 3D"
            print(f"    ✅ {func_name} on 3D: {result_3d.shape}")
            passed += 1
        except Exception as e:
            print(f"    ❌ {func_name} on 3D: {e}")
    
    print(f"  Geometric Module: {passed}/{total} tests passed")
    return passed == total

def test_coherence_module():
    """Test all functions in the coherence module."""
    print("\n🧪 Testing Coherence Module...")
    test_data = create_test_data()
    
    # Coherence functions typically work on 3D volumes
    data_3d = test_data['3d']
    
    functions_to_test = [
        'semblance', 'eigenstructure_coherence', 'energy_ratio_coherence',
        'c3_coherence', 'variance_coherence'
    ]
    
    passed = 0
    total = len(functions_to_test)
    
    for func_name in functions_to_test:
        try:
            func = getattr(seisjax, func_name)
            result = func(data_3d)
            assert not jnp.any(jnp.isnan(result)), f"NaN values in {func_name}"
            print(f"    ✅ {func_name}: {result.shape}")
            passed += 1
        except Exception as e:
            print(f"    ❌ {func_name}: {e}")
    
    print(f"  Coherence Module: {passed}/{total} tests passed")
    return passed == total

def test_curvature_module():
    """Test all functions in the curvature module."""
    print("\n🧪 Testing Curvature Module...")
    test_data = create_test_data()
    
    # Curvature functions work on 2D data only (surfaces)
    data_2d = test_data['2d']
    
    functions_to_test = [
        'mean_curvature', 'gaussian_curvature', 'maximum_curvature', 'minimum_curvature',
        'curvedness', 'shape_index', 'dip_curvature', 'strike_curvature',
        'contour_curvature', 'profile_curvature', 'planform_curvature',
        'most_positive_curvature', 'most_negative_curvature', 'total_curvature',
        'tangential_curvature'
    ]
    
    # Special function that returns tuple
    tuple_function = 'principal_curvatures'
    
    passed = 0
    total = len(functions_to_test) + 1  # regular functions on 2D + tuple function on 2D
    
    # Test regular functions on 2D data only
    for func_name in functions_to_test:
        func = getattr(seisjax, func_name)
        
        try:
            result_2d = func(data_2d)
            assert not jnp.any(jnp.isnan(result_2d)), f"NaN values in {func_name} on 2D"
            print(f"    ✅ {func_name} on 2D: {result_2d.shape}")
            passed += 1
        except Exception as e:
            print(f"    ❌ {func_name} on 2D: {e}")
    
    # Test tuple function
    func = getattr(seisjax, tuple_function)
    try:
        k1_2d, k2_2d = func(data_2d)
        print(f"    ✅ {tuple_function} on 2D: k1 {k1_2d.shape}, k2 {k2_2d.shape}")
        passed += 1
    except Exception as e:
        print(f"    ❌ {tuple_function} on 2D: {e}")
    
    print(f"  Curvature Module: {passed}/{total} tests passed")
    return passed == total

def test_utils_module():
    """Test utility functions."""
    print("\n🧪 Testing Utils Module...")
    test_data = create_test_data()
    
    passed = 0
    total = 0
    
    # Test some utility functions that are publicly accessible
    try:
        # Test bandpass filter
        from seisjax.utils import bandpass_filter
        data = test_data['1d']
        filtered = bandpass_filter(data, low_freq=10, high_freq=50, fs=test_data['fs'])
        assert filtered.shape == data.shape
        print(f"    ✅ bandpass_filter: {filtered.shape}")
        passed += 1
        total += 1
    except Exception as e:
        print(f"    ❌ bandpass_filter: {e}")
        total += 1
    
    try:
        # Test rescale_volume
        from seisjax.utils import rescale_volume
        data = test_data['3d']
        rescaled = rescale_volume(data, percentiles=(2.0, 98.0))
        assert rescaled.shape == data.shape
        print(f"    ✅ rescale_volume: {rescaled.shape}")
        passed += 1
        total += 1
    except Exception as e:
        print(f"    ❌ rescale_volume: {e}")
        total += 1
    
    try:
        # Test normalize_volume
        from seisjax.utils import normalize_volume
        data = test_data['3d']
        normalized = normalize_volume(data)
        assert normalized.shape == data.shape
        print(f"    ✅ normalize_volume: {normalized.shape}")
        passed += 1
        total += 1
    except Exception as e:
        print(f"    ❌ normalize_volume: {e}")
        total += 1
    
    print(f"  Utils Module: {passed}/{total} tests passed")
    return passed == total

def test_different_axes():
    """Test functions with different axis parameters."""
    print("\n🧪 Testing Different Axis Values...")
    test_data = create_test_data()
    
    data_3d = test_data['3d']  # Shape: (20, 30, 100)
    
    axis_functions = [
        ('envelope', {}),
        ('instantaneous_phase', {}),
        ('instantaneous_frequency', {'fs': test_data['fs']}),
        ('dominant_frequency', {'fs': test_data['fs']}),
        ('spectral_centroid', {'fs': test_data['fs']})
    ]
    
    passed = 0
    total = len(axis_functions) * 3  # 3 different axes
    
    for func_name, extra_kwargs in axis_functions:
        func = getattr(seisjax, func_name)
        
        for axis in [0, 1, 2]:
            try:
                result = func(data_3d, axis=axis, **extra_kwargs)
                print(f"    ✅ {func_name} axis={axis}: {result.shape}")
                passed += 1
            except Exception as e:
                print(f"    ❌ {func_name} axis={axis}: {e}")
    
    print(f"  Different Axes: {passed}/{total} tests passed")
    return passed == total

def test_performance():
    """Test performance and JIT compilation."""
    print("\n🧪 Testing Performance and JIT...")
    test_data = create_test_data()
    
    # Test JIT compilation with a few representative functions
    data_3d = test_data['3d']
    
    functions_to_time = [
        ('envelope', {'axis': -1}),
        ('dominant_frequency', {'axis': -1, 'fs': test_data['fs']}),
        ('mean_curvature', {}),
        ('semblance', {})
    ]
    
    for func_name, kwargs in functions_to_time:
        try:
            func = getattr(seisjax, func_name)
            
            # First call for JIT compilation
            _ = func(data_3d, **kwargs)
            
            # Timed calls
            start_time = time.time()
            for _ in range(5):
                result = func(data_3d, **kwargs)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 5
            print(f"    ✅ {func_name}: {avg_time:.4f}s avg, shape {result.shape}")
            
        except Exception as e:
            print(f"    ❌ {func_name}: {e}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting Comprehensive SeisJAX Function Tests...")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    
    test_results = []
    
    # Run all test modules
    test_results.append(test_attributes_module())
    test_results.append(test_spectral_module())
    test_results.append(test_geometric_module())
    test_results.append(test_coherence_module())
    test_results.append(test_curvature_module())
    test_results.append(test_utils_module())
    test_results.append(test_different_axes())
    
    # Performance test (always runs but doesn't affect pass/fail)
    test_performance()
    
    passed_modules = sum(test_results)
    total_modules = len(test_results)
    
    print(f"\n🏆 Final Results: {passed_modules}/{total_modules} modules passed all tests")
    
    if passed_modules == total_modules:
        print("🎉 All SeisJAX functions are working correctly!")
        return True
    else:
        print("❌ Some functions failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)