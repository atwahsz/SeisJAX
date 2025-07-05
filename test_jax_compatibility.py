#!/usr/bin/env python3
"""
Test script to verify JAX compatibility fixes for SeisJAX library.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os

# Add the seisjax directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'seisjax'))

try:
    import seisjax
    print("✅ SeisJAX imported successfully")
except ImportError as e:
    print(f"❌ Failed to import SeisJAX: {e}")
    sys.exit(1)

def test_shape_consistency():
    """Test that conditional logic handles different input shapes correctly."""
    print("\n🧪 Testing shape consistency fixes...")
    
    # Test with different input shapes to verify both branches work
    test_shapes = [(2,), (10,), (100,), (1000,)]
    
    for shape in test_shapes:
        print(f"  Testing shape {shape}...")
        
        # Create test data
        test_data = jnp.ones(shape)
        
        try:
            # Test envelope (uses _hilbert_jax -> _unwrap_1d)
            envelope_result = seisjax.envelope(test_data, axis=0)
            assert envelope_result.shape == shape, f"Envelope shape mismatch: {envelope_result.shape} vs {shape}"
            
            # Test instantaneous phase (uses _jax_unwrap -> _unwrap_1d)
            phase_result = seisjax.instantaneous_phase(test_data, axis=0)
            assert phase_result.shape == shape, f"Phase shape mismatch: {phase_result.shape} vs {shape}"
            
            # Test instantaneous frequency (uses _jax_gradient -> _gradient_1d)
            freq_result = seisjax.instantaneous_frequency(test_data, axis=0, fs=1000.0)
            assert freq_result.shape == shape, f"Frequency shape mismatch: {freq_result.shape} vs {shape}"
            
            print(f"    ✅ Shape {shape} passed all tests")
            
        except Exception as e:
            print(f"    ❌ Shape {shape} failed: {e}")
            return False
    
    return True

def test_3d_volume():
    """Test with 3D volume similar to seismic data."""
    print("\n🧪 Testing 3D volume processing...")
    
    # Create a small 3D volume
    volume = jnp.ones((10, 10, 50))
    
    try:
        print("  Testing envelope...")
        envelope_vol = seisjax.envelope(volume, axis=2)
        assert envelope_vol.shape == volume.shape
        print("    ✅ Envelope test passed")
        
        print("  Testing instantaneous phase...")
        phase_vol = seisjax.instantaneous_phase(volume, axis=2)
        assert phase_vol.shape == volume.shape
        print("    ✅ Phase test passed")
        
        print("  Testing instantaneous frequency...")
        freq_vol = seisjax.instantaneous_frequency(volume, axis=2, fs=1000.0)
        assert freq_vol.shape == volume.shape
        print("    ✅ Frequency test passed")
        
        print("  Testing cosine instantaneous phase...")
        cos_phase_vol = seisjax.cosine_instantaneous_phase(volume, axis=2)
        assert cos_phase_vol.shape == volume.shape
        print("    ✅ Cosine phase test passed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 3D volume test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality with 1D, 2D, and 3D inputs."""
    print("\n🧪 Testing basic functionality...")
    
    # Test 1D
    x1d = jnp.sin(jnp.linspace(0, 10*jnp.pi, 100))
    
    try:
        envelope_1d = seisjax.envelope(x1d, axis=0)
        phase_1d = seisjax.instantaneous_phase(x1d, axis=0)
        freq_1d = seisjax.instantaneous_frequency(x1d, axis=0, fs=100.0)
        
        print("  ✅ 1D tests passed")
        
        # Test 2D
        x2d = jnp.sin(jnp.linspace(0, 10*jnp.pi, 100).reshape(10, 10))
        
        envelope_2d = seisjax.envelope(x2d, axis=1)
        phase_2d = seisjax.instantaneous_phase(x2d, axis=1)
        freq_2d = seisjax.instantaneous_frequency(x2d, axis=1, fs=100.0)
        
        print("  ✅ 2D tests passed")
        
        # Test 3D
        x3d = jnp.sin(jnp.linspace(0, 10*jnp.pi, 1000).reshape(10, 10, 10))
        
        envelope_3d = seisjax.envelope(x3d, axis=2)
        phase_3d = seisjax.instantaneous_phase(x3d, axis=2)
        freq_3d = seisjax.instantaneous_frequency(x3d, axis=2, fs=100.0)
        
        print("  ✅ 3D tests passed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test performance with timing."""
    print("\n🧪 Testing performance...")
    
    # Create test data
    volume = jnp.ones((20, 20, 100))
    
    # Warm up JIT
    _ = seisjax.envelope(volume, axis=2)
    
    # Time the computation
    start_time = time.time()
    envelope_result = seisjax.envelope(volume, axis=2)
    phase_result = seisjax.instantaneous_phase(volume, axis=2)
    freq_result = seisjax.instantaneous_frequency(volume, axis=2, fs=1000.0)
    end_time = time.time()
    
    print(f"  ⏱️  Computed 3 attributes on {volume.shape} volume in {end_time - start_time:.3f} seconds")
    print(f"  📊 Results shapes: {envelope_result.shape}, {phase_result.shape}, {freq_result.shape}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting JAX compatibility tests...")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    
    tests = [
        test_shape_consistency,
        test_basic_functionality,
        test_3d_volume,
        test_performance
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🏆 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! JAX compatibility fixes are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 