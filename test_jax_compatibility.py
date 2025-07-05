#!/usr/bin/env python3
"""
Test script to verify JAX compatibility of SeisJAX functions.
"""

import numpy as np
import jax.numpy as jnp
import jax
from jax import random
import sys
import os

# Add the current directory to the path so we can import seisjax
sys.path.insert(0, os.getcwd())

try:
    import seisjax
    print("✅ Successfully imported seisjax")
except ImportError as e:
    print(f"❌ Failed to import seisjax: {e}")
    exit(1)

def test_basic_functionality():
    """Test basic SeisJAX functionality."""
    print("\n🧪 Testing Basic Functionality...")
    
    # Create test data
    key = random.PRNGKey(42)
    test_data_1d = random.normal(key, (100,))
    test_data_2d = random.normal(key, (50, 30))
    test_data_3d = random.normal(key, (20, 15, 10))
    
    try:
        # Test 1D envelope
        env_1d = seisjax.envelope(test_data_1d, axis=0)
        print(f"✅ 1D envelope: {env_1d.shape}")
        
        # Test 2D envelope
        env_2d = seisjax.envelope(test_data_2d, axis=0)
        print(f"✅ 2D envelope: {env_2d.shape}")
        
        # Test 3D envelope  
        env_3d = seisjax.envelope(test_data_3d, axis=0)
        print(f"✅ 3D envelope: {env_3d.shape}")
        
        # Test instantaneous phase
        phase_1d = seisjax.instantaneous_phase(test_data_1d, axis=0)
        print(f"✅ 1D instantaneous phase: {phase_1d.shape}")
        
        # Test instantaneous frequency
        freq_1d = seisjax.instantaneous_frequency(test_data_1d, axis=0)
        print(f"✅ 1D instantaneous frequency: {freq_1d.shape}")
        
        print("✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_axes():
    """Test functions with different axis values."""
    print("\n🔄 Testing Different Axes...")
    
    # Create 3D test data
    key = random.PRNGKey(123)
    test_data = random.normal(key, (20, 15, 10))
    
    try:
        # Test axis=0
        env_axis0 = seisjax.envelope(test_data, axis=0)
        print(f"✅ Envelope axis=0: {env_axis0.shape}")
        
        # Test axis=1
        env_axis1 = seisjax.envelope(test_data, axis=1)
        print(f"✅ Envelope axis=1: {env_axis1.shape}")
        
        # Test axis=2
        env_axis2 = seisjax.envelope(test_data, axis=2)
        print(f"✅ Envelope axis=2: {env_axis2.shape}")
        
        # Test axis=-1
        env_axis_neg1 = seisjax.envelope(test_data, axis=-1)
        print(f"✅ Envelope axis=-1: {env_axis_neg1.shape}")
        
        print("✅ All axis tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Axis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test performance with timing."""
    print("\n⚡ Testing Performance...")
    
    # Create larger test data
    key = random.PRNGKey(456)
    large_data = random.normal(key, (100, 50, 30))
    
    try:
        import time
        
        # Time envelope computation
        start_time = time.time()
        env_large = seisjax.envelope(large_data, axis=0)
        end_time = time.time()
        
        print(f"✅ Large envelope computation: {env_large.shape}")
        print(f"⏱️  Time taken: {end_time - start_time:.4f} seconds")
        
        # Time instantaneous phase
        start_time = time.time()
        phase_large = seisjax.instantaneous_phase(large_data, axis=0)
        end_time = time.time()
        
        print(f"✅ Large phase computation: {phase_large.shape}")
        print(f"⏱️  Time taken: {end_time - start_time:.4f} seconds")
        
        print("✅ Performance test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spectral_functions():
    """Test spectral decomposition functions."""
    print("\n🌈 Testing Spectral Functions...")
    
    try:
        # Create test data
        key = random.PRNGKey(789)
        test_data = random.normal(key, (100, 32))
        fs = 250.0
        
        # Test spectral decomposition
        frequencies, times, stft_result = seisjax.spectral_decomposition(
            test_data, 
            fs=fs,
            window_length=0.1,
            hop_length=2,
            axis=0
        )
        
        print(f"✅ Spectral decomposition: {stft_result.shape}")
        print(f"📊 Frequencies: {len(frequencies)}, Times: {len(times)}")
        
        # Test RGB frequency blend
        rgb_blend = seisjax.rgb_frequency_blend(
            test_data,
            fs=fs,
            freq_red=15.0,
            freq_green=35.0,
            freq_blue=60.0,
            axis=0
        )
        
        print(f"✅ RGB frequency blend: {rgb_blend.shape}")
        print("✅ All spectral tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Spectral test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting SeisJAX JAX Compatibility Tests")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")
    
    tests = [
        test_basic_functionality,
        test_different_axes,
        test_performance,
        test_spectral_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SeisJAX is JAX-compatible!")
        return 0
    else:
        print("😞 Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main()) 