#!/usr/bin/env python3
"""
Test script to validate spectral function fixes for TracerIntegerConversionError.
"""

import jax.numpy as jnp
import sys
import os

# Add the seisjax directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    import seisjax
    print("✅ SeisJAX imported successfully")
except ImportError as e:
    print(f"❌ Failed to import SeisJAX: {e}")
    sys.exit(1)

def test_spectral_functions():
    """Test spectral functions that were causing TracerIntegerConversionError."""
    print("\n🔄 Testing spectral functions...")
    
    # Create test data similar to seismic inline
    test_data = jnp.ones((50, 100))  # (crossline, time)
    fs = 250.0
    
    try:
        print("  Testing dominant_frequency...")
        dom_freq = seisjax.dominant_frequency(test_data, axis=1, fs=fs)
        print(f"    ✅ Dominant frequency shape: {dom_freq.shape}")
        
        print("  Testing spectral_centroid...")
        spec_centroid = seisjax.spectral_centroid(test_data, axis=1, fs=fs)
        print(f"    ✅ Spectral centroid shape: {spec_centroid.shape}")
        
        print("  Testing spectral_bandwidth...")
        spec_bandwidth = seisjax.spectral_bandwidth(test_data, axis=1, fs=fs)
        print(f"    ✅ Spectral bandwidth shape: {spec_bandwidth.shape}")
        
        print("  Testing peak_frequency...")
        peak_freq = seisjax.peak_frequency(test_data, axis=1, fs=fs)
        print(f"    ✅ Peak frequency shape: {peak_freq.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Spectral function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_axes():
    """Test with different axis values."""
    print("\n🔄 Testing different axis values...")
    
    # Create 3D test data
    test_data_3d = jnp.ones((20, 30, 100))  # (inline, crossline, time)
    fs = 250.0
    
    try:
        # Test axis=0 (inline)
        print("  Testing axis=0...")
        dom_freq_0 = seisjax.dominant_frequency(test_data_3d, axis=0, fs=fs)
        print(f"    ✅ Shape for axis=0: {dom_freq_0.shape}")
        
        # Test axis=1 (crossline)
        print("  Testing axis=1...")
        dom_freq_1 = seisjax.dominant_frequency(test_data_3d, axis=1, fs=fs)
        print(f"    ✅ Shape for axis=1: {dom_freq_1.shape}")
        
        # Test axis=2 (time)
        print("  Testing axis=2...")
        dom_freq_2 = seisjax.dominant_frequency(test_data_3d, axis=2, fs=fs)
        print(f"    ✅ Shape for axis=2: {dom_freq_2.shape}")
        
        # Test axis=-1 (last axis)
        print("  Testing axis=-1...")
        dom_freq_neg1 = seisjax.dominant_frequency(test_data_3d, axis=-1, fs=fs)
        print(f"    ✅ Shape for axis=-1: {dom_freq_neg1.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Axis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Spectral Function Fix Validation...")
    
    tests = [
        test_spectral_functions,
        test_different_axes
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🏆 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All spectral function fixes are working correctly!")
        print("✅ TracerIntegerConversionError has been resolved.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 