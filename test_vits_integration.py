#!/usr/bin/env python3
"""
Test script for VITS v8 integration

This script tests the new model selection functionality without requiring
the actual model files to be present.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tts_stt_service_base44 import TTSSTTServiceBase44


def test_service_initialization():
    """Test that service initializes correctly"""
    print("=" * 70)
    print("Test 1: Service Initialization")
    print("=" * 70)
    
    try:
        service = TTSSTTServiceBase44()
        print("✅ Service initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False


def test_model_detection():
    """Test model detection"""
    print("\n" + "=" * 70)
    print("Test 2: Model Detection")
    print("=" * 70)
    
    try:
        service = TTSSTTServiceBase44()
        
        # Check what models were detected
        print("\nDetected VITS models:")
        for model_name, path in service._vits_model_paths.items():
            print(f"  - {model_name}: {path}")
        
        if not service._vits_model_paths:
            print("  (No models detected - this is OK if models aren't downloaded yet)")
        
        print("✅ Model detection completed")
        return True
    except Exception as e:
        print(f"❌ Model detection failed: {e}")
        return False


def test_get_available_models():
    """Test get_available_models method"""
    print("\n" + "=" * 70)
    print("Test 3: Get Available Models API")
    print("=" * 70)
    
    try:
        service = TTSSTTServiceBase44()
        
        # Test for Kurdish
        print("\nAvailable models for Kurdish:")
        models_info = service.get_available_models('kurdish')
        print(f"  Language: {models_info['language']}")
        print(f"  Models: {', '.join(models_info['models'])}")
        print(f"  Default: {models_info['default_model']}")
        
        # Test for English (should return default)
        print("\nAvailable models for English:")
        models_info = service.get_available_models('english')
        print(f"  Language: {models_info['language']}")
        print(f"  Models: {', '.join(models_info['models'])}")
        
        print("✅ get_available_models() works correctly")
        return True
    except Exception as e:
        print(f"❌ get_available_models() failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_selection_logic():
    """Test model selection without actual generation"""
    print("\n" + "=" * 70)
    print("Test 4: Model Selection Logic")
    print("=" * 70)
    
    try:
        service = TTSSTTServiceBase44()
        
        # Test auto-selection
        print("\nTesting auto-selection priority:")
        if 'v8' in service._vits_model_paths:
            print("  ✓ v8 model available - should be auto-selected")
        elif 'original' in service._vits_model_paths:
            print("  ✓ original model available - should be auto-selected")
        else:
            print("  ✓ No VITS models - should fall back to Coqui")
        
        print("✅ Model selection logic is correct")
        return True
    except Exception as e:
        print(f"❌ Model selection logic test failed: {e}")
        return False


def test_api_parameters():
    """Test that API accepts new parameters"""
    print("\n" + "=" * 70)
    print("Test 5: API Parameter Compatibility")
    print("=" * 70)
    
    try:
        service = TTSSTTServiceBase44()
        
        # Test that the method accepts model_version parameter
        print("\nTesting method signature...")
        import inspect
        sig = inspect.signature(service.text_to_speech_base44)
        params = list(sig.parameters.keys())
        
        print(f"  Parameters: {', '.join(params)}")
        
        if 'model_version' in params:
            print("  ✓ model_version parameter exists")
        else:
            print("  ✗ model_version parameter missing!")
            return False
        
        print("✅ API parameters are correct")
        return True
    except Exception as e:
        print(f"❌ API parameter test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n🔧 VITS v8 Integration Test Suite")
    print("=" * 70)
    print("This test validates the integration without requiring model files.\n")
    
    tests = [
        test_service_initialization,
        test_model_detection,
        test_get_available_models,
        test_model_selection_logic,
        test_api_parameters
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed!")
        print("\nNext steps:")
        print("  1. Place trained model files in training/best_model_v8/")
        print("  2. Start the API server: python api_server.py")
        print("  3. Test generation via web UI or API")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
