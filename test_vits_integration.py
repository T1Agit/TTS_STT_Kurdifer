#!/usr/bin/env python3
"""
Simple test to verify VITS service structure without requiring model downloads.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all necessary modules can be imported"""
    print("=" * 70)
    print("VITS Service Import Test")
    print("=" * 70)
    
    try:
        print("\n1. Importing vits_tts_service...")
        from vits_tts_service import VitsTTSService
        print("   ✅ VitsTTSService imported successfully")
        
        print("\n2. Checking class attributes...")
        service = VitsTTSService.__new__(VitsTTSService)  # Don't call __init__
        assert hasattr(VitsTTSService, 'MODELS'), "Missing MODELS attribute"
        assert 'original' in VitsTTSService.MODELS, "Missing 'original' model"
        assert 'trained_v8' in VitsTTSService.MODELS, "Missing 'trained_v8' model"
        print("   ✅ Model configuration looks good")
        
        print("\n3. Checking methods...")
        methods = [
            'list_available_models',
            'generate_speech',
            'verify_kurdish_chars'
        ]
        for method in methods:
            assert hasattr(VitsTTSService, method), f"Missing method: {method}"
            print(f"   ✅ {method} exists")
        
        print("\n4. Testing model info...")
        models = VitsTTSService.MODELS
        for version, info in models.items():
            print(f"   - {version}: {info['description']}")
        
        print("\n" + "=" * 70)
        print("✅ All import tests passed!")
        print("=" * 70)
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except AssertionError as e:
        print(f"   ❌ Assertion error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def test_service_integration():
    """Test TTS service integration"""
    print("\n" + "=" * 70)
    print("TTS Service Integration Test")
    print("=" * 70)
    
    try:
        print("\n1. Importing main TTS service...")
        from tts_stt_service_base44 import TTSSTTServiceBase44
        print("   ✅ TTSSTTServiceBase44 imported successfully")
        
        print("\n2. Checking VITS integration methods...")
        service = TTSSTTServiceBase44.__new__(TTSSTTServiceBase44)
        methods = [
            '_get_vits_service',
            '_generate_speech_vits'
        ]
        for method in methods:
            assert hasattr(TTSSTTServiceBase44, method), f"Missing method: {method}"
            print(f"   ✅ {method} exists")
        
        print("\n3. Checking text_to_speech_base44 signature...")
        import inspect
        sig = inspect.signature(TTSSTTServiceBase44.text_to_speech_base44)
        params = list(sig.parameters.keys())
        print(f"   Parameters: {params}")
        assert 'model_version' in params, "Missing model_version parameter"
        assert 'use_vits' in params, "Missing use_vits parameter"
        print("   ✅ Method signature updated correctly")
        
        print("\n" + "=" * 70)
        print("✅ All integration tests passed!")
        print("=" * 70)
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except AssertionError as e:
        print(f"   ❌ Assertion error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def test_api_integration():
    """Test API server integration"""
    print("\n" + "=" * 70)
    print("API Server Integration Test")
    print("=" * 70)
    
    try:
        print("\n1. Checking API server code...")
        with open('api_server.py', 'r') as f:
            api_code = f.read()
        
        # Check for model_version support
        assert 'model_version' in api_code, "API doesn't handle model_version"
        print("   ✅ API server has model_version support")
        
        # Check for models endpoint
        assert "@app.route('/models'" in api_code, "Missing /models endpoint"
        print("   ✅ /models endpoint exists")
        
        print("\n2. Checking HTML frontend...")
        with open('index.html', 'r') as f:
            html_code = f.read()
        
        # Check for model selector
        assert 'modelGroup' in html_code, "Missing model selector in HTML"
        assert 'trained_v8' in html_code, "Missing trained_v8 option"
        print("   ✅ HTML has model selection UI")
        
        print("\n" + "=" * 70)
        print("✅ All API integration tests passed!")
        print("=" * 70)
        return True
        
    except FileNotFoundError as e:
        print(f"   ❌ File not found: {e}")
        return False
    except AssertionError as e:
        print(f"   ❌ Assertion error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    results = []
    
    results.append(("Import Tests", test_imports()))
    results.append(("Service Integration", test_service_integration()))
    results.append(("API Integration", test_api_integration()))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All tests passed! The integration is ready.")
        print("\nNext steps:")
        print("1. Add the trained model files to training/best_model_v8/")
        print("2. Test with actual Kurdish text synthesis")
        print("3. Compare original vs trained_v8 model quality")
    else:
        print("❌ Some tests failed. Please review the errors above.")
    print("=" * 70)
    
    sys.exit(0 if all_passed else 1)
