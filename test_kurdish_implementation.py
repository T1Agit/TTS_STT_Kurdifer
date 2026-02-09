#!/usr/bin/env python3
"""
Test Kurdish TTS implementation without downloading the actual model.
This tests the code structure and routing logic.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tts_stt_service_base44 import TTSSTTServiceBase44


def test_language_routing():
    """Test that language routing works correctly"""
    print("=" * 70)
    print("Testing Kurdish TTS Implementation")
    print("=" * 70)
    
    service = TTSSTTServiceBase44()
    
    # Test 1: Language code mapping
    print("\n📝 Test 1: Language Code Mapping")
    print("-" * 70)
    
    test_cases = [
        ("english", "en"),
        ("kurdish", "ku"),
        ("german", "de"),
        ("french", "fr"),
        ("turkish", "tr"),
        ("en", "en"),
        ("ku", "ku"),
    ]
    
    all_passed = True
    for input_lang, expected_code in test_cases:
        try:
            result = service._get_language_code(input_lang)
            if result == expected_code:
                print(f"✅ {input_lang:12s} → {result}")
            else:
                print(f"❌ {input_lang:12s} → {result} (expected {expected_code})")
                all_passed = False
        except Exception as e:
            print(f"❌ {input_lang:12s} → Error: {e}")
            all_passed = False
    
    # Test 2: TTS Engine Selection
    print("\n🎤 Test 2: TTS Engine Selection")
    print("-" * 70)
    
    engines = [
        ("en", "gTTS", False),
        ("de", "gTTS", False),
        ("fr", "gTTS", False),
        ("tr", "gTTS", False),
        ("ku", "Coqui TTS", True),
    ]
    
    for lang_code, expected_engine, should_use_coqui in engines:
        uses_coqui = service._uses_coqui_tts(lang_code)
        if uses_coqui == should_use_coqui:
            print(f"✅ {lang_code} → {expected_engine}")
        else:
            print(f"❌ {lang_code} → Wrong engine selected")
            all_passed = False
    
    # Test 3: Error Handling
    print("\n⚠️  Test 3: Error Handling")
    print("-" * 70)
    
    # Test invalid language
    try:
        service._get_language_code("invalid_language")
        print("❌ Should have raised error for invalid language")
        all_passed = False
    except ValueError:
        print("✅ Correctly rejected invalid language")
    
    # Test 4: Method Existence
    print("\n🔍 Test 4: Method Existence")
    print("-" * 70)
    
    required_methods = [
        "_get_language_code",
        "_uses_coqui_tts",
        "_generate_speech_coqui",
        "text_to_speech_base44",
    ]
    
    for method_name in required_methods:
        if hasattr(service, method_name):
            print(f"✅ Method exists: {method_name}")
        else:
            print(f"❌ Missing method: {method_name}")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All tests passed!")
        print("=" * 70)
        print("\n📝 Implementation Summary:")
        print("   • Kurdish (ku) routes to Coqui TTS")
        print("   • Other languages (en, de, fr, tr) use gTTS")
        print("   • Language code normalization works correctly")
        print("   • Error handling implemented")
        print("\n⚠️  Note: Actual TTS generation not tested")
        print("   To test full functionality:")
        print("   1. Install: pip install coqui-tts")
        print("   2. Run: python setup_kurdish_tts.py")
        print("   3. Run: python tts_stt_service_base44.py")
        return 0
    else:
        print("❌ Some tests failed!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(test_language_routing())
