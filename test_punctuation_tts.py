#!/usr/bin/env python3
"""
Test punctuation-aware TTS functionality
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_text_splitting():
    """Test the text splitting on punctuation"""
    print("=" * 70)
    print("Text Splitting Test")
    print("=" * 70)
    
    try:
        from vits_tts_service import VitsTTSService
        
        # Create service instance (without initializing models)
        service = VitsTTSService.__new__(VitsTTSService)
        
        # Test cases
        test_cases = [
            ("Silav, tu çawa yî? Ez baş im.", 
             [("Silav", ","), ("tu çawa yî", "?"), ("Ez baş im", ".")]),
            ("Hello world", 
             [("Hello world", "")]),
            ("Test. Another sentence.", 
             [("Test", "."), ("Another sentence", ".")]),
            ("First, second; third: fourth! Fifth? End.", 
             [("First", ","), ("second", ";"), ("third", ":"), ("fourth", "!"), ("Fifth", "?"), ("End", ".")]),
            ("", 
             []),
            ("   ", 
             []),
        ]
        
        all_passed = True
        for text, expected in test_cases:
            print(f"\nTest: '{text}'")
            result = service._split_text_on_punctuation(text)
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
            
            if result == expected:
                print("  ✅ PASSED")
            else:
                print("  ❌ FAILED")
                all_passed = False
        
        if all_passed:
            print("\n" + "=" * 70)
            print("✅ All text splitting tests passed!")
            print("=" * 70)
            return True
        else:
            print("\n" + "=" * 70)
            print("❌ Some text splitting tests failed!")
            print("=" * 70)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_silence_duration():
    """Test silence duration mapping"""
    print("\n" + "=" * 70)
    print("Silence Duration Test")
    print("=" * 70)
    
    try:
        from vits_tts_service import VitsTTSService
        
        # Create service instance (without initializing models)
        service = VitsTTSService.__new__(VitsTTSService)
        
        # Expected silence durations
        expected_durations = {
            '.': 500,
            '?': 500,
            '!': 500,
            ',': 250,
            ';': 350,
            ':': 300,
            '': 0,
            'x': 0,  # Unknown punctuation
        }
        
        all_passed = True
        for punctuation, expected_duration in expected_durations.items():
            result = service._get_silence_duration(punctuation)
            status = "✅" if result == expected_duration else "❌"
            print(f"  {status} '{punctuation}' -> {result}ms (expected: {expected_duration}ms)")
            
            if result != expected_duration:
                all_passed = False
        
        if all_passed:
            print("\n" + "=" * 70)
            print("✅ All silence duration tests passed!")
            print("=" * 70)
            return True
        else:
            print("\n" + "=" * 70)
            print("❌ Some silence duration tests failed!")
            print("=" * 70)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_method_presence():
    """Test that required methods exist"""
    print("\n" + "=" * 70)
    print("Method Presence Test")
    print("=" * 70)
    
    try:
        from vits_tts_service import VitsTTSService
        
        required_methods = [
            '_split_text_on_punctuation',
            '_get_silence_duration',
            '_generate_segment_audio',
            '_preprocess_and_generate',
            'generate_speech'
        ]
        
        all_passed = True
        for method_name in required_methods:
            if hasattr(VitsTTSService, method_name):
                print(f"  ✅ {method_name} exists")
            else:
                print(f"  ❌ {method_name} missing")
                all_passed = False
        
        if all_passed:
            print("\n" + "=" * 70)
            print("✅ All required methods present!")
            print("=" * 70)
            return True
        else:
            print("\n" + "=" * 70)
            print("❌ Some required methods missing!")
            print("=" * 70)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    
    results.append(("Method Presence", test_method_presence()))
    results.append(("Text Splitting", test_text_splitting()))
    results.append(("Silence Duration", test_silence_duration()))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Please review the errors above.")
    print("=" * 70)
    
    sys.exit(0 if all_passed else 1)
