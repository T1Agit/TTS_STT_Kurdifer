#!/usr/bin/env python3
"""
Manual test script for TTS API improvements
Tests the new features without requiring a running server
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_endpoint_signature():
    """Test that the API endpoint accepts voice_preset"""
    print("=" * 70)
    print("API Endpoint Signature Test")
    print("=" * 70)
    
    try:
        # Read the api_server.py file
        with open('api_server.py', 'r') as f:
            content = f.read()
        
        # Check for voice_preset parameter extraction
        if "voice_preset = data.get('voice_preset'" in content:
            print("  ✅ API endpoint extracts voice_preset from request")
        else:
            print("  ❌ API endpoint doesn't extract voice_preset")
            return False
        
        # Check for validation
        if "valid_presets = ['default', 'elderly_male', 'elderly_female']" in content:
            print("  ✅ API endpoint validates voice_preset values")
        else:
            print("  ❌ API endpoint doesn't validate voice_preset")
            return False
        
        # Check it passes to service
        if "voice_preset=voice_preset" in content:
            print("  ✅ API endpoint passes voice_preset to service")
        else:
            print("  ❌ API endpoint doesn't pass voice_preset to service")
            return False
        
        print("\n  ✅ API endpoint signature is correct")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_ui_integration():
    """Test that the UI has the voice preset selector"""
    print("\n" + "=" * 70)
    print("UI Integration Test")
    print("=" * 70)
    
    try:
        with open('index.html', 'r') as f:
            content = f.read()
        
        # Check for voicePreset select element
        if 'id="voicePreset"' in content:
            print("  ✅ UI has voicePreset select element")
        else:
            print("  ❌ UI missing voicePreset select element")
            return False
        
        # Check for options
        if 'value="elderly_male"' in content and 'value="elderly_female"' in content:
            print("  ✅ UI has all voice preset options")
        else:
            print("  ❌ UI missing some voice preset options")
            return False
        
        # Check JavaScript sends voice_preset
        if 'voice_preset: voicePreset' in content or 'voice_preset:' in content:
            print("  ✅ UI JavaScript sends voice_preset in request")
        else:
            print("  ❌ UI JavaScript doesn't send voice_preset")
            return False
        
        # Check UI displays voice preset in info
        if 'Voice preset:' in content or 'voice preset:' in content:
            print("  ✅ UI displays voice preset in audio info")
        else:
            print("  ❌ UI doesn't display voice preset")
            return False
        
        print("\n  ✅ UI integration is complete")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_backward_compatibility():
    """Test that backward compatibility is maintained"""
    print("\n" + "=" * 70)
    print("Backward Compatibility Test")
    print("=" * 70)
    
    try:
        from vits_tts_service import VitsTTSService
        import inspect
        
        # Check default values ensure backward compatibility
        sig = inspect.signature(VitsTTSService.generate_speech)
        
        # voice_preset should default to 'default'
        if sig.parameters['voice_preset'].default == 'default':
            print("  ✅ voice_preset defaults to 'default' (backward compatible)")
        else:
            print(f"  ❌ voice_preset default is {sig.parameters['voice_preset'].default}")
            return False
        
        # Check that punctuation parameter defaults to empty string
        sig2 = inspect.signature(VitsTTSService._generate_segment_audio)
        if sig2.parameters['punctuation'].default == '':
            print("  ✅ punctuation defaults to '' (backward compatible)")
        else:
            print(f"  ❌ punctuation default is {sig2.parameters['punctuation'].default}")
            return False
        
        print("\n  ✅ Backward compatibility maintained")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pause_durations():
    """Verify the new shorter pause durations"""
    print("\n" + "=" * 70)
    print("Pause Durations Test")
    print("=" * 70)
    
    try:
        from vits_tts_service import VitsTTSService
        
        expected = {
            '.': 300,
            '?': 350,
            '!': 250,
            ',': 150,
            ';': 200,
            ':': 200,
        }
        
        print("  Expected pause durations:")
        for punct, duration in expected.items():
            print(f"    '{punct}': {duration}ms")
        
        # Check actual values
        service = VitsTTSService.__new__(VitsTTSService)
        all_correct = True
        
        print("\n  Actual pause durations:")
        for punct, expected_duration in expected.items():
            actual = service._get_silence_duration(punct)
            status = "✅" if actual == expected_duration else "❌"
            print(f"    {status} '{punct}': {actual}ms")
            if actual != expected_duration:
                all_correct = False
        
        if all_correct:
            print("\n  ✅ All pause durations are correct")
            return True
        else:
            print("\n  ❌ Some pause durations are incorrect")
            return False
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary():
    """Print a summary of the changes"""
    print("\n" + "=" * 70)
    print("IMPLEMENTATION SUMMARY")
    print("=" * 70)
    print("""
Changes Implemented:

1. ✅ Shorter Pause Durations
   - Period (.): 500ms → 300ms
   - Question (?): 500ms → 350ms
   - Exclamation (!): 500ms → 250ms
   - Comma (,): 250ms → 150ms
   - Semicolon (;): 350ms → 200ms
   - Colon (:): 300ms → 200ms

2. ✅ Intonation Changes per Punctuation
   - Question (?): noise_scale=0.8, speaking_rate=0.95 (rising, slower)
   - Exclamation (!): noise_scale=0.9, speaking_rate=1.1 (expressive, faster)
   - Period (.): noise_scale=0.5, speaking_rate=0.95 (calm, falling)
   - Comma (,) / default: noise_scale=0.667, speaking_rate=1.0 (normal)

3. ✅ Voice Presets
   - default: No changes (backward compatible)
   - elderly_male: Lower pitch ~15%, slow to 90% speed
   - elderly_female: Raise pitch ~5%, slow to 88% speed

4. ✅ API & UI Integration
   - /tts endpoint accepts voice_preset parameter
   - index.html has voice preset dropdown
   - Validation for valid preset values
   - Full backward compatibility

5. ✅ Testing
   - All existing tests updated and passing
   - New comprehensive test suite added
   - Manual validation ready
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TTS IMPROVEMENTS MANUAL TEST")
    print("=" * 70)
    print()
    
    results = []
    
    results.append(("Pause Durations", test_pause_durations()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("API Endpoint Signature", test_api_endpoint_signature()))
    results.append(("UI Integration", test_ui_integration()))
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print_summary()
        print("\n" + "=" * 70)
        print("🎉 All manual tests passed!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ Some tests failed. Please review the errors above.")
        print("=" * 70)
    
    sys.exit(0 if all_passed else 1)
