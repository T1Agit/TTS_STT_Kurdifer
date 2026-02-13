#!/usr/bin/env python3
"""
Test TTS improvements: intonation changes and voice presets
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_intonation_parameters():
    """Test that _generate_segment_audio accepts punctuation parameter"""
    print("=" * 70)
    print("Intonation Parameters Test")
    print("=" * 70)
    
    try:
        from vits_tts_service import VitsTTSService
        import inspect
        
        # Check that _generate_segment_audio has punctuation parameter
        sig = inspect.signature(VitsTTSService._generate_segment_audio)
        params = list(sig.parameters.keys())
        
        print(f"  _generate_segment_audio parameters: {params}")
        
        if 'punctuation' in params:
            print("  ✅ punctuation parameter exists")
            
            # Check default value
            default = sig.parameters['punctuation'].default
            if default == '':
                print(f"  ✅ punctuation default value is correct: '{default}'")
                return True
            else:
                print(f"  ❌ punctuation default value incorrect: '{default}' (expected '')")
                return False
        else:
            print("  ❌ punctuation parameter missing")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_voice_preset_method():
    """Test that _apply_voice_preset method exists"""
    print("\n" + "=" * 70)
    print("Voice Preset Method Test")
    print("=" * 70)
    
    try:
        from vits_tts_service import VitsTTSService
        
        if hasattr(VitsTTSService, '_apply_voice_preset'):
            print("  ✅ _apply_voice_preset method exists")
            
            # Check method signature
            import inspect
            sig = inspect.signature(VitsTTSService._apply_voice_preset)
            params = list(sig.parameters.keys())
            
            print(f"  Parameters: {params}")
            
            required_params = ['self', 'audio_segment', 'preset']
            if all(p in params for p in required_params):
                print("  ✅ All required parameters present")
                return True
            else:
                print(f"  ❌ Missing parameters. Expected: {required_params}")
                return False
        else:
            print("  ❌ _apply_voice_preset method missing")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generate_speech_signature():
    """Test that generate_speech has voice_preset parameter"""
    print("\n" + "=" * 70)
    print("Generate Speech Signature Test")
    print("=" * 70)
    
    try:
        from vits_tts_service import VitsTTSService
        import inspect
        
        sig = inspect.signature(VitsTTSService.generate_speech)
        params = list(sig.parameters.keys())
        
        print(f"  generate_speech parameters: {params}")
        
        if 'voice_preset' in params:
            print("  ✅ voice_preset parameter exists")
            
            # Check default value
            default = sig.parameters['voice_preset'].default
            if default == 'default':
                print(f"  ✅ voice_preset default value is correct: '{default}'")
                return True
            else:
                print(f"  ❌ voice_preset default value incorrect: '{default}' (expected 'default')")
                return False
        else:
            print("  ❌ voice_preset parameter missing")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_voice_preset_logic():
    """Test voice preset application logic (without actual audio generation)"""
    print("\n" + "=" * 70)
    print("Voice Preset Logic Test")
    print("=" * 70)
    
    try:
        from vits_tts_service import VitsTTSService
        from pydub import AudioSegment
        import io
        
        # Create a mock audio segment (1 second of silence at 16000 Hz)
        service = VitsTTSService.__new__(VitsTTSService)
        mock_audio = AudioSegment.silent(duration=1000, frame_rate=16000)
        
        print(f"  Original audio: duration={len(mock_audio)}ms, frame_rate={mock_audio.frame_rate}")
        
        # Test default preset (should return unchanged)
        result_default = service._apply_voice_preset(mock_audio, 'default')
        if len(result_default) == len(mock_audio) and result_default.frame_rate == mock_audio.frame_rate:
            print("  ✅ 'default' preset: no change")
        else:
            print("  ❌ 'default' preset: unexpected change")
            return False
        
        # Test elderly_male preset
        result_male = service._apply_voice_preset(mock_audio, 'elderly_male')
        print(f"  'elderly_male': duration={len(result_male)}ms, frame_rate={result_male.frame_rate}")
        if result_male.frame_rate == 16000:
            print("  ✅ 'elderly_male' preset: frame_rate correct")
        else:
            print(f"  ❌ 'elderly_male' preset: frame_rate incorrect ({result_male.frame_rate})")
            return False
        
        # Test elderly_female preset
        result_female = service._apply_voice_preset(mock_audio, 'elderly_female')
        print(f"  'elderly_female': duration={len(result_female)}ms, frame_rate={result_female.frame_rate}")
        if result_female.frame_rate == 16000:
            print("  ✅ 'elderly_female' preset: frame_rate correct")
        else:
            print(f"  ❌ 'elderly_female' preset: frame_rate incorrect ({result_female.frame_rate})")
            return False
        
        # Test unknown preset (should return unchanged)
        result_unknown = service._apply_voice_preset(mock_audio, 'unknown')
        if len(result_unknown) == len(mock_audio):
            print("  ✅ 'unknown' preset: no change (as expected)")
        else:
            print("  ❌ 'unknown' preset: unexpected change")
            return False
        
        print("\n  ✅ All voice preset logic tests passed")
        return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_integration():
    """Test that API integration has voice_preset support"""
    print("\n" + "=" * 70)
    print("API Integration Test")
    print("=" * 70)
    
    try:
        # Check api_server.py
        with open('api_server.py', 'r') as f:
            api_content = f.read()
        
        if 'voice_preset' in api_content:
            print("  ✅ api_server.py contains voice_preset")
        else:
            print("  ❌ api_server.py missing voice_preset")
            return False
        
        # Check tts_stt_service_base44.py
        with open('tts_stt_service_base44.py', 'r') as f:
            service_content = f.read()
        
        if 'voice_preset' in service_content:
            print("  ✅ tts_stt_service_base44.py contains voice_preset")
        else:
            print("  ❌ tts_stt_service_base44.py missing voice_preset")
            return False
        
        # Check index.html
        with open('index.html', 'r') as f:
            html_content = f.read()
        
        if 'voicePreset' in html_content and 'elderly_male' in html_content:
            print("  ✅ index.html contains voicePreset selector")
        else:
            print("  ❌ index.html missing voicePreset selector")
            return False
        
        print("\n  ✅ All API integration checks passed")
        return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    
    results.append(("Intonation Parameters", test_intonation_parameters()))
    results.append(("Voice Preset Method", test_voice_preset_method()))
    results.append(("Generate Speech Signature", test_generate_speech_signature()))
    results.append(("Voice Preset Logic", test_voice_preset_logic()))
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
        print("🎉 All TTS improvement tests passed!")
    else:
        print("❌ Some tests failed. Please review the errors above.")
    print("=" * 70)
    
    sys.exit(0 if all_passed else 1)
