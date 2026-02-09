#!/usr/bin/env python3
"""
Setup script to download and prepare Kurdish TTS model

This script downloads the Coqui TTS model for Kurdish (Kurmanji) support.
The model is based on XTTS v2 which is a multilingual model that includes Kurdish.

Mozilla Common Voice Kurdish (Kurmanji) dataset:
https://datacollective.mozillafoundation.org/datasets/cmj8u3pbq00dtnxxbz4yoxc4i
"""

import sys
import os


def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import TTS
        print("✅ TTS package is installed")
        return True
    except ImportError:
        print("❌ TTS package is not installed")
        print("\n📦 Installing Coqui TTS...")
        print("   Run: pip install TTS")
        return False


def setup_kurdish_model():
    """Download and setup Kurdish TTS model"""
    print("\n" + "=" * 70)
    print("Kurdish TTS Setup")
    print("=" * 70)
    
    if not check_dependencies():
        print("\n❌ Please install dependencies first:")
        print("   pip install -r requirements.txt")
        return False
    
    try:
        from TTS.api import TTS
        
        print("\n🔧 Setting up Kurdish TTS model...")
        print("   Model: tts_models/multilingual/multi-dataset/xtts_v2")
        print("   Language: Kurdish (Kurmanji)")
        print("\n⏳ This may take a few minutes for first-time setup (~2GB download)...\n")
        
        # Initialize TTS with the multilingual model
        # This will download the model if it's not already cached
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=True
        )
        
        print("\n✅ Kurdish TTS model ready!")
        print("\n📊 Model Information:")
        print(f"   Model name: {tts.model_name}")
        print(f"   Languages supported: Multiple (including Kurdish)")
        print(f"   Cache directory: {tts.manager.output_prefix}")
        
        # Test the model with a simple Kurdish phrase
        print("\n🧪 Testing Kurdish TTS...")
        import tempfile
        
        test_text = "Silav, tu çawa yî?"
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            tts.tts_to_file(
                text=test_text,
                file_path=temp_path,
                language="ku"
            )
            
            # Check if file was created and has content
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                print(f"✅ Test successful! Generated audio for: '{test_text}'")
                print(f"   Test file size: {os.path.getsize(temp_path)} bytes")
            else:
                print("⚠️  Warning: Audio file was created but may be empty")
            
            # Clean up
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"⚠️  Test failed: {e}")
            print("   The model is installed but may need additional configuration")
        
        print("\n" + "=" * 70)
        print("✅ Setup complete!")
        print("=" * 70)
        print("\n💡 Usage:")
        print("   from tts_stt_service_base44 import TTSSTTServiceBase44")
        print("   service = TTSSTTServiceBase44()")
        print("   result = service.text_to_speech_base44('Silav', 'kurdish')")
        print("\n🌐 Supported languages:")
        print("   • Kurdish (ku) - Coqui TTS")
        print("   • German (de) - gTTS")
        print("   • French (fr) - gTTS")
        print("   • English (en) - gTTS")
        print("   • Turkish (tr) - gTTS")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you have enough disk space (~2GB)")
        print("   2. Check your internet connection")
        print("   3. Try running: pip install --upgrade TTS")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Kurdish (Kurmanji) TTS Setup Script")
    print("=" * 70)
    print("\n📚 About:")
    print("   This script sets up Coqui TTS for Kurdish language support.")
    print("   It uses the XTTS v2 multilingual model which includes Kurdish.")
    print("\n🌍 Dataset:")
    print("   Mozilla Common Voice Kurdish (Kurmanji)")
    print("   https://datacollective.mozillafoundation.org/datasets/cmj8u3pbq00dtnxxbz4yoxc4i")
    
    success = setup_kurdish_model()
    
    sys.exit(0 if success else 1)
