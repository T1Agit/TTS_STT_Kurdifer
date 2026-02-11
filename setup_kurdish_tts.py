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
        print("✅ Coqui TTS package is installed")
        return True
    except ImportError:
        print("❌ Coqui TTS package is not installed")
        print("\n📦 Installing Coqui TTS...")
        print("   Run: pip install coqui-tts")
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
    
    # Check for fine-tuned model first
    model_dirs = ["models/kurdish", "./models/kurdish"]
    fine_tuned_found = False
    
    for model_dir in model_dirs:
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            print(f"\n✅ Found fine-tuned Kurdish model at: {model_dir}")
            print("   Your Kurdish TTS is ready to use!")
            fine_tuned_found = True
            break
    
    if fine_tuned_found:
        print("\n💡 The service will use your fine-tuned model for Kurdish TTS.")
        return True
    
    # No fine-tuned model, provide instructions
    print("\n⚠️  No fine-tuned Kurdish model found.")
    print("\n📋 You have two options:")
    print("\n   Option 1: Train a fine-tuned Kurdish model (Recommended)")
    print("   --------------------------------------------------------")
    print("   1. Download Mozilla Common Voice Kurdish corpus:")
    print("      https://datacollective.mozillafoundation.org/datasets/cmj8u3pbq00dtnxxbz4yoxc4i")
    print("   2. Extract to: cv-corpus-24.0-2025-12-05-kmr/")
    print("   3. Run training script:")
    print("      python train_kurdish_xtts.py --corpus_path cv-corpus-24.0-2025-12-05-kmr/cv-corpus-24.0-2025-12-05/kmr/")
    print("   4. Wait for training to complete (~2-4 hours on RTX 2070)")
    print("   5. Model will be saved to: models/kurdish/")
    print("\n   Option 2: Use voice cloning fallback (Works immediately)")
    print("   --------------------------------------------------------")
    print("   The service will use Turkish phonetics as a proxy for Kurdish.")
    print("   This works but may not sound as natural as a fine-tuned model.")
    print("   No additional setup needed - just start using the service!")
    
    try:
        from TTS.api import TTS
        
        print("\n🔧 Downloading base XTTS v2 model...")
        print("   Model: tts_models/multilingual/multi-dataset/xtts_v2")
        print("\n⏳ This may take a few minutes (~2GB download)...\n")
        
        # Initialize TTS with the multilingual model
        # This will download the model if it's not already cached
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=True
        )
        
        print("\n✅ Base XTTS v2 model downloaded!")
        print("\n📊 Model Information:")
        print(f"   Model name: {tts.model_name}")
        print(f"   Cache directory: {tts.manager.output_prefix}")
        
        print("\n" + "=" * 70)
        print("✅ Setup complete!")
        print("=" * 70)
        print("\n💡 Usage:")
        print("   from tts_stt_service_base44 import TTSSTTServiceBase44")
        print("   service = TTSSTTServiceBase44()")
        print("   result = service.text_to_speech_base44('Silav', 'kurdish')")
        print("\n🌐 Current mode: Voice cloning fallback (Turkish phonetics)")
        print("   For better results, train a fine-tuned model with train_kurdish_xtts.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you have enough disk space (~2GB)")
        print("   2. Check your internet connection")
        print("   3. Try running: pip install --upgrade coqui-tts")
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
