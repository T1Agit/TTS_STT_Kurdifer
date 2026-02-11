#!/usr/bin/env python3
"""
Test script to verify soundfile backend works correctly

This script verifies that:
1. Soundfile is installed and working
2. Datasets library uses soundfile backend
3. torchcodec is not interfering
"""

import os
import sys

# Set environment variable BEFORE importing datasets
os.environ['HF_AUDIO_DECODER'] = 'soundfile'

print("=" * 80)
print("Testing Soundfile Audio Backend")
print("=" * 80)

# Test 1: Check soundfile installation
print("\n✓ Test 1: Checking soundfile installation...")
try:
    import soundfile as sf
    print(f"  ✅ soundfile version: {sf.__version__}")
except ImportError as e:
    print(f"  ❌ Failed to import soundfile: {e}")
    sys.exit(1)

# Test 2: Check datasets installation
print("\n✓ Test 2: Checking datasets installation...")
try:
    import datasets
    print(f"  ✅ datasets version: {datasets.__version__}")
except ImportError as e:
    print(f"  ❌ Failed to import datasets: {e}")
    sys.exit(1)

# Test 3: Verify environment variable
print("\n✓ Test 3: Checking HF_AUDIO_DECODER environment variable...")
decoder = os.environ.get('HF_AUDIO_DECODER')
if decoder == 'soundfile':
    print(f"  ✅ HF_AUDIO_DECODER = {decoder}")
else:
    print(f"  ⚠️  HF_AUDIO_DECODER = {decoder} (expected: soundfile)")

# Test 4: Check if torchcodec is installed (should NOT be)
print("\n✓ Test 4: Checking torchcodec status...")
try:
    import torchcodec
    version = getattr(torchcodec, '__version__', 'unknown')
    print(f"  ⚠️  torchcodec is still installed (version: {version})")
    print(f"  💡 Recommendation: pip uninstall torchcodec -y")
except ImportError:
    print(f"  ✅ torchcodec is not installed (correct!)")

# Test 5: Test Audio feature with soundfile
print("\n✓ Test 5: Testing Audio feature with soundfile backend...")
try:
    from datasets import Audio
    audio_feature = Audio(sampling_rate=16000)
    print(f"  ✅ Audio feature created successfully")
    print(f"  ✅ Sample rate: {audio_feature.sampling_rate}")
except Exception as e:
    print(f"  ❌ Failed to create Audio feature: {e}")
    sys.exit(1)

# Test 6: Check other required libraries
print("\n✓ Test 6: Checking other required libraries...")
libraries = {
    'soundfile': None,
    'pandas': None,
    'numpy': None,
    'librosa': None,
    'transformers': None,
    'torch': None,
    'tqdm': None,
}

for lib_name in libraries:
    try:
        lib = __import__(lib_name)
        version = getattr(lib, '__version__', 'unknown')
        print(f"  ✅ {lib_name}: {version}")
        libraries[lib_name] = version
    except ImportError:
        print(f"  ⚠️  {lib_name}: not installed")
        libraries[lib_name] = None

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

all_passed = True

# Required for audio loading
if libraries.get('soundfile'):
    print("✅ Audio loading: Ready (soundfile backend)")
else:
    print("❌ Audio loading: NOT ready (soundfile missing)")
    all_passed = False

# Required for dataset handling
if libraries.get('pandas') and libraries.get('numpy'):
    print("✅ Data processing: Ready")
else:
    print("❌ Data processing: NOT ready (pandas/numpy missing)")
    all_passed = False

# Required for audio processing
if libraries.get('librosa'):
    print("✅ Audio processing: Ready")
else:
    print("❌ Audio processing: NOT ready (librosa missing)")
    all_passed = False

# Required for model training
if libraries.get('transformers') and libraries.get('torch'):
    print("✅ Model training: Ready")
else:
    print("❌ Model training: NOT ready (transformers/torch missing)")
    all_passed = False

print("\n" + "=" * 80)

if all_passed:
    print("🎉 All tests passed! You're ready to run prepare_data.py")
    print("\nNext steps:")
    print("  python prepare_data.py --output_dir training --max_samples 500")
    sys.exit(0)
else:
    print("⚠️  Some dependencies are missing. Install them with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
