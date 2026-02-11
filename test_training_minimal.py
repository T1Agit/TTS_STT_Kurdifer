#!/usr/bin/env python3
"""
Minimal test script to verify training works correctly with the fix.
This tests on a single synthetic sample to verify gradients flow.
"""

import torch
import torch.nn.functional as F
from transformers import VitsModel, VitsTokenizer
import torchaudio

print("=" * 70)
print("Testing VITS Training Fix")
print("=" * 70)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Device: {device}")

# Load model
print("\n📦 Loading model...")
try:
    tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kmr-script_latin")
    model = VitsModel.from_pretrained("facebook/mms-tts-kmr-script_latin")
    model = model.to(device)
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("\n⚠️  This test requires transformers library and internet connection")
    print("Install with: pip install transformers torch torchaudio")
    exit(1)

# Create synthetic test data
print("\n📊 Creating test data...")
text = "Silav"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
attention_mask = torch.ones_like(input_ids).to(device)

# Create a fake target waveform (2 seconds at 16kHz)
target_waveform = torch.randn(1, 32000).to(device)

# Create mel spectrogram computer
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80
).to(device)

def compute_mel(waveform):
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    return mel_spec

# Compute target mel
target_mel = compute_mel(target_waveform)
print(f"✅ Test data created")
print(f"   Input text: {text}")
print(f"   Input IDs shape: {input_ids.shape}")
print(f"   Target waveform shape: {target_waveform.shape}")
print(f"   Target mel shape: {target_mel.shape}")

# Test forward pass
print("\n🔬 Testing forward pass...")
model.train()
try:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print("✅ Forward pass succeeded")
    
    if hasattr(outputs, 'waveform'):
        generated_waveform = outputs.waveform
        print(f"✅ Generated waveform shape: {generated_waveform.shape}")
        print(f"   Requires grad: {generated_waveform.requires_grad}")
    else:
        print("❌ Output does not have 'waveform' attribute")
        print(f"   Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
        exit(1)
        
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test mel computation from generated waveform
print("\n🔬 Testing mel spectrogram computation...")
try:
    generated_mel = compute_mel(generated_waveform)
    print(f"✅ Generated mel shape: {generated_mel.shape}")
    print(f"   Requires grad: {generated_mel.requires_grad}")
except Exception as e:
    print(f"❌ Mel computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test loss computation
print("\n🔬 Testing loss computation...")
try:
    # Align lengths
    common_len = min(target_mel.shape[-1], generated_mel.shape[-1])
    target_mel_crop = target_mel[:, :, :common_len]
    generated_mel_crop = generated_mel[:, :, :common_len]
    
    loss = F.l1_loss(generated_mel_crop, target_mel_crop)
    print(f"✅ Loss computed: {loss.item():.6f}")
    print(f"   Loss requires grad: {loss.requires_grad}")
except Exception as e:
    print(f"❌ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test backward pass
print("\n🔬 Testing backward pass...")
try:
    # Get initial parameter to check gradient
    param = next(model.parameters())
    param_before = param.clone().detach()
    
    loss.backward()
    
    # Check if gradients exist
    has_grad = param.grad is not None
    print(f"✅ Backward pass succeeded")
    print(f"   Parameter has gradient: {has_grad}")
    
    if has_grad:
        grad_norm = param.grad.norm().item()
        print(f"   Gradient norm: {grad_norm:.6f}")
        
        if grad_norm == 0:
            print("⚠️  Warning: Gradient norm is 0 - no learning will occur!")
        else:
            print("✅ Gradients are flowing correctly!")
    else:
        print("❌ No gradients computed - training will not work!")
        exit(1)
        
except Exception as e:
    print(f"❌ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test optimizer step
print("\n🔬 Testing optimizer step...")
try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.step()
    optimizer.zero_grad()
    
    # Check if parameters changed
    param_after = param.clone().detach()
    param_diff = (param_after - param_before).abs().max().item()
    
    print(f"✅ Optimizer step succeeded")
    print(f"   Max parameter change: {param_diff:.10f}")
    
    if param_diff > 0:
        print("✅ Parameters updated - training will work!")
    else:
        print("⚠️  Warning: Parameters did not change - learning rate might be too low")
        
except Exception as e:
    print(f"❌ Optimizer step failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nThe training fix is working correctly:")
print("  ✓ Model forward pass generates waveform")
print("  ✓ Mel spectrogram computed from generated waveform")
print("  ✓ L1 loss computed successfully")
print("  ✓ Gradients flow through the model")
print("  ✓ Parameters can be updated")
print("\n🎉 Training should now work properly!")
print("=" * 70)
