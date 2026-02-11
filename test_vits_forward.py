#!/usr/bin/env python3
"""
Test script to understand VitsModel forward pass behavior
"""

import torch
from transformers import VitsModel, VitsTokenizer

# Load model and tokenizer
print("Loading model...")
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kmr-script_latin")
model = VitsModel.from_pretrained("facebook/mms-tts-kmr-script_latin")

# Test text
text = "Silav"

# Tokenize
print(f"\nTokenizing text: {text}")
input_ids = tokenizer(text, return_tensors="pt").input_ids
print(f"Input IDs shape: {input_ids.shape}")
print(f"Input IDs: {input_ids}")

# Test forward pass in eval mode
print("\n=== Testing forward pass in eval mode ===")
model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids)
    print(f"Output type: {type(outputs)}")
    print(f"Output attributes: {dir(outputs)}")
    if hasattr(outputs, 'waveform'):
        print(f"Waveform shape: {outputs.waveform.shape}")
    if hasattr(outputs, 'spectrogram'):
        print(f"Spectrogram shape: {outputs.spectrogram.shape}")

# Test forward pass in train mode
print("\n=== Testing forward pass in train mode ===")
model.train()
try:
    outputs = model(input_ids=input_ids)
    print(f"✅ Forward pass succeeded in train mode")
    print(f"Output type: {type(outputs)}")
    if hasattr(outputs, 'waveform'):
        print(f"Waveform shape: {outputs.waveform.shape}")
        print(f"Waveform requires_grad: {outputs.waveform.requires_grad}")
except Exception as e:
    print(f"❌ Forward pass failed in train mode: {e}")

# Test if we can pass labels
print("\n=== Testing with labels parameter ===")
try:
    fake_mel = torch.randn(1, 80, 100)  # [batch, n_mels, time]
    outputs = model(input_ids=input_ids, labels=fake_mel)
    print(f"✅ Forward pass with labels succeeded")
    print(f"Has loss attribute: {hasattr(outputs, 'loss')}")
except Exception as e:
    print(f"❌ Forward pass with labels failed: {e}")

print("\n=== Done ===")
