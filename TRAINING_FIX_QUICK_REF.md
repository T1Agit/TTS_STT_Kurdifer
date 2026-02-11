# Training Fix - Quick Reference

## What Was Fixed

The training script was **silently skipping all samples** because `VitsModel.forward()` doesn't accept a `labels` parameter. Every batch threw an error that was caught and ignored.

## Key Changes

### Before (Broken)
```python
# ❌ This doesn't work - VitsModel has no 'labels' parameter
outputs = model(input_ids=input_ids, labels=mel_specs)
loss = outputs.loss  # ❌ This attribute doesn't exist
```

### After (Fixed)
```python
# ✅ Generate waveform, compute mel, calculate loss manually
outputs = model(input_ids=input_ids)
generated_waveform = outputs.waveform
generated_mel = compute_mel(generated_waveform)
target_mel = compute_mel(target_waveform)
loss = F.l1_loss(generated_mel, target_mel)
```

## Quick Test

```bash
# Test the fix works
python test_training_minimal.py

# Expected: "✅ ALL TESTS PASSED!"
```

## Training Now Works

```bash
# Small test (2-3 minutes)
python prepare_data.py --max_samples 100
python train_vits.py --epochs 1 --max_samples 100

# Full training (2-4 hours)
python prepare_data.py
python train_vits.py --epochs 10
```

## Expected Behavior

| Metric | Before (Broken) | After (Fixed) |
|--------|----------------|---------------|
| **Time per epoch** | 0.2 min | 15-30 min |
| **Loss** | 0.0 | 1.0-3.0 → 0.3-0.8 |
| **Samples processed** | 0 (all skipped) | 42,139 |
| **VRAM usage** | 0.14 GB | 4-6 GB |
| **Errors** | Silent | Logged (first 10) |

## Files Modified

- ✅ `train_vits.py` - Fixed training loop
- ✅ `train_feedback.py` - Fixed feedback loop
- ✅ `test_training_minimal.py` - Validation test (NEW)
- ✅ `TRAINING_BUG_FIX.md` - Full documentation (NEW)
- ✅ `TRAINING_FIX_QUICK_REF.md` - This file (NEW)

## Troubleshooting

### Loss is still 0.0
- Run `test_training_minimal.py` to diagnose
- Check error messages (not silent anymore)

### OOM (Out of Memory)
```bash
python train_vits.py --batch_size 2  # Reduce from 4 to 2
```

### Takes too long
```bash
# Test on subset first
python train_vits.py --max_samples 100
```

## More Info

- Full details: `TRAINING_BUG_FIX.md`
- Training guide: `VITS_TRAINING_README.md`
- Test script: `test_training_minimal.py`
