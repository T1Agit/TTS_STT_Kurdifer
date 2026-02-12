# VITS Training Script - Optimized for 9-10 Hours

## Overview

This optimized training script is designed to efficiently train the VITS/MMS Kurdish TTS model within a 9.5 hour target window, maximizing GPU utilization on an RTX 2070 8GB.

## Key Optimizations

### 1. **RAM Pre-loading** (5.21 GB)
- All 42,139 WAV files are loaded into RAM at startup using `soundfile`
- Eliminates disk I/O bottleneck during training
- Typical load time: ~2-3 minutes

### 2. **Pre-computed Mel Spectrograms**
- Target mel spectrograms are computed once during dataset initialization
- Stored in RAM alongside waveforms
- Reduces computation during training loops

### 3. **Pinned Memory + Non-blocking Transfers**
- Enables asynchronous CPU→GPU data transfer
- Uses `pin_memory=True` in DataLoader
- Adds `non_blocking=True` to `.to(device)` calls
- Maximizes GPU utilization

### 4. **Speed Benchmarking**
- Runs on 200 samples (configurable with `--benchmark_samples`)
- Measures actual samples/sec throughput
- Used for auto-calibration

### 5. **Auto-calculated Epochs**
- Target: 9.5 hours (configurable with `--target_hours`)
- Formula: `epochs = target_time / (samples_per_epoch / samples_per_sec)`
- Ensures training fits within time budget

### 6. **Enhanced Progress Tracking**
- Loss per batch
- Samples/sec (real-time)
- VRAM usage (current/total)
- Epoch ETA
- Total training ETA

### 7. **Best Model Saving**
- Saves checkpoint every epoch
- Tracks best model by loss
- Saves best model separately

### 8. **Cosine LR Schedule with Warmup**
- Linear warmup: 500 steps (configurable with `--warmup_steps`)
- Cosine decay: Smooth reduction to 0
- Better convergence than constant LR

## Usage

```bash
# Basic usage (auto-calibrates for 9.5 hours)
python train_vits.py

# Custom target time (e.g., 5 hours)
python train_vits.py --target_hours 5.0

# Adjust batch size for different VRAM
python train_vits.py --batch_size 16  # For >8GB VRAM
python train_vits.py --batch_size 4   # For <8GB VRAM

# Test on small subset
python train_vits.py --max_samples 1000
```

## Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `training` | Directory with wavs/ and metadata.csv |
| `--checkpoint_dir` | `training/checkpoints` | Checkpoint save directory |
| `--output_dir` | `training/final_model` | Final model save directory |
| `--model_name` | `facebook/mms-tts-kmr-script_latin` | Base model |
| `--batch_size` | `8` | Batch size (increase for better GPU util) |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation |
| `--learning_rate` | `2e-5` | Initial learning rate |
| `--target_hours` | `9.5` | Target training time in hours |
| `--warmup_steps` | `500` | LR warmup steps |
| `--fp16` | `True` | Mixed precision training |
| `--max_samples` | `0` | Limit samples (0 = all) |
| `--benchmark_samples` | `200` | Samples for speed benchmark |

## Expected Performance

### Previous (disk I/O bottleneck):
- Speed: ~1.7 samples/sec
- VRAM: 0.47 GB / 8 GB (6% utilization)
- Time for 1 epoch (42,139 samples): ~6.9 hours

### Optimized (with all improvements):
- **Expected speed: 10-15 samples/sec** (5-10x faster)
- **Expected VRAM: 3-5 GB / 8 GB** (40-60% utilization)
- **Time for 1 epoch: ~1.0-1.5 hours**
- **Epochs in 9.5 hours: 6-9 epochs**

## Output Structure

```
training/
├── checkpoints/
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   ├── ...
│   └── best_model/          # Best model by loss
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer files
├── final_model/             # Final model after all epochs
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
└── wavs/                    # Audio files (pre-loaded into RAM)
```

## Technical Details

### Audio Loading
- **Uses `soundfile`** (NOT `torchaudio.load`)
- Reason: torchcodec is broken on Windows with Python 3.14
- Format: float32, 16kHz mono
- Resampling: librosa (if needed)

### Mel Spectrogram Parameters
- Sample rate: 16,000 Hz
- FFT size: 1,024
- Hop length: 256
- Mel bins: 80
- Scale: Log scale with min clamp at 1e-5

### Memory Requirements
- Audio waveforms: ~5.21 GB
- Mel spectrograms: ~2-3 GB
- Model: ~0.5 GB
- **Total RAM: ~8-9 GB recommended**
- **Total VRAM: 8 GB (RTX 2070)**

## Troubleshooting

### Out of RAM
```bash
# Disable mel pre-computation (saves 2-3 GB RAM)
# Edit train_vits.py line 694: precompute_mels=False
```

### Out of VRAM
```bash
# Reduce batch size
python train_vits.py --batch_size 4

# Increase gradient accumulation
python train_vits.py --gradient_accumulation_steps 8
```

### Too slow/fast
```bash
# Adjust target time
python train_vits.py --target_hours 12.0  # More epochs
python train_vits.py --target_hours 6.0   # Fewer epochs
```

## Files Modified

1. **`train_vits.py`** - Main training script (fully optimized)
2. **`train_feedback.py`** - Feedback training (uses soundfile)
3. **`prepare_data.py`** - Data prep (already uses soundfile)
4. **`requirements.txt`** - Dependencies (no torchcodec)
5. **`.gitignore`** - Excludes training artifacts

## Next Steps

1. **Prepare data**: Run `python prepare_data.py` if not done
2. **Start training**: Run `python train_vits.py`
3. **Monitor progress**: Watch loss, samples/sec, VRAM, and ETA
4. **Test model**: Load from `training/best_model/` or `training/final_model/`
5. **Collect feedback**: Use `train_feedback.py` for incremental improvements

## References

- Base model: [facebook/mms-tts-kmr-script_latin](https://huggingface.co/facebook/mms-tts-kmr-script_latin)
- Dataset: Kurdish Common Voice (via `amedcj/kurmanji-commonvoice`)
- Architecture: VITS (Variational Inference TTS)
- Parameters: 36M (trainable: varies by fine-tuning strategy)
