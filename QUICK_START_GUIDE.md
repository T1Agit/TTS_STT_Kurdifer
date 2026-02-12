# Quick Start Guide: Optimized VITS Training

## Prerequisites

### System Requirements
- **OS:** Windows (tested), Linux (should work)
- **GPU:** NVIDIA RTX 2070 8GB or similar (8GB VRAM minimum)
- **RAM:** 16GB minimum (8-9GB used during training)
- **Python:** 3.8+ (tested with 3.14)
- **PyTorch:** 2.0+ with CUDA support

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify PyTorch with CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Step 1: Prepare Data

```bash
# Download and prepare Kurdish Common Voice dataset
# This creates training/wavs/ and training/metadata.csv
python prepare_data.py

# Optional: Limit to fewer samples for testing
python prepare_data.py --max_samples 1000
```

**Expected output:**
- ~42,139 WAV files in `training/wavs/`
- Metadata file at `training/metadata.csv`
- Total size: ~5.21 GB

## Step 2: Run Training

### Quick Test (100 samples, ~2 minutes)
```bash
python train_vits.py --max_samples 100 --benchmark_samples 10
```

### Full Training (9.5 hours target)
```bash
python train_vits.py
```

### Custom Training Time
```bash
# 5 hours
python train_vits.py --target_hours 5.0

# 12 hours (overnight)
python train_vits.py --target_hours 12.0
```

### Adjust for Different GPU Memory
```bash
# For 4-6 GB VRAM
python train_vits.py --batch_size 4 --gradient_accumulation_steps 8

# For 12+ GB VRAM
python train_vits.py --batch_size 16 --gradient_accumulation_steps 2
```

## What to Expect

### Initial Loading (2-3 minutes)
```
📦 Loading model and tokenizer...
✅ Model loaded successfully
   Parameters: 36.00M
   Trainable parameters: XX.XXM

📊 Loading dataset from training...
📝 Loading metadata...
✅ Found 42139 samples
🔄 Pre-loading all audio into RAM (using soundfile)...
[Progress bar showing WAV loading]
✅ Loaded 42139 samples into RAM (5210.00 MB)
🔄 Pre-computing mel spectrograms...
[Progress bar showing mel computation]
✅ Pre-computed mel spectrograms (2500.00 MB)
📊 Total RAM usage: 7710.00 MB (7.53 GB)
```

### Benchmarking (30-60 seconds)
```
⏱️  Benchmarking training speed on 25 batches...
✅ Benchmark complete: 12.50 samples/sec
   Total samples: 200
   Elapsed time: 16.00s
```

### Training Plan
```
📊 TRAINING PLAN
======================================================================
Total samples: 42,139
Samples per epoch: 42,139
Benchmark speed: 12.50 samples/sec
Estimated time per epoch: 56.2 minutes
Target training time: 9.5 hours (570.0 minutes)
Auto-calculated epochs: 10
Estimated total time: 9.37 hours
======================================================================
```

### Training Progress
```
🚀 Starting training for 10 epochs...
======================================================================

📍 Epoch 1/10
Epoch 1/10: 100%|████████| 5267/5267 [56:12<00:00, 1.56it/s, loss=18.4521, samples/s=12.3, VRAM=4.32/8.00GB, ETA=0.0m]
✅ Epoch 1 complete
   Average loss: 18.9234
   Samples/sec: 12.48
   Elapsed time: 0.94 hours
   Total ETA: 8.43 hours
💾 Saved checkpoint to training/checkpoints/checkpoint_epoch_1.pt
💎 New best model saved! Loss: 18.9234

📍 Epoch 2/10
[Similar progress...]
```

## Monitoring

### Key Metrics to Watch

1. **Loss:** Should decrease over time
   - Initial: ~19.0-20.0
   - After 1 epoch: ~18.0-19.0
   - After 10 epochs: ~16.0-17.0 (expected)

2. **Samples/sec:** Should be 10-15+
   - <5: Problem with GPU utilization
   - 10-15: Expected performance
   - >15: Great performance!

3. **VRAM:** Should be 3-5 GB
   - <1 GB: GPU underutilized (check batch size)
   - 3-5 GB: Good utilization
   - >7 GB: Risk of OOM, reduce batch size

4. **ETA:** Should match your target
   - Check total ETA after epoch 1
   - Adjust `--target_hours` if needed

## Output Files

After training completes, you'll have:

```
training/
├── checkpoints/
│   ├── checkpoint_epoch_1.pt          # Epoch 1 checkpoint
│   ├── checkpoint_epoch_2.pt          # Epoch 2 checkpoint
│   ├── ...
│   ├── checkpoint_epoch_10.pt         # Final epoch checkpoint
│   └── best_model/                    # Best model by loss ⭐
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.json
└── final_model/                       # Final model (last epoch)
    ├── config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.json
```

**Which model to use?**
- **For production:** Use `best_model/` (lowest loss)
- **For checkpointing:** Use `final_model/` (latest state)

## Testing the Model

```python
from transformers import VitsModel, VitsTokenizer
import torch

# Load the trained model
model_path = "training/checkpoints/best_model"
model = VitsModel.from_pretrained(model_path)
tokenizer = VitsTokenizer.from_pretrained(model_path)

# Generate speech
text = "Silav, çawa yî?"  # Kurdish: "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    waveform = outputs.waveform.squeeze().cpu().numpy()

# Save to file
import soundfile as sf
sf.write("output.wav", waveform, 16000)
print("✅ Generated output.wav")
```

## Troubleshooting

### Problem: "Out of RAM"
```bash
# Solution: Disable mel pre-computation
# Edit train_vits.py line 694: precompute_mels=False
```

### Problem: "Out of VRAM" / CUDA OOM
```bash
# Solution: Reduce batch size
python train_vits.py --batch_size 4 --gradient_accumulation_steps 8
```

### Problem: "Very slow (< 5 samples/sec)"
- Check GPU is being used: `nvidia-smi`
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Try increasing batch size: `--batch_size 16`

### Problem: "FileNotFoundError: metadata.csv"
```bash
# Solution: Run data preparation first
python prepare_data.py
```

### Problem: "Model not learning (loss not decreasing)"
- Check learning rate: Try `--learning_rate 1e-5` (lower) or `--learning_rate 5e-5` (higher)
- Check batch size: Try larger effective batch size
- Verify data quality: Check `training/metadata.csv` for correct format

## Advanced Usage

### Resume from Checkpoint
```python
# Load checkpoint and continue training
checkpoint = torch.load("training/checkpoints/checkpoint_epoch_5.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# Continue training...
```

### Feedback Training
```bash
# After collecting user feedback
# Place corrected audio in training/feedback/
python train_feedback.py
```

## Performance Benchmarks

### Expected on RTX 2070 8GB:
- **Loading time:** 2-3 minutes
- **Benchmark time:** 30-60 seconds
- **Training speed:** 10-15 samples/sec
- **Epoch time:** 50-70 minutes
- **Total time (10 epochs):** 8-12 hours

### If you see different results:
- Much slower: Check GPU utilization, reduce batch size, or check disk speed
- Much faster: Great! You can increase batch size or target more epochs

## Getting Help

1. **Check logs:** Training prints detailed error messages
2. **Review documentation:**
   - `OPTIMIZED_TRAINING_README.md` - Full documentation
   - `VITS_IMPLEMENTATION_SUMMARY.md` - Implementation details
3. **Common issues:** See troubleshooting section above
4. **GPU monitoring:** Use `nvidia-smi -l 1` to watch GPU usage

## Next Steps

1. ✅ Prepare data: `python prepare_data.py`
2. ✅ Test training: `python train_vits.py --max_samples 100`
3. ✅ Full training: `python train_vits.py`
4. ✅ Monitor progress: Watch loss, samples/sec, VRAM
5. ✅ Test model: Load `best_model/` and generate speech
6. ✅ Collect feedback: Use `train_feedback.py` for improvements

**Happy training!** 🚀
