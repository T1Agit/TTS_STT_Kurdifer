# Quick Start: VITS/MMS Kurdish TTS Fine-tuning

## One-Line Quick Test (Windows)

```bash
# Install dependencies
pip install -r requirements.txt

# Test with 100 samples (~5 minutes)
python prepare_data.py --max_samples 100 --output_dir training && python train_vits.py --epochs 2 --max_samples 100
```

## Production Training (Full Dataset)

```bash
# Step 1: Prepare all data (~1 hour)
python prepare_data.py --output_dir training

# Step 2: Train model (~8-12 hours on RTX 2070)
python train_vits.py --epochs 20 --batch_size 4

# Step 3: Deploy model
cp -r training/final_model models/kurdish
```

## Feedback Loop (Continuous Improvement)

```bash
# Add feedback files to training/feedback/
# Format: audio_001.wav + audio_001.txt

# Run incremental training
python train_feedback.py --feedback_dir training/feedback --epochs 5
```

## Key Parameters

### prepare_data.py
- `--max_samples N` - Process N samples (0 = all)
- `--target_sr 16000` - Target sample rate
- `--min_upvotes 2` - Quality filter
- `--output_dir training` - Output directory

### train_vits.py
- `--epochs 20` - Training epochs
- `--batch_size 4` - Batch size (reduce if OOM)
- `--gradient_accumulation_steps 8` - Gradient accumulation
- `--fp16` - Use mixed precision (default: True)
- `--max_samples N` - Use N samples (0 = all)

### train_feedback.py
- `--feedback_dir training/feedback` - Feedback directory
- `--epochs 5` - Training epochs
- `--learning_rate 1e-5` - Lower LR for fine-tuning

## Requirements

- **GPU**: RTX 2070 or better (8GB+ VRAM)
- **OS**: Windows (uses soundfile workaround)
- **Python**: 3.8+ (tested on 3.14)
- **Time**: 30 min (500 samples) to 12 hours (full)

## Dataset

- **Source**: Kurdish Common Voice (amedcj/kurmanji-commonvoice)
- **Samples**: 42,139 high-quality samples
- **Duration**: ~2-5 seconds per sample
- **Total**: ~58 hours of audio

## Troubleshooting

**Out of Memory?**
```bash
python train_vits.py --batch_size 2 --gradient_accumulation_steps 16
```

**Dataset not downloading?**
- Check internet connection
- Verify HuggingFace access

**CUDA not available?**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Documentation

- **Full Guide**: `VITS_TRAINING_README.md`
- **Implementation Details**: `IMPLEMENTATION_VITS_PIPELINE.md`

## Model Info

- **Base Model**: facebook/mms-tts-kmr-script_latin
- **Architecture**: VITS (36M parameters)
- **Sample Rate**: 16kHz
- **Vocabulary**: 36 Kurdish characters

## Output Structure

```
training/
├── wavs/              # Processed audio (16kHz WAV)
├── metadata.csv       # filename|text pairs
├── checkpoints/       # Training checkpoints
└── final_model/       # Fine-tuned model
```

## Next Steps After Training

1. Test the model:
   ```bash
   cp -r training/final_model models/kurdish
   python tts_stt_service_base44.py
   ```

2. Collect feedback from users

3. Improve with feedback:
   ```bash
   python train_feedback.py
   ```

4. Repeat steps 2-3 for continuous improvement

---

**For detailed documentation, see `VITS_TRAINING_README.md`**
