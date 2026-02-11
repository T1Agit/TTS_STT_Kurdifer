# XTTS v2 Fine-Tuning Guide for Kurdish (Kurmanji)

This guide explains how to fine-tune XTTS v2 on Kurdish Common Voice data to add proper Kurdish language support to the TTS system.

## Overview

The `train_kurdish_xtts.py` script implements **actual fine-tuning** of XTTS v2 on Kurdish data, unlike the previous placeholder implementation. It uses Coqui TTS's built-in fine-tuning API to train the model on Mozilla Common Voice Kurdish (Kurmanji) dataset.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 2070 or better)
- **Disk Space**: ~20GB for processed data
- **RAM**: 16GB+ recommended

### Software Requirements
- **Python**: 3.8+
- **CUDA**: Compatible with your GPU
- **PyTorch**: 2.0+ with CUDA support

### Dependencies
Install all dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `coqui-tts>=0.27.0` - XTTS v2 model and training
- `librosa>=0.11.0` - Audio processing
- `pandas>=1.5.0` - TSV parsing
- `soundfile>=0.12.0` - WAV file I/O
- `tqdm>=4.65.0` - Progress bars
- `torch` - Deep learning framework

### Dataset
Download Mozilla Common Voice Kurdish (Kurmanji) corpus:
- Dataset: [Common Voice Kurdish v24.0](https://commonvoice.mozilla.org/en/datasets)
- Size: ~2.12 GB (91,298 clips total, 64,397 validated)
- Extract to: `cv-corpus-24.0-2025-12-05-kmr/cv-corpus-24.0-2025-12-05/kmr/`

## Training Modes

The script supports three training modes based on the `--max_samples` parameter:

### 1. Quick Test (Default)
**Time**: ~30 minutes  
**Samples**: 500 clips  
**Purpose**: Test the pipeline, verify setup

```bash
python train_kurdish_xtts.py --max_samples 500
```

### 2. Medium Training
**Time**: 2-3 hours  
**Samples**: 5,000 clips  
**Purpose**: Reasonably good model for testing

```bash
python train_kurdish_xtts.py --max_samples 5000
```

### 3. Full Training
**Time**: 8-12 hours  
**Samples**: All validated clips (~64K)  
**Purpose**: Best quality production model

```bash
python train_kurdish_xtts.py --max_samples 0
```

## Usage

### Basic Usage

```bash
# Quick test with default settings
python train_kurdish_xtts.py

# Specify corpus location
python train_kurdish_xtts.py --corpus_path "D:\path\to\cv-corpus-24.0-2025-12-05-kmr\cv-corpus-24.0-2025-12-05\kmr"

# Full training with custom output directory
python train_kurdish_xtts.py --max_samples 0 --output_dir "D:\kurdish_training\processed"
```

### Resume Training

Training can be interrupted and resumed from the last checkpoint:

```bash
python train_kurdish_xtts.py --resume
```

### Advanced Options

```bash
python train_kurdish_xtts.py \
  --corpus_path "path/to/corpus" \
  --output_dir "processed_audio" \
  --model_dir "models/kurdish" \
  --tsv_file "validated.tsv" \
  --max_samples 5000 \
  --epochs 10 \
  --batch_size 2 \
  --grad_accum_steps 16 \
  --learning_rate 5e-6 \
  --save_step 1000 \
  --resume
```

## Training Parameters

### For 8GB VRAM (RTX 2070)
- **Batch Size**: 2 (keep small)
- **Gradient Accumulation**: 16 steps (effective batch size: 32)
- **Learning Rate**: 5e-6 (low for fine-tuning)
- **Mixed Precision**: fp16 enabled (saves VRAM)
- **Save Checkpoint**: Every 1000 steps

### Adjusting for Different GPUs

**For 6GB VRAM or less:**
```bash
python train_kurdish_xtts.py \
  --batch_size 1 \
  --grad_accum_steps 32 \
  --max_samples 500
```

**For 16GB+ VRAM:**
```bash
python train_kurdish_xtts.py \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --max_samples 0
```

## Output Structure

After training, the following structure is created:

```
models/kurdish/
├── best_model.pth          # Final trained model weights
├── config.json             # Model configuration
├── speakers.pth            # Speaker embeddings
├── checkpoints/            # Training checkpoints
│   ├── checkpoint_1000.pth
│   ├── checkpoint_2000.pth
│   └── ...
└── training_log.txt        # Training progress log

processed_audio/
├── wavs/                   # Processed WAV files
│   ├── common_voice_kmr_24839000.wav
│   ├── common_voice_kmr_24839001.wav
│   └── ...
└── metadata.csv            # LJSpeech format metadata
```

## Pipeline Stages

### Stage 1: Data Preparation
1. Load Common Voice TSV file (validated.tsv)
2. Filter clips by quality:
   - Minimum 2 upvotes
   - Maximum 0 downvotes
   - Valid sentence text
3. Convert MP3 → WAV (22050Hz, mono)
4. Filter by duration (2-15 seconds)
5. Normalize audio
6. Create LJSpeech format metadata file

**Output**: Processed WAV files and metadata.csv

### Stage 2: Fine-Tuning
1. Download base XTTS v2 model
2. Configure dataset with BaseDatasetConfig
3. Load training samples (90/10 train/eval split)
4. Configure XTTS model:
   - Add Kurdish ('ku') to language list
   - Set training hyperparameters
   - Enable mixed precision
5. Initialize Trainer
6. Run training with trainer.fit()
7. Save model weights and checkpoints

**Output**: Fine-tuned model in models/kurdish/

## Using the Fine-Tuned Model

Once training is complete, the TTS service automatically detects and uses the fine-tuned model:

```python
from tts_stt_service_base44 import TTSSTTServiceBase44

service = TTSSTTServiceBase44()

# This will now use the fine-tuned Kurdish model
result = service.text_to_speech_base44(
    text="Silav, tu çawa yî?",
    language="kurdish"
)
```

The service will:
1. Check for `models/kurdish/best_model.pth` and `config.json`
2. If found, load the fine-tuned model
3. Use `language="ku"` for Kurdish TTS
4. Print status message: "Generating with fine-tuned Kurdish model"

If no trained model exists, it falls back to:
- Turkish phonetics as proxy for Kurdish
- Voice cloning mode
- Status message: "Using Turkish phonetics as proxy for Kurdish"

## Troubleshooting

### Out of Memory (OOM)
**Problem**: GPU runs out of memory during training

**Solutions**:
1. Reduce batch size: `--batch_size 1`
2. Increase gradient accumulation: `--grad_accum_steps 32`
3. Close other GPU-intensive applications
4. Use fewer samples: `--max_samples 500`

### Slow Training
**Problem**: Training is very slow

**Solutions**:
1. Ensure CUDA is properly installed
2. Check GPU is being used (not CPU)
3. Verify PyTorch CUDA version matches your CUDA installation
4. Use mixed precision (enabled by default)

### Import Errors
**Problem**: `ModuleNotFoundError` for TTS or trainer

**Solutions**:
```bash
# Install Coqui TTS with all dependencies
pip install coqui-tts>=0.27.0

# Or use the setup script
python setup_kurdish_tts.py
```

### Dataset Not Found
**Problem**: Script can't find corpus or clips directory

**Solutions**:
1. Verify corpus path: `--corpus_path "full/path/to/kmr/"`
2. Check directory structure:
   ```
   cv-corpus-24.0-2025-12-05-kmr/
   └── cv-corpus-24.0-2025-12-05/
       └── kmr/
           ├── clips/
           ├── validated.tsv
           └── ...
   ```
3. Use absolute paths on Windows: `"D:\path\to\corpus"`

### Checkpoint Loading Fails
**Problem**: Resume fails to load checkpoint

**Solutions**:
1. Check checkpoint directory exists: `models/kurdish/checkpoints/`
2. Verify checkpoint files: `checkpoint_*.pth`
3. If corrupted, start fresh without `--resume`

## Performance Benchmarks

### Training Time (RTX 2070, 8GB VRAM)
- **500 samples**: ~30 minutes
- **5,000 samples**: ~2-3 hours
- **64,000 samples**: ~8-12 hours

### Model Quality
- **500 samples**: Basic quality, good for testing
- **5,000 samples**: Good quality, reasonable for production
- **64,000 samples**: Best quality, production-ready

## Technical Details

### Fine-Tuning Approach
- **Architecture**: XTTS v2 (multilingual TTS model)
- **Method**: Update GPT-2 decoder weights, keep encoder frozen
- **Training**: Supervised learning on Kurdish audio-text pairs
- **Languages**: Adds 'ku' (Kurdish) to supported languages

### Data Processing
- **Input Format**: MP3 audio + TSV metadata
- **Output Format**: WAV (22050Hz, mono) + LJSpeech metadata
- **Quality Filter**: Upvotes ≥2, downvotes ≤0, duration 2-15s
- **Normalization**: Audio amplitude normalization

### Model Configuration
- **Base Model**: tts_models/multilingual/multi-dataset/xtts_v2
- **Batch Size**: 2 (for 8GB VRAM)
- **Effective Batch Size**: 32 (with gradient accumulation)
- **Learning Rate**: 5e-6 (low for fine-tuning)
- **Mixed Precision**: FP16 enabled
- **Optimizer**: AdamW (default in Trainer)

## Windows-Specific Notes

### File Paths
The script uses `Path` objects for cross-platform compatibility:
```python
# Works on both Windows and Linux
corpus_path = Path("D:/TTS_STT_Kurdifer/cv-corpus-24.0-2025-12-05-kmr/...")
```

### Multiprocessing
Windows requires explicit multiprocessing guard:
```python
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    # ... rest of code
```

### DataLoader
Set `num_workers=0` on Windows to avoid multiprocessing issues (handled automatically by Trainer).

## Next Steps

After training:

1. **Test the model**:
   ```bash
   python tts_stt_service_base44.py
   ```

2. **Check service output**:
   - Should see: "Fine-tuned Kurdish model loaded successfully"
   - Should use: "language='ku'"

3. **Compare quality**:
   - Generate audio with fine-tuned model
   - Compare to Turkish phonetic fallback
   - Kurdish should sound more natural

4. **Deploy**:
   - Copy `models/kurdish/` to production server
   - Service auto-detects and uses fine-tuned model
   - No code changes needed

## References

- [Coqui TTS Documentation](https://tts.readthedocs.io/)
- [XTTS v2 Paper](https://arxiv.org/abs/2311.00430)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/)
- [Kurdish TTS Implementation Guide](./KURDISH_TTS_IMPLEMENTATION.md)

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review training logs: `models/kurdish/training_log.txt`
3. Open an issue on GitHub with:
   - Error message
   - Command used
   - GPU/system info
   - Training logs
