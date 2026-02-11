# VITS/MMS Kurdish TTS Fine-tuning Pipeline

This directory contains scripts for fine-tuning the `facebook/mms-tts-kmr-script_latin` model on Kurdish Common Voice data for improved Kurdish Text-to-Speech.

## Overview

The pipeline consists of three main scripts:

1. **`prepare_data.py`** - Prepares the Kurdish Common Voice dataset for training
2. **`train_vits.py`** - Fine-tunes the VITS/MMS model on the prepared data
3. **`train_feedback.py`** - Enables incremental fine-tuning based on user feedback

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 2070 or better)
- **Disk Space**: ~20GB for processed data
- **RAM**: 16GB+ recommended

### Software
- **Python**: 3.8+ (tested on 3.14)
- **CUDA**: Compatible with your GPU
- **PyTorch**: 2.0+ with CUDA support

### Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

Key packages:
- `datasets>=2.14.0` - For loading Kurdish Common Voice dataset
- `transformers>=4.35.0` - For VITS model
- `torch>=2.0.0` - Deep learning framework
- `torchaudio>=2.0.0` - Audio processing
- `soundfile>=0.12.0` - Audio I/O (Windows-compatible)
- `librosa>=0.10.0` - Audio resampling
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `huggingface_hub` - HuggingFace model hub integration
- `tqdm>=4.65.0` - Progress bars

## Quick Start

### Step 1: Prepare Data

Download and prepare the Kurdish Common Voice dataset:

```bash
python prepare_data.py --output_dir training --max_samples 1000
```

This will:
- Load the `amedcj/kurmanji-commonvoice` dataset (using soundfile workaround for Windows)
- Filter for high-quality samples (up_votes >= 2, down_votes == 0)
- Resample audio to 16kHz mono
- Save WAV files to `training/wavs/`
- Create `training/metadata.csv` with format: `filename|text`
- Print statistics (samples, duration, speakers)

**Options:**
- `--output_dir`: Output directory (default: `training`)
- `--target_sr`: Target sample rate in Hz (default: 16000)
- `--max_samples`: Maximum samples to process, 0 = all (default: 0)
- `--min_upvotes`: Minimum up_votes for quality filter (default: 2)
- `--max_downvotes`: Maximum down_votes for quality filter (default: 0)

### Step 2: Fine-tune Model

Train the VITS/MMS model on the prepared data:

```bash
python train_vits.py --epochs 10 --batch_size 4
```

This will:
- Load the `facebook/mms-tts-kmr-script_latin` base model
- Fine-tune on data from `training/wavs/` + `training/metadata.csv`
- Use gradient accumulation and FP16 for 8GB VRAM optimization
- Compute mel spectrograms during training
- Save checkpoints to `training/checkpoints/` (every 2 epochs)
- Save final model to `training/final_model/`
- Log training progress and loss per epoch

**Options:**
- `--data_dir`: Directory with wavs/ and metadata.csv (default: `training`)
- `--checkpoint_dir`: Directory to save checkpoints (default: `training/checkpoints`)
- `--output_dir`: Directory to save final model (default: `training/final_model`)
- `--model_name`: Base model to fine-tune (default: `facebook/mms-tts-kmr-script_latin`)
- `--batch_size`: Batch size (default: 4 for 8GB VRAM)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--epochs`: Number of training epochs (default: 10)
- `--fp16`: Use mixed precision training (default: True)
- `--max_samples`: Maximum samples to use, 0 = all (default: 0)

**Effective Batch Size:** `batch_size × gradient_accumulation_steps = 4 × 8 = 32`

### Step 3: Feedback Loop (Optional)

After deploying your model, collect user feedback and improve it:

1. Create `training/feedback/` directory
2. Add corrected audio+text pairs:
   - `audio_001.wav` (user's corrected pronunciation)
   - `audio_001.txt` (correct text transcription)
   - `audio_002.wav`
   - `audio_002.txt`
   - etc.
3. Run incremental fine-tuning:

```bash
python train_feedback.py --feedback_dir training/feedback --epochs 5
```

This will:
- Load existing fine-tuned model from `training/final_model/`
- Train on feedback samples with lower learning rate (1e-5)
- Save updated model to `training/feedback_model/`

**Options:**
- `--feedback_dir`: Directory with WAV+TXT pairs (default: `training/feedback`)
- `--model_dir`: Directory with existing model (default: `training/final_model`)
- `--output_dir`: Directory to save updated model (default: `training/feedback_model`)
- `--base_model`: Fallback base model (default: `facebook/mms-tts-kmr-script_latin`)
- `--batch_size`: Batch size (default: 2)
- `--learning_rate`: Learning rate (default: 1e-5, lower for fine-tuning)
- `--epochs`: Number of epochs (default: 5)
- `--fp16`: Use mixed precision (default: True)

## Dataset Information

### Kurdish Common Voice Dataset

- **Dataset**: `amedcj/kurmanji-commonvoice`
- **Total samples**: 45,992
- **High quality samples**: 42,139 (up_votes >= 2, down_votes == 0)
- **Unique speakers**: 458
- **Gender distribution**: 17,109 male, 6,136 female, 22,747 unknown
- **Audio**: ~2-5 seconds per sample, originally 32kHz sample rate
- **Text column**: `sentence`
- **Audio column**: `path` (Audio type)

### Sample Texts

```
0: 2.88s - Em ê nebin.
1: 3.89s - Ez vê li xwe mikur tînim
2: 4.64s - Min dengekî lêdanê ji derî bihîst.
3: 4.21s - Dema ez zarok bûm ez qet bi fransî neaxivîm.
4: 2.16s - Rojbûna te pîroz be
```

## Model Information

### Base Model: facebook/mms-tts-kmr-script_latin

- **Architecture**: VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
- **Parameters**: 36M
- **Sample Rate**: 16kHz
- **Vocabulary Size**: 36 characters

### Tokenizer Vocabulary

```python
{
    'n': 0, 'h': 1, 'ş': 2, 'ê': 3, 'e': 4, 'p': 5, 'c': 6, 'x': 7,
    'w': 8, 'j': 9, 'd': 10, 's': 11, 'ç': 12, '-': 13, 'o': 14,
    'î': 15, 'm': 16, 'û': 17, 'k': 18, 'l': 19, 'a': 20, 'b': 21,
    '_': 22, 'z': 23, "'": 24, 'u': 25, 'f': 26, 'v': 27, 'q': 28,
    ' ': 29, 'y': 30, 't': 31, 'i': 32, 'g': 33, 'r': 34, '<unk>': 35
}
```

## Training Tips

### For RTX 2070 8GB VRAM

The scripts are optimized for 8GB VRAM:

1. **Batch size**: 4 (adjust down to 2 if OOM)
2. **Gradient accumulation**: 8 (effective batch size = 32)
3. **Mixed precision (FP16)**: Enabled by default
4. **Data workers**: 0 (for Windows compatibility)

### Memory Management

If you encounter OOM errors:
- Reduce `--batch_size` to 2 or 1
- Increase `--gradient_accumulation_steps` to maintain effective batch size
- Use `--max_samples` to train on subset of data first

### Training Time Estimates

- **500 samples**: ~30 minutes
- **5,000 samples**: 2-3 hours
- **42,000 samples (full)**: 8-12 hours

(Times may vary based on GPU and settings)

## Output Structure

```
training/
├── wavs/                    # Processed WAV files (16kHz mono)
│   ├── audio_000000.wav
│   ├── audio_000001.wav
│   └── ...
├── metadata.csv             # Training metadata (filename|text)
├── checkpoints/             # Training checkpoints
│   ├── checkpoint_epoch_2.pt
│   ├── checkpoint_epoch_4.pt
│   └── ...
├── final_model/             # Final fine-tuned model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
├── feedback/                # User feedback (optional)
│   ├── audio_001.wav
│   ├── audio_001.txt
│   └── ...
└── feedback_model/          # Feedback-updated model (optional)
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

## Windows Compatibility

### Audio Loading Workaround

The scripts use a soundfile-based workaround for Windows (torchcodec is broken on Windows):

```python
ds = load_dataset("amedcj/kurmanji-commonvoice", split="train")
ds = ds.cast_column("path", Audio(decode=False))  # Get raw bytes
# Then decode manually:
audio_bytes = sample["path"]["bytes"]
audio_data, sr = sf.read(io.BytesIO(audio_bytes))
```

This solution is **confirmed working** on Windows machines.

## Integration with Base44 App

The fine-tuned model can be integrated with the existing TTS service:

1. Copy the fine-tuned model to your deployment:
   ```bash
   cp -r training/final_model models/kurdish
   ```

2. The `tts_stt_service_base44.py` will automatically detect and use the fine-tuned model

3. For feedback loop:
   - Users record corrections in the Base44 app
   - Save as WAV+TXT pairs to `training/feedback/`
   - Run `train_feedback.py` to improve the model
   - Deploy updated model

## Troubleshooting

### OOM (Out of Memory) Errors

```bash
# Reduce batch size
python train_vits.py --batch_size 2

# Or train on subset first
python prepare_data.py --max_samples 1000
python train_vits.py --max_samples 1000
```

### Dataset Not Found

Make sure you have internet connection to download the dataset:
```bash
# Test dataset access
python -c "from datasets import load_dataset; ds = load_dataset('amedcj/kurmanji-commonvoice', split='train'); print(f'Found {len(ds)} samples')"
```

### CUDA Errors

Ensure PyTorch is installed with CUDA support:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is not available, install PyTorch with CUDA:
```bash
# For CUDA 12.8 (adjust version as needed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## License

This pipeline uses the following open-source components:
- Facebook MMS TTS model (CC-BY-NC 4.0)
- Mozilla Common Voice dataset (CC0)
- Transformers library (Apache 2.0)

## Citation

If you use this pipeline, please cite:

```bibtex
@article{pratap2023mms,
  title={Scaling Speech Technology to 1,000+ Languages},
  author={Pratap, Vineel and others},
  journal={arXiv preprint arXiv:2305.13516},
  year={2023}
}

@article{ardila2020common,
  title={Common Voice: A Massively-Multilingual Speech Corpus},
  author={Ardila, Rosana and others},
  journal={LREC 2020},
  year={2020}
}
```
