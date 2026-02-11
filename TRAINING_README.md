# Kurdish MMS TTS Training Guide

This guide explains how to prepare data and train the MMS (Massively Multilingual Speech) TTS model for Kurdish (Kurmanji) using the Common Voice dataset.

## 🎯 Overview

The training pipeline consists of three main scripts:

1. **`prepare_data.py`** - Prepares Kurdish Common Voice data for training
2. **`train_vits.py`** - Fine-tunes the MMS model on prepared data
3. **`train_feedback.py`** - Incremental fine-tuning based on user feedback

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 2070 or better (8GB+ VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: 20GB+ free space
- **OS**: Windows, Linux, or macOS

### Software Requirements
- **Python**: 3.8+ (tested on 3.14)
- **PyTorch**: 2.0+ with CUDA support (tested on 2.10.0+cu128)
- **CUDA**: 11.8+ or 12.x

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

Dependencies include:
- `datasets>=2.14.0` - HuggingFace datasets library
- `transformers>=4.30.0` - HuggingFace transformers
- `accelerate>=0.20.0` - Training acceleration
- `torch>=2.0.0` - PyTorch
- `soundfile>=0.12.0` - Audio I/O
- `librosa>=0.11.0` - Audio processing

## 📊 Dataset Information

**Dataset**: [amedcj/kurmanji-commonvoice](https://huggingface.co/datasets/amedcj/kurmanji-commonvoice)

**Statistics**:
- **45,992 total samples**
- **42,139 high quality** (2+ upvotes, 0 downvotes)
- **458 unique speakers** (17,109 male, 6,136 female, 22,747 unknown)

**Columns**:
- `client_id` - Speaker ID
- `path` - Audio file path
- `sentence` - Kurdish text
- `up_votes` - Number of upvotes
- `down_votes` - Number of downvotes
- `gender` - Speaker gender
- `age` - Speaker age
- `text` - Normalized text

## 🔧 Step 1: Data Preparation

The `prepare_data.py` script handles the audio loading issue by using `Audio(decode=False)` to bypass the `torchcodec` dependency, then manually decoding with `soundfile`.

### Basic Usage

```bash
# Prepare all high-quality samples
python prepare_data.py

# Limit to 1000 samples for testing
python prepare_data.py --max_samples 1000

# Custom quality threshold
python prepare_data.py --min_upvotes 3 --max_downvotes 0
```

### Options

```
--dataset           Dataset name (default: amedcj/kurmanji-commonvoice)
--output_dir        Output directory (default: training)
--min_upvotes       Minimum upvotes (default: 2)
--max_downvotes     Maximum downvotes (default: 0)
--target_sr         Target sample rate in Hz (default: 16000)
--max_samples       Max samples to process (default: None = all)
```

### What It Does

1. **Loads dataset** with `decode=False` to get raw audio bytes
2. **Manually decodes** audio using `soundfile` 
3. **Filters** for quality based on votes
4. **Resamples** to 16kHz mono
5. **Normalizes** audio levels
6. **Filters** by duration (1-15 seconds)
7. **Saves** WAV files to `training/wavs/`
8. **Creates** `training/metadata.csv` with format: `file|text`

### Output Structure

```
training/
├── wavs/
│   ├── audio_000000.wav
│   ├── audio_000001.wav
│   └── ...
└── metadata.csv
```

### Example Output

```
================================================================================
📊 Data Preparation Statistics
================================================================================

✅ Successfully processed samples: 5000
📊 Total audio duration: 350.5 minutes (5.84 hours)
📊 Average audio length: 4.2 seconds

📝 Text Statistics:
   Average text length: 42.3 characters
   Min text length: 8 characters
   Max text length: 150 characters

📁 Output Files:
   WAV files: training/wavs/
   Metadata: training/metadata.csv
```

## 🚀 Step 2: Model Fine-Tuning

The `train_vits.py` script fine-tunes the `facebook/mms-tts-kmr-script_latin` model (36M parameters) on your prepared data.

### Basic Usage

```bash
# Start training with default settings
python train_vits.py

# Custom settings for 8GB VRAM
python train_vits.py \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --epochs 10 \
  --learning_rate 1e-5

# Resume from checkpoint
python train_vits.py --resume_from training/checkpoints/checkpoint-5000
```

### Options

```
--model_name                Model to fine-tune (default: facebook/mms-tts-kmr-script_latin)
--metadata_path             Metadata file (default: training/metadata.csv)
--wavs_dir                  WAV files directory (default: training/wavs)
--output_dir                Checkpoint directory (default: training/checkpoints)
--epochs                    Number of epochs (default: 10)
--batch_size                Batch size (default: 4)
--gradient_accumulation_steps  Gradient accumulation (default: 4)
--learning_rate             Learning rate (default: 1e-5)
--warmup_steps              Warmup steps (default: 500)
--save_steps                Save every N steps (default: 1000)
--no_mixed_precision        Disable FP16 training
```

### Training Configuration

**For RTX 2070 (8GB VRAM)**:
- Batch size: 4
- Gradient accumulation: 4
- Effective batch size: 16
- Mixed precision: FP16
- Memory usage: ~6-7GB

**For RTX 3090 (24GB VRAM)**:
- Batch size: 8-16
- Gradient accumulation: 2-4
- Effective batch size: 32-64
- Mixed precision: FP16
- Memory usage: ~18-20GB

### Training Output

```
================================================================================
MMS Fine-Tuning for Kurdish (Kurmanji)
================================================================================

📦 Loading model: facebook/mms-tts-kmr-script_latin
✅ Tokenizer loaded
   Vocabulary size: 36
✅ Model loaded
   Total parameters: 36.5M
   Trainable parameters: 36.5M

📊 Training Configuration:
   Epochs: 10
   Batch size: 4
   Gradient accumulation: 4
   Effective batch size: 16
   Learning rate: 1e-5
   Total training steps: 3125
   Mixed precision: True

📈 Epoch 1/10
100%|████████████| 781/781 [12:34<00:00, 1.03it/s, loss=0.234, lr=1.0e-05]

📊 Epoch 1 Summary:
   Average Loss: 0.2340
   Learning Rate: 1.00e-05
```

### Output Structure

```
training/checkpoints/
├── checkpoint-1000/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
├── checkpoint-2000/
├── best_model/
├── final_model/
└── training_log.txt
```

## 🔄 Step 3: Incremental Fine-Tuning (Feedback Loop)

The `train_feedback.py` script enables continuous improvement through user feedback from the Base44 app.

### Adding Feedback Samples

```bash
# Add single feedback sample
python train_feedback.py \
  --add_sample audio_correction.wav "Corrected Kurdish text"

# Or manually add to feedback directory
# 1. Copy audio to: feedback/wavs/
# 2. Add entry to: feedback/metadata.csv (format: filename|text)
```

### Running Feedback Training

```bash
# Train on feedback data
python train_feedback.py

# With custom settings
python train_feedback.py \
  --model_path training/checkpoints/best_model \
  --feedback_weight 2.0 \
  --epochs 5 \
  --learning_rate 5e-6 \
  --archive_feedback
```

### Options

```
--model_path        Base model to improve (default: training/checkpoints/best_model)
--feedback_dir      Feedback data directory (default: feedback)
--feedback_weight   Repeat feedback samples N times (default: 2.0)
--epochs            Training epochs (default: 5)
--batch_size        Batch size (default: 4)
--learning_rate     Learning rate (default: 5e-6, lower for stability)
--archive_feedback  Archive feedback after training
--add_sample        Add single sample: AUDIO_PATH TEXT
```

### Feedback Workflow

1. **Collect**: Users provide corrections via Base44 app
2. **Store**: Feedback saved to `feedback/wavs/` and `feedback/metadata.csv`
3. **Train**: Run `train_feedback.py` to incorporate feedback
4. **Test**: Evaluate improved model
5. **Deploy**: Replace production model if quality improves
6. **Archive**: Move processed feedback to archive

### Output

```
================================================================================
Feedback Loop Training for Kurdish MMS TTS
================================================================================

📊 Found 50 feedback samples

🔄 Merging feedback data with base training data...
   Feedback weight: 2.0x
✅ Copied 5000 base training samples
✅ Added 50 feedback samples (weighted 2.0x)
📊 Total training samples: 5100

💾 Saved checkpoint to training/checkpoints_feedback/checkpoint-500
💾 Saved best model (loss: 0.1234)

✅ Incremental Fine-Tuning Complete!
```

## 🎤 Model Information

### MMS (Massively Multilingual Speech)

**Model**: `facebook/mms-tts-kmr-script_latin`

**Specifications**:
- Parameters: 36M
- Architecture: VITS (Variational Inference with adversarial learning for end-to-end TTS)
- Sample rate: 16kHz
- Vocabulary: 36 characters (includes Kurdish special chars: ê, î, û, ş, ç)

**Kurdish Vocabulary**:
```python
{
  'n': 0, 'h': 1, 'ş': 2, 'ê': 3, 'e': 4, 'p': 5, 'c': 6, 'x': 7,
  'w': 8, 'j': 9, 'd': 10, 's': 11, 'ç': 12, '-': 13, 'o': 14,
  'î': 15, 'm': 16, 'û': 17, 'k': 18, 'l': 19, 'a': 20, 'b': 21,
  '_': 22, 'z': 23, "'": 24, 'u': 25, 'f': 26, 'v': 27, 'q': 28,
  ' ': 29, 'y': 30, 't': 31, 'i': 32, 'g': 33, 'r': 34, '<unk>': 35
}
```

## 🔧 Troubleshooting

### Memory Issues

**Problem**: CUDA out of memory

**Solution**:
```bash
# Reduce batch size and increase gradient accumulation
python train_vits.py \
  --batch_size 2 \
  --gradient_accumulation_steps 8
```

### Dataset Loading Issues

**Problem**: "To support decoding audio data, please install 'torchcodec'"

**Solution**: The `prepare_data.py` script handles this automatically by using `Audio(decode=False)` and manually decoding with `soundfile`. No action needed.

### Audio Processing Errors

**Problem**: Failed to process audio files

**Solution**:
```bash
# Check audio file integrity
python -c "import soundfile as sf; sf.read('training/wavs/audio_000000.wav')"

# Verify sample rate
python -c "import soundfile as sf; info = sf.info('training/wavs/audio_000000.wav'); print(info)"
```

### Training Not Improving

**Problem**: Loss not decreasing

**Solutions**:
- Reduce learning rate: `--learning_rate 5e-6`
- Increase warmup steps: `--warmup_steps 1000`
- Check data quality in `training/metadata.csv`
- Ensure audio and text match correctly

## 📈 Training Best Practices

### Data Quality

- ✅ Use `--min_upvotes 2` for quality filtering
- ✅ Keep audio duration between 1-15 seconds
- ✅ Ensure text transcriptions are accurate
- ✅ Remove noise and normalize audio

### Training Strategy

1. **Initial Training**: Start with 10 epochs on full dataset
2. **Evaluation**: Test on diverse Kurdish texts
3. **Feedback Collection**: Gather user corrections
4. **Incremental Training**: Fine-tune with feedback (5 epochs, lower LR)
5. **A/B Testing**: Compare before/after quality
6. **Deployment**: Deploy if improvement verified
7. **Repeat**: Continue feedback loop

### Hyperparameter Tuning

**Learning Rate**:
- Initial training: `1e-5`
- Feedback training: `5e-6`
- If unstable: reduce by 50%

**Batch Size**:
- 8GB VRAM: batch_size=4, grad_accum=4
- 16GB VRAM: batch_size=8, grad_accum=2
- 24GB VRAM: batch_size=16, grad_accum=1

**Epochs**:
- Initial: 10-20 epochs
- Feedback: 3-5 epochs (to avoid overfitting)

## 🎯 Expected Results

### Training Time

**For 5000 samples on RTX 2070**:
- Data preparation: ~10-15 minutes
- Training (10 epochs): ~2-3 hours
- Per epoch: ~15-20 minutes

**For 40,000 samples on RTX 3090**:
- Data preparation: ~45-60 minutes
- Training (10 epochs): ~8-12 hours
- Per epoch: ~50-70 minutes

### Quality Metrics

- **Initial model** (no training): Baseline quality
- **After fine-tuning**: Significant improvement in naturalness and pronunciation
- **After feedback loop**: Further refinement based on user corrections

## 🔗 Integration with Base44 App

### API Endpoint for Feedback

```python
# Example Flask endpoint for collecting feedback
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    audio_file = request.files['audio']
    corrected_text = request.form['text']
    
    # Save audio
    audio_path = f"feedback/wavs/{timestamp}.wav"
    audio_file.save(audio_path)
    
    # Append to metadata
    with open('feedback/metadata.csv', 'a') as f:
        f.write(f"{timestamp}.wav|{corrected_text}\n")
    
    return {'status': 'success'}
```

### Automated Training Pipeline

```bash
#!/bin/bash
# Weekly feedback training

# Check if feedback exists
if [ -f "feedback/metadata.csv" ]; then
    echo "Starting feedback training..."
    python train_feedback.py --archive_feedback
    echo "Training complete!"
else
    echo "No feedback data found"
fi
```

## 📚 Additional Resources

- [MMS Paper](https://arxiv.org/abs/2305.13516)
- [VITS Paper](https://arxiv.org/abs/2106.06103)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Kurdish Common Voice](https://commonvoice.mozilla.org/ku)

## 🤝 Contributing

Contributions welcome! Please:
1. Test changes on your local setup
2. Document any new features
3. Update this README if adding functionality
4. Follow existing code style

## 📝 License

MIT License - Same as parent project

---

**Made with ❤️ for the Kurdish community**
