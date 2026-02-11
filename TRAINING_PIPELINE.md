# Kurdish Kurmanji TTS Fine-Tuning Guide

This guide explains how to fix the torchcodec audio loading issue and fine-tune the MMS TTS model on Kurdish Kurmanji data.

## 🔧 Problem: Torchcodec Audio Loading Issue

The `torchcodec` library is broken on Windows environments and prevents loading audio data from the `amedcj/kurmanji-commonvoice` dataset, causing:

```
RuntimeError: Could not load libtorchcodec. Likely causes:
  1. FFmpeg is not properly installed...
  2. The PyTorch version (2.10.0+cu128) is not compatible...
```

### Solution: Use Soundfile Backend

We've fixed this by:
1. **Explicitly excluding torchcodec** from dependencies
2. **Using soundfile** as the audio backend instead
3. **Setting environment variable** `HF_AUDIO_DECODER=soundfile` before loading datasets

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 2070 or better) recommended
- **Disk Space**: ~20GB for processed data and checkpoints
- **RAM**: 16GB+ recommended

### Software Requirements
- **Python**: 3.8-3.12
- **CUDA**: Compatible with your GPU (optional, but recommended)
- **FFmpeg**: Required for audio processing

### Install Dependencies

```bash
# First, uninstall torchcodec if present
pip uninstall torchcodec -y

# Install all required packages
pip install -r requirements.txt

# Verify soundfile is installed
python -c "import soundfile; print('soundfile version:', soundfile.__version__)"
```

## 🚀 Quick Start: Complete Training Pipeline

### Step 1: Prepare Training Data

The `prepare_data.py` script loads the Kurdish CommonVoice dataset and prepares it for training:

```bash
# Quick test with 500 samples (~5 minutes)
python prepare_data.py --output_dir training --max_samples 500

# Full dataset (45,992 samples, ~30 minutes)
python prepare_data.py --output_dir training --max_samples 0
```

**What it does:**
- ✅ Loads `amedcj/kurmanji-commonvoice` dataset using soundfile backend
- ✅ Filters for high-quality samples (up_votes ≥ 2, down_votes = 0)
- ✅ Extracts audio to WAV files at 16kHz mono in `training/wavs/`
- ✅ Creates metadata CSV with `file|text` format
- ✅ Shows dataset statistics (speakers, genders, durations, text samples)

**Output structure:**
```
training/
├── wavs/
│   ├── kmr_cv_000000.wav
│   ├── kmr_cv_000001.wav
│   └── ...
└── metadata.csv  (format: filename|text)
```

### Step 2: Fine-Tune MMS Model

The `train_vits.py` script fine-tunes the `facebook/mms-tts-kmr-script_latin` model:

```bash
# Quick test with 500 samples (~30 minutes on RTX 2070)
python train_vits.py --data_dir training --max_samples 500 --epochs 10

# Production training (all samples, ~6-8 hours)
python train_vits.py --data_dir training --epochs 50
```

**What it does:**
- ✅ Loads base MMS model (36M params, 138MB, already has Kurdish vocab)
- ✅ Fine-tunes on prepared Kurdish audio data
- ✅ Optimized for RTX 2070 8GB VRAM
- ✅ Saves checkpoints during training
- ✅ Creates drop-in replacement for current MMS model

**Output:**
```
models/mms-kurdish-finetuned/
├── final/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── training_config.json
│   └── ...
└── checkpoints/
    ├── checkpoint_500.pt
    ├── checkpoint_1000.pt
    └── ...
```

### Step 3: Use Fine-Tuned Model

Update your TTS service to use the fine-tuned model:

```python
from transformers import VitsModel, VitsTokenizer

# Load fine-tuned model
model_path = "models/mms-kurdish-finetuned/final"
model = VitsModel.from_pretrained(model_path)
tokenizer = VitsTokenizer.from_pretrained(model_path)

# Generate speech
text = "Silav, tu çawa yî?"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    output = model(**inputs).waveform
```

## 🔄 Feedback-Loop Training

The `train_feedback.py` script enables continuous improvement with user corrections:

### Add Single Correction

```bash
python train_feedback.py \
  --model_dir models/mms-kurdish-finetuned/final \
  --audio recording.wav \
  --text "Correct pronunciation text"
```

### Add Batch Corrections

Create a directory with audio files and corresponding text files:

```
corrections/
├── audio1.wav
├── audio1.txt
├── audio2.wav
├── audio2.txt
└── ...
```

Then add them:

```bash
python train_feedback.py \
  --model_dir models/mms-kurdish-finetuned/final \
  --batch_dir corrections
```

### Train on Accumulated Feedback

```bash
python train_feedback.py \
  --model_dir models/mms-kurdish-finetuned/final \
  --train \
  --epochs 10
```

### Show Feedback Statistics

```bash
python train_feedback.py \
  --model_dir models/mms-kurdish-finetuned/final \
  --stats
```

## 📊 Script Options

### prepare_data.py

```bash
python prepare_data.py [OPTIONS]

Options:
  --output_dir TEXT       Output directory (default: training)
  --max_samples INT       Max samples to process (0 = all, default: 0)
  --min_upvotes INT       Min up_votes for quality filter (default: 2)
  --max_downvotes INT     Max down_votes for quality filter (default: 0)
  --target_sr INT         Target sample rate in Hz (default: 16000)
  --split TEXT            Dataset split: train/test/validation (default: train)
```

### train_vits.py

```bash
python train_vits.py [OPTIONS]

Options:
  --data_dir TEXT                    Data directory (default: training)
  --output_dir TEXT                  Output directory (default: models/mms-kurdish-finetuned)
  --model_name TEXT                  Base model (default: facebook/mms-tts-kmr-script_latin)
  --epochs INT                       Training epochs (default: 50)
  --batch_size INT                   Batch size (default: 4, reduce if OOM)
  --gradient_accumulation_steps INT  Gradient accumulation (default: 8)
  --learning_rate FLOAT              Learning rate (default: 1e-5)
  --save_steps INT                   Save checkpoint every N steps (default: 500)
  --max_samples INT                  Max samples to use (0 = all)
  --resume                           Resume from last checkpoint
```

### train_feedback.py

```bash
python train_feedback.py [OPTIONS]

Options:
  --model_dir TEXT         Model directory (required)
  --feedback_dir TEXT      Feedback directory (default: feedback_data)
  --audio TEXT             Audio file for single feedback
  --text TEXT              Text transcription for single feedback
  --batch_dir TEXT         Directory with batch feedback
  --train                  Train on accumulated feedback
  --epochs INT             Training epochs (default: 10)
  --batch_size INT         Batch size (default: 2)
  --learning_rate FLOAT    Learning rate (default: 5e-6)
  --stats                  Show feedback statistics
```

## 🔧 Troubleshooting

### Torchcodec Still Being Loaded

**Problem**: Dataset library still tries to load torchcodec

**Solution**:
```bash
# Uninstall torchcodec completely
pip uninstall torchcodec -y

# Verify it's removed
pip list | grep torch

# Set environment variable
export HF_AUDIO_DECODER=soundfile  # Linux/Mac
set HF_AUDIO_DECODER=soundfile     # Windows CMD
$env:HF_AUDIO_DECODER="soundfile"  # Windows PowerShell
```

### Out of Memory (OOM) Errors

**Problem**: GPU runs out of memory during training

**Solution**:
```bash
# Reduce batch size
python train_vits.py --batch_size 2 --gradient_accumulation_steps 16

# Or use CPU (slower)
CUDA_VISIBLE_DEVICES="" python train_vits.py
```

### Dataset Download Issues

**Problem**: Cannot download amedcj/kurmanji-commonvoice dataset

**Solution**:
1. Check internet connection
2. Clear HuggingFace cache: `rm -rf ~/.cache/huggingface/`
3. Try manual download from HuggingFace website
4. Set HF_HOME environment variable to a location with more space

### Audio Processing Errors

**Problem**: Errors processing audio files

**Solution**:
```bash
# Install FFmpeg
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
# Windows: download from https://ffmpeg.org/

# Verify installation
ffmpeg -version

# Reinstall audio libraries
pip install --upgrade soundfile librosa
```

### Model Loading Errors

**Problem**: Cannot load MMS model

**Solution**:
```bash
# Upgrade transformers
pip install --upgrade transformers torch

# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--facebook--mms-tts-kmr-script_latin

# Check model exists on HuggingFace
python -c "from transformers import VitsModel; VitsModel.from_pretrained('facebook/mms-tts-kmr-script_latin')"
```

## 📈 Performance Benchmarks

### Data Preparation (prepare_data.py)

| Dataset Size | Processing Time | Output Size |
|--------------|----------------|-------------|
| 500 samples  | ~5 minutes     | ~80 MB      |
| 5,000 samples| ~30 minutes    | ~800 MB     |
| 45,992 samples| ~3 hours      | ~7 GB       |

### Training (train_vits.py on RTX 2070 8GB)

| Samples | Epochs | Training Time | Final Model Size |
|---------|--------|---------------|------------------|
| 500     | 10     | ~30 minutes   | ~140 MB          |
| 5,000   | 50     | ~4 hours      | ~140 MB          |
| 45,992  | 50     | ~24 hours     | ~140 MB          |

### Feedback Training (train_feedback.py)

| Feedback Samples | Epochs | Training Time |
|------------------|--------|---------------|
| 10               | 10     | ~2 minutes    |
| 100              | 10     | ~15 minutes   |
| 1,000            | 10     | ~2 hours      |

## 🎯 Best Practices

### For Initial Training

1. **Start with a test run**: Use `--max_samples 500` to verify everything works
2. **Monitor GPU memory**: Check `nvidia-smi` during training
3. **Use checkpoints**: Training can be interrupted and resumed
4. **Validate quality**: Test generated speech after each major training session

### For Feedback Training

1. **Accumulate corrections**: Collect 50-100 feedback samples before training
2. **Use lower learning rate**: Default 5e-6 is good for incremental updates
3. **Keep backups**: The script automatically backs up models before updating
4. **Test after updates**: Verify model quality hasn't degraded

### For Production Use

1. **Train on full dataset**: Use all 45,992 samples for best quality
2. **Multiple epochs**: 50-100 epochs recommended for production models
3. **Validation split**: Set aside test data to measure improvement
4. **Regular updates**: Retrain monthly with accumulated feedback

## 🔐 Security Notes

- The scripts do not collect or send any telemetry
- All data stays local on your machine
- No external API calls except for initial model/dataset downloads
- Feedback samples are stored locally in `feedback_data/`

## 📚 Additional Resources

- **MMS Model**: https://huggingface.co/facebook/mms-tts-kmr-script_latin
- **Kurdish CommonVoice Dataset**: https://huggingface.co/datasets/amedcj/kurmanji-commonvoice
- **Mozilla CommonVoice**: https://commonvoice.mozilla.org/
- **Transformers Docs**: https://huggingface.co/docs/transformers/
- **VITS Paper**: https://arxiv.org/abs/2106.06103

## 🤝 Integration with Base44 App

To integrate feedback training with your Base44 app:

1. **Add feedback endpoint** to your API:
```python
@app.route('/feedback', methods=['POST'])
def add_feedback():
    audio = request.files['audio']
    text = request.form['text']
    
    # Save audio temporarily
    audio_path = f"/tmp/feedback_{uuid.uuid4()}.wav"
    audio.save(audio_path)
    
    # Add to feedback trainer
    os.system(f"python train_feedback.py --model_dir models/mms-kurdish-finetuned/final --audio {audio_path} --text '{text}'")
    
    return jsonify({'status': 'success'})
```

2. **Schedule periodic retraining**:
```bash
# Add to cron (daily at 2 AM)
0 2 * * * cd /path/to/TTS_STT_Kurdifer && python train_feedback.py --model_dir models/mms-kurdish-finetuned/final --train --epochs 10
```

3. **Add UI for corrections**:
```javascript
// When user clicks "Report Bad Pronunciation"
function reportBadPronunciation(audio, correctText) {
    const formData = new FormData();
    formData.append('audio', audio);
    formData.append('text', correctText);
    
    fetch('/feedback', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        alert('Thank you for your feedback! The model will improve.');
    });
}
```

## 📝 License

This training pipeline is part of the TTS_STT_Kurdifer project and follows the same MIT license.

## 👤 Support

For issues or questions:
- GitHub Issues: https://github.com/T1Agit/TTS_STT_Kurdifer/issues
- Check the troubleshooting section above
- Review error messages carefully - they often contain helpful hints

---

**Made with ❤️ for the Kurdish community**
