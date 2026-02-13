# VITS v8 Model Integration Guide

This document explains how to integrate and use the fine-tuned VITS v8 model for Kurdish TTS in the Base44 application.

## Overview

The Base44 app now supports multiple TTS models for Kurdish, including:
- **VITS v8**: Your fine-tuned model (best quality)
- **Original**: Facebook's `mms-tts-kmr-script_latin` base model
- **Coqui XTTS v2**: Fallback using Turkish phonetics

## Quick Start

### 1. Place Your Trained Model

Copy your trained model files to the designated directory:

```bash
# Your trained model should be at:
training/best_model_v8/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.json
└── special_tokens_map.json
```

These files are generated automatically when you run `train_vits.py` with `--output_dir training/best_model_v8`.

### 2. Install Dependencies

Ensure all required packages are installed:

```bash
pip install -r requirements.txt
```

Key dependencies for VITS support:
- `transformers>=4.30.0` - HuggingFace transformers for VITS model
- `torch>=2.0.0` - PyTorch for model inference
- `scipy>=1.10.0` - For audio file I/O

### 3. Start the Server

```bash
python api_server.py
```

The service will automatically detect available models on startup:

```
✅ VITS model 'original' available from HuggingFace: facebook/mms-tts-kmr-script_latin
✅ VITS model 'v8' found at: training/best_model_v8
```

## Using the Models

### Web Interface

1. Open http://localhost:5000 in your browser
2. Select "Kurdish" as the language
3. A "Model" dropdown will appear with options:
   - **Auto (best available)**: Automatically selects v8 if available
   - **VITS v8 (Fine-tuned)**: Uses your trained model
   - **Original (Facebook MMS)**: Uses the base model
   - **Coqui XTTS v2 (Fallback)**: Uses Turkish phonetics proxy

4. Enter Kurdish text and click "Generate Speech"
5. The model used will be displayed in the audio info

### Python API

```python
from tts_stt_service_base44 import TTSSTTServiceBase44

service = TTSSTTServiceBase44()

# Auto-select best model (prefers v8)
result = service.text_to_speech_base44(
    text="Silav, tu çawa yî?",
    language="kurdish"
)

# Explicitly use v8 model
result = service.text_to_speech_base44(
    text="Silav, tu çawa yî?",
    language="kurdish",
    model_version="v8"
)

# Use original Facebook model
result = service.text_to_speech_base44(
    text="Silav, tu çawa yî?",
    language="kurdish",
    model_version="original"
)

# Use Coqui fallback
result = service.text_to_speech_base44(
    text="Silav, tu çawa yî?",
    language="kurdish",
    model_version="coqui"
)

print(f"Used model: {result['model']}")
```

### REST API

```bash
# Auto-select best model
curl -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Silav, tu çawa yî?",
    "language": "kurdish"
  }'

# Use v8 model
curl -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Silav, tu çawa yî?",
    "language": "kurdish",
    "model": "v8"
  }'

# Use original model
curl -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Silav, tu çawa yî?",
    "language": "kurdish",
    "model": "original"
  }'

# List available models
curl http://localhost:5000/models?language=kurdish
```

## A/B Testing Models

You can easily compare models by generating the same text with different models:

```python
from tts_stt_service_base44 import TTSSTTServiceBase44

service = TTSSTTServiceBase44()
text = "Silav, tu çawa yî?"

# Generate with all models
result_v8 = service.text_to_speech_base44(text, "kurdish", model_version="v8")
result_original = service.text_to_speech_base44(text, "kurdish", model_version="original")
result_coqui = service.text_to_speech_base44(text, "kurdish", model_version="coqui")

# Save for comparison
service.save_audio_from_base44(result_v8['audio'], "output_v8.mp3")
service.save_audio_from_base44(result_original['audio'], "output_original.mp3")
service.save_audio_from_base44(result_coqui['audio'], "output_coqui.mp3")

print(f"V8 model: {len(result_v8['audio'])} chars")
print(f"Original model: {len(result_original['audio'])} chars")
print(f"Coqui model: {len(result_coqui['audio'])} chars")
```

## Model Selection Logic

The system uses the following priority when `model_version=None` (auto-select):

1. **v8** (if available) - Your fine-tuned model
2. **original** (if available) - Facebook MMS base model
3. **coqui** (fallback) - Coqui XTTS v2 with Turkish phonetics

This ensures the best quality is always used when available.

## Architecture

### Model Detection

On initialization, `TTSSTTServiceBase44` checks for models in these locations:

**For v8 model:**
- `training/best_model_v8/config.json`
- `./training/best_model_v8/config.json`
- `<script_dir>/training/best_model_v8/config.json`

**For original model:**
- HuggingFace Hub: `facebook/mms-tts-kmr-script_latin`

### Model Loading

Models are loaded lazily (on first use) to minimize startup time and memory usage:

1. First request for a model triggers loading
2. Model is cached in memory for subsequent requests
3. Model is moved to GPU if available
4. Model is set to eval mode for inference

### Audio Generation Flow

```
Text Input
    ↓
Language Detection (Kurdish?)
    ↓
Model Selection (auto or explicit)
    ↓
├─→ VITS Model (v8 or original)
│   ├─→ Load model from HuggingFace/local
│   ├─→ Tokenize text
│   ├─→ Generate waveform (16kHz)
│   ├─→ Export to WAV
│   └─→ Convert to MP3
│
└─→ Coqui XTTS v2 (fallback)
    ├─→ Load Coqui TTS
    ├─→ Generate with Turkish phonetics
    └─→ Convert to MP3
    ↓
Base44 Encoding
    ↓
API Response
```

## Performance

### Memory Usage

- **v8 model**: ~2GB VRAM/RAM during inference
- **original model**: ~2GB VRAM/RAM during inference
- **coqui model**: ~4GB VRAM/RAM during inference

### Inference Speed (RTX 2070)

- **v8 model**: ~300-500ms per sentence
- **original model**: ~300-500ms per sentence
- **coqui model**: ~1-2s per sentence

CPU-only inference will be slower (2-5x).

## Troubleshooting

### Model Not Detected

If your v8 model is not detected:

1. Check that `training/best_model_v8/config.json` exists
2. Verify the model files are complete (pytorch_model.bin, etc.)
3. Check server logs for error messages
4. Try absolute paths in `_vits_model_paths`

### CUDA Out of Memory

If you get OOM errors during inference:

1. Use CPU-only mode (model will auto-detect)
2. Close other GPU applications
3. Reduce batch size in training (doesn't affect inference)
4. Use the Coqui fallback model (smaller footprint)

### Import Errors

If you get "transformers not found" or similar:

```bash
pip install transformers torch scipy
```

### Audio Quality Issues

If audio quality is poor:

1. Check training loss - should be < 0.7 for good quality
2. Try the original model for comparison
3. Ensure model completed training (not interrupted)
4. Check that decoder wasn't accidentally trained (should be frozen)

## Future Improvements

To improve the model further:

1. **More training data**: Increase dataset size
2. **Longer training**: Train for more epochs
3. **Fine-tune decoder**: Unfreeze decoder for final epochs
4. **Feedback loop**: Use `train_feedback.py` with user corrections

See `VITS_TRAINING_README.md` for detailed training instructions.

## Technical Details

### Model Architecture

- **Base**: `facebook/mms-tts-kmr-script_latin`
- **Type**: VITS (Variational Inference with adversarial learning for TTS)
- **Parameters**: 36M total
- **Fine-tuned**: text_encoder, duration_predictor
- **Frozen**: decoder
- **Sample rate**: 16000 Hz
- **Vocabulary**: 36 Kurdish characters

### Training Configuration

Based on your training:
- **Loss**: 0.6326 (epoch 9)
- **Epochs**: 14 total
- **Dataset**: Kurdish Common Voice
- **Batch size**: 4 (effective 32 with grad accumulation)
- **Learning rate**: 2e-5
- **Precision**: FP16 (mixed precision)

## API Reference

### New Methods

#### `get_available_models(language='kurdish')`

Returns available models for a language.

**Returns:**
```python
{
    'language': 'ku',
    'models': ['v8', 'original', 'coqui'],
    'default_model': 'v8',
    'model_info': {
        'v8': {
            'type': 'VITS',
            'path': 'training/best_model_v8',
            'description': 'VITS model: v8'
        },
        ...
    }
}
```

### Updated Methods

#### `text_to_speech_base44(text, language='en', audio_format='mp3', model_version=None)`

New parameter: `model_version` - Select which model to use for Kurdish TTS.

**Options:**
- `None` (default): Auto-select best available
- `'v8'`: Use fine-tuned v8 model
- `'original'`: Use Facebook MMS base model
- `'coqui'`: Use Coqui XTTS v2 fallback

**Returns:** Dictionary with additional `'model'` field indicating which model was used.

### New API Endpoints

#### `GET /models?language=<lang>`

Get available models for a language.

**Example:**
```bash
curl "http://localhost:5000/models?language=kurdish"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "language": "ku",
    "models": ["v8", "original", "coqui"],
    "default_model": "v8",
    "model_info": { ... }
  }
}
```

## License

This integration maintains compatibility with:
- Facebook MMS TTS model: CC-BY-NC 4.0
- Mozilla Common Voice dataset: CC0
- Transformers library: Apache 2.0

## Support

For issues or questions:
1. Check this guide
2. Review `VITS_TRAINING_README.md`
3. Check server logs
4. Open an issue on GitHub
