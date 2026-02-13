# VITS Kurdish TTS Model Integration Guide

This guide explains how to use the integrated VITS TTS system for Kurdish text-to-speech with support for multiple model versions.

## Overview

The TTS service now supports two Kurdish TTS model options:

1. **Original** - Base `facebook/mms-tts-kmr-script_latin` model from HuggingFace
2. **Trained v8** - Fine-tuned model with improved Kurdish pronunciation

## Architecture

### Components

1. **vits_tts_service.py** - Core VITS TTS inference service
   - Loads and manages VITS models
   - Generates speech from Kurdish text
   - Supports multiple model versions
   - Verifies Kurdish character support

2. **tts_stt_service_base44.py** - Main TTS/STT service (updated)
   - Integrates VITS TTS with existing service
   - Falls back to Coqui TTS if VITS fails
   - Supports model selection via parameters

3. **api_server.py** - Flask API server (updated)
   - Accepts `model_version` parameter in `/tts` endpoint
   - New `/models` endpoint to list available models

4. **index.html** - Web frontend (updated)
   - Model selection dropdown (shown only for Kurdish)
   - Displays which model was used for generation
   - A/B comparison capability

## Usage

### Web Interface

1. Open the web interface (http://localhost:5000 when running locally)
2. Select "Kurdish (Kurdî)" as the language
3. A "Kurdish TTS Model" dropdown will appear with options:
   - **Original (MMS Base Model)** - Always available
   - **Trained v8 (Fine-tuned)** - Available when model files are present
4. Enter Kurdish text (e.g., "Silav, tu çawa yî?")
5. Click "Generate Speech"
6. The audio info will show which model and engine were used

### API Usage

#### Generate Speech with Model Selection

```bash
curl -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Silav, tu çawa yî?",
    "language": "kurdish",
    "model_version": "trained_v8"
  }'
```

Response:
```json
{
  "success": true,
  "audio": "ABC123...",
  "format": "mp3",
  "language": "ku",
  "model": "trained_v8",
  "engine": "vits"
}
```

#### List Available Models

```bash
curl http://localhost:5000/models
```

Response:
```json
{
  "success": true,
  "models": {
    "original": {
      "description": "Base MMS Kurdish TTS model",
      "type": "huggingface",
      "status": "available",
      "loaded": false
    },
    "trained_v8": {
      "description": "Fine-tuned Kurdish TTS v8",
      "type": "local",
      "status": "not_found",
      "loaded": false
    }
  }
}
```

### Python API Usage

```python
from tts_stt_service_base44 import TTSSTTServiceBase44

# Initialize service
service = TTSSTTServiceBase44()

# Generate speech with original model
result = service.text_to_speech_base44(
    text="Silav, tu çawa yî?",
    language="kurdish",
    model_version="original"
)

# Generate speech with trained model
result = service.text_to_speech_base44(
    text="Silav, tu çawa yî?",
    language="kurdish",
    model_version="trained_v8"
)

# Audio is Base44 encoded
audio_base44 = result['audio']
print(f"Generated with {result.get('model', 'unknown')} using {result.get('engine', 'unknown')}")
```

## Adding the Trained Model

### Required Files

Place these files in `training/best_model_v8/`:

1. **config.json** - Model configuration
2. **pytorch_model.bin** - Model weights
3. **tokenizer_config.json** - Tokenizer configuration
4. **vocab.json** - Vocabulary (if different from base)
5. **special_tokens_map.json** - Special tokens (if different from base)

### From Training Output

If you used `train_vits.py`:

```bash
# Copy final model
cp -r training/final_model/* training/best_model_v8/

# Or copy from specific checkpoint
cp -r training/checkpoints/checkpoint_epoch_10/* training/best_model_v8/
```

### Model Structure

```
training/best_model_v8/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.json (optional)
├── special_tokens_map.json (optional)
└── README.md
```

## Kurdish Character Support

The VITS tokenizer supports all Kurdish special characters:

| Character | Token ID | Example |
|-----------|----------|---------|
| ê         | 3        | hêvî (hope) |
| î         | 15       | zîrek (smart) |
| û         | 17       | dûr (far) |
| ç         | 12       | çiya (mountain) |
| ş         | 2        | şêr (lion) |

You can verify character support:

```python
from vits_tts_service import VitsTTSService

service = VitsTTSService()
char_support = service.verify_kurdish_chars('original')
print(char_support)
```

## Model Comparison

To A/B test models:

1. Generate audio with original model
2. Generate audio with trained_v8 model
3. Compare quality, pronunciation, and naturalness
4. Use the better model for production

Example:

```python
texts = [
    "Silav, tu çawa yî?",
    "Rojbûna te pîroz be",
    "Ez te hez dikim"
]

for text in texts:
    print(f"\nText: {text}")
    
    # Original
    result_orig = service.text_to_speech_base44(text, "kurdish", model_version="original")
    service.save_audio_from_base44(result_orig['audio'], f"orig_{text[:10]}.mp3")
    
    # Trained
    result_trained = service.text_to_speech_base44(text, "kurdish", model_version="trained_v8")
    service.save_audio_from_base44(result_trained['audio'], f"trained_{text[:10]}.mp3")
```

## Fallback Behavior

The service implements a robust fallback chain:

1. **Primary**: VITS TTS with selected model
2. **Fallback 1**: VITS TTS with original model (if trained_v8 fails)
3. **Fallback 2**: Coqui TTS (if VITS is unavailable)

This ensures the service always works, even if:
- The trained model is missing
- VITS libraries are not installed
- There's an error during generation

## Future Model Versions

To add future models (v9, v10, etc.):

1. Update `vits_tts_service.py` MODELS dict:

```python
MODELS = {
    'original': { ... },
    'trained_v8': { ... },
    'trained_v9': {
        'path': 'training/best_model_v9',
        'description': 'Fine-tuned Kurdish TTS v9',
        'type': 'local'
    }
}
```

2. Add the model files to the specified path
3. Update the frontend dropdown in `index.html`:

```html
<option value="trained_v9">Trained v9 (Latest)</option>
```

4. The service automatically handles the new model

## Performance

### Model Loading

- **First request**: 5-15 seconds (model download + load)
- **Cached requests**: <1 second
- Models are cached in memory after first load

### Generation Speed

- **Short text (5-10 words)**: 1-3 seconds
- **Long text (20-50 words)**: 3-8 seconds
- Speed depends on CPU/GPU and text length

### Memory Usage

- **Base model**: ~150 MB
- **Trained model**: ~150 MB
- **Both cached**: ~300 MB

## Troubleshooting

### Model Not Found

**Symptom**: API returns "Model directory not found"

**Solution**: Ensure model files are in `training/best_model_v8/` with correct structure

```bash
ls -la training/best_model_v8/
# Should show: config.json, pytorch_model.bin, etc.
```

### VITS Import Error

**Symptom**: "VITS TTS service not available"

**Solution**: Install dependencies

```bash
pip install torch torchaudio transformers
```

### Generation Fails

**Symptom**: "VITS TTS generation failed"

**Solution**: Service automatically falls back to Coqui TTS. Check logs for specific error.

### Model Downloads Fail

**Symptom**: "Cannot download model from HuggingFace"

**Solution**: Check internet connection and HuggingFace availability. The service will use cached model if available.

## Testing

Run the integration test:

```bash
python test_vits_integration.py
```

This verifies:
- All imports work correctly
- Service integration is complete
- API endpoints are configured
- Frontend UI is updated

## Security Notes

- Model files in `training/` are excluded from git (see `.gitignore`)
- Only model metadata (README.md) is committed
- Large model files should be stored separately (e.g., cloud storage)
- Use environment variables for any API keys or secrets

## License

- Base model: CC-BY-NC 4.0 (Facebook MMS)
- Integration code: Same as repository license
- Trained models: Subject to training data license

## Support

For issues or questions:
1. Check this documentation
2. Run `test_vits_integration.py` to verify setup
3. Check service logs for error messages
4. Open an issue on GitHub with error details
