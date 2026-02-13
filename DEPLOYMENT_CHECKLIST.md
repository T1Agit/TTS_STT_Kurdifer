# VITS v8 Deployment Checklist

Use this checklist to deploy your fine-tuned VITS v8 model into the Base44 application.

## Prerequisites

- [ ] Fine-tuned VITS model is trained and saved
- [ ] Model training completed successfully with acceptable loss (< 0.7)
- [ ] Model files are accessible on your system

## Step 1: Prepare Model Files

- [ ] Locate your trained model files (output from `train_vits.py`)
- [ ] Verify these files exist:
  - [ ] `config.json` - Model configuration
  - [ ] `pytorch_model.bin` - Model weights
  - [ ] `tokenizer_config.json` - Tokenizer configuration
  - [ ] `vocab.json` - Vocabulary mapping
  - [ ] `special_tokens_map.json` - Special tokens

**Location of trained model:** `_____________________________`

## Step 2: Deploy Model to Application

### Option A: Local Development

```bash
# Copy model files to the designated directory
cp -r <your_model_path>/* training/best_model_v8/

# Or if your model is in training/final_model:
cp -r training/final_model/* training/best_model_v8/
```

- [ ] Model files copied to `training/best_model_v8/`
- [ ] All 5 required files present in directory

### Option B: Production Deployment

```bash
# Package model files
tar -czf vits_v8_model.tar.gz training/best_model_v8/

# Upload to server (example with scp)
scp vits_v8_model.tar.gz user@server:/path/to/app/

# On server, extract
cd /path/to/app
tar -xzf vits_v8_model.tar.gz
```

- [ ] Model packaged and uploaded
- [ ] Model extracted in correct location on server

## Step 3: Install Dependencies

```bash
# Ensure all required packages are installed
pip install -r requirements.txt

# Verify transformers and torch are installed
python -c "import transformers, torch; print('✅ Dependencies OK')"
```

- [ ] All dependencies installed
- [ ] transformers >= 4.30.0
- [ ] torch >= 2.0.0
- [ ] scipy >= 1.10.0

## Step 4: Test Model Loading

```bash
# Run integration tests
python test_vits_integration.py
```

Expected output:
```
✅ Service initialized successfully
✅ VITS model 'v8' found at: training/best_model_v8
✅ All tests passed!
```

- [ ] Integration tests pass
- [ ] v8 model is detected
- [ ] No errors in model detection

## Step 5: Start the Server

```bash
# Start the Flask API server
python api_server.py
```

Expected console output:
```
✅ VITS model 'original' available from HuggingFace: facebook/mms-tts-kmr-script_latin
✅ VITS model 'v8' found at: training/best_model_v8
🚀 Starting server on port 5000
```

- [ ] Server starts without errors
- [ ] v8 model is detected on startup
- [ ] Server accessible at http://localhost:5000

## Step 6: Test via Web UI

1. Open browser to http://localhost:5000
2. Select "Kurdish" language
3. Model dropdown should appear with options
4. Enter Kurdish text: "Silav, tu çawa yî?"
5. Select "VITS v8 (Fine-tuned)" model
6. Click "Generate Speech"

- [ ] Web UI loads correctly
- [ ] Model dropdown appears for Kurdish
- [ ] v8 model is available in dropdown
- [ ] Audio generates successfully
- [ ] Model info shows "v8" in output

## Step 7: Test via API

```bash
# Test auto-select (should use v8)
curl -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Silav, tu çawa yî?", "language": "kurdish"}'

# Test explicit v8 selection
curl -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Silav, tu çawa yî?", "language": "kurdish", "model": "v8"}'

# List available models
curl http://localhost:5000/models?language=kurdish
```

- [ ] Auto-select returns v8 model
- [ ] Explicit v8 selection works
- [ ] Models endpoint lists v8 as available
- [ ] API returns valid Base44-encoded audio

## Step 8: A/B Comparison

Run the example script to compare models:

```bash
python example_vits_usage.py
# Select option 4 (A/B comparison)
```

This will generate 3 audio files for comparison:
- `output_comparison_v8.mp3`
- `output_comparison_original.mp3`
- `output_comparison_coqui.mp3`

- [ ] All 3 audio files generated
- [ ] v8 model produces audible output
- [ ] Quality comparison shows v8 improvement

## Step 9: Production Deployment

### Environment Variables

If deploying to production (e.g., Railway, Heroku):

```bash
# Set any required environment variables
export PORT=5000
export MODEL_PATH=training/best_model_v8
```

- [ ] Environment variables configured
- [ ] Model path is accessible in production
- [ ] Server starts in production environment

### Model Persistence

Ensure model files are:
- [ ] Not in `.gitignore` (actual model files should be tracked or separately deployed)
- [ ] Accessible at runtime
- [ ] Backed up (model files are valuable!)

## Step 10: Monitoring

After deployment, monitor:
- [ ] Server logs for model loading errors
- [ ] Memory usage (v8 model uses ~2GB)
- [ ] Generation time (should be 300-500ms per sentence)
- [ ] Audio quality feedback from users

## Troubleshooting

If you encounter issues, check:

### Model Not Detected
- Verify `training/best_model_v8/config.json` exists
- Check file permissions (readable by server process)
- Review server logs for detection messages

### Import Errors
```bash
pip install transformers torch scipy
```

### CUDA/GPU Issues
- Model will auto-detect and use CPU if CUDA unavailable
- Check `torch.cuda.is_available()` returns True if GPU expected
- Verify CUDA version matches PyTorch installation

### Audio Quality Issues
- Check training loss was < 0.7
- Compare with original model
- Verify model wasn't corrupted during transfer

### Memory Issues
- Close other applications
- Use CPU mode if GPU memory is limited
- Consider using the original model as fallback

## Rollback Plan

If v8 model has issues:

1. Remove/rename the model directory:
   ```bash
   mv training/best_model_v8 training/best_model_v8.backup
   ```

2. Restart server - will auto-fallback to original or coqui

3. Investigate issues with backed up model

- [ ] Rollback plan tested
- [ ] Backup of model files exists

## Success Criteria

Your deployment is successful if:
- ✅ v8 model is detected on startup
- ✅ API accepts model selection parameter
- ✅ Web UI shows model dropdown for Kurdish
- ✅ Audio generates with v8 model
- ✅ Audio quality meets expectations
- ✅ No errors in server logs

## Next Steps

After successful deployment:
1. Collect user feedback on audio quality
2. Compare v8 with original model
3. Consider further fine-tuning with `train_feedback.py`
4. Monitor usage and performance metrics
5. Update model as needed for continuous improvement

## Support Resources

- **Integration Guide**: `VITS_V8_INTEGRATION.md`
- **Training Guide**: `VITS_TRAINING_README.md`
- **Usage Examples**: `example_vits_usage.py`
- **Tests**: `test_vits_integration.py`

---

**Deployment Date:** _____________________

**Deployed By:** _____________________

**Model Version:** v8

**Training Loss:** _____________________

**Notes:** 
_____________________________________________________________
_____________________________________________________________
_____________________________________________________________
