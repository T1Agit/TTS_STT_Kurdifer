# VITS Training v8 and Testing v6

This repository now includes the latest working training script (v8) and test script (v6) for Kurdish TTS model training.

## New Scripts

### 1. `train_vits_v8.py` - Training Script with Normalized Loss

The v8 training script includes several improvements over previous versions:

**Key Features:**
- **Normalized Loss**: Prevents gradient collapse and the "silent model" problem
- **Amplitude Preservation**: Maintains stable audio amplitude around 0.85
- **Real-time Monitoring**: Tracks loss, amplitude status, and training speed
- **Performance Optimized**: ~7 samples/second training speed, ~3.6 minutes per epoch with 1500 samples

**Training Characteristics:**
- Amplitude stays stable at ~0.85 (OK status)
- Loss decreasing progressively (e.g., Epoch 1: 0.6413 → Epoch 5: 0.5892 → Epoch 10: 0.5124)
- Speed: ~7 sps (samples per second)
- Time: ~3.6 min per epoch with 1500 samples

**Usage:**

```bash
# Quick training with 1500 samples (default)
python train_vits_v8.py

# Custom training
python train_vits_v8.py \
  --data_dir training \
  --max_samples 1500 \
  --epochs 10 \
  --batch_size 4 \
  --amplitude_target 0.85 \
  --amplitude_weight 0.1
```

**Parameters:**
- `--data_dir`: Directory containing wavs/ and metadata.csv (default: training)
- `--max_samples`: Maximum samples to use (default: 1500)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 4 for 8GB VRAM)
- `--amplitude_target`: Target amplitude for generated audio (default: 0.85)
- `--amplitude_weight`: Weight for amplitude preservation loss (default: 0.1)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8)

**Output:**
- Checkpoints saved to `training/checkpoints/`
- Final model saved to `training/final_model/`
- Training statistics saved to `training/final_model/training_stats.json`

### 2. `test_v6.py` - Model Comparison Script

The v6 test script compares the original and trained models:

**Key Features:**
- Loads both original and trained models
- Generates audio samples with both models
- Compares quality metrics:
  - Audio amplitude
  - Silence detection
  - Generation time
  - Energy levels
- Saves comparison results and audio files

**Usage:**

```bash
# Basic comparison test
python test_v6.py

# Custom test with specific texts
python test_v6.py \
  --original_model facebook/mms-tts-kmr-script_latin \
  --trained_model training/final_model \
  --output_dir test_outputs \
  --test_texts "Silav!" "Tu çawa yî?" "Spas!"
```

**Parameters:**
- `--original_model`: Original/base model name (default: facebook/mms-tts-kmr-script_latin)
- `--trained_model`: Path to trained model directory (default: training/final_model)
- `--output_dir`: Directory to save test results (default: test_outputs)
- `--test_texts`: Test texts to generate (default: Kurdish greetings)
- `--device`: Device to use: 'cuda', 'cpu', or 'auto' (default: auto)

**Output:**
- Comparison results saved to `test_outputs/comparison_results.json`
- Audio files saved to:
  - `test_outputs/original_audio/` - Original model outputs
  - `test_outputs/trained_audio/` - Trained model outputs

## Training Workflow

### Step 1: Prepare Data

First, prepare your Kurdish Common Voice dataset:

```bash
python prepare_data.py --output_dir training --max_samples 1500
```

### Step 2: Train Model with v8

Train the model using the v8 script with normalized loss:

```bash
python train_vits_v8.py --max_samples 1500 --epochs 10
```

Monitor the output for:
- Loss values (should decrease)
- Amplitude status (should be "OK" at ~0.85)
- Training speed (~7 sps)

### Step 3: Test the Model

Compare the trained model against the original:

```bash
python test_v6.py
```

Review the results in `test_outputs/comparison_results.json` and listen to the generated audio files.

## Understanding the Training Metrics

### Amplitude Status

The v8 script monitors amplitude in real-time:
- **✅ OK**: Amplitude within 0.1 of target (0.75-0.95 for target 0.85)
- **⚠️  WARNING**: Amplitude within 0.2 of target (0.65-1.05)
- **❌ CRITICAL**: Amplitude deviation > 0.2 from target

### Loss Metrics

- **Total Loss**: Combined normalized reconstruction loss + amplitude preservation loss
- **Reconstruction Loss**: How well the model reconstructs mel spectrograms
- **Amplitude Loss**: Penalty for deviating from target amplitude

### Training Speed

- **sps (samples per second)**: Actual training speed
- Expected: ~7 sps on RTX 2070 with 8GB VRAM
- Time per epoch: ~3.6 minutes for 1500 samples

## Comparison Metrics

The test script provides comprehensive comparison:

### Audio Quality Metrics
- **Max Amplitude**: Peak audio level
- **Mean Amplitude**: Average audio level
- **RMS Amplitude**: Root mean square amplitude
- **Energy**: Total audio energy
- **Zero Crossing Rate**: Measure of audio content

### Comparison Results
- **Amplitude Comparison**: Which model has better amplitude
- **Silence Check**: Whether models produce silent output
- **Generation Speed**: Time taken to generate audio
- **Energy Comparison**: Which model produces higher energy output

## Troubleshooting

### Model Goes Silent

If the model produces silent or very quiet audio:
1. Check the amplitude status during training
2. Increase `--amplitude_weight` (e.g., 0.2)
3. Ensure normalized loss is being used (it is in v8)

### Loss Not Decreasing

If loss doesn't decrease:
1. Reduce learning rate: `--learning_rate 1e-5`
2. Increase training samples: `--max_samples 3000`
3. Check data quality in `training/wavs/`

### Out of Memory

If you get OOM errors:
1. Reduce batch size: `--batch_size 2`
2. Increase gradient accumulation: `--gradient_accumulation_steps 16`
3. Use fewer samples: `--max_samples 1000`

## Version History

### Version 8 (Current)
- ✅ Normalized loss to prevent gradient collapse
- ✅ Amplitude preservation to prevent silent models
- ✅ Real-time monitoring of amplitude status
- ✅ Performance metrics tracking
- ✅ Default to 1500 samples for ~3.6 min epochs

### Test Version 6 (Current)
- ✅ Comprehensive model comparison
- ✅ Quality metrics analysis
- ✅ Audio file generation and saving
- ✅ JSON results export

## Integration with Existing Pipeline

These scripts integrate seamlessly with the existing VITS training pipeline:

1. **Data Preparation**: Use existing `prepare_data.py`
2. **Training**: Use new `train_vits_v8.py`
3. **Testing**: Use new `test_v6.py`
4. **Feedback Loop**: Use existing `train_feedback.py` for incremental improvements

## License

Same as the main repository. See the root LICENSE file for details.

## Support

For issues or questions:
1. Check this README
2. Review training logs and output
3. Open an issue on GitHub with:
   - Error message
   - Command used
   - GPU/system info
   - Training logs
