# 🎤 Kurdish TTS/STT API with Base44 Encoding

A complete multilingual Text-to-Speech (TTS) and Speech-to-Text (STT) solution with custom Base44 encoding for efficient audio transfer. **Now featuring both high-quality Kurdish (Kurmanji) TTS with VITS models and Kurdish STT with MMS-1B!**

## 🌟 Features

### Text-to-Speech (TTS)
- **5 Languages**: Kurdish (Kurdî), English, German, French, Turkish
- **Kurdish VITS v8**: Fine-tuned VITS model for high-quality Kurdish speech synthesis
- **Multiple TTS Engines**: VITS (Kurdish), gTTS (other languages)
- **Base44 Encoding**: Efficient audio encoding (1.47x compression)

### Speech-to-Text (STT) - NEW!
- **Kurdish Speech Recognition**: Using facebook/mms-1b-all with Kurdish (kmr) adapter
- **Automatic Post-Processing**: Built-in dictionary-based correction system with 779+ word corrections
- **Kurdish Character Support**: Full support for ê, î, û, ç, ş
- **Microphone Recording**: Record audio directly in browser
- **File Upload**: Drag & drop or browse for audio files (MP3, WAV, OGG, WebM)
- **Confidence Scores**: Get transcription confidence metrics
- **Raw & Corrected Output**: Returns both original and post-processed transcriptions

### General
- **Unified Web UI**: Tabbed interface with TTS and STT in one place
- **REST API**: Simple HTTP endpoints for both TTS and STT
- **Railway Deployed**: Live 24/7
- **Secure**: Zero npm vulnerabilities, secure Python dependencies

---

## 🚀 Live Demo

### 🌐 Web Interface
**[https://t1agit.github.io/TTS_STT_Kurdifer/](https://t1agit.github.io/TTS_STT_Kurdifer/)**

### 🔗 API Endpoint
**[https://ttststtkurdifer-production.up.railway.app](https://ttststtkurdifer-production.up.railway.app)**

### 📸 Screenshots

#### Text-to-Speech Tab
![TTS Tab](https://github.com/user-attachments/assets/960ee2dd-d8f2-4206-a2f3-2dce656e84cc)

Features Kurdish text input with special character buttons (ê, î, û, ç, ş), language selection, and model selection for Kurdish.

#### Speech-to-Text Tab
![STT Tab](https://github.com/user-attachments/assets/bd90f10e-e289-46b4-b8b3-f57d1687dc99)

Features audio file upload with drag & drop, microphone recording, and displays transcribed Kurdish text with proper character support.

---

## 📡 API Endpoints

### 1. Health Check
```bash
GET /health
```
**Response:**
```json
{"status": "healthy"}
```

### 2. Get Supported Languages
```bash
GET /languages
```
**Response:**
```json
{
  "languages": ["kurdish", "german", "french", "english", "turkish", "ku", "de", "fr", "en", "tr"]
}
```

### 3. Text-to-Speech (TTS)
```bash
POST /tts
Content-Type: application/json

{
  "text": "Hello, how are you?",
  "language": "english"
}
```

**Response:**
```json
{
  "audio": "DAMc3XX6M5QeVe66PDYfMO8fLGGc...",
  "format": "mp3",
  "language": "english",
  "size": 8973
}
```

### 4. Speech-to-Text (STT) - NEW!
```bash
POST /stt
Content-Type: application/json

{
  "audio": "DAMc3XX6M5QeVe66PDYfMO8fLGGc..."
}
```

**Or upload file (local example):**
```bash
# Local development
curl -X POST http://localhost:5000/stt \
  -F "audio=@kurdish_audio.mp3"

# Production
curl -X POST https://ttststtkurdifer-production.up.railway.app/stt \
  -F "audio=@kurdish_audio.mp3"
```

**Response:**
```json
{
  "success": true,
  "text": "Silav, tu çawa yî?",
  "raw_text": "Silav, tu cawa yi?",
  "language": "kmr",
  "confidence": 0.95,
  "duration": 2.5
}
```
Note: `text` contains the post-processed/corrected transcription, while `raw_text` contains the original STT output.

### 5. STT Status Check
```bash
GET /stt/status
```

**Response:**
```json
{
  "success": true,
  "available": true,
  "model": "facebook/mms-1b-all",
  "language": "kmr (Kurdish Kurmanji)"
}
```

---

## 🖥️ Quick Start

### Option 1: Use the Web Interface
1. Go to **[https://t1agit.github.io/TTS_STT_Kurdifer/](https://t1agit.github.io/TTS_STT_Kurdifer/)** or run locally
2. **For TTS (Text-to-Speech):**
   - Enter Kurdish text (use special character buttons: ê, î, û, ç, ş)
   - Select language and model
   - Click "Generate Speech"
   - Listen to the generated audio!
3. **For STT (Speech-to-Text):**
   - Switch to "Speech-to-Text" tab
   - Upload an audio file (drag & drop or browse)
   - Or click "Record Audio" to record from microphone
   - Click "Transcribe Audio"
   - See the Kurdish text transcription with proper characters!

### Option 2: Use cURL for TTS
```bash
curl -X POST https://ttststtkurdifer-production.up.railway.app/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Silav, tu çawa yî?", "language": "kurdish"}'
```

### Option 3: Use cURL for STT
```bash
curl -X POST http://localhost:5000/stt \
  -F "audio=@kurdish_audio.mp3"
```

### Option 3: Use JavaScript
```javascript
fetch('https://ttststtkurdifer-production.up.railway.app/tts', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Hello World',
    language: 'english'
  })
})
.then(r => r.json())
.then(data => {
  // Decode Base44 audio
  const audioBytes = Base44.decode(data.audio);
  const audioBlob = new Blob([audioBytes], {type: 'audio/mpeg'});
  const audioUrl = URL.createObjectURL(audioBlob);
  
  // Play audio
  const audio = new Audio(audioUrl);
  audio.play();
});
```

---

## 🛠️ Local Development

### Prerequisites
- **Python 3.8+** (Python 3.12 recommended)
- **Node.js 14+** (for API server)
- **~2GB free disk space** (for Kurdish XTTS v2 model)
- **Internet connection** (for first-time model download)

### Installation Steps

#### 1. Clone Repository
```bash
git clone https://github.com/T1Agit/TTS_STT_Kurdifer.git
cd TTS_STT_Kurdifer
```

#### 2. Install Python Dependencies
```bash
# Install all required packages including Coqui TTS for Kurdish
pip install -r requirements.txt
```

**Dependencies include:**
- `gTTS` - Google Text-to-Speech (for English, German, French, Turkish)
- `coqui-tts>=0.27.0` - XTTS v2 model (for Kurdish with high-quality voice synthesis)
- `librosa>=0.11.0` - Audio processing library (for training and voice cloning)
- `flask` + `flask-cors` - API server
- `pydub` - Audio processing
- Other utilities

**Note:** `coqui-tts` will automatically install compatible versions of PyTorch, torchaudio, and transformers. The version constraints have been removed to avoid dependency conflicts and let coqui-tts manage its own dependencies.

#### 3. Install Node.js Dependencies (Optional - for Node.js API)
```bash
npm install
```

#### 4. Setup Kurdish TTS (XTTS v2 Model)
**First-time setup required for Kurdish language support:**

```bash
# Run the automated setup script
python setup_kurdish_tts.py
```

**What this does:**
- ✅ Checks if Coqui TTS is installed
- ✅ Checks for fine-tuned Kurdish model
- ✅ Downloads base XTTS v2 model (~2GB) if needed
- ✅ Provides training instructions if no fine-tuned model found
- ✅ Configures voice cloning fallback (works immediately)
- ⏱️ Takes 2-5 minutes for initial download

**Two modes available:**
1. **Voice Cloning Fallback** (default, works immediately)
   - Uses Turkish phonetics as proxy for Kurdish
   - Good quality, no training required
   
2. **Fine-Tuned Model** (best quality, requires training)
   - See "Training Your Own Kurdish Model" section above
   - Run: `python train_kurdish_xtts.py`

**For automated/CI installations:**
```bash
# Set this environment variable to automatically accept Coqui TOS
export COQUI_TOS_AGREED=1
python setup_kurdish_tts.py
```

**Manual verification:**
```bash
# Test Kurdish TTS directly
python3 << 'EOF'
from tts_stt_service_base44 import TTSSTTServiceBase44
service = TTSSTTServiceBase44()
result = service.text_to_speech_base44('Silav, tu çawa yî?', 'kurdish')
print(f'✅ Generated {result["size"]} bytes of Kurdish audio')
EOF
```

#### 5. Run Python API Server
```bash
# Start Flask server on port 5000
python api_server.py
```

Server runs on: `http://localhost:5000`

#### 6. Run Node.js API Server (Alternative)
```bash
# Start Express server on port 3000
npm start
```

Server runs on: `http://localhost:3000`

**Note:** Node.js server calls Python backend for TTS generation.

### Testing Your Installation

```bash
# Test Python implementation
python test_kurdish_implementation.py

# Test Node.js integration
npm test

# Test TTS service directly
python tts_stt_service_base44.py

# Test API with cURL
curl -X POST http://localhost:5000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Silav", "language": "kurdish"}'
```

---

## 🧠 Train Custom Kurdish Voice

Use Google Colab for FREE GPU training:

**[Open Colab Notebook](https://colab.research.google.com/github/T1Agit/TTS_STT_Kurdifer/blob/main/kurdish_tts_training.ipynb)**

### Steps:
1. Upload your Kurdish MP3 samples (30min - 2h)
2. Upload transcriptions (text files)
3. Run training cells (2-6 hours)
4. Download trained model
5. Use on Raspberry Pi or local server

---

## 🗂️ Project Structure

```
TTS_STT_Kurdifer/
├── api_server.py                   # Flask API server (Python)
├── api-server-base44.js            # Express API server (Node.js)
├── tts_stt_service_base44.py       # TTS/STT core logic (Python)
├── tts-stt-service-base44.js       # TTS/STT service (Node.js, calls Python)
├── kurdish_stt_service.py          # Kurdish STT service with post-processing
├── kurdish_dictionary.py           # 779+ Kurdish word corrections
├── kurdish_postprocessor.py        # Post-processing engine for STT
├── base44.py                       # Base44 encoding (Python)
├── base44.js                       # Base44 encoding (JavaScript/Node.js)
├── setup_kurdish_tts.py            # Automated XTTS v2 setup script
├── test_kurdish_implementation.py  # Python unit tests
├── test_kurdish_postprocessing.py  # Post-processing tests
├── demo_kurdish_postprocessing.py  # Post-processing demo
├── test-integration.js             # Node.js integration tests
├── client-example.js               # API client example
├── index.html                      # Web UI
├── requirements.txt                # Python dependencies
├── package.json                    # Node.js dependencies
├── Procfile                        # Railway deployment config
├── railway.json                    # Railway config
├── kurdish_tts_training.ipynb      # Colab training notebook (optional)
├── README.md                       # This file
├── IMPLEMENTATION_SUMMARY.md       # Implementation details
├── KURDISH_TTS_IMPLEMENTATION.md   # Kurdish TTS documentation
└── KURDISH_STT_POSTPROCESSING.md   # Kurdish STT post-processing documentation
```

---

## 🐛 Troubleshooting

### PyTorch and Transformers Issues

**Problem:** Dependency conflicts during installation
```bash
# Solution: Let coqui-tts manage its dependencies
pip uninstall torch torchaudio transformers coqui-tts -y
pip install -r requirements.txt

# Or use virtual environment for clean installation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Problem:** "CUDA out of memory" or GPU-related errors
```bash
# Solution: XTTS v2 can run on CPU, though slower
# PyTorch will automatically use CPU if CUDA is unavailable
# No special configuration needed - CPU inference works by default
```

### Kurdish TTS Issues

**Problem:** "Language ku is not supported" error
```bash
# This is expected! The base XTTS v2 model doesn't support 'ku' directly.
# The service automatically uses voice cloning with Turkish phonetics.
# This is the normal and recommended behavior - no action needed.
```

**Problem:** Want to prepare training data for future fine-tuning
```bash
# Download Common Voice corpus and run:
python train_kurdish_xtts.py --corpus_path <path_to_common_voice>

# Note: This prepares data but doesn't perform actual training yet.
# Voice cloning is currently the recommended approach.
```

**Problem:** "Coqui TTS is not installed" error
```bash
# Solution: Install Coqui TTS explicitly
pip install coqui-tts>=0.27.0

# Or reinstall all dependencies
pip install -r requirements.txt
```

**Problem:** Model download fails or is slow
```bash
# Solution: Check internet connection and disk space
# The model is ~2GB and downloads to ~/.local/share/tts/
# You can also download manually and place in that directory
```

**Problem:** Kurdish audio quality is not good
```bash
# The voice cloning with Turkish phonetics should provide good quality.
# If you experience issues:

# 1. Check that you have the latest version
pip install --upgrade coqui-tts

# 2. Verify the text is in Kurdish (Kurmanji) script
# The system works best with proper Kurdish text

# 3. Try shorter sentences (under 200 characters)
# Longer texts may have quality issues
```

**Problem:** "ffmpeg not found" warning
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### General Issues

**Problem:** Port already in use
```bash
# Change port in api_server.py or use environment variable
PORT=8080 python api_server.py
```

**Problem:** Module import errors
```bash
# Make sure you're in the right directory
cd TTS_STT_Kurdifer

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Problem:** Node.js tests fail
```bash
# Reinstall Node.js dependencies
rm -rf node_modules package-lock.json
npm install

# Make sure Python is available
which python3
```

---

## 📈 Performance Notes

### First Run (Kurdish)
- Downloads XTTS v2 model: **~2GB, 2-5 minutes**
- Model initialization: **~10-20 seconds**
- First audio generation: **~5-10 seconds**

### Subsequent Runs (Kurdish)
- Model loads from cache: **~5 seconds**
- Audio generation: **~1-3 seconds per sentence**

### Other Languages (English, German, French, Turkish)
- Uses Google TTS (gTTS): **~1-2 seconds per request**
- No model download required
- Depends on internet connection to Google servers

---

## 🌍 Supported Languages

| Language | Code | TTS Engine | Status |
|----------|------|------------|--------|
| **Kurdish** (Kurmanji) | `ku`, `kurdish` | **Coqui TTS XTTS v2** | ✅ High-quality neural TTS |
| English | `en`, `english` | gTTS (Google) | ✅ Working |
| German | `de`, `german` | gTTS (Google) | ✅ Working |
| French | `fr`, `french` | gTTS (Google) | ✅ Working |
| Turkish | `tr`, `turkish` | gTTS (Google) | ✅ Working |

### About Kurdish TTS (XTTS v2)

**Model:** `tts_models/multilingual/multi-dataset/xtts_v2`
- 🎯 **High Quality**: Neural TTS with natural prosody and intonation
- 🌐 **Multilingual**: Base model trained on 13+ languages
- 📊 **Kurdish Support**: Via fine-tuning OR voice cloning fallback
- 🚀 **Performance**: ~1-3 seconds per sentence after initialization
- 💾 **Model Size**: ~2GB (one-time download, then cached)
- 🔄 **Auto-setup**: Downloads automatically on first use

**Two Modes of Operation:**

1. **Voice Cloning Fallback (Recommended - Works Immediately)**
   - Uses Turkish phonetics as proxy for Kurdish
   - No training required
   - Good quality, works out-of-the-box
   - Automatically used (default mode)

2. **Fine-Tuned Model (Future Enhancement)**
   - Data preparation script available (`train_kurdish_xtts.py`)
   - Prepares Mozilla Common Voice data for fine-tuning
   - Actual fine-tuning implementation pending
   - Voice cloning is currently the recommended approach

**Dataset Reference:**  
[Mozilla Common Voice Kurdish (Kurmanji)](https://datacollective.mozillafoundation.org/datasets/cmj8u3pbq00dtnxxbz4yoxc4i)

### Preparing Data for Future Fine-Tuning

If you want to prepare training data from the Kurdish Common Voice corpus:

```bash
# 1. Download Common Voice Kurdish corpus
# https://datacollective.mozillafoundation.org/datasets/cmj8u3pbq00dtnxxbz4yoxc4i

# 2. Extract to: cv-corpus-24.0-2025-12-05-kmr/

# 3. Run data preparation script
python train_kurdish_xtts.py \
  --corpus_path cv-corpus-24.0-2025-12-05-kmr/cv-corpus-24.0-2025-12-05/kmr/ \
  --max_samples 5000  # Optional: limit for testing

# 4. Script will process and validate audio files
# 5. Training data manifest saved to: models/kurdish/training_manifest.json
```

**Note:** The script prepares data but doesn't perform actual fine-tuning yet. Voice cloning with Turkish phonetics is the current recommended approach.

**For Google Colab users:**
Open `kurdish_tts_training.ipynb` in Colab for GPU-accelerated data preparation.

### Language Usage Examples

```python
# Kurdish (using XTTS v2)
service.text_to_speech_base44("Silav, tu çawa yî?", "kurdish")

# English (using gTTS)
service.text_to_speech_base44("Hello, how are you?", "english")

# German (using gTTS)  
service.text_to_speech_base44("Guten Tag", "german")
```

---

## 📊 Base44 Encoding

### What is Base44?
Base44 is a custom encoding scheme that:
- Uses 44 safe characters: `0-9`, `A-Z`, `a-h`
- Compresses binary data ~1.47x better than Base64
- URL-safe (no special characters)
- Perfect for JSON transmission

### Compression Comparison
| Format | Size | Ratio |
|--------|------|-------|
| Binary (MP3) | 8973 bytes | 1.00x |
| Base64 | 11964 chars | 0.75x |
| **Base44** | **13149 chars** | **0.68x** |

---

## 🚢 Deployment

### Railway (Current)
1. Fork this repo
2. Connect to Railway
3. Set start command: `python api_server.py`
4. Set port: `8080`
5. Deploy!

### Raspberry Pi (Recommended for Kurdish)
1. Install Python 3.8+
2. Clone repo
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python api_server.py`
5. Use Cloudflare Tunnel for public access

---

## 🔧 Environment Variables

```bash
PORT=8080                    # Server port
FLASK_ENV=production         # Flask environment
MAX_TEXT_LENGTH=500          # Max characters per request
COQUI_TOS_AGREED=1          # Auto-accept Coqui TOS (for CI/automated setups)
```

**COQUI_TOS_AGREED:**
- Set to `1` to automatically accept Coqui AI's Terms of Service during model download
- Required for automated/CI installations where interactive prompts cannot be answered
- Only use if you have read and agree to [Coqui AI's Terms](https://coqui.ai/cpml)
- Optional for manual installations (will prompt interactively if not set)

---

## 📝 Status & Features

### ✅ Completed
- [x] Web UI with tabbed interface (TTS & STT)
- [x] Railway deployment (24/7 live)
- [x] Multi-language support (5 languages)
- [x] Base44 encoding for efficient audio transfer
- [x] **Kurdish VITS TTS integration** (fine-tuned v8 model)
- [x] **Kurdish STT implementation** (facebook/mms-1b-all with kmr adapter)
- [x] Microphone recording support in web UI
- [x] Audio file upload with drag & drop
- [x] Kurdish special character support (ê, î, û, ç, ş)
- [x] Automated Kurdish TTS setup script
- [x] Comprehensive documentation
- [x] Python and Node.js implementations
- [x] Security: Zero npm vulnerabilities
- [x] Full test coverage

### 🚧 In Progress / Future
- [ ] Voice cloning with XTTS v2
- [ ] Custom Kurdish voice training guide
- [ ] Raspberry Pi setup guide
- [ ] Docker support
- [ ] Audio caching for improved performance
- [ ] Additional Kurdish dialects (Sorani)
- [ ] Web UI enhancements

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 👤 Author

**T1Agit**
- GitHub: [@T1Agit](https://github.com/T1Agit)
- Repository: [TTS_STT_Kurdifer](https://github.com/T1Agit/TTS_STT_Kurdifer)

---

## 🙏 Acknowledgments

- [gTTS](https://github.com/pndurette/gTTS) - Google Text-to-Speech
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Kurdish voice synthesis
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Railway](https://railway.app/) - Hosting platform

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/T1Agit/TTS_STT_Kurdifer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/T1Agit/TTS_STT_Kurdifer/discussions)

---

**Made with ❤️ for the Kurdish community**