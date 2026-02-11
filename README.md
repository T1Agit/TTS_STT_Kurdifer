# 🎤 Kurdish TTS/STT API with Base44 Encoding

A multilingual Text-to-Speech (TTS) and Speech-to-Text (STT) API with custom Base44 encoding for efficient audio transfer.

## 🌟 Features

- **5 Languages**: Kurdish (Kurdî), English, German, French, Turkish
- **Base44 Encoding**: Efficient audio encoding (1.47x compression)
- **REST API**: Simple HTTP endpoints
- **Web UI**: Browser-based interface
- **Railway Deployed**: Live 24/7

---

## 🚀 Live Demo

### 🌐 Web Interface
**[https://t1agit.github.io/TTS_STT_Kurdifer/](https://t1agit.github.io/TTS_STT_Kurdifer/)**

### 🔗 API Endpoint
**[https://ttststtkurdifer-production.up.railway.app](https://ttststtkurdifer-production.up.railway.app)**

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

---

## 🖥️ Quick Start

### Option 1: Use the Web Interface
1. Go to **[https://t1agit.github.io/TTS_STT_Kurdifer/](https://t1agit.github.io/TTS_STT_Kurdifer/)**
2. Enter text
3. Select language
4. Click "Generate Speech"
5. Listen to the audio!

### Option 2: Use cURL
```bash
curl -X POST https://ttststtkurdifer-production.up.railway.app/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Silav, tu çawa yî?", "language": "kurdish"}'
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

- **Python**: 3.8 - 3.12 (tested with 3.12.3)
- **Disk Space**: ~2GB for XTTS v2 model (Kurdish TTS)
- **Internet**: Required for first-time model download

### 1. Clone Repository
```bash
git clone https://github.com/T1Agit/TTS_STT_Kurdifer.git
cd TTS_STT_Kurdifer
```

### 2. Install Dependencies

**Important**: Kurdish TTS requires PyTorch and XTTS v2:

```bash
# Install all dependencies (includes PyTorch 2.x and Coqui TTS)
pip install -r requirements.txt

# This will install:
# - PyTorch 2.x and torchaudio (for XTTS v2)
# - Coqui TTS 0.27.x (with XTTS v2 model support)
# - gTTS (for other languages)
# - Flask and other dependencies
```

**Version Notes**:
- `torch>=2.0.0,<2.5.0` - Tested and verified with 2.0-2.4.1. Versions capped at <2.5.0 for compatibility with Coqui TTS 0.27.x (2.5.x untested, 2.6+ has breaking changes to `torch.load()` API).
- `torchaudio>=2.0.0,<2.5.0` - Must match torch version for compatibility
- `transformers>=4.33.0,<5.0.0` - Version 5.x removed APIs (like `isin_mps_friendly`) needed by Coqui TTS 0.27.x
- `coqui-tts>=0.27.0,<0.28.0` - XTTS v2 multilingual model support

### 3. Setup Kurdish TTS (Optional Test)

Test the XTTS v2 model installation:

```bash
# Set environment variable to agree to non-commercial license
export COQUI_TOS_AGREED=1

# Run setup script
python setup_kurdish_tts.py
```

This will:
- Download XTTS v2 model (~2GB, first time only)
- Prompt for license agreement (or skip with COQUI_TOS_AGREED=1)
- Test Kurdish TTS with sample phrase
- Verify model is working correctly

**License Note**: XTTS v2 uses [Coqui Public Model License (CPML)](https://coqui.ai/cpml) for non-commercial use. Commercial use requires contacting licensing@coqui.ai.

### 4. Run Server
```bash
python api_server.py
```

Server runs on: `http://localhost:5000`

### 5. Test TTS
```bash
python tts_stt_service_base44.py
```

---

## 🎯 XTTS v2 & Kurdish Support

### What is XTTS v2?

**XTTS v2** (eXtensible Text-to-Speech version 2) is Coqui TTS's state-of-the-art multilingual TTS model:

- **Model**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Languages**: 16+ languages including **Kurdish (Kurmanji)**
- **Quality**: Neural TTS with natural prosody and intonation
- **Size**: ~2GB (downloads automatically on first use)
- **Speed**: 1-3 seconds per sentence after initialization

### Kurdish Language Support

This project uses XTTS v2 for **Kurdish (Kurmanji)** text-to-speech:

```python
from tts_stt_service_base44 import TTSSTTServiceBase44

service = TTSSTTServiceBase44()
result = service.text_to_speech_base44("Silav, tu çawa yî?", "kurdish")
# Returns Base44-encoded MP3 audio
```

**Features**:
- ✅ Native Kurdish (Kurmanji) support via XTTS v2
- ✅ Automatic model download and caching
- ✅ High-quality neural voice synthesis
- ✅ No training required (pre-trained model)
- ✅ Works offline after initial download

**Language Routing**:
- **Kurdish (`ku`)**: Uses Coqui TTS with XTTS v2 model
- **Other languages** (`en`, `de`, `fr`, `tr`): Uses Google TTS (gTTS)

### First-Time Setup

The XTTS v2 model downloads automatically when you first use Kurdish TTS:

```bash
# Test installation and download model
python setup_kurdish_tts.py

# Or just use the API - model downloads on first Kurdish TTS request
python api_server.py
```

**Download Progress**:
- Size: ~2GB
- Time: 2-5 minutes (depends on internet speed)
- Location: Cached in `~/.local/share/tts/` (Linux/Mac) or `%USERPROFILE%\.local\share\tts\` (Windows)
- One-time only: Subsequent calls use cached model

### Model Compatibility

**Supported PyTorch Versions**:
- ✅ PyTorch 2.0.x - 2.4.x (tested and recommended: 2.4.1)
- ❌ PyTorch 2.5.x - Not supported (untested with Coqui TTS 0.27.x)
- ❌ PyTorch 2.6+ - Not supported (breaking changes with `torch.load()`)

**Version Resolution**:
```bash
# Check your versions
pip list | grep -E "torch|coqui"

# If you have PyTorch 2.6+, downgrade to 2.4.1:
pip install torch==2.4.1 torchaudio==2.4.1
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
├── api_server.py              # Flask API server
├── tts_stt_service_base44.py  # TTS/STT core logic
├── base44.py                  # Base44 encoding (Python)
├── base44.js                  # Base44 encoding (JavaScript)
├── index.html                 # Web UI
├── requirements.txt           # Python dependencies
├── Procfile                   # Railway deployment config
├── kurdish_tts_training.ipynb # Colab training notebook
└── README.md                  # This file
```

---

## 🌍 Supported Languages

| Language | Code | TTS Engine | Model |
|----------|------|------------|-------|
| English | `en`, `english` | gTTS (Google) | Online API |
| German | `de`, `german` | gTTS (Google) | Online API |
| French | `fr`, `french` | gTTS (Google) | Online API |
| Turkish | `tr`, `turkish` | gTTS (Google) | Online API |
| **Kurdish (Kurmanji)** | `ku`, `kurdish` | **Coqui TTS** | **XTTS v2** (~2GB) |

**Kurdish TTS Details**:
- Uses state-of-the-art XTTS v2 multilingual model
- No training required - pre-trained model included
- Automatic download on first use (~2GB, cached locally)
- High-quality neural voice with natural prosody
- Works offline after initial download

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
PORT=8080                     # Server port
FLASK_ENV=production          # Flask environment
MAX_TEXT_LENGTH=500           # Max characters per request
COQUI_TOS_AGREED=1            # Auto-agree to Coqui CPML license (non-commercial)
```

---

## 🔧 Troubleshooting

### PyTorch Version Issues

**Problem**: Import errors or model loading failures

**Solution**:
```bash
# Check your PyTorch version
pip list | grep torch

# If PyTorch >= 2.5, downgrade to 2.4.1
pip install torch==2.4.1 torchaudio==2.4.1
```

**Why**: PyTorch 2.5.x is untested with Coqui TTS 0.27.x. PyTorch 2.6+ has breaking changes to `torch.load()` API that are not compatible.

### Transformers Compatibility

**Problem**: `ImportError: cannot import name 'isin_mps_friendly'`

**Solution**:
```bash
# Downgrade transformers to 4.x
pip install 'transformers>=4.33.0,<5.0.0'
```

**Why**: Coqui TTS 0.27.x requires transformers 4.x. Version 5.x removed some APIs.

### License Agreement Prompt

**Problem**: Model download prompts for license agreement

**Solution**:
```bash
# Set environment variable to auto-agree (non-commercial use)
export COQUI_TOS_AGREED=1

# Then run your script
python setup_kurdish_tts.py
```

**Note**: This agrees to the non-commercial [Coqui CPML license](https://coqui.ai/cpml). For commercial use, contact licensing@coqui.ai.

### Model Download Failures

**Problem**: Network errors during model download

**Solution**:
1. Check internet connection
2. Ensure firewall allows access to huggingface.co
3. Verify ~2GB free disk space
4. Retry after some time if servers are busy

**Cache Location**:
- Linux/Mac: `~/.local/share/tts/`
- Windows: `%USERPROFILE%\.local\share\tts\`

### Missing Dependencies

**Problem**: `ImportError: No module named 'torch'` or similar

**Solution**:
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# If issues persist, use a clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📝 To-Do

- [x] Web UI
- [x] Railway deployment
- [x] Multi-language support
- [x] Base44 encoding
- [ ] Kurdish voice training guide
- [ ] Speech-to-Text (STT) implementation
- [ ] Voice cloning
- [ ] Raspberry Pi setup guide
- [ ] Docker support

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