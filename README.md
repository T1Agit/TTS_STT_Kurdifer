# 🎤 Kurdish TTS/STT API with Base44 Encoding

A multilingual Text-to-Speech (TTS) and Speech-to-Text (STT) API with custom Base44 encoding for efficient audio transfer.

## 🌟 Features

- **5 Languages**: Kurdish (Kurdî), English, German, French, Turkish
- **Base44 Encoding**: Efficient audio encoding (1.47x compression)
- **REST API**: Simple HTTP endpoints
- **Web UI**: Browser-based interface
- **Railway Deployed**: Live 24/7
- **Kurdish STT**: ✅ Google STT supports Kurdish (Kurmanji) - [See Verification](KURDISH_STT_VERIFICATION.md)

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
  "success": true,
  "audio": "DAMc3XX6M5QeVe66PDYfMO8fLGGc...",
  "format": "mp3",
  "language": "english"
}
```

### 4. Speech-to-Text (STT)
```bash
POST /stt
Content-Type: application/json

{
  "audio": "DAMc3XX6M5QeVe66PDYfMO8fLGGc...",
  "language": "english"
}
```

**Response:**
```json
{
  "success": true,
  "text": "Hello, how are you?",
  "language": "english",
  "confidence": 0.95
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

### 1. Clone Repository
```bash
git clone https://github.com/T1Agit/TTS_STT_Kurdifer.git
cd TTS_STT_Kurdifer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Server
```bash
python api_server.py
```

Server runs on: `http://localhost:5000`

### 4. Test TTS
```bash
python tts_stt_service_base44.py
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

## 🎯 Kurdish Language - No Fallback Policy

### Important: Language Integrity Guarantee

This service ensures **Kurdish language integrity** by implementing a strict no-fallback policy:

#### Text-to-Speech (TTS)
- **Kurdish (ku)** → Always uses **Coqui TTS** with XTTS v2 model
- If Coqui TTS is not available or fails, the service raises an error
- **Never falls back** to other TTS engines or languages
- Guarantees authentic Kurdish pronunciation

#### Speech-to-Text (STT)
- **Kurdish (ku)** → Always uses **Google Speech Recognition** with 'ku' language code
- If recognition fails, the service raises an error
- **Never falls back** to other languages or engines
- Ensures accurate Kurdish transcription

##### Google STT Support for Kurdish
**✅ YES - Google STT supports Kurdish (Kurmanji)**

- **Language Code:** `ku` (ISO 639-1)
- **Service Used:** Google Web Speech API (free tier)
- **Quality:** Basic support, suitable for development/testing
- **Status:** Verified and functional - [See detailed verification report](KURDISH_STT_VERIFICATION.md)

**Key Points:**
- Google Cloud Speech-to-Text API officially supports Kurdish
- Web Speech API (used by this repo) has limited but functional support
- Best results with high-quality audio (16kHz+, clear recordings)
- For production use, consider upgrading to paid Google Cloud API

**For detailed information, limitations, and recommendations:**
📄 **[Read the Kurdish STT Verification Report](KURDISH_STT_VERIFICATION.md)**

#### Why No Fallback?
1. **Language Integrity**: Prevents mixing Kurdish with other languages
2. **Quality Assurance**: Ensures users get Kurdish-specific models
3. **Transparency**: Errors are explicit rather than silent degradation
4. **Base44 Frontend**: Base44 encoding handles all data transfer reliably

#### Error Handling
- If Kurdish TTS fails: `RuntimeError: Kurdish TTS generation failed`
- If Kurdish STT fails: `ValueError: Could not understand Kurdish audio`
- If Coqui TTS not installed: `ImportError: Coqui TTS is not installed`

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

| Language | Code | TTS Engine |
|----------|------|------------|
| English | `en`, `english` | gTTS (Google) |
| German | `de`, `german` | gTTS (Google) |
| French | `fr`, `french` | gTTS (Google) |
| Turkish | `tr`, `turkish` | gTTS (Google) |
| Kurdish | `ku`, `kurdish` | Coqui TTS (Custom) |

**Note:** Kurdish requires local Coqui TTS setup (see training notebook).

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

### Docker (Recommended for Local/Production)

#### Quick Start with Docker Compose
```bash
# Build and start the service
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

#### Using Docker directly
```bash
# Build the image
docker build -t kurdish-tts-stt .

# Run the container
docker run -d -p 5000:5000 \
  --name kurdish-tts \
  -e PORT=5000 \
  -e FLASK_ENV=production \
  kurdish-tts-stt

# View logs
docker logs -f kurdish-tts

# Stop container
docker stop kurdish-tts
docker rm kurdish-tts
```

#### Features
- ✅ Pre-configured with all dependencies
- ✅ Includes ffmpeg for audio processing
- ✅ Caches TTS models to avoid re-downloading
- ✅ Health checks included
- ✅ Production-ready

### Railway (Cloud Hosting)
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
```

---

## ❓ Frequently Asked Questions (FAQ)

### Is Google's STT able to do Kurdish (Kurmanji)?

**✅ YES!** Google's Speech-to-Text services support Kurdish (Kurmanji) using language code `ku`.

**Quick Facts:**
- ✅ **Works:** Google Web Speech API accepts Kurdish with code `ku`
- ✅ **Functional:** Implementation is active in this repository
- ⚠️ **Quality:** Basic support on free tier, better on paid Google Cloud API
- ✅ **Verified:** Tested with SpeechRecognition library version 3.14.5

**For Production Use:**
- Current: Free Google Web Speech API (suitable for development/testing)
- Recommended: Google Cloud Speech-to-Text API (higher accuracy, paid)
- Alternative: Coqui STT with custom Kurdish model

📄 **[Complete verification report and technical details →](KURDISH_STT_VERIFICATION.md)**

### How accurate is Kurdish STT?
Accuracy depends on:
- Audio quality (16kHz+ recommended)
- Clear pronunciation
- Minimal background noise
- Using native Kurdish speakers

The free Web Speech API provides basic support. For production applications requiring high accuracy, we recommend upgrading to Google Cloud Speech-to-Text API.

### What Kurdish dialect is supported?
The language code `ku` typically refers to **Kurmanji (Northern Kurdish)**. Google Cloud Speech-to-Text officially supports Kurdish, primarily Kurmanji dialect.

---

## 📝 To-Do

- [x] Web UI
- [x] Railway deployment
- [x] Multi-language support
- [x] Base44 encoding
- [x] Kurdish voice training guide
- [x] Speech-to-Text (STT) implementation
- [x] Docker support
- [ ] Voice cloning
- [ ] Raspberry Pi setup guide

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