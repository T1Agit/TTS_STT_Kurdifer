# 🎤 TTS/STT Service with Base44 Encoding

A comprehensive Text-to-Speech (TTS) and Speech-to-Text (STT) service with Base44 encoding support for multiple languages including Kurdish, German, French, English, and Turkish.

## 🌍 Supported Languages

| Language | Code | Flag |
|----------|------|------|
| Kurdish  | ku   | 🟥⚪🟩 |
| German   | de   | 🇩🇪 |
| French   | fr   | 🇫🇷 |
| English  | en   | 🇬🇧 |
| Turkish  | tr   | 🇹🇷 |

## ✨ Features

- 🎯 **Multi-language Support**: Process text and audio in 5 different languages
- 🔐 **Base44 Encoding**: Efficient binary-to-text encoding with 44-character alphabet
- 🎤 **Text-to-Speech**: Convert text to natural-sounding speech
- 🎧 **Speech-to-Text**: Transcribe audio to text (with API integration)
- 🚀 **REST API**: Easy-to-use HTTP endpoints
- 📦 **Batch Processing**: Convert multiple texts simultaneously
- 🐍 **Python & JavaScript**: Full implementation in both languages
- 💾 **File Operations**: Save and load audio files
- 🌐 **CORS Enabled**: Ready for web applications

## 📋 Installation

### Python Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Note: You may need system dependencies for audio processing
# On Ubuntu/Debian:
sudo apt-get install ffmpeg portaudio19-dev python3-pyaudio

# On macOS:
brew install ffmpeg portaudio
```

### JavaScript/Node.js Setup

```bash
# Install Node.js dependencies
npm install

# Or using yarn
yarn install
```

## 🚀 Quick Start

### Starting the API Server

```bash
# Start the server
npm start

# Or for development with auto-reload
npm run dev
```

The server will start on `http://localhost:3000`

### Running Examples

```bash
# Test Base44 encoding
node base44.js
python base44.py

# Test TTS/STT service
npm run demo

# Test API client
npm run client
```

## 📚 API Documentation

### Health Check

Check if the server is running and get service information.

```bash
curl http://localhost:3000/health
```

**Response:**
```json
{
  "status": "healthy",
  "encoding": "Base44",
  "supportedLanguages": ["ku", "de", "fr", "en", "tr"],
  "endpoints": [
    "GET /health",
    "GET /api/languages",
    "POST /api/tts",
    "POST /api/stt",
    "POST /api/tts/batch"
  ],
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

### Get Supported Languages

Get a list of all supported languages.

```bash
curl http://localhost:3000/api/languages
```

**Response:**
```json
{
  "success": true,
  "languages": ["ku", "de", "fr", "en", "tr"],
  "count": 5
}
```

### Text to Speech

Convert text to speech with Base44 encoding.

```bash
curl -X POST http://localhost:3000/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you today?",
    "language": "en"
  }'
```

**Request Body:**
- `text` (required): Text to convert to speech
- `language` (optional): Target language (default: "en")

**Response:**
```json
{
  "success": true,
  "data": {
    "audio": "PAX9PFN5ObNNUJFAVZF...",
    "language": "en",
    "format": "mp3",
    "text": "Hello, how are you today?",
    "size": 12345,
    "encodedSize": 18000,
    "compressionRatio": 1.46
  }
}
```

### Speech to Text

Convert Base44 encoded audio to text.

```bash
curl -X POST http://localhost:3000/api/stt \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "PAX9PFN5ObNNUJFAVZF...",
    "language": "en"
  }'
```

**Request Body:**
- `audio` (required): Base44 encoded audio data
- `language` (optional): Source language (default: "en")

**Response:**
```json
{
  "success": true,
  "data": {
    "text": "Hello, how are you today?",
    "language": "en",
    "confidence": 1.0,
    "audioSize": 12345
  }
}
```

### Batch Text to Speech

Convert multiple texts to speech at once.

```bash
curl -X POST http://localhost:3000/api/tts/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Hello World",
      "Good morning",
      "Thank you"
    ],
    "language": "en"
  }'
```

**Request Body:**
- `texts` (required): Array of texts to convert
- `language` (optional): Target language (default: "en")

**Response:**
```json
{
  "success": true,
  "count": 3,
  "data": [
    {
      "audio": "PAX9PFN5...",
      "language": "en",
      "format": "mp3",
      "text": "Hello World",
      "size": 8000,
      "encodedSize": 11680,
      "compressionRatio": 1.46
    },
    ...
  ]
}
```

## 💻 Code Examples

### JavaScript Client

```javascript
const { TTSSTTClient } = require('./client-example');

// Create client
const client = new TTSSTTClient('http://localhost:3000');

// Text to Speech
const ttsResult = await client.textToSpeech(
  'Silav, tu çawa yî?',
  'kurdish'
);
console.log(ttsResult.data.audio); // Base44 encoded audio

// Speech to Text
const sttResult = await client.speechToText(
  ttsResult.data.audio,
  'kurdish'
);
console.log(sttResult.data.text); // Transcribed text

// Batch processing
const batchResult = await client.batchTextToSpeech(
  ['Hello', 'Goodbye', 'Thank you'],
  'english'
);
console.log(`Processed ${batchResult.count} texts`);

// Get supported languages
const languages = await client.getSupportedLanguages();
console.log(languages.languages); // ['ku', 'de', 'fr', 'en', 'tr']
```

### Python Usage

```python
from tts_stt_service_base44 import TTSSTTServiceBase44

# Create service instance
service = TTSSTTServiceBase44()

# Text to Speech
result = service.text_to_speech_base44(
    text="Silav, tu çawa yî?",
    language="kurdish",
    format="mp3"
)
print(f"Audio encoded: {result['encoded_size']} chars")

# Save audio to file
service.save_audio_from_base44(
    result['audio'],
    'output.mp3'
)

# Speech to Text from file
transcription = service.speech_to_text_from_file(
    'output.mp3',
    language='kurdish'
)
print(f"Transcribed: {transcription['text']}")
```

### Base44 Encoding Examples

#### JavaScript
```javascript
const { encode, decode } = require('./base44');

// Encode
const data = Buffer.from('Hello, World!');
const encoded = encode(data);
console.log(encoded); // "PAX9PFN5ObNNUJFAVZF"

// Decode
const decoded = decode(encoded);
console.log(decoded.toString()); // "Hello, World!"
```

#### Python
```python
from base44 import encode, decode

# Encode
data = b"Hello, World!"
encoded = encode(data)
print(encoded)  # "PAX9PFN5ObNNUJFAVZF"

# Decode
decoded = decode(encoded)
print(decoded.decode())  # "Hello, World!"
```

## 🔤 Base44 Encoding

Base44 is a binary-to-text encoding scheme that uses 44 characters from the alphabet:
```
ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefgh-_
```

### Why Base44?

- **Efficient**: ~1.46x size increase (compared to 1.33x for Base64)
- **URL-safe**: Uses only alphanumeric characters and safe symbols
- **Readable**: Uses familiar characters
- **No padding**: Handles leading zeros correctly

### Compression Ratios

| Original Format | Size | Base44 Size | Ratio |
|----------------|------|-------------|-------|
| MP3 Audio      | 10KB | ~14.6KB     | 1.46x |
| Text           | 1KB  | ~1.46KB     | 1.46x |
| Binary Data    | 5KB  | ~7.3KB      | 1.46x |

## 🌐 Example Texts

### Kurdish
- "Silav, tu çawa yî?" (Hello, how are you?)
- "Silav, tu îro çawa yî?" (Hello, how are you today?)
- "Spas dikim" (Thank you)

### German
- "Guten Tag, wie geht es Ihnen?" (Good day, how are you?)
- "Danke schön" (Thank you)

### French
- "Bonjour, comment allez-vous aujourd'hui?" (Hello, how are you today?)
- "Merci beaucoup" (Thank you very much)

### Turkish
- "Merhaba, bugün nasılsınız?" (Hello, how are you today?)
- "Teşekkür ederim" (Thank you)

### English
- "Hello, how are you today?"
- "Thank you very much"

## 🧪 Testing

### Run All Tests

```bash
# Test Base44 encoding
npm test

# Test individual components
node base44.js                    # Base44 encoding tests
node tts-stt-service-base44.js   # TTS/STT service demo
python base44.py                  # Python Base44 tests
python tts_stt_service_base44.py # Python TTS demo
```

### Manual Testing

1. Start the server: `npm start`
2. In another terminal, run the client: `npm run client`
3. Check the output files: `output_*.mp3`

## 🔧 Configuration

### Environment Variables

- `PORT`: Server port (default: 3000)

```bash
PORT=8080 npm start
```

### Server Configuration

Edit `api-server-base44.js`:
- `JSON body limit`: Currently 50MB
- `CORS`: Enabled for all origins (modify for production)

## 📁 Project Structure

```
.
├── base44.js                      # JavaScript Base44 implementation
├── base44.py                      # Python Base44 implementation
├── tts-stt-service-base44.js     # Node.js TTS/STT service
├── tts_stt_service_base44.py     # Python TTS/STT service
├── api-server-base44.js          # Express REST API server
├── client-example.js             # API client example
├── package.json                   # Node.js dependencies
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🐛 Troubleshooting

### Common Issues

**"Module not found" error:**
```bash
npm install
```

**Python audio errors:**
```bash
# Install system dependencies
sudo apt-get install ffmpeg portaudio19-dev  # Ubuntu/Debian
brew install ffmpeg portaudio                 # macOS
```

**Port already in use:**
```bash
# Use a different port
PORT=8080 npm start
```

**Speech recognition not working:**
The speech-to-text feature requires integration with an external API (e.g., Google Speech-to-Text). The current implementation includes a placeholder that needs to be replaced with actual API integration.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

---

## 🔗 API Quick Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/health` | Health check |
| GET    | `/api/languages` | Get supported languages |
| POST   | `/api/tts` | Text to speech |
| POST   | `/api/stt` | Speech to text |
| POST   | `/api/tts/batch` | Batch text to speech |

## 💡 Tips

- Use batch processing for multiple texts to improve performance
- Base44 encoded audio can be safely transmitted over HTTP
- Kurdish language uses Google's Kurdish (Kurmanji) TTS
- For production, consider adding authentication to API endpoints
- Use environment variables for sensitive configuration

---

Made with ❤️ for multilingual speech processing