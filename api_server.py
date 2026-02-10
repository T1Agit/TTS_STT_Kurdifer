"""
Flask API Server for Kurdish TTS/STT Service
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tts_stt_service_base44 import TTSSTTServiceBase44

app = Flask(__name__)
CORS(app)

service = TTSSTTServiceBase44()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Kurdish TTS/STT API',
        'version': '1.0',
        'status': 'running'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/tts', methods=['POST'])
def tts():
    data = request.json
    text = data.get('text')
    language = data.get('language', 'english')
    
    if not text:
        return jsonify({'success': False, 'error': 'Missing text'}), 400
    
    try:
        result = service.text_to_speech_base44(text, language)
        return jsonify({
            'success': True,
            'audio': result['audio'],
            'format': result['format'],
            'language': result['language']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stt', methods=['POST'])
def stt():
    data = request.json
    audio_base44 = data.get('audio')
    language = data.get('language', 'english')
    
    if not audio_base44:
        return jsonify({'success': False, 'error': 'Missing audio data'}), 400
    
    try:
        result = service.speech_to_text_from_base44(audio_base44, language)
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result['language'],
            'confidence': result.get('confidence', 1.0)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/languages', methods=['GET'])
def languages():
    return jsonify({'languages': list(service.SUPPORTED_LANGUAGES.keys())})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)