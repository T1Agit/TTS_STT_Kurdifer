"""
Flask API Server for Kurdish TTS/STT Service
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tts_stt_service_base44 import TTSSTTServiceBase44

app = Flask(__name__, static_folder='.')
CORS(app)

service = TTSSTTServiceBase44()

@app.route('/')
def index():
    """Serve the main web UI"""
    return send_from_directory('.', 'index.html')

@app.route('/base44.js')
def base44_js():
    """Serve the base44.js library"""
    return send_from_directory('.', 'base44.js')

@app.route('/api', methods=['GET'])
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
    model = data.get('model', None)  # Optional model selection for Kurdish
    
    if not text:
        return jsonify({'success': False, 'error': 'Missing text'}), 400
    
    try:
        result = service.text_to_speech_base44(text, language, model_version=model)
        return jsonify({
            'success': True,
            'audio': result['audio'],
            'format': result['format'],
            'language': result['language'],
            'model': result.get('model', 'default')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/languages', methods=['GET'])
def languages():
    return jsonify({'languages': list(service.SUPPORTED_LANGUAGES.keys())})

@app.route('/models', methods=['GET'])
def models():
    """Get available models for a language"""
    language = request.args.get('language', 'kurdish')
    try:
        available_models = service.get_available_models(language)
        return jsonify({'success': True, 'data': available_models})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)