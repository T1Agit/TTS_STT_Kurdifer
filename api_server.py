"""
Flask API Server for Kurdish TTS/STT Service
"""

import os
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tts_stt_service_base44 import TTSSTTServiceBase44
from base44 import decode

app = Flask(__name__, static_folder='.')
CORS(app)

service = TTSSTTServiceBase44()

# Lazy load STT service
_stt_service = None

def get_stt_service():
    """Lazy initialization of STT service"""
    global _stt_service
    if _stt_service is None:
        try:
            from kurdish_stt_service import KurdishSTTService
            _stt_service = KurdishSTTService()
            print("✅ Kurdish STT service initialized")
        except Exception as e:
            print(f"⚠️  Could not initialize STT service: {e}")
            _stt_service = False
    return _stt_service if _stt_service is not False else None

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
    voice_preset = data.get('voice_preset', 'default')  # Support voice presets
    
    # Default model_version based on language
    if language in ['kurdish', 'ku']:
        model_version = data.get('model_version', 'trained_v8')  # Kurdish default: trained_v8
    else:
        model_version = data.get('model_version', 'original')  # Other languages: original
    
    if not text:
        return jsonify({'success': False, 'error': 'Missing text'}), 400
    
    # Validate model_version is only used for Kurdish
    if 'model_version' in data and model_version != 'original' and language not in ['kurdish', 'ku']:
        return jsonify({
            'success': False, 
            'error': 'model_version parameter is only supported for Kurdish language'
        }), 400
    
    # Validate voice_preset values
    valid_presets = ['default', 'elderly_male', 'elderly_female']
    if voice_preset not in valid_presets:
        return jsonify({
            'success': False,
            'error': f'Invalid voice_preset. Must be one of: {", ".join(valid_presets)}'
        }), 400
    
    try:
        result = service.text_to_speech_base44(
            text, 
            language,
            model_version=model_version,
            voice_preset=voice_preset
        )
        return jsonify({
            'success': True,
            'audio': result['audio'],
            'format': result['format'],
            'language': result['language'],
            'model': result.get('model', 'unknown'),
            'engine': result.get('engine', 'unknown'),
            'voice_preset': voice_preset
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/languages', methods=['GET'])
def languages():
    return jsonify({'languages': list(service.SUPPORTED_LANGUAGES.keys())})

@app.route('/models', methods=['GET'])
def models():
    """List available TTS models for Kurdish"""
    try:
        vits_service = service._get_vits_service()
        if vits_service:
            available_models = vits_service.list_available_models()
            return jsonify({
                'success': True,
                'models': available_models
            })
        else:
            return jsonify({
                'success': False,
                'error': 'VITS service not available'
            }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stt', methods=['POST'])
def stt():
    """Speech-to-Text endpoint for Kurdish audio transcription"""
    try:
        # Get STT service
        stt_service = get_stt_service()
        if stt_service is None:
            return jsonify({
                'success': False,
                'error': 'STT service not available. Model may not be loaded.'
            }), 503
        
        # Check if audio data is provided in JSON (Base44 encoded)
        if request.is_json:
            data = request.json
            audio_base44 = data.get('audio')
            
            if not audio_base44:
                return jsonify({
                    'success': False,
                    'error': 'Missing audio data'
                }), 400
            
            # Decode Base44 to audio bytes
            audio_bytes = decode(audio_base44)
        
        # Check if audio file is uploaded
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400
            
            # Read file bytes
            audio_bytes = audio_file.read()
        
        else:
            return jsonify({
                'success': False,
                'error': 'No audio data provided. Send Base44 encoded audio in JSON or upload a file.'
            }), 400
        
        # Transcribe audio
        result = stt_service.transcribe_from_bytes(audio_bytes, return_confidence=True)
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result['language'],
            'confidence': result.get('confidence', 1.0),
            'duration': result.get('duration', 0)
        })
        
    except Exception as e:
        print(f"❌ STT Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stt/status', methods=['GET'])
def stt_status():
    """Check if STT service is available"""
    stt_service = get_stt_service()
    if stt_service is not None:
        return jsonify({
            'success': True,
            'available': True,
            'model': 'facebook/mms-1b-all',
            'language': 'kmr (Kurdish Kurmanji)'
        })
    else:
        return jsonify({
            'success': True,
            'available': False,
            'message': 'STT service not loaded. Model download may be required.'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)