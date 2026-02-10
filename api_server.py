from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from tts_stt_service_base44 import TTSSTTServiceBase44

app = Flask(__name__)
CORS(app)

@app.route('/tts', methods=['POST'])
def tts():
    data = request.json
    text = data.get('text')
    # Implement the logic to call TTSSTTServiceBase44 and return audio
    audio_data = TTSSTTServiceBase44.text_to_speech(text)
    return jsonify({"audio": base64.b64encode(audio_data).decode('utf-8')}), 200

@app.route('/stt', methods=['POST'])
def stt():
    data = request.json
    audio_base64 = data.get('audio')
    audio_data = base64.b64decode(audio_base64)
    # Implement the logic to call TTSSTTServiceBase44 and return text
    text = TTSSTTServiceBase44.speech_to_text(audio_data)
    return jsonify({"text": text}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/languages', methods=['GET'])
def languages():
    # Implement logic to return supported languages
    languages = TTSSTTServiceBase44.get_supported_languages()
    return jsonify({"languages": languages}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)