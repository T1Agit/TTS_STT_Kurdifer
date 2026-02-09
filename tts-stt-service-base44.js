/**
 * TTS/STT Service with Base44 Encoding
 * 
 * Provides Text-to-Speech and Speech-to-Text functionality with Base44 encoding
 * for Kurdish, German, French, English, and Turkish languages.
 */

const gtts = require('node-gtts');
const { encode, decode } = require('./base44');
const fs = require('fs').promises;
const path = require('path');

class TTSSTTServiceBase44 {
    /**
     * TTS/STT Service with Base44 encoding support
     * 
     * Supports: Kurdish (ku), German (de), French (fr), English (en), Turkish (tr)
     */
    
    static SUPPORTED_LANGUAGES = {
        'kurdish': 'ku', 'german': 'de', 'french': 'fr',
        'english': 'en', 'turkish': 'tr',
        'ku': 'ku', 'de': 'de', 'fr': 'fr', 'en': 'en', 'tr': 'tr'
    };

    constructor() {
        this.ttsCache = {};
    }

    /**
     * Normalize language input to language code
     * @param {string} language - Language name or code
     * @returns {string} Language code (e.g., 'en', 'ku', 'de')
     */
    _getLanguageCode(language) {
        const langLower = language.toLowerCase().trim();
        if (!(langLower in TTSSTTServiceBase44.SUPPORTED_LANGUAGES)) {
            throw new Error(
                `Unsupported language: ${language}. ` +
                `Supported: ${Object.values(TTSSTTServiceBase44.SUPPORTED_LANGUAGES).join(', ')}`
            );
        }
        return TTSSTTServiceBase44.SUPPORTED_LANGUAGES[langLower];
    }

    /**
     * Check if language requires Coqui TTS (via Python)
     * @private
     * @param {string} langCode - Language code
     * @returns {boolean} True if language needs Coqui TTS
     */
    _usesCoquiTTS(langCode) {
        return langCode === 'ku';
    }

    /**
     * Generate speech using Python service (for Coqui TTS/Kurdish)
     * @private
     * @param {string} text - Text to convert
     * @param {string} langCode - Language code
     * @returns {Promise<Buffer>} Audio buffer
     */
    _generateSpeechPython(text, langCode) {
        return new Promise((resolve, reject) => {
            const { spawn } = require('child_process');
            
            // Escape text for Python string
            const escapedText = text.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n');
            
            const pythonCode = `
from tts_stt_service_base44 import TTSSTTServiceBase44
import json
import sys

try:
    service = TTSSTTServiceBase44()
    result = service.text_to_speech_base44("${escapedText}", "${langCode}")
    # Return only the essential data as JSON
    output = {
        'audio': result['audio'],
        'language': result['language'],
        'format': result['format'],
        'size': result['size'],
        'encoded_size': result['encoded_size']
    }
    print(json.dumps(output))
except Exception as e:
    print(json.dumps({'error': str(e)}), file=sys.stderr)
    sys.exit(1)
`;
            
            const python = spawn('python3', ['-c', pythonCode]);
            
            let stdout = '';
            let stderr = '';
            
            python.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            python.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            python.on('close', (code) => {
                if (code !== 0) {
                    reject(new Error(`Python TTS failed: ${stderr || 'Unknown error'}`));
                    return;
                }
                
                try {
                    // Find JSON in stdout (skip any log lines)
                    const lines = stdout.trim().split('\n');
                    const jsonLine = lines[lines.length - 1]; // Last line should be JSON
                    const result = JSON.parse(jsonLine);
                    
                    if (result.error) {
                        reject(new Error(result.error));
                        return;
                    }
                    
                    // Convert Base44 audio to Buffer
                    const { decode } = require('./base44');
                    const audioBuffer = decode(result.audio);
                    
                    resolve(audioBuffer);
                } catch (error) {
                    reject(new Error(`Failed to parse Python output: ${error.message}`));
                }
            });
            
            python.on('error', (error) => {
                reject(new Error(`Failed to spawn Python: ${error.message}`));
            });
        });
    }

    /**
     * Generate speech using node-gtts
     * @private
     * @param {string} text - Text to convert
     * @param {string} langCode - Language code
     * @returns {Promise<Buffer>} Audio buffer
     */
    _generateSpeech(text, langCode) {
        return new Promise((resolve, reject) => {
            try {
                const tts = gtts(langCode);
                const chunks = [];
                
                const stream = tts.stream(text);
                
                stream.on('data', (chunk) => {
                    chunks.push(chunk);
                });
                
                stream.on('end', () => {
                    resolve(Buffer.concat(chunks));
                });
                
                stream.on('error', (error) => {
                    reject(error);
                });
            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Recognize speech from audio buffer
     * @private
     * @param {Buffer} audioBuffer - Audio buffer
     * @param {string} langCode - Language code
     * @returns {Promise<string>} Transcribed text
     */
    async _recognizeSpeech(audioBuffer, langCode) {
        // Note: Speech recognition requires external API or service
        // This is a placeholder implementation
        throw new Error(
            'Speech recognition requires external API integration. ' +
            'Please integrate with Google Speech-to-Text API or similar service.'
        );
    }

    /**
     * Convert text to speech and encode to Base44
     * @param {string} text - Text to convert to speech
     * @param {string} language - Target language (e.g., 'en', 'english', 'ku')
     * @returns {Promise<Object>} Object with audio data and metadata
     */
    async textToSpeechBase44(text, language = 'en') {
        try {
            // Get language code
            const langCode = this._getLanguageCode(language);
            
            console.log(`🎤 Generating speech: '${text.substring(0, 50)}...' in ${langCode}`);
            
            // Check if we need to use Coqui TTS via Python for Kurdish
            let audioBuffer;
            if (this._usesCoquiTTS(langCode)) {
                // Use Python service with Coqui TTS for Kurdish
                audioBuffer = await this._generateSpeechPython(text, langCode);
            } else {
                // Generate speech using node-gtts for other languages
                audioBuffer = await this._generateSpeech(text, langCode);
            }
            
            // Encode to Base44
            const audioBase44 = encode(audioBuffer);
            
            console.log(`✅ Success! Size: ${audioBuffer.length} bytes → ${audioBase44.length} chars`);
            
            const compressionRatio = audioBase44.length / audioBuffer.length;
            
            return {
                audio: audioBase44,
                language: langCode,
                format: 'mp3',
                text: text,
                size: audioBuffer.length,
                encodedSize: audioBase44.length,
                compressionRatio: compressionRatio
            };
            
        } catch (error) {
            console.error(`❌ Error in textToSpeechBase44: ${error.message}`);
            throw error;
        }
    }

    /**
     * Decode Base44 audio and convert to text
     * @param {string} audioBase44 - Base44 encoded audio
     * @param {string} language - Source language for recognition
     * @returns {Promise<Object>} Object with transcribed text and metadata
     */
    async speechToTextFromBase44(audioBase44, language = 'en') {
        try {
            // Get language code
            const langCode = this._getLanguageCode(language);
            
            console.log(`🎧 Decoding audio (${audioBase44.length} chars) for ${langCode}...`);
            
            // Decode Base44 to bytes
            const audioBuffer = decode(audioBase44);
            
            // Recognize speech
            const text = await this._recognizeSpeech(audioBuffer, langCode);
            
            console.log(`✅ Transcribed: '${text}'`);
            
            return {
                text: text,
                language: langCode,
                confidence: 1.0,
                audioSize: audioBuffer.length
            };
            
        } catch (error) {
            console.error(`❌ Error in speechToTextFromBase44: ${error.message}`);
            throw error;
        }
    }

    /**
     * Read audio file and convert to text
     * @param {string} filePath - Path to audio file
     * @param {string} language - Source language for recognition
     * @returns {Promise<Object>} Object with transcribed text and metadata
     */
    async speechToTextFromFile(filePath, language = 'en') {
        try {
            console.log(`📁 Reading audio file: ${filePath}`);
            
            // Read audio file
            const audioBuffer = await fs.readFile(filePath);
            
            // Encode to Base44
            const audioBase44 = encode(audioBuffer);
            
            // Call speechToTextFromBase44
            return await this.speechToTextFromBase44(audioBase44, language);
            
        } catch (error) {
            console.error(`❌ Error in speechToTextFromFile: ${error.message}`);
            throw error;
        }
    }

    /**
     * Decode Base44 audio and save to file
     * @param {string} audioBase44 - Base44 encoded audio
     * @param {string} outputPath - Path to save audio file
     * @returns {Promise<void>}
     */
    async saveAudioFromBase44(audioBase44, outputPath) {
        try {
            console.log(`💾 Saving audio to: ${outputPath}`);
            
            // Decode Base44 to bytes
            const audioBuffer = decode(audioBase44);
            
            // Write to file
            await fs.writeFile(outputPath, audioBuffer);
            
            console.log(`✅ Saved ${audioBuffer.length} bytes`);
            
        } catch (error) {
            console.error(`❌ Error in saveAudioFromBase44: ${error.message}`);
            throw error;
        }
    }
}

// Export
module.exports = { TTSSTTServiceBase44 };

// Demo function
async function main() {
    console.log("=".repeat(70));
    console.log("TTS/STT Service with Base44 Encoding - Demo");
    console.log("=".repeat(70));
    
    // Initialize service
    const service = new TTSSTTServiceBase44();
    
    // Test examples for all languages
    const testCases = [
        ["Hello, how are you today?", "english"],
        ["Silav, tu çawa yî?", "kurdish"],
        ["Guten Tag, wie geht es Ihnen?", "german"],
        ["Bonjour, comment allez-vous aujourd'hui?", "french"],
        ["Merhaba, bugün nasılsınız?", "turkish"],
    ];
    
    console.log("\n🎤 Testing Text-to-Speech with Base44 encoding...\n");
    
    const results = [];
    
    for (const [text, lang] of testCases) {
        try {
            console.log(`\n--- ${lang.toUpperCase()} ---`);
            const result = await service.textToSpeechBase44(text, lang);
            results.push([text, lang, result]);
            
            console.log(`Text: ${result.text}`);
            console.log(`Language: ${result.language}`);
            console.log(`Format: ${result.format}`);
            console.log(`Audio size: ${result.size} bytes`);
            console.log(`Encoded size: ${result.encodedSize} chars`);
            console.log(`Compression ratio: ${result.compressionRatio.toFixed(2)}x`);
            console.log(`Base44 preview: ${result.audio.substring(0, 60)}...`);
            
            // Save audio file
            const outputFile = `output_${result.language}.mp3`;
            await service.saveAudioFromBase44(result.audio, outputFile);
            
        } catch (error) {
            console.error(`Error with ${lang}: ${error.message}`);
        }
    }
    
    console.log("\n" + "=".repeat(70));
    console.log("✅ Demo completed!");
    console.log("=".repeat(70));
    
    // Summary
    console.log("\n📊 Summary:");
    console.log(`  Languages tested: ${testCases.length}`);
    console.log(`  Successful: ${results.length}`);
    if (results.length > 0) {
        const avgRatio = results.reduce((sum, r) => sum + r[2].compressionRatio, 0) / results.length;
        console.log(`  Average compression ratio: ${avgRatio.toFixed(2)}x`);
    }
    console.log(`\n  Audio files saved:`);
    for (const [, , result] of results) {
        console.log(`    - output_${result.language}.mp3`);
    }
}

// Run demo if executed directly
if (require.main === module) {
    main().catch(console.error);
}
