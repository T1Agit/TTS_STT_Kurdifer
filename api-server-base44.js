/**
 * REST API Server for TTS/STT with Base44 Encoding
 * 
 * Express.js server providing endpoints for Text-to-Speech and Speech-to-Text
 * with Base44 encoding support.
 */

const express = require('express');
const cors = require('cors');
const { TTSSTTServiceBase44 } = require('./tts-stt-service-base44');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Initialize service
const service = new TTSSTTServiceBase44();

// Request logging middleware
app.use((req, res, next) => {
    console.log(`📨 ${req.method} ${req.path} - ${new Date().toISOString()}`);
    next();
});

/**
 * GET /health
 * Health check endpoint
 */
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        encoding: 'Base44',
        supportedLanguages: Object.values(TTSSTTServiceBase44.SUPPORTED_LANGUAGES).filter((v, i, a) => a.indexOf(v) === i),
        endpoints: [
            'GET /health',
            'GET /api/languages',
            'POST /api/tts',
            'POST /api/stt',
            'POST /api/tts/batch'
        ],
        timestamp: new Date().toISOString()
    });
});

/**
 * GET /api/languages
 * Get all supported languages
 */
app.get('/api/languages', (req, res) => {
    const languages = Object.values(TTSSTTServiceBase44.SUPPORTED_LANGUAGES)
        .filter((v, i, a) => a.indexOf(v) === i);
    
    res.json({
        success: true,
        languages: languages,
        count: languages.length
    });
});

/**
 * POST /api/tts
 * Convert text to speech with Base44 encoding
 * Body: { text: string, language?: string }
 */
app.post('/api/tts', async (req, res) => {
    try {
        const { text, language = 'en' } = req.body;
        
        if (!text) {
            return res.status(400).json({
                success: false,
                error: 'Text is required'
            });
        }
        
        console.log(`🎤 TTS Request: "${text.substring(0, 50)}..." (${language})`);
        
        const result = await service.textToSpeechBase44(text, language);
        
        res.json({
            success: true,
            data: result
        });
        
    } catch (error) {
        console.error(`❌ TTS Error: ${error.message}`);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

/**
 * POST /api/stt
 * Convert speech to text from Base44 encoded audio
 * Body: { audio: string (base44), language?: string }
 */
app.post('/api/stt', async (req, res) => {
    try {
        const { audio, language = 'en' } = req.body;
        
        if (!audio) {
            return res.status(400).json({
                success: false,
                error: 'Audio data is required'
            });
        }
        
        console.log(`🎧 STT Request: ${audio.length} chars (${language})`);
        
        const result = await service.speechToTextFromBase44(audio, language);
        
        res.json({
            success: true,
            data: result
        });
        
    } catch (error) {
        console.error(`❌ STT Error: ${error.message}`);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

/**
 * POST /api/tts/batch
 * Convert multiple texts to speech with Base44 encoding
 * Body: { texts: string[], language?: string }
 */
app.post('/api/tts/batch', async (req, res) => {
    try {
        const { texts, language = 'en' } = req.body;
        
        if (!texts || !Array.isArray(texts)) {
            return res.status(400).json({
                success: false,
                error: 'Texts array is required'
            });
        }
        
        console.log(`🎤 Batch TTS Request: ${texts.length} texts (${language})`);
        
        // Process all texts concurrently
        const results = await Promise.all(
            texts.map(text => service.textToSpeechBase44(text, language))
        );
        
        res.json({
            success: true,
            count: results.length,
            data: results
        });
        
    } catch (error) {
        console.error(`❌ Batch TTS Error: ${error.message}`);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(`❌ Server Error: ${err.message}`);
    res.status(500).json({
        success: false,
        error: err.message || 'Internal server error'
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        success: false,
        error: 'Endpoint not found'
    });
});

// Start server
app.listen(PORT, () => {
    console.log("\n" + "=".repeat(70));
    console.log("🚀 TTS/STT API Server with Base44 Encoding");
    console.log("=".repeat(70));
    console.log(`\n📡 Server running on: http://localhost:${PORT}`);
    console.log(`\n📚 Available endpoints:`);
    console.log(`  • GET  /health              - Health check`);
    console.log(`  • GET  /api/languages       - Get supported languages`);
    console.log(`  • POST /api/tts             - Text to speech`);
    console.log(`  • POST /api/stt             - Speech to text`);
    console.log(`  • POST /api/tts/batch       - Batch text to speech`);
    console.log(`\n🌍 Supported languages:`);
    const languages = Object.values(TTSSTTServiceBase44.SUPPORTED_LANGUAGES)
        .filter((v, i, a) => a.indexOf(v) === i);
    console.log(`  ${languages.join(', ')}`);
    console.log("\n" + "=".repeat(70));
});

module.exports = app;
