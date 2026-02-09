/**
 * Client Example for TTS/STT API with Base44 Encoding
 * 
 * Demonstrates how to interact with the TTS/STT API server
 */

const axios = require('axios');

class TTSSTTClient {
    /**
     * Create a new TTS/STT API client
     * @param {string} baseUrl - Base URL of the API server
     */
    constructor(baseUrl = 'http://localhost:3000') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }

    /**
     * Convert text to speech
     * @param {string} text - Text to convert
     * @param {string} language - Target language
     * @returns {Promise<Object>} TTS result
     */
    async textToSpeech(text, language = 'en') {
        try {
            const response = await this.client.post('/api/tts', {
                text,
                language
            });
            return response.data;
        } catch (error) {
            throw new Error(`TTS request failed: ${error.message}`);
        }
    }

    /**
     * Convert speech to text
     * @param {string} audioBase44 - Base44 encoded audio
     * @param {string} language - Source language
     * @returns {Promise<Object>} STT result
     */
    async speechToText(audioBase44, language = 'en') {
        try {
            const response = await this.client.post('/api/stt', {
                audio: audioBase44,
                language
            });
            return response.data;
        } catch (error) {
            throw new Error(`STT request failed: ${error.message}`);
        }
    }

    /**
     * Convert multiple texts to speech in batch
     * @param {string[]} texts - Texts to convert
     * @param {string} language - Target language
     * @returns {Promise<Object>} Batch TTS results
     */
    async batchTextToSpeech(texts, language = 'en') {
        try {
            const response = await this.client.post('/api/tts/batch', {
                texts,
                language
            });
            return response.data;
        } catch (error) {
            throw new Error(`Batch TTS request failed: ${error.message}`);
        }
    }

    /**
     * Get supported languages
     * @returns {Promise<Object>} Supported languages
     */
    async getSupportedLanguages() {
        try {
            const response = await this.client.get('/api/languages');
            return response.data;
        } catch (error) {
            throw new Error(`Get languages request failed: ${error.message}`);
        }
    }

    /**
     * Check server health
     * @returns {Promise<Object>} Health status
     */
    async healthCheck() {
        try {
            const response = await this.client.get('/health');
            return response.data;
        } catch (error) {
            throw new Error(`Health check failed: ${error.message}`);
        }
    }
}

// Demo function
async function demo() {
    console.log("=".repeat(70));
    console.log("TTS/STT API Client - Demo");
    console.log("=".repeat(70));
    
    const client = new TTSSTTClient('http://localhost:3000');
    
    try {
        // Check server health
        console.log("\n🏥 Checking server health...");
        const health = await client.healthCheck();
        console.log(`✅ Server status: ${health.status}`);
        console.log(`📦 Encoding: ${health.encoding}`);
        console.log(`🌍 Supported languages: ${health.supportedLanguages.join(', ')}`);
        
        // Get supported languages
        console.log("\n🌍 Getting supported languages...");
        const languagesResult = await client.getSupportedLanguages();
        console.log(`✅ Found ${languagesResult.count} languages: ${languagesResult.languages.join(', ')}`);
        
        // Example 1: Kurdish TTS
        console.log("\n" + "-".repeat(70));
        console.log("📝 Example 1: Kurdish Text-to-Speech");
        console.log("-".repeat(70));
        
        const kurdishText = "Silav, tu îro çawa yî?";
        console.log(`Text: ${kurdishText}`);
        console.log("Language: Kurdish (ku)");
        
        const ttsResult = await client.textToSpeech(kurdishText, 'kurdish');
        console.log(`\n✅ TTS Success!`);
        console.log(`  Audio size: ${ttsResult.data.size} bytes`);
        console.log(`  Encoded size: ${ttsResult.data.encodedSize} chars`);
        console.log(`  Compression ratio: ${ttsResult.data.compressionRatio.toFixed(2)}x`);
        console.log(`  Base44 preview: ${ttsResult.data.audio.substring(0, 60)}...`);
        
        // Example 2: Batch TTS for all languages
        console.log("\n" + "-".repeat(70));
        console.log("📝 Example 2: Batch Text-to-Speech (All Languages)");
        console.log("-".repeat(70));
        
        const batchTexts = [
            "Hello, how are you today?",
            "Guten Tag, wie geht es Ihnen?",
            "Bonjour, comment allez-vous?",
            "Merhaba, nasılsınız?",
            "Silav, tu çawa yî?"
        ];
        
        const languages = ['english', 'german', 'french', 'turkish', 'kurdish'];
        
        console.log(`\nProcessing ${batchTexts.length} texts in different languages...\n`);
        
        for (let i = 0; i < batchTexts.length; i++) {
            const text = batchTexts[i];
            const lang = languages[i];
            
            console.log(`${i + 1}. ${lang.toUpperCase()}: "${text}"`);
            
            const result = await client.textToSpeech(text, lang);
            console.log(`   ✅ Size: ${result.data.size} bytes → ${result.data.encodedSize} chars (${result.data.compressionRatio.toFixed(2)}x)`);
        }
        
        // Example 3: Batch processing
        console.log("\n" + "-".repeat(70));
        console.log("📝 Example 3: Batch API (Multiple English Texts)");
        console.log("-".repeat(70));
        
        const englishTexts = [
            "Good morning!",
            "How are you?",
            "Thank you very much!"
        ];
        
        console.log(`\nProcessing batch of ${englishTexts.length} texts...`);
        englishTexts.forEach((text, i) => console.log(`  ${i + 1}. "${text}"`));
        
        const batchResult = await client.batchTextToSpeech(englishTexts, 'english');
        console.log(`\n✅ Batch processing complete!`);
        console.log(`  Total processed: ${batchResult.count}`);
        console.log(`  Results:`);
        batchResult.data.forEach((result, i) => {
            console.log(`    ${i + 1}. ${result.size} bytes → ${result.encodedSize} chars`);
        });
        
        // Summary
        console.log("\n" + "=".repeat(70));
        console.log("✅ Demo completed successfully!");
        console.log("=".repeat(70));
        
        console.log("\n💡 Usage Examples:");
        console.log("\nNode.js:");
        console.log(`
const { TTSSTTClient } = require('./client-example');
const client = new TTSSTTClient('http://localhost:3000');

// Convert text to speech
const result = await client.textToSpeech('Hello World', 'english');
console.log(result.data.audio); // Base44 encoded audio

// Get supported languages
const languages = await client.getSupportedLanguages();
console.log(languages.languages);
        `);
        
        console.log("\ncURL:");
        console.log(`
# Text to Speech
curl -X POST http://localhost:3000/api/tts \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Hello World", "language": "en"}'

# Get Languages
curl http://localhost:3000/api/languages

# Batch TTS
curl -X POST http://localhost:3000/api/tts/batch \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["Hello", "World"], "language": "en"}'
        `);
        
    } catch (error) {
        console.error(`\n❌ Error: ${error.message}`);
        console.error("\n💡 Make sure the API server is running:");
        console.error("   npm start");
        process.exit(1);
    }
}

// Export client class
module.exports = { TTSSTTClient };

// Run demo if executed directly
if (require.main === module) {
    demo().catch(console.error);
}
