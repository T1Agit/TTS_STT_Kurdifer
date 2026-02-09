#!/usr/bin/env node

/**
 * Integration Test for Base44 TTS/STT Service
 * Tests the complete workflow without external API dependencies
 */

const { encode, decode } = require('./base44');
const fs = require('fs');
const os = require('os');

console.log("=".repeat(70));
console.log("Base44 TTS/STT Integration Test");
console.log("=".repeat(70));

// Test 1: Base44 Encoding/Decoding
console.log("\n📝 Test 1: Base44 Encoding/Decoding");
console.log("-".repeat(70));

const testData = [
    "Hello, World!",
    "Silav, tu çawa yî?",
    "Guten Tag!",
    "Bonjour!",
    "Merhaba!"
];

let allPassed = true;

testData.forEach((text, i) => {
    const original = Buffer.from(text, 'utf-8');
    const encoded = encode(original);
    const decoded = decode(encoded);
    
    const passed = Buffer.compare(original, decoded) === 0;
    const status = passed ? "✅" : "❌";
    
    console.log(`${status} Test ${i + 1}: ${text}`);
    console.log(`   Original size: ${original.length} bytes`);
    console.log(`   Encoded size:  ${encoded.length} chars`);
    console.log(`   Ratio:         ${(encoded.length / original.length).toFixed(2)}x`);
    console.log(`   Encoded:       ${encoded.substring(0, 40)}...`);
    
    if (!passed) {
        allPassed = false;
        console.log(`   ❌ Mismatch detected!`);
    }
    console.log();
});

// Test 2: File Operations
console.log("\n📁 Test 2: File Operations (Save/Load)");
console.log("-".repeat(70));

const testContent = "This is a test file for Base44 encoding.";
const testBuffer = Buffer.from(testContent);
const testEncoded = encode(testBuffer);

// Simulate saving
const tempDir = os.tmpdir();
const testFilePath = `${tempDir}/test_base44.txt`;

try {
    fs.writeFileSync(testFilePath, testEncoded);
    console.log("✅ Saved encoded data to file");
    
    // Simulate loading
    const loadedEncoded = fs.readFileSync(testFilePath, 'utf-8');
    const loadedBuffer = decode(loadedEncoded);
    const loadedContent = loadedBuffer.toString('utf-8');
    
    if (loadedContent === testContent) {
        console.log("✅ Loaded and decoded data successfully");
        console.log(`   Content: "${loadedContent}"`);
    } else {
        console.log("❌ Content mismatch after load");
        allPassed = false;
    }
    
    // Cleanup
    fs.unlinkSync(testFilePath);
} catch (error) {
    console.log(`❌ File operation error: ${error.message}`);
    allPassed = false;
}

// Test 3: Large Data Handling
console.log("\n📦 Test 3: Large Data Handling");
console.log("-".repeat(70));

const sizes = [1024, 10240, 102400]; // 1KB, 10KB, 100KB

sizes.forEach(size => {
    const largeData = Buffer.alloc(size);
    for (let i = 0; i < size; i++) {
        largeData[i] = i % 256;
    }
    
    const startEncode = Date.now();
    const encoded = encode(largeData);
    const encodeTime = Date.now() - startEncode;
    
    const startDecode = Date.now();
    const decoded = decode(encoded);
    const decodeTime = Date.now() - startDecode;
    
    const passed = Buffer.compare(largeData, decoded) === 0;
    const status = passed ? "✅" : "❌";
    
    console.log(`${status} ${(size / 1024).toFixed(0)}KB data:`);
    console.log(`   Encoded size:   ${encoded.length} chars`);
    console.log(`   Compression:    ${(encoded.length / size).toFixed(2)}x`);
    console.log(`   Encode time:    ${encodeTime}ms`);
    console.log(`   Decode time:    ${decodeTime}ms`);
    
    if (!passed) allPassed = false;
});

// Test 4: Language Support Validation
console.log("\n🌍 Test 4: Language Code Validation");
console.log("-".repeat(70));

const { TTSSTTServiceBase44 } = require('./tts-stt-service-base44');
const service = new TTSSTTServiceBase44();

const testLanguages = [
    ['english', 'en'],
    ['kurdish', 'ku'],
    ['german', 'de'],
    ['french', 'fr'],
    ['turkish', 'tr'],
    ['en', 'en'],
    ['de', 'de']
];

testLanguages.forEach(([input, expected]) => {
    try {
        const result = service._getLanguageCode(input);
        if (result === expected) {
            console.log(`✅ ${input.padEnd(10)} → ${result}`);
        } else {
            console.log(`❌ ${input.padEnd(10)} → ${result} (expected: ${expected})`);
            allPassed = false;
        }
    } catch (error) {
        console.log(`❌ ${input.padEnd(10)} → Error: ${error.message}`);
        allPassed = false;
    }
});

// Test 5: Kurdish TTS Engine Selection
console.log("\n🇹🇯 Test 5: Kurdish TTS Engine Selection");
console.log("-".repeat(70));

try {
    const usesGTTS_en = !service._usesCoquiTTS('en');
    const usesCoqui_ku = service._usesCoquiTTS('ku');
    
    if (usesGTTS_en) {
        console.log("✅ English (en) uses gTTS");
    } else {
        console.log("❌ English (en) should use gTTS");
        allPassed = false;
    }
    
    if (usesCoqui_ku) {
        console.log("✅ Kurdish (ku) uses Coqui TTS");
    } else {
        console.log("❌ Kurdish (ku) should use Coqui TTS");
        allPassed = false;
    }
} catch (error) {
    console.log(`❌ Error testing TTS engine selection: ${error.message}`);
    allPassed = false;
}

// Test 6: Error Handling
console.log("\n⚠️  Test 6: Error Handling");
console.log("-".repeat(70));

// Test invalid language
try {
    service._getLanguageCode('invalid_language');
    console.log("❌ Should have thrown error for invalid language");
    allPassed = false;
} catch (error) {
    console.log("✅ Correctly rejected invalid language");
}

// Test invalid Base44 string
try {
    decode("INVALID@CHARS!");
    console.log("❌ Should have thrown error for invalid characters");
    allPassed = false;
} catch (error) {
    console.log("✅ Correctly rejected invalid Base44 string");
}

// Summary
console.log("\n" + "=".repeat(70));
if (allPassed) {
    console.log("✅ All integration tests passed!");
    console.log("=".repeat(70));
    console.log("\n✨ The Base44 encoding and service structure is working correctly!");
    console.log("\n📝 Note: TTS/STT functionality requires external API access:");
    console.log("   - Google Text-to-Speech API for audio generation");
    console.log("   - Google Speech-to-Text API for transcription");
    console.log("\n🚀 To test the full API server:");
    console.log("   1. npm start              (start the server)");
    console.log("   2. npm run client         (run client examples)");
    console.log("   3. curl http://localhost:3000/health");
    process.exit(0);
} else {
    console.log("❌ Some tests failed!");
    console.log("=".repeat(70));
    process.exit(1);
}
