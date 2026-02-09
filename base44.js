/**
 * Base44 Encoding/Decoding Module
 * 
 * Implements Base44 encoding using a 44-character alphabet:
 * ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefgh-_
 */

class Base44 {
    // First 44 characters
    static ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefgh-_".substring(0, 44);
    static BASE = 44n;

    /**
     * Encode bytes to Base44 string
     * @param {Buffer|Uint8Array} data - Bytes to encode
     * @returns {string} Base44 encoded string
     */
    static encode(data) {
        if (!data || data.length === 0) {
            return "";
        }

        // Convert bytes to BigInt
        let num = 0n;
        for (let i = 0; i < data.length; i++) {
            num = (num << 8n) | BigInt(data[i]);
        }

        // Convert to base44
        if (num === 0n) {
            return this.ALPHABET[0];
        }

        const result = [];
        while (num > 0n) {
            const remainder = Number(num % this.BASE);
            result.push(this.ALPHABET[remainder]);
            num = num / this.BASE;
        }

        // Reverse to get correct order
        result.reverse();

        // Handle leading zeros
        let leadingZeros = 0;
        for (let i = 0; i < data.length; i++) {
            if (data[i] === 0) {
                leadingZeros++;
            } else {
                break;
            }
        }

        return this.ALPHABET[0].repeat(leadingZeros) + result.join('');
    }

    /**
     * Decode Base44 string to bytes
     * @param {string} encoded - Base44 encoded string
     * @returns {Buffer} Decoded bytes
     */
    static decode(encoded) {
        if (!encoded || encoded.length === 0) {
            return Buffer.from([]);
        }

        // Count leading zeros
        let leadingZeros = 0;
        for (let i = 0; i < encoded.length; i++) {
            if (encoded[i] === this.ALPHABET[0]) {
                leadingZeros++;
            } else {
                break;
            }
        }

        // Convert from base44 to BigInt
        let num = 0n;
        for (let i = 0; i < encoded.length; i++) {
            const char = encoded[i];
            const index = this.ALPHABET.indexOf(char);
            if (index === -1) {
                throw new Error(`Invalid character in Base44 string: ${char}`);
            }
            num = num * this.BASE + BigInt(index);
        }

        // Convert BigInt to bytes
        if (num === 0n) {
            // If the entire string is just zeros, return the correct number of zero bytes
            return Buffer.alloc(Math.max(leadingZeros, 1));
        }

        // Convert to byte array
        const bytes = [];
        let tempNum = num;
        while (tempNum > 0n) {
            bytes.unshift(Number(tempNum & 0xFFn));
            tempNum = tempNum >> 8n;
        }

        // Create buffer with leading zeros
        const result = Buffer.alloc(leadingZeros + bytes.length);
        result.fill(0, 0, leadingZeros);
        
        for (let i = 0; i < bytes.length; i++) {
            result[leadingZeros + i] = bytes[i];
        }

        return result;
    }
}

/**
 * Encode bytes to Base44 string
 * @param {Buffer|Uint8Array} data - Bytes to encode
 * @returns {string} Base44 encoded string
 */
function encode(data) {
    return Base44.encode(data);
}

/**
 * Decode Base44 string to bytes
 * @param {string} encoded - Base44 encoded string
 * @returns {Buffer} Decoded bytes
 */
function decode(encoded) {
    return Base44.decode(encoded);
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Base44, encode, decode };
}

// Test if running directly
if (require.main === module) {
    console.log("=".repeat(60));
    console.log("Base44 Encoding/Decoding Test");
    console.log("=".repeat(60));

    // Test cases
    const testCases = [
        Buffer.from("Hello, World!"),
        Buffer.from("Kurdish: Silav, tu çawa yî?", 'utf-8'),
        Buffer.from("German: Guten Tag!"),
        Buffer.from("French: Bonjour!"),
        Buffer.from("Turkish: Merhaba!"),
        Buffer.from("English: Hello!"),
        Buffer.from([0x00, 0x00, 0x01, 0x02]), // Test leading zeros
        Buffer.from([]), // Empty
        Buffer.from([0x00]), // Single zero
        Buffer.from(Array.from({ length: 256 }, (_, i) => i)), // All possible bytes
    ];

    console.log("\n✅ Running test cases...\n");

    let allPassed = true;
    testCases.forEach((testData, i) => {
        try {
            // Encode
            const encoded = encode(testData);

            // Decode
            const decoded = decode(encoded);

            // Verify
            const passed = Buffer.compare(decoded, testData) === 0;
            
            if (passed) {
                const status = "✅ PASS";
                if (testData.length <= 50) {
                    console.log(`Test ${i + 1}: ${status}`);
                    console.log(`  Original: ${testData.toString('hex').substring(0, 100)}`);
                    console.log(`  Encoded:  ${encoded.substring(0, 80)}${encoded.length > 80 ? '...' : ''}`);
                    console.log(`  Size:     ${testData.length} bytes → ${encoded.length} chars`);
                    console.log(`  Ratio:    ${(encoded.length / Math.max(testData.length, 1)).toFixed(2)}x`);
                } else {
                    console.log(`Test ${i + 1}: ${status} (large data: ${testData.length} bytes)`);
                    console.log(`  Encoded size: ${encoded.length} chars`);
                    console.log(`  Ratio: ${(encoded.length / testData.length).toFixed(2)}x`);
                }
            } else {
                const status = "❌ FAIL";
                allPassed = false;
                console.log(`Test ${i + 1}: ${status}`);
                console.log(`  Original: ${testData.toString('hex')}`);
                console.log(`  Decoded:  ${decoded.toString('hex')}`);
            }
        } catch (error) {
            allPassed = false;
            console.log(`Test ${i + 1}: ❌ ERROR - ${error.message}`);
        }
        console.log();
    });

    console.log("=".repeat(60));
    if (allPassed) {
        console.log("✅ All tests passed!");
    } else {
        console.log("❌ Some tests failed!");
    }
    console.log("=".repeat(60));
}
