/**
 * Base44 Encoding/Decoding for Browser
 * Browser-compatible version using Uint8Array instead of Node.js Buffer
 */

const Base44 = {
    ALPHABET: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh",
    BASE: 44n,

    /**
     * Encode Uint8Array to Base44 string
     */
    encode(data) {
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
    },

    /**
     * Decode Base44 string to Uint8Array
     */
    decode(encoded) {
        if (!encoded || encoded.length === 0) {
            return new Uint8Array(0);
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
            return new Uint8Array(Math.max(leadingZeros, 1));
        }

        // Convert to byte array
        const bytes = [];
        let tempNum = num;
        while (tempNum > 0n) {
            bytes.unshift(Number(tempNum & 0xFFn));
            tempNum = tempNum >> 8n;
        }

        // Create Uint8Array with leading zeros
        const result = new Uint8Array(leadingZeros + bytes.length);
        for (let i = 0; i < bytes.length; i++) {
            result[leadingZeros + i] = bytes[i];
        }

        return result;
    }
};