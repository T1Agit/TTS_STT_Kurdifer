"""
Base44 Encoding/Decoding Module

Implements Base44 encoding using a 44-character alphabet:
A-Z (26 chars), a-z (26 chars, but we only use first 8), 0-9 (10 chars) = 44 chars total
Alphabet: ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefgh-_
"""


class Base44:
    """Base44 encoder/decoder using 44-character alphabet"""
    
    # First 44 characters: A-Z (26), 0-9 (10), a-h (8) = 44 chars
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefgh-_"[:44]
    BASE = 44
    
    @classmethod
    def encode(cls, data: bytes) -> str:
        """
        Encode bytes to Base44 string
        
        Args:
            data: Bytes to encode
            
        Returns:
            Base44 encoded string
        """
        if not data:
            return ""
        
        # Convert bytes to integer
        num = int.from_bytes(data, byteorder='big')
        
        # Convert to base44
        if num == 0:
            return cls.ALPHABET[0]
        
        result = []
        while num > 0:
            num, remainder = divmod(num, cls.BASE)
            result.append(cls.ALPHABET[remainder])
        
        # Reverse to get correct order
        result.reverse()
        
        # Handle leading zeros
        leading_zeros = len(data) - len(data.lstrip(b'\x00'))
        return cls.ALPHABET[0] * leading_zeros + ''.join(result)
    
    @classmethod
    def decode(cls, encoded: str) -> bytes:
        """
        Decode Base44 string to bytes
        
        Args:
            encoded: Base44 encoded string
            
        Returns:
            Decoded bytes
        """
        if not encoded:
            return b""
        
        # Count leading zeros
        leading_zeros = len(encoded) - len(encoded.lstrip(cls.ALPHABET[0]))
        
        # Convert from base44 to integer
        num = 0
        for char in encoded:
            if char not in cls.ALPHABET:
                raise ValueError(f"Invalid character in Base44 string: {char}")
            num = num * cls.BASE + cls.ALPHABET.index(char)
        
        # Convert integer to bytes
        if num == 0:
            # If the entire string is just zeros, return the correct number of zero bytes
            return b'\x00' * max(leading_zeros, 1)
        
        byte_length = (num.bit_length() + 7) // 8
        result = num.to_bytes(byte_length, byteorder='big')
        
        # Add leading zero bytes
        return b'\x00' * leading_zeros + result


def encode(data: bytes) -> str:
    """Encode bytes to Base44 string"""
    return Base44.encode(data)


def decode(encoded: str) -> bytes:
    """Decode Base44 string to bytes"""
    return Base44.decode(encoded)


if __name__ == "__main__":
    print("=" * 60)
    print("Base44 Encoding/Decoding Test")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        b"Hello, World!",
        "Kurdish: Silav, tu çawa yî?".encode('utf-8'),
        b"German: Guten Tag!",
        b"French: Bonjour!",
        b"Turkish: Merhaba!",
        b"English: Hello!",
        b"\x00\x00\x01\x02",  # Test leading zeros
        b"",  # Empty
        b"\x00",  # Single zero
        bytes(range(256)),  # All possible bytes
    ]
    
    print("\n✅ Running test cases...\n")
    
    all_passed = True
    for i, test_data in enumerate(test_cases, 1):
        try:
            # Encode
            encoded = encode(test_data)
            
            # Decode
            decoded = decode(encoded)
            
            # Verify
            if decoded == test_data:
                status = "✅ PASS"
                if len(test_data) <= 50:
                    print(f"Test {i}: {status}")
                    print(f"  Original: {test_data[:50]}")
                    print(f"  Encoded:  {encoded[:80]}{'...' if len(encoded) > 80 else ''}")
                    print(f"  Size:     {len(test_data)} bytes → {len(encoded)} chars")
                    print(f"  Ratio:    {len(encoded) / max(len(test_data), 1):.2f}x")
                else:
                    print(f"Test {i}: {status} (large data: {len(test_data)} bytes)")
                    print(f"  Encoded size: {len(encoded)} chars")
                    print(f"  Ratio: {len(encoded) / len(test_data):.2f}x")
            else:
                status = "❌ FAIL"
                all_passed = False
                print(f"Test {i}: {status}")
                print(f"  Original: {test_data}")
                print(f"  Decoded:  {decoded}")
        except Exception as e:
            all_passed = False
            print(f"Test {i}: ❌ ERROR - {e}")
        
        print()
    
    print("=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)
