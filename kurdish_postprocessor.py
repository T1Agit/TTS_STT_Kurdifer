#!/usr/bin/env python3
"""
Kurdish Post-Processor for STT

This module provides post-processing capabilities for Kurdish STT output,
correcting common mistakes made by the facebook/mms-1b-all model.
"""

import re
from typing import Dict, Optional
from kurdish_dictionary import KURDISH_CORRECTIONS


class KurdishPostProcessor:
    """
    Post-processor for Kurdish Speech-to-Text output
    
    Applies word-level corrections from the Kurdish dictionary to fix
    common STT mistakes including:
    - Missing special characters (ê, î, û, ç, ş)
    - Vowel confusion (e/ê, i/î, u/û)
    - Consonant confusion (c/ç, s/ş)
    """
    
    def __init__(self):
        """Initialize the Kurdish post-processor"""
        self.corrections = KURDISH_CORRECTIONS
        print(f"📖 Kurdish Post-Processor initialized with {len(self.corrections)} corrections")
    
    def correct_transcription(self, text: str) -> str:
        """
        Apply corrections to transcribed text
        
        Args:
            text: The raw transcription from STT
            
        Returns:
            Corrected text with Kurdish words properly formatted
        """
        if not text or not isinstance(text, str):
            return text
        
        # Split text into words while preserving punctuation and spacing
        # This regex splits on word boundaries but keeps punctuation separate
        words = re.findall(r'\w+|[^\w\s]', text)
        
        corrected_words = []
        for word in words:
            # Skip punctuation
            if not re.match(r'\w', word):
                corrected_words.append(word)
                continue
            
            # Try to find correction in dictionary
            corrected = self._correct_word(word)
            corrected_words.append(corrected)
        
        # Reconstruct the text
        result = self._reconstruct_text(words, corrected_words)
        return result
    
    def _correct_word(self, word: str) -> str:
        """
        Correct a single word using the dictionary
        
        Handles case sensitivity by:
        1. Checking exact match first
        2. Checking lowercase match and preserving case pattern
        
        Args:
            word: The word to correct
            
        Returns:
            Corrected word with case preserved
        """
        # Try exact match first
        if word in self.corrections:
            return self.corrections[word]
        
        # Try lowercase match
        lower_word = word.lower()
        if lower_word in self.corrections:
            corrected = self.corrections[lower_word]
            return self._preserve_case(word, corrected)
        
        # No correction found, return original
        return word
    
    def _preserve_case(self, original: str, corrected: str) -> str:
        """
        Preserve the case pattern of the original word in the corrected word
        
        Args:
            original: The original word with its case pattern
            corrected: The corrected word (usually lowercase from dictionary)
            
        Returns:
            Corrected word with case pattern matching original
        """
        if not original or not corrected:
            return corrected
        
        # If original is all uppercase, return corrected in uppercase
        if original.isupper():
            return corrected.upper()
        
        # If original has first letter capitalized, capitalize corrected
        if original[0].isupper() and (len(original) == 1 or original[1:].islower()):
            return corrected[0].upper() + corrected[1:] if len(corrected) > 1 else corrected.upper()
        
        # Otherwise, return corrected as-is
        return corrected
    
    def _reconstruct_text(self, original_tokens: list, corrected_tokens: list) -> str:
        """
        Reconstruct text from tokens, preserving spacing
        
        Args:
            original_tokens: Original tokens from text
            corrected_tokens: Corrected tokens
            
        Returns:
            Reconstructed text string
        """
        if not corrected_tokens:
            return ""
        
        result = []
        for i, token in enumerate(corrected_tokens):
            # Add the token
            result.append(token)
            
            # Add space after word tokens (not after punctuation at end)
            if i < len(corrected_tokens) - 1:
                # Check if current token is a word and next is also a word
                is_word = re.match(r'\w', token)
                next_is_word = re.match(r'\w', corrected_tokens[i + 1])
                
                if is_word and next_is_word:
                    result.append(' ')
        
        return ''.join(result)
    
    def get_correction_stats(self, original: str, corrected: str) -> Dict[str, int]:
        """
        Get statistics about corrections applied
        
        Args:
            original: Original text
            corrected: Corrected text
            
        Returns:
            Dictionary with statistics (words_corrected, total_words)
        """
        original_words = re.findall(r'\w+', original)
        corrected_words = re.findall(r'\w+', corrected)
        
        corrections_count = sum(
            1 for orig, corr in zip(original_words, corrected_words)
            if orig != corr
        )
        
        return {
            'words_corrected': corrections_count,
            'total_words': len(original_words),
            'correction_rate': corrections_count / len(original_words) if original_words else 0
        }


def test_postprocessor():
    """Test function for Kurdish post-processor"""
    print("=" * 70)
    print("Kurdish Post-Processor Test")
    print("=" * 70)
    
    processor = KurdishPostProcessor()
    
    # Test cases with common mistakes
    test_cases = [
        "silav cawa yi",
        "ez bas im",
        "nave te ci ye",
        "tu cend sali yi",
        "ez ji te hez dikim",
        "rojbas de bav",
        "pirtuk li ser mase ye",
        "car cin penc ses",
    ]
    
    print("\n🔧 Testing corrections:")
    print("-" * 70)
    
    for original in test_cases:
        corrected = processor.correct_transcription(original)
        stats = processor.get_correction_stats(original, corrected)
        
        if original != corrected:
            print(f"✅ '{original}'")
            print(f"   → '{corrected}'")
            print(f"   ({stats['words_corrected']}/{stats['total_words']} words corrected)")
        else:
            print(f"⚪ '{original}' (no changes)")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_postprocessor()
