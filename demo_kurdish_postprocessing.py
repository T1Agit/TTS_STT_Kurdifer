#!/usr/bin/env python3
"""
Demo: Kurdish STT Post-Processing

This script demonstrates the post-processing capabilities without requiring
the full STT model to be loaded.
"""

from kurdish_postprocessor import KurdishPostProcessor
from kurdish_dictionary import get_corrections_count


def main():
    print("=" * 80)
    print(" " * 20 + "Kurdish STT Post-Processing Demo")
    print("=" * 80)
    
    # Initialize post-processor
    processor = KurdishPostProcessor()
    
    print(f"\n📊 Dictionary Statistics:")
    print(f"   Total corrections: {get_corrections_count()}")
    print(f"   Covers: greetings, pronouns, verbs, family, numbers, time, nature,")
    print(f"           body parts, food, adjectives, prepositions, education, and more")
    
    # Demonstrate corrections
    print("\n" + "=" * 80)
    print("Sample Corrections (simulating raw STT output)")
    print("=" * 80)
    
    examples = [
        # Common greetings
        ("silav", "Hello/Greetings"),
        ("rojbas", "Good day"),
        ("sevbas", "Good evening"),
        
        # Common phrases
        ("tu cawa yi", "How are you?"),
        ("ez bas im", "I am fine"),
        ("nave te ci ye", "What is your name?"),
        ("cend sali yi", "How old are you?"),
        
        # Numbers
        ("car cin penc ses", "4 5 6"),
        ("yek du se", "1 2 3"),
        
        # Common words
        ("de bav bira xwisk", "mother father brother sister"),
        ("nan gost sir cay", "bread meat milk tea"),
        ("ser cav dest pe", "head eye hand foot"),
        
        # Longer phrases
        ("ez dixwazim ku te bibînim", "I want to see you"),
        ("nav min ahmed e", "My name is Ahmed"),
    ]
    
    print("\n📝 Single Words:")
    print("-" * 80)
    for wrong, meaning in examples[:3]:
        corrected = processor.correct_transcription(wrong)
        if wrong != corrected:
            print(f"  '{wrong:15s}' → '{corrected:15s}'  ({meaning})")
        else:
            print(f"  '{wrong:15s}' → (no change)      ({meaning})")
    
    print("\n💬 Common Phrases:")
    print("-" * 80)
    for wrong, meaning in examples[3:7]:
        corrected = processor.correct_transcription(wrong)
        stats = processor.get_correction_stats(wrong, corrected)
        
        print(f"\n  Input:     '{wrong}'")
        print(f"  Output:    '{corrected}'")
        print(f"  Meaning:   {meaning}")
        print(f"  Corrected: {stats['words_corrected']}/{stats['total_words']} words")
    
    print("\n🔢 Numbers:")
    print("-" * 80)
    for wrong, meaning in examples[7:9]:
        corrected = processor.correct_transcription(wrong)
        print(f"  '{wrong:20s}' → '{corrected:20s}'  ({meaning})")
    
    print("\n📚 Word Lists:")
    print("-" * 80)
    for wrong, meaning in examples[9:12]:
        corrected = processor.correct_transcription(wrong)
        print(f"  '{wrong:30s}' → '{corrected:30s}'")
        print(f"  ({meaning})")
    
    print("\n📖 Longer Sentences:")
    print("-" * 80)
    for wrong, meaning in examples[12:]:
        corrected = processor.correct_transcription(wrong)
        stats = processor.get_correction_stats(wrong, corrected)
        
        print(f"\n  Input:     '{wrong}'")
        print(f"  Output:    '{corrected}'")
        print(f"  Meaning:   {meaning}")
        if stats['words_corrected'] > 0:
            print(f"  Corrected: {stats['words_corrected']}/{stats['total_words']} words ({stats['correction_rate']*100:.0f}%)")
    
    # Special characters explanation
    print("\n" + "=" * 80)
    print("Kurdish Special Characters Handled:")
    print("=" * 80)
    print("  • ç (c-cedilla) - replaces simple 'c' in many words")
    print("  • ş (s-cedilla) - replaces simple 's' in many words")
    print("  • ê (e-circumflex) - used instead of 'e' in many words")
    print("  • î (i-circumflex) - used instead of 'i' in many words")
    print("  • û (u-circumflex) - used instead of 'u' in many words")
    
    print("\n" + "=" * 80)
    print("✨ Post-processing helps correct common STT mistakes automatically!")
    print("=" * 80)
    
    # Integration note
    print("\n📝 Integration with kurdish_stt_service.py:")
    print("   • Post-processor is initialized when STT service starts")
    print("   • Corrections are applied automatically after transcription")
    print("   • Returns both 'raw_text' and 'text' (corrected) in response")
    print("   • Preserves original case (lowercase, UPPERCASE, Capitalized)")
    print()


if __name__ == "__main__":
    main()
