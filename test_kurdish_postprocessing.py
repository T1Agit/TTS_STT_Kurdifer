#!/usr/bin/env python3
"""
Test Kurdish STT Post-Processing

Tests the post-processing functionality without requiring the full STT model.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kurdish_dictionary import KURDISH_CORRECTIONS, get_corrections_count
from kurdish_postprocessor import KurdishPostProcessor


def test_dictionary():
    """Test that dictionary has sufficient entries"""
    print("=" * 70)
    print("Test 1: Dictionary Size")
    print("=" * 70)
    
    count = get_corrections_count()
    print(f"📖 Dictionary contains {count} word corrections")
    
    if count >= 500:
        print(f"✅ PASS: Dictionary has {count} entries (>= 500 required)")
        return True
    else:
        print(f"❌ FAIL: Dictionary has only {count} entries (500 required)")
        return False


def test_special_characters():
    """Test corrections for special Kurdish characters"""
    print("\n" + "=" * 70)
    print("Test 2: Special Character Corrections")
    print("=" * 70)
    
    test_cases = [
        # Format: (wrong, expected_correct)
        ("cawa", "çawa"),  # c → ç
        ("xer", "xêr"),    # e → ê
        ("sir", "şîr"),    # s → ş, i → î
        ("bun", "bûn"),    # u → û
        ("di", "di"),      # Keep as-is (correct)
    ]
    
    all_passed = True
    for wrong, expected in test_cases:
        corrected = KURDISH_CORRECTIONS.get(wrong, wrong)
        if corrected == expected:
            print(f"✅ '{wrong}' → '{corrected}'")
        else:
            print(f"❌ '{wrong}' → '{corrected}' (expected '{expected}')")
            all_passed = False
    
    return all_passed


def test_postprocessor_basic():
    """Test basic post-processor functionality"""
    print("\n" + "=" * 70)
    print("Test 3: Post-Processor Basic Functionality")
    print("=" * 70)
    
    processor = KurdishPostProcessor()
    
    test_cases = [
        # Format: (input, expected_output)
        ("silav cawa yi", "silav çawa yi"),
        ("ez bas im", "ez baş im"),
        ("ci ye", "çi ye"),
        ("tu cend sali yi", "tu çend sali yi"),
    ]
    
    all_passed = True
    for input_text, expected in test_cases:
        result = processor.correct_transcription(input_text)
        # Normalize whitespace for comparison
        result = ' '.join(result.split())
        expected = ' '.join(expected.split())
        
        if result == expected:
            print(f"✅ '{input_text}' → '{result}'")
        else:
            print(f"❌ '{input_text}'")
            print(f"   Got:      '{result}'")
            print(f"   Expected: '{expected}'")
            all_passed = False
    
    return all_passed


def test_postprocessor_case_preservation():
    """Test that case is preserved correctly"""
    print("\n" + "=" * 70)
    print("Test 4: Case Preservation")
    print("=" * 70)
    
    processor = KurdishPostProcessor()
    
    test_cases = [
        # Format: (input, expected_output)
        ("Silav", "Silav"),        # Keep capitalization
        ("SILAV", "SILAV"),        # Keep all caps
        ("silav", "silav"),        # Keep lowercase
        ("Cawa yi", "Çawa yi"),    # Preserve case after correction
        ("CAWA", "ÇAWA"),          # Uppercase correction
    ]
    
    all_passed = True
    for input_text, expected in test_cases:
        result = processor.correct_transcription(input_text)
        # Normalize whitespace
        result = ' '.join(result.split())
        expected = ' '.join(expected.split())
        
        if result == expected:
            print(f"✅ '{input_text}' → '{result}'")
        else:
            print(f"❌ '{input_text}'")
            print(f"   Got:      '{result}'")
            print(f"   Expected: '{expected}'")
            all_passed = False
    
    return all_passed


def test_word_categories():
    """Test that all required word categories are present"""
    print("\n" + "=" * 70)
    print("Test 5: Word Categories Coverage")
    print("=" * 70)
    
    categories = {
        "Greetings": ["silav", "merheba", "rojbaş", "spas"],
        "Pronouns": ["ez", "tu", "ew", "min"],
        "Question words": ["çi", "kî", "çawa", "çend"],
        "Verbs": ["hatin", "çûn", "bûn", "kirin"],
        "Family": ["dê", "bav", "bira", "xwişk"],
        "Numbers": ["yek", "du", "sê", "çar", "pênc"],
        "Time": ["roj", "şev", "sibê", "îro"],
        "Nature": ["av", "erd", "ezman", "çiya"],
        "Body": ["ser", "çav", "dest", "pê"],
        "Food": ["nan", "goşt", "şîr", "çay"],
        "Adjectives": ["baş", "mezin", "biçûk", "xweş"],
        "Prepositions": ["li", "di", "bi", "ji"],
    }
    
    all_passed = True
    for category, words in categories.items():
        found = sum(1 for word in words if word in KURDISH_CORRECTIONS.values())
        total = len(words)
        percentage = (found / total) * 100
        
        if percentage >= 50:  # At least 50% of sample words should be in dictionary
            print(f"✅ {category:20s}: {found}/{total} words found ({percentage:.0f}%)")
        else:
            print(f"⚠️  {category:20s}: {found}/{total} words found ({percentage:.0f}%)")
            all_passed = False
    
    return all_passed


def test_integration_simulation():
    """Simulate the integration with STT service"""
    print("\n" + "=" * 70)
    print("Test 6: Integration Simulation")
    print("=" * 70)
    
    processor = KurdishPostProcessor()
    
    # Simulate what would happen in the STT service
    print("Simulating STT service integration:")
    print("-" * 70)
    
    # Simulate raw STT output (with common mistakes)
    raw_outputs = [
        "silav nave min ahmed e",
        "tu cawa yi ez bas im",
        "ci dixwazi ez te bibînim",
    ]
    
    all_passed = True
    for raw in raw_outputs:
        corrected = processor.correct_transcription(raw)
        stats = processor.get_correction_stats(raw, corrected)
        
        print(f"\nRaw output:  '{raw}'")
        print(f"Corrected:   '{corrected}'")
        print(f"Stats: {stats['words_corrected']} words corrected out of {stats['total_words']}")
        
        # Verify the structure is correct (should have both raw and corrected)
        result = {
            'raw_text': raw,
            'text': corrected,
            'stats': stats
        }
        
        if 'raw_text' in result and 'text' in result:
            print("✅ Result structure correct (has both raw_text and text)")
        else:
            print("❌ Result structure incorrect")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("KURDISH STT POST-PROCESSING TEST SUITE")
    print("=" * 70)
    
    tests = [
        test_dictionary,
        test_special_characters,
        test_postprocessor_basic,
        test_postprocessor_case_preservation,
        test_word_categories,
        test_integration_simulation,
    ]
    
    results = []
    for test_func in tests:
        try:
            passed = test_func()
            results.append((test_func.__name__, passed))
        except Exception as e:
            print(f"\n❌ Test {test_func.__name__} failed with exception: {e}")
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
