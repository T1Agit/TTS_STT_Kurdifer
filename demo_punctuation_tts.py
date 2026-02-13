#!/usr/bin/env python3
"""
Demo script to showcase punctuation-aware TTS
This demonstrates the text splitting and silence insertion without requiring model downloads
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vits_tts_service import VitsTTSService

def demo_text_processing():
    """Demo the text processing without model inference"""
    print("=" * 70)
    print("Punctuation-Aware TTS Demo")
    print("=" * 70)
    
    # Create service instance
    service = VitsTTSService.__new__(VitsTTSService)
    
    # Example text from the problem statement
    test_text = "Silav, tu çawa yî? Ez baş im."
    
    print(f"\n📝 Original text:")
    print(f"   '{test_text}'")
    
    # Split into segments
    segments = service._split_text_on_punctuation(test_text)
    
    print(f"\n🔍 Text will be split into {len(segments)} segments:")
    for i, (segment_text, punctuation) in enumerate(segments, 1):
        silence_ms = service._get_silence_duration(punctuation)
        print(f"\n   Segment {i}:")
        print(f"      Text: '{segment_text}'")
        print(f"      Punctuation: '{punctuation}'")
        print(f"      Silence after: {silence_ms}ms")
    
    print("\n📊 Processing summary:")
    print(f"   • Total segments: {len(segments)}")
    total_silence = sum(service._get_silence_duration(p) for _, p in segments)
    print(f"   • Total silence time: {total_silence}ms")
    
    # Show more examples
    print("\n" + "=" * 70)
    print("Additional Examples")
    print("=" * 70)
    
    examples = [
        "Hello world",
        "First sentence. Second sentence.",
        "Question one? Question two!",
        "Item one, item two, item three.",
        "Start; middle: end.",
    ]
    
    for example in examples:
        segments = service._split_text_on_punctuation(example)
        print(f"\n'{example}'")
        print(f"  → {len(segments)} segment(s):")
        for segment_text, punctuation in segments:
            silence = service._get_silence_duration(punctuation)
            print(f"     • '{segment_text}'{' + ' + str(silence) + 'ms pause' if silence else ''}")
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("\n💡 Key features:")
    print("   • Splits text on punctuation marks (. ? ! , ; :)")
    print("   • Preserves punctuation with each segment")
    print("   • Inserts appropriate silence gaps:")
    print("      - Period, Question, Exclamation: 500ms")
    print("      - Comma: 250ms")
    print("      - Semicolon: 350ms")
    print("      - Colon: 300ms")
    print("   • Skips empty segments")
    print("   • Concatenates all audio with pydub")
    print("=" * 70)

if __name__ == "__main__":
    demo_text_processing()
