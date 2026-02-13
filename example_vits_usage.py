#!/usr/bin/env python3
"""
Example: Using VITS v8 Model Integration

This script demonstrates how to use the new model selection feature
for Kurdish TTS with the Base44 app.
"""

from tts_stt_service_base44 import TTSSTTServiceBase44


def example_auto_selection():
    """Example 1: Auto-select best available model"""
    print("=" * 70)
    print("Example 1: Auto-select best available model")
    print("=" * 70)
    
    service = TTSSTTServiceBase44()
    
    # Auto-select best model (prefers v8 if available)
    result = service.text_to_speech_base44(
        text="Silav, tu çawa yî?",
        language="kurdish"
    )
    
    print(f"Model used: {result['model']}")
    print(f"Audio size: {result['size']} bytes")
    
    # Save audio
    service.save_audio_from_base44(result['audio'], 'output_auto.mp3')
    print("Saved to: output_auto.mp3\n")


def example_explicit_v8():
    """Example 2: Explicitly use VITS v8 model"""
    print("=" * 70)
    print("Example 2: Explicitly use VITS v8 model")
    print("=" * 70)
    
    service = TTSSTTServiceBase44()
    
    # Use v8 model explicitly
    result = service.text_to_speech_base44(
        text="Silav, tu çawa yî?",
        language="kurdish",
        model_version="v8"
    )
    
    print(f"Model used: {result['model']}")
    print(f"Audio size: {result['size']} bytes")
    
    # Save audio
    service.save_audio_from_base44(result['audio'], 'output_v8.mp3')
    print("Saved to: output_v8.mp3\n")


def example_explicit_original():
    """Example 3: Use original Facebook MMS model"""
    print("=" * 70)
    print("Example 3: Use original Facebook MMS model")
    print("=" * 70)
    
    service = TTSSTTServiceBase44()
    
    # Use original model explicitly
    result = service.text_to_speech_base44(
        text="Silav, tu çawa yî?",
        language="kurdish",
        model_version="original"
    )
    
    print(f"Model used: {result['model']}")
    print(f"Audio size: {result['size']} bytes")
    
    # Save audio
    service.save_audio_from_base44(result['audio'], 'output_original.mp3')
    print("Saved to: output_original.mp3\n")


def example_ab_comparison():
    """Example 4: A/B comparison of all models"""
    print("=" * 70)
    print("Example 4: A/B comparison of all models")
    print("=" * 70)
    
    service = TTSSTTServiceBase44()
    text = "Rojbûna te pîroz be"
    
    # Test all available models
    models = ['v8', 'original', 'coqui']
    
    for model in models:
        print(f"\nGenerating with {model} model...")
        try:
            result = service.text_to_speech_base44(
                text=text,
                language="kurdish",
                model_version=model
            )
            
            filename = f'output_comparison_{model}.mp3'
            service.save_audio_from_base44(result['audio'], filename)
            
            print(f"  ✓ Model: {result['model']}")
            print(f"  ✓ Size: {result['size']} bytes")
            print(f"  ✓ Saved to: {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\nComparison complete! Listen to the files to compare quality.\n")


def example_list_models():
    """Example 5: List available models"""
    print("=" * 70)
    print("Example 5: List available models")
    print("=" * 70)
    
    service = TTSSTTServiceBase44()
    
    models_info = service.get_available_models('kurdish')
    
    print(f"\nLanguage: {models_info['language']}")
    print(f"Default model: {models_info['default_model']}")
    print(f"\nAvailable models:")
    
    for model_name in models_info['models']:
        info = models_info['model_info'].get(model_name, {})
        model_type = info.get('type', 'Unknown')
        description = info.get('description', 'No description')
        
        print(f"  • {model_name}")
        print(f"    Type: {model_type}")
        print(f"    Description: {description}")


def main():
    """Run all examples"""
    print("\n🎤 VITS v8 Integration Examples\n")
    
    examples = [
        ("Auto-select best model", example_auto_selection),
        ("Explicitly use v8", example_explicit_v8),
        ("Use original model", example_explicit_original),
        ("A/B comparison", example_ab_comparison),
        ("List available models", example_list_models)
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nSelect an example to run (1-5), or 'all' to run all:")
    choice = input("> ").strip().lower()
    
    if choice == 'all':
        for name, func in examples:
            print()
            try:
                func()
            except Exception as e:
                print(f"❌ Error in '{name}': {e}\n")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        name, func = examples[int(choice) - 1]
        try:
            func()
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
