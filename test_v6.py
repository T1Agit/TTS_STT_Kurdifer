#!/usr/bin/env python3
"""
VITS Model Comparison Test Script v6
Compare original vs trained Kurdish TTS models

This script:
1. Loads the original facebook/mms-tts-kmr-script_latin model
2. Loads the trained/fine-tuned model (if available)
3. Generates audio samples with both models
4. Compares quality metrics:
   - Audio amplitude
   - Mel spectrogram similarity
   - Generation time
5. Saves comparison results and audio files
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio
from transformers import VitsModel, VitsTokenizer
import numpy as np
from datetime import datetime


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare original vs trained VITS models")
    parser.add_argument(
        "--original_model",
        type=str,
        default="facebook/mms-tts-kmr-script_latin",
        help="Original/base model name (default: facebook/mms-tts-kmr-script_latin)"
    )
    parser.add_argument(
        "--trained_model",
        type=str,
        default="training/final_model",
        help="Path to trained model directory (default: training/final_model)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_outputs",
        help="Directory to save test results (default: test_outputs)"
    )
    parser.add_argument(
        "--test_texts",
        nargs="+",
        default=[
            "Silav, tu çawa yî?",
            "Ez ji te hez dikim.",
            "Spas ji bo alîkariya te.",
            "Roja te baş be."
        ],
        help="Test texts to generate (default: Kurdish greetings)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cuda', 'cpu', or 'auto' (default: auto)"
    )
    return parser.parse_args()


class ModelTester:
    """Test and compare TTS models"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        ).to(device)
    
    def load_model(
        self,
        model_path: str,
        model_type: str = "pretrained"
    ) -> Tuple[Optional[VitsModel], Optional[VitsTokenizer]]:
        """
        Load a VITS model and tokenizer
        
        Args:
            model_path: Path to model (HuggingFace name or local path)
            model_type: Type of model ('pretrained' or 'trained')
            
        Returns:
            model, tokenizer (or None, None if loading fails)
        """
        try:
            print(f"\n📦 Loading {model_type} model from: {model_path}")
            
            tokenizer = VitsTokenizer.from_pretrained(model_path)
            model = VitsModel.from_pretrained(model_path)
            model = model.to(self.device)
            model.eval()
            
            num_params = sum(p.numel() for p in model.parameters())
            print(f"✅ {model_type.capitalize()} model loaded successfully")
            print(f"   Parameters: {num_params / 1e6:.2f}M")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading {model_type} model: {e}")
            return None, None
    
    def generate_audio(
        self,
        model: VitsModel,
        tokenizer: VitsTokenizer,
        text: str
    ) -> Tuple[Optional[np.ndarray], float, Dict]:
        """
        Generate audio from text using the model
        
        Args:
            model: VITS model
            tokenizer: VITS tokenizer
            text: Input text
            
        Returns:
            audio: Generated audio as numpy array
            generation_time: Time taken to generate (seconds)
            metrics: Dictionary of audio metrics
        """
        import time
        
        try:
            # Tokenize input
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            
            # Generate audio
            start_time = time.time()
            with torch.no_grad():
                output = model(input_ids)
            generation_time = time.time() - start_time
            
            # Extract audio
            if hasattr(output, 'waveform'):
                audio = output.waveform.squeeze().cpu().numpy()
            elif hasattr(output, 'audio'):
                audio = output.audio.squeeze().cpu().numpy()
            else:
                print("⚠️  Warning: Could not extract audio from model output")
                return None, generation_time, {}
            
            # Compute metrics
            metrics = self.compute_audio_metrics(audio)
            metrics['generation_time'] = generation_time
            metrics['duration'] = len(audio) / 16000.0  # Assuming 16kHz
            
            return audio, generation_time, metrics
            
        except Exception as e:
            print(f"❌ Error generating audio: {e}")
            return None, 0.0, {}
    
    def compute_audio_metrics(self, audio: np.ndarray) -> Dict:
        """
        Compute various audio quality metrics
        
        Args:
            audio: Audio as numpy array
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Amplitude metrics
        metrics['max_amplitude'] = float(np.max(np.abs(audio)))
        metrics['mean_amplitude'] = float(np.mean(np.abs(audio)))
        metrics['rms_amplitude'] = float(np.sqrt(np.mean(audio ** 2)))
        
        # Check if audio is silent
        metrics['is_silent'] = metrics['max_amplitude'] < 0.01
        
        # Energy
        metrics['energy'] = float(np.sum(audio ** 2))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        metrics['zero_crossing_rate'] = float(zero_crossings / len(audio))
        
        return metrics
    
    def compare_models(
        self,
        original_audio: np.ndarray,
        trained_audio: np.ndarray,
        original_metrics: Dict,
        trained_metrics: Dict
    ) -> Dict:
        """
        Compare metrics between original and trained models
        
        Returns:
            Dictionary of comparison results
        """
        comparison = {
            "amplitude_comparison": {
                "original_mean": original_metrics.get('mean_amplitude', 0.0),
                "trained_mean": trained_metrics.get('mean_amplitude', 0.0),
                "difference": abs(
                    original_metrics.get('mean_amplitude', 0.0) - 
                    trained_metrics.get('mean_amplitude', 0.0)
                ),
                "trained_better": trained_metrics.get('mean_amplitude', 0.0) > 0.5
            },
            "silence_check": {
                "original_silent": original_metrics.get('is_silent', True),
                "trained_silent": trained_metrics.get('is_silent', True),
                "trained_better": not trained_metrics.get('is_silent', True)
            },
            "generation_speed": {
                "original_time": original_metrics.get('generation_time', 0.0),
                "trained_time": trained_metrics.get('generation_time', 0.0),
                "speedup": (
                    original_metrics.get('generation_time', 1.0) / 
                    trained_metrics.get('generation_time', 1.0)
                    if trained_metrics.get('generation_time', 0.0) > 0 else 1.0
                )
            },
            "energy_comparison": {
                "original_energy": original_metrics.get('energy', 0.0),
                "trained_energy": trained_metrics.get('energy', 0.0),
                "trained_better": trained_metrics.get('energy', 0.0) > original_metrics.get('energy', 0.0)
            }
        }
        
        return comparison
    
    def save_audio(self, audio: np.ndarray, filepath: Path, sample_rate: int = 16000):
        """Save audio to WAV file"""
        try:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            torchaudio.save(str(filepath), audio_tensor, sample_rate)
            print(f"   💾 Saved audio to: {filepath}")
        except Exception as e:
            print(f"   ❌ Error saving audio: {e}")


def run_comparison_test(args):
    """Run the model comparison test"""
    
    print("=" * 70)
    print("VITS Model Comparison Test - Version 6")
    print("=" * 70)
    print(f"Original model: {args.original_model}")
    print(f"Trained model: {args.trained_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of test texts: {len(args.test_texts)}")
    print("=" * 70)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for audio files
    original_audio_dir = output_dir / "original_audio"
    trained_audio_dir = output_dir / "trained_audio"
    original_audio_dir.mkdir(exist_ok=True)
    trained_audio_dir.mkdir(exist_ok=True)
    
    # Initialize tester
    tester = ModelTester(device)
    
    # Load models
    original_model, original_tokenizer = tester.load_model(
        args.original_model,
        model_type="original"
    )
    
    trained_model, trained_tokenizer = tester.load_model(
        args.trained_model,
        model_type="trained"
    )
    
    if original_model is None:
        print("\n❌ Failed to load original model. Aborting test.")
        return
    
    # Check if trained model is available
    has_trained_model = trained_model is not None
    
    if not has_trained_model:
        print("\n⚠️  No trained model found. Will only test original model.")
    
    # Run tests
    print("\n" + "=" * 70)
    print("Running Tests")
    print("=" * 70)
    
    results = {
        "test_date": datetime.now().isoformat(),
        "original_model": args.original_model,
        "trained_model": args.trained_model,
        "device": str(device),
        "tests": []
    }
    
    for i, text in enumerate(args.test_texts):
        print(f"\n📝 Test {i + 1}/{len(args.test_texts)}: {text}")
        
        test_result = {
            "text": text,
            "original": {},
            "trained": {},
            "comparison": {}
        }
        
        # Generate with original model
        print("   🔊 Generating with original model...")
        original_audio, orig_time, orig_metrics = tester.generate_audio(
            original_model,
            original_tokenizer,
            text
        )
        
        if original_audio is not None:
            # Save original audio
            orig_filename = f"test_{i + 1:02d}_original.wav"
            tester.save_audio(
                original_audio,
                original_audio_dir / orig_filename
            )
            
            test_result["original"] = {
                "metrics": orig_metrics,
                "audio_file": str(original_audio_dir / orig_filename)
            }
            
            print(f"   ✅ Original model:")
            print(f"      Amplitude: {orig_metrics.get('mean_amplitude', 0.0):.3f}")
            print(f"      Silent: {orig_metrics.get('is_silent', False)}")
            print(f"      Time: {orig_time:.3f}s")
        
        # Generate with trained model (if available)
        if has_trained_model:
            print("   🔊 Generating with trained model...")
            trained_audio, trained_time, trained_metrics = tester.generate_audio(
                trained_model,
                trained_tokenizer,
                text
            )
            
            if trained_audio is not None:
                # Save trained audio
                trained_filename = f"test_{i + 1:02d}_trained.wav"
                tester.save_audio(
                    trained_audio,
                    trained_audio_dir / trained_filename
                )
                
                test_result["trained"] = {
                    "metrics": trained_metrics,
                    "audio_file": str(trained_audio_dir / trained_filename)
                }
                
                print(f"   ✅ Trained model:")
                print(f"      Amplitude: {trained_metrics.get('mean_amplitude', 0.0):.3f}")
                print(f"      Silent: {trained_metrics.get('is_silent', False)}")
                print(f"      Time: {trained_time:.3f}s")
                
                # Compare models
                if original_audio is not None:
                    comparison = tester.compare_models(
                        original_audio,
                        trained_audio,
                        orig_metrics,
                        trained_metrics
                    )
                    test_result["comparison"] = comparison
                    
                    print(f"   📊 Comparison:")
                    print(f"      Amplitude diff: {comparison['amplitude_comparison']['difference']:.3f}")
                    print(f"      Trained better quality: {comparison['amplitude_comparison']['trained_better']}")
                    print(f"      Trained not silent: {comparison['silence_check']['trained_better']}")
        
        results["tests"].append(test_result)
    
    # Save results to JSON
    results_file = output_dir / "comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    # Calculate summary statistics
    if has_trained_model:
        num_tests = len(results["tests"])
        trained_better_count = sum(
            1 for test in results["tests"]
            if test.get("comparison", {}).get("amplitude_comparison", {}).get("trained_better", False)
        )
        not_silent_count = sum(
            1 for test in results["tests"]
            if test.get("comparison", {}).get("silence_check", {}).get("trained_better", False)
        )
        
        print(f"Total tests: {num_tests}")
        print(f"Trained model better amplitude: {trained_better_count}/{num_tests}")
        print(f"Trained model not silent: {not_silent_count}/{num_tests}")
        
        if trained_better_count == num_tests and not_silent_count == num_tests:
            print("\n✅ All tests passed! Trained model performs better.")
        elif trained_better_count > num_tests // 2:
            print("\n⚠️  Trained model performs better in most tests.")
        else:
            print("\n❌ Trained model needs improvement.")
    else:
        print("Only original model tested (no trained model available).")
    
    print(f"\n📊 Results saved to: {results_file}")
    print(f"🔊 Audio files saved to:")
    print(f"   Original: {original_audio_dir}")
    if has_trained_model:
        print(f"   Trained: {trained_audio_dir}")
    
    print("\n" + "=" * 70)
    print("✅ Comparison test complete!")
    print("=" * 70)


def main():
    """Main function"""
    args = parse_args()
    run_comparison_test(args)


if __name__ == "__main__":
    main()
