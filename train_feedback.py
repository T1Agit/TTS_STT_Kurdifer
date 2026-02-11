#!/usr/bin/env python3
"""
Feedback Loop Training Script for Kurdish MMS TTS

This script provides incremental fine-tuning capability for the MMS model based on
user feedback from the Base44 app. It accepts new audio+text correction pairs and
incrementally fine-tunes the model to improve quality.

Usage:
  1. Collect user feedback: audio file + corrected text
  2. Add feedback data to feedback/wavs/ and feedback/metadata.csv
  3. Run this script to incrementally fine-tune the model
  4. Deploy improved model

Integration with Base44 app:
  - Accept user corrections via API endpoint
  - Store feedback data automatically
  - Run periodic fine-tuning (e.g., weekly)
  - A/B test improved model before deployment
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import shutil

# Audio processing
import soundfile as sf
import librosa
import numpy as np

# HuggingFace
from transformers import VitsModel, VitsTokenizer
import torch
from torch.utils.data import Dataset

# Import our training utilities
try:
    from train_vits import MMSFineTuner, KurdishTTSDataset
except ImportError:
    print("❌ Could not import training utilities from train_vits.py")
    print("   Make sure train_vits.py is in the same directory")
    sys.exit(1)

print("=" * 80)
print("Feedback Loop Training for Kurdish MMS TTS")
print("=" * 80)


class FeedbackDataManager:
    """Manager for feedback data collection and processing"""
    
    def __init__(
        self,
        feedback_dir: str = "feedback",
        base_metadata_path: str = "training/metadata.csv",
        base_wavs_dir: str = "training/wavs"
    ):
        """
        Initialize feedback data manager
        
        Args:
            feedback_dir: Directory for feedback data
            base_metadata_path: Path to original training metadata
            base_wavs_dir: Directory with original training WAVs
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_wavs_dir = self.feedback_dir / "wavs"
        self.feedback_metadata_path = self.feedback_dir / "metadata.csv"
        
        self.base_metadata_path = Path(base_metadata_path)
        self.base_wavs_dir = Path(base_wavs_dir)
        
        # Create directories
        self.feedback_wavs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Feedback directory: {self.feedback_dir}")
        print(f"✅ Feedback WAVs: {self.feedback_wavs_dir}")
        print(f"✅ Feedback metadata: {self.feedback_metadata_path}")
    
    def add_feedback_sample(
        self,
        audio_path: str,
        corrected_text: str,
        sample_id: Optional[str] = None,
        target_sr: int = 16000
    ) -> bool:
        """
        Add a new feedback sample
        
        Args:
            audio_path: Path to audio file
            corrected_text: Corrected text transcription
            sample_id: Optional sample ID (auto-generated if not provided)
            target_sr: Target sample rate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate sample ID if not provided
            if sample_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                sample_id = f"feedback_{timestamp}"
            
            # Load and process audio
            audio, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            # Save to feedback directory
            output_filename = f"{sample_id}.wav"
            output_path = self.feedback_wavs_dir / output_filename
            sf.write(output_path, audio, target_sr)
            
            # Append to metadata
            with open(self.feedback_metadata_path, 'a', encoding='utf-8') as f:
                f.write(f"{output_filename}|{corrected_text}\n")
            
            print(f"✅ Added feedback sample: {output_filename}")
            print(f"   Text: {corrected_text}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to add feedback sample: {e}")
            return False
    
    def get_feedback_count(self) -> int:
        """Get number of feedback samples"""
        if not self.feedback_metadata_path.exists():
            return 0
        
        with open(self.feedback_metadata_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    
    def merge_with_base_data(
        self,
        output_dir: str = "training_with_feedback",
        feedback_weight: float = 2.0
    ) -> Tuple[str, str]:
        """
        Merge feedback data with base training data
        
        Args:
            output_dir: Output directory for merged data
            feedback_weight: Weight for feedback samples (repeat N times)
            
        Returns:
            Tuple of (merged_metadata_path, merged_wavs_dir)
        """
        output_dir = Path(output_dir)
        output_wavs_dir = output_dir / "wavs"
        output_metadata_path = output_dir / "metadata.csv"
        
        output_wavs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 Merging feedback data with base training data...")
        print(f"   Feedback weight: {feedback_weight}x")
        
        # Copy base training data
        base_count = 0
        if self.base_metadata_path.exists():
            with open(self.base_metadata_path, 'r', encoding='utf-8') as f_in:
                with open(output_metadata_path, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        line = line.strip()
                        if line:
                            f_out.write(line + '\n')
                            base_count += 1
                            
                            # Copy WAV file
                            filename = line.split('|')[0]
                            src = self.base_wavs_dir / filename
                            dst = output_wavs_dir / filename
                            if src.exists() and not dst.exists():
                                shutil.copy2(src, dst)
        
        print(f"✅ Copied {base_count} base training samples")
        
        # Add feedback data (with repetition for higher weight)
        feedback_count = 0
        if self.feedback_metadata_path.exists():
            with open(self.feedback_metadata_path, 'r', encoding='utf-8') as f_in:
                with open(output_metadata_path, 'a', encoding='utf-8') as f_out:
                    for line in f_in:
                        line = line.strip()
                        if line:
                            # Repeat feedback samples for higher weight
                            for _ in range(int(feedback_weight)):
                                f_out.write(line + '\n')
                            feedback_count += 1
                            
                            # Copy WAV file
                            filename = line.split('|')[0]
                            src = self.feedback_wavs_dir / filename
                            dst = output_wavs_dir / filename
                            if src.exists() and not dst.exists():
                                shutil.copy2(src, dst)
        
        print(f"✅ Added {feedback_count} feedback samples (weighted {feedback_weight}x)")
        print(f"📊 Total training samples: {base_count + int(feedback_count * feedback_weight)}")
        
        return str(output_metadata_path), str(output_wavs_dir)
    
    def archive_feedback(self, archive_dir: str = "feedback_archive"):
        """
        Archive processed feedback data
        
        Args:
            archive_dir: Directory to archive feedback data
        """
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_subdir = archive_dir / f"feedback_{timestamp}"
        
        # Move feedback data to archive
        if self.feedback_dir.exists():
            shutil.move(str(self.feedback_dir), str(archive_subdir))
            print(f"✅ Archived feedback data to {archive_subdir}")
            
            # Recreate feedback directory
            self.feedback_wavs_dir.mkdir(parents=True, exist_ok=True)


def main():
    """Main feedback training pipeline"""
    parser = argparse.ArgumentParser(
        description="Incremental fine-tuning with user feedback"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="training/checkpoints/best_model",
        help="Path to base model to fine-tune"
    )
    parser.add_argument(
        "--feedback_dir",
        type=str,
        default="feedback",
        help="Directory containing feedback data"
    )
    parser.add_argument(
        "--base_metadata",
        type=str,
        default="training/metadata.csv",
        help="Path to base training metadata"
    )
    parser.add_argument(
        "--base_wavs_dir",
        type=str,
        default="training/wavs",
        help="Directory containing base training WAVs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/checkpoints_feedback",
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--feedback_weight",
        type=float,
        default=2.0,
        help="Weight for feedback samples (repeat N times)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate (lower for incremental training)"
    )
    parser.add_argument(
        "--archive_feedback",
        action="store_true",
        help="Archive feedback data after training"
    )
    parser.add_argument(
        "--add_sample",
        type=str,
        nargs=2,
        metavar=("AUDIO_PATH", "TEXT"),
        help="Add a single feedback sample: audio_path text"
    )
    
    args = parser.parse_args()
    
    # Initialize feedback manager
    feedback_manager = FeedbackDataManager(
        feedback_dir=args.feedback_dir,
        base_metadata_path=args.base_metadata,
        base_wavs_dir=args.base_wavs_dir
    )
    
    # Handle single sample addition
    if args.add_sample:
        audio_path, text = args.add_sample
        success = feedback_manager.add_feedback_sample(audio_path, text)
        if success:
            print("\n✅ Feedback sample added successfully")
            print(f"   Total feedback samples: {feedback_manager.get_feedback_count()}")
        return 0 if success else 1
    
    # Check if there's feedback data to train on
    feedback_count = feedback_manager.get_feedback_count()
    if feedback_count == 0:
        print("\n⚠️  No feedback data found!")
        print(f"   Add feedback samples to: {args.feedback_dir}/wavs/")
        print(f"   And update metadata: {args.feedback_dir}/metadata.csv")
        print("\n💡 Or use --add_sample flag to add individual samples:")
        print("   python train_feedback.py --add_sample audio.wav 'Corrected text'")
        return 1
    
    print(f"\n📊 Found {feedback_count} feedback samples")
    
    # Merge feedback with base data
    merged_metadata, merged_wavs = feedback_manager.merge_with_base_data(
        output_dir="training_with_feedback",
        feedback_weight=args.feedback_weight
    )
    
    # Check if base model exists
    if not Path(args.model_path).exists():
        print(f"\n⚠️  Base model not found: {args.model_path}")
        print("   Using default model: facebook/mms-tts-kmr-script_latin")
        model_name = "facebook/mms-tts-kmr-script_latin"
    else:
        print(f"\n✅ Using base model: {args.model_path}")
        model_name = args.model_path
    
    print("\n" + "=" * 80)
    print("Starting Incremental Fine-Tuning")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Base model: {model_name}")
    print(f"   Feedback samples: {feedback_count}")
    print(f"   Feedback weight: {args.feedback_weight}x")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Output: {args.output_dir}")
    
    # Initialize fine-tuner
    fine_tuner = MMSFineTuner(
        model_name=model_name,
        output_dir=args.output_dir
    )
    
    # Create dataset with merged data
    print("\n📊 Loading training dataset...")
    train_dataset = KurdishTTSDataset(
        metadata_path=merged_metadata,
        wavs_dir=merged_wavs,
        tokenizer=fine_tuner.tokenizer,
        target_sr=16000
    )
    
    # Start training
    fine_tuner.train(
        train_dataset=train_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        save_steps=500
    )
    
    # Archive feedback if requested
    if args.archive_feedback:
        feedback_manager.archive_feedback()
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ Incremental Fine-Tuning Complete!")
    print("=" * 80)
    print(f"\n📁 Output locations:")
    print(f"   Improved model: {args.output_dir}/best_model/")
    print(f"   Training log: {args.output_dir}/training_log.txt")
    
    print("\n💡 Next Steps:")
    print(f"   1. Test improved model")
    print(f"   2. Compare with previous version (A/B testing)")
    print(f"   3. Deploy if quality improves")
    print(f"   4. Continue collecting feedback for next iteration")
    
    if not args.archive_feedback:
        print(f"\n⚠️  Feedback data not archived")
        print(f"   Use --archive_feedback flag to archive after training")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
