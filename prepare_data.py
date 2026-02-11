#!/usr/bin/env python3
"""
Kurdish Data Preparation Script for MMS Fine-Tuning

This script prepares the Kurdish Common Voice dataset for MMS TTS fine-tuning by:
1. Loading audio data with decode=False to bypass torchcodec dependency
2. Manually decoding audio with soundfile
3. Filtering for quality (up_votes >= 2, down_votes == 0)
4. Resampling to 16kHz mono
5. Saving processed WAVs to training/wavs/
6. Creating training/metadata.csv in format: file|text

Dataset: amedcj/kurmanji-commonvoice on HuggingFace
Target Model: facebook/mms-tts-kmr-script_latin (36M params)
Hardware: RTX 2070 8GB VRAM
Python: 3.14, PyTorch 2.10.0+cu128, Windows
"""

import os
import sys
import io
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Audio processing
import soundfile as sf
import librosa

# HuggingFace datasets
from datasets import load_dataset, Audio

print("=" * 80)
print("Kurdish Data Preparation for MMS Fine-Tuning")
print("=" * 80)


class KurdishDataPreparation:
    """Handler for Kurdish Common Voice dataset preparation"""
    
    def __init__(self, output_dir: str = "training"):
        """
        Initialize data preparation handler
        
        Args:
            output_dir: Directory to save processed audio and metadata
        """
        self.output_dir = Path(output_dir)
        self.wavs_dir = self.output_dir / "wavs"
        self.wavs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Output directory: {self.output_dir}")
        print(f"✅ WAVs directory: {self.wavs_dir}")
    
    def load_dataset_with_audio_bytes(self, dataset_name: str = "amedcj/kurmanji-commonvoice") -> Dict:
        """
        Load dataset with decode=False to get raw audio bytes
        
        Args:
            dataset_name: HuggingFace dataset name
            
        Returns:
            Dataset object
        """
        print(f"\n📊 Loading dataset: {dataset_name}")
        print("   Using Audio(decode=False) to bypass torchcodec...")
        
        try:
            # Load dataset
            ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
            print(f"   Total samples: {len(ds)}")
            
            # Cast audio column to Audio(decode=False) to get raw bytes
            ds = ds.cast_column("path", Audio(decode=False))
            print("✅ Dataset loaded with raw audio bytes")
            
            return ds
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def filter_quality_samples(self, ds, min_upvotes: int = 2, max_downvotes: int = 0) -> List[int]:
        """
        Filter dataset for quality samples
        
        Args:
            ds: Dataset object
            min_upvotes: Minimum upvotes required
            max_downvotes: Maximum downvotes allowed
            
        Returns:
            List of valid sample indices
        """
        print("\n🔍 Filtering for quality samples...")
        print(f"   Criteria: up_votes >= {min_upvotes}, down_votes <= {max_downvotes}")
        
        valid_indices = []
        
        for idx in range(len(ds)):
            sample = ds[idx]
            
            # Check votes
            up_votes = sample.get('up_votes', 0)
            down_votes = sample.get('down_votes', 0)
            
            if up_votes >= min_upvotes and down_votes <= max_downvotes:
                # Check text is not empty
                text = sample.get('text', '').strip()
                if text:
                    valid_indices.append(idx)
        
        print(f"✅ Found {len(valid_indices)} quality samples ({len(valid_indices)/len(ds)*100:.1f}%)")
        
        # Show statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(ds)}")
        print(f"   High quality: {len(valid_indices)}")
        
        # Count speakers
        if 'client_id' in ds.column_names:
            client_ids = [ds[idx]['client_id'] for idx in valid_indices]
            unique_speakers = len(set(client_ids))
            print(f"   Unique speakers: {unique_speakers}")
        
        # Count by gender
        if 'gender' in ds.column_names:
            genders = [ds[idx].get('gender', 'unknown') for idx in valid_indices]
            gender_counts = pd.Series(genders).value_counts()
            print(f"   Gender distribution:")
            for gender, count in gender_counts.items():
                print(f"      {gender}: {count}")
        
        return valid_indices
    
    def decode_audio_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Decode audio bytes using soundfile
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Use BytesIO to create file-like object from bytes
            audio_io = io.BytesIO(audio_bytes)
            
            # Read audio with soundfile
            audio_data, sample_rate = sf.read(audio_io)
            
            return audio_data, sample_rate
            
        except Exception as e:
            raise Exception(f"Failed to decode audio: {e}")
    
    def process_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        target_sr: int = 16000
    ) -> np.ndarray:
        """
        Process audio: resample to 16kHz mono and normalize
        
        Args:
            audio_data: Audio data array
            sample_rate: Original sample rate
            target_sr: Target sample rate (default: 16000)
            
        Returns:
            Processed audio data
        """
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sample_rate != target_sr:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=target_sr
            )
        
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        return audio_data
    
    def prepare_training_data(
        self,
        ds,
        valid_indices: List[int],
        target_sr: int = 16000,
        max_samples: int = None
    ) -> List[Tuple[str, str]]:
        """
        Process audio files and create training data
        
        Args:
            ds: Dataset object
            valid_indices: List of valid sample indices
            target_sr: Target sample rate
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            List of tuples (wav_filename, text)
        """
        print(f"\n🔄 Processing audio files...")
        
        # Limit samples if specified
        if max_samples is not None and max_samples > 0:
            valid_indices = valid_indices[:max_samples]
            print(f"   Processing {len(valid_indices)} samples (limited by max_samples)")
        else:
            print(f"   Processing {len(valid_indices)} samples")
        
        processed_data = []
        failed_count = 0
        
        for idx in tqdm(valid_indices, desc="Processing"):
            try:
                sample = ds[idx]
                
                # Get audio bytes
                audio_bytes = sample['path']['bytes']
                
                # Decode audio
                audio_data, sr = self.decode_audio_bytes(audio_bytes)
                
                # Process audio (resample, normalize)
                audio_data = self.process_audio(audio_data, sr, target_sr)
                
                # Check duration (should be between 1-15 seconds)
                duration = len(audio_data) / target_sr
                if duration < 1.0 or duration > 15.0:
                    continue
                
                # Create filename
                wav_filename = f"audio_{idx:06d}.wav"
                wav_path = self.wavs_dir / wav_filename
                
                # Save WAV file
                sf.write(wav_path, audio_data, target_sr)
                
                # Get text
                text = sample['text'].strip()
                
                # Add to processed data
                processed_data.append((wav_filename, text))
                
            except Exception as e:
                failed_count += 1
                continue
        
        print(f"\n✅ Successfully processed: {len(processed_data)} files")
        if failed_count > 0:
            print(f"⚠️  Failed to process: {failed_count} files")
        
        return processed_data
    
    def create_metadata_csv(
        self,
        processed_data: List[Tuple[str, str]]
    ) -> str:
        """
        Create metadata.csv file in format: file|text
        
        Args:
            processed_data: List of (filename, text) tuples
            
        Returns:
            Path to metadata file
        """
        metadata_path = self.output_dir / "metadata.csv"
        
        print(f"\n📝 Creating metadata file: {metadata_path}")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for filename, text in processed_data:
                # Format: file|text (as expected by MMS training)
                f.write(f"{filename}|{text}\n")
        
        print(f"✅ Created metadata with {len(processed_data)} entries")
        
        return str(metadata_path)
    
    def show_statistics(self, processed_data: List[Tuple[str, str]]):
        """Display statistics about processed data"""
        print("\n" + "=" * 80)
        print("📊 Data Preparation Statistics")
        print("=" * 80)
        
        print(f"\n✅ Successfully processed samples: {len(processed_data)}")
        
        # Calculate total audio duration
        total_duration = 0
        for filename, _ in processed_data:
            wav_path = self.wavs_dir / filename
            if wav_path.exists():
                info = sf.info(str(wav_path))
                total_duration += info.duration
        
        print(f"📊 Total audio duration: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)")
        print(f"📊 Average audio length: {total_duration/len(processed_data):.1f} seconds")
        
        # Text statistics
        texts = [text for _, text in processed_data]
        text_lengths = [len(text) for text in texts]
        
        print(f"\n📝 Text Statistics:")
        print(f"   Average text length: {np.mean(text_lengths):.1f} characters")
        print(f"   Min text length: {np.min(text_lengths)} characters")
        print(f"   Max text length: {np.max(text_lengths)} characters")
        
        print(f"\n📁 Output Files:")
        print(f"   WAV files: {self.wavs_dir}/")
        print(f"   Metadata: {self.output_dir}/metadata.csv")
        
        print("\n" + "=" * 80)


def main():
    """Main data preparation pipeline"""
    parser = argparse.ArgumentParser(
        description="Prepare Kurdish Common Voice data for MMS fine-tuning"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="amedcj/kurmanji-commonvoice",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training",
        help="Directory to save processed audio and metadata"
    )
    parser.add_argument(
        "--min_upvotes",
        type=int,
        default=2,
        help="Minimum upvotes required"
    )
    parser.add_argument(
        "--max_downvotes",
        type=int,
        default=0,
        help="Maximum downvotes allowed"
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16000,
        help="Target sample rate (Hz)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for all)"
    )
    
    args = parser.parse_args()
    
    print("\n📋 Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Min upvotes: {args.min_upvotes}")
    print(f"   Max downvotes: {args.max_downvotes}")
    print(f"   Target sample rate: {args.target_sr} Hz")
    print(f"   Max samples: {args.max_samples or 'All'}")
    
    # Initialize data preparation
    data_prep = KurdishDataPreparation(output_dir=args.output_dir)
    
    # Load dataset
    ds = data_prep.load_dataset_with_audio_bytes(args.dataset)
    
    # Filter for quality
    valid_indices = data_prep.filter_quality_samples(
        ds,
        min_upvotes=args.min_upvotes,
        max_downvotes=args.max_downvotes
    )
    
    if len(valid_indices) == 0:
        print("\n❌ No valid samples found!")
        return 1
    
    # Process audio files
    processed_data = data_prep.prepare_training_data(
        ds,
        valid_indices,
        target_sr=args.target_sr,
        max_samples=args.max_samples
    )
    
    if len(processed_data) == 0:
        print("\n❌ No audio files were successfully processed!")
        return 1
    
    # Create metadata file
    data_prep.create_metadata_csv(processed_data)
    
    # Show statistics
    data_prep.show_statistics(processed_data)
    
    print("\n✅ Data preparation complete!")
    print("\n💡 Next Steps:")
    print("   1. Run training script: python train_vits.py")
    print("   2. Monitor training progress in training/checkpoints/")
    print("   3. Test fine-tuned model with your TTS service")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Data preparation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
