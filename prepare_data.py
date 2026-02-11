#!/usr/bin/env python3
"""
Data Preparation Script for Kurdish Kurmanji TTS Training

This script:
1. Loads amedcj/kurmanji-commonvoice dataset using soundfile backend
2. Filters for high-quality samples (up_votes >= 2, down_votes == 0)
3. Extracts audio to WAV files at 16kHz mono
4. Creates metadata CSV with file|text format
5. Shows dataset statistics (speakers, genders, durations, text samples)

Usage:
    python prepare_data.py --output_dir training --max_samples 1000
    
    # For full dataset (45,992 samples):
    python prepare_data.py --output_dir training --max_samples 0
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment variable to use soundfile backend BEFORE importing datasets
os.environ['HF_AUDIO_DECODER'] = 'soundfile'

import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Import datasets after setting environment variable
from datasets import load_dataset, Audio

print("=" * 80)
print("Kurdish Kurmanji TTS Data Preparation")
print("=" * 80)
print(f"Audio backend: {os.environ.get('HF_AUDIO_DECODER', 'default')}")
print("=" * 80)


class KurdishDataPreparation:
    """Handler for Kurdish Kurmanji dataset preparation"""
    
    def __init__(self, output_dir: str = "training"):
        """
        Initialize data preparation handler
        
        Args:
            output_dir: Base directory for output files
        """
        self.output_dir = Path(output_dir)
        self.wavs_dir = self.output_dir / "wavs"
        self.wavs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Output directory: {self.output_dir.absolute()}")
        print(f"✅ WAV files will be saved to: {self.wavs_dir.absolute()}")
    
    def load_dataset(self, split: str = "train") -> tuple:
        """
        Load Kurdish Kurmanji CommonVoice dataset using soundfile backend
        
        Args:
            split: Dataset split to load (train, test, validation)
            
        Returns:
            Tuple of (dataset, original_size)
        """
        print(f"\n📊 Loading amedcj/kurmanji-commonvoice dataset (split: {split})...")
        print("   This may take a few minutes on first run...")
        
        try:
            # Load dataset with soundfile audio decoder
            dataset = load_dataset(
                "amedcj/kurmanji-commonvoice",
                split=split,
                trust_remote_code=True
            )
            
            # Cast audio column to use soundfile explicitly
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
            
            original_size = len(dataset)
            print(f"   ✅ Loaded {original_size:,} samples")
            
            # Show available columns
            print(f"   Available columns: {', '.join(dataset.column_names)}")
            
            return dataset, original_size
            
        except Exception as e:
            print(f"   ❌ Error loading dataset: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Ensure you have internet connection")
            print("   2. Try: pip uninstall torchcodec -y")
            print("   3. Ensure soundfile is installed: pip install soundfile")
            raise
    
    def filter_quality_samples(
        self,
        dataset,
        min_upvotes: int = 2,
        max_downvotes: int = 0
    ):
        """
        Filter dataset for high-quality samples
        
        Args:
            dataset: HuggingFace dataset
            min_upvotes: Minimum up_votes required
            max_downvotes: Maximum down_votes allowed
            
        Returns:
            Filtered dataset
        """
        print(f"\n🔍 Filtering for quality (up_votes >= {min_upvotes}, down_votes <= {max_downvotes})...")
        initial_count = len(dataset)
        
        # Check if voting columns exist
        has_votes = 'up_votes' in dataset.column_names and 'down_votes' in dataset.column_names
        
        if has_votes:
            # Filter by votes
            dataset = dataset.filter(
                lambda x: x['up_votes'] >= min_upvotes and x['down_votes'] <= max_downvotes,
                desc="Filtering by votes"
            )
            filtered_count = len(dataset)
            print(f"   ✅ Kept {filtered_count:,} / {initial_count:,} samples ({filtered_count/initial_count*100:.1f}%)")
        else:
            print(f"   ⚠️  No voting columns found, keeping all {initial_count:,} samples")
        
        # Filter out samples without text
        dataset = dataset.filter(
            lambda x: x.get('text') or x.get('sentence'),
            desc="Filtering empty text"
        )
        
        final_count = len(dataset)
        if final_count < len(dataset) if has_votes else initial_count:
            print(f"   ✅ After removing empty text: {final_count:,} samples")
        
        return dataset
    
    def process_audio(
        self,
        dataset,
        max_samples: int = 0,
        target_sr: int = 16000
    ) -> List[Dict]:
        """
        Process audio files: convert to WAV at target sample rate
        
        Args:
            dataset: Filtered dataset
            max_samples: Maximum samples to process (0 = all)
            target_sr: Target sample rate in Hz
            
        Returns:
            List of metadata dictionaries
        """
        print(f"\n🎵 Processing audio files...")
        
        total_samples = len(dataset)
        if max_samples > 0 and max_samples < total_samples:
            dataset = dataset.select(range(max_samples))
            print(f"   Limited to {max_samples:,} samples (out of {total_samples:,})")
        else:
            print(f"   Processing all {total_samples:,} samples")
        
        metadata = []
        processed = 0
        skipped = 0
        
        for idx, sample in enumerate(tqdm(dataset, desc="Converting audio")):
            try:
                # Get audio data
                audio_data = sample['audio']
                audio_array = audio_data['array']
                sample_rate = audio_data['sampling_rate']
                
                # Get text (try different column names)
                text = sample.get('text') or sample.get('sentence', '')
                if not text or not text.strip():
                    skipped += 1
                    continue
                
                # Resample if needed
                if sample_rate != target_sr:
                    import librosa
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sample_rate,
                        target_sr=target_sr
                    )
                
                # Normalize audio
                audio_array = audio_array.astype(np.float32)
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array)) * 0.95
                
                # Generate output filename
                wav_filename = f"kmr_cv_{idx:06d}.wav"
                wav_path = self.wavs_dir / wav_filename
                
                # Save as WAV
                sf.write(wav_path, audio_array, target_sr)
                
                # Add to metadata
                metadata.append({
                    'filename': wav_filename,
                    'text': text.strip(),
                    'duration': len(audio_array) / target_sr,
                    'client_id': sample.get('client_id', ''),
                    'gender': sample.get('gender', ''),
                    'age': sample.get('age', ''),
                })
                
                processed += 1
                
            except Exception as e:
                print(f"\n   ⚠️  Error processing sample {idx}: {e}")
                skipped += 1
                continue
        
        print(f"\n   ✅ Successfully processed: {processed:,} files")
        if skipped > 0:
            print(f"   ⚠️  Skipped: {skipped} files")
        
        return metadata
    
    def create_metadata_csv(self, metadata: List[Dict]) -> Path:
        """
        Create metadata CSV in file|text format for training
        
        Args:
            metadata: List of metadata dictionaries
            
        Returns:
            Path to metadata CSV file
        """
        print(f"\n📝 Creating metadata CSV...")
        
        csv_path = self.output_dir / "metadata.csv"
        
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            
            for item in metadata:
                # Format: filename|text
                writer.writerow([item['filename'], item['text']])
        
        print(f"   ✅ Saved to: {csv_path.absolute()}")
        print(f"   Format: filename|text")
        print(f"   Total entries: {len(metadata):,}")
        
        return csv_path
    
    def show_statistics(self, metadata: List[Dict]):
        """
        Display dataset statistics
        
        Args:
            metadata: List of metadata dictionaries
        """
        print("\n" + "=" * 80)
        print("📊 DATASET STATISTICS")
        print("=" * 80)
        
        # Total samples
        total = len(metadata)
        print(f"\n✅ Total samples: {total:,}")
        
        # Duration statistics
        durations = [item['duration'] for item in metadata]
        total_duration = sum(durations)
        avg_duration = np.mean(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)
        
        print(f"\n⏱️  Duration statistics:")
        print(f"   Total duration: {total_duration/3600:.2f} hours ({total_duration:.0f} seconds)")
        print(f"   Average: {avg_duration:.2f} seconds")
        print(f"   Min: {min_duration:.2f} seconds")
        print(f"   Max: {max_duration:.2f} seconds")
        
        # Speaker statistics
        speakers = [item['client_id'] for item in metadata if item['client_id']]
        if speakers:
            unique_speakers = len(set(speakers))
            print(f"\n👥 Speakers: {unique_speakers:,} unique")
        
        # Gender statistics
        genders = [item['gender'] for item in metadata if item['gender']]
        if genders:
            gender_counts = Counter(genders)
            print(f"\n⚧️  Gender distribution:")
            for gender, count in gender_counts.most_common():
                print(f"   {gender}: {count:,} ({count/total*100:.1f}%)")
        
        # Age statistics
        ages = [item['age'] for item in metadata if item['age']]
        if ages:
            age_counts = Counter(ages)
            print(f"\n🎂 Age distribution:")
            for age, count in sorted(age_counts.items()):
                print(f"   {age}: {count:,} ({count/total*100:.1f}%)")
        
        # Text statistics
        text_lengths = [len(item['text']) for item in metadata]
        avg_text_len = np.mean(text_lengths)
        print(f"\n📝 Text statistics:")
        print(f"   Average length: {avg_text_len:.0f} characters")
        print(f"   Min length: {min(text_lengths)} characters")
        print(f"   Max length: {max(text_lengths)} characters")
        
        # Sample texts
        print(f"\n📄 Sample texts (first 5):")
        for i, item in enumerate(metadata[:5], 1):
            text = item['text']
            if len(text) > 80:
                text = text[:77] + "..."
            print(f"   {i}. {text}")
        
        print("\n" + "=" * 80)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Prepare Kurdish Kurmanji TTS training data from CommonVoice dataset"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='training',
        help='Output directory for processed data (default: training)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=0,
        help='Maximum samples to process (0 = all, default: 0)'
    )
    parser.add_argument(
        '--min_upvotes',
        type=int,
        default=2,
        help='Minimum up_votes for quality filter (default: 2)'
    )
    parser.add_argument(
        '--max_downvotes',
        type=int,
        default=0,
        help='Maximum down_votes for quality filter (default: 0)'
    )
    parser.add_argument(
        '--target_sr',
        type=int,
        default=16000,
        help='Target sample rate in Hz (default: 16000)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'test', 'validation'],
        help='Dataset split to use (default: train)'
    )
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Max samples: {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"  Quality filter: up_votes >= {args.min_upvotes}, down_votes <= {args.max_downvotes}")
    print(f"  Target sample rate: {args.target_sr} Hz")
    print(f"  Dataset split: {args.split}")
    print()
    
    try:
        # Initialize handler
        handler = KurdishDataPreparation(args.output_dir)
        
        # Load dataset
        dataset, original_size = handler.load_dataset(args.split)
        
        # Filter for quality
        dataset = handler.filter_quality_samples(
            dataset,
            min_upvotes=args.min_upvotes,
            max_downvotes=args.max_downvotes
        )
        
        # Process audio
        metadata = handler.process_audio(
            dataset,
            max_samples=args.max_samples,
            target_sr=args.target_sr
        )
        
        if not metadata:
            print("\n❌ No samples were processed successfully!")
            return 1
        
        # Create metadata CSV
        csv_path = handler.create_metadata_csv(metadata)
        
        # Show statistics
        handler.show_statistics(metadata)
        
        print(f"\n✅ Data preparation complete!")
        print(f"\n📁 Output files:")
        print(f"   WAV files: {handler.wavs_dir.absolute()}")
        print(f"   Metadata: {csv_path.absolute()}")
        print(f"\n🚀 Ready for training with train_vits.py or train_feedback.py")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
