#!/usr/bin/env python3
"""
Data Preparation Script for VITS/MMS Kurdish TTS Fine-tuning

This script:
1. Loads the Kurdish Common Voice dataset using soundfile workaround for Windows
2. Filters high-quality samples (up_votes >= 2, down_votes == 0)
3. Resamples audio to 16kHz mono
4. Saves WAV files to training/wavs/
5. Creates training/metadata.csv with format: filename|text

IMPORTANT: This script sets HF_AUDIO_DECODER="soundfile" at import time to work around
torchcodec issues on Windows. This environment variable affects how the datasets library
decodes audio files.
"""

import os
# Set audio decoder to soundfile for Windows compatibility (must be before datasets import)
os.environ["HF_AUDIO_DECODER"] = "soundfile"

import io
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf
import librosa
import numpy as np
from datasets import load_dataset, Audio
from tqdm import tqdm
import pandas as pd


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prepare Kurdish TTS training data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training",
        help="Output directory for training data (default: training)"
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum number of samples to process (0 = all, default: 0)"
    )
    parser.add_argument(
        "--min_upvotes",
        type=int,
        default=2,
        help="Minimum up_votes for quality filter (default: 2)"
    )
    parser.add_argument(
        "--max_downvotes",
        type=int,
        default=0,
        help="Maximum down_votes for quality filter (default: 0)"
    )
    return parser.parse_args()


def load_kurdish_dataset():
    """
    Load Kurdish Common Voice dataset using soundfile workaround
    
    This uses the confirmed working solution for Windows:
    - Load with Audio(decode=False) to get raw bytes
    - Decode manually with soundfile
    """
    print("📦 Loading Kurdish Common Voice dataset...")
    print("   Using soundfile workaround for Windows compatibility")
    
    # Load dataset with Audio(decode=False) to get raw bytes
    ds = load_dataset("amedcj/kurmanji-commonvoice", split="train")
    ds = ds.cast_column("path", Audio(decode=False))
    
    print(f"✅ Loaded {len(ds)} samples")
    return ds


def filter_high_quality(ds, min_upvotes: int = 2, max_downvotes: int = 0):
    """
    Filter dataset for high quality samples
    
    Args:
        ds: Dataset to filter
        min_upvotes: Minimum up_votes required
        max_downvotes: Maximum down_votes allowed
        
    Returns:
        Filtered dataset
    """
    print(f"\n🔍 Filtering for quality (up_votes >= {min_upvotes}, down_votes == {max_downvotes})...")
    initial_count = len(ds)
    
    def quality_filter(sample):
        return (
            sample.get("up_votes", 0) >= min_upvotes and
            sample.get("down_votes", 0) <= max_downvotes and
            sample.get("sentence", "").strip() != ""
        )
    
    ds = ds.filter(quality_filter)
    filtered_count = len(ds)
    
    print(f"✅ Filtered: {filtered_count} / {initial_count} samples ({filtered_count/initial_count*100:.1f}%)")
    return ds


def decode_audio_with_soundfile(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode audio bytes using soundfile
    
    Args:
        audio_bytes: Raw audio bytes
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    return audio_data, sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate using librosa
    
    Args:
        audio: Audio data
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono if stereo
    
    Args:
        audio: Audio data (can be 1D or 2D)
        
    Returns:
        Mono audio (1D array)
    """
    if audio.ndim == 2:
        # Convert stereo to mono by averaging channels
        audio = audio.mean(axis=1)
    return audio


def get_speaker_stats(ds) -> Dict:
    """Calculate speaker statistics from dataset"""
    try:
        unique_speakers = set()
        gender_counts = {"male": 0, "female": 0, "other": 0, "unknown": 0}
        
        for sample in ds:
            client_id = sample.get("client_id", "unknown")
            unique_speakers.add(client_id)
            
            gender = sample.get("gender", "").lower()
            if gender in ["male", "man", "m"]:
                gender_counts["male"] += 1
            elif gender in ["female", "woman", "f"]:
                gender_counts["female"] += 1
            elif gender:
                gender_counts["other"] += 1
            else:
                gender_counts["unknown"] += 1
        
        return {
            "unique_speakers": len(unique_speakers),
            "gender_counts": gender_counts
        }
    except Exception as e:
        print(f"⚠️  Could not calculate speaker stats: {e}")
        return {"unique_speakers": 0, "gender_counts": {}}


def process_and_save_dataset(
    ds,
    output_dir: Path,
    target_sr: int = 16000,
    max_samples: int = 0
):
    """
    Process dataset and save WAV files with metadata
    
    Args:
        ds: Dataset to process
        output_dir: Output directory
        target_sr: Target sample rate
        max_samples: Maximum samples to process (0 = all)
    """
    # Create output directories
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Processing and saving audio files...")
    print(f"   Output directory: {output_dir}")
    print(f"   Target sample rate: {target_sr} Hz")
    
    # Determine number of samples to process
    num_samples = len(ds)
    if max_samples > 0:
        num_samples = min(max_samples, num_samples)
        print(f"   Processing: {num_samples} / {len(ds)} samples")
    else:
        print(f"   Processing: all {num_samples} samples")
    
    # Process samples
    metadata_rows = []
    total_duration = 0.0
    success_count = 0
    error_count = 0
    
    for idx in tqdm(range(num_samples), desc="Processing audio"):
        try:
            sample = ds[idx]
            
            # Get text
            text = sample.get("sentence", "").strip()
            if not text:
                error_count += 1
                continue
            
            # Get audio bytes
            audio_bytes = sample["path"]["bytes"]
            
            # Decode audio with soundfile
            audio_data, orig_sr = decode_audio_with_soundfile(audio_bytes)
            
            # Ensure mono
            audio_data = ensure_mono(audio_data)
            
            # Resample to target sample rate
            audio_data = resample_audio(audio_data, orig_sr, target_sr)
            
            # Calculate duration
            duration = len(audio_data) / target_sr
            total_duration += duration
            
            # Generate filename (kmr_XXXXX.wav format)
            filename = f"kmr_{idx:05d}.wav"
            output_path = wavs_dir / filename
            
            # Save as WAV
            sf.write(str(output_path), audio_data, target_sr)
            
            # Add to metadata
            metadata_rows.append({
                "filename": filename,
                "text": text
            })
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Only print first 5 errors
                print(f"\n⚠️  Error processing sample {idx}: {e}")
    
    # Save metadata to CSV
    metadata_path = output_dir / "metadata.csv"
    print(f"\n💾 Saving metadata to {metadata_path}...")
    
    df = pd.DataFrame(metadata_rows)
    # Save with pipe separator and no index or header
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['filename']}|{row['text']}\n")
    
    print(f"✅ Saved {len(metadata_rows)} entries to metadata.csv")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("📊 DATASET STATISTICS")
    print("=" * 70)
    print(f"Total samples processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    if success_count > 0:
        print(f"Average duration: {total_duration/success_count:.2f} seconds")
    print(f"WAV files saved to: {wavs_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Estimate training time
    if success_count > 0:
        print("\n⏱️  ESTIMATED TRAINING TIME")
        # Rough estimates based on typical training:
        # ~1-2 seconds per sample per epoch on RTX 2070
        seconds_per_sample = 1.5
        num_epochs = 10  # Default epochs
        estimated_seconds = success_count * seconds_per_sample * num_epochs
        estimated_hours = estimated_seconds / 3600
        print(f"Estimated time for {num_epochs} epochs: ~{estimated_hours:.1f} hours")
        print(f"  (Based on ~{seconds_per_sample:.1f}s per sample on RTX 2070 8GB)")
        print(f"  Actual time may vary based on GPU, batch size, and settings")
    
    # Get speaker stats
    print("\n📊 SPEAKER STATISTICS")
    stats = get_speaker_stats(ds)
    if stats["unique_speakers"] > 0:
        print(f"Unique speakers: {stats['unique_speakers']}")
        gender_counts = stats["gender_counts"]
        print(f"Gender distribution:")
        print(f"  Male: {gender_counts.get('male', 0)}")
        print(f"  Female: {gender_counts.get('female', 0)}")
        print(f"  Other: {gender_counts.get('other', 0)}")
        print(f"  Unknown: {gender_counts.get('unknown', 0)}")
    
    print("\n" + "=" * 70)
    print("✅ Data preparation complete!")
    print("=" * 70)
    
    # Show sample texts
    print("\n📝 Sample texts (first 5):")
    for i, row in enumerate(metadata_rows[:5]):
        print(f"  {i}: {row['text']}")


def main():
    """Main function"""
    args = parse_args()
    
    print("=" * 70)
    print("VITS/MMS Kurdish TTS Data Preparation")
    print("=" * 70)
    
    # Load dataset
    ds = load_kurdish_dataset()
    
    # Filter for quality
    ds = filter_high_quality(ds, args.min_upvotes, args.max_downvotes)
    
    # Process and save
    output_dir = Path(args.output_dir)
    process_and_save_dataset(
        ds,
        output_dir,
        target_sr=args.target_sr,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
