#!/usr/bin/env python3
"""
XTTS v2 Fine-Tuning Script for Kurdish (Kurmanji)

This script actually fine-tunes the XTTS v2 multilingual model on Kurdish Common Voice data
using Coqui TTS's built-in fine-tuning API.

Requirements:
- Mozilla Common Voice Kurdish corpus (cv-corpus-24.0-2025-12-05-kmr)
- 8GB+ VRAM (RTX 2070 or better)
- ~20GB disk space for processed data
- Coqui TTS 0.27.5+ with all dependencies

Dataset: Mozilla Common Voice Kurdish (Kurmanji) v24.0
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import shutil
from datetime import datetime

# Audio processing
import librosa
import soundfile as sf
from tqdm import tqdm

# Deep learning
import torch

print("=" * 80)
print("XTTS v2 Fine-Tuning for Kurdish (Kurmanji)")
print("=" * 80)


class CommonVoiceDataPreparation:
    """Handler for Mozilla Common Voice Kurdish dataset preparation"""
    
    def __init__(self, corpus_path: str, output_dir: str):
        """
        Initialize dataset handler
        
        Args:
            corpus_path: Path to Common Voice corpus directory
            output_dir: Directory to save processed audio files
        """
        self.corpus_path = Path(corpus_path)
        self.clips_dir = self.corpus_path / "clips"
        self.output_dir = Path(output_dir)
        self.wavs_dir = self.output_dir / "wavs"
        self.wavs_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if corpus exists
        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus path not found: {self.corpus_path}\n"
                f"Please download the Mozilla Common Voice Kurdish corpus and extract it."
            )
        
        if not self.clips_dir.exists():
            raise FileNotFoundError(
                f"Clips directory not found: {self.clips_dir}\n"
                f"Expected structure: {self.corpus_path}/clips/"
            )
        
        print(f"✅ Found corpus at: {self.corpus_path}")
        print(f"✅ Found clips at: {self.clips_dir}")
    
    def load_tsv(self, tsv_name: str = "validated.tsv") -> pd.DataFrame:
        """Load and parse Common Voice TSV file"""
        tsv_path = self.corpus_path / tsv_name
        
        if not tsv_path.exists():
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")
        
        print(f"\n📊 Loading {tsv_name}...")
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"   Total entries: {len(df)}")
        
        return df
    
    def filter_quality_clips(
        self,
        df: pd.DataFrame,
        min_upvotes: int = 2,
        max_downvotes: int = 0
    ) -> pd.DataFrame:
        """Filter clips for quality"""
        print("\n🔍 Filtering for quality...")
        initial_count = len(df)
        
        # Filter by votes
        if 'up_votes' in df.columns and 'down_votes' in df.columns:
            df = df[
                (df['up_votes'] >= min_upvotes) &
                (df['down_votes'] <= max_downvotes)
            ]
            print(f"   After vote filtering: {len(df)} clips ({len(df)/initial_count*100:.1f}%)")
        
        # Remove entries with missing sentence
        df = df.dropna(subset=['sentence'])
        df = df[df['sentence'].str.strip() != '']
        print(f"   After text filtering: {len(df)} clips ({len(df)/initial_count*100:.1f}%)")
        
        return df
    
    def process_audio_file(
        self,
        audio_path: Path,
        output_path: Path,
        target_sr: int = 22050
    ) -> Tuple[bool, float]:
        """
        Load and process audio file, converting MP3 to WAV
        
        Returns:
            Tuple of (success, duration)
        """
        try:
            # Skip if already processed
            if output_path.exists():
                audio, sr = librosa.load(str(output_path), sr=target_sr, mono=True)
                duration = len(audio) / sr
                return True, duration
            
            # Load audio with librosa (handles MP3 files)
            audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
            
            # Check duration (2-15 seconds is good for training)
            duration = len(audio) / sr
            if duration < 2.0 or duration > 15.0:
                return False, duration
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Save as WAV
            sf.write(output_path, audio, sr)
            
            return True, duration
            
        except Exception as e:
            return False, 0.0
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_sr: int = 22050,
        max_samples: Optional[int] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Process and prepare training data
        
        Returns:
            List of tuples (wav_path, text, speaker_id)
        """
        print("\n🔄 Processing audio files...")
        
        processed_data = []
        total = len(df) if max_samples is None or max_samples == 0 else min(len(df), max_samples)
        
        # Use tqdm for progress bar - iterate over index to properly track progress
        df_subset = df.head(total)
        for idx in tqdm(df_subset.index, desc="Processing audio"):
            row = df_subset.loc[idx]
            
            # Get audio file path
            audio_filename = row['path']
            audio_path = self.clips_dir / audio_filename
            
            if not audio_path.exists():
                continue
            
            # Output WAV path
            wav_filename = audio_filename.replace('.mp3', '.wav')
            output_path = self.wavs_dir / wav_filename
            
            # Process audio
            success, duration = self.process_audio_file(audio_path, output_path, target_sr)
            
            if not success:
                continue
            
            # Add to training data
            # Format: (wav_path, text, speaker_id)
            processed_data.append((
                str(output_path.relative_to(self.output_dir)),
                row['sentence'].strip(),
                row.get('client_id', 'unknown')
            ))
        
        print(f"\n✅ Processed {len(processed_data)} valid audio clips")
        return processed_data
    
    def create_metadata_file(self, processed_data: List[Tuple[str, str, str]]) -> str:
        """
        Create metadata file in LJSpeech format
        
        Format: wavs/filename|text|text
        
        Returns:
            Path to metadata file
        """
        metadata_path = self.output_dir / "metadata.csv"
        
        print(f"\n📝 Creating metadata file: {metadata_path}")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for wav_path, text, speaker_id in processed_data:
                # LJSpeech format: wavs/filename|text|text
                f.write(f"{wav_path}|{text}|{text}\n")
        
        print(f"✅ Created metadata with {len(processed_data)} entries")
        return str(metadata_path)


class XTTSv2FineTuner:
    """XTTS v2 fine-tuning trainer for Kurdish"""
    
    def __init__(
        self,
        output_model_dir: str = "models/kurdish",
        language: str = "ku"
    ):
        """
        Initialize fine-tuning trainer
        
        Args:
            output_model_dir: Directory to save fine-tuned model
            language: Target language code
        """
        self.output_model_dir = Path(output_model_dir)
        self.output_model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_model_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.language = language
        
        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🖥️  Device: {self.device}")
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {vram:.1f} GB")
            
            if vram < 7.5:
                print(f"   ⚠️  Warning: Low VRAM detected. Training may be slow or fail.")
                print(f"   ℹ️  Consider using smaller batch size or gradient accumulation")
        else:
            print("   ⚠️  No GPU detected. Training on CPU will be very slow.")
    
    def download_base_model(self):
        """Download base XTTS v2 model"""
        try:
            from TTS.utils.manage import ModelManager
            
            print(f"\n🔧 Downloading base XTTS v2 model...")
            manager = ModelManager()
            model_path = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
            print(f"✅ Base model downloaded to: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"❌ Error downloading base model: {e}")
            raise
    
    def train(
        self,
        dataset_path: str,
        metadata_file: str,
        num_epochs: int = 10,
        batch_size: int = 2,
        grad_accum_steps: int = 16,
        learning_rate: float = 5e-6,
        save_step: int = 1000,
        resume: bool = False
    ):
        """
        Fine-tune XTTS v2 model on Kurdish data
        
        Args:
            dataset_path: Path to processed dataset directory
            metadata_file: Path to metadata.csv file
            num_epochs: Number of training epochs
            batch_size: Batch size (keep small for 8GB VRAM)
            grad_accum_steps: Gradient accumulation steps
            learning_rate: Learning rate for fine-tuning
            save_step: Save checkpoint every N steps
            resume: Whether to resume from last checkpoint
        """
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts, XttsArgs
            from TTS.config.shared_configs import BaseDatasetConfig
            from TTS.tts.datasets import load_tts_samples
            from trainer import Trainer, TrainerArgs
            
            print("\n" + "=" * 80)
            print("Starting XTTS v2 Fine-Tuning")
            print("=" * 80)
            
            # Step 1: Download base model
            base_model_path = self.download_base_model()
            
            # Step 2: Configure dataset
            print("\n📊 Configuring dataset...")
            dataset_config = BaseDatasetConfig(
                formatter="ljspeech",
                meta_file_train=metadata_file,
                path=dataset_path,
                language=self.language
            )
            
            # Step 3: Load training samples
            print("📚 Loading training samples...")
            train_samples, eval_samples = load_tts_samples(
                dataset_config,
                eval_split=True,
                eval_split_max_size=0.1,
                eval_split_size=0.1
            )
            
            print(f"   Train samples: {len(train_samples)}")
            print(f"   Eval samples: {len(eval_samples)}")
            
            # Step 4: Configure XTTS model
            print("\n🔧 Configuring XTTS model...")
            config = XttsConfig()
            config.load_json(os.path.join(base_model_path, "config.json"))
            
            # Update config for fine-tuning
            config.batch_size = batch_size
            config.grad_accum_steps = grad_accum_steps
            config.num_epochs = num_epochs
            config.save_step = save_step
            config.print_step = 100
            config.plot_step = 500
            config.log_model_step = 1000
            config.lr = learning_rate
            config.use_grad_scaler = True  # Use mixed precision (fp16)
            
            # Configure for Kurdish
            config.languages = config.languages if hasattr(config, 'languages') else []
            if self.language not in config.languages:
                config.languages.append(self.language)
            
            # Save config
            config_path = self.output_model_dir / "config.json"
            config.save_json(str(config_path))
            print(f"✅ Config saved to: {config_path}")
            
            # Step 5: Initialize model
            print("\n🏗️  Initializing model...")
            model = Xtts.init_from_config(config)
            model.load_checkpoint(config, checkpoint_dir=base_model_path, eval=False)
            
            # Move to device
            if self.device == "cuda":
                model = model.cuda()
            
            # Step 6: Configure trainer
            print("\n🎯 Configuring trainer...")
            trainer_args = TrainerArgs()
            trainer_args.restore_path = None
            trainer_args.skip_train_epoch = False
            trainer_args.use_accelerate = True  # Use accelerate for better performance
            
            # Check for resume
            if resume:
                checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
                if checkpoints:
                    # Sort checkpoints by step number, with error handling
                    valid_checkpoints = []
                    for cp in checkpoints:
                        try:
                            # Extract step number from checkpoint_<step>.pth
                            parts = cp.stem.split('_')
                            if len(parts) >= 2:
                                step_num = int(parts[-1])
                                valid_checkpoints.append((step_num, cp))
                        except (ValueError, IndexError):
                            print(f"   ⚠️  Skipping invalid checkpoint name: {cp.name}")
                            continue
                    
                    if valid_checkpoints:
                        # Get checkpoint with highest step number
                        latest_checkpoint = max(valid_checkpoints, key=lambda x: x[0])[1]
                        trainer_args.restore_path = str(latest_checkpoint)
                        print(f"   ℹ️  Resuming from: {latest_checkpoint}")
                    else:
                        print(f"   ⚠️  No valid checkpoints found to resume from")
                else:
                    print(f"   ⚠️  No checkpoints found to resume from")
            
            # Step 7: Initialize trainer
            trainer = Trainer(
                args=trainer_args,
                config=config,
                output_path=str(self.output_model_dir),
                model=model,
                train_samples=train_samples,
                eval_samples=eval_samples
            )
            
            # Step 8: Start training
            print("\n" + "=" * 80)
            print("🚀 Starting Training...")
            print("=" * 80)
            print(f"   Epochs: {num_epochs}")
            print(f"   Batch size: {batch_size}")
            print(f"   Gradient accumulation: {grad_accum_steps}")
            print(f"   Effective batch size: {batch_size * grad_accum_steps}")
            print(f"   Learning rate: {learning_rate}")
            print(f"   Save every: {save_step} steps")
            print(f"   Device: {self.device}")
            print("=" * 80)
            
            # Create training log
            log_file = self.output_model_dir / "training_log.txt"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Train samples: {len(train_samples)}\n")
                f.write(f"Eval samples: {len(eval_samples)}\n")
                f.write(f"Epochs: {num_epochs}\n")
                f.write(f"Batch size: {batch_size}\n")
                f.write(f"Gradient accumulation: {grad_accum_steps}\n")
                f.write(f"Learning rate: {learning_rate}\n")
                f.write(f"{'=' * 80}\n")
            
            # Run training
            trainer.fit()
            
            # Step 9: Save final model
            print("\n💾 Saving final model...")
            final_model_path = self.output_model_dir / "best_model.pth"
            torch.save(model.state_dict(), final_model_path)
            print(f"✅ Final model saved to: {final_model_path}")
            
            # Save speakers embedding if exists
            if hasattr(model, 'speaker_manager') and model.speaker_manager is not None:
                speakers_path = self.output_model_dir / "speakers.pth"
                torch.save(model.speaker_manager.speaker_embeddings, speakers_path)
                print(f"✅ Speaker embeddings saved to: {speakers_path}")
            
            # Update training log
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Final model: {final_model_path}\n")
            
            print("\n" + "=" * 80)
            print("✅ Fine-Tuning Complete!")
            print("=" * 80)
            
        except ImportError as e:
            print(f"\n❌ Import Error: {e}")
            print("\n⚠️  Required modules not available.")
            print("   This script requires Coqui TTS with Trainer support.")
            print("   Please ensure you have installed:")
            print("   - TTS>=0.27.5")
            print("   - trainer (from Coqui TTS)")
            print("\n   Installation:")
            print("   pip install TTS>=0.27.5")
            raise
        except Exception as e:
            print(f"\n❌ Training Error: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(
        description="Fine-tune XTTS v2 on Kurdish Common Voice data"
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="cv-corpus-24.0-2025-12-05-kmr/cv-corpus-24.0-2025-12-05/kmr/",
        help="Path to Common Voice Kurdish corpus directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_audio",
        help="Directory to save processed audio files"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/kurdish",
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--tsv_file",
        type=str,
        default="validated.tsv",
        help="TSV file to use (validated.tsv, train.tsv, etc.)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of samples to process (0 for all, default: 500 for quick test)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (keep small for 8GB VRAM)"
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps (to compensate for small batch size)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate for fine-tuning"
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    args = parser.parse_args()
    
    # Convert max_samples=0 to None (all samples)
    max_samples = None if args.max_samples == 0 else args.max_samples
    
    print("\n📋 Configuration:")
    print(f"   Corpus path: {args.corpus_path}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Model dir: {args.model_dir}")
    print(f"   TSV file: {args.tsv_file}")
    print(f"   Max samples: {max_samples or 'All'}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.grad_accum_steps}")
    print(f"   Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Save step: {args.save_step}")
    print(f"   Resume: {args.resume}")
    
    # Step 1: Prepare dataset
    print("\n" + "=" * 80)
    print("STEP 1: Data Preparation")
    print("=" * 80)
    
    data_prep = CommonVoiceDataPreparation(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir
    )
    
    # Load TSV
    df = data_prep.load_tsv(args.tsv_file)
    
    # Filter for quality
    df = data_prep.filter_quality_clips(
        df,
        min_upvotes=2,
        max_downvotes=0
    )
    
    # Process audio files
    processed_data = data_prep.prepare_training_data(
        df,
        target_sr=22050,
        max_samples=max_samples
    )
    
    if len(processed_data) == 0:
        print("\n❌ No valid training data found!")
        print("   Please check your corpus path and TSV file.")
        return 1
    
    # Create metadata file
    metadata_file = data_prep.create_metadata_file(processed_data)
    
    # Step 2: Fine-tune model
    print("\n" + "=" * 80)
    print("STEP 2: Fine-Tuning XTTS v2")
    print("=" * 80)
    
    trainer = XTTSv2FineTuner(
        output_model_dir=args.model_dir,
        language="ku"
    )
    
    trainer.train(
        dataset_path=args.output_dir,
        metadata_file=metadata_file,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        save_step=args.save_step,
        resume=args.resume
    )
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ Training Pipeline Complete!")
    print("=" * 80)
    print(f"\n📁 Output locations:")
    print(f"   Processed audio: {args.output_dir}/wavs/")
    print(f"   Metadata: {metadata_file}")
    print(f"   Model directory: {args.model_dir}/")
    print(f"   Best model: {args.model_dir}/best_model.pth")
    print(f"   Config: {args.model_dir}/config.json")
    print(f"   Checkpoints: {args.model_dir}/checkpoints/")
    print(f"   Training log: {args.model_dir}/training_log.txt")
    
    print("\n💡 Next Steps:")
    print("   1. Test the fine-tuned model with tts_stt_service_base44.py")
    print("   2. The service will automatically load the trained model")
    print(f"   3. Run: python tts_stt_service_base44.py")
    
    return 0


if __name__ == "__main__":
    # Windows compatibility: multiprocessing guard
    import multiprocessing
    multiprocessing.freeze_support()
    
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
