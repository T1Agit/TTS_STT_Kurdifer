#!/usr/bin/env python3
"""
VITS/MMS Fine-tuning Script for Kurdish TTS - Optimized for 9-10 hours

This script fine-tunes the facebook/mms-tts-kmr-script_latin model on Kurdish data.
Optimized for RTX 2070 8GB VRAM with maximum GPU utilization.

Key optimizations:
- Pre-loads all 42,139 WAVs into RAM (5.21 GB)
- Pre-computes target mel spectrograms
- Uses pinned memory + non_blocking=True for fast CPU→GPU transfer
- Benchmarks speed on 200 samples to measure actual samples/sec
- Auto-calculates epochs to fit within 9.5 hours target
- Shows progress with loss, samples/sec, VRAM, epoch ETA, total ETA
- Saves checkpoints every epoch + best model
- Uses cosine LR schedule with warmup

Base model: facebook/mms-tts-kmr-script_latin (VITS architecture, 36M params)
Dataset: Kurdish Common Voice prepared by prepare_data.py
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import VitsModel, VitsTokenizer, VitsConfig
import numpy as np
from tqdm import tqdm
import soundfile as sf
import librosa


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune VITS/MMS for Kurdish TTS")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="training",
        help="Directory containing wavs/ and metadata.csv (default: training)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="training/checkpoints",
        help="Directory to save checkpoints (default: training/checkpoints)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/final_model",
        help="Directory to save final model (default: training/final_model)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/mms-tts-kmr-script_latin",
        help="Base model to fine-tune (default: facebook/mms-tts-kmr-script_latin)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training (default: 8, increased for better GPU utilization)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--target_hours",
        type=float,
        default=9.5,
        help="Target training time in hours (default: 9.5)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for LR scheduler (default: 500)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use mixed precision training (default: True)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to use (0 = all, default: 0)"
    )
    parser.add_argument(
        "--benchmark_samples",
        type=int,
        default=200,
        help="Number of samples for speed benchmark (default: 200)"
    )
    return parser.parse_args()


class KurdishTTSDataset(Dataset):
    """Dataset for Kurdish TTS training with full RAM pre-loading"""
    
    def __init__(
        self,
        data_dir: Path,
        tokenizer: VitsTokenizer,
        max_samples: int = 0,
        precompute_mels: bool = True,
        device: torch.device = None
    ):
        """
        Initialize dataset with full RAM pre-loading
        
        Args:
            data_dir: Directory containing wavs/ and metadata.csv
            tokenizer: VITS tokenizer
            max_samples: Maximum samples to load (0 = all)
            precompute_mels: Whether to pre-compute mel spectrograms
            device: Device for mel computation (CPU or CUDA)
        """
        self.data_dir = Path(data_dir)
        self.wavs_dir = self.data_dir / "wavs"
        self.tokenizer = tokenizer
        self.precompute_mels = precompute_mels
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Parse metadata (format: filename|text)
        print("📝 Loading metadata...")
        metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|', 1)
                if len(parts) == 2:
                    filename, text = parts
                    wav_path = self.wavs_dir / filename
                    if wav_path.exists():
                        metadata.append((wav_path, text))
        
        # Limit samples if requested
        if max_samples > 0:
            metadata = metadata[:max_samples]
        
        print(f"✅ Found {len(metadata)} samples")
        
        # Pre-load all audio into RAM
        print("🔄 Pre-loading all audio into RAM (using soundfile)...")
        self.samples = []
        total_size_mb = 0
        
        for wav_path, text in tqdm(metadata, desc="Loading audio"):
            try:
                # Load audio using soundfile (NOT torchaudio)
                waveform, sample_rate = sf.read(str(wav_path), dtype='float32')
                
                # Ensure mono
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                
                # Resample to 16kHz if needed using librosa
                if sample_rate != 16000:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
                
                # Convert to torch tensor
                waveform_tensor = torch.from_numpy(waveform).float()
                
                # Tokenize text
                input_ids = self.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
                
                # Calculate size
                total_size_mb += waveform_tensor.numel() * 4 / 1024 / 1024
                
                self.samples.append({
                    "input_ids": input_ids,
                    "waveform": waveform_tensor,
                    "text": text,
                    "filename": wav_path.name
                })
                
            except Exception as e:
                print(f"⚠️  Error loading {wav_path}: {e}")
                continue
        
        print(f"✅ Loaded {len(self.samples)} samples into RAM ({total_size_mb:.2f} MB)")
        
        # Pre-compute mel spectrograms if requested
        if precompute_mels:
            print("🔄 Pre-computing mel spectrograms...")
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=256,
                n_mels=80
            )
            
            if device is not None and device.type == 'cuda':
                mel_transform = mel_transform.to(device)
            
            mel_size_mb = 0
            for i, sample in enumerate(tqdm(self.samples, desc="Computing mels")):
                waveform = sample["waveform"]
                if device is not None and device.type == 'cuda':
                    waveform = waveform.to(device)
                
                mel_spec = mel_transform(waveform)
                mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
                
                # Move back to CPU for storage
                if device is not None and device.type == 'cuda':
                    mel_spec = mel_spec.cpu()
                
                mel_size_mb += mel_spec.numel() * 4 / 1024 / 1024
                sample["mel_spec"] = mel_spec
            
            print(f"✅ Pre-computed mel spectrograms ({mel_size_mb:.2f} MB)")
            total_size_mb += mel_size_mb
        
        print(f"📊 Total RAM usage: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample (already in RAM)"""
        return self.samples[idx]


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80
) -> torch.Tensor:
    """
    Compute mel spectrogram from waveform
    
    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel filterbanks
        
    Returns:
        Mel spectrogram tensor
    """
    # Note: This function should be called with a pre-initialized transform
    # for better performance during training
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    mel_spec = mel_transform(waveform)
    
    # Convert to log scale
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    return mel_spec


class MelSpectrogramComputer:
    """Helper class to compute mel spectrograms with cached transform"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        device: torch.device = None
    ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        if device is not None:
            self.mel_transform = self.mel_transform.to(device)
    
    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from waveform"""
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec


def collate_fn(batch: List[Dict], pin_memory: bool = True) -> Dict:
    """
    Collate function for DataLoader with pinned memory support
    
    Pads sequences to same length in batch
    """
    # Find max lengths
    max_text_len = max(item["input_ids"].shape[0] for item in batch)
    max_audio_len = max(item["waveform"].shape[0] for item in batch)
    
    # Check if mels are pre-computed
    has_mels = "mel_spec" in batch[0]
    if has_mels:
        max_mel_len = max(item["mel_spec"].shape[-1] for item in batch)
    
    # Pad sequences
    input_ids_list = []
    attention_mask_list = []
    waveforms_list = []
    mel_specs_list = [] if has_mels else None
    
    for item in batch:
        # Pad text
        text_len = item["input_ids"].shape[0]
        pad_len = max_text_len - text_len
        input_ids = F.pad(item["input_ids"], (0, pad_len), value=0)
        attention_mask = torch.cat([
            torch.ones(text_len),
            torch.zeros(pad_len)
        ])
        
        # Pad audio
        audio_pad_len = max_audio_len - item["waveform"].shape[0]
        waveform = F.pad(item["waveform"], (0, audio_pad_len), value=0.0)
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        waveforms_list.append(waveform)
        
        # Pad mel if pre-computed
        if has_mels:
            mel_pad_len = max_mel_len - item["mel_spec"].shape[-1]
            mel_spec = F.pad(item["mel_spec"], (0, mel_pad_len), value=0.0)
            mel_specs_list.append(mel_spec)
    
    result = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "waveforms": torch.stack(waveforms_list)
    }
    
    if has_mels:
        result["mel_specs"] = torch.stack(mel_specs_list)
    
    # Pin memory for faster CPU→GPU transfer
    if pin_memory:
        for key in result:
            if isinstance(result[key], torch.Tensor):
                result[key] = result[key].pin_memory()
    
    return result


def benchmark_training_speed(
    model: VitsModel,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 25,
    use_fp16: bool = True
) -> float:
    """
    Benchmark training speed on a subset of data
    
    Args:
        model: VITS model
        dataloader: DataLoader
        device: Device
        num_batches: Number of batches to benchmark
        use_fp16: Whether to use FP16
        
    Returns:
        Samples per second
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    mel_computer = MelSpectrogramComputer(device=device)
    
    print(f"\n⏱️  Benchmarking training speed on {num_batches} batches...")
    
    total_samples = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        batch_size = batch["input_ids"].shape[0]
        total_samples += batch_size
        
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        
        # Use pre-computed mels if available, otherwise compute
        if "mel_specs" in batch:
            mel_specs = batch["mel_specs"].to(device, non_blocking=True)
        else:
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            mel_specs = []
            for waveform in waveforms:
                mel_spec = mel_computer.compute(waveform)
                mel_specs.append(mel_spec)
            
            max_mel_len = max(mel.shape[-1] for mel in mel_specs)
            mel_specs_padded = []
            for mel in mel_specs:
                pad_len = max_mel_len - mel.shape[-1]
                mel_padded = F.pad(mel, (0, pad_len), value=0.0)
                mel_specs_padded.append(mel_padded)
            mel_specs = torch.stack(mel_specs_padded)
        
        # Forward + backward
        if use_fp16:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=mel_specs
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=mel_specs
            )
            loss = outputs.loss
            loss.backward()
        
        # Clear gradients
        model.zero_grad()
    
    elapsed = time.time() - start_time
    samples_per_sec = total_samples / elapsed
    
    print(f"✅ Benchmark complete: {samples_per_sec:.2f} samples/sec")
    print(f"   Total samples: {total_samples}")
    print(f"   Elapsed time: {elapsed:.2f}s")
    
    return samples_per_sec


def train_epoch(
    model: VitsModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    gradient_accumulation_steps: int = 1,
    use_fp16: bool = True
) -> Tuple[float, float]:
    """
    Train for one epoch with detailed progress tracking
    
    Returns:
        Tuple of (average loss, samples per second)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Create GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    
    # Create mel spectrogram computer (cached transform)
    mel_computer = MelSpectrogramComputer(device=device)
    
    # Track timing
    epoch_start_time = time.time()
    batch_times = []
    samples_processed = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        batch_start_time = time.time()
        
        batch_size = batch["input_ids"].shape[0]
        samples_processed += batch_size
        
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        
        # Use pre-computed mels if available, otherwise compute on-the-fly
        if "mel_specs" in batch:
            mel_specs = batch["mel_specs"].to(device, non_blocking=True)
        else:
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            mel_specs = []
            for waveform in waveforms:
                mel_spec = mel_computer.compute(waveform)
                mel_specs.append(mel_spec)
            
            # Stack mel spectrograms (pad to same length)
            max_mel_len = max(mel.shape[-1] for mel in mel_specs)
            mel_specs_padded = []
            for mel in mel_specs:
                pad_len = max_mel_len - mel.shape[-1]
                mel_padded = F.pad(mel, (0, pad_len), value=0.0)
                mel_specs_padded.append(mel_padded)
            mel_specs = torch.stack(mel_specs_padded)
        
        # Forward pass with mixed precision
        if use_fp16:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=mel_specs
                )
                loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=mel_specs
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
        
        # Update weights after accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Track batch time
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        samples_per_sec = samples_processed / (time.time() - epoch_start_time)
        
        # VRAM usage
        if device.type == "cuda":
            vram_used_gb = torch.cuda.memory_allocated(device) / 1e9
            vram_total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        else:
            vram_used_gb = 0
            vram_total_gb = 0
        
        # Estimate epoch ETA
        batches_remaining = len(dataloader) - batch_idx - 1
        if len(batch_times) > 10:
            avg_batch_time = sum(batch_times[-10:]) / len(batch_times[-10:])
        else:
            avg_batch_time = sum(batch_times) / len(batch_times)
        epoch_eta_seconds = batches_remaining * avg_batch_time
        epoch_eta_minutes = epoch_eta_seconds / 60
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
            "samples/s": f"{samples_per_sec:.1f}",
            "VRAM": f"{vram_used_gb:.2f}/{vram_total_gb:.2f}GB",
            "ETA": f"{epoch_eta_minutes:.1f}m"
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    samples_per_sec = samples_processed / (time.time() - epoch_start_time)
    
    return avg_loss, samples_per_sec


def save_checkpoint(
    model: VitsModel,
    tokenizer: VitsTokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path
):
    """Save training checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, checkpoint_path)
    
    print(f"💾 Saved checkpoint to {checkpoint_path}")


def save_final_model(
    model: VitsModel,
    tokenizer: VitsTokenizer,
    output_dir: Path
):
    """Save final trained model"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"💾 Saved final model to {output_dir}")


def main():
    """Main training function"""
    args = parse_args()
    
    print("=" * 70)
    print("VITS/MMS Kurdish TTS Fine-tuning - OPTIMIZED")
    print("=" * 70)
    print(f"Base model: {args.model_name}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Target training time: {args.target_hours} hours")
    print(f"Mixed precision (FP16): {args.fp16}")
    print("=" * 70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load tokenizer and model
    print(f"\n📦 Loading model and tokenizer...")
    try:
        tokenizer = VitsTokenizer.from_pretrained(args.model_name)
        model = VitsModel.from_pretrained(args.model_name)
        model = model.to(device)
        print("✅ Model loaded successfully")
        
        # Print model size
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Parameters: {num_params / 1e6:.2f}M")
        print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Load dataset with pre-loading
    print(f"\n📊 Loading dataset from {args.data_dir}...")
    try:
        dataset = KurdishTTSDataset(
            data_dir=Path(args.data_dir),
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            precompute_mels=True,  # Pre-compute mel spectrograms
            device=device if device.type == "cuda" else None
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise
    
    # Create dataloader with pinned memory
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pin_memory=True),
        num_workers=0,  # Use 0 for Windows compatibility
        pin_memory=True  # Enable pinned memory for faster transfers
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Benchmark training speed
    benchmark_batches = min(args.benchmark_samples // args.batch_size, len(dataloader))
    samples_per_sec = benchmark_training_speed(
        model=model,
        dataloader=dataloader,
        device=device,
        num_batches=benchmark_batches,
        use_fp16=args.fp16 and device.type == "cuda"
    )
    
    # Auto-calculate number of epochs to fit target training time
    total_samples = len(dataset)
    samples_per_epoch = total_samples
    seconds_per_epoch = samples_per_epoch / samples_per_sec
    target_seconds = args.target_hours * 3600
    auto_epochs = max(1, int(target_seconds / seconds_per_epoch))
    
    print(f"\n📊 TRAINING PLAN")
    print("=" * 70)
    print(f"Total samples: {total_samples:,}")
    print(f"Samples per epoch: {samples_per_epoch:,}")
    print(f"Benchmark speed: {samples_per_sec:.2f} samples/sec")
    print(f"Estimated time per epoch: {seconds_per_epoch / 60:.1f} minutes")
    print(f"Target training time: {args.target_hours} hours ({target_seconds / 60:.1f} minutes)")
    print(f"Auto-calculated epochs: {auto_epochs}")
    print(f"Estimated total time: {auto_epochs * seconds_per_epoch / 3600:.2f} hours")
    print("=" * 70)
    
    # Setup learning rate scheduler with warmup and cosine decay
    total_steps = len(dataloader) * auto_epochs // args.gradient_accumulation_steps
    warmup_steps = args.warmup_steps
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\n📈 Learning rate schedule:")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Schedule: Linear warmup + Cosine decay")
    
    # Training loop
    print(f"\n🚀 Starting training for {auto_epochs} epochs...")
    print("=" * 70)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    best_model_path = None
    training_start_time = time.time()
    
    for epoch in range(auto_epochs):
        print(f"\n📍 Epoch {epoch + 1}/{auto_epochs}")
        
        # Train for one epoch
        avg_loss, epoch_samples_per_sec = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch + 1,
            total_epochs=auto_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_fp16=args.fp16 and device.type == "cuda"
        )
        
        # Calculate total ETA
        elapsed_time = time.time() - training_start_time
        epochs_remaining = auto_epochs - epoch - 1
        estimated_time_per_epoch = elapsed_time / (epoch + 1)
        total_eta_seconds = epochs_remaining * estimated_time_per_epoch
        total_eta_hours = total_eta_seconds / 3600
        
        print(f"✅ Epoch {epoch + 1} complete")
        print(f"   Average loss: {avg_loss:.4f}")
        print(f"   Samples/sec: {epoch_samples_per_sec:.2f}")
        print(f"   Elapsed time: {elapsed_time / 3600:.2f} hours")
        print(f"   Total ETA: {total_eta_hours:.2f} hours")
        
        # Save checkpoint every epoch
        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=avg_loss,
            checkpoint_dir=checkpoint_dir
        )
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_dir = checkpoint_dir / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_model_dir))
            tokenizer.save_pretrained(str(best_model_dir))
            best_model_path = best_model_dir
            print(f"💎 New best model saved! Loss: {best_loss:.4f}")
    
    # Save final model
    print("\n" + "=" * 70)
    print("💾 Saving final model...")
    output_dir = Path(args.output_dir)
    save_final_model(model, tokenizer, output_dir)
    
    total_training_time = time.time() - training_start_time
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)
    print(f"\n📊 TRAINING SUMMARY")
    print(f"Total training time: {total_training_time / 3600:.2f} hours")
    print(f"Total epochs: {auto_epochs}")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"\n💾 SAVED MODELS")
    print(f"Final model: {output_dir}")
    print(f"Best model: {best_model_path}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
