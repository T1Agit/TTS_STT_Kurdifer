#!/usr/bin/env python3
"""
VITS/MMS Fine-tuning Script for Kurdish TTS

This script fine-tunes the facebook/mms-tts-kmr-script_latin model on Kurdish data.
Optimized for RTX 2070 8GB VRAM using gradient accumulation and mixed precision.

Base model: facebook/mms-tts-kmr-script_latin (VITS architecture, 36M params)
Dataset: Kurdish Common Voice prepared by prepare_data.py
"""

import os
import argparse
import json
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
        default=4,
        help="Batch size for training (default: 4 for 8GB VRAM)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=3,
        help="Minimum number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate (default: 500)"
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        default=False,
        help="Use cosine annealing LR scheduler after warmup (default: False, use --use_scheduler to enable)"
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
    return parser.parse_args()


class KurdishTTSDataset(Dataset):
    """Dataset for Kurdish TTS training"""
    
    def __init__(
        self,
        data_dir: Path,
        tokenizer: VitsTokenizer,
        max_samples: int = 0
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing wavs/ and metadata.csv
            tokenizer: VITS tokenizer
            max_samples: Maximum samples to load (0 = all)
        """
        self.data_dir = Path(data_dir)
        self.wavs_dir = self.data_dir / "wavs"
        self.tokenizer = tokenizer
        
        # Cache resampler (assuming all audio is 16kHz already from preprocessing)
        self.resampler_cache = {}
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Parse metadata (format: filename|text)
        self.samples = []
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
                        self.samples.append((wav_path, text))
        
        # Limit samples if requested
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        
        print(f"✅ Loaded {len(self.samples)} samples from {metadata_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        wav_path, text = self.samples[idx]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(str(wav_path))
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed (with caching)
        if sample_rate != 16000:
            if sample_rate not in self.resampler_cache:
                self.resampler_cache[sample_rate] = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = self.resampler_cache[sample_rate](waveform)
        
        # Tokenize text
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        
        return {
            "input_ids": input_ids,
            "waveform": waveform.squeeze(0),
            "text": text
        }


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


class MultiScaleMelLoss(nn.Module):
    """Multi-scale mel spectrogram loss for better gradient flow"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        device: torch.device = None
    ):
        super().__init__()
        
        # Create mel transforms at different scales (different FFT sizes)
        # Smaller FFT -> better time resolution
        # Larger FFT -> better frequency resolution
        self.scales = [
            # (n_fft, hop_length, weight)
            (512, 128, 0.25),    # Fine scale - good for transients
            (1024, 256, 0.5),    # Medium scale - balanced (original)
            (2048, 512, 0.25),   # Coarse scale - good for overall structure
        ]
        
        self.mel_computers = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            for n_fft, hop_length, _ in self.scales
        ])
        
        if device is not None:
            self.to(device)
    
    def forward(
        self,
        predicted_waveform: torch.Tensor,
        target_waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-scale mel loss
        
        Args:
            predicted_waveform: Predicted audio waveform from model
            target_waveform: Target audio waveform
            
        Returns:
            Combined loss across all scales
        """
        total_loss = 0.0
        
        for mel_transform, (n_fft, hop_length, weight) in zip(self.mel_computers, self.scales):
            # Compute mels at this scale for both predicted and target
            predicted_mel = mel_transform(predicted_waveform)
            predicted_mel = torch.log(torch.clamp(predicted_mel, min=1e-5))
            
            target_mel = mel_transform(target_waveform)
            target_mel = torch.log(torch.clamp(target_mel, min=1e-5))
            
            # Ensure both have same temporal dimension
            if predicted_mel.shape[-1] != target_mel.shape[-1]:
                # Resize to match the smaller dimension
                min_len = min(predicted_mel.shape[-1], target_mel.shape[-1])
                predicted_mel = predicted_mel[..., :min_len]
                target_mel = target_mel[..., :min_len]
            
            # Compute L1 loss at this scale
            loss = F.l1_loss(predicted_mel, target_mel)
            total_loss += weight * loss
        
        return total_loss


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader
    
    Pads sequences to same length in batch
    """
    # Find max lengths
    max_text_len = max(item["input_ids"].shape[0] for item in batch)
    max_audio_len = max(item["waveform"].shape[0] for item in batch)
    
    # Pad sequences
    input_ids_list = []
    attention_mask_list = []
    waveforms_list = []
    
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
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "waveforms": torch.stack(waveforms_list)
    }


def train_epoch(
    model: VitsModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    use_fp16: bool = True,
    use_multiscale_loss: bool = True,
    epoch_num: int = 1
) -> float:
    """
    Train for one epoch
    
    Args:
        model: VITS model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (can be None)
        device: Device to train on
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_fp16: Use mixed precision training
        use_multiscale_loss: Use multi-scale mel loss instead of single-scale
        epoch_num: Current epoch number (for logging)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Create GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    
    # Create mel loss computer
    if use_multiscale_loss:
        mel_loss_fn = MultiScaleMelLoss(device=device)
    else:
        mel_computer = MelSpectrogramComputer(device=device)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        waveforms = batch["waveforms"].to(device)
        
        # Forward pass with mixed precision
        if use_fp16:
            with torch.cuda.amp.autocast():
                # Generate mel spectrogram predictions
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # Get predicted waveform or mel
                if hasattr(outputs, 'waveform'):
                    # If model outputs waveform, convert to mel
                    predicted_waveform = outputs.waveform
                    if use_multiscale_loss:
                        # Multi-scale loss computes mels internally from waveform
                        loss = mel_loss_fn(predicted_waveform, waveforms)
                    else:
                        # Compute mel from predicted waveform
                        predicted_mels = []
                        target_mels = []
                        for pred_wav, target_wav in zip(predicted_waveform, waveforms):
                            pred_mel = mel_computer.compute(pred_wav)
                            target_mel = mel_computer.compute(target_wav)
                            predicted_mels.append(pred_mel)
                            target_mels.append(target_mel)
                        
                        # Pad and stack
                        max_len = max(max(m.shape[-1] for m in predicted_mels),
                                     max(m.shape[-1] for m in target_mels))
                        
                        predicted_mels = torch.stack([
                            F.pad(m, (0, max_len - m.shape[-1])) for m in predicted_mels
                        ])
                        target_mels = torch.stack([
                            F.pad(m, (0, max_len - m.shape[-1])) for m in target_mels
                        ])
                        
                        loss = F.l1_loss(predicted_mels, target_mels)
                else:
                    # Fallback: use model's built-in loss if available
                    # Compute target mels
                    target_mels = []
                    for waveform in waveforms:
                        mel = mel_computer.compute(waveform)
                        target_mels.append(mel)
                    
                    # Pad target mels
                    max_mel_len = max(mel.shape[-1] for mel in target_mels)
                    target_mels_padded = []
                    for mel in target_mels:
                        pad_len = max_mel_len - mel.shape[-1]
                        mel_padded = F.pad(mel, (0, pad_len), value=0.0)
                        target_mels_padded.append(mel_padded)
                    target_mels = torch.stack(target_mels_padded).to(device)
                    
                    # Use model's forward pass with labels
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=target_mels
                    )
                    loss = outputs.loss
                
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
        else:
            # Same logic without mixed precision
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            if hasattr(outputs, 'waveform'):
                predicted_waveform = outputs.waveform
                if use_multiscale_loss:
                    loss = mel_loss_fn(predicted_waveform, waveforms)
                else:
                    predicted_mels = []
                    target_mels = []
                    for pred_wav, target_wav in zip(predicted_waveform, waveforms):
                        pred_mel = mel_computer.compute(pred_wav)
                        target_mel = mel_computer.compute(target_wav)
                        predicted_mels.append(pred_mel)
                        target_mels.append(target_mel)
                    
                    max_len = max(max(m.shape[-1] for m in predicted_mels),
                                 max(m.shape[-1] for m in target_mels))
                    
                    predicted_mels = torch.stack([
                        F.pad(m, (0, max_len - m.shape[-1])) for m in predicted_mels
                    ])
                    target_mels = torch.stack([
                        F.pad(m, (0, max_len - m.shape[-1])) for m in target_mels
                    ])
                    
                    loss = F.l1_loss(predicted_mels, target_mels)
            else:
                target_mels = []
                for waveform in waveforms:
                    mel = mel_computer.compute(waveform)
                    target_mels.append(mel)
                
                max_mel_len = max(mel.shape[-1] for mel in target_mels)
                target_mels_padded = []
                for mel in target_mels:
                    pad_len = max_mel_len - mel.shape[-1]
                    mel_padded = F.pad(mel, (0, pad_len), value=0.0)
                    target_mels_padded.append(mel_padded)
                target_mels = torch.stack(target_mels_padded).to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target_mels
                )
                loss = outputs.loss
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
        
        # Update weights after accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # Step scheduler after each optimizer step
            if scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Update progress bar with current LR
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
            "lr": f"{current_lr:.2e}"
        })
    
    return total_loss / num_batches if num_batches > 0 else 0.0


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
    print("VITS/MMS Kurdish TTS Fine-tuning")
    print("=" * 70)
    print(f"Base model: {args.model_name}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs} (minimum: {args.min_epochs})")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Use scheduler: {args.use_scheduler}")
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
        
        # CRITICAL: Ensure ALL parameters require gradients
        print("\n🔧 Ensuring all model parameters require gradients...")
        params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        for param in model.parameters():
            param.requires_grad = True
        
        params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("✅ Model loaded successfully")
        
        # Print model size
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {num_params / 1e6:.2f}M")
        print(f"   Trainable (initially loaded): {params_before / 1e6:.2f}M")
        print(f"   Trainable (after unfreezing): {params_after / 1e6:.2f}M")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load dataset
    print(f"\n📊 Loading dataset from {args.data_dir}...")
    try:
        dataset = KurdishTTSDataset(
            data_dir=Path(args.data_dir),
            tokenizer=tokenizer,
            max_samples=args.max_samples
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Calculate total training steps for scheduler
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    # Enforce minimum epochs
    actual_epochs = max(args.epochs, args.min_epochs)
    total_steps = steps_per_epoch * actual_epochs
    
    print(f"\n📊 Training configuration:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Requested epochs: {args.epochs}")
    print(f"   Actual epochs (enforced minimum): {actual_epochs}")
    print(f"   Total training steps: {total_steps}")
    
    # Setup learning rate scheduler with warmup
    scheduler = None
    if args.use_scheduler:
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step: int):
            """
            Learning rate schedule with warmup and cosine decay
            """
            if current_step < args.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, args.warmup_steps))
            else:
                # Cosine annealing after warmup with minimum LR of 10% of max
                progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
                return 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))  # Decays from 1.0 to 0.1
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"✅ Learning rate scheduler configured (warmup + cosine annealing)")
        print(f"   Warmup steps: {args.warmup_steps}")
        print(f"   Max LR: {args.learning_rate:.2e}")
        print(f"   Min LR: {args.learning_rate * 0.1:.2e} (10% of max)")
    
    # Training loop
    print(f"\n🚀 Starting training for {actual_epochs} epochs...")
    print("=" * 70)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    for epoch in range(actual_epochs):
        print(f"\n📍 Epoch {epoch + 1}/{actual_epochs}")
        
        # Train for one epoch
        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_fp16=args.fp16 and device.type == "cuda",
            use_multiscale_loss=True,  # Enable multi-scale mel loss
            epoch_num=epoch + 1
        )
        
        print(f"✅ Epoch {epoch + 1} complete - Average loss: {avg_loss:.4f}")
        
        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss,
                checkpoint_dir=checkpoint_dir
            )
    
    # Save final model
    print("\n" + "=" * 70)
    print("💾 Saving final model...")
    output_dir = Path(args.output_dir)
    save_final_model(model, tokenizer, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)
    print(f"\nFinal model saved to: {output_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
