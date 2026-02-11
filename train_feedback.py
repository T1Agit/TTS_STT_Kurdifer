#!/usr/bin/env python3
"""
Feedback-based Fine-tuning Script for VITS/MMS Kurdish TTS

This script enables incremental fine-tuning based on user feedback from the Base44 app.
Users can record corrections when the model mispronounces words, and this script
will incorporate those corrections into the model.

Expected input structure:
- training/feedback/
  - audio_001.wav (corrected pronunciation)
  - audio_001.txt (correct text transcription)
  - audio_002.wav
  - audio_002.txt
  - ...
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import VitsModel, VitsTokenizer
from tqdm import tqdm


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune VITS/MMS with user feedback")
    parser.add_argument(
        "--feedback_dir",
        type=str,
        default="training/feedback",
        help="Directory containing feedback WAV+TXT pairs (default: training/feedback)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="training/final_model",
        help="Directory with existing fine-tuned model (default: training/final_model)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/feedback_model",
        help="Directory to save updated model (default: training/feedback_model)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="facebook/mms-tts-kmr-script_latin",
        help="Base model if fine-tuned model doesn't exist (default: facebook/mms-tts-kmr-script_latin)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training (default: 2)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5, lower for fine-tuning)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use mixed precision training (default: True)"
    )
    return parser.parse_args()


class FeedbackDataset(Dataset):
    """Dataset for feedback-based training"""
    
    def __init__(self, feedback_dir: Path, tokenizer: VitsTokenizer):
        """
        Initialize feedback dataset
        
        Args:
            feedback_dir: Directory containing WAV+TXT pairs
            tokenizer: VITS tokenizer
        """
        self.feedback_dir = Path(feedback_dir)
        self.tokenizer = tokenizer
        
        # Cache resampler
        self.resampler_cache = {}
        
        # Scan for WAV+TXT pairs
        self.samples = self._scan_feedback_pairs()
        
        if len(self.samples) == 0:
            raise ValueError(f"No feedback pairs found in {feedback_dir}")
        
        print(f"✅ Found {len(self.samples)} feedback samples")
    
    def _scan_feedback_pairs(self) -> List[Tuple[Path, str]]:
        """Scan directory for WAV+TXT pairs"""
        samples = []
        
        if not self.feedback_dir.exists():
            print(f"⚠️  Feedback directory does not exist: {self.feedback_dir}")
            return samples
        
        # Find all WAV files
        wav_files = list(self.feedback_dir.glob("*.wav"))
        
        for wav_path in wav_files:
            # Look for corresponding TXT file
            txt_path = wav_path.with_suffix('.txt')
            
            if txt_path.exists():
                # Read text
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if text:
                    samples.append((wav_path, text))
                else:
                    print(f"⚠️  Empty text file: {txt_path}")
            else:
                print(f"⚠️  Missing text file for: {wav_path}")
        
        return samples
    
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
            "text": text,
            "filename": wav_path.name
        }


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0
) -> torch.Tensor:
    """Compute mel spectrogram from waveform"""
    # Note: This function should be called with a pre-initialized transform
    # for better performance during training
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax
    )
    
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    return mel_spec


class MelSpectrogramComputer:
    """Helper class to compute mel spectrograms with cached transform"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        device: torch.device = None
    ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax
        )
        if device is not None:
            self.mel_transform = self.mel_transform.to(device)
    
    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram from waveform"""
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader"""
    max_text_len = max(item["input_ids"].shape[0] for item in batch)
    max_audio_len = max(item["waveform"].shape[0] for item in batch)
    
    input_ids_list = []
    attention_mask_list = []
    waveforms_list = []
    filenames = []
    
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
        filenames.append(item["filename"])
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "waveforms": torch.stack(waveforms_list),
        "filenames": filenames
    }


def train_on_feedback(
    model: VitsModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    use_fp16: bool = True
):
    """Train model on feedback data"""
    model.train()
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    
    # Create mel spectrogram computer (cached transform)
    mel_computer = MelSpectrogramComputer(device=device)
    
    print(f"\n🚀 Training on feedback for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\n📍 Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            waveforms = batch["waveforms"].to(device)
            
            # Compute mel spectrograms with cached transform
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
            mel_specs = torch.stack(mel_specs_padded).to(device)
            
            # Forward pass
            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=mel_specs
                    )
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=mel_specs
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"✅ Epoch {epoch + 1} complete - Average loss: {avg_loss:.4f}")
    
    return avg_loss


def main():
    """Main function"""
    args = parse_args()
    
    print("=" * 70)
    print("VITS/MMS Feedback-based Fine-tuning")
    print("=" * 70)
    print(f"Feedback directory: {args.feedback_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Load model
    print(f"\n📦 Loading model...")
    model_dir = Path(args.model_dir)
    
    try:
        if model_dir.exists():
            print(f"   Loading fine-tuned model from {model_dir}")
            tokenizer = VitsTokenizer.from_pretrained(str(model_dir))
            model = VitsModel.from_pretrained(str(model_dir))
        else:
            print(f"   Fine-tuned model not found, loading base model: {args.base_model}")
            tokenizer = VitsTokenizer.from_pretrained(args.base_model)
            model = VitsModel.from_pretrained(args.base_model)
        
        model = model.to(device)
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load feedback dataset
    print(f"\n📊 Loading feedback data from {args.feedback_dir}...")
    try:
        dataset = FeedbackDataset(
            feedback_dir=Path(args.feedback_dir),
            tokenizer=tokenizer
        )
    except Exception as e:
        print(f"❌ Error loading feedback data: {e}")
        print("\nTo use this script:")
        print("1. Create directory: training/feedback/")
        print("2. Add WAV+TXT pairs:")
        print("   - audio_001.wav + audio_001.txt")
        print("   - audio_002.wav + audio_002.txt")
        print("   - etc.")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Setup optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Train on feedback
    final_loss = train_on_feedback(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        use_fp16=args.fp16 and device.type == "cuda"
    )
    
    # Save updated model
    print("\n" + "=" * 70)
    print("💾 Saving updated model...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"✅ Model saved to {output_dir}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ Feedback training complete!")
    print("=" * 70)
    print(f"\nUpdated model saved to: {output_dir}")
    print(f"Samples processed: {len(dataset)}")
    print(f"Final loss: {final_loss:.4f}")
    print("\n💡 To continue collecting feedback:")
    print(f"   1. Add more WAV+TXT pairs to {args.feedback_dir}")
    print(f"   2. Run this script again")
    print(f"   3. The model will be further improved!")


if __name__ == "__main__":
    main()
