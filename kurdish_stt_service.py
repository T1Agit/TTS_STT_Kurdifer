#!/usr/bin/env python3
"""
Kurdish STT (Speech-to-Text) Service

Uses facebook/mms-1b-all model with Kurdish (Kurmanji) language adapter (kmr)
for speech recognition with proper Kurdish character support (ê, î, û, ç, ş).
"""

import io
import os
import tempfile
from typing import Dict, Optional, Any
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, AutoProcessor


class KurdishSTTService:
    """
    Kurdish Speech-to-Text service using MMS-1B-All model
    
    Supports Kurdish (Kurmanji) with proper character handling for:
    ê, î, û, ç, ş
    """
    
    # Model configuration
    MODEL_ID = "facebook/mms-1b-all"
    TARGET_LANG = "kmr"  # Kurdish (Kurmanji)
    SAMPLE_RATE = 16000
    
    def __init__(self):
        """Initialize the Kurdish STT service"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.current_lang = None
        
        print(f"🎙️ Kurdish STT Service initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.MODEL_ID}")
        print(f"   Target language: {self.TARGET_LANG}")
    
    def _load_model(self):
        """
        Lazy load the MMS model and processor with Kurdish adapter
        """
        if self.model is not None and self.current_lang == self.TARGET_LANG:
            return
        
        print(f"📦 Loading {self.MODEL_ID} with {self.TARGET_LANG} adapter...")
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            
            # Load model
            self.model = Wav2Vec2ForCTC.from_pretrained(self.MODEL_ID)
            
            # Load target language adapter
            self.model.load_adapter(self.TARGET_LANG)
            self.model.set_target_lang(self.TARGET_LANG)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.current_lang = self.TARGET_LANG
            
            print(f"✅ Model loaded successfully")
            print(f"   Language adapter: {self.TARGET_LANG}")
            print(f"   Vocab size: {len(self.processor.tokenizer)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _load_audio(self, audio_input) -> torch.Tensor:
        """
        Load audio from various input types
        
        Args:
            audio_input: Can be file path (str), bytes, or BytesIO
            
        Returns:
            Audio waveform tensor normalized to 16kHz
        """
        try:
            if isinstance(audio_input, str):
                # Load from file path
                waveform, sample_rate = torchaudio.load(audio_input)
            elif isinstance(audio_input, bytes):
                # Load from bytes
                audio_buffer = io.BytesIO(audio_input)
                waveform, sample_rate = torchaudio.load(audio_buffer)
            elif isinstance(audio_input, io.BytesIO):
                # Load from BytesIO
                waveform, sample_rate = torchaudio.load(audio_input)
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.SAMPLE_RATE
                )
                waveform = resampler(waveform)
            
            return waveform.squeeze()
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            raise
    
    def transcribe(
        self,
        audio_input,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe Kurdish speech to text
        
        Args:
            audio_input: Audio file path, bytes, or BytesIO object
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with transcribed text and metadata
        """
        try:
            # Load model if not already loaded
            self._load_model()
            
            # Load and preprocess audio
            waveform = self._load_audio(audio_input)
            
            print(f"🎧 Transcribing audio ({len(waveform)} samples)...")
            
            # Process audio
            inputs = self.processor(
                waveform,
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt"
            )
            
            # Move to device
            input_values = inputs.input_values.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            print(f"✅ Transcribed: '{transcription}'")
            
            result = {
                'text': transcription,
                'language': self.TARGET_LANG,
                'sample_rate': self.SAMPLE_RATE,
                'duration': len(waveform) / self.SAMPLE_RATE
            }
            
            # Add confidence scores if requested
            if return_confidence:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence = probs.max(dim=-1).values.mean().item()
                result['confidence'] = confidence
            
            return result
            
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            raise
    
    def transcribe_from_file(
        self,
        file_path: str,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to Kurdish text
        
        Args:
            file_path: Path to audio file (supports mp3, wav, ogg, etc.)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with transcribed text and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        return self.transcribe(file_path, return_confidence=return_confidence)
    
    def transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio bytes to Kurdish text
        
        Args:
            audio_bytes: Audio data as bytes
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with transcribed text and metadata
        """
        return self.transcribe(audio_bytes, return_confidence=return_confidence)
    
    def verify_kurdish_chars(self) -> Dict:
        """
        Verify that Kurdish special characters are supported
        
        Returns:
            Dictionary with character support information
        """
        self._load_model()
        
        # Kurdish special characters to test
        kurdish_chars = ['ê', 'î', 'û', 'ç', 'ş']
        
        results = {}
        vocab = self.processor.tokenizer.get_vocab()
        
        for char in kurdish_chars:
            if char in vocab:
                results[char] = {
                    'supported': True,
                    'token_id': vocab[char]
                }
            else:
                # Try to tokenize and see what happens
                tokens = self.processor.tokenizer.tokenize(char)
                results[char] = {
                    'supported': len(tokens) == 1,
                    'tokens': tokens
                }
        
        return results


def test_kurdish_stt():
    """Test function for Kurdish STT service"""
    print("=" * 70)
    print("Kurdish STT Service Test")
    print("=" * 70)
    
    # Initialize service
    service = KurdishSTTService()
    
    # Test Kurdish character support
    print("\n🔤 Testing Kurdish character support:")
    char_support = service.verify_kurdish_chars()
    for char, info in char_support.items():
        if info['supported']:
            token_id = info.get('token_id', 'N/A')
            print(f"   ✅ '{char}' -> supported (token_id: {token_id})")
        else:
            print(f"   ⚠️  '{char}' -> tokens: {info.get('tokens', [])}")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)
    print("\n📝 Note: To test actual transcription, provide Kurdish audio files:")
    print("   service.transcribe_from_file('path/to/kurdish_audio.mp3')")


if __name__ == "__main__":
    test_kurdish_stt()
