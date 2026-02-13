#!/usr/bin/env python3
"""
VITS Kurdish TTS Inference Service

Uses HuggingFace transformers to load and run fine-tuned VITS models
for Kurdish TTS with support for multiple model versions.
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict
import torch
import torchaudio
from transformers import VitsModel, VitsTokenizer
from pydub import AudioSegment


class VitsTTSService:
    """
    VITS-based TTS service with support for multiple model versions
    
    Supports:
    - Original: facebook/mms-tts-kmr-script_latin (base model)
    - trained_v8: Fine-tuned model from training/best_model_v8/
    """
    
    MODELS = {
        'original': {
            'name': 'facebook/mms-tts-kmr-script_latin',
            'description': 'Base MMS Kurdish TTS model',
            'type': 'huggingface'
        },
        'trained_v8': {
            'path': 'training/best_model_v8',
            'description': 'Fine-tuned Kurdish TTS v8',
            'type': 'local'
        }
    }
    
    def __init__(self, default_model: str = 'original'):
        """
        Initialize VITS TTS service
        
        Args:
            default_model: Default model to use ('original' or 'trained_v8')
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_cache = {}
        self.tokenizers_cache = {}
        self.default_model = default_model
        
        print(f"🎤 VITS TTS Service initialized")
        print(f"   Device: {self.device}")
        print(f"   Default model: {default_model}")
    
    def _get_model_info(self, model_version: str) -> Dict:
        """Get model information"""
        if model_version not in self.MODELS:
            raise ValueError(
                f"Unknown model version: {model_version}. "
                f"Available: {list(self.MODELS.keys())}"
            )
        return self.MODELS[model_version]
    
    def _load_model(self, model_version: str):
        """
        Load model and tokenizer for specified version
        
        Args:
            model_version: Model version to load ('original' or 'trained_v8')
        """
        # Check cache
        if model_version in self.models_cache:
            return self.models_cache[model_version], self.tokenizers_cache[model_version]
        
        model_info = self._get_model_info(model_version)
        
        print(f"📦 Loading {model_version} model...")
        
        try:
            if model_info['type'] == 'huggingface':
                # Load from HuggingFace
                model_name = model_info['name']
                print(f"   Loading from HuggingFace: {model_name}")
                tokenizer = VitsTokenizer.from_pretrained(model_name)
                model = VitsModel.from_pretrained(model_name)
            else:
                # Load from local path
                model_path = model_info['path']
                
                # Check if model exists
                model_dir = Path(model_path)
                if not model_dir.exists():
                    raise FileNotFoundError(
                        f"Model directory not found: {model_path}\n"
                        f"Please ensure the trained model is saved to this location."
                    )
                
                config_path = model_dir / "config.json"
                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Model config not found: {config_path}\n"
                        f"The model directory should contain config.json and pytorch_model.bin"
                    )
                
                print(f"   Loading from local path: {model_path}")
                tokenizer = VitsTokenizer.from_pretrained(str(model_dir))
                model = VitsModel.from_pretrained(str(model_dir))
            
            # Move to device
            model = model.to(self.device)
            model.eval()
            
            # Cache the model
            self.models_cache[model_version] = model
            self.tokenizers_cache[model_version] = tokenizer
            
            print(f"✅ Model {model_version} loaded successfully")
            print(f"   Tokenizer vocab size: {len(tokenizer)}")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading model {model_version}: {e}")
            raise
    
    def list_available_models(self) -> Dict:
        """
        List all available models
        
        Returns:
            Dictionary with model information
        """
        available = {}
        for version, info in self.MODELS.items():
            status = 'available'
            
            # Check if local model exists
            if info['type'] == 'local':
                model_path = Path(info['path'])
                if not model_path.exists() or not (model_path / 'config.json').exists():
                    status = 'not_found'
            
            available[version] = {
                'description': info['description'],
                'type': info['type'],
                'status': status,
                'loaded': version in self.models_cache
            }
        
        return available
    
    def generate_speech(
        self,
        text: str,
        model_version: Optional[str] = None,
        output_format: str = 'mp3'
    ) -> bytes:
        """
        Generate speech from text using specified model
        
        Args:
            text: Kurdish text to synthesize
            model_version: Model version to use (default: self.default_model)
            output_format: Output audio format ('mp3', 'wav')
            
        Returns:
            Audio bytes in specified format
        """
        if model_version is None:
            model_version = self.default_model
        
        print(f"🎵 Generating speech with {model_version} model")
        print(f"   Text: {text[:50]}...")
        
        try:
            # Load model
            model, tokenizer = self._load_model(model_version)
            
            # Tokenize text
            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            
            # Generate speech
            with torch.no_grad():
                outputs = model(input_ids)
                waveform = outputs.waveform.squeeze()
            
            # Move to CPU and convert to numpy
            waveform_cpu = waveform.cpu()
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Save waveform (VITS outputs at 16kHz)
            torchaudio.save(
                temp_path,
                waveform_cpu.unsqueeze(0),
                sample_rate=16000,
                format='wav'
            )
            
            # Read the WAV file
            with open(temp_path, 'rb') as f:
                wav_bytes = f.read()
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Convert to desired format if needed
            if output_format.lower() == 'mp3':
                audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
                output_buffer = io.BytesIO()
                audio.export(output_buffer, format="mp3")
                audio_bytes = output_buffer.getvalue()
            else:
                audio_bytes = wav_bytes
            
            print(f"✅ Speech generated successfully")
            print(f"   Output size: {len(audio_bytes)} bytes")
            print(f"   Format: {output_format}")
            
            return audio_bytes
            
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            raise
    
    def verify_kurdish_chars(self, model_version: Optional[str] = None) -> Dict:
        """
        Verify that Kurdish special characters are supported in tokenizer
        
        Args:
            model_version: Model version to check (default: self.default_model)
            
        Returns:
            Dictionary with character support information
        """
        if model_version is None:
            model_version = self.default_model
        
        model, tokenizer = self._load_model(model_version)
        
        # Kurdish special characters to test
        kurdish_chars = ['ê', 'î', 'û', 'ç', 'ş']
        
        results = {}
        vocab = tokenizer.get_vocab()
        
        for char in kurdish_chars:
            if char in vocab:
                results[char] = {
                    'supported': True,
                    'token_id': vocab[char]
                }
            else:
                # Try to tokenize and see what happens
                tokens = tokenizer.tokenize(char)
                results[char] = {
                    'supported': False,
                    'tokens': tokens
                }
        
        return results


def test_vits_service():
    """Test function for VITS TTS service"""
    print("=" * 70)
    print("VITS TTS Service Test")
    print("=" * 70)
    
    # Initialize service
    service = VitsTTSService(default_model='original')
    
    # List available models
    print("\n📋 Available models:")
    models = service.list_available_models()
    for version, info in models.items():
        status_icon = "✅" if info['status'] == 'available' else "❌"
        print(f"   {status_icon} {version}: {info['description']} ({info['status']})")
    
    # Test Kurdish character support
    print("\n🔤 Testing Kurdish character support:")
    char_support = service.verify_kurdish_chars('original')
    for char, info in char_support.items():
        if info['supported']:
            print(f"   ✅ '{char}' -> token_id: {info['token_id']}")
        else:
            print(f"   ⚠️  '{char}' -> tokens: {info['tokens']}")
    
    # Test speech generation
    print("\n🎵 Testing speech generation:")
    test_texts = [
        "Silav, tu çawa yî?",
        "Rojbûna te pîroz be",
        "Ez te hez dikim"
    ]
    
    for text in test_texts:
        try:
            print(f"\n   Text: {text}")
            audio_bytes = service.generate_speech(text, model_version='original')
            print(f"   Generated: {len(audio_bytes)} bytes")
            
            # Save test output
            output_file = f"test_vits_{text[:10].replace(' ', '_')}.mp3"
            with open(output_file, 'wb') as f:
                f.write(audio_bytes)
            print(f"   Saved to: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_vits_service()
