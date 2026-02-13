"""
TTS/STT Service with Base44 Encoding

Provides Text-to-Speech and Speech-to-Text functionality with Base44 encoding
for Kurdish, German, French, English, and Turkish languages.
"""

import io
import os
import tempfile
from typing import Dict
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr
from base44 import encode, decode


class TTSSTTServiceBase44:
    """
    TTS/STT Service with Base44 encoding support
    
    Supports: Kurdish (ku), German (de), French (fr), English (en), Turkish (tr)
    """
    
    SUPPORTED_LANGUAGES = {
        'kurdish': 'ku', 'german': 'de', 'french': 'fr',
        'english': 'en', 'turkish': 'tr',
        'ku': 'ku', 'de': 'de', 'fr': 'fr', 'en': 'en', 'tr': 'tr'
    }
    
    def __init__(self):
        """Initialize the TTS/STT service"""
        self.recognizer = sr.Recognizer()
        self._coqui_tts = None  # Lazy initialization for Coqui TTS
        self._fine_tuned_model_path = None  # Path to fine-tuned Kurdish model
        self._use_fine_tuned = False  # Flag to indicate if fine-tuned model is loaded
        self._vits_service = None  # Lazy initialization for VITS TTS
        self._check_fine_tuned_model()
    
    def _get_language_code(self, language: str) -> str:
        """
        Normalize language input to language code
        
        Args:
            language: Language name or code
            
        Returns:
            Language code (e.g., 'en', 'ku', 'de')
        """
        lang_lower = language.lower().strip()
        if lang_lower not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {', '.join(set(self.SUPPORTED_LANGUAGES.values()))}"
            )
        return self.SUPPORTED_LANGUAGES[lang_lower]
    
    def _check_fine_tuned_model(self):
        """Check if a fine-tuned Kurdish model exists"""
        model_dirs = [
            "models/kurdish",
            "./models/kurdish",
            os.path.join(os.path.dirname(__file__), "models", "kurdish")
        ]
        
        for model_dir in model_dirs:
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                self._fine_tuned_model_path = model_dir
                print(f"✅ Found fine-tuned Kurdish model at: {model_dir}")
                return
        
        self._fine_tuned_model_path = None
    
    def _uses_coqui_tts(self, lang_code: str) -> bool:
        """
        Check if language requires Coqui TTS
        
        Args:
            lang_code: Language code (e.g., 'ku', 'en')
            
        Returns:
            True if language needs Coqui TTS, False otherwise
        """
        return lang_code == 'ku'
    
    def _get_vits_service(self):
        """Lazy initialization of VITS TTS service"""
        if self._vits_service is None:
            try:
                from vits_tts_service import VitsTTSService
                self._vits_service = VitsTTSService(default_model='original')
                print("✅ VITS TTS service initialized")
            except ImportError as e:
                print(f"⚠️  VITS TTS not available: {e}")
                self._vits_service = None
        return self._vits_service
    
    def _generate_speech_vits(self, text: str, model_version: str = 'original', voice_preset: str = 'default') -> bytes:
        """
        Generate speech using VITS TTS for Kurdish
        
        Args:
            text: Text to convert to speech
            model_version: VITS model version ('original' or 'trained_v8')
            voice_preset: Voice preset ('default', 'elderly_male', 'elderly_female')
            
        Returns:
            Audio bytes in MP3 format
        """
        try:
            vits_service = self._get_vits_service()
            if vits_service is None:
                raise ImportError("VITS TTS service not available")
            
            # Generate speech using VITS
            audio_bytes = vits_service.generate_speech(
                text=text,
                model_version=model_version,
                output_format='mp3',
                voice_preset=voice_preset
            )
            
            return audio_bytes
            
        except Exception as e:
            print(f"❌ VITS TTS error: {e}")
            raise RuntimeError(f"VITS TTS generation failed: {e}") from e
    
    def _generate_speech_coqui(self, text: str, lang_code: str) -> bytes:
        """
        Generate speech using Coqui TTS for Kurdish
        
        If a fine-tuned Kurdish model exists, uses it with language='ku'.
        Otherwise, falls back to voice cloning with Turkish phonetics as proxy.
        
        Args:
            text: Text to convert to speech
            lang_code: Language code (should be 'ku')
            
        Returns:
            Audio bytes in MP3 format
        """
        try:
            from TTS.api import TTS
            
            # Lazy initialization of Coqui TTS
            if self._coqui_tts is None:
                print("🔧 Initializing Coqui TTS for Kurdish...")
                
                # Check if we have a fine-tuned model
                if self._fine_tuned_model_path:
                    # Check for required model files
                    model_path = os.path.join(self._fine_tuned_model_path, "best_model.pth")
                    config_path = os.path.join(self._fine_tuned_model_path, "config.json")
                    
                    if os.path.exists(model_path) and os.path.exists(config_path):
                        print(f"   ✅ Loading fine-tuned Kurdish model from: {self._fine_tuned_model_path}")
                        try:
                            # Load fine-tuned model
                            self._coqui_tts = TTS(
                                model_path=config_path,
                                progress_bar=False
                            )
                            self._coqui_tts.load_checkpoint(
                                config_path=config_path,
                                checkpoint_path=model_path
                            )
                            self._use_fine_tuned = True
                            print("   ✅ Fine-tuned Kurdish model loaded successfully")
                            print("   ℹ️  Using language='ku' for Kurdish TTS")
                        except Exception as e:
                            print(f"   ⚠️  Error loading fine-tuned model: {e}")
                            print("   ℹ️  Falling back to base model with Turkish phonetics")
                            self._use_fine_tuned = False
                            self._coqui_tts = TTS(
                                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                                progress_bar=False
                            )
                    else:
                        print(f"   ⚠️  Fine-tuned model files not complete at: {self._fine_tuned_model_path}")
                        print("   ℹ️  Falling back to base model with Turkish phonetics")
                        self._use_fine_tuned = False
                        self._coqui_tts = TTS(
                            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                            progress_bar=False
                        )
                else:
                    # No fine-tuned model, use base model
                    print("   ℹ️  No fine-tuned model found, using base XTTS v2")
                    print("   ℹ️  Using Turkish phonetics as proxy for Kurdish")
                    self._use_fine_tuned = False
                    self._coqui_tts = TTS(
                        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                        progress_bar=False
                    )
                
                if self._use_fine_tuned:
                    print("✅ Coqui TTS initialized (fine-tuned Kurdish mode)")
                else:
                    print("✅ Coqui TTS initialized (voice cloning mode)")
            
            # Generate speech to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Generate audio
            if self._use_fine_tuned:
                # Use fine-tuned model with Kurdish language code
                print("   Generating with fine-tuned Kurdish model (language='ku')...")
                self._coqui_tts.tts_to_file(
                    text=text,
                    file_path=temp_path,
                    language="ku"
                )
            else:
                # Fallback to voice cloning with Turkish phonetics
                print("   Using Turkish phonetics as proxy for Kurdish...")
                self._generate_with_voice_cloning(text, temp_path)
            
            # Read the generated file
            with open(temp_path, 'rb') as f:
                wav_bytes = f.read()
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Convert WAV to MP3 using pydub
            audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format="mp3")
            
            return output_buffer.getvalue()
            
        except ImportError as e:
            raise ImportError(
                "Coqui TTS is not installed. Please run: pip install TTS\n"
                "Or run: python setup_kurdish_tts.py"
            ) from e
        except Exception as e:
            print(f"❌ Coqui TTS error: {e}")
            raise RuntimeError(f"Coqui TTS generation failed: {e}") from e
    
    def _generate_with_voice_cloning(self, text: str, output_path: str):
        """
        Generate speech using Turkish as phonetic proxy for Kurdish
        
        Args:
            text: Kurdish text to synthesize
            output_path: Path to save WAV file
        """
        # Use Turkish language with XTTS v2 as phonetic proxy
        # Turkish and Kurdish (Kurmanji) share similar phonology
        try:
            self._coqui_tts.tts_to_file(
                text=text,
                file_path=output_path,
                language="tr"  # Turkish as phonetic proxy for Kurdish
            )
        except Exception as e:
            # If even Turkish fails, try English as last resort
            print(f"   ⚠️  Turkish fallback failed: {e}")
            print("   Using English as last resort...")
            self._coqui_tts.tts_to_file(
                text=text,
                file_path=output_path,
                language="en"
            )
    
    def text_to_speech_base44(
        self,
        text: str,
        language: str = 'en',
        audio_format: str = 'mp3',
        model_version: str = None,
        use_vits: bool = True,
        voice_preset: str = 'default'
    ) -> Dict[str, str]:
        """
        Convert text to speech and encode to Base44
        
        Args:
            text: Text to convert to speech
            language: Target language (e.g., 'en', 'english', 'ku')
            audio_format: Audio format ('mp3', 'wav', 'ogg')
            model_version: Model version for Kurdish VITS ('original', 'trained_v8')
            use_vits: Use VITS for Kurdish instead of Coqui TTS (default: True)
            voice_preset: Voice preset ('default', 'elderly_male', 'elderly_female')
            
        Returns:
            Dictionary with audio data and metadata
        """
        try:
            # Get language code
            lang_code = self._get_language_code(language)
            
            print(f"🎤 Generating speech: '{text[:50]}...' in {lang_code}")
            
            # Check if we need to use Kurdish TTS
            if self._uses_coqui_tts(lang_code):
                # Try VITS first if enabled
                if use_vits:
                    try:
                        print(f"   Using VITS TTS (model: {model_version or 'original'}, preset: {voice_preset})")
                        audio_bytes = self._generate_speech_vits(
                            text,
                            model_version=model_version or 'original',
                            voice_preset=voice_preset
                        )
                        
                        # Encode to Base44
                        audio_base44 = encode(audio_bytes)
                        
                        print(f"✅ Success! Size: {len(audio_bytes)} bytes → {len(audio_base44)} chars")
                        
                        return {
                            'audio': audio_base44,
                            'language': lang_code,
                            'format': audio_format,
                            'text': text,
                            'size': len(audio_bytes),
                            'encoded_size': len(audio_base44),
                            'compression_ratio': len(audio_base44) / len(audio_bytes),
                            'model': model_version or 'original',
                            'engine': 'vits'
                        }
                    except Exception as e:
                        print(f"⚠️  VITS TTS failed: {e}")
                        print("   Falling back to Coqui TTS...")
                        # Fall through to Coqui TTS
                
                # Generate speech using Coqui TTS
                audio_bytes = self._generate_speech_coqui(text, lang_code)
                
                # Encode to Base44
                audio_base44 = encode(audio_bytes)
                
                print(f"✅ Success! Size: {len(audio_bytes)} bytes → {len(audio_base44)} chars")
                
                return {
                    'audio': audio_base44,
                    'language': lang_code,
                    'format': audio_format,
                    'text': text,
                    'size': len(audio_bytes),
                    'encoded_size': len(audio_base44),
                    'compression_ratio': len(audio_base44) / len(audio_bytes),
                    'engine': 'coqui'
                }
            
            # Generate speech using gTTS for other languages
            tts = gTTS(text=text, lang=lang_code)
            
            # Save to bytes buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Load with pydub for format conversion
            audio = AudioSegment.from_file(audio_buffer, format="mp3")
            
            # Convert to desired format
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format=audio_format)
            audio_bytes = output_buffer.getvalue()
            
            # Encode to Base44
            audio_base44 = encode(audio_bytes)
            
            print(f"✅ Success! Size: {len(audio_bytes)} bytes → {len(audio_base44)} chars")
            
            return {
                'audio': audio_base44,
                'language': lang_code,
                'format': audio_format,
                'text': text,
                'size': len(audio_bytes),
                'encoded_size': len(audio_base44),
                'compression_ratio': len(audio_base44) / len(audio_bytes)
            }
            
        except Exception as e:
            print(f"❌ Error in text_to_speech_base44: {e}")
            raise
    
    def speech_to_text_from_base44(
        self,
        audio_base44: str,
        language: str = 'en',
        source_format: str = 'mp3'
    ) -> Dict[str, str]:
        """
        Decode Base44 audio and convert to text
        
        Args:
            audio_base44: Base44 encoded audio
            language: Source language for recognition
            source_format: Audio format of source
            
        Returns:
            Dictionary with transcribed text and metadata
        """
        try:
            # Get language code
            lang_code = self._get_language_code(language)
            
            print(f"🎧 Decoding audio ({len(audio_base44)} chars) for {lang_code}...")
            
            # Decode Base44 to bytes
            audio_bytes = decode(audio_base44)
            
            # Load audio with pydub
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format=source_format
            )
            
            # Convert to WAV for speech recognition
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            # Recognize speech
            with sr.AudioFile(wav_buffer) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language=lang_code)
            
            print(f"✅ Transcribed: '{text}'")
            
            return {
                'text': text,
                'language': lang_code,
                'confidence': 1.0,  # Google API doesn't provide confidence
                'audio_size': len(audio_bytes)
            }
            
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            raise ValueError("Could not understand audio")
        except sr.RequestError as e:
            print(f"❌ API error: {e}")
            raise
        except Exception as e:
            print(f"❌ Error in speech_to_text_from_base44: {e}")
            raise
    
    def speech_to_text_from_file(
        self,
        audio_file_path: str,
        language: str = 'en'
    ) -> Dict[str, str]:
        """
        Read audio file and convert to text
        
        Args:
            audio_file_path: Path to audio file
            language: Source language for recognition
            
        Returns:
            Dictionary with transcribed text and metadata
        """
        try:
            print(f"📁 Reading audio file: {audio_file_path}")
            
            # Read audio file
            with open(audio_file_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Encode to Base44
            audio_base44 = encode(audio_bytes)
            
            # Detect format from file extension
            _, ext = os.path.splitext(audio_file_path)
            source_format = ext[1:] if ext else 'mp3'
            
            # Call speech_to_text_from_base44
            return self.speech_to_text_from_base44(
                audio_base44,
                language,
                source_format
            )
            
        except Exception as e:
            print(f"❌ Error in speech_to_text_from_file: {e}")
            raise
    
    def save_audio_from_base44(
        self,
        audio_base44: str,
        output_path: str
    ):
        """
        Decode Base44 audio and save to file
        
        Args:
            audio_base44: Base44 encoded audio
            output_path: Path to save audio file
        """
        try:
            print(f"💾 Saving audio to: {output_path}")
            
            # Decode Base44 to bytes
            audio_bytes = decode(audio_base44)
            
            # Write to file
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            
            print(f"✅ Saved {len(audio_bytes)} bytes")
            
        except Exception as e:
            print(f"❌ Error in save_audio_from_base44: {e}")
            raise


if __name__ == "__main__":
    print("=" * 70)
    print("TTS/STT Service with Base44 Encoding - Demo")
    print("=" * 70)
    
    # Initialize service
    service = TTSSTTServiceBase44()
    
    # Test examples for all languages
    test_cases = [
        ("Hello, how are you today?", "english"),
        ("Silav, tu çawa yî?", "kurdish"),
        ("Guten Tag, wie geht es Ihnen?", "german"),
        ("Bonjour, comment allez-vous aujourd'hui?", "french"),
        ("Merhaba, bugün nasılsınız?", "turkish"),
    ]
    
    print("\n🎤 Testing Text-to-Speech with Base44 encoding...\n")
    
    results = []
    for text, lang in test_cases:
        try:
            print(f"\n--- {lang.upper()} ---")
            result = service.text_to_speech_base44(text, lang)
            results.append((text, lang, result))
            
            print(f"Text: {result['text']}")
            print(f"Language: {result['language']}")
            print(f"Format: {result['format']}")
            print(f"Audio size: {result['size']} bytes")
            print(f"Encoded size: {result['encoded_size']} chars")
            print(f"Compression ratio: {result['compression_ratio']:.2f}x")
            print(f"Base44 preview: {result['audio'][:60]}...")
            
            # Save audio file
            output_file = f"output_{result['language']}.mp3"
            service.save_audio_from_base44(result['audio'], output_file)
            
        except Exception as e:
            print(f"Error with {lang}: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Demo completed!")
    print("=" * 70)
    
    # Summary
    print("\n📊 Summary:")
    print(f"  Languages tested: {len(test_cases)}")
    print(f"  Successful: {len(results)}")
    if results:
        avg_ratio = sum(r[2]['compression_ratio'] for r in results) / len(results)
        print(f"  Average compression ratio: {avg_ratio:.2f}x")
    print(f"\n  Audio files saved:")
    for _, lang, result in results:
        print(f"    - output_{result['language']}.mp3")
