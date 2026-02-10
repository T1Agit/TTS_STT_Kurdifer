"""
TTS/STT Service with Base44 Encoding

Provides Text-to-Speech and Speech-to-Text functionality with Base44 encoding
for Kurdish, German, French, English, and Turkish languages.

IMPORTANT - NO FALLBACK POLICY:
- Kurdish (ku) ALWAYS uses Coqui TTS for TTS - never falls back to other engines
- Kurdish (ku) ALWAYS uses Google STT with 'ku' language code - never falls back
- If Kurdish TTS/STT fails, an error is raised instead of falling back to another language
- This ensures Kurdish language integrity and prevents mixing languages
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
    
    def _uses_coqui_tts(self, lang_code: str) -> bool:
        """
        Check if language requires Coqui TTS
        
        Args:
            lang_code: Language code (e.g., 'ku', 'en')
            
        Returns:
            True if language needs Coqui TTS, False otherwise
        """
        return lang_code == 'ku'
    
    def _generate_speech_coqui(self, text: str, lang_code: str) -> bytes:
        """
        Generate speech using Coqui TTS for Kurdish
        
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
                # Use a multilingual model that supports Kurdish
                # Note: First-time initialization will download ~2GB of model data
                # and may take 2-5 minutes depending on network speed.
                # Subsequent calls will use cached model and be much faster.
                self._coqui_tts = TTS(
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                    progress_bar=False
                )
                print("✅ Coqui TTS initialized")
            
            # Generate speech to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Generate audio using Coqui TTS
            # XTTS v2 supports Kurdish (ku) as part of its multilingual capabilities
            self._coqui_tts.tts_to_file(
                text=text,
                file_path=temp_path,
                language="ku"
            )
            
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
            # NO FALLBACK: Kurdish TTS must use Coqui TTS only
            # If Coqui TTS fails, we raise an error instead of falling back to other engines
            raise RuntimeError(f"Kurdish TTS generation failed: {e}") from e
    
    def text_to_speech_base44(
        self,
        text: str,
        language: str = 'en',
        audio_format: str = 'mp3'
    ) -> Dict[str, str]:
        """
        Convert text to speech and encode to Base44
        
        Args:
            text: Text to convert to speech
            language: Target language (e.g., 'en', 'english', 'ku')
            audio_format: Audio format ('mp3', 'wav', 'ogg')
            
        Returns:
            Dictionary with audio data and metadata
        """
        try:
            # Get language code
            lang_code = self._get_language_code(language)
            
            print(f"🎤 Generating speech: '{text[:50]}...' in {lang_code}")
            
            # Check if we need to use Coqui TTS for Kurdish
            if self._uses_coqui_tts(lang_code):
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
                    'compression_ratio': len(audio_base44) / len(audio_bytes)
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
            # NO FALLBACK: If Kurdish speech cannot be recognized, we raise an error
            # instead of falling back to another language
            raise ValueError("Could not understand Kurdish audio")
        except sr.RequestError as e:
            print(f"❌ API error: {e}")
            # NO FALLBACK: API errors are raised directly without fallback
            raise RuntimeError(f"Kurdish STT API error: {e}") from e
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
