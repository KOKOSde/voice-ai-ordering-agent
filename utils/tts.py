"""
Text-to-Speech utilities for voice responses.
Supports multiple TTS engines: gTTS, ElevenLabs, and Twilio's built-in TTS.
"""

import os
import tempfile
import logging
import hashlib
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache directory for generated audio
CACHE_DIR = Path(tempfile.gettempdir()) / "voice_ai_tts_cache"
CACHE_DIR.mkdir(exist_ok=True)


def text_to_speech(
    text: str,
    engine: str = "gtts",
    voice: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Convert text to speech audio file.
    
    Args:
        text: Text to convert to speech
        engine: TTS engine to use ('gtts', 'elevenlabs', 'pyttsx3')
        voice: Voice ID/name (engine-specific)
        output_path: Optional output file path
    
    Returns:
        Path to generated audio file
    """
    # Generate cache key
    cache_key = hashlib.md5(f"{text}:{engine}:{voice}".encode()).hexdigest()
    cached_path = CACHE_DIR / f"{cache_key}.mp3"
    
    # Return cached file if exists
    if cached_path.exists():
        logger.debug(f"Returning cached TTS: {cached_path}")
        return str(cached_path)
    
    # Generate audio
    output = output_path or str(cached_path)
    
    if engine == "elevenlabs":
        output = _generate_elevenlabs(text, voice, output)
    elif engine == "pyttsx3":
        output = _generate_pyttsx3(text, voice, output)
    else:  # Default to gTTS
        output = _generate_gtts(text, output)
    
    return output


def _generate_gtts(text: str, output_path: str) -> str:
    """Generate speech using Google Text-to-Speech (gTTS)."""
    try:
        from gtts import gTTS
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_path)
        logger.info(f"Generated gTTS audio: {output_path}")
        return output_path
        
    except ImportError:
        logger.error("gTTS not installed. Install with: pip install gtts")
        raise
    except Exception as e:
        logger.error(f"gTTS generation failed: {e}")
        raise


def _generate_elevenlabs(text: str, voice: Optional[str], output_path: str) -> str:
    """Generate speech using ElevenLabs API."""
    try:
        import requests
        
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            logger.warning("ELEVENLABS_API_KEY not set, falling back to gTTS")
            return _generate_gtts(text, output_path)
        
        # Default to a natural voice
        voice_id = voice or "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"Generated ElevenLabs audio: {output_path}")
        return output_path
        
    except ImportError:
        logger.warning("requests not available, falling back to gTTS")
        return _generate_gtts(text, output_path)
    except Exception as e:
        logger.error(f"ElevenLabs generation failed: {e}")
        return _generate_gtts(text, output_path)


def _generate_pyttsx3(text: str, voice: Optional[str], output_path: str) -> str:
    """Generate speech using pyttsx3 (offline TTS)."""
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Set voice if specified
        if voice:
            voices = engine.getProperty('voices')
            for v in voices:
                if voice.lower() in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break
        
        # Set rate for clarity
        engine.setProperty('rate', 150)
        
        # Save to file
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        logger.info(f"Generated pyttsx3 audio: {output_path}")
        return output_path
        
    except ImportError:
        logger.warning("pyttsx3 not installed, falling back to gTTS")
        return _generate_gtts(text, output_path)
    except Exception as e:
        logger.error(f"pyttsx3 generation failed: {e}")
        return _generate_gtts(text, output_path)


def get_audio_url(audio_path: str, base_url: Optional[str] = None) -> str:
    """
    Get a URL for an audio file.
    For Twilio, this needs to be a publicly accessible URL.
    
    Args:
        audio_path: Local path to audio file
        base_url: Base URL for the server (e.g., ngrok URL)
    
    Returns:
        URL to access the audio file
    """
    base = base_url or os.getenv("BASE_URL", "http://localhost:8000")
    
    # In production, you'd upload to cloud storage or serve from your server
    # For local development with ngrok, files can be served from a static directory
    filename = os.path.basename(audio_path)
    return f"{base}/audio/{filename}"


def preprocess_text_for_speech(text: str) -> str:
    """
    Preprocess text for more natural speech output.
    Handles abbreviations, numbers, and special formatting.
    """
    import re
    
    # Expand common abbreviations
    abbreviations = {
        "Dr.": "Doctor",
        "Mr.": "Mister",
        "Mrs.": "Missus",
        "Ms.": "Miss",
        "Jr.": "Junior",
        "Sr.": "Senior",
        "St.": "Street",
        "Ave.": "Avenue",
        "Blvd.": "Boulevard",
        "oz.": "ounces",
        "lb.": "pounds",
    }
    
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    
    # Handle prices (e.g., "$14.99" -> "14 dollars and 99 cents")
    def price_to_words(match):
        price = match.group(0)
        dollars, cents = price[1:].split(".")
        if cents == "00":
            return f"{dollars} dollars"
        elif cents.endswith("0"):
            return f"{dollars} dollars and {cents[0]}0 cents"
        else:
            return f"{dollars} dollars and {cents} cents"
    
    text = re.sub(r'\$\d+\.\d{2}', price_to_words, text)
    
    # Handle phone numbers
    def phone_to_words(match):
        phone = match.group(0)
        # Keep simple for now
        return phone.replace("-", " ")
    
    text = re.sub(r'\d{3}-\d{3}-\d{4}', phone_to_words, text)
    
    # Add pauses for better pacing
    text = text.replace(". ", "... ")
    text = text.replace("! ", "... ")
    text = text.replace("? ", "... ")
    
    return text


class TTSService:
    """
    High-level TTS service with engine selection and caching.
    """
    
    def __init__(
        self,
        default_engine: str = "gtts",
        default_voice: Optional[str] = None
    ):
        self.default_engine = default_engine
        self.default_voice = default_voice
        self._check_engines()
    
    def _check_engines(self):
        """Check available TTS engines."""
        self.available_engines = ["gtts"]  # gTTS always available (with internet)
        
        try:
            import pyttsx3
            self.available_engines.append("pyttsx3")
        except ImportError:
            pass
        
        if os.getenv("ELEVENLABS_API_KEY"):
            self.available_engines.append("elevenlabs")
        
        logger.info(f"Available TTS engines: {self.available_engines}")
    
    async def synthesize(
        self,
        text: str,
        engine: Optional[str] = None,
        voice: Optional[str] = None,
        preprocess: bool = True
    ) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            engine: TTS engine (default uses instance default)
            voice: Voice ID/name
            preprocess: Whether to preprocess text for natural speech
        
        Returns:
            Path to generated audio file
        """
        engine = engine or self.default_engine
        voice = voice or self.default_voice
        
        if preprocess:
            text = preprocess_text_for_speech(text)
        
        return text_to_speech(text, engine, voice)
    
    def clear_cache(self):
        """Clear the TTS cache directory."""
        import shutil
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(exist_ok=True)
        logger.info("TTS cache cleared")

