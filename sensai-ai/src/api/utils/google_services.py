import base64
import json
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.cloud import speech
from google.cloud.speech import RecognitionConfig, RecognitionAudio
from api.settings import settings
from api.config import google_plan_to_model_name

logger = logging.getLogger(__name__)

# Configure Google Gemini
if settings.google_gemini_api_key:
    genai.configure(api_key=settings.google_gemini_api_key)


class GoogleSpeechToText:
    """Google Speech-to-Text service for audio transcription"""
    
    def __init__(self):
        self.client = speech.SpeechClient()
    
    async def transcribe_audio(self, audio_data: bytes, language_code: str = "en-US") -> str:
        """
        Transcribe audio data to text using Google Speech-to-Text
        
        Args:
            audio_data: Audio data in bytes (WAV format)
            language_code: Language code (default: en-US)
            
        Returns:
            Transcribed text
        """
        try:
            # Configure recognition
            config = RecognitionConfig(
                encoding=RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,  # Adjust based on your audio
                language_code=language_code,
                enable_automatic_punctuation=True,
            )
            
            # Create recognition audio
            audio = RecognitionAudio(content=audio_data)
            
            # Perform transcription
            response = self.client.recognize(config=config, audio=audio)
            
            # Extract transcript
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript + " "
            
            logger.info(f"[GoogleSpeechToText] Transcribed {len(audio_data)} bytes to text")
            return transcript.strip()
            
        except Exception as e:
            logger.error(f"[GoogleSpeechToText] Error transcribing audio: {e}")
            raise


class GoogleGeminiLLM:
    """Google Gemini LLM service for AI chat and reasoning"""
    
    def __init__(self):
        if not settings.google_gemini_api_key:
            raise ValueError("Google Gemini API key not configured")
    
    def get_model(self, model_type: str) -> genai.GenerativeModel:
        """
        Get the appropriate Gemini model based on type
        
        Args:
            model_type: Type of model (reasoning, text, text-mini, audio, router)
            
        Returns:
            Configured Gemini model
        """
        model_name = google_plan_to_model_name.get(model_type, "gemini-1.5-pro")
        return genai.GenerativeModel(model_name)
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model_type: str = "text",
        max_tokens: int = 4096,
        temperature: float = 0.0
    ) -> str:
        """
        Generate chat completion using Gemini
        
        Args:
            messages: List of message dictionaries
            model_type: Type of model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        try:
            model = self.get_model(model_type)
            
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
                elif msg["role"] == "assistant":
                    gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
                elif msg["role"] == "system":
                    # Gemini doesn't have system messages, prepend to first user message
                    if gemini_messages and gemini_messages[0]["role"] == "user":
                        gemini_messages[0]["parts"][0]["text"] = f"{msg['content']}\n\n{gemini_messages[0]['parts'][0]['text']}"
                    else:
                        # If no user message yet, create one
                        gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
            
            # Generate response
            response = model.generate_content(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
            
            logger.info(f"[GoogleGeminiLLM] Generated response with {len(response.text)} characters")
            return response.text
            
        except Exception as e:
            logger.error(f"[GoogleGeminiLLM] Error generating response: {e}")
            raise
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model_type: str = "text",
        max_tokens: int = 4096,
        temperature: float = 0.0
    ):
        """
        Stream chat completion using Gemini
        
        Args:
            messages: List of message dictionaries
            model_type: Type of model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            Generated response chunks
        """
        try:
            model = self.get_model(model_type)
            
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
                elif msg["role"] == "assistant":
                    gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
                elif msg["role"] == "system":
                    # Gemini doesn't have system messages, prepend to first user message
                    if gemini_messages and gemini_messages[0]["role"] == "user":
                        gemini_messages[0]["parts"][0]["text"] = f"{msg['content']}\n\n{gemini_messages[0]['parts'][0]['text']}"
                    else:
                        # If no user message yet, create one
                        gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
            
            # Stream response
            response = model.generate_content(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                ),
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    logger.info(f"[GoogleGeminiLLM] Streaming chunk: {len(chunk.text)} characters")
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"[GoogleGeminiLLM] Error streaming response: {e}")
            raise


# Global instances
# Initialize Google Speech-to-Text service
try:
    if settings.google_application_credentials or settings.google_cloud_project_id:
        google_speech_to_text = GoogleSpeechToText()
    else:
        google_speech_to_text = None
        logger.warning("Google Speech-to-Text not configured - missing GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_PROJECT_ID")
except Exception as e:
    google_speech_to_text = None
    logger.warning(f"Failed to initialize Google Speech-to-Text: {e}")

google_gemini_llm = GoogleGeminiLLM() if settings.google_gemini_api_key else None


def transcribe_audio_with_google(audio_data: bytes) -> str:
    """
    Transcribe audio using Google Speech-to-Text
    
    Args:
        audio_data: Audio data in bytes
        
    Returns:
        Transcribed text
    """
    if not google_speech_to_text:
        # Fallback to a placeholder transcription
        logger.warning("Google Speech-to-Text not configured, returning placeholder text")
        return "[Audio transcription not available - Google credentials not configured]"
    
    return google_speech_to_text.transcribe_audio(audio_data)


def get_user_audio_message_for_chat_history_google(uuid: str) -> List[Dict]:
    """
    Get user audio message for chat history using Google Speech-to-Text
    
    Args:
        uuid: File UUID
        
    Returns:
        List of message dictionaries
    """
    from api.utils.s3 import download_file_from_s3_as_bytes, get_media_upload_s3_key_from_uuid
    import os
    
    # Download audio file
    if settings.s3_folder_name:
        audio_data = download_file_from_s3_as_bytes(
            get_media_upload_s3_key_from_uuid(uuid, "wav")
        )
    else:
        with open(os.path.join(settings.local_upload_folder, f"{uuid}.wav"), "rb") as f:
            audio_data = f.read()
    
    # Transcribe audio
    transcript = transcribe_audio_with_google(audio_data)
    
    return [
        {
            "type": "text",
            "text": f"Student's Response: {transcript}",
        }
    ] 