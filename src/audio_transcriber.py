import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def transcribe_audio(file_path: str, translate_to_english: bool = False) -> str:
    """Transcribes an audio file (mp3, wav, m4a) into text using Whisper via Groq.

    If translate_to_english is True, uses Groq's translation endpoint instead —
    the doctor-patient conversation can be in any supported language, and the
    output text will be in English, ready for the existing English-trained
    SOAP generation and NER pipeline.
    """
    with open(file_path, "rb") as audio_file:
        if translate_to_english:
            result = client.audio.translations.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="text",
            )
        else:
            result = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="text",
            )
    return result