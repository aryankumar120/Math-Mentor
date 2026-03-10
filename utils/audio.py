import os
import tempfile
from groq import Groq

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def transcribe_audio(audio_bytes, filename="audio.wav"):
    client = _get_client()
    ext = os.path.splitext(filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                prompt="This is a math problem. It may contain terms like: square root, integral, derivative, raised to the power, factorial, summation, limit, infinity, log, sine, cosine, tangent.",
            )
        return transcript.text, 0.85
    finally:
        os.unlink(tmp_path)
