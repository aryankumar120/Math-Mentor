import os
import io
import base64
from groq import Groq

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def extract_text(image_bytes):
    client = _get_client()

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from this image exactly as written. This is a math problem — preserve all mathematical notation, symbols, equations, and numbers precisely. Output only the extracted text, nothing else.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0.1,
    )

    extracted = resp.choices[0].message.content.strip()
    confidence = 0.92 if extracted else 0.0
    return extracted, confidence, []
