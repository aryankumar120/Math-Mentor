import os
import json
from groq import Groq

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def parse_problem(raw_text, ocr_corrections=None):
    client = _get_client()

    system = """You are a math problem parser for JEE-level questions. Given raw text (which may have OCR/speech errors), clean it and return a structured JSON object.

Output this exact JSON format:
{
  "problem_text": "cleaned and corrected problem statement",
  "topic": "algebra|probability|calculus|linear_algebra",
  "variables": ["list", "of", "variables"],
  "constraints": ["list of constraints like x > 0"],
  "needs_clarification": false,
  "clarification_reason": ""
}

Set needs_clarification to true only if the problem is genuinely ambiguous or critically incomplete."""

    user_msg = f"Parse this math problem:\n\n{raw_text}"
    if ocr_corrections:
        user_msg += f"\n\nApply these known corrections if relevant: {json.dumps(ocr_corrections)}"

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    return json.loads(resp.choices[0].message.content)
