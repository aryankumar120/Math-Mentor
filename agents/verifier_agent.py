import os
import json
from groq import Groq

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def verify_solution(parsed_problem, solution):
    client = _get_client()

    system = """You are a rigorous math solution verifier for JEE problems. Check the provided solution for:
1. Mathematical correctness (are the steps valid?)
2. Unit and domain consistency (are domain restrictions respected?)
3. Edge cases (could there be special cases overlooked?)
4. Final answer validity

Return this exact JSON format:
{
  "is_correct": true,
  "confidence": 0.9,
  "issues": ["list any issues found, empty if none"],
  "needs_hitl": false,
  "hitl_reason": ""
}

Set needs_hitl to true if confidence < 0.72 or if there are significant issues."""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Problem:\n{json.dumps(parsed_problem, indent=2)}\n\nSolution:\n{json.dumps(solution, indent=2)}",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    return json.loads(resp.choices[0].message.content)
