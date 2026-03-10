import os
import json
from groq import Groq

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def route_problem(parsed_problem):
    client = _get_client()

    system = """You are a math problem router for JEE-level questions. Given a structured math problem, determine the best solution strategy and return a JSON object.

Output this exact JSON format:
{
  "strategy": "algebraic|probabilistic|calculus|linear_algebraic|combinatorial",
  "subtopic": "specific subtopic (e.g. quadratic equations, bayes theorem, optimization)",
  "difficulty": "easy|medium|hard",
  "requires_computation": false,
  "approach": "brief 1-2 sentence description of the solution approach",
  "key_formulas": ["list of key formulas or theorems to use"]
}"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Route this problem:\n{json.dumps(parsed_problem, indent=2)}",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    return json.loads(resp.choices[0].message.content)
