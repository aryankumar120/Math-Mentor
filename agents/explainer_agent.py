import os
import json
from groq import Groq

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def explain_solution(parsed_problem, solution, routing_info):
    client = _get_client()

    system = """You are a friendly JEE math tutor. Create a clear, student-friendly step-by-step explanation.

Return this exact JSON format:
{
  "steps": [
    {
      "title": "Short descriptive title for this step",
      "content": "Plain English explanation text here. Put every equation on its own line as a display block."
    }
  ],
  "key_concepts": ["concept 1", "concept 2"],
  "tip": "one exam tip or common mistake to avoid",
  "summary": "one-sentence summary of the core approach"
}

STRICT RULES for writing math in the content field:
- NEVER write LaTeX commands like \\frac, \\lim, \\sin inside a sentence mixed with words.
- ALL equations must go on their OWN separate line as a display block surrounded by $$ on each side.
- Example of correct format:
    "We apply the standard limit identity:\\n\\n$$\\\\lim_{x \\\\to 0} \\\\frac{\\\\sin x}{x} = 1$$\\n\\nThis lets us simplify the expression."
- Use simple readable notation inside prose text: write sin(3x)/5x, not LaTeX fractions.
- Each step title should be meaningful (e.g. Identify the Formula, Rewrite the Expression).
- Write content as if talking to the student directly.
- Aim for 3-5 steps."""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Problem:\n{json.dumps(parsed_problem, indent=2)}\n\nSolution:\n{json.dumps(solution, indent=2)}\n\nTopic/Strategy:\n{routing_info.get('subtopic', '')} - {routing_info.get('approach', '')}",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )

    result = json.loads(resp.choices[0].message.content)

    # When a model writes bare \frac, \to, \times etc. in a JSON string value,
    # Python's json.loads interprets the escape sequences:
    #   \f  → form-feed  (chr 12, \x0c)  →  ruins \frac, \forall, etc.
    #   \t  → tab        (chr  9, \x09)  →  ruins \to, \times, \theta, etc.
    #   \b  → backspace  (chr  8, \x08)  →  ruins \beta, \binom, etc.
    #   \r  → CR         (chr 13, \x0d)  →  ruins \right, \rho, etc.
    # Reconstruct them so LaTeX renders correctly.
    _repairs = [
        ("\x0c", "\\f"),   # form-feed → \f  (restores \frac, \forall …)
        ("\x09o",  "\\to"),
        ("\x09imes", "\\times"),
        ("\x09heta", "\\theta"),
        ("\x09au",   "\\tau"),
        ("\x08eta",  "\\beta"),
        ("\x08inom", "\\binom"),
        ("\x0dight", "\\right"),
        ("\x0dho",   "\\rho"),
    ]

    def _fix(text):
        if not isinstance(text, str):
            return text
        for bad, good in _repairs:
            text = text.replace(bad, good)
        return text

    for step in result.get("steps", []):
        if "content" in step:
            step["content"] = _fix(step["content"])

    return result
