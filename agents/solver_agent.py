import os
import json
from groq import Groq

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


def solve_problem(parsed_problem, routing_info, rag_chunks, similar_solutions=None):
    client = _get_client()

    rag_context = ""
    if rag_chunks:
        parts = []
        for i, chunk in enumerate(rag_chunks, 1):
            parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
        rag_context = "\n\n".join(parts)

    system = f"""You are a JEE math solver. Solve problems step by step with full working.

Retrieved Knowledge (use this to inform your solution):
{rag_context}

Return this exact JSON format:
{{
  "solution_steps": ["step 1 - include key equations in LaTeX using $...$ for inline or $$...$$ for display", "step 2", ...],
  "final_answer": "write the answer as plain readable text. Use simple notation like: x = 3/5 or k < 25/4 or alpha + beta = 7. Do NOT use LaTeX or dollar signs here.",
  "confidence": 0.85,
  "used_sources": [1, 2],
  "method": "name of the method used"
}}

IMPORTANT: In solution_steps, every equation must use LaTeX. Use $$...$$ for display equations, $...$ for inline.
For final_answer, write plain readable text only — no dollar signs, no backslashes.
used_sources should list the source numbers [1..{len(rag_chunks)}] that directly helped."""

    user_content = f"Problem:\n{json.dumps(parsed_problem, indent=2)}\n\nRouting:\n{json.dumps(routing_info, indent=2)}"

    if similar_solutions:
        solved_patterns = []
        for s in similar_solutions[:2]:
            if s.get("solution"):
                solved_patterns.append(
                    {
                        "problem": s.get("input_text", "")[:200],
                        "answer": s["solution"].get("final_answer", ""),
                        "method": s["solution"].get("method", ""),
                    }
                )
        if solved_patterns:
            user_content += f"\n\nSimilar solved problems (for reference):\n{json.dumps(solved_patterns, indent=2)}"

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    return json.loads(resp.choices[0].message.content)
