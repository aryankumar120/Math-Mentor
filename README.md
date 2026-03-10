# 📐 Math Mentor

A **Reliable Multimodal Math Mentor** — a multi-agent AI system that solves JEE-level math problems from text, images, or audio. It explains solutions step-by-step, uses a RAG pipeline over a structured knowledge base, and learns from user feedback over time.

---

## What It Does

- Accepts math problems via **typed text**, **uploaded image** (OCR), or **audio** (speech-to-text)
- Routes each problem through **5 specialised agents**: Parser → Router → Solver → Verifier → Explainer
- Retrieves relevant formulas and theory from a **vector knowledge base** (RAG) before solving
- Triggers **Human-in-the-Loop (HITL)** review when the problem is ambiguous or confidence is low
- Stores every interaction in a **self-learning memory** and reuses similar past solutions
- Renders clean **step-by-step solutions with rendered math equations** in a Streamlit UI

---

## Architecture



---

## Project Structure

```
Math Mentor/
├── app.py                  # Streamlit UI + pipeline orchestration
├── .env                    # Your API keys (not committed)
├── .env.example            # Key template
├── requirements.txt
│
├── agents/
│   ├── parser_agent.py     # Parses & cleans raw input
│   ├── router_agent.py     # Classifies topic & strategy
│   ├── solver_agent.py     # Solves using RAG + memory
│   ├── verifier_agent.py   # Verifies correctness
│   └── explainer_agent.py  # Generates explanation
│
├── rag/
│   ├── knowledge_base/     # 17 domain .txt files
│   ├── embedder.py         # Builds FAISS index from KB
│   └── retriever.py        # Semantic search at query time
│
├── memory/
│   └── store.py            # Interaction store + similarity search
│
├── utils/
│   ├── ocr.py              # Image → text via Groq vision
│   └── audio.py            # Audio → text via Groq Whisper
│
└── data/                   # Generated at runtime (gitignored)
    ├── memory.json
    └── memory_index.bin
```

---

## Setup & Run

### 1. Prerequisites
- Python 3.9+
- A free [Groq API key](https://console.groq.com)

### 2. Clone & install

```bash
git clone <your-repo-url>
cd "Math Mentor"
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API key

```bash
cp .env.example .env
```

Open `.env` and set:

```
GROQ_API_KEY=gsk_your_key_here
```

### 4. Build the knowledge base index

Only needed once (or after editing knowledge base files):

```bash
python -m rag.embedder
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

Thank you, hope you like it.
