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

<img width="889" height="291" alt="Screenshot 2026-03-11 at 1 32 59 AM" src="https://github.com/user-attachments/assets/b758fd2a-027a-420c-ae28-4d457240e2a9" />


---

## Project Structure

```
Math Mentor/
├── app.py                  
├── .env                    
├── .env.example            #Important to read before starting
├── requirements.txt
│
├── agents/
│   ├── parser_agent.py     
│   ├── router_agent.py     
│   ├── solver_agent.py     
│   ├── verifier_agent.py   
│   └── explainer_agent.py  
│
├── rag/
│   ├── knowledge_base/     
│   ├── embedder.py         
│   └── retriever.py        
│
├── memory/
│   └── store.py           
│
├── utils/
│   ├── ocr.py              
│   └── audio.py            
│
└── data/                   
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


---

Thank you, hope you like it.
