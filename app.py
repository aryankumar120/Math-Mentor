import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud: inject secrets into env vars if present
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from agents.parser_agent import parse_problem
from agents.router_agent import route_problem
from agents.solver_agent import solve_problem
from agents.verifier_agent import verify_solution
from agents.explainer_agent import explain_solution
from rag.retriever import retrieve
from memory.store import (
    store_interaction,
    find_similar,
    update_feedback,
    get_ocr_corrections,
)
from utils.ocr import extract_text
from utils.audio import transcribe_audio

st.set_page_config(page_title="Math Mentor", page_icon="📐", layout="wide")


def init_state():
    defaults = {
        "trace": [],
        "result": None,
        "record_id": None,
        "extracted_text": "",
        "extraction_conf": 1.0,
        "hitl_active": False,
        "hitl_reason": "",
        "hitl_stage": "",
        "feedback_submitted": False,
        "input_mode": "Text",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def add_trace(agent, action, detail=""):
    st.session_state.trace.append(
        {"agent": agent, "action": action, "detail": detail}
    )


def run_pipeline(problem_text, input_type, force_solve=False):
    st.session_state.trace = []
    st.session_state.result = None
    st.session_state.hitl_active = False
    st.session_state.feedback_submitted = False

    add_trace("Parser Agent", "Parsing raw input")
    ocr_corrections = get_ocr_corrections()
    parsed = parse_problem(problem_text, ocr_corrections or None)
    add_trace(
        "Parser Agent",
        f"Topic: {parsed.get('topic', 'unknown')}",
        parsed.get("problem_text", "")[:120],
    )

    if parsed.get("needs_clarification") and not force_solve:
        st.session_state.hitl_active = True
        st.session_state.hitl_reason = parsed.get("clarification_reason", "Problem is ambiguous or incomplete.")
        st.session_state.hitl_stage = "parser"
        st.session_state.result = {"parsed": parsed, "stage": "parser_hitl"}
        return

    add_trace("Router Agent", "Classifying problem type and strategy")
    routing = route_problem(parsed)
    add_trace(
        "Router Agent",
        f"Strategy: {routing.get('strategy', '')} | Difficulty: {routing.get('difficulty', '')}",
        routing.get("approach", ""),
    )

    add_trace("RAG", "Retrieving relevant knowledge")
    query = f"{parsed.get('topic', '')} {parsed.get('problem_text', '')}"
    rag_chunks = retrieve(query, top_k=5)
    sources = list(dict.fromkeys(c["source"].replace(".txt", "") for c in rag_chunks))
    add_trace("RAG", f"Retrieved {len(rag_chunks)} chunks", "Sources: " + ", ".join(sources))

    similar = find_similar(problem_text, top_k=3)
    if similar:
        add_trace("Memory", f"Found {len(similar)} similar solved problems", "Reusing solution patterns")

    add_trace("Solver Agent", "Solving with RAG context")
    solution = solve_problem(parsed, routing, rag_chunks, similar or None)
    add_trace(
        "Solver Agent",
        f"Answer: {solution.get('final_answer', '')}",
        f"Method: {solution.get('method', '')}",
    )

    add_trace("Verifier Agent", "Checking correctness and domain constraints")
    verification = verify_solution(parsed, solution)
    conf_pct = f"{verification.get('confidence', 0):.0%}"
    issues = verification.get("issues", [])
    add_trace(
        "Verifier Agent",
        f"Confidence: {conf_pct} | Correct: {verification.get('is_correct', '?')}",
        (", ".join(issues) if issues else "No issues found"),
    )

    if verification.get("needs_hitl"):
        st.session_state.hitl_active = True
        st.session_state.hitl_reason = verification.get("hitl_reason", "Verifier confidence is low.")
        st.session_state.hitl_stage = "verifier"

    add_trace("Explainer Agent", "Generating student-friendly explanation")
    explanation = explain_solution(parsed, solution, routing)
    add_trace("Explainer Agent", explanation.get("summary", "Explanation ready"))

    record_id = store_interaction(
        problem_text, input_type, parsed, rag_chunks, solution, verification
    )
    st.session_state.record_id = record_id

    st.session_state.result = {
        "parsed": parsed,
        "routing": routing,
        "rag_chunks": rag_chunks,
        "solution": solution,
        "verification": verification,
        "explanation": explanation,
        "stage": "complete",
    }


def confidence_color(score):
    if score >= 0.80:
        return "green"
    if score >= 0.60:
        return "orange"
    return "red"


st.title("📐 Math Mentor")
st.caption("JEE-level math solver — Algebra · Probability · Calculus · Linear Algebra")

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Select input type", ["Text", "Image", "Audio"], index=0)

# Clear all output when the user switches input mode
if mode != st.session_state.input_mode:
    st.session_state.result = None
    st.session_state.trace = []
    st.session_state.hitl_active = False
    st.session_state.hitl_reason = ""
    st.session_state.hitl_stage = ""
    st.session_state.extracted_text = ""
    st.session_state.extraction_conf = 1.0
    st.session_state.feedback_submitted = False
    st.session_state.record_id = None

st.session_state.input_mode = mode

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.markdown(
    "Multi-agent system with RAG pipeline, OCR, speech-to-text, and self-learning memory."
)

problem_text_final = ""
input_type_label = mode.lower()
ready_to_solve = False

if mode == "Text":
    with st.form("text_input_form"):
        typed = st.text_area(
            "Type your math problem here",
            height=120,
            placeholder="e.g. Find all values of k for which x² + kx + 1 = 0 has two distinct real roots.",
        )
        solve_from_form = st.form_submit_button("🔍 Solve", type="primary")
    if solve_from_form and typed.strip():
        with st.spinner("Running agents..."):
            run_pipeline(typed.strip(), "text")
        st.rerun()

elif mode == "Image":
    uploaded_img = st.file_uploader("Upload a photo or screenshot of the problem", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        col_img, col_ocr = st.columns([1, 1])
        with col_img:
            st.image(uploaded_img, caption="Uploaded image", use_container_width=True)

        img_bytes = uploaded_img.read()

        if st.button("Extract Text from Image"):
            with st.spinner("Running OCR..."):
                text, conf, _ = extract_text(img_bytes)
                st.session_state.extracted_text = text
                st.session_state.extraction_conf = conf

        with col_ocr:
            if st.session_state.extracted_text:
                conf = st.session_state.extraction_conf
                conf_label = f"{conf:.0%}"
                if conf < 0.60:
                    st.warning(f"OCR confidence is low ({conf_label}). Please review and correct the extracted text before solving.")
                    st.session_state.hitl_active = True
                    st.session_state.hitl_reason = f"OCR confidence is low ({conf_label}). Please verify the extracted text."
                    st.session_state.hitl_stage = "ocr"
                else:
                    st.success(f"OCR confidence: {conf_label}")

                edited = st.text_area(
                    "Extracted text (edit if needed)",
                    value=st.session_state.extracted_text,
                    height=150,
                    key="ocr_edit",
                )
                if edited.strip():
                    problem_text_final = edited.strip()
                    ready_to_solve = True

                if conf < 0.60 and st.session_state.hitl_stage == "ocr":
                    st.info("You can edit the text above and proceed. Your correction will be saved to improve future OCR.")

elif mode == "Audio":
    st.markdown("**Upload audio** (.wav, .mp3, .m4a, .ogg) or **record in browser**:")

    try:
        from audio_recorder_streamlit import audio_recorder
        recorded_bytes = audio_recorder(text="Click to record", pause_threshold=3.0, icon_size="2x")
        if recorded_bytes:
            st.audio(recorded_bytes, format="audio/wav")
            if st.button("Transcribe Recording"):
                with st.spinner("Transcribing..."):
                    text, conf = transcribe_audio(recorded_bytes, "recording.wav")
                    st.session_state.extracted_text = text
                    st.session_state.extraction_conf = conf
    except ImportError:
        st.info("Install `audio-recorder-streamlit` to enable in-browser recording.")

    uploaded_audio = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded_audio:
        st.audio(uploaded_audio)
        if st.button("Transcribe Audio File"):
            with st.spinner("Transcribing with Whisper..."):
                audio_bytes = uploaded_audio.read()
                text, conf = transcribe_audio(audio_bytes, uploaded_audio.name)
                st.session_state.extracted_text = text
                st.session_state.extraction_conf = conf

    if st.session_state.extracted_text and mode == "Audio":
        conf = st.session_state.extraction_conf
        if conf < 0.70:
            st.warning("Transcription may be unclear. Please confirm below.")
            st.session_state.hitl_active = True
            st.session_state.hitl_reason = "Audio transcription confidence is low. Please verify the text."
            st.session_state.hitl_stage = "asr"
        else:
            st.success(f"Transcription complete (confidence: {conf:.0%})")

        edited_audio = st.text_area(
            "Transcript (edit if needed)",
            value=st.session_state.extracted_text,
            height=120,
            key="asr_edit",
        )
        if edited_audio.strip():
            problem_text_final = edited_audio.strip()
            ready_to_solve = True


if mode != "Text":
    st.markdown("---")
    solve_col, _ = st.columns([1, 3])
    with solve_col:
        solve_clicked = st.button("🔍 Solve", disabled=not ready_to_solve, type="primary")

    if solve_clicked and problem_text_final:
        if st.session_state.hitl_stage == "ocr" and st.session_state.extraction_conf < 0.60:
            orig_text = st.session_state.extracted_text
            if orig_text != problem_text_final:
                update_feedback(
                    -1,
                    {
                        "type": "ocr_correction",
                        "original": orig_text,
                        "corrected": problem_text_final,
                    },
                )
        with st.spinner("Running agents..."):
            run_pipeline(problem_text_final, input_type_label)
        st.rerun()


result = st.session_state.result

if result:
    if st.session_state.hitl_active and st.session_state.hitl_stage in ("parser", "verifier"):
        stage = st.session_state.hitl_stage
        st.warning(f"**Human Review Required** — {st.session_state.hitl_reason}")

        if stage == "parser":
            st.markdown("The parser found the problem ambiguous. Please clarify your question (or submit as-is):")
            clarified = st.text_area("Clarified problem", value=problem_text_final, height=100, key="hitl_clarify")
            if st.button("Resubmit with clarification"):
                st.session_state.hitl_active = False
                with st.spinner("Re-running with clarified input..."):
                    run_pipeline(clarified, input_type_label, force_solve=True)
                st.rerun()

        elif stage == "verifier":
            st.markdown("The verifier is not fully confident. You can review the solution below and approve or reject it.")
            sol = result.get("solution", {})
            st.markdown(f"**Proposed answer:** `{sol.get('final_answer', 'N/A')}`")
            hitl_action = st.radio("Action", ["Approve", "Reject and re-solve", "Edit answer"], horizontal=True)

            if hitl_action == "Edit answer":
                edited_ans = st.text_input("Corrected answer", value=sol.get("final_answer", ""))
                if st.button("Save corrected answer"):
                    result["solution"]["final_answer"] = edited_ans
                    result["solution"]["human_corrected"] = True
                    st.session_state.hitl_active = False
                    if st.session_state.record_id is not None:
                        update_feedback(
                            st.session_state.record_id,
                            {"type": "correction", "corrected_answer": edited_ans},
                        )
                    st.success("Answer updated and saved as learning signal.")
                    st.rerun()

            elif hitl_action == "Reject and re-solve":
                if st.button("Re-solve"):
                    st.session_state.hitl_active = False
                    with st.spinner("Re-running solver..."):
                        run_pipeline(problem_text_final, input_type_label)
                    st.rerun()

            elif hitl_action == "Approve":
                if st.button("Approve solution"):
                    st.session_state.hitl_active = False
                    if st.session_state.record_id is not None:
                        update_feedback(
                            st.session_state.record_id,
                            {"type": "approved_despite_low_confidence"},
                        )
                    st.rerun()

    if result.get("stage") == "complete":
        parsed = result["parsed"]
        solution = result["solution"]
        verification = result["verification"]
        explanation = result["explanation"]
        rag_chunks = result["rag_chunks"]

        st.markdown("---")

        top_left, top_right = st.columns([3, 1])
        with top_left:
            st.subheader("Problem")
            st.markdown(f"> {parsed.get('problem_text', '')}")
            st.markdown(
                f"**Topic:** `{parsed.get('topic', '')}` &nbsp;|&nbsp; **Strategy:** `{result['routing'].get('strategy', '')}` &nbsp;|&nbsp; **Difficulty:** `{result['routing'].get('difficulty', '')}`"
            )
        with top_right:
            conf = verification.get("confidence", 0)
            color = confidence_color(conf)
            st.markdown(f"**Confidence**")
            st.progress(conf)
            st.markdown(f"<span style='color:{color}; font-size:1.4em; font-weight:bold'>{conf:.0%}</span>", unsafe_allow_html=True)
            if not verification.get("is_correct", True):
                st.error("Verifier flagged issues")

        st.markdown("---")

        main_col, side_col = st.columns([2, 1])

        with main_col:
            st.subheader("Solution")
            steps = solution.get("solution_steps", [])
            for i, step in enumerate(steps, 1):
                st.markdown(f"**Step {i}**")
                st.markdown(step)
                st.markdown("<hr style='margin:8px 0;opacity:0.15'>", unsafe_allow_html=True)

            st.markdown("---")
            answer_val = solution.get('final_answer', 'N/A')
            st.markdown("### ✅ Answer:")
            st.markdown(f"#### {answer_val}")

            if verification.get("issues"):
                with st.expander("Verifier notes"):
                    for issue in verification["issues"]:
                        st.warning(issue)

            st.markdown("---")
            st.subheader("Explanation")
            exp_steps = explanation.get("steps", [])
            if exp_steps:
                for i, s in enumerate(exp_steps, 1):
                    title = s.get("title", f"Step {i}")
                    content = s.get("content", "")
                    st.markdown(f"#### ➡ Step {i}: {title}")
                    st.markdown(content)
                    st.markdown("<hr style='margin:10px 0;opacity:0.15'>", unsafe_allow_html=True)
            else:
                st.markdown(explanation.get("explanation", ""))

            concepts = explanation.get("key_concepts", [])
            if concepts:
                st.markdown("**Key concepts:** " + " · ".join(f"`{c}`" for c in concepts))

            tip = explanation.get("tip", "")
            if tip:
                st.info(f"**Exam tip:** {tip}")

        with side_col:
            with st.expander("Retrieved Context", expanded=True):
                if rag_chunks:
                    for chunk in rag_chunks:
                        src = chunk["source"].replace(".txt", "").replace("_", " ").title()
                        score_pct = f"{chunk['score']:.0%}"
                        st.markdown(f"**{src}** — relevance: {score_pct}")
                        st.caption(chunk["text"][:300] + "...")
                        st.markdown("<hr style='opacity:0.15'>", unsafe_allow_html=True)
                else:
                    st.write("No relevant chunks retrieved.")

            with st.expander("Agent Trace", expanded=False):
                for entry in st.session_state.trace:
                    st.markdown(f"**{entry['agent']}** — {entry['action']}")
                    if entry.get("detail"):
                        st.caption(entry["detail"])
                    st.markdown("---")

        st.markdown("---")
        st.subheader("Feedback")
        if not st.session_state.feedback_submitted:
            fb_col1, fb_col2 = st.columns([1, 2])
            with fb_col1:
                fb = st.radio("Was this answer correct?", ["✅ Correct", "❌ Incorrect"], horizontal=True, key="fb_radio")
            with fb_col2:
                comment = st.text_input("Comment (optional)", key="fb_comment")
            if st.button("Submit Feedback"):
                fb_type = "correct" if "Correct" in fb else "incorrect"
                payload = {"type": fb_type}
                if comment:
                    payload["comment"] = comment
                if st.session_state.record_id is not None:
                    update_feedback(st.session_state.record_id, payload)
                st.session_state.feedback_submitted = True
                st.success("Feedback saved! It will be used as a learning signal.")
                st.rerun()
        else:
            st.success("Thank you for your feedback.")

        similar_shown = find_similar(parsed.get("problem_text", ""), top_k=2)
        similar_shown = [s for s in similar_shown if s.get("id") != st.session_state.record_id]
        if similar_shown:
            with st.expander(f"Similar problems from memory ({len(similar_shown)})"):
                for s in similar_shown:
                    st.markdown(f"**Problem:** {s.get('input_text', '')[:150]}...")
                    if s.get("solution"):
                        st.markdown(f"**Answer:** `{s['solution'].get('final_answer', 'N/A')}`")
                    st.caption(f"Similarity: {s.get('similarity', 0):.0%} | Stored: {s.get('timestamp', '')[:10]}")
                    st.markdown("---")
