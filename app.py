import os
import sys
import tempfile

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from VectorStore.pipeline import get_chunks, build_vectorstore, get_retriever, retrieve_all_languages, RETRIEVAL_THRESHOLD
from stt import transcribe
from translate import translate_from_english
from llm import generate_answer
from TTS.pipeline import synthesize

PATH = os.path.join(os.path.dirname(__file__), "schemes_multilingual (1).csv")
LANG_NAMES = {"hi": "Hindi", "pa": "Punjabi", "en": "English"}


def confidence_badge(score: float, is_percent: bool = False) -> str:
    """Return a coloured emoji label for a confidence score."""
    val = score if is_percent else score * 100
    if val >= 70:
        return f"🟢 {val:.0f}%"
    elif val >= 40:
        return f"🟡 {val:.0f}%"
    else:
        return f"🔴 {val:.0f}%"


@st.cache_resource(show_spinner="Building RAG pipeline...")
def load_pipeline():
    chunks = get_chunks(PATH)
    vectorstore = build_vectorstore(chunks)
    retriever = get_retriever(vectorstore, top_k=5)
    return chunks, retriever


def process_turn(audio_bytes: bytes, chunks, retriever):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    try:
        with st.status("Processing your question...", expanded=True) as status:

            st.write("🎙️ Transcribing audio...")
            query_en, query_native, lang = transcribe(audio_path)
            st.write(f"Detected language: **{LANG_NAMES.get(lang, lang)}** | Query: *{query_en}*")

            st.write("🔍 Retrieving & re-ranking schemes...")
            results, retrieval_scores = retrieve_all_languages(
                query_en, retriever, chunks, top_k=5, query_native=query_native
            )
            retrieval_scores = {k: float(v) for k, v in retrieval_scores.items()}
            scheme_names = [docs[0].metadata["scheme_name"] for docs in results.values()]
            st.write(f"Matched {len(results)} scheme(s)")

            st.write("🤖 Generating answer...")
            answer_en, gen_confidence, updated_history = generate_answer(
                query_en, results, st.session_state.history
            )
            st.session_state.history = updated_history

            st.write("🌐 Translating...")
            answer = translate_from_english(answer_en, lang)

            st.write("🔊 Synthesizing speech...")
            out_fd, out_path = tempfile.mkstemp(suffix=".wav")
            os.close(out_fd)  # release the handle so synthesize() can write to it
            synthesize(answer, lang, out_path)
            with open(out_path, "rb") as f:
                audio_out = f.read()
            os.unlink(out_path)

            status.update(label="Done!", state="complete", expanded=False)

        return {
            "query_en": query_en,
            "query_native": query_native,
            "lang": lang,
            "schemes": scheme_names,
            "retrieval_scores": retrieval_scores,
            "answer_en": answer_en,
            "gen_confidence": gen_confidence,
            "answer": answer,
            "audio": audio_out,
        }

    finally:
        os.unlink(audio_path)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceBot – Government Schemes",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ VoiceBot")
st.caption("Ask about Indian government schemes in Hindi, Punjabi, or English")

chunks, retriever = load_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []
if "turns" not in st.session_state:
    st.session_state.turns = []

# ── Conversation display ──────────────────────────────────────────────────────
for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.write(turn["query_native"])
        st.caption(f"Language: {LANG_NAMES.get(turn['lang'], turn['lang'])}")

    with st.chat_message("assistant"):
        # Low confidence warning
        avg_retrieval = sum(turn["retrieval_scores"].values()) / max(len(turn["retrieval_scores"]), 1)
        if avg_retrieval < RETRIEVAL_THRESHOLD or turn["gen_confidence"] < 40:
            st.warning("⚠️ Low confidence — answer may not be fully reliable.")

        st.write(turn["answer"])
        st.audio(turn["audio"], format="audio/wav")

        with st.expander("📊 Confidence scores"):
            st.markdown("**Generator confidence**")
            st.progress(turn["gen_confidence"] / 100, text=confidence_badge(turn["gen_confidence"], is_percent=True))

            st.markdown("**Retriever confidence (per scheme)**")
            for i, (sid, score) in enumerate(turn["retrieval_scores"].items()):
                label = turn["schemes"][i] if i < len(turn["schemes"]) else sid
                st.progress(score, text=f"{label}: {confidence_badge(score)}")

        with st.expander("🔍 Retrieved schemes"):
            for name in turn["schemes"]:
                st.write(f"• {name}")

        with st.expander("🇬🇧 English answer"):
            st.write(turn["answer_en"])

# ── Audio input ───────────────────────────────────────────────────────────────
st.divider()
audio_input = st.audio_input("🎤 Record your question")

if audio_input:
    if st.button("Submit", type="primary", use_container_width=True):
        result = process_turn(audio_input.getvalue(), chunks, retriever)
        st.session_state.turns.append(result)
        st.rerun()

if st.session_state.turns:
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.turns = []
        st.rerun()
