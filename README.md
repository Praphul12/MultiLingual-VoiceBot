# VoiceBot — Multilingual Government Schemes Assistant

A voice-first RAG chatbot that answers questions about Indian government schemes in **Hindi, Punjabi, and English**. Users speak a question; the bot retrieves relevant schemes, generates a grounded answer, and replies in both text and synthesized speech — all in the user's language.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Decisions](#component-decisions)
3. [Confidence & Anti-Hallucination](#confidence--anti-hallucination)
4. [Project Structure](#project-structure)
5. [Setup](#setup)
6. [Running with Docker](#running-with-docker)
7. [Environment Variables](#environment-variables)

---

## Architecture Overview

```
Audio Input
    │
    ▼
┌─────────────────────────────┐
│  STT  (Whisper medium)      │  → English text  (for LLM + cross-encoder)
│                             │  → Native text   (for FAISS query + UI display)
│                             │  → Language code (hi / pa / en)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  FAISS Vector Store         │  Queried with both English + native query
│  (indic-sentence-bert-nli)  │  150+ chunks across 3 languages, 50 schemes
└─────────────┬───────────────┘
              │ candidate docs
              ▼
┌─────────────────────────────┐
│  Cross-Encoder Re-ranker    │  ms-marco-MiniLM-L-6-v2
│  (English chunks only)      │  Min-max normalised confidence scores
└─────────────┬───────────────┘
              │ top-5 schemes
              ▼
┌─────────────────────────────┐
│  LLM  (Llama 3.1 8B)        │  Via Groq API — JSON output with confidence
│  English context + history  │  Anti-hallucination system prompt
└─────────────┬───────────────┘
              │ English answer
              ▼
┌─────────────────────────────┐
│  Translation                │  Google Translate (deep-translator)
│  English → target language  │  Skipped if user spoke English
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  TTS  (Facebook MMS-VITS)   │  Per-language VITS models (CPU-fast)
│  hi / pa / en               │  Optional: AI4Bharat Parler-TTS (GPU)
└─────────────┬───────────────┘
              │
              ▼
        Audio + Text Response
```

---

## Component Decisions

### Embedding Model: `l3cube-pune/indic-sentence-bert-nli`
Chosen over `ai4bharat/indic-bert` (base ALBERT, no pooling layer — cannot produce sentence embeddings) and `paraphrase-multilingual-mpnet-base-v2` (not retrieval-optimised, low similarity scores ~0.29). The L3Cube model is fine-tuned with Sentence-BERT methodology on NLI data specifically for Indian languages, mapping English, Hindi, and Punjabi into a shared semantic space for cross-lingual retrieval without translation.

### Cross-Lingual Retrieval: ID-based lookup
Rejected the approach of translating the query into all three languages and running three separate FAISS queries. The cross-lingual embedding model already handles language differences at retrieval time. Once the right scheme IDs are identified, all language versions are fetched directly from the source data — no translation errors, no extra latency.

### STT: OpenAI Whisper (medium)
Chosen over `ai4bharat/indic-conformer-600m-multilingual`, which has no English language mask (English queries fail entirely) and whose LID mode does not exist in the deployed model. Whisper handles Hindi, Punjabi, English, and Hinglish in one model with zero-cost automatic language detection.

### Vector Store: FAISS
Chosen over manual NumPy cosine similarity, which requires storing all vectors in memory and recomputing on every query. FAISS provides optimised similarity search via LangChain's `Retriever` interface, making it easy to swap backends (Chroma, Pinecone) later.

### LLM: Groq API (Llama 3.1 8B)
Replaced self-hosted open-source models (Airavata/Mistral-7B) due to:
- Airavata: gated repo access required
- Mistral-7B: 14 GB download, OOM on CPU inference
- Phi-3-mini: meta-device errors with `device_map="auto"`

Groq provides sub-second inference with no local GPU requirement. `llama-3.1-8b-instant` is sufficient for extractive QA over short retrieved context.

### Translation: Google Translate (`deep-translator`)
Chosen over `ai4bharat/indictrans2-indic-en-dist-200M` (IndicTrans2), which requires a Cython extension that fails to build on Windows and whose custom tokenizer is incompatible with newer `transformers` versions. For short queries and answers, Google Translate quality is sufficient.

### TTS Backend: Facebook MMS-VITS (default)
Parler-TTS (`ai4bharat/indic-parler-tts`) is autoregressive and takes 3–4 minutes on CPU. MMS-VITS is non-autoregressive and generates speech in seconds. Both backends are kept in `TTS/pipeline.py` switchable via a single `BACKEND` flag.

### Per-Language Chunking
Each scheme is chunked separately per language rather than joining all three into one document. Joining dilutes the embedding — the vector becomes a mix of three languages and does not represent any one well. Separate per-language chunks with a shared `id` metadata field enable ID-based cross-language lookup after retrieval.

---

## Confidence & Anti-Hallucination

**Retrieval confidence** is the min-max normalised cross-encoder score for each retrieved scheme within the current query's candidate batch. Displayed as a progress bar per scheme in the UI.

**Generator confidence** is self-reported by the LLM (0–100) in its JSON output. It is capped at 40 if the model reports `grounded: false`.

A warning banner is shown when:
- Average retrieval confidence < 0.4 (RETRIEVAL_THRESHOLD), or
- Generator confidence < 40%

---

## Project Structure

```
VoiceBot/
├── app.py                          # Streamlit UI (multi-turn conversation)
├── main.py                         # CLI entry point (multi-turn loop)
├── stt.py                          # Speech-to-text (Whisper)
├── translate.py                    # English → target language (Google Translate)
├── llm.py                          # LLM answer generation (Groq)
├── VectorStore/
│   ├── pipeline.py                 # Chunking, FAISS, cross-encoder re-ranking
│   └── __init__.py
├── TTS/
│   ├── pipeline.py                 # VITS + Parler TTS backends
│   └── __init__.py
├── schemes_multilingual (1).csv    # Source data: 50 schemes × 3 languages
├── decision.md                     # Architectural decision records
├── pyproject.toml                  # Dependencies (managed with uv)
├── uv.lock
├── Dockerfile
└── docker-compose.yml
```

---

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg (for audio processing)
- A [Groq API key](https://console.groq.com/)

### Install

```bash
git clone <repo-url>
cd VoiceBot
cp .env.example .env        # add your GROQ_API_KEY
uv sync
```

### Run

```bash
# Streamlit UI
uv run streamlit run app.py

# CLI (processes a fixed audio file in a loop)
uv run python main.py
```

---

## Running with Docker

```bash
# Build and start
docker compose up --build

# The app is available at http://localhost:8501
```

Model weights (Whisper, HuggingFace, Torch) are persisted in named Docker volumes so they are not re-downloaded on container restart.

> **First run is slow** — Whisper medium (~1.5 GB), the embedding model, cross-encoder, and VITS TTS models are all downloaded on first use.

### GPU / Parler-TTS (optional)

The default VITS backend runs on CPU. To use Parler-TTS, set `BACKEND = "parler"` in `TTS/pipeline.py` and add a GPU runtime to `docker-compose.yml`:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for Llama 3.1 8B inference |

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```
