# VoiceBot — Multilingual Government Schemes Assistant

A voice-first RAG chatbot that answers questions about Indian government schemes in **Hindi, Punjabi, and English**. Users speak a question; the bot retrieves relevant schemes, generates a grounded answer, and replies in both text and synthesized speech — all in the user's language.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Confidence & Anti-Hallucination](#confidence--anti-hallucination)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Running with Docker](#running-with-docker)
6. [Environment Variables](#environment-variables)

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
