# Architectural Decisions

## 1. Embedding Model: l3cube-pune/indic-sentence-bert-nli

**Chosen over:** `ai4bharat/indic-bert`, `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

**Why:**
- `ai4bharat/indic-bert` is a base ALBERT model, not a sentence-transformer. It has no pooling layer and cannot produce sentence-level embeddings directly.
- `paraphrase-multilingual-mpnet-base-v2` is not retrieval-optimized and gave low similarity scores (~0.29).
- `l3cube-pune/indic-sentence-bert-nli` is fine-tuned with Sentence-BERT methodology on NLI data specifically for Indian languages. It maps English, Hindi, and Punjabi into a shared semantic space, enabling cross-lingual retrieval without any translation.

---

## 2. Cross-lingual Retrieval: ID-based lookup vs. Query Translation

**Chosen:** ID-based lookup
**Rejected:** Translating the query into all 3 languages and running 3 separate retrievals

**Why ID-based lookup is better:**
- The cross-lingual embedding model already handles language differences at retrieval time — a Hindi or Hinglish query correctly retrieves relevant chunks regardless of the chunk's language.
- Once the right scheme IDs are retrieved, all language versions are fetched directly from the original data — no translation errors, no extra model calls.
- Translation approach would require 3 separate translation model calls (IndicTrans2) + 3 retrieval calls + deduplication logic, all of which add latency and potential for compounding errors.
- Hinglish / code-switched queries are handled naturally by the embedding model without needing to detect or separate the languages first.

---

## 3. STT Model: OpenAI Whisper (medium)

**Chosen over:** ai4bharat/indic-conformer-600m-multilingual

**Why:**
- IndicConformer is Indic-only — it has no English language mask, so English queries fail entirely.
- Whisper's multilingual model handles Hindi, Punjabi, English, and Hinglish in a single model with automatic language detection (`result["language"]`).
- Removes the need for a separate language detection heuristic (script ratio scoring) that was needed to disambiguate Hindi vs Punjabi vs English outputs from IndicConformer.
- Simpler pipeline: one model, one call, language detected automatically.

---

## 4. Vector Store: FAISS over manual cosine similarity

**Chosen over:** NumPy cosine similarity

**Why:**
- Manual cosine similarity required storing all vectors in memory and recomputing on every query — does not scale.
- FAISS provides optimized similarity search via LangChain's standard `Retriever` interface, making it easy to swap backends later (e.g. Chroma, Pinecone).
- `vectorstore.as_retriever()` integrates cleanly into LangChain chains for future RAG expansion.

---

## 5. Per-language Document Chunking

**Chosen over:** Joining all languages into one document

**Why:**
- Joining all languages into a single chunk dilutes the embedding — the vector becomes a mix of 3 languages and doesn't represent any one language well.
- Separate documents per language allow the embedding model to produce clean, language-specific vectors while the `language` metadata field enables filtering during retrieval if needed.
- The ID field in metadata ties all language versions of the same scheme together, enabling the ID-based cross-language lookup (Decision 2).

---

## 6. LLM: ai4bharat/Airavata (open-source) over Claude API

**Chosen over:** Anthropic Claude API

**Why:**
- Airavata is an open-source instruction-tuned LLM from AI4Bharat, fine-tuned on Hindi+English data on top of Mistral 7B (OpenHermes-2.5).
- Keeps the entire pipeline open-source and self-hosted — no external API dependency or cost per query.
- Sufficient for extractive QA over short retrieved context in Hindi/English.

---

## 7. Language Detection: Whisper built-in detection

**Chosen:** Whisper's automatic language detection (`result["language"]`)
**Rejected:** `langdetect` library, LLM-based detection, IndicConformer LID mode

**Why:**
- Whisper detects language from audio as part of transcription — zero extra cost.
- IndicConformer's `"lid"` mode turned out to not exist in the deployed model (returns `None`).
- `langdetect` works on text and struggles with short Indic queries.
- Whisper handles Hindi, Punjabi, English, and Hinglish naturally.

---

## 8. Query Translation: Google Translate (deep-translator) over IndicTrans2

**Chosen:** `deep-translator` (GoogleTranslator) for query → English translation
**Rejected:** `ai4bharat/indictrans2-indic-en-dist-200M`

**Why:**
- IndicTrans2 requires `IndicTransToolkit` which has a Cython extension that fails to build on Windows due to missing SDK headers.
- IndicTrans2's custom tokenizer (`IndicTransTokenizer`) is incompatible with newer versions of `transformers`.
- Google Translate via `deep-translator` requires no model download, no compilation, and handles all languages including Punjabi.
- For short queries (the only text being translated), quality is sufficient.

---

## 9. Pivot Language for RAG: Translate query to English before retrieval

**Chosen:** Translate query → English → retrieve → LLM answers in detected language
**Rejected:** Retrieve using original-language query, then translate answer back

**Why:**
- English query retrieves against English chunks which are the most complete and reliably embedded.
- The LLM (Airavata) receives English context regardless of the user's language, ensuring consistent answer quality.
- Airavata is instructed to answer directly in the detected language, eliminating a separate answer-translation step.
