import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
from VectorStore.pipeline import get_chunks, build_vectorstore, get_retriever, retrieve_all_languages
from stt import transcribe
from translate import translate_from_english
from llm import generate_answer
from TTS.pipeline import synthesize


PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemes_multilingual (1).csv")
AUDIO_PATH = r"C:\Users\praph\Downloads\WhatsApp Ptt 2026-03-27 at 12.34.05.wav"
OUTPUT_AUDIO_PATH = r"C:\Courses\VoiceBot\response.wav"


def main():
    # Step 1: Build RAG pipeline once
    print("Building RAG pipeline...")
    chunks = get_chunks(PATH)
    print(f"Total chunks: {len(chunks)}")
    vectorstore = build_vectorstore(chunks)
    retriever = get_retriever(vectorstore, top_k=5)

    history = []
    turn = 1

    while True:
        print(f"\n{'─' * 40}")
        print(f"Turn {turn} — press Enter to process '{AUDIO_PATH}', or type 'q' to quit.")
        user_input = input("> ").strip()
        if user_input.lower() == "q":
            print("Ending conversation.")
            break

        # Step 2: STT
        print(f"\n[STT] Transcribing: {AUDIO_PATH}")
        query_en, query_native, lang = transcribe(AUDIO_PATH)

        # Step 3: Retrieve
        results, retrieval_scores = retrieve_all_languages(
            query_en, retriever, chunks, top_k=5, query_native=query_native
        )
        print(f"\n[RETRIEVE] Matched {len(results)} scheme(s):")
        for scheme_id, docs in results.items():
            print(f"  - [{scheme_id}] {docs[0].metadata['scheme_name']} (score: {retrieval_scores[scheme_id]:.2f})")

        # Step 4: Generate with history
        print("\n[LLM] Generating English answer...")
        answer_en, gen_confidence, history = generate_answer(query_en, results, history)
        print(f"Generator confidence: {gen_confidence}%")
        print(f"English answer: {answer_en}")

        # Step 5: Translate
        answer = translate_from_english(answer_en, lang)
        print(f"\n[TRANSLATE] Answer in '{lang}':\n{answer}")

        # Step 6: TTS
        print(f"\n[TTS] Synthesizing speech → {OUTPUT_AUDIO_PATH}")
        synthesize(answer, lang, OUTPUT_AUDIO_PATH)
        print("Done.")

        turn += 1


if __name__ == "__main__":
    main()
