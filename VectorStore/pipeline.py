from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import csv
LANGUAGES = ["english", "hindi", "punjabi"]

# Schemes with re-ranker sigmoid score below this are considered low-confidence
RETRIEVAL_THRESHOLD = 0.4

embed_model = HuggingFaceEmbeddings(
    model_name="l3cube-pune/indic-sentence-bert-nli"
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



def get_chunks(doc):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    with open(doc, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata = {
                "id": row["id"],
                "scheme_name": row["scheme_name"],
                "category": row["category"],
                "source_url": row["source_url"],
            }
            for lang in LANGUAGES:
                split_texts = splitter.split_text(row[lang])
                for chunk in split_texts:
                    # Prepend scheme name so embeddings capture name-based queries
                    content = f"{row['scheme_name']}: {chunk}"
                    chunks.append(Document(page_content=content, metadata={**metadata, "language": lang}))
    return chunks


def build_vectorstore(chunks: list[Document]) -> FAISS:
    return FAISS.from_documents(chunks, embed_model)


def get_retriever(vectorstore: FAISS, top_k: int = 5):
    # Over-fetch widely so the cross-encoder sees enough unique schemes
    return vectorstore.as_retriever(search_kwargs={"k": top_k * len(LANGUAGES) * 2})


def retrieve_all_languages(
    query: str,
    retriever,
    chunks: list[Document],
    top_k: int = 5,
    query_native: str = None,
) -> tuple[dict[str, list[Document]], dict[str, float]]:
    """
    Retrieve and re-rank top matching schemes.

    Queries FAISS with both the English query and (optionally) the native-language query,
    then merges candidates before cross-encoder re-ranking.

    Returns:
        (id_to_chunks, scheme_scores) where scheme_scores maps scheme_id -> min-max normalised confidence (0-1)
    """
    candidates = retriever.invoke(query)

    # Also retrieve with native query and merge (deduplicating by doc id)
    if query_native and query_native != query:
        native_candidates = retriever.invoke(query_native)
        seen_ids = {id(d) for d in candidates}
        for doc in native_candidates:
            if id(doc) not in seen_ids:
                candidates.append(doc)
                seen_ids.add(id(doc))

    # Re-rank using English chunks only
    english_candidates = [doc for doc in candidates if doc.metadata["language"] == "english"]
    pairs = [(query, doc.page_content) for doc in english_candidates]
    raw_scores = reranker.predict(pairs)

    # Normalise to 0-1 with min-max within the batch
    # (sigmoid skews too heavily — a score of -5 gives 0.7% even if it's the best match)
    min_s, max_s = min(raw_scores), max(raw_scores)
    if max_s > min_s:
        norm_scores = [(s - min_s) / (max_s - min_s) for s in raw_scores]
    else:
        norm_scores = [0.5] * len(raw_scores)

    scored = sorted(
        zip(norm_scores, english_candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    # Deduplicate by scheme_id, keeping top_k unique schemes
    seen: dict[str, float] = {}
    for score, doc in scored:
        sid = doc.metadata["id"]
        if sid not in seen:
            seen[sid] = score
        if len(seen) == top_k:
            break

    matched_ids = list(seen.keys())

    # Build lookup: scheme_id -> all language chunks
    id_to_chunks: dict[str, list[Document]] = {sid: [] for sid in matched_ids}
    for chunk in chunks:
        if chunk.metadata["id"] in id_to_chunks:
            id_to_chunks[chunk.metadata["id"]].append(chunk)

    return id_to_chunks, seen
