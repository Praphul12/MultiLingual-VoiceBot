import os
import json
from groq import Groq

MODEL_ID = "llama-3.1-8b-instant"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def generate_answer(
    query_en: str,
    context_docs: dict,
    history: list = None,
) -> tuple[str, int, list]:
    """
    Generate an English answer using Llama 3.1 8B via Groq API.

    Returns:
        (answer, confidence 0-100, updated_history)
    """
    if history is None:
        history = []

    context_parts = []
    for scheme_id, docs in context_docs.items():
        english_docs = [d for d in docs if d.metadata["language"] == "english"]
        if not english_docs:
            english_docs = docs
        scheme_name = docs[0].metadata["scheme_name"]
        category = docs[0].metadata["category"]
        source_url = docs[0].metadata.get("source_url", "")
        prefix = f"{scheme_name}: "
        content = " ".join(
            d.page_content[len(prefix):] if d.page_content.startswith(prefix) else d.page_content
            for d in english_docs
        )
        context_parts.append(
            f"### Scheme: {scheme_name}\n"
            f"Category: {category}\n"
            f"Official URL: {source_url if source_url else 'Not available'}\n"
            f"Details: {content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant for Indian government schemes. "
            "Answer ONLY using facts explicitly stated in the provided context. "
            "Do NOT invent scheme names, eligibility criteria, benefit amounts, or URLs. "
            "If the context does not contain enough information, say so honestly. "
            "Use the closest relevant scheme even if the name doesn't match exactly.\n\n"
            "You MUST respond with a JSON object in this exact format:\n"
            '{"answer": "<your answer in simple plain English, under 120 words>", '
            '"confidence": <integer 0-100 reflecting how well the context supports your answer>, '
            '"grounded": <true if every fact in your answer comes from the context, false otherwise>}'
        ),
    }

    user_message = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query_en}",
    }

    messages = [system] + history + [user_message]

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=350,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        answer = parsed.get("answer", raw)
        confidence = int(parsed.get("confidence", 50))
        grounded = parsed.get("grounded", True)
        # Penalise confidence if LLM itself says it's not grounded
        if not grounded:
            confidence = min(confidence, 40)
    except (json.JSONDecodeError, ValueError):
        answer = raw
        confidence = 50

    updated_history = history + [
        user_message,
        {"role": "assistant", "content": json.dumps({"answer": answer, "confidence": confidence})},
    ]
    return answer, confidence, updated_history
