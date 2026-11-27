import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://qdrant:6333"
OLLAMA_URL = "http://ollama:11434"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM = "qwen2.5:1.5b"

_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def search(question, chunk_size, k=3):
    qv = embedder().encode(question).tolist()
    return QdrantClient(url=QDRANT_URL).query_points(
        collection_name=f"chunks_{chunk_size}", query=qv, limit=k).points


def rag_prompt(question, hits):
    context = "\n---\n".join(h.payload["chunk_text"] for h in hits)
    return ("Ты помощник по внутренним регламентам компании. Отвечай ТОЛЬКО по контексту "
            "ниже, кратко. Если в контексте нет ответа — напиши «В регламентах не найдено».\n\n"
            f"Контекст:\n{context}\n\nВопрос: {question}")


def chat(prompt, model=LLM):
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=600, json={
        "model": model, "stream": False,
        "options": {"temperature": 0, "num_predict": 300, "repeat_penalty": 1.15},
        "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
