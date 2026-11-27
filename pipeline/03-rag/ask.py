import argparse

from common import LLM, chat, rag_prompt, search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--model", default=LLM)
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("question")
    a = ap.parse_args()
    if a.no_rag:
        print(f"=== БЕЗ RAG ({a.model}) ===\n{chat(a.question, a.model)}")
        return
    hits = search(a.question, a.chunk_size, a.k)
    print(f"=== С RAG ({a.model}, chunks_{a.chunk_size}) ===\n{chat(rag_prompt(a.question, hits), a.model)}")
    print("\n--- источники ---")
    for h in hits:
        print(f"  doc_id={h.payload['doc_id']}  {h.payload['source']}  sim={h.score:.3f}")


if __name__ == "__main__":
    main()
