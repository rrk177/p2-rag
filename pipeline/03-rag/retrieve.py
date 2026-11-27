import argparse

from common import search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("question")
    a = ap.parse_args()
    print(f"Вопрос: {a.question}\n")
    for h in search(a.question, a.chunk_size, a.k):
        print(f"doc_id={h.payload['doc_id']}  {h.payload['source']}  sim={h.score:.3f}")
        print("  " + h.payload["chunk_text"][:180].replace("\n", " ") + "\n")


if __name__ == "__main__":
    main()
