import argparse
import csv
from pathlib import Path

from common import LLM, chat, rag_prompt, search

ROOT = Path(__file__).resolve().parents[2]


def cell(t):
    return t.replace("|", "/").replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--n", type=int, default=6, help="сколько вопросов (каждый 3-й из gold — по одному на документ)")
    a = ap.parse_args()
    gold = list(csv.DictReader((ROOT / "data" / "gold.csv").open(encoding="utf-8")))[::3][:a.n]
    rows = ["| Вопрос | Без RAG | С RAG | Источники (doc_id) |", "|---|---|---|---|"]
    for g in gold:
        q = g["question"]
        no_rag = chat(q)
        hits = search(q, a.chunk_size)
        with_rag = chat(rag_prompt(q, hits))
        src = ", ".join(str(h.payload["doc_id"]) for h in hits)
        rows.append(f"| {cell(q)} | {cell(no_rag)} | {cell(with_rag)} | {src} (верный {g['doc_id']}) |")
        print(f"[compare] {q}")
    out = ROOT / "results" / "compare.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(f"# Без RAG / с RAG ({LLM}, chunks_{a.chunk_size})\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"[compare] -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
