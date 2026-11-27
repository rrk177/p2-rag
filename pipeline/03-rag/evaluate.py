import argparse
import csv
from pathlib import Path

from common import search

GOLD = Path(__file__).resolve().parents[2] / "data" / "gold.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, nargs="+", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--recall-at", type=int, default=3)
    a = ap.parse_args()
    gold = list(csv.DictReader(GOLD.open(encoding="utf-8")))

    summary = []
    for size in a.chunk_size:
        hits, mrr = 0, 0.0
        print(f"\n=== chunks_{size} ===")
        for g in gold:
            got = [h.payload["doc_id"] for h in search(g["question"], size, a.k)]
            want = int(g["doc_id"])
            hits += want in got[:a.recall_at]
            rank = next((i + 1 for i, d in enumerate(got) if d == want), None)
            mrr += 1 / rank if rank else 0
            print(f"Q: {g['question'][:60]:<60} gold={want} top-{a.k}={got}")
        n = len(gold)
        summary.append((size, hits, n, mrr / n))
        print(f"chunk_size={size}: recall@{a.recall_at} = {hits}/{n} = {hits / n:.2f}, MRR = {mrr / n:.3f}")

    print("\n| chunk_size | recall@%d | MRR |\n|---|---|---|" % a.recall_at)
    for size, h, n, m in summary:
        print(f"| {size} | {h}/{n} = {h / n:.2f} | {m:.3f} |")


if __name__ == "__main__":
    main()
