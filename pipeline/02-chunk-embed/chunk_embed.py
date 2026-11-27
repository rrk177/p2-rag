import argparse

from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384


def chunk(text, size, overlap):
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--overlap", type=int, default=40)
    args = ap.parse_args()
    coll = f"chunks_{args.chunk_size}"

    model = SentenceTransformer(MODEL_NAME)
    mc = Minio("minio:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
    qd = QdrantClient(url="http://qdrant:6333")

    if qd.collection_exists(coll):
        qd.delete_collection(coll)
    qd.create_collection(
        coll,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )

    names = sorted(o.object_name for o in mc.list_objects("rag", prefix="documents/", recursive=True)
                   if o.object_name.endswith(".md"))
    if not names:
        print("[chunk] в MinIO нет документов — сначала стадии 0 и 1")
        return

    points, pid = [], 1
    for name in names:
        file = name.split("/")[-1]
        doc_id = int(file.split("-")[0])
        text = mc.get_object("rag", name).read().decode("utf-8")
        pieces = chunk(text, args.chunk_size, args.overlap)
        vectors = model.encode(pieces)
        for i, (piece, vec) in enumerate(zip(pieces, vectors)):
            points.append(PointStruct(id=pid, vector=vec.tolist(), payload={
                "doc_id": doc_id, "source": file, "chunk_index": i,
                "chunk_text": piece, "num_chars": len(piece),
            }))
            pid += 1
        print(f"[chunk] {file}: {len(pieces)} чанков")

    qd.upsert(collection_name=coll, points=points)
    print(f"[chunk] итого {len(points)} чанков -> Qdrant '{coll}' (cosine, HNSW m=16)")


if __name__ == "__main__":
    main()
