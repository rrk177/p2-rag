import hashlib
import json
import time
from pathlib import Path

from minio import Minio

ROOT = Path(__file__).resolve().parents[2]
RAW, MD = ROOT / "data" / "raw", ROOT / "data" / "md"
BUCKET = "rag"


def main():
    mc = Minio("minio:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
    if not mc.bucket_exists(BUCKET):
        mc.make_bucket(BUCKET)
        print(f"[load] создан bucket s3://{BUCKET}")

    manifest = []
    for md in sorted(MD.glob("[0-9][0-9]-*.md")):
        doc_id = int(md.name[:2])
        raw = next(RAW.glob(md.stem + ".*"))
        mc.fput_object(BUCKET, f"raw/{raw.name}", str(raw))
        mc.fput_object(BUCKET, f"documents/{md.name}", str(md))
        manifest.append({
            "doc_id": doc_id,
            "title": md.stem.split("-", 1)[1].replace("-", " "),
            "source_format": raw.suffix.lstrip(".").lower(),
            "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "raw_path": f"s3://{BUCKET}/raw/{raw.name}",
            "md_path": f"s3://{BUCKET}/documents/{md.name}",
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"[load] doc_id={doc_id} {raw.name} -> raw/, {md.name} -> documents/")

    out = Path(__file__).parent / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[load] готово: {len(manifest)} документов, реестр {out.name}")


if __name__ == "__main__":
    main()
