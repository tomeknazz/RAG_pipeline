import os
import uuid
import fitz
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTIONS = {
    "sprzet": "docs/sprzet/",
    "technika": "docs/technika/",
    "historia": "docs/historia/",
    "fotografia": "docs/foto/",
}


def chunk_text(text, size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks


def read_docs(folder):
    docs = []
    if not os.path.exists(folder):
        return docs
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if fname.endswith(".pdf"):
            pdf = fitz.open(fpath)
            text = "".join(p.get_text() for p in pdf)
            docs.append({"text": text, "source": fname})
        elif fname.endswith(".txt"):
            with open(fpath, encoding="utf-8") as f:
                docs.append({"text": f.read(), "source": fname})
    return docs


for name, folder in COLLECTIONS.items():
    docs = read_docs(folder)
    chunks = [
        {"text": c, "source": d["source"]}
        for d in docs
        for c in chunk_text(d["text"])
    ]
    if not chunks:
        print(f"⚠️  Brak dokumentów w {folder}")
        continue

    embeddings = model.encode([c["text"] for c in chunks])

    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    client.upsert(
        collection_name=name,
        points=[
            PointStruct(id=str(uuid.uuid4()), vector=e.tolist(), payload=c)
            for e, c in zip(embeddings, chunks)
        ]
    )
    print(f"[{name}] — {len(chunks)} chunków")
