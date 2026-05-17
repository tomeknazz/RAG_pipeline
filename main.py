import os
import uuid

import fitz  # pymupdf
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from transformers import pipeline

llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    max_new_tokens=512
)

docs = []
for fname in os.listdir("docs/"):
    fpath = os.path.join("docs", fname)

    if fname.endswith(".pdf"):
        pdf = fitz.open(fpath)
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()
        docs.append({"text": text, "source": fname})

    elif fname.endswith(".txt"):
        with open(fpath, "r", encoding="utf-8") as f:
            docs.append({"text": f.read(), "source": fname})

print(f"Wczytano {len(docs)} dokumentów")
for d in docs:
    print(f"  {d['source']}: {len(d['text'])} znaków")


def chunk_text(text, size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks


all_chunks = []
for doc in docs:
    for chunk in chunk_text(doc["text"]):
        all_chunks.append({"text": chunk, "source": doc["source"]})

model = SentenceTransformer("all-MiniLM-L6-v2")  # lekki, dobry model
embeddings = model.encode([c["text"] for c in all_chunks])

client = QdrantClient("localhost", port=6333)
client.recreate_collection(
    collection_name="photography",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

points = [
    PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload=chunk)
    for emb, chunk in zip(embeddings, all_chunks)
]
client.upsert(collection_name="photography", points=points)
print(f"Zaindeksowano {len(points)} chunków!")
