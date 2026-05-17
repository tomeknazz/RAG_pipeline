# poprzednie
# 10 audio do embedingow przez clipa i recznie piszemy podobienstwo


#  znalezc 10 dokumentow odnosnie jakiejs wiedyz (fotorgrafia)
# postawic qdrant
# stworzyc pipeline zeby przerobic na embeddingi
# pobrac llm z hugging face
# przetworzyć prompta i dodać RAG


from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import pipeline
import os, uuid

import fitz  # pymupdf
import os

llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",  # lub "google/flan-t5-base" (lżejszy)
    device_map="auto",
    max_new_tokens=512
)

docs = []
for fname in os.listdir("docs/"):
    fpath = os.path.join("docs", fname)

    if fname.endswith(".pdf"):
        # Czytaj PDF
        pdf = fitz.open(fpath)
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()
        docs.append({"text": text, "source": fname})

    elif fname.endswith(".txt"):
        # Czytaj TXT z poprawnym encodingiem
        with open(fpath, "r", encoding="utf-8") as f:
            docs.append({"text": f.read(), "source": fname})

print(f"Wczytano {len(docs)} dokumentów")
for d in docs:
    print(f"  {d['source']}: {len(d['text'])} znaków")

# 2. Podziel na chunki (np. co 500 znaków)
def chunk_text(text, size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])
    return chunks

all_chunks = []
for doc in docs:
    for chunk in chunk_text(doc["text"]):
        all_chunks.append({"text": chunk, "source": doc["source"]})

# 3. Stwórz embeddingi
model = SentenceTransformer("all-MiniLM-L6-v2")  # lekki, dobry model
embeddings = model.encode([c["text"] for c in all_chunks])

# 4. Zapisz do Qdrant
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
