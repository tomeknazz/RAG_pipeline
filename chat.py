from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from transformers import pipeline

print("Ładowanie modeli...")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient("localhost", port=6333)

llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)

def rag_query(question: str):
    query_vec = model.encode([question])[0].tolist()

    # Nowe API qdrant-client
    results = client.query_points(
        collection_name="photography",
        query=query_vec,
        limit=3
    ).points

    context = "\n\n".join([r.payload["text"] for r in results])
    sources = list(set([r.payload["source"] for r in results]))
    print(f"\n📚 Źródła: {', '.join(sources)}")

    prompt = f"""<|system|>
Jesteś ekspertem od fotografii. Odpowiadaj krótko i na temat, tylko na podstawie kontekstu.</s>
<|user|>
Kontekst: {context}

Pytanie: {question}</s>
<|assistant|>"""

    output = llm(prompt)[0]["generated_text"]
    response = output.split("<|assistant|>")[-1].strip()
    return response


# ── Pętla czatu ──────────────────────────────────────────
print("\n🎞️  Asystent fotograficzny gotowy!")
print("Wpisz pytanie lub 'quit' żeby wyjść.\n")

while True:
    question = input("Ty: ").strip()

    if not question:
        continue

    if question.lower() in ("quit", "exit", "q"):
        print("Do widzenia!")
        break

    print("\nAsystent: ", end="", flush=True)
    answer = rag_query(question)
    print(answer)
    print("-" * 60)