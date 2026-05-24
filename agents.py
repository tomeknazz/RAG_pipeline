from agno.agent import Agent
from agno.models.ollama import Ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient("localhost", port=6333)


def get_context(collection: str, query: str) -> str:
    try:
        vec = embed_model.encode([query])[0].tolist()
        results = qdrant.query_points(
            collection_name=collection,
            query=vec,
            limit=3
        ).points
        if not results:
            return "Brak danych w bazie wiedzy."
        return "\n\n---\n\n".join(
            f"[{r.payload['source']}]\n{r.payload['text']}"
            for r in results
        )
    except Exception as e:
        return f"Błąd wyszukiwania: {e}"


def make_agent(name: str, role: str, collection: str, topic: str) -> Agent:
    class RagAgent(Agent):
        def run(self, *args, **kwargs):
            # Wyciągnij pytanie niezależnie od nazwy argumentu
            message = kwargs.get("input") or kwargs.get("message") or (args[0] if args else "")

            context = get_context(collection, str(message))
            self.instructions = f"""Jesteś {role}.
Odpowiadaj TYLKO na podstawie poniższego kontekstu.
Jeśli nie ma odpowiedzi w kontekście, powiedz że nie wiem.

=== KONTEKST ===
{context}
================"""
            return super().run(*args, **kwargs)

    return RagAgent(
        name=name,
        role=role,
        model=Ollama(id="llama3.2"),
    )


sprzet_agent = make_agent(
    name="SprzętAgent",
    role="Ekspert od sprzętu budowlanego",
    collection="sprzet",
    topic="ciężki sprzęt budowlany",
)

technika_agent = make_agent(
    name="TechnikaAgent",
    role="Ekspert od techniki",
    collection="technika",
    topic="technika i maszyny",
)

historia_agent = make_agent(
    name="HistoriaAgent",
    role="Historyk",
    collection="historia",
    topic="historia i bitwy",
)

photo_agent = make_agent(
    name="PhotoAgent",
    role="Ekspert od fotografii",
    collection="fotografia",
    topic="fotografia i zdjęcia",
)
