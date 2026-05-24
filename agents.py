from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools import Toolkit
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient("localhost", port=6333)

# ── Klasa zamiast dekoratora ─────────────────────────────
class RagToolkit(Toolkit):
    def __init__(self, collection: str):
        super().__init__(name=f"search_{collection}")
        self.collection = collection
        self.register(self.search)

    def search(self, query: str) -> str:
        """Przeszukaj bazę wiedzy i zwróć pasujące fragmenty."""
        vec = embed_model.encode([query])[0].tolist()
        results = qdrant.query_points(
            collection_name=self.collection,
            query=vec,
            limit=3
        ).points
        if not results:
            return "Brak wyników w bazie wiedzy."
        parts = [f"[{r.payload['source']}]\n{r.payload['text']}" for r in results]
        return "\n\n---\n\n".join(parts)

# ── Definicje agentów ────────────────────────────────────
model = Ollama(id="llama3.2")

sprzet_agent = Agent(
    name="SprzętAgent",
    role="Ekspert od sprzętu",
    model=model,
    tools=[RagToolkit("sprzet")],
    instructions="Odpowiadaj tylko na pytania o ciężki sprzęt budowlany. Zawsze szukaj w bazie wiedzy.",
)

technika_agent = Agent(
    name="TechnikaAgent",
    role="Ekspert od techniki",
    model=model,
    tools=[RagToolkit("technika")],
    instructions="Odpowiadaj na pytania o technikę. Zawsze szukaj w bazie wiedzy.",
)

historia_agent = Agent(
    name="HistoriaAgent",
    role="Historyk",
    model=model,
    tools=[RagToolkit("historia")],
    instructions="Odpowiadaj na pytania historyczne. Zawsze szukaj w bazie wiedzy.",
)

photo_agent = Agent(
    name="PhotoAgent",
    role="Ekspert od fotografii",
    model=model,
    tools=[RagToolkit("fotografia")],
    instructions="Odpowiadaj na pytania o fotografię. Zawsze szukaj w bazie wiedzy.",
)