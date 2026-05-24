import phoenix as px
from agno.agent import Agent
from agno.models.ollama import Ollama
from phoenix.otel import register

from agents import sprzet_agent, technika_agent, historia_agent, photo_agent

px.launch_app()
register(project_name="multi-agent-rag", auto_instrument=True)

# Router agent
router_agent = Agent(
    name="RouterAgent",
    model=Ollama(id="llama3.2"),
    instructions="""Jesteś routerem. Twoim jedynym zadaniem jest wybrać właściwą kategorię dla pytania.

Dostępne kategorie:
- sprzet: ciężki sprzęt budowlany, spycharka, koparka, wywrotka, dźwig
- technika: CNC, roboty, maszyny, inżynieria, sterowniki, wiertła
- historia: historia, bitwy, wojny, daty historyczne, wydarzenia
- fotografia: aparaty, ISO, obiektywy, zdjęcia, ekspozycja, głębia ostrości

Odpowiedz TYLKO jednym słowem — nazwą kategorii. Nic więcej.""",
)

AGENTS = {
    "sprzet": sprzet_agent,
    "technika": technika_agent,
    "historia": historia_agent,
    "fotografia": photo_agent,
}


def route(question: str) -> str:
    response = router_agent.run(question)

    # Wyciągnij tekst z odpowiedzi
    if hasattr(response, "content"):
        answer = response.content.strip().lower()
    else:
        answer = str(response).strip().lower()

    # Sprawdź czy model zwrócił poprawną kategorię
    for key in AGENTS:
        if key in answer:
            return key


# ── Pętla czatu ──────────────────────────────────────────
print("\n🎞️  Multi-Agent RAG gotowy!")
print("📊 Phoenix: http://localhost:6006")
print("Wpisz pytanie lub 'quit'\n")

while True:
    question = input("Ty: ").strip()
    if not question:
        continue
    if question.lower() in ("quit", "exit", "q"):
        break

    agent_key = route(question)
    print(f"\n🔀 Router → [{agent_key}]")

    agent = AGENTS[agent_key]
    agent.print_response(question, stream=True)
    print("-" * 60)
