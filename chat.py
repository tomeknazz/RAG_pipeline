import phoenix as px
from phoenix.otel import register
from agno.team import Team
from agno.models.ollama import Ollama
from agents import sprzet_agent, technika_agent, historia_agent, photo_agent

# ── 1. Uruchom Phoenix lokalnie ──────────────────────────
px.launch_app()  # http://localhost:6006

register(
    project_name="photo-multi-agent-rag",
    auto_instrument=True  # ← cała magia w jednej linii
)

# ── 2. Zespół agentów z wbudowanym routerem ──────────────
team = Team(
    name="FotografiaTeam",
    mode="route",  # router automatycznie wybiera agenta
    model=Ollama(id="llama3.2"),
    members=[
        sprzet_agent,
        technika_agent,
        historia_agent,
        photo_agent,
    ],
    instructions="""
    Jesteś routerem. Przekazuj pytania do odpowiedniego agenta:
    - SprzętAgent: Wywrotka, Ciężki sprzęt, wywrotka
    - TechnikaAgent: CNC, Roboty, technika
    - HistoriaAgent: historia bitew, wojna światowa
    - PhotoAgent: fotografia, ISO, zdjęcia
    """,
    show_members_responses=True,
)

# ── 3. Pętla czatu ───────────────────────────────────────
print("\n🎞️  Multi-Agent RAG (AGNO) gotowy!")
print("📊 Phoenix: http://localhost:6006")
print("Wpisz pytanie lub 'quit'\n")

while True:
    question = input("Ty: ").strip()
    if not question:
        continue
    if question.lower() in ("quit", "exit", "q"):
        break

    team.print_response(question, stream=True)
    print("-" * 60)
