# Multi-Agent RAG System

A multi-agent RAG system built with **Agno**, **Qdrant**, **Ollama**, and **Arize Phoenix**. A router agent classifies user intent and delegates questions to specialized agents, each backed by its own knowledge base.

---

## Architecture

```
User Question
      ↓
RouterAgent (llama3.2)
      ↓
 ┌────┴─────────────────────────┐
 ▼          ▼         ▼         ▼
SprzętAgent  TechnikaAgent  HistoriaAgent  PhotoAgent
 ↓               ↓               ↓              ↓
Qdrant       Qdrant          Qdrant         Qdrant
(sprzet)    (technika)      (historia)    (fotografia)
      ↓
Arize Phoenix (observability)
```

- **RouterAgent** — an LLM-based agent that classifies intent and selects the right specialist
- **Specialist agents** — each has its own Qdrant collection and system prompt
- **Similarity fallback** — if the router returns an unexpected answer, the system picks the collection with the highest cosine similarity score
- **Arize Phoenix** — full observability: traces every span from routing to LLM response

---

## Stack

| Component | Technology |
|---|---|
| Agent framework | [Agno](https://agno.com) |
| Vector database | [Qdrant](https://qdrant.tech) |
| LLM (local) | [Ollama](https://ollama.com) + `llama3.2` |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| PDF parsing | PyMuPDF (`fitz`) |
| Observability | [Arize Phoenix](https://phoenix.arize.com) |

---

## Project Structure

```
├── docs/
│   ├── sprzet/       # Heavy machinery documents
│   ├── technika/     # Engineering & technology documents
│   ├── historia/     # History documents
│   └── foto/         # Photography documents
├── index.py          # Indexes PDFs/TXTs into Qdrant
├── agents.py         # Specialist agent definitions
├── chat.py           # Router agent + main chat loop
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tomeknazz/RAG_pipeline
cd RAG_pipeline
```

### 2. Install dependencies

```bash
pip install agno qdrant-client sentence-transformers pymupdf
pip install arize-phoenix openinference-instrumentation-agno
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
pip install transformers torch
```

### 3. Start Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest
```

### 4. Pull the LLM with Ollama

Download Ollama from [ollama.com](https://ollama.com/download), then:

```bash
ollama pull llama3.2
```

### 5. Add your documents

Place `.pdf` or `.txt` files into the appropriate folders:

```
docs/sprzet/      ← heavy machinery
docs/technika/    ← engineering
docs/historia/    ← history
docs/foto/        ← photography
```

### 6. Index documents into Qdrant

```bash
python index.py
```

### 7. Run the chat

```bash
python chat.py
```

Phoenix dashboard will be available at `http://localhost:6006`.

---

## How It Works

### Indexing (`index.py`)

1. Reads `.pdf` and `.txt` files from each `docs/` subfolder
2. Splits documents into 500-character chunks with 50-character overlap
3. Encodes chunks using `all-MiniLM-L6-v2`
4. Stores embeddings and metadata in Qdrant (one collection per category)

### Routing (`chat.py`)

1. User submits a question
2. `RouterAgent` (Agno + llama3.2) returns a single category name
3. If the response is invalid, a **similarity fallback** queries all Qdrant collections and picks the one with the highest score
4. The matched specialist agent is called

### RAG (`agents.py`)

Each specialist agent overrides `run()` to:

1. Encode the user question into a vector
2. Query its Qdrant collection for the top 3 most relevant chunks
3. Inject the retrieved context directly into its system prompt
4. Call the LLM with the enriched prompt — no tool calling required

---

## Observability with Arize Phoenix

All agent interactions are automatically traced via OpenInference instrumentation.

Open `http://localhost:6006` to see:

- **Traces** — full request tree: router → specialist agent → LLM
- **Spans** — individual steps with latency
- **Inputs / Outputs** — exact prompts and responses at each stage

---

## Extending the System

### Add a new knowledge domain

1. Create a new folder under `docs/`, e.g. `docs/prawo/`
2. Add it to `COLLECTIONS` in `index.py`:
   ```python
   "prawo": "docs/prawo/"
   ```
3. Re-run `python index.py`
4. Add a new agent in `agents.py`:
   ```python
   prawo_agent = make_agent(
       name="PrawoAgent",
       role="Ekspert od prawa",
       collection="prawo",
       topic="prawo i przepisy",
   )
   ```
5. Register it in `AGENTS` in `chat.py` and update the router instructions

### Switch the LLM

Replace `llama3.2` with any model supported by Ollama:

```python
model = Ollama(id="qwen2.5")   # better quality, ~4.7GB
model = Ollama(id="mistral")   # good balance, ~4.1GB
```

---

## Known Limitations

- `llama3.2` does not support tool calling — context is injected via system prompt instead
- Documents must be re-indexed after any changes to the `docs/` folder

---

## License

MIT
