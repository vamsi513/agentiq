# AgentIQ — Multi-Step Agentic Research Assistant

> An agentic AI system that autonomously routes queries between document retrieval and live web search to generate grounded, cited answers.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-purple?logo=chainlink&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-red?logo=streamlit&logoColor=white)

---

## 🚀 Live Demo

**[AgentIQ on Streamlit Cloud](https://agentiq.streamlit.app)**

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │           User Query                 │
                        └──────────────┬──────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────┐
                        │         Router Node                  │
                        │   (LLM classifies query intent)      │
                        └──────┬───────────────┬──────────────┘
                               │               │              │
              ┌────────────────▼──┐   ┌────────▼────────┐  ┌─▼──────────────┐
              │  FAISS Retrieval  │   │  Tavily Web     │  │  Direct LLM    │
              │  (local AI/ML     │   │  Search         │  │  (GPT-4o-mini  │
              │   corpus)         │   │  (live results) │  │   knowledge)   │
              └────────────────┬──┘   └────────┬────────┘  └─┬──────────────┘
                               │               │              │
                        ┌──────▼───────────────▼──────────────▼──────┐
                        │              Generator Node                  │
                        │   (synthesises context → cited answer)       │
                        └──────────────────┬──────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │         Streamed Response + Citations        │
                        │         (via FastAPI SSE / Streamlit)        │
                        └─────────────────────────────────────────────┘

  Memory: MemorySaver checkpoints every turn → full multi-turn history per session
```

---

## Features

- **Multi-step agentic reasoning** with LangGraph StateGraph — router, retriever, web search, and generator nodes wired with conditional edges
- **Hybrid retrieval** — semantic FAISS search over a local AI/ML research corpus + live Tavily web search for current information
- **Persistent conversation memory** with LangGraph MemorySaver checkpointing across all turns in a session
- **Real-time streaming responses** via FastAPI Server-Sent Events (SSE) with token-level output
- **RAGAS evaluation** — answer relevance **0.84**, faithfulness **0.91** across 50 multi-hop QA pairs
- **Fully deployed** on Streamlit Cloud with a dark-themed chat interface, route badges, and collapsible source citations

---

## Tech Stack

| Category | Technology |
|---|---|
| Agent Orchestration | LangGraph 0.2.28 |
| LLM Abstraction | LangChain 0.3.7 |
| Language Model | OpenAI GPT-4o-mini |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Vector Store | FAISS (faiss-cpu 1.9.0) |
| Web Search | Tavily Search API |
| Backend API | FastAPI 0.115.4 + uvicorn |
| Streaming | Server-Sent Events (SSE) |
| Data Validation | Pydantic v2 |
| Frontend | Streamlit 1.40.0 |
| Evaluation | RAGAS 0.1.21 |
| Testing | pytest 8.3.3 |
| Language | Python 3.11 |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/vamsi513/agentiq.git
cd agentiq
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. The FAISS index is built automatically on first run (~30 seconds).

### 6. (Optional) Run the FastAPI backend separately

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`.

### 7. Run tests

```bash
pytest tests/ -v
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here
TAVILY_API_KEY=tvly-your-tavily-api-key-here

# Optional (defaults shown)
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_TOKENS=1024
TEMPERATURE=0.2
TOP_K_RETRIEVAL=5
MAX_WEB_RESULTS=5
LOG_LEVEL=INFO
```

Get API keys:
- OpenAI: https://platform.openai.com/api-keys
- Tavily: https://tavily.com (free tier: 1,000 searches/month)

---

## Project Structure

```
agentiq/
├── app.py                        # Streamlit frontend entry point
├── requirements.txt              # All dependencies with pinned versions
├── .env.example                  # Environment variables template
├── .gitignore
├── README.md
├── config.py                     # Central config — loads from .env
├── agent/
│   ├── __init__.py
│   ├── graph.py                  # LangGraph StateGraph with conditional routing
│   ├── nodes.py                  # Router, retriever, web_search, generator nodes
│   ├── state.py                  # AgentState TypedDict definition
│   └── memory.py                 # MemorySaver setup and thread management
├── retrieval/
│   ├── __init__.py
│   ├── vectorstore.py            # FAISS index build, save, load, query
│   ├── embeddings.py             # sentence-transformers wrapper
│   └── documents/
│       └── sample_docs.txt       # 10 AI/ML research documents
├── tools/
│   ├── __init__.py
│   └── web_search.py             # Tavily API wrapper with error handling
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app — /chat, /chat/stream, /health
│   ├── schemas.py                # Pydantic request/response models
│   └── streaming.py              # SSE streaming helper
├── evaluation/
│   ├── __init__.py
│   ├── eval_runner.py            # RAGAS evaluation pipeline
│   ├── test_queries.json         # 50 test question/answer pairs
│   └── evaluation_results.json   # RAGAS results (committed)
└── tests/
    ├── __init__.py
    ├── test_agent.py             # Unit tests for graph nodes
    ├── test_retrieval.py         # Unit tests for vectorstore
    └── test_tools.py             # Unit tests for web search tool
```

---

## Evaluation Results

Evaluated on 50 curated AI/ML multi-hop question-answer pairs using RAGAS 0.1.21.

| Metric | Score |
|---|---|
| Answer Relevancy | **0.84** |
| Faithfulness | **0.91** |
| Queries Evaluated | **50** |
| Avg Latency | **187ms** |
| Min Latency | 98ms |
| Max Latency | 413ms |

### Route distribution across 50 queries

| Route | Count | % |
|---|---|---|
| 📄 Document Retrieval | 25 | 50% |
| 🧠 Direct LLM | 22 | 44% |
| 🌐 Web Search | 3 | 6% |

### Run evaluation yourself

```bash
python -m evaluation.eval_runner
```

Results are written to `evaluation/evaluation_results.json`.

---

## How It Works

### 1. Router Node
The router receives the user query and uses GPT-4o-mini with a strict classification prompt to decide which tool to invoke. It returns exactly one of three routing decisions: `retrieval`, `web_search`, or `direct`. Unknown or malformed responses are sanitised to `retrieval` as a safe default.

### 2. Retriever Node
When routing to `retrieval`, this node queries the FAISS vector index built from the local AI/ML research corpus. The query is encoded with `all-MiniLM-L6-v2`, L2-normalised, and searched against the index using inner-product (cosine) similarity. The top-5 chunks are returned with their similarity scores and metadata.

### 3. Web Search Node
When routing to `web_search`, this node calls the Tavily Search API with `search_depth="advanced"` and returns up to 5 results. If Tavily is unavailable, rate-limited, or misconfigured, the node returns a fallback result and the generator answers from LLM knowledge — the app never crashes.

### 4. Generator Node
The generator receives the assembled context (from retrieval or web search) and the full conversation history. It constructs a prompt that instructs GPT-4o-mini to answer grounded in the provided context with inline citations. Conversation history from all previous turns is included for multi-turn coherence.

### 5. Memory (MemorySaver)
Every graph run is checkpointed by LangGraph's MemorySaver under a stable `thread_id` (the session ID). This means the agent remembers everything said earlier in the session without any extra code — the full state graph is restored on each invocation.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + API key status |
| `POST` | `/chat` | Full JSON response (non-streaming) |
| `POST` | `/chat/stream` | Token-by-token SSE stream |

### Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "session_id": "my-session-1"}'
```

---

## Deployment — Streamlit Cloud

1. Push repository to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file: `app.py`
5. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-your-key"
   TAVILY_API_KEY = "tvly-your-key"
   ```
6. Click **Deploy**

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built by [Vamsi](https://github.com/vamsi513) as a production-quality agentic AI portfolio project.

---

*Deployed on [Streamlit Cloud](https://agentiq.streamlit.app) · Source on [GitHub](https://github.com/vamsi513/agentiq)*
