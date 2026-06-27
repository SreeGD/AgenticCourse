# Session 00b — Engineering Foundations for AI

**Track:** 0 — Foundations (Optional)
**Duration:** ~2 hours
**Prerequisites:** None
**Status:** Not Started

---

## Why This Session Exists

Sessions 1–46 assume you already speak Python async, FastAPI, Docker, and vector databases
fluently. If any of those feel shaky, work through this session first. It is entirely optional for
engineers with production Python web-service experience.

---

## 1. Async Python Patterns

### The Event Loop in One Mental Model

```
Thread           Event Loop
  │                │
  │  await io()   │──► schedules coroutine
  │◄──────────────│       ↓
  │               │  runs other tasks while I/O is in-flight
  │               │       ↓
  │  resume ◄─────│  I/O completes; resumes original coroutine
```

Python's `asyncio` event loop is **single-threaded** but multiplexes I/O-bound work so the
thread is never idle waiting for the network. For LLM calls—which can take 2–30 seconds—this
is critical: one process can serve hundreds of simultaneous requests.

### Key Rules

| Situation | Use |
|-----------|-----|
| Calls an API or DB | `async def` + `await` |
| Pure CPU work | plain `def` (or `asyncio.run_in_executor`) |
| Fire-and-forget background task | `asyncio.create_task()` |
| Running from a script | `asyncio.run(main())` |

### Example — async vs sync Claude call

```python
import asyncio
import anthropic

# sync (blocks the thread for the whole round-trip)
def ask_sync(prompt: str) -> str:
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# async (yields control while waiting; preferred in FastAPI)
async def ask_async(prompt: str) -> str:
    client = anthropic.AsyncAnthropic()
    resp = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text
```

> **Rule:** Inside a FastAPI route handler, always use `async def` and `await`
> so the event loop can serve other requests during the LLM call.

---

## 2. FastAPI + Pydantic Anatomy

### Request Lifecycle

```
HTTP POST /chat
     │
     ▼
FastAPI deserialises JSON → validates against ChatRequest (Pydantic)
     │
     ▼  validation fails → 422 Unprocessable Entity (automatic)
     │  validation passes ↓
     ▼
async def chat(req: ChatRequest) → calls Claude → wraps in ChatResponse
     │
     ▼
FastAPI serialises ChatResponse → JSON response
```

### The Lab's Core Structure

```python
# 1. Schema layer — Pydantic enforces types and constraints
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)   # empty string → 422

class ChatResponse(BaseModel):
    reply: str

# 2. App factory — makes create_app() testable without spinning up a server
def create_app() -> FastAPI:
    app = FastAPI(title="AgenticCourse Chat Skeleton")
    client = anthropic.Anthropic()           # one client, reused per request

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": req.message}],
        )
        return ChatResponse(reply=response.content[0].text)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app
```

### Why `create_app()` Instead of a Module-Level `app`?

A module-level `app = FastAPI()` runs when Python imports the file. `create_app()` defers
construction so tests can patch `anthropic.Anthropic` **before** the client is instantiated:

```python
with patch("anthropic.Anthropic", return_value=mock_client):
    lab = _load_lab()                 # Anthropic() is called here, sees the mock
    client = TestClient(lab.create_app())
```

### Pydantic Field Constraints Cheat Sheet

| Constraint | Type | Example |
|------------|------|---------|
| `min_length=1` | `str` | non-empty string |
| `max_length=4096` | `str` | bounded input |
| `gt=0` | `int` / `float` | positive number |
| `ge=0, le=1` | `float` | probability range |
| `pattern=r"^[a-z]+"` | `str` | regex validation |

---

## 3. Docker Layer Diagram

Docker builds images in layers. Each `RUN`, `COPY`, and `ADD` instruction creates one layer.
Layers are cached; only layers **below** a changed line are rebuilt.

```
┌─────────────────────────────────────────────┐
│  Layer 4 — COPY labs/00b_*.py .             │  ← changes every code edit
├─────────────────────────────────────────────┤
│  Layer 3 — pip install fastapi uvicorn ...  │  ← changes when deps change
├─────────────────────────────────────────────┤
│  Layer 2 — apt-get install gcc              │  ← rarely changes
├─────────────────────────────────────────────┤
│  Layer 1 — python:3.11-slim (base image)   │  ← pull once
└─────────────────────────────────────────────┘
```

**Key insight:** Put `COPY` of application code **last**. That way a code-only change only
invalidates layer 4; layers 1–3 are served from cache and the rebuild takes seconds.

### Dockerfile.00b Explained

```dockerfile
FROM python:3.11-slim          # Layer 1 — minimal base
WORKDIR /app

RUN apt-get update \
    && apt-get install -y gcc \
    && rm -rf /var/lib/apt/lists/*   # Layer 2 — build tools; chained to minimise layers

RUN pip install fastapi uvicorn anthropic python-dotenv   # Layer 3 — deps

COPY labs/00b_engineering_foundations.py .   # Layer 4 — app code (cache invalidated here)

EXPOSE 8000
CMD ["python", "00b_engineering_foundations.py"]
```

---

## 4. pgvector Schema Snippet

pgvector extends PostgreSQL with a `vector` column type and approximate-nearest-neighbour
indexes (IVFFlat, HNSW). It is the simplest way to add semantic search to an existing
Postgres stack.

```sql
-- Enable the extension (once per database)
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table with a 1536-dim embedding column (OpenAI ada-002 / Cohere size)
-- For Anthropic voyage-3: 1024 dims
CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   vector(1024)     -- dimension must match your embedding model
);

-- HNSW index — fast approximate search, good recall at ef_construction=64
CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Semantic search: top-5 nearest neighbours to a query vector
SELECT id, content, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS score
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
```

> `<=>` is cosine distance. Use `<->` for L2 (Euclidean) or `<#>` for inner product.

### Why pgvector in docker-compose?

The `db` service in `docker-compose.00b.yml` runs `pgvector/pgvector:pg16`—a pre-built image
with the extension already compiled. No manual `CREATE EXTENSION` step needed beyond the SQL
above; the image initialises a default database named `agentic` with user `agentic`.

---

## 5. Run It

### Prerequisites

```bash
# Copy your Anthropic key into the project root
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

### Option A — Run locally (no Docker)

```bash
pip install fastapi uvicorn anthropic python-dotenv
python labs/00b_engineering_foundations.py
# Server starts at http://localhost:8000
```

Test the endpoint:

```bash
curl -s -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What is pgvector?"}' | python -m json.tool
```

### Option B — Run with Docker Compose

```bash
docker compose -f labs/docker/docker-compose.00b.yml up --build
```

This starts:
- `api` — FastAPI app on port 8000
- `db` — pgvector/Postgres on port 5432 (healthcheck ensures it is ready before API starts)

Check health:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Connect to Postgres to try the pgvector schema:

```bash
docker compose -f labs/docker/docker-compose.00b.yml exec db \
    psql -U agentic -d agentic \
    -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT '[1,2,3]'::vector;"
```

### Run the Tests

```bash
pytest tests/unit/test_00b_engineering_foundations.py -v
```

Expected output:

```
PASSED tests/unit/test_00b_engineering_foundations.py::test_chat_endpoint_returns_text
PASSED tests/unit/test_00b_engineering_foundations.py::test_chat_endpoint_rejects_empty_message
2 passed in 0.90s
```

---

## Key Takeaways

1. **Async by default in FastAPI** — `async def` route handlers let the event loop serve other
   requests while Claude is thinking.
2. **Pydantic validates at the boundary** — invalid input never reaches your business logic;
   FastAPI returns 422 automatically.
3. **Layer your Dockerfile** — dependencies before code; code last. Saves minutes per iteration.
4. **pgvector = Postgres + ANN search** — one `CREATE EXTENSION` away from semantic retrieval
   inside your existing Postgres stack.
5. **Factory pattern (`create_app`)** — makes your FastAPI app unit-testable without a live
   server or real API keys.

---

## Try This

After completing this session, extend the `/chat` endpoint to:
1. Accept an optional `system_prompt: str = ""` field in `ChatRequest`.
2. Pass it as the `system` parameter in the `client.messages.create()` call.
3. Add a test that verifies the system prompt reaches the mock client.
