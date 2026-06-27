"""Session 00b — Engineering Foundations: FastAPI + Claude + Docker + pgvector skeleton."""
import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

load_dotenv()

MAX_TOKENS = 1024


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    """Response body returned by the /chat endpoint."""

    reply: str


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="AgenticCourse Chat Skeleton")
    client = anthropic.Anthropic()

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        """Send a message to Claude and return the reply."""
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": req.message}],
        )
        return ChatResponse(reply=response.content[0].text)

    @app.get("/health")
    async def health() -> dict:
        """Return service liveness status."""
        return {"status": "ok"}

    return app


def main() -> None:
    """Start the development server."""
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
