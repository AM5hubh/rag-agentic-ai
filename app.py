import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import build_rag_graph


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph

    logger.info("Starting RAG graph initialization...")

    try:
        index_name = os.getenv(
            "PINECONE_INDEX_NAME",
            "agentic-ai-index",
        )

        graph = build_rag_graph(index_name=index_name)

        logger.info("RAG graph initialized successfully.")

        yield

    except Exception:
        logger.exception("Failed to initialize the RAG graph.")
        raise

    finally:
        graph = None
        logger.info("RAG API shutting down.")


app = FastAPI(
    title="Agentic AI RAG API",
    description="Document-grounded question answering using RAG.",
    version="1.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Question about the Agentic AI ebook.",
    )


class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float


@app.get("/")
async def root():
    return {"status": "Agentic AI RAG API is running"}


@app.get("/health")
async def health():
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="RAG graph is not initialized.",
        )

    return {"status": "healthy"}


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="RAG graph is not initialized.",
        )

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    initial_state = {
        "question": query,
        "context": [],
        "answer": "",
        "score": 0.0,
    }

    try:
        result = graph.invoke(initial_state)

        return QueryResponse(
            final_answer=result.get(
                "answer",
                "No answer was generated.",
            ),
            retrieved_context=result.get("context", []),
            confidence_score=float(result.get("score", 0.0)),
        )

    except Exception as exc:
        logger.exception("Error processing RAG query.")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question.",
        ) from exc
