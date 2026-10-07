import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import build_rag_graph


graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph

    print("Starting RAG graph initialization...", flush=True)

    graph = build_rag_graph(
        index_name=os.getenv(
            "PINECONE_INDEX_NAME",
            # "rag-agentic-ai-minilm"
            "agentic-ai-index"
        )
    )

    print("RAG graph initialized successfully.", flush=True)

    yield

    print("Shutting down...", flush=True)


app = FastAPI(
    title="Agentic AI RAG API",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[str]
    confidence_score: float


@app.get("/")
async def root():
    return {"status": "Agentic AI RAG API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0
    }

    result = graph.invoke(initial_state)

    return QueryResponse(
        final_answer=result["answer"],
        retrieved_context=result["context"],
        confidence_score=result["score"]
    )