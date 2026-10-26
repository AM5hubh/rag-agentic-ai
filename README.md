# Agentic AI RAG Chatbot

A document-grounded Retrieval-Augmented Generation (RAG) API for answering questions from `Ebook-Agentic-AI.pdf`. The application retrieves the five most relevant chunks from Pinecone, then sends only those chunks to the language model through a LangGraph workflow:

```text
Question -> retrieve -> generate -> response
```

The API returns the generated answer, the retrieved context chunks, and the retrieval average confidence score.

## Tech Stack

- Python 3.10+
- LangChain and LangGraph
- FastAPI and Uvicorn
- Pinecone for vector storage
- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- OpenRouter-compatible ChatOpenAI endpoint for generation
- PyPDF for PDF extraction

## Project Structure

```text
rag-agentic-ai/
├── data/
│   └── Ebook-Agentic-AI.pdf
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py
│   └── ingestion.py
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

## Prerequisites

Install the following before starting:

- Python 3.10 or newer
- A Pinecone account and API key
- An OpenRouter API key
- A Pinecone index with a dimension compatible with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) and cosine similarity

The PDF knowledge source must be available at:

```text
data/Ebook-Agentic-AI.pdf
```

## Setup

Run these commands from the repository root, `rag-agentic-ai/`.

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv venv
venv\Scripts\activate
```

Verify the Python version:

```powershell
python --version
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and provide real values:

```env
OPENAI_API_KEY=your_openrouter_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
```

`OPENAI_API_KEY` is the environment variable read by LangChain's `ChatOpenAI` client. In this project, requests are routed to the OpenRouter endpoint configured in `src/graph.py`.

Do not commit `.env` or any API keys.

## Ingestion

Ingestion is a one-time process, or should be rerun whenever the source PDF changes. It loads the PDF, splits it into 1,000-character chunks with 200-character overlap, creates Hugging Face embeddings, and writes the chunks to Pinecone.

Confirm that `data/Ebook-Agentic-AI.pdf` exists, then run:

```powershell
python -m src.ingestion
```

Expected output includes the number of loaded documents, chunks, and the Pinecone index receiving the vectors.

The Pinecone index name used by ingestion comes from `PINECONE_INDEX_NAME` in `.env`.

## Run the API

Start the FastAPI development server from the repository root:

```powershell
uvicorn app:app --reload
```

The API is available at:

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

The application exposes:

```text
POST /chat
```

Request body:

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

PowerShell test request:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What is Agentic AI according to the eBook?"}'
```

Example response shape:

```json
{
  "final_answer": "...",
  "retrieved_context": [
    "Relevant document chunk 1...",
    "Relevant document chunk 2..."
  ],
  "confidence_score": 0.82
}
```

## Grounding Behavior

The generator receives only the retrieved document chunks. For a question that is not supported by the eBook, it is instructed to return:

```text
I cannot answer based on the provided document.
```

Useful validation questions include:

```text
What is Agentic AI according to the eBook?
How do AI agents differ from traditional automation systems?
What role does memory play in Agentic AI workflows?
Who won the 2022 FIFA World Cup?
```

The last question should demonstrate the out-of-document refusal behavior.

## Troubleshooting

### `ModuleNotFoundError`

Make sure the virtual environment is active and dependencies are installed:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Pinecone index errors

Check that the index name in `.env` exactly matches an existing Pinecone index and that its dimension is compatible with the MiniLM embeddings.

### Missing API key errors

Confirm that `.env` is in the repository root and contains non-empty values for `OPENAI_API_KEY` and `PINECONE_API_KEY`.
