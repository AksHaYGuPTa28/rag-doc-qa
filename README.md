# RAG-Based Enterprise Document Q&A Assistant

An end-to-end Retrieval-Augmented Generation pipeline for natural-language
Q&A over internal documents, with a streaming chat UI.

**Tech stack:** Python, FastAPI, LangChain (text splitting), OpenAI API,
Claude API, Gemini API, Chroma (vector DB), React / Next.js.

## Architecture

```
                 ingest                                  ask
   PDF/DOCX/TXT ────────► chunk (LangChain) ──► embed ──► Chroma vector store
                                                              │
   question ──► hybrid retrieval (vector + BM25, RRF fusion) ┘
                    │
                    ▼
             cross-encoder rerank (optional, graceful fallback)
                    │
                    ▼
        grounded prompt (context injection + citation + guardrails)
                    │
                    ▼
    OpenAI, Claude, or Gemini ──► streamed answer (SSE) + source citations
                        └─► or: function-calling → structured JSON answer
```

| Concern | Implementation |
|---|---|
| Ingestion | `pypdf` / `docx2txt` loaders, `langchain-text-splitters` recursive chunking |
| Embeddings | Pluggable: OpenAI (`text-embedding-3-small`) or local `sentence-transformers` (no API key needed) |
| Vector DB | Chroma, persisted to disk |
| Hybrid search | Dense vector similarity + BM25 keyword search, combined via Reciprocal Rank Fusion |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescoring of the fused shortlist, falls back to fusion order if `sentence-transformers` isn't installed |
| Generation | Pluggable: OpenAI, Claude, or Gemini, switchable via `LLM_PROVIDER` |
| Grounding | System prompt forces citation markers `[n]` and an explicit "not in the documents" fallback instead of hallucinating |
| Structured output | Function/tool calling forces a `{answer, citations, confidence}` JSON payload (`POST /api/chat/structured`) |
| Streaming | Server-Sent Events token-by-token, terminated by a `sources` event with citation metadata |
| Frontend | Next.js (App Router) chat interface: streaming responses, inline source citations, drag-free file upload |

## Project layout

```
backend/
  app/
    config.py            settings (env vars)
    models/schemas.py     pydantic models
    ingestion/            file loading + chunking
    embeddings/           OpenAI / local embedder (pluggable)
    llm/                  OpenAI / Claude / Gemini chat + structured output (pluggable)
    retrieval/            Chroma vector store, hybrid BM25+vector search, reranker
    rag/                  prompts + pipeline orchestration
    api/                  FastAPI routes (documents, chat)
    main.py               app entrypoint
frontend/
  app/                    Next.js pages
  components/             ChatWindow, MessageBubble, SourceCitations, DocumentUpload
  lib/                    API client (SSE streaming fetch), types
```

## Setup

### Backend

Requires Python 3.10+.

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env         # then fill in API keys
uvicorn app.main:app --reload --port 8000
```

`sentence-transformers` (used for local embeddings and cross-encoder
reranking) is a large install pulling in `torch`. The app runs fine without
it -- it just falls back to OpenAI-only embeddings and skips the rerank
step -- so feel free to remove it from `requirements.txt` if you want a
lighter install and don't need those two features.

Minimum required env vars in `backend/.env`:
- `LLM_PROVIDER` (`openai`, `anthropic`, or `gemini`) + the matching API key
- `EMBEDDING_PROVIDER` (`openai` needs `OPENAI_API_KEY`; `local` needs nothing)

### Frontend

```bash
cd frontend
npm install
copy .env.local.example .env.local
npm run dev
```

Open http://localhost:3000, upload a PDF/DOCX/TXT/MD file, then ask a
question about it.

## API

- `POST /api/documents/upload` -- multipart file upload, chunks + embeds + indexes it
- `GET /api/documents` -- list indexed documents
- `POST /api/chat/stream` -- SSE stream: `{query, history}` in, `token` events then a final `sources` event
- `POST /api/chat/structured` -- same input, returns a single `{answer, citations, confidence}` JSON object via function/tool calling
