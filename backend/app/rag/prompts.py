from app.models.schemas import ChatMessage, ScoredChunk, SourceMeta

SYSTEM_PROMPT = """You are an enterprise document Q&A assistant. Answer the \
user's question using ONLY the numbered CONTEXT provided below.

Rules:
- Cite every factual claim with the matching bracketed marker, e.g. [1], [2], \
placed right after the sentence it supports.
- If the answer is not contained in the context, say plainly that the indexed \
documents don't cover it. Never guess or fall back on outside knowledge.
- Prefer quoting or closely paraphrasing the source text over inventing detail.
- Keep the answer concise and directly responsive to the question."""

# Shared JSON schema for the "structured answer" mode, where the LLM is forced
# via function/tool calling to return machine-readable citations instead of
# (or in addition to) a free-text streamed answer.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The grounded answer to the user's question, with inline [n] markers.",
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "marker": {"type": "integer", "description": "Matches the [n] marker used in the answer."},
                    "source": {"type": "string"},
                    "page": {"type": ["integer", "null"]},
                    "quote": {"type": "string", "description": "Short supporting quote from that source."},
                },
                "required": ["marker", "source", "quote"],
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["answer", "citations", "confidence"],
}


def build_context_block(scored_chunks: list[ScoredChunk]) -> str:
    lines = []
    for i, sc in enumerate(scored_chunks, start=1):
        page_str = f", p.{sc.chunk.page}" if sc.chunk.page else ""
        lines.append(f"[{i}] (source: {sc.chunk.source}{page_str})\n{sc.chunk.text}")
    return "\n\n".join(lines)


def build_messages(query: str, scored_chunks: list[ScoredChunk], history: list[ChatMessage]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": h.role, "content": h.content})

    user_content = f"CONTEXT:\n{build_context_block(scored_chunks)}\n\nQUESTION:\n{query}"
    messages.append({"role": "user", "content": user_content})
    return messages


def build_sources(scored_chunks: list[ScoredChunk]) -> list[SourceMeta]:
    sources = []
    for i, sc in enumerate(scored_chunks, start=1):
        preview = sc.chunk.text[:220] + ("..." if len(sc.chunk.text) > 220 else "")
        sources.append(SourceMeta(marker=i, source=sc.chunk.source, page=sc.chunk.page, preview=preview))
    return sources
