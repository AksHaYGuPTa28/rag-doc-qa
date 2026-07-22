import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.dependencies import get_pipeline
from app.models.schemas import AnswerWithCitations, ChatRequest
from app.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/stream")
def stream_chat(request: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """Server-Sent Events: token deltas as they're generated, then a final
    'sources' event carrying the citations used to ground the answer."""
    token_stream, sources = pipeline.stream_answer(request.query, request.history)

    def event_generator():
        try:
            for token in token_stream:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        sources_payload = [s.model_dump() for s in sources]
        yield f"data: {json.dumps({'type': 'sources', 'content': sources_payload})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/structured", response_model=AnswerWithCitations)
def structured_chat(request: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """Non-streaming path: forces a machine-readable {answer, citations,
    confidence} JSON object via function/tool calling instead of free text."""
    return pipeline.answer_structured(request.query, request.history)
