from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.config import get_settings
from app.dependencies import get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()  # build eagerly so misconfiguration (missing API keys) fails at startup
    yield


app = FastAPI(title="RAG Enterprise Document Q&A", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}
