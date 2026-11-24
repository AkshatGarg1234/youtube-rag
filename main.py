from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="YouTube Playlist RAG API",
    description="AI-powered YouTube video revision assistant",
    version="1.0.0",
)

app.include_router(router)