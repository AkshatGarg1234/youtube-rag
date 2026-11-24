
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.retrieval.retriever import search_chunks
from app.generation.groq_llm import (
    answer_question,
    answer_question_stream,
    GenerationError,
)


router = APIRouter()


# --------------------------------------------------
# REQUEST SCHEMAS
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    language: str = Field(default="hinglish")
    code_language: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_language(language: str):

    if language not in {"english", "hinglish"}:
        raise HTTPException(
            status_code=400,
            detail="Language must be english or hinglish.",
        )


def validate_code_language(code_language: str | None):

    if code_language is not None:

        if code_language not in {"python", "cpp", "java"}:
            raise HTTPException(
                status_code=400,
                detail="Code language must be python, cpp, or java.",
            )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@router.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "YouTube Playlist RAG API",
    }


# --------------------------------------------------
# SEARCH ENDPOINT
# --------------------------------------------------

@router.post("/search")
def search(request: SearchRequest):

    try:
        results = search_chunks(
            question=request.question,
            top_k=request.top_k,
        )

        return {
            "question": request.question,
            "total_results": len(results),
            "results": results,
        }

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Search failed.",
        )


# --------------------------------------------------
# NORMAL ASK ENDPOINT
# --------------------------------------------------

@router.post("/ask")
def ask(request: AskRequest):

    validate_language(request.language)
    validate_code_language(request.code_language)

    try:

        answer = answer_question(
            question=request.question,
            top_k=request.top_k,
            language=request.language,
            code_language=request.code_language,
        )

        return {
            "question": request.question,
            "language": request.language,
            "code_language": request.code_language,
            "answer": answer,
        }

    except (ValueError, GenerationError) as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )


# --------------------------------------------------
# STREAMING ASK ENDPOINT
# --------------------------------------------------

@router.post("/ask/stream")
def ask_stream(request: AskRequest):

    validate_language(request.language)
    validate_code_language(request.code_language)

    def event_generator():

        try:

            for token in answer_question_stream(
                question=request.question,
                top_k=request.top_k,
                language=request.language,
                code_language=request.code_language,
            ):

                event = json.dumps(
                    {"token": token},
                    ensure_ascii=False,
                )

                yield f"data: {event}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as exc:

            error_event = json.dumps(
                {"error": str(exc)},
                ensure_ascii=False,
            )

            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
