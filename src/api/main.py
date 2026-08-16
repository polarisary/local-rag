import os
import time
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Enterprise Local RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    folder_path: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/ingest")
def ingest(req: IngestRequest):
    from src.etl.pipeline import run_folder_ingest

    folder = req.folder_path or os.getenv("INGEST_FOLDER", "/app/test_data")
    try:
        stats = run_folder_ingest(folder)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}")
    return {"code": 0, "msg": "ingest finished", "data": stats}


@app.get("/api/v1/models")
def list_models():
    model_name = os.getenv("LLM_MODEL", "deepseek-chat")
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "local-rag",
            }
        ],
    }


class RagQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4


@app.post("/api/v1/rag/query")
def rag_query(req: RagQueryRequest):
    from src.rag.llm_chain import query_rag

    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    try:
        result = query_rag(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rag failed: {e}")
    return {"code": 0, "msg": "ok", "data": result}


@app.post("/api/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    from src.rag.llm_chain import query_rag

    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message found")
    user_query = str(user_msgs[-1].content)

    try:
        result = query_rag(user_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rag failed: {e}")

    answer = result.get("answer", "")
    references = result.get("references", [])

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model or os.getenv("LLM_MODEL", "deepseek-chat"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "extra": {
            "references": references,
        },
    }
