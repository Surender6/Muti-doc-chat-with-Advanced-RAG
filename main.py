from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.model.models import UploadResponse, ChatRequest,ChatResponse
from multi_doc_chat.utils.document_ops import FastAPIFileAdapter
from multi_doc_chat.utils.model_loader import ModelLoader


## fastapi initialization

app = FastAPI(title="MultiDocChat", version="0.1.0")

# CORS (optional for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static and templates

BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# simple in-memory chat history
SESSIONS: Dict[str, List[dict]] = {}

## routes
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/",response_class=HTMLResponse)
def home(request:Request) ->HTMLResponse:
    return templates.TemplateResponse("index.html",{"request":request})

import traceback

@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        wrapped_files = [FastAPIFileAdapter(f) for f in files]

        ingestor = ChatIngestor(use_session_dirs=True)
        session_id = ingestor.session_id

        ingestor.built_retriver(
            uploaded_files=wrapped_files,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        SESSIONS[session_id] = []

        return UploadResponse(
            session_id=session_id,
            indexed=True,
            message="Indexing complete with MMR"
        )

    except Exception as e:
        print("❌ INDEXING ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


import traceback
import os

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    message = req.message.strip()

    if not session_id or session_id not in SESSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired session_id. Re-upload documents."
        )

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    try:
        # 🔍 TEMP DEBUG — keep for now
        print("🔑 GROQ_API_KEY visible in /chat:", bool(os.getenv("GROQ_API_KEY")))
        print("📌 Session ID:", session_id)

        # build RAG and load retriever from persisted FAISS with MMR
        model_loader = ModelLoader()
        rag = ConversationalRAG(session_id=session_id,model_loader=model_loader)

        index_path = Path("faiss_index") / session_id

        rag.load_retriever_from_faiss(
            index_path=index_path,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # use simple in-memory history and convert to BaseMessage list
        simple = SESSIONS.get(session_id, [])
        lc_history = []

        for m in simple:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_history.append(AIMessage(content=content))

        answer = rag.invoke(message, chat_history=lc_history)

        # update history
        simple.append({"role": "user", "content": message})
        simple.append({"role": "assistant", "content": answer})
        SESSIONS[session_id] = simple

        return ChatResponse(answer=answer)

    except DocumentPortalException as e:
        print("\n❌ DOCUMENT PORTAL ERROR =================")
        print(e)
        traceback.print_exc()
        print("❌ END DOCUMENT PORTAL ERROR =============\n")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        print("\n❌ CHAT ERROR ============================")
        print("Exception type:", type(e))
        print("Exception message:", e)
        traceback.print_exc()
        print("❌ END CHAT ERROR ========================\n")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
