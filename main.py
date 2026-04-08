import os
import json
import uuid
import asyncio
from datetime import datetime
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()

# Cloud Run metadata
SERVICE_NAME = os.getenv("K_SERVICE", "ai-backend")
REVISION = os.getenv("K_REVISION", "local")
PORT = int(os.getenv("PORT", "8080"))

# OpenAI
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Initialize OpenAI client (will be None if API key not set)
try:
    client = OpenAI()  # reads OPENAI_API_KEY from env automatically
except Exception as e:
    print(f"Warning: OpenAI client initialization failed: {e}")
    client = None

# Static UI
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------- Session Storage ----------
SESSIONS_FILE = os.getenv("SESSIONS_FILE", "sessions.json")
_sessions: dict = {}


def _load_sessions():
    global _sessions
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r") as f:
                _sessions = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load sessions file: {e}")


def _save_sessions():
    try:
        with open(SESSIONS_FILE, "w") as f:
            json.dump(_sessions, f)
    except Exception as e:
        print(f"Warning: Could not save sessions file: {e}")


_load_sessions()


# ---------- Models ----------
Role = Literal["developer", "user", "assistant"]

class ChatMessage(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[ChatMessage] = Field(default_factory=list)
    model: Optional[str] = None
    instructions: Optional[str] = None  # optional extra developer instruction


class ChatResponse(BaseModel):
    reply: str
    model: str
    service: str
    revision: str


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    instructions: Optional[str] = None


class SessionChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    model: Optional[str] = None
    instructions: Optional[str] = None  # overrides session-level instructions


# ---------- System endpoints ----------
@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/status")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "revision": REVISION,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/version")
def version():
    return {"service": SERVICE_NAME, "revision": REVISION, "port": PORT}

@app.get("/ready")
def ready():
    checks = {
        "openai_configured": os.getenv("OPENAI_API_KEY") is not None,
        "static_files": os.path.exists("static"),
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


# ---------- UI ----------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Missing static/index.html in repo.")

@app.get("/status", response_class=HTMLResponse)
def status():
    try:
        with open("static/status.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Missing static/status.html in repo.")


# ---------- Test Endpoints ----------
@app.get("/test/stream")
async def test_stream():
    async def generate():
        for i in range(1, 11):
            yield f"data: {json.dumps({'chunk': i, 'text': f'Chunk {i}'})}\n\n"
            await asyncio.sleep(0.1)
        yield f"data: {json.dumps({'done': True})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/test/openai")
async def test_openai():
    async def generate():
        try:
            if client is None:
                yield f"data: {json.dumps({'error': 'OpenAI client not initialized. Check OPENAI_API_KEY.'})}\n\n"
                return
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "Count to 3"}],
                stream=True
            )
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield f"data: {json.dumps({'text': delta.content})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/test/whisper")
async def test_whisper(audio: UploadFile = File(...)):
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized. Check OPENAI_API_KEY.")
    try:
        # Save temp file
        temp_path = f"/tmp/{audio.filename}"
        with open(temp_path, "wb") as f:
            f.write(await audio.read())

        # Transcribe
        with open(temp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

        os.remove(temp_path)
        return {"transcript": transcript.text, "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Chat ----------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI client not initialized. Check OPENAI_API_KEY environment variable."
        )

    model = req.model or DEFAULT_MODEL

    # Build messages for Chat Completions API
    messages = []
    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})
    for m in req.history:
        # Map 'developer' role to 'system' for Chat API
        role = "system" if m.role == "developer" else m.role
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        reply = response.choices[0].message.content or ""
        return ChatResponse(
            reply=reply,
            model=model,
            service=SERVICE_NAME,
            revision=REVISION,
        )
    except Exception as e:
        # Keep it simple + actionable
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI call failed: {type(e).__name__}: {str(e)}"
        )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if client is None:
        async def error_stream():
            yield f"data: {json.dumps({'error': 'OpenAI client not initialized'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    model = req.model or DEFAULT_MODEL

    # Build messages for Chat Completions API
    messages = []
    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})
    for m in req.history:
        # Map 'developer' role to 'system' for Chat API
        role = "system" if m.role == "developer" else m.role
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    async def generate():
        try:
            print(f"[STREAM] Starting Chat Completions stream for model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

            chunk_count = 0
            for chunk in response:
                chunk_count += 1

                # Chat Completions streaming uses choices[0].delta.content
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        print(f"[STREAM] Chunk {chunk_count}: {delta.content}")
                        yield f"data: {json.dumps({'text': delta.content})}\n\n"

            print(f"[STREAM] Stream complete. Total chunks: {chunk_count}")
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            print(f"[STREAM] Error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------- Sessions ----------

@app.post("/sessions", status_code=201)
def create_session(req: CreateSessionRequest = None):
    if req is None:
        req = CreateSessionRequest()
    session_id = uuid.uuid4().hex[:8]
    now = datetime.utcnow().isoformat()
    _sessions[session_id] = {
        "id": session_id,
        "title": req.title or f"Session {session_id}",
        "instructions": req.instructions,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    _save_sessions()
    return _sessions[session_id]


@app.get("/sessions")
def list_sessions():
    return [
        {
            "id": s["id"],
            "title": s["title"],
            "message_count": len(s["messages"]),
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
        }
        for s in sorted(_sessions.values(), key=lambda x: x["updated_at"], reverse=True)
    ]


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return s


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    del _sessions[session_id]
    _save_sessions()
    return {"deleted": session_id}


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def session_chat(session_id: str, req: SessionChatRequest):
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized. Check OPENAI_API_KEY.")

    model = req.model or DEFAULT_MODEL
    instructions = req.instructions or s.get("instructions")

    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    for m in s["messages"]:
        role = "system" if m["role"] == "developer" else m["role"]
        messages.append({"role": role, "content": m["content"]})
    messages.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(model=model, messages=messages)
        reply = response.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {type(e).__name__}: {str(e)}")

    now = datetime.utcnow().isoformat()
    s["messages"].append({"role": "user", "content": req.message})
    s["messages"].append({"role": "assistant", "content": reply})
    s["updated_at"] = now
    _save_sessions()

    return ChatResponse(reply=reply, model=model, service=SERVICE_NAME, revision=REVISION)


@app.post("/sessions/{session_id}/chat/stream")
async def session_chat_stream(session_id: str, req: SessionChatRequest):
    s = _sessions.get(session_id)
    if not s:
        async def _err():
            yield f"data: {json.dumps({'error': f'Session not found'})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")
    if client is None:
        async def _err():
            yield f"data: {json.dumps({'error': 'OpenAI client not initialized'})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    model = req.model or DEFAULT_MODEL
    instructions = req.instructions or s.get("instructions")

    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    for m in s["messages"]:
        role = "system" if m["role"] == "developer" else m["role"]
        messages.append({"role": role, "content": m["content"]})
    messages.append({"role": "user", "content": req.message})

    # Save user message immediately so a concurrent device sees it
    s["messages"].append({"role": "user", "content": req.message})

    async def generate():
        chunks = []
        try:
            response = client.chat.completions.create(model=model, messages=messages, stream=True)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    chunks.append(text)
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if chunks:
                s["messages"].append({"role": "assistant", "content": "".join(chunks)})
                s["updated_at"] = datetime.utcnow().isoformat()
                _save_sessions()

    return StreamingResponse(generate(), media_type="text/event-stream")


# Optional: nicer 404
@app.exception_handler(404)
def not_found(_, __):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "hint": "Try /ui, /, /health, /version, or POST /chat"},
    )
