import os
import json
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
            response = client.responses.create(
                model=DEFAULT_MODEL,
                input=[{"role": "user", "content": "Count to 3"}],
                stream=True
            )
            for chunk in response:
                if hasattr(chunk, 'output_text_delta') and chunk.output_text_delta:
                    yield f"data: {json.dumps({'text': chunk.output_text_delta})}\n\n"
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

    # Build input items for the Responses API (roles supported) :contentReference[oaicite:3]{index=3}
    input_items = []

    # Optional: a base developer instruction
    if req.instructions:
        input_items.append({"role": "developer", "content": req.instructions})

    # Add prior messages (client-side memory for now)
    for m in req.history:
        input_items.append({"role": m.role, "content": m.content})

    # Add the new user message
    input_items.append({"role": "user", "content": req.message})

    try:
        response = client.responses.create(
            model=model,
            input=input_items,
            tools=[{"type": "web_search_preview"}],  # Enable web search
        )
        reply = response.output_text or ""
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

    # Build input items
    input_items = []
    if req.instructions:
        input_items.append({"role": "developer", "content": req.instructions})
    for m in req.history:
        input_items.append({"role": m.role, "content": m.content})
    input_items.append({"role": "user", "content": req.message})

    async def generate():
        try:
            print(f"[STREAM] Starting stream for model: {model}")
            response = client.responses.create(
                model=model,
                input=input_items,
                stream=True,
                tools=[{"type": "web_search_preview"}],  # Enable web search
            )

            chunk_count = 0
            for chunk in response:
                chunk_count += 1
                print(f"[STREAM] Chunk {chunk_count}: {chunk}")
                print(f"[STREAM] Chunk type: {type(chunk)}")
                print(f"[STREAM] Chunk dir: {dir(chunk)}")

                # Try multiple possible attribute names
                text_delta = None
                if hasattr(chunk, 'output_text_delta'):
                    text_delta = chunk.output_text_delta
                    print(f"[STREAM] Found output_text_delta: {text_delta}")
                elif hasattr(chunk, 'delta'):
                    if hasattr(chunk.delta, 'content'):
                        text_delta = chunk.delta.content
                        print(f"[STREAM] Found delta.content: {text_delta}")
                    elif hasattr(chunk.delta, 'text'):
                        text_delta = chunk.delta.text
                        print(f"[STREAM] Found delta.text: {text_delta}")
                elif hasattr(chunk, 'content'):
                    text_delta = chunk.content
                    print(f"[STREAM] Found content: {text_delta}")
                elif hasattr(chunk, 'text'):
                    text_delta = chunk.text
                    print(f"[STREAM] Found text: {text_delta}")

                if text_delta:
                    print(f"[STREAM] Yielding text: {text_delta}")
                    yield f"data: {json.dumps({'text': text_delta})}\n\n"
                else:
                    print(f"[STREAM] No text found in chunk")

            print(f"[STREAM] Stream complete. Total chunks: {chunk_count}")
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            print(f"[STREAM] Error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Optional: nicer 404
@app.exception_handler(404)
def not_found(_, __):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "hint": "Try /ui, /, /health, /version, or POST /chat"},
    )
