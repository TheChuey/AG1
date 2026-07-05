import os
import subprocess
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

class ModelManager:
    """Detect models installed via Ollama or other local providers.
    Currently supports scanning the Ollama models directory (~/.ollama/models).
    Returns a list of model identifiers.
    """


    @staticmethod
    def get_installed_models() -> list[str]:
        """Return a list of installed model identifiers.
        Tries the Ollama models directory first, filtering out internal folders.
        Falls back to `ollama list --format json` if the CLI is available.
        """
        models_dir = os.path.expanduser("~/.ollama/models")
        if os.path.isdir(models_dir):
            try:
                candidates = []
                for name in os.listdir(models_dir):
                    path = os.path.join(models_dir, name)
                    if os.path.isdir(path) and name not in ("blobs", "manifests"):
                        if any(fname.startswith("modelfile") or fname.endswith(".gguf") for fname in os.listdir(path)):
                            candidates.append(name)
                if candidates:
                    return candidates
            except Exception:
                pass
        try:
            result = subprocess.run(["ollama", "list", "--format", "json"], capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            return [item.get("model", "").split(":")[0] for item in data if isinstance(item, dict) and "model" in item]
        except Exception:
            return []


from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from chatbot.bot import ChatBot
from chatbot.config import Config
from chatbot.rag import KnowledgeManager
from chatbot.utils.code_executor import CodeExecutor
from chatbot.utils.log_viewer import LogViewer
from chatbot.pipelines.research import ResearchPipeline



config = Config(
    system_prompt_path=os.path.join(os.path.dirname(BASE_DIR), "prompts", "system_prompt.md"),
    chroma_persist_dir=os.path.join(BASE_DIR, "resources", "chroma_db"),
)

bot = ChatBot(config)
knowledge = KnowledgeManager(config)
code_exec = CodeExecutor()
log_viewer = LogViewer(log_path=os.path.join(BASE_DIR, "resources", "chat_logs.jsonl"))
research_pipeline = ResearchPipeline(model_name=config.model)

prompt_loaded = False
prompt_path = config.system_prompt_path
if os.path.exists(prompt_path):
    bot.load_prompt(prompt_path)
    prompt_loaded = True

app = FastAPI(title="AI Chatbot Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
# Serve frontend static files under /static
app.mount("/static", StaticFiles(directory=os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))), name="static")

# Root endpoint to serve index.html
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "index.html")))

# --- Models ---

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    agent: Optional[str] = None
    session_id: Optional[str] = None

class ResearchRequest(BaseModel):
    query: str

class SwitchModelRequest(BaseModel):
    model: str
    provider: Optional[str] = None

# --- Health endpoint ---

@app.get("/api/health")
async def health():
    try:
        bot.llm.send_request("ping")
        return {"status": "ok", "model": config.model}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})

# --- Endpoints ---

@app.get("/api/models/installed")
async def get_installed_models():
    """Return a list of locally installed LLM model IDs (e.g., Ollama)."""
    models = ModelManager.get_installed_models()
    if not models:
        models = ["qwen2.5-coder", "llama3.2", "mistral"]
    return {"installed_models": models}

# Existing models endpoint remains unchanged
@app.get("/api/models")
async def get_models():
    return {
        "models": [
            {"id": "qwen2.5-coder", "name": "Qwen 2.5 Coder", "provider": "ollama"},
            {"id": "llama3.2", "name": "Llama 3.2", "provider": "ollama"},
            {"id": "mistral", "name": "Mistral", "provider": "ollama"},
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai"},
            {"id": "claude-3-5-sonnet", "name": "Claude 3.5 Sonnet", "provider": "anthropic"},
        ]
    }


@app.get("/api/config")
async def get_config():
    return {
        "provider": config.provider,
        "model": config.model,
        "temperature": config.temperature,
        "prompt_loaded": prompt_loaded,
    }

@app.post("/api/chat")
async def chat(data: ChatRequest):
    if not prompt_loaded:
        bot.load_prompt_text("You are a helpful AI assistant.")
    if data.model and data.model != config.model:
        bot.llm.switch_model(data.model)
        config.model = data.model
    try:
        response = bot.chat(data.message, session_id=data.session_id)
    except Exception as e:
        # Log the error and return a JSON error payload so the frontend can parse it.
        import traceback, sys
        tb = traceback.format_exc()
        print(f"[Chat Endpoint Error] {e}\n{tb}", file=sys.stderr)
        return JSONResponse(status_code=500, content={"error": str(e)})
    log_viewer.append_entry({
        "timestamp": str(__import__("datetime").datetime.now()),
        "message": data.message,
        "reply": response,
        "model": config.model,
    })
    return {"reply": response, "session_id": bot.current_session_id}

@app.post("/api/chat/reset")
async def reset_chat():
    bot.reset()
    return {"status": "reset", "session_id": bot.current_session_id}

@app.post("/api/model/switch")
async def switch_model(data: SwitchModelRequest):
    bot.llm.switch_model(data.model, provider=data.provider)
    config.model = data.model
    if data.provider:
        config.provider = data.provider
    return {"status": "switched", "model": data.model, "provider": data.provider or config.provider}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    save_path = os.path.join(BASE_DIR, "resources", file.filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    if file.filename.lower().endswith(".pdf"):
        result = knowledge.load_pdf(save_path)
    else:
        text = content.decode("utf-8")
        knowledge.load_texts([text], metadatas=[{"filename": file.filename}])
        result = f"Loaded text file: {file.filename}"
    return {"reply": result}

@app.post("/api/rag/add")
async def rag_add(data: dict):
    """Add arbitrary text (e.g., research report) to the RAG knowledge base.
    Expects JSON with a `content` field (string)."""
    content = data.get("content")
    if not content:
        raise HTTPException(status_code=400, detail="Missing 'content' in request.")
    # Load as a synthetic document
    knowledge.load_texts([content], metadatas=[{"filename": "research.txt"}])
    return {"reply": "Research content added to RAG knowledge base."}

@app.post("/api/rag/query")
async def rag_query(data: ChatRequest):
    if not knowledge.is_ready():
        raise HTTPException(status_code=400, detail="No documents loaded. Upload a file first.")
    response = knowledge.query(data.message)
    return {"reply": response, "source": "RAG"}

@app.post("/api/mode/set")
async def set_mode(data: dict):
    """Set the chatbot mode: normal, business, or thinking"""
    mode = data.get("mode", "normal")
    if mode not in ("normal", "business", "thinking"):
        raise HTTPException(status_code=400, detail="Invalid mode")
    bot.set_mode(mode)
    return {"status": "mode set", "mode": mode}


@app.post("/api/research")
async def research(data: ResearchRequest):
    result = research_pipeline.run(data.query)
    report = result.get("final_report", "No results")
    urls = result.get("discovered_urls", [])
    return {
        "report": report,
        "urls": urls,
        "url_count": len(urls),
    }



@app.post("/api/code/execute")
async def execute_code(data: ChatRequest):
    result = code_exec.execute(data.message)
    return result

@app.get("/api/logs")
async def get_logs():
    return {"entries": log_viewer.read_entries()}

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
