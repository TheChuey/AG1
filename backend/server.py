import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
from moduleV2 import AgentSessionRegistry, AgentConversationLogger, ModelDiscoveryEngine, AgentTelemetryTracker, MultiAgentOrchestrationMatrix



config = Config(
    system_prompt_path=os.path.join(os.path.dirname(BASE_DIR), "prompts", "system_prompt.md"),
    chroma_persist_dir=os.path.join(BASE_DIR, "resources", "chroma_db"),
)

bot = ChatBot(config)
knowledge = KnowledgeManager(config)
code_exec = CodeExecutor()
log_viewer = LogViewer(log_path=os.path.join(BASE_DIR, "resources", "chat_logs.jsonl"))
research_pipeline = ResearchPipeline(model_name=config.model)

# Module-level management instances
session_data_mgr = AgentSessionRegistry(config_dir=os.path.join(BASE_DIR, "resources", "sessions"))
prompt_bridge = AgentConversationLogger(storage_dir=os.path.join(os.path.dirname(BASE_DIR), "prompts"))
llm_mgr = ModelDiscoveryEngine(host="localhost", port=11434)
chat_mgr = AgentConversationLogger(storage_dir=os.path.join(BASE_DIR, "resources"))

prompt_loaded = False
prompt_path = config.system_prompt_path
if os.path.exists(prompt_path):
    bot.load_prompt(prompt_path)
    prompt_loaded = True
    print(f"[Server] system prompt loaded from {prompt_path}")

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
    models = llm_mgr.fetch_configured_models()
    if models:
        return {"status": "ok", "model": config.model, "models_available": models}
    return JSONResponse(
        status_code=503,
        content={"status": "error", "detail": "Ollama not reachable via module LLMManager"},
    )

# --- Endpoints ---

@app.get("/api/models/installed")
async def get_installed_models():
    """Return a list of locally installed LLM model IDs (e.g., Ollama)."""
    models = ModelDiscoveryEngine().get_installed_models()
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
    agent_type = data.agent or "chat"
    print(f"[API] POST /api/chat | agent={agent_type} session={data.session_id} msg={data.message[:60]}")
    if not prompt_loaded:
        bot.load_prompt_text("You are a helpful AI assistant.")
    if data.model and data.model != config.model:
        bot.llm.switch_model(data.model)
        config.model = data.model
    sid = data.session_id or bot.current_session_id
    session_data_mgr.register_session_meta(sid, agent_type)
    try:
        response = bot.chat(data.message, session_id=data.session_id)
    except Exception as e:
        import traceback, sys
        tb = traceback.format_exc()
        print(f"[API] POST /api/chat ERROR: {e}\n{tb}", file=sys.stderr)
        return JSONResponse(status_code=500, content={"error": str(e)})
    log_viewer.append_entry({
        "timestamp": str(__import__("datetime").datetime.now()),
        "message": data.message,
        "reply": response,
        "model": config.model,
        "agent": agent_type,
    })
    print(f"[API] POST /api/chat OK | session={bot.current_session_id}")
    return {"reply": response, "session_id": bot.current_session_id}

@app.post("/api/chat/reset")
async def reset_chat():
    print(f"[API] POST /api/chat/reset")
    bot.reset()
    session_data_mgr.register_session_meta(bot.current_session_id, "chat")
    return {"status": "reset", "session_id": bot.current_session_id}

@app.post("/api/model/switch")
async def switch_model(data: SwitchModelRequest):
    print(f"[API] POST /api/model/switch | model={data.model} provider={data.provider}")
    bot.llm.switch_model(data.model, provider=data.provider)
    config.model = data.model
    if data.provider:
        config.provider = data.provider
    return {"status": "switched", "model": data.model, "provider": data.provider or config.provider}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    print(f"[API] POST /api/upload | file={file.filename}")
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
    print(f"[API] POST /api/upload OK | {result}")
    return {"reply": result}

@app.post("/api/rag/add")
async def rag_add(data: dict):
    print(f"[API] POST /api/rag/add")
    content = data.get("content")
    if not content:
        raise HTTPException(status_code=400, detail="Missing 'content' in request.")
    knowledge.load_texts([content], metadatas=[{"filename": "research.txt"}])
    return {"reply": "Research content added to RAG knowledge base."}

@app.post("/api/rag/query")
async def rag_query(data: ChatRequest):
    print(f"[API] POST /api/rag/query | msg={data.message[:60]}")
    if not knowledge.is_ready():
        raise HTTPException(status_code=400, detail="No documents loaded. Upload a file first.")
    response = knowledge.query(data.message)
    return {"reply": response, "source": "RAG"}

@app.post("/api/mode/set")
async def set_mode(data: dict):
    mode = data.get("mode", "normal")
    print(f"[API] POST /api/mode/set | mode={mode}")
    if mode not in ("normal", "business", "thinking"):
        raise HTTPException(status_code=400, detail="Invalid mode")
    bot.set_mode(mode)
    return {"status": "mode set", "mode": mode}


@app.post("/api/research")
async def research(data: ResearchRequest):
    print(f"[API] POST /api/research | query={data.query[:60]}")
    session_data_mgr.register_session_meta(f"research_{id(data)}", "research")
    result = research_pipeline.run(data.query)
    report = result.get("final_report", "No results")
    urls = result.get("discovered_urls", [])
    print(f"[API] POST /api/research OK | {len(urls)} URLs, report={len(report)} chars")
    return {
        "report": report,
        "urls": urls,
        "url_count": len(urls),
    }



@app.post("/api/code/execute")
async def execute_code(data: ChatRequest):
    print(f"[API] POST /api/code/execute | code={data.message[:60]}")
    result = code_exec.execute(data.message)
    print(f"[API] POST /api/code/execute | success={result.get('success')}")
    return result

@app.get("/api/logs")
async def get_logs():
    print(f"[API] GET /api/logs")
    entries = log_viewer.read_entries()
    print(f"[API] GET /api/logs | {len(entries)} entries")
    return {"entries": entries}

@app.get("/api/chats")
async def get_chats():
    print(f"[API] GET /api/chats")
    chats = chat_mgr.get_recent_chats_log()
    print(f"[API] GET /api/chats | {len(chats)} sessions")
    return {"chats": chats}

if __name__ == "__main__":
    print(f"[Server] starting FastAPI on 127.0.0.1:8000")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
