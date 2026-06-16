"""
Server FastAPI
Espone l'agent LangChain via HTTP per il frontend HTML.
"""

import os
import sys
import threading
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from agent import CustomerSupportAgent

# ─────────────────────────────────────────────
# AGENT GLOBALE
# ─────────────────────────────────────────────

agent_instance: Optional[CustomerSupportAgent] = None
agent_lock = threading.Lock() # perché agent.py ha stato condiviso non thread-safe, quindi così si mitiga quel problema


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inizializza l'agent all'avvio del server."""
    global agent_instance
    print("🔄 Inizializzo Customer Support Agent (caricamento knowledge base + ChromaDB)...")
    agent_instance = CustomerSupportAgent()
    print("✅ Agent pronto. Server in ascolto.")
    yield
    print("👋 Server in chiusura.")


app = FastAPI(
    title="AI Customer Support API",
    description="Backend AI per il supporto clienti",
    lifespan=lifespan,
)

# Permette al file HTML aperto in browser di chiamare il server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    client_name: Optional[str] = None 


class ChatResponse(BaseModel):
    risposta: str
    escalation: bool
    source: str  # RAG | CUSTOMER | ESCALATION | GENERAL | RATE_LIMIT
    tools_used: list[str]
    rate_limited: bool = False


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    """Serve direttamente il file HTML della demo."""
    html_path = os.path.join(os.path.dirname(__file__), "..", "demo.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "ok", "message": "Server attivo. Frontend non trovato."}


@app.get("/health")
def health():
    return {"status": "ok", "agent_loaded": agent_instance is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="Agent non ancora inizializzato")
    with agent_lock:
        try:
            result = agent_instance.ask(req.message, nome_cliente=req.client_name)
            return ChatResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")


@app.post("/reset")
def reset():
    """Resetta la memoria della conversazione."""
    if agent_instance is None:
        raise HTTPException(status_code=503, detail="Agent non ancora inizializzato")
    agent_instance.reset_memory()
    return {"status": "ok", "message": "Memoria resettata"}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
