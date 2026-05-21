import sys

from fastapi.middleware.cors import CORSMiddleware

sys.path.append('./scripts_subagentes')
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

from langchain.messages import HumanMessage
from scripts_subagentes.main_agent import main_agent

app = FastAPI(title="Game Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatboxgamboxd.netlify.app/"],
    allow_methods=["https://chatboxgamboxd.netlify.app/"],
    allow_headers=["https://chatboxgamboxd.netlify.app/"],
)
# ── Schemas ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    analisis_final: str
    subagent_json: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await main_agent.ainvoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    output = result.get("structured_response")
    if output is None:
        raise HTTPException(status_code=500, detail="El agente no devolvió respuesta estructurada.")

    return ChatResponse(
        session_id=session_id,
        analisis_final=output.analisis_final,
        subagent_json=output.subagent_json if isinstance(output.subagent_json, dict) else {},
    )


@app.get("/state/{session_id}")
async def get_state(session_id: str):
    """Devuelve el GameState actual para una sesión."""
    config = {"configurable": {"thread_id": session_id}}
    snapshot = main_agent.get_state(config)

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")

    state = snapshot.values
    return {
        "session_id": session_id,
        "full_game_title": state.get("full_game_title"),
        "platform": state.get("platform"),
        "genres": state.get("genres", []),
        "tags": state.get("tags", []),
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Limpia la memoria de una sesión (útil para reiniciar la conversación)."""

    return {"status": "ok", "session_id": session_id}


@app.get("/health")
async def health():
    return {"status": "ok"}