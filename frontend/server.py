from __future__ import annotations

import os
import sys
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
import asyncio

# Make sure project root is importable when running from frontend/
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import backend engine without modifying it
from src.engine.engine import Engine
from src.engine.deck import Deck
from src.models import AIModel, AskAgentIfWantsToSpeakEventPublic, AgentResponseToQuestioningEventPublic, ChancellorPlayPolicyEventPublic, VoteChancellorYesNoEventPublic, ChooseAgentToVoteOutEventPublic, PresidentPickChancellorEventPublic

app = FastAPI(title="Secret Hitler LLM API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    # Include "null" so file:// origins can fetch the API during local dev
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Very simple single-game registry for demo purposes
class GameState(BaseModel):
    game_id: str

# In-memory state
_engine: Engine | None = None
_game_id: str | None = None
_game_running: bool = False
_game_task: asyncio.Task | None = None


def _serialize_event(ev: Any) -> Dict[str, Any]:
    # Map Pydantic models to plain dicts + type tags for the UI
    d = ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
    if isinstance(ev, AskAgentIfWantsToSpeakEventPublic):
        d["event_type"] = "ask-to-speak"
    elif isinstance(ev, AgentResponseToQuestioningEventPublic):
        d["event_type"] = "answer"
    elif isinstance(ev, ChancellorPlayPolicyEventPublic):
        d["event_type"] = "chancellor-play-policy"
    elif isinstance(ev, VoteChancellorYesNoEventPublic):
        d["event_type"] = "vote-chancellor"
    elif isinstance(ev, ChooseAgentToVoteOutEventPublic):
        d["event_type"] = "choose-to-execute"
    elif isinstance(ev, PresidentPickChancellorEventPublic):
        d["event_type"] = "president-pick-chancellor"
    else:
        d["event_type"] = type(ev).__name__
    return d


@app.post("/api/create", response_model=GameState)
def api_create_game() -> GameState:
    global _engine, _game_id
    try:
        deck = Deck()
        # default five slots of models (can be None)
        ai_models = [
            AIModel.OPENAI_GPT_5,
            AIModel.OPENAI_GPT_5_MINI,
            AIModel.OPENAI_GPT_5_NANO,
            AIModel.OPENAI_GPT_4_1,
            None,
        ]
        print(f"Creating engine with ai_models: {ai_models}")
        _engine = Engine(
            deck=deck,
            ai_models=ai_models,
            fascist_policies_to_win=6,
            liberal_policies_to_win=5
        )
        print("Engine created successfully")
    except Exception as e:
        print(f"Error creating engine: {e}")
        raise
    
    # For demo purposes, assign a random Chancellor so we can see both titles
    agent_ids = list(_engine.agents_by_id.keys())
    current_president_id = _engine.president_rotation[_engine.current_president_idx]
    available_for_chancellor = [aid for aid in agent_ids if aid != current_president_id]
    if available_for_chancellor:
        _engine.current_chancellor_id = available_for_chancellor[0]
    
    _game_id = "local-demo"
    return GameState(game_id=_game_id)


@app.post("/api/{game_id}/start")
async def api_start_game(game_id: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    global _game_running, _game_task
    
    if _engine is None or _game_id != game_id:
        return {"ok": False, "error": "game not found"}
    
    if _game_running:
        return {"ok": False, "error": "game already running"}
    
    # Create queues for async communication
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    
    async def run_game_background():
        global _game_running
        try:
            print(f"Starting game {game_id} in background...")
            _game_running = True
            await _engine.run(input_queue, output_queue)
            print(f"Game {game_id} completed!")
        except Exception as e:
            print(f"Error running game: {e}")
        finally:
            _game_running = False
    
    # Start the game in the background
    _game_task = asyncio.create_task(run_game_background())
    
    return {"ok": True, "message": "Game started in background"}


@app.post("/api/{game_id}/discussion")
async def api_run_discussion(game_id: str) -> Dict[str, Any]:
    if _engine is None or _game_id != game_id:
        return {"ok": False, "error": "game not found"}
    
    # Create queues for async communication
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()
    
    try:
        # Run a single discourse round
        await _engine._discourse(input_queue, output_queue)
        return {"ok": True}
    except Exception as e:
        print(f"Error running discussion: {e}")
        return {"ok": False, "error": str(e)}


@app.get("/api/{game_id}/state")
def api_state(game_id: str) -> Dict[str, Any]:
    if _engine is None or _game_id != game_id:
        return {"ok": False, "error": "game not found"}

    liberal = _engine.liberal_policies_played
    fascist = _engine.fascist_policies_played
    events = [_serialize_event(ev) for ev in _engine.public_events]
    
    # Create player info with names and roles
    available_names = ["Alice", "Bob", "Charlie", "Diana", "You"]
    name_index = 0
    players = []
    
    for i, (agent_id, agent) in enumerate(_engine.agents_by_id.items()):
        current_president_id = _engine.president_rotation[_engine.current_president_idx]
        is_president = agent_id == current_president_id
        is_chancellor = agent_id == _engine.current_chancellor_id
        
        title_parts = []
        if is_president:
            title_parts.append("President")
        if is_chancellor:
            title_parts.append("Chancellor")
        
        title = " & ".join(title_parts) if title_parts else ""
        
        # Hitler gets special name and role label
        if agent.role.value == "hitler":
            player_name = "Hitler"
            role_label = "(Fascist)"
            is_you = False
        else:
            player_name = available_names[name_index] if name_index < len(available_names) else f"Player{name_index+1}"
            role_label = f"({agent.role.value.title()})"
            is_you = player_name == "You"
            name_index += 1
        
        players.append({
            "id": agent_id,
            "name": player_name,
            "role": role_label,
            "title": title,
            "is_you": is_you,
            "model": agent.ai_model
        })
    
    return {
        "ok": True,
        "game_id": _game_id,
        "liberal_policies": liberal,
        "fascist_policies": fascist,
        "liberal_policies_to_win": _engine.liberal_policies_to_win,
        "fascist_policies_to_win": _engine.fascist_policies_to_win,
        "events": events,
        "players": players,
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
