from __future__ import annotations

import os
import sys
import uvicorn
import uuid
import asyncio
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict

# Make sure project root is importable when running from frontend/
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.engine.engine_api import EngineAPI
from src.engine.deck import Deck
from src.models import AIModel, AskAgentIfWantsToSpeakEventPublic, AgentResponseToQuestioningEventPublic, ChancellorPlayPolicyEventPublic, VoteChancellorYesNoEventPublic, ChooseAgentToVoteOutEventPublic, PresidentPickChancellorEventPublic

# FastAPI app setup
app = FastAPI(title="Secret Hitler LLM API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple models
class GameState(BaseModel):
    game_id: str

# Global state
_api = EngineAPI()
_game_id: str | None = None
_game_running = False


def _serialize_event(ev: Any) -> Dict[str, Any]:
    # Map Pydantic models to plain dicts + type tags for the UI
    d = ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
    
    # Don't replace agent IDs - the frontend JavaScript already has
    # a playerLookup mechanism that maps agent IDs to names
    
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
async def api_create_game() -> GameState:
    global _game_id, _game_running
    
    print("DEBUG: Frontend requested game creation")
    
    # End any existing game first
    if _game_id and _game_id in _api.games:
        print(f"DEBUG: Ending previous game {_game_id}")
        # Clean up previous game
        if _game_id in _api.tasks:
            _api.tasks[_game_id].cancel()
            del _api.tasks[_game_id]
        if _game_id in _api.engines:
            del _api.engines[_game_id]
        if _game_id in _api.games:
            del _api.games[_game_id]
        _game_running = False
    
    # Just generate a new game ID, but don't create the game yet
    _game_id = str(uuid.uuid4())
    
    print(f"DEBUG: Prepared game ID {_game_id} - waiting for user to click 'Start Game'")
    return GameState(game_id=_game_id)


@app.post("/api/{game_id}/start")
async def api_start_game(game_id: str) -> Dict[str, Any]:
    global _game_running
    
    if _game_id != game_id:
        return {"ok": False, "error": "game not found"}
    
    if _game_running:
        return {"ok": False, "error": "game already running"}
    
    print(f"DEBUG: User clicked 'Start Game' for {game_id}")
    
    # NOW create and start the actual game
    ai_models = [AIModel.OPENAI_GPT_5] * 5  # All AI players
    
    await _api.create(
        game_id=_game_id,
        deck=Deck(),
        ai_models=ai_models
    )
    
    _game_running = True
    print(f"DEBUG: Game {game_id} created and started - AI players are now playing")
    return {"ok": True, "message": "Game started"}


@app.post("/api/{game_id}/discussion")
async def api_run_discussion(game_id: str) -> Dict[str, Any]:
    # Discussion is handled automatically by the EngineAPI game loop
    return {"ok": True, "message": "Discussion handled by game loop"}


@app.get("/api/{game_id}/state")
def api_state(game_id: str) -> Dict[str, Any]:
    print(f"DEBUG: Frontend requesting state for game_id: {game_id}")
    print(f"DEBUG: Server has game_id: {_game_id}")
    print(f"DEBUG: Available games in API: {list(_api.games.keys())}")
    
    # If no games exist, return error
    if not _api.games:
        return {"ok": False, "error": "no game exists"}
    
    # Use the first available game (for demo purposes)
    actual_game_id = list(_api.games.keys())[0]
    engine = _api.engines[actual_game_id]  # Use engines dict, not games dict
    print(f"DEBUG: Using game {actual_game_id} for frontend display")
    
    # Get public events from engine
    events = [_serialize_event(ev) for ev in engine.public_events]
    
    # Find the most recent chancellor from events (since engine state might be cleared)
    most_recent_chancellor = None
    for event in reversed(engine.public_events):
        if hasattr(event, 'chancellor_id') and event.chancellor_id:
            most_recent_chancellor = event.chancellor_id
            break
    
    print(f"DEBUG: Engine current_chancellor_id: {engine.current_chancellor_id}")
    print(f"DEBUG: Most recent chancellor from events: {most_recent_chancellor}")
    
    # Create player info
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    players = []
    
    for i, (agent_id, agent) in enumerate(engine.agents_by_id.items()):
        is_president = agent_id == engine.president_rotation[engine.current_president_idx]
        # Use most recent chancellor from events if engine state is None
        is_chancellor = agent_id == (engine.current_chancellor_id or most_recent_chancellor)
        
        print(f"DEBUG: Agent {agent_id} - President: {is_president}, Chancellor: {is_chancellor}")
        
        title_parts = []
        if is_president: title_parts.append("President")
        if is_chancellor: title_parts.append("Chancellor")
        
        players.append({
            "id": agent_id,
            "name": names[i],
            "role": f"({agent.role.value.title()})",
            "title": " & ".join(title_parts),
            "is_you": False,  # All AI players
            "model": agent.ai_model
        })
    
    return {
        "ok": True,
        "game_id": actual_game_id,
        "liberal_policies": engine.liberal_policies_played,
        "fascist_policies": engine.fascist_policies_played,
        "liberal_policies_to_win": 5,
        "fascist_policies_to_win": 6,
        "events": events,
        "players": players,
    }


# Mount static files AFTER all API routes are defined
# This ensures API routes take precedence over static file serving
app.mount("/", StaticFiles(directory=os.path.dirname(__file__), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
