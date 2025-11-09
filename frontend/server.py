from __future__ import annotations

import os
import sys
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
from src.engine.protocol import ModelOutput
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

# Static files will be mounted after API routes are defined

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
def api_create_game() -> GameState:
    global _engine, _game_id
    try:
        deck = Deck()
        # 4 AI players + 1 RL agent (None = RL agent)
        ai_models = [
            AIModel.OPENAI_GPT_5,
            AIModel.OPENAI_GPT_5_MINI,
            AIModel.OPENAI_GPT_5_NANO,
            AIModel.OPENAI_GPT_4_1,
            None,  # RL agent
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
            print(f"Engine state: liberal={_engine.liberal_policies_played}, fascist={_engine.fascist_policies_played}")
            print(f"Current events: {len(_engine.public_events)}")
            _game_running = True
            
            # Start a task to handle RL agent responses
            async def handle_rl_agent():
                import json
                import random
                while _game_running:
                    try:
                        # Wait for RL agent decision request
                        model_input = await asyncio.wait_for(output_queue.get(), timeout=1.0)
                        print(f"RL agent decision requested: {model_input.tool_call.name}")
                        
                        # TODO: Replace this with your actual RL agent logic
                        # This is a simple placeholder that makes random decisions
                        tool_name = model_input.tool_call.name
                        
                        if tool_name == "ask-agent-if-wants-to-speak":
                            # Sometimes speak, sometimes don't
                            if random.random() < 0.3:  # 30% chance to speak
                                arguments = {
                                    "question_or_statement": "I think we should be careful about this nomination.",
                                    "ask_directed_question_to_agent_id": None
                                }
                            else:
                                arguments = {
                                    "question_or_statement": None,
                                    "ask_directed_question_to_agent_id": None
                                }
                        elif tool_name == "president-pick-chancellor":
                            # Pick a random agent (excluding self - agent_4 is the RL agent)
                            available_agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
                            arguments = {"agent_id": random.choice(available_agents)}
                        elif tool_name == "vote-chancellor-yes-no":
                            # Random vote
                            arguments = {"choice": random.choice([True, False])}
                        elif tool_name == "president-choose-card-to-discard":
                            # Random card choice (0, 1, or 2)
                            arguments = {"card_index": random.randint(0, 2)}
                        elif tool_name == "chancellor-play-policy":
                            # Random card choice (0 or 1)
                            arguments = {"card_index": random.randint(0, 1)}
                        elif tool_name == "choose-agent-to-vote-out":
                            # Pick a random agent to execute
                            available_agents = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"]
                            arguments = {"agent_id": random.choice(available_agents)}
                        elif tool_name == "agent-response-to-question-tool":
                            # Respond to a question
                            arguments = {"response": "I'm not sure, but I trust the process."}
                        else:
                            # Default empty arguments
                            arguments = {}
                        
                        # Create proper ModelOutput format
                        function_calling_json = json.dumps({
                            "tool_name": tool_name,
                            "arguments": arguments
                        })
                        
                        model_output = ModelOutput(
                            function_calling_json=function_calling_json,
                            reasoning="RL agent decision"
                        )
                        
                        await input_queue.put(model_output)
                        print(f"RL agent responded: {arguments}")
                        
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"Error handling RL agent: {e}")
                        import traceback
                        traceback.print_exc()
                        break
            
            # Run both the game and RL agent handler concurrently
            await asyncio.gather(
                _engine.run(input_queue, output_queue),
                handle_rl_agent()
            )
            print(f"Game {game_id} completed!")
        except Exception as e:
            import traceback
            print(f"Error running game: {e}")
            print(f"Full traceback:")
            traceback.print_exc()
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
    print(f"DEBUG: Found {len(_engine.public_events)} public events, serialized to {len(events)} events")
    print(f"DEBUG: Game running: {_game_running}")
    if events:
        print(f"DEBUG: First event: {events[0]}")
        # Show all event types for debugging
        event_types = [ev.get("event_type", "unknown") for ev in events]
        print(f"DEBUG: All event types: {event_types}")
        # Show any ask-to-speak events specifically
        chat_events = [ev for ev in events if ev.get("event_type") in ["ask-to-speak", "answer"]]
        print(f"DEBUG: Chat events found: {len(chat_events)}")
    
    # Mock events removed - chat display confirmed working!
    
    # Create player info with names and roles
    available_names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
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
        
        # Assign name and determine if this is the human player
        player_name = available_names[name_index] if name_index < len(available_names) else f"Player{name_index+1}"
        is_you = agent.ai_model is None  # Human player has no AI model
        
        # Set role label
        if agent.role.value == "hitler":
            role_label = "(Hitler)"
        else:
            role_label = f"({agent.role.value.title()})"
        
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


# Mount static files AFTER all API routes are defined
# This ensures API routes take precedence over static file serving
app.mount("/", StaticFiles(directory=os.path.dirname(__file__), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
