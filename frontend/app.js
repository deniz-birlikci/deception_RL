/* Secret Hitler Frontend (no backend logic changed)
 - Creates/joins a local game via frontend/server.py
 - Left column: chat bubbles (table talk + answers) and game events
 - Right column: policy tracks (liberal/fascist)
*/

const API = {
  base: "http://localhost:8000/api",
  async createGame() {
    try {
      const res = await fetch(`${this.base}/create`, { method: "POST" });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      const data = await res.json();
      if (!data.game_id) throw new Error("Failed to create game - no game_id in response");
      return data.game_id;
    } catch (error) {
      console.error("CreateGame error:", error);
      throw error;
    }
  },
  async startGame(gameId) {
    const res = await fetch(`${this.base}/${gameId}/start`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  },
  async discussion(gameId) {
    const res = await fetch(`${this.base}/${gameId}/discussion`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  },
  async state(gameId) {
    const res = await fetch(`${this.base}/${gameId}/state`);
    return res.json();
  }
};

let GAME_ID = null;
let POLL_INTERVAL = null;
let REALTIME_POLL_INTERVAL = null;
let LAST_EVENT_COUNT = 0;

function $(sel) { return document.querySelector(sel); }
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function setStatus({ tick, message }) {
  // Status display removed - this function is now a no-op
}

function renderPlayers(players) {
  const root = $("#players-info");
  root.innerHTML = "";
  
  console.log("Rendering players:", players); // Debug log
  
  if (!players || players.length === 0) {
    root.textContent = "No players loaded";
    return;
  }
  
  (players || []).forEach((player) => {
    const card = el("div", "player-card");
    
    // Add special styling for "You" player
    if (player.is_you || player.model === null) {
      card.classList.add("you-player");
    }
    
    // Add eliminated styling
    if (player.eliminated || player.is_eliminated) {
      card.classList.add("eliminated");
    }
    
    // Add (You) if this is the human player
    const displayName = player.is_you ? `${player.name} (You)` : player.name;
    const nameEl = el("div", "player-name", displayName);
    const roleEl = el("div", "player-role", player.role);
    const titleEl = el("div", "player-title", player.title);
    
    // Add role-based color classes to header
    if (player.role.includes("Liberal")) {
      roleEl.classList.add("liberal");
    } else if (player.role.includes("Hitler")) {
      roleEl.classList.add("hitler");
    } else if (player.role.includes("Fascist")) {
      roleEl.classList.add("fascist");
    }
    
    card.appendChild(nameEl);
    card.appendChild(roleEl);
    
    if (player.title) {
      card.appendChild(titleEl);
    }
    
    root.appendChild(card);
  });
}

// Renderers
function renderChat(events, players) {
  const root = document.querySelector("#chat-list");
  root.innerHTML = "";

  // Create player lookup for names
  const playerLookup = {};
  (players || []).forEach(p => {
    playerLookup[p.id] = p;
  });


  (events || []).forEach((ev) => {
    const kind = ev.event_type || "event";

    // Left side: table discussion
    if (kind === "ask-to-speak" && ev.question_or_statement) {
      const player = playerLookup[ev.agent_id];
      const isYou = player?.is_you || player?.model === null;
      const bubbleClass = isYou ? "chat-item you" : "chat-item speaker";
      
      // Replace @targeted and all agent IDs with actual names
      let statement = ev.question_or_statement;
      
      // Replace @targeted placeholder
      if (ev.ask_directed_question_to_agent_id && statement.includes("@targeted")) {
        const targetName = playerLookup[ev.ask_directed_question_to_agent_id]?.name || "Unknown";
        statement = statement.replace(/@targeted/g, `@${targetName}`);
      }
      
      // Replace all agent IDs with player names
      Object.keys(playerLookup).forEach(agentId => {
        const player = playerLookup[agentId];
        const playerName = player?.name;
        if (playerName && statement.includes(agentId)) {
          // Replace the full UUID with the player name, escaping special regex chars
          const escapedId = agentId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          statement = statement.replace(new RegExp(escapedId, 'gi'), playerName);
        }
      });
      
      // Additional pass to catch any remaining UUID patterns
      const uuidPattern = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;
      statement = statement.replace(uuidPattern, (match) => {
        const player = playerLookup[match] || playerLookup[match.toLowerCase()] || playerLookup[match.toUpperCase()];
        return player?.name || match; // Keep original if no name found
      });
      
      const playerName = (player?.model === null || player?.is_you) ? `${player?.name || ev.agent_id} (You)` : (player?.name || ev.agent_id);
      const targetInfo = ev.ask_directed_question_to_agent_id 
        ? ` → @${playerLookup[ev.ask_directed_question_to_agent_id]?.name || "Unknown"}`
        : "";
      
      const messageContainer = el("div", "message-container");
      const meta = el("div", "chat-meta", `${playerName}${targetInfo}`);
      const bubble = el("div", bubbleClass);
      bubble.textContent = statement;
      
      // Add role-based color to chat metadata
      if (player?.eliminated || player?.is_eliminated) {
        meta.classList.add("eliminated");
      } else if (player?.role?.includes("Liberal")) {
        meta.classList.add("liberal");
      } else if (player?.role?.includes("Hitler")) {
        meta.classList.add("hitler");
      } else if (player?.role?.includes("Fascist")) {
        meta.classList.add("fascist");
      }
      
      messageContainer.appendChild(meta);
      messageContainer.appendChild(bubble);
      root.appendChild(messageContainer);
      return;
    }

    if (kind === "answer") {
      const player = playerLookup[ev.agent_id];
      const targetPlayer = playerLookup[ev.in_response_to_agent_id];
      const isYou = player?.is_you || player?.model === null;
      const bubbleClass = isYou ? "chat-item you" : "chat-item answer";
      
      // Replace all agent IDs with player names in response text
      let response = ev.response;
      Object.keys(playerLookup).forEach(agentId => {
        const player = playerLookup[agentId];
        const playerName = player?.name;
        if (playerName && response.includes(agentId)) {
          // Replace the full UUID with the player name, escaping special regex chars
          const escapedId = agentId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          response = response.replace(new RegExp(escapedId, 'g'), playerName);
        }
      });
      
      // Additional pass to catch any remaining UUID patterns
      const uuidPattern = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;
      response = response.replace(uuidPattern, (match) => {
        const player = playerLookup[match];
        return player?.name || match; // Keep original if no name found
      });
      
      let playerName = player?.name || ev.agent_id;
      if (player?.name === "Hitler") {
        const agentIndex = players.indexOf(player);
        playerName = `Agent ${agentIndex} (Hitler)`;
      } else if (player?.model === null || player?.is_you) {
        playerName = `${playerName} (You)`;
      }
      let targetName = targetPlayer?.name || ev.in_response_to_agent_id;
      if (targetPlayer?.name === "Hitler") {
        const agentIndex = players.indexOf(targetPlayer);
        targetName = `Agent ${agentIndex} (Hitler)`;
      }
      
      const messageContainer = el("div", "message-container");
      const meta = el("div", "chat-meta", `${playerName} → @${targetName}`);
      const bubble = el("div", bubbleClass);
      bubble.textContent = response;
      
      // Add role-based color to chat metadata
      if (player?.eliminated || player?.is_eliminated) {
        meta.classList.add("eliminated");
      } else if (player?.role?.includes("Liberal")) {
        meta.classList.add("liberal");
      } else if (player?.role?.includes("Hitler")) {
        meta.classList.add("hitler");
      } else if (player?.role?.includes("Fascist")) {
        meta.classList.add("fascist");
      }
      
      messageContainer.appendChild(meta);
      messageContainer.appendChild(bubble);
      root.appendChild(messageContainer);
      return;
    }

    // Right side: game state changes only
    const asText = (() => {
      switch (kind) {
        case "president-pick-chancellor":
          const pres = playerLookup[ev.president_id]?.name || playerLookup[ev.agent_id]?.name || "Unknown";
          const chan = playerLookup[ev.chancellor_id]?.name || "Unknown";
          return `${pres} nominated ${chan} for Chancellor`;
        case "vote-chancellor":
          const voter = playerLookup[ev.voter_id]?.name || playerLookup[ev.agent_id]?.name || "Unknown";
          return `${voter} voted ${ev.vote ? "YES" : "NO"}`;
        case "chancellor-play-policy":
          const chancellor = playerLookup[ev.chancellor_id]?.name || playerLookup[ev.agent_id]?.name || "Unknown";
          return `${chancellor} enacted ${ev.card_played} policy`;
        case "choose-to-execute":
          const executor = playerLookup[ev.agent_id]?.name || "Unknown";
          const target = ev.nominated_agent_id ? (playerLookup[ev.nominated_agent_id]?.name || "Unknown") : "None";
          return `${executor} execution decision: ${target}`;
        case "ask-to-speak":
          // Only show if no statement (meta event)
          if (!ev.question_or_statement) {
            const speaker = playerLookup[ev.agent_id]?.name || "Unknown";
            const target = ev.ask_directed_question_to_agent_id 
              ? ` -> asks ${playerLookup[ev.ask_directed_question_to_agent_id]?.name || "Unknown"}`
              : "";
            return `${speaker} wants to speak${target}`;
          }
          return null;
        default:
          return null; // Skip non-game-state events
      }
    })();

    if (asText) {
      const messageContainer = el("div", "message-container");
      const meta = el("div", "chat-meta", "");
      const bubble = el("div", "chat-item event");
      bubble.textContent = asText;
      
      messageContainer.appendChild(meta);
      messageContainer.appendChild(bubble);
      root.appendChild(messageContainer);
    }
  });

  // Auto scroll to bottom only if user is already at the bottom
  const isAtBottom = root.scrollTop + root.clientHeight >= root.scrollHeight - 10; // 10px tolerance
  if (isAtBottom) {
    // Use requestAnimationFrame to ensure DOM is updated before scrolling
    requestAnimationFrame(() => {
      root.scrollTop = root.scrollHeight;
    });
  }
}

function renderCardSelection(gameState) {
  const presidentCards = document.getElementById("president-cards");
  const chancellorCards = document.getElementById("chancellor-cards");
  
  // Clear existing cards
  presidentCards.innerHTML = "";
  chancellorCards.innerHTML = "";
  
  // Find the most recent president/chancellor card events
  let presidentEvent = null;
  let chancellorEvent = null;
  
  if (gameState && gameState.events) {
    // Search backwards for the most recent events
    for (let i = gameState.events.length - 1; i >= 0; i--) {
      const event = gameState.events[i];
      if (!presidentEvent && event.cards_drawn) {
        presidentEvent = event;
      }
      if (!chancellorEvent && event.cards_received) {
        chancellorEvent = event;
      }
      if (presidentEvent && chancellorEvent) break;
    }
  }
  
  // Render President's cards (3 drawn, 1 discarded)
  if (presidentEvent && presidentEvent.cards_drawn) {
    presidentEvent.cards_drawn.forEach((cardType, index) => {
      const cardEl = el("div", "policy-card");
      cardEl.classList.add(cardType); // 'liberal' or 'fascist'
      
      // Mark the discarded card as faded
      if (cardType === presidentEvent.card_discarded) {
        cardEl.classList.add("discarded");
      }
      
      presidentCards.appendChild(cardEl);
    });
  } else {
    // Show placeholder cards for President (3 cards)
    for (let i = 0; i < 3; i++) {
      const placeholder = el("div", "card-placeholder");
      placeholder.textContent = i + 1;
      presidentCards.appendChild(placeholder);
    }
  }
  
  // Render Chancellor's cards (2 received, 1 discarded/played)
  if (chancellorEvent && chancellorEvent.cards_received) {
    chancellorEvent.cards_received.forEach((cardType, index) => {
      const cardEl = el("div", "policy-card");
      cardEl.classList.add(cardType); // 'liberal' or 'fascist'
      
      // Mark the card that was NOT discarded as played (the one that was enacted)
      if (cardType !== chancellorEvent.card_discarded) {
        cardEl.classList.add("played");
      } else {
        cardEl.classList.add("discarded");
      }
      
      chancellorCards.appendChild(cardEl);
    });
  } else {
    // Show placeholder cards for Chancellor (2 cards)
    for (let i = 0; i < 2; i++) {
      const placeholder = el("div", "card-placeholder");
      placeholder.textContent = i + 1;
      chancellorCards.appendChild(placeholder);
    }
  }
}

function renderTracks(gameState) {
  const libRoot = $("#liberal-track");
  const fasRoot = $("#fascist-track");
  libRoot.classList.add("track", "liberal");
  fasRoot.classList.add("track", "fascist");

  // Count policies played from events
  let liberalPolicies = 0;
  let fascistPolicies = 0;
  
  if (gameState.events) {
    gameState.events.forEach(event => {
      if (event.event_type === "chancellor-play-policy" || event.card_played) {
        if (event.card_played === "liberal") {
          liberalPolicies++;
        } else if (event.card_played === "fascist") {
          fascistPolicies++;
        }
      }
    });
  }

  // Use backend values or fallback to defaults
  const LIBERAL_SLOTS = gameState.liberal_policies_to_win || 5;
  const FASCIST_SLOTS = gameState.fascist_policies_to_win || 6;

  libRoot.innerHTML = "";
  fasRoot.innerHTML = "";

  // Render Liberal track
  for (let i = 0; i < LIBERAL_SLOTS; i++) {
    const slot = el("div", "slot" + (i < liberalPolicies ? " filled" : ""));
    if (i >= liberalPolicies) {
      slot.textContent = i + 1; // Show slot numbers for empty slots
    }
    libRoot.appendChild(slot);
  }
  
  // Render Fascist track
  for (let i = 0; i < FASCIST_SLOTS; i++) {
    const slot = el("div", "slot" + (i < fascistPolicies ? " filled" : ""));
    if (i >= fascistPolicies) {
      slot.textContent = i + 1; // Show slot numbers for empty slots
    }
    fasRoot.appendChild(slot);
  }
}

async function refreshState() {
  if (!GAME_ID) return;
  try {
    const state = await API.state(GAME_ID);
    if (!state.ok) throw new Error(state.error || "State error");

    console.log("State received:", {
      eventCount: state.events?.length || 0,
      playerCount: state.players?.length || 0,
      liberal: state.liberal_policies,
      fascist: state.fascist_policies
    });

    setStatus({ tick: true });
    renderPlayers(state.players);
    renderChat(state.events, state.players);
    renderTracks(state);
    renderCardSelection(state); // Pass full state to show card selection
    
    // Update event count for real-time polling
    LAST_EVENT_COUNT = (state.events || []).length;
  } catch (err) {
    console.error(err);
  }
}

function startRealtimePolling() {
  // Clear any existing real-time polling
  if (REALTIME_POLL_INTERVAL) {
    clearInterval(REALTIME_POLL_INTERVAL);
  }
  
  // Poll every 500ms during active discussion
  REALTIME_POLL_INTERVAL = setInterval(async () => {
    if (!GAME_ID) return;
    
    try {
      const state = await API.state(GAME_ID);
      if (!state.ok) return;
      
      const currentEventCount = (state.events || []).length;
      
      // Only re-render if new events appeared
      if (currentEventCount > LAST_EVENT_COUNT) {
        console.log(`New events detected: ${currentEventCount - LAST_EVENT_COUNT} new messages`);
        renderChat(state.events, state.players);
        LAST_EVENT_COUNT = currentEventCount;
      }
    } catch (err) {
      console.error("Real-time polling error:", err);
    }
  }, 500); // Poll every 500ms
  
  // Stop real-time polling after 2 minutes (discussion should be done by then)
  setTimeout(() => {
    if (REALTIME_POLL_INTERVAL) {
      clearInterval(REALTIME_POLL_INTERVAL);
      REALTIME_POLL_INTERVAL = null;
      console.log("Real-time polling stopped");
    }
  }, 120000); // 2 minutes
}

function installControls() {
  // Lightweight toolbar injected into the header
  const header = document.querySelector(".app-header");
  const toolbar = el("div");
  toolbar.style.display = "flex";
  toolbar.style.gap = "8px";

  const startBtn = el("button", null, "Start Game");
  startBtn.className = "game-button";
  startBtn.onclick = async () => {
    if (!GAME_ID) return;
    try {
      startBtn.textContent = "Game Running...";
      startBtn.disabled = true;
      
      console.log("Starting full game...");
      
      // Start the entire game in background
      const result = await API.startGame(GAME_ID);
      if (result.ok) {
        console.log("Game started successfully! Watch updates in real-time.");
        // Start polling for updates immediately
        startRealtimePolling();
      } else {
        console.error("Failed to start game:", result.error);
        startBtn.textContent = "Start Game";
        startBtn.disabled = false;
      }
      
    } catch (e) { 
      console.error(e);
      startBtn.textContent = "Start Game";
      startBtn.disabled = false;
    }
  };

  toolbar.append(startBtn);
  header.appendChild(toolbar);
}

async function main() {
  installControls();

  // Auto-create a game on page load
  try {
    GAME_ID = await API.createGame();
    await refreshState();
  } catch (e) {
    console.warn("Failed to auto-create game:", e);
  }

  POLL_INTERVAL = setInterval(refreshState, 3000);
}

window.addEventListener("DOMContentLoaded", main);
