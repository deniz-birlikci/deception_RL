/* Secret Hitler Frontend (no backend logic changed)
 - Creates/joins a local game via frontend/server.py
 - Left column: chat bubbles (table talk + answers) and game events
 - Right column: policy tracks (liberal/fascist)
*/

const API = {
  base: "http://localhost:8000/api",
  async createGame() {
    const res = await fetch(`${this.base}/create`, { method: "POST" });
    const data = await res.json();
    if (!data.game_id) throw new Error("Failed to create game");
    return data.game_id;
  },
  async discussion(gameId) {
    const res = await fetch(`${this.base}/${gameId}/discussion`, { method: "POST" });
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
    if (player.is_you) {
      card.classList.add("you-player");
    }
    
    const nameEl = el("div", "player-name", player.name);
    const roleEl = el("div", "player-role", player.role);
    const titleEl = el("div", "player-title", player.title);
    
    // Add role-based color classes to header
    if (player.role.includes("Liberal")) {
      roleEl.classList.add("liberal");
    } else if (player.role.includes("Fascist")) {
      if (player.name === "Hitler") {
        roleEl.classList.add("hitler");
      } else {
        roleEl.classList.add("fascist");
      }
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

  console.log("Player lookup:", playerLookup); // Debug log
  console.log("Events to render:", events); // Debug log

  (events || []).forEach((ev) => {
    const kind = ev.event_type || "event";

    // Left side: table discussion
    if (kind === "ask-to-speak" && ev.question_or_statement) {
      const player = playerLookup[ev.agent_id];
      const isYou = player?.is_you;
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
        const playerName = playerLookup[agentId]?.name;
        if (playerName && statement.includes(agentId)) {
          // Replace the full UUID with the player name
          statement = statement.replace(new RegExp(agentId, 'g'), playerName);
        }
      });
      
      const playerName = player?.name || ev.agent_id;
      const targetInfo = ev.ask_directed_question_to_agent_id 
        ? ` → @${playerLookup[ev.ask_directed_question_to_agent_id]?.name || "Unknown"}`
        : "";
      
      const messageContainer = el("div", "message-container");
      const meta = el("div", "chat-meta", `${playerName}${targetInfo}`);
      const bubble = el("div", bubbleClass);
      bubble.textContent = statement;
      
      // Add role-based color to chat metadata
      if (player?.role?.includes("Liberal")) {
        meta.classList.add("liberal");
      } else if (player?.role?.includes("Fascist")) {
        if (player.name === "Hitler") {
          meta.classList.add("hitler");
        } else {
          meta.classList.add("fascist");
        }
      }
      
      messageContainer.appendChild(meta);
      messageContainer.appendChild(bubble);
      root.appendChild(messageContainer);
      return;
    }

    if (kind === "answer") {
      const player = playerLookup[ev.agent_id];
      const targetPlayer = playerLookup[ev.in_response_to_agent_id];
      const isYou = player?.is_you;
      const bubbleClass = isYou ? "chat-item you" : "chat-item answer";
      
      // Replace all agent IDs with player names in response text
      let response = ev.response;
      Object.keys(playerLookup).forEach(agentId => {
        const playerName = playerLookup[agentId]?.name;
        if (playerName && response.includes(agentId)) {
          response = response.replace(new RegExp(agentId, 'g'), playerName);
        }
      });
      
      const playerName = player?.name || ev.agent_id;
      const targetName = targetPlayer?.name || ev.in_response_to_agent_id;
      
      const messageContainer = el("div", "message-container");
      const meta = el("div", "chat-meta", `${playerName} → @${targetName}`);
      const bubble = el("div", bubbleClass);
      bubble.textContent = response;
      
      // Add role-based color to chat metadata
      if (player?.role?.includes("Liberal")) {
        meta.classList.add("liberal");
      } else if (player?.role?.includes("Fascist")) {
        if (player.name === "Hitler") {
          meta.classList.add("hitler");
        } else {
          meta.classList.add("fascist");
        }
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
      const meta = el("div", "chat-meta", new Date().toLocaleTimeString());
      const bubble = el("div", "chat-item event");
      bubble.textContent = asText;
      
      messageContainer.appendChild(meta);
      messageContainer.appendChild(bubble);
      root.appendChild(messageContainer);
    }
  });

  // Auto scroll to bottom only if user is already at the bottom
  const isAtBottom = root.scrollTop + root.clientHeight >= root.scrollHeight - 5; // 5px tolerance
  if (isAtBottom) {
    root.scrollTop = root.scrollHeight;
  }
}

function renderCardSelection(cardData) {
  const presidentCards = document.getElementById("president-cards");
  const chancellorCards = document.getElementById("chancellor-cards");
  
  // Clear existing cards
  presidentCards.innerHTML = "";
  chancellorCards.innerHTML = "";
  
  if (!cardData) return;
  
  // Render President's cards (3 drawn, 1 discarded)
  if (cardData.presidentDraw) {
    cardData.presidentDraw.forEach((card, index) => {
      const cardEl = el("div", "policy-card");
      cardEl.classList.add(card.type); // 'liberal' or 'fascist'
      
      if (card.discarded) {
        cardEl.classList.add("discarded");
      }
      
      presidentCards.appendChild(cardEl);
    });
  }
  
  // Render Chancellor's cards (2 received, 1 played)
  if (cardData.chancellorReceive) {
    cardData.chancellorReceive.forEach((card, index) => {
      const cardEl = el("div", "policy-card");
      cardEl.classList.add(card.type); // 'liberal' or 'fascist'
      
      if (card.played) {
        cardEl.classList.add("played");
      }
      
      chancellorCards.appendChild(cardEl);
    });
  }
}

function renderTracks({ liberal, fascist, liberalSlotsTotal, fascistSlotsTotal }) {
  const libRoot = $("#liberal-track");
  const fasRoot = $("#fascist-track");
  libRoot.classList.add("track", "liberal");
  fasRoot.classList.add("track", "fascist");

  // Use backend values or fallback to defaults
  const LIBERAL_SLOTS = liberalSlotsTotal || 5;
  const FASCIST_SLOTS = fascistSlotsTotal || 6;

  libRoot.innerHTML = "";
  fasRoot.innerHTML = "";

  for (let i = 0; i < LIBERAL_SLOTS; i++) {
    const slot = el("div", "slot" + (i < liberal ? " filled" : ""));
    libRoot.appendChild(slot);
  }
  for (let i = 0; i < FASCIST_SLOTS; i++) {
    const slot = el("div", "slot" + (i < fascist ? " filled" : ""));
    fasRoot.appendChild(slot);
  }
}

async function refreshState() {
  if (!GAME_ID) return;
  try {
    const state = await API.state(GAME_ID);
    if (!state.ok) throw new Error(state.error || "State error");

    setStatus({ tick: true });
    renderPlayers(state.players);
    renderChat(state.events, state.players);
    renderTracks({ 
      liberal: state.liberal_policies, 
      fascist: state.fascist_policies,
      liberalSlotsTotal: state.liberal_policies_to_win,
      fascistSlotsTotal: state.fascist_policies_to_win
    });
    
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
      startBtn.textContent = "Running...";
      startBtn.disabled = true;
      
      console.log("Starting discussion round...");
      const start = Date.now();
      
      // Start the discussion round (non-blocking)
      API.discussion(GAME_ID).then(() => {
        const elapsed = ((Date.now() - start) / 1000).toFixed(1);
        console.log(`Discussion round completed in ${elapsed}s`);
        startBtn.textContent = "Start Game";
        startBtn.disabled = false;
      }).catch(e => {
        console.error(e);
        startBtn.textContent = "Start Game";
        startBtn.disabled = false;
      });
      
      // Start polling for new messages immediately
      startRealtimePolling();
      
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
