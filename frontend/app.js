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
    console.log(`DEBUG: Rendering player ${player.name} with title: "${player.title}"`);
    
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

  // Debug: Log event order to verify chronological rendering
  if (events && events.length > 0) {
    console.log(`Rendering ${events.length} events in order:`);
    events.slice(-5).forEach((ev, index) => { // Show last 5 events
      if (ev.question_or_statement) {
        console.log(`  ${events.length - 5 + index + 1}: ${ev.question_or_statement.substring(0, 40)}...`);
      } else {
        console.log(`  ${events.length - 5 + index + 1}: ${ev.event_type || 'unknown event'}`);
      }
    });
  }

  // Track rounds and insert dividers
  let roundCounter = 0;
  let lastEventWasPolicy = false;

  (events || []).forEach((ev, index) => {
    const kind = ev.event_type || "event";

    // Add Round 1 divider at the very beginning
    if (index === 0 && kind === "president-pick-chancellor") {
      roundCounter = 1;
      const divider = el("div", "round-divider");
      divider.textContent = `——— Round 1 ———`;
      divider.style.textAlign = "center";
      divider.style.color = "#9aa0a6";
      divider.style.fontSize = "12px";
      divider.style.margin = "10px 0";
      divider.style.userSelect = "none";
      root.appendChild(divider);
    }

    // Insert a round divider immediately after a policy is enacted
    if (kind === "chancellor-play-policy") {
      roundCounter += 1;
      // First render the policy event
      // (this will be handled by the normal event rendering below)
      
      // Then add the round divider after this policy
      lastEventWasPolicy = true;
    }

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
      
      // Replace all agent IDs with player names (case-insensitive, multiple formats)
      Object.keys(playerLookup).forEach(agentId => {
        const player = playerLookup[agentId];
        const playerName = player?.name;
        if (playerName) {
          // Replace various formats: agent_0, Agent_0, AGENT_0, etc.
          const patterns = [
            agentId, // exact: agent_0
            agentId.charAt(0).toUpperCase() + agentId.slice(1), // Agent_0
            agentId.toUpperCase(), // AGENT_0
            agentId.replace('_', ''), // agent0
            agentId.replace('_', '').charAt(0).toUpperCase() + agentId.replace('_', '').slice(1) // Agent0
          ];
          
          patterns.forEach(pattern => {
            if (statement.includes(pattern)) {
              const escapedPattern = pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
              statement = statement.replace(new RegExp(`\\b${escapedPattern}\\b`, 'gi'), playerName);
            }
          });
        }
      });
      
      // Additional pass to catch any remaining UUID patterns
      const uuidPattern = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;
      statement = statement.replace(uuidPattern, (match) => {
        const player = playerLookup[match] || playerLookup[match.toLowerCase()] || playerLookup[match.toUpperCase()];
        return player?.name || match; // Keep original if no name found
      });
      
      const playerName = (player?.model === null || player?.is_you) ? `${player?.name || ev.agent_id} (You)` : (player?.name || ev.agent_id);
      
      // Detect if message is addressing someone
      let targetInfo = "";
      let addressedPlayer = null;
      
      // Check if there's an explicit target
      if (ev.ask_directed_question_to_agent_id) {
        addressedPlayer = playerLookup[ev.ask_directed_question_to_agent_id]?.name || "Unknown";
      } else {
        // Auto-detect addressing by looking for player names in the message
        Object.keys(playerLookup).forEach(agentId => {
          const targetPlayerName = playerLookup[agentId]?.name;
          if (targetPlayerName && targetPlayerName !== player?.name) {
            // Check if message starts with the player's name (common addressing pattern)
            const addressingPatterns = [
              `${targetPlayerName}:`,
              `${targetPlayerName},`,
              `@${targetPlayerName}`,
              `${targetPlayerName} `,
            ];
            
            for (const pattern of addressingPatterns) {
              if (statement.toLowerCase().startsWith(pattern.toLowerCase()) || 
                  statement.toLowerCase().includes(pattern.toLowerCase())) {
                addressedPlayer = targetPlayerName;
                break;
              }
            }
          }
        });
      }
      
      if (addressedPlayer) {
        targetInfo = ` → @${addressedPlayer}`;
      }
      
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
      
      // Replace all agent IDs with player names in response text (case-insensitive, multiple formats)
      let response = ev.response;
      Object.keys(playerLookup).forEach(agentId => {
        const player = playerLookup[agentId];
        const playerName = player?.name;
        if (playerName) {
          // Replace various formats: agent_0, Agent_0, AGENT_0, etc.
          const patterns = [
            agentId, // exact: agent_0
            agentId.charAt(0).toUpperCase() + agentId.slice(1), // Agent_0
            agentId.toUpperCase(), // AGENT_0
            agentId.replace('_', ''), // agent0
            agentId.replace('_', '').charAt(0).toUpperCase() + agentId.replace('_', '').slice(1) // Agent0
          ];
          
          patterns.forEach(pattern => {
            if (response.includes(pattern)) {
              const escapedPattern = pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
              response = response.replace(new RegExp(`\\b${escapedPattern}\\b`, 'gi'), playerName);
            }
          });
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
    
    // Add round divider immediately after a policy is enacted
    if (kind === "chancellor-play-policy") {
      const divider = el("div", "round-divider");
      divider.textContent = `——— Round ${roundCounter} ———`;
      divider.style.textAlign = "center";
      divider.style.color = "#9aa0a6";
      divider.style.fontSize = "12px";
      divider.style.margin = "10px 0";
      divider.style.userSelect = "none";
      root.appendChild(divider);
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
  console.log("🎴 CARD SELECTION: Function called!");
  console.log("🎴 CARD SELECTION: GameState events length:", gameState?.events?.length || 0);
  
  const presidentCards = document.getElementById("president-cards");
  const chancellorCards = document.getElementById("chancellor-cards");
  
  if (!presidentCards || !chancellorCards) {
    console.warn("Card selection elements not found in DOM");
    return;
  }
  
  // Clear existing cards
  presidentCards.innerHTML = "";
  chancellorCards.innerHTML = "";
  
  console.log("DEBUG: Rendering card selection, checking events...");
  console.log("DEBUG: Total events:", gameState?.events?.length || 0);
  
  // Find the most recent president/chancellor card events
  let presidentEvent = null;
  let chancellorEvent = null;
  
  if (gameState && gameState.events) {
    console.log("DEBUG: Searching through events for card messages...");
    // Search backwards for the most recent events
    for (let i = gameState.events.length - 1; i >= 0; i--) {
      const event = gameState.events[i];
      
      // Debug: log each event we're checking
      if (event.question_or_statement) {
        console.log(`DEBUG: Checking event ${i}: "${event.question_or_statement.substring(0, 50)}..."`);
      }
      
      // Look for president card draw messages: "You drew 3 cards: [LIBERAL, FASCIST, FASCIST] and discarded FASCIST"
      if (!presidentEvent && event.question_or_statement && event.question_or_statement.includes("You drew 3 cards:")) {
        const text = event.question_or_statement;
        const cardsMatch = text.match(/\[(.*?)\]/);
        const discardedMatch = text.match(/discarded (\w+)/);
        
        if (cardsMatch && discardedMatch) {
          presidentEvent = {
            cards_drawn: cardsMatch[1].split(', ').map(card => card.toLowerCase()),
            card_discarded: discardedMatch[1].toLowerCase()
          };
          console.log("DEBUG: Found president event:", presidentEvent);
        }
      }
      
      // Look for chancellor card receive messages: "You received 2 cards from President agent_2: [FASCIST, LIBERAL] and discarded FASCIST"
      if (!chancellorEvent && event.question_or_statement && event.question_or_statement.includes("You received 2 cards from President")) {
        const text = event.question_or_statement;
        const cardsMatch = text.match(/\[(.*?)\]/);
        const discardedMatch = text.match(/discarded (\w+)/);
        
        if (cardsMatch && discardedMatch) {
          chancellorEvent = {
            cards_received: cardsMatch[1].split(', ').map(card => card.toLowerCase()),
            card_discarded: discardedMatch[1].toLowerCase()
          };
          console.log("DEBUG: Found chancellor event:", chancellorEvent);
        }
      }
      
      if (presidentEvent && chancellorEvent) break;
    }
  } else {
    console.log("DEBUG: No gameState or events available");
  }
  
  console.log("DEBUG: Final results - presidentEvent:", presidentEvent);
  console.log("DEBUG: Final results - chancellorEvent:", chancellorEvent);
  
  // Render President's cards (3 drawn, 1 discarded)
  if (presidentEvent && presidentEvent.cards_drawn) {
    console.log("DEBUG: Rendering president cards:", presidentEvent.cards_drawn);
    presidentEvent.cards_drawn.forEach((cardType, index) => {
      const cardEl = el("div", "policy-card");
      cardEl.classList.add(cardType); // 'liberal' or 'fascist'
      cardEl.textContent = cardType.toUpperCase();
      
      // Basic styling
      cardEl.style.padding = "8px 12px";
      cardEl.style.margin = "2px";
      cardEl.style.border = "2px solid #333";
      cardEl.style.borderRadius = "4px";
      cardEl.style.fontSize = "12px";
      cardEl.style.fontWeight = "bold";
      
      if (cardType === 'liberal') {
        cardEl.style.backgroundColor = "#4CAF50";
        cardEl.style.color = "white";
      } else {
        cardEl.style.backgroundColor = "#f44336";
        cardEl.style.color = "white";
      }
      
      // Mark the discarded card as faded
      if (cardType === presidentEvent.card_discarded) {
        cardEl.classList.add("discarded");
        cardEl.style.opacity = "0.5";
        cardEl.style.textDecoration = "line-through";
      }
      
      presidentCards.appendChild(cardEl);
    });
  } else {
    console.log("DEBUG: No president event found, showing placeholders");
    // Show placeholder cards for President (3 cards)
    for (let i = 0; i < 3; i++) {
      const placeholder = el("div", "card-placeholder");
      placeholder.textContent = i + 1;
      placeholder.style.padding = "8px 12px";
      placeholder.style.margin = "2px";
      placeholder.style.border = "2px dashed #666";
      placeholder.style.borderRadius = "4px";
      placeholder.style.backgroundColor = "transparent";
      placeholder.style.color = "#999";
      presidentCards.appendChild(placeholder);
    }
  }
  
  // Render Chancellor's cards (2 received, 1 discarded/played)
  if (chancellorEvent && chancellorEvent.cards_received) {
    console.log("DEBUG: Rendering chancellor cards:", chancellorEvent.cards_received);
    chancellorEvent.cards_received.forEach((cardType, index) => {
      const cardEl = el("div", "policy-card");
      cardEl.classList.add(cardType); // 'liberal' or 'fascist'
      cardEl.textContent = cardType.toUpperCase();
      
      // Basic styling
      cardEl.style.padding = "8px 12px";
      cardEl.style.margin = "2px";
      cardEl.style.border = "2px solid #333";
      cardEl.style.borderRadius = "4px";
      cardEl.style.fontSize = "12px";
      cardEl.style.fontWeight = "bold";
      
      if (cardType === 'liberal') {
        cardEl.style.backgroundColor = "#4CAF50";
        cardEl.style.color = "white";
      } else {
        cardEl.style.backgroundColor = "#f44336";
        cardEl.style.color = "white";
      }
      
      // Mark the card that was NOT discarded as played (the one that was enacted)
      if (cardType !== chancellorEvent.card_discarded) {
        cardEl.classList.add("played");
        cardEl.style.border = "3px solid gold";
        cardEl.style.boxShadow = "0 0 8px rgba(255, 215, 0, 0.6)";
      } else {
        cardEl.classList.add("discarded");
        cardEl.style.opacity = "0.5";
        cardEl.style.textDecoration = "line-through";
      }
      
      chancellorCards.appendChild(cardEl);
    });
  } else {
    console.log("DEBUG: No chancellor event found, showing placeholders");
    // Show placeholder cards for Chancellor (2 cards)
    for (let i = 0; i < 2; i++) {
      const placeholder = el("div", "card-placeholder");
      placeholder.textContent = i + 1;
      placeholder.style.padding = "8px 12px";
      placeholder.style.margin = "2px";
      placeholder.style.border = "2px dashed #666";
      placeholder.style.borderRadius = "4px";
      placeholder.style.backgroundColor = "transparent";
      placeholder.style.color = "#999";
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
    const prevEventCount = LAST_EVENT_COUNT;
    const state = await API.state(GAME_ID);
    if (!state.ok) {
      // If game not found, try to create a new one
      if (state.error === "game not found" || state.error === "no game exists") {
        console.log("Game not found, creating new game...");
        GAME_ID = await API.createGame();
        return; // Will retry on next poll
      }
      throw new Error(state.error || "State error");
    }
    
    // Update GAME_ID if server provides it (prevents future mismatches)
    if (state.game_id && state.game_id !== GAME_ID) {
      console.log(`Syncing game ID: ${GAME_ID} -> ${state.game_id}`);
      GAME_ID = state.game_id;
    }
    
    renderPlayers(state.players);
    renderChat(state.events, state.players);
    renderTracks(state);
    renderCardSelection(state); // Pass full state to show card selection
    
    // Update event count and show new message indicator
    const newEventCount = (state.events || []).length;
    if (newEventCount > prevEventCount) {
      const newMessages = newEventCount - prevEventCount;
      console.log(`📨 ${newMessages} new message(s) received - Total: ${newEventCount}`);
      
      // Log the actual new messages for debugging
      const newEvents = (state.events || []).slice(prevEventCount);
      newEvents.forEach((event, index) => {
        if (event.question_or_statement) {
          console.log(`  New message ${prevEventCount + index + 1}: ${event.question_or_statement.substring(0, 50)}...`);
        } else {
          console.log(`  New event ${prevEventCount + index + 1}: ${event.event_type || 'unknown'}`);
        }
      });
      
      // Scroll to bottom to show new messages
      const chatList = document.querySelector("#chat-list");
      if (chatList) {
        chatList.scrollTop = chatList.scrollHeight;
      }
    }
    
    LAST_EVENT_COUNT = newEventCount;
  } catch (error) {
    console.error("Failed to refresh state:", error);
  }
}

function startRealtimePolling() {
  // Clear any existing real-time polling
  if (REALTIME_POLL_INTERVAL) {
    clearInterval(REALTIME_POLL_INTERVAL);
  }
  
  console.log("Starting aggressive real-time polling for live updates");
  REALTIME_POLL_INTERVAL = setInterval(async () => {
    if (!GAME_ID) return;
    
    try {
      const prevEventCount = LAST_EVENT_COUNT; // Define prevEventCount here
      await refreshState(); // Use the main refresh function which handles everything
      
      // If we got new events, log them
      if (LAST_EVENT_COUNT > prevEventCount) {
        console.log(`📨 ${LAST_EVENT_COUNT - prevEventCount} new events detected via real-time polling`);
      }
    } catch (error) {
      console.error("Real-time polling error:", error);
    }
  }, 100); // Poll every 100ms for maximum responsiveness
  
  // Stop aggressive polling after 5 minutes, but keep checking periodically
  setTimeout(() => {
    if (REALTIME_POLL_INTERVAL) {
      clearInterval(REALTIME_POLL_INTERVAL);
      REALTIME_POLL_INTERVAL = null;
      console.log("Aggressive polling stopped, switching to normal polling");
      
      // Continue with slower polling
      REALTIME_POLL_INTERVAL = setInterval(async () => {
        try {
          await refreshState();
        } catch (error) {
          console.error("Polling error:", error);
        }
      }, 2000); // Poll every 2 seconds after aggressive period
    }
  }, 300000); // 5 minutes of aggressive polling
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

  POLL_INTERVAL = setInterval(refreshState, 1000); // Poll every 1 second for more responsive updates
  
  // Start aggressive polling during active gameplay
  startRealtimePolling();
}

window.addEventListener("DOMContentLoaded", main);
