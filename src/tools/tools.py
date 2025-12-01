from typing import Any


def generate_tools(
    allowed_tools: list[str] | None = None,
    eligible_agent_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    tool_schemas = {
        "president-pick-chancellor": {
            "type": "function",
            "function": {
                "name": "president-pick-chancellor",
                "description": "As the President, nominate a player to be Chancellor for this round. The nominated player must be eligible (not term-limited from the previous government). After nomination, all players will vote on whether to approve this government. Choose wisely based on who you trust or want to test.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "enum": eligible_agent_ids,
                            "description": "The unique identifier of an agent/player in the game. This ID must correspond to a valid player currently in the game.",
                        }
                    },
                    "required": ["agent_id"],
                    "additionalProperties": False,
                },
            },
        },
        "vote-chancellor-yes-no": {
            "type": "function",
            "function": {
                "name": "vote-chancellor-yes-no",
                "description": "Vote on whether to approve the proposed government (President and Chancellor pair). Vote true (yes) if you want to approve this government and allow them to enact a policy, or false (no) if you want to reject it. All players vote simultaneously. If the vote fails, the election tracker advances and a new President nominates a Chancellor.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "choice": {
                            "type": "boolean",
                            "description": "A boolean value representing a yes/no choice. Use true for yes/approve and false for no/reject.",
                        }
                    },
                    "required": ["choice"],
                    "additionalProperties": False,
                },
            },
        },
        "president-choose-card-to-discard": {
            "type": "function",
            "function": {
                "name": "president-choose-card-to-discard",
                "description": "As President, you have drawn three policy cards and must discard one of them. Select the card_index (0, 1, or 2) of the card you want to discard. The remaining two cards will be passed to the Chancellor, who will then choose which one to enact. Use this power strategically to influence the game outcome.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "card_index": {
                            "type": "integer",
                            "enum": [0, 1, 2],
                            "description": "The zero-based index of a policy card (0, 1, or 2). Use this to select which card to discard from the available options.",
                        }
                    },
                    "required": ["card_index"],
                    "additionalProperties": False,
                },
            },
        },
        "chancellor-play-policy": {
            "type": "function",
            "function": {
                "name": "chancellor-play-policy",
                "description": "As Chancellor, you have received two policy cards from the President and must choose one to enact by selecting its card_index (0 or 1). The other card will be discarded. The enacted policy will be revealed to all players and added to the board. Choose carefully as this may trigger special presidential powers.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "card_index": {
                            "type": "integer",
                            "enum": [0, 1],
                            "description": "The zero-based index of a policy card (0 or 1). Use this to select which card to play from the available options.",
                        }
                    },
                    "required": ["card_index"],
                    "additionalProperties": False,
                },
            },
        },
        "choose-agent-to-vote-out": {
            "type": "function",
            "function": {
                "name": "choose-agent-to-vote-out",
                "description": "As President with executive power, choose a player to execute (remove from the game). Provide the agent_id of the player you want to eliminate, or null if you choose not to use this power (if allowed by game rules). Executed players are revealed and removed from the game permanently. Use this power to eliminate suspected impostors or to create confusion and mistrust among crewmates.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": ["string", "null"],
                            "enum": (eligible_agent_ids or []) + [None],
                            "description": "The unique identifier of an agent/player to vote out, or null to skip. When the president has the power to execute a player, provide the agent_id of the player to eliminate, or null to decline using this power if allowed.",
                        }
                    },
                    "required": ["agent_id"],
                    "additionalProperties": False,
                },
            },
        },
        "ask-agent-if-wants-to-speak": {
            "type": "function",
            "function": {
                "name": "ask-agent-if-wants-to-speak",
                "description": "Indicate if you want to speak during the discourse phase. Optionally provide a question_or_statement you want to make, and/or specify an ask_directed_question_to_agent_id to direct your question to a specific agent.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question_or_statement": {
                            "type": ["string", "null"],
                            "description": "An optional string value, or null if not provided.",
                        },
                        "ask_directed_question_to_agent_id": {
                            "type": ["string", "null"],
                            "enum": (eligible_agent_ids or []) + [None],
                            "description": "The unique identifier of an agent/player to vote out, or null to skip. When the president has the power to execute a player, provide the agent_id of the player to eliminate, or null to decline using this power if allowed.",
                        },
                    },
                    "required": [
                        "question_or_statement",
                        "ask_directed_question_to_agent_id",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "agent-response-to-question-tool": {
            "type": "function",
            "function": {
                "name": "agent-response-to-question-tool",
                "description": "Respond to a question or statement that was directed at you during the discourse phase.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "A string value.",
                        }
                    },
                    "required": ["response"],
                    "additionalProperties": False,
                },
            },
        },
    }

    tool_names = allowed_tools or list(tool_schemas.keys())
    return [tool_schemas[name] for name in tool_names if name in tool_schemas]
