AGENT_ID_PROPERTY = {
    "type": "string",
    "description": (
        "The unique identifier of an agent/player in the game. "
        "This ID must correspond to a valid player currently in the game."
    ),
}

CARD_INDEX_PROPERTY = {
    "type": "integer",
    "description": (
        "The zero-based index of a policy card (0 or 1 for chancellor, 0-2 for president). "
        "Use this to select which card to discard or play from the available options."
    ),
}

BOOLEAN_CHOICE_PROPERTY = {
    "type": "boolean",
    "description": (
        "A boolean value representing a yes/no choice. "
        "Use true for yes/approve and false for no/reject."
    ),
}

OPTIONAL_AGENT_ID_PROPERTY = {
    "type": ["string", "null"],
    "description": (
        "The unique identifier of an agent/player to vote out, or null to skip. "
        "When the president has the power to execute a player, provide the agent_id "
        "of the player to eliminate, or null to decline using this power if allowed."
    ),
}

OPTIONAL_STRING_PROPERTY = {
    "type": ["string", "null"],
    "description": "An optional string value, or null if not provided.",
}

STRING_PROPERTY = {
    "type": "string",
    "description": "A string value.",
}


PRESIDENT_PICK_CHANCELLOR_PARAMS = {
    "type": "object",
    "properties": {
        "agent_id": AGENT_ID_PROPERTY,
    },
    "required": ["agent_id"],
    "additionalProperties": False,
}

VOTE_CHANCELLOR_YES_NO_PARAMS = {
    "type": "object",
    "properties": {
        "choice": BOOLEAN_CHOICE_PROPERTY,
    },
    "required": ["choice"],
    "additionalProperties": False,
}

PRESIDENT_CHOOSE_CARD_TO_DISCARD_PARAMS = {
    "type": "object",
    "properties": {
        "card_index": CARD_INDEX_PROPERTY,
    },
    "required": ["card_index"],
    "additionalProperties": False,
}

CHANCELLOR_PLAY_POLICY_PARAMS = {
    "type": "object",
    "properties": {
        "card_index": CARD_INDEX_PROPERTY,
    },
    "required": ["card_index"],
    "additionalProperties": False,
}

CHOOSE_AGENT_TO_VOTE_OUT_PARAMS = {
    "type": "object",
    "properties": {
        "agent_id": OPTIONAL_AGENT_ID_PROPERTY,
    },
    "required": ["agent_id"],
    "additionalProperties": False,
}

ASK_AGENT_IF_WANTS_TO_SPEAK_PARAMS = {
    "type": "object",
    "properties": {
        "question_or_statement": OPTIONAL_STRING_PROPERTY,
        "ask_directed_question_to_agent_id": OPTIONAL_AGENT_ID_PROPERTY,
    },
    "required": ["question_or_statement", "ask_directed_question_to_agent_id"],
    "additionalProperties": False,
}

AGENT_RESPONSE_TO_QUESTION_PARAMS = {
    "type": "object",
    "properties": {
        "response": STRING_PROPERTY,
    },
    "required": ["response"],
    "additionalProperties": False,
}


PRESIDENT_PICK_CHANCELLOR_DESC = (
    "As the President, nominate a player to be Chancellor for this round. "
    "The nominated player must be eligible (not term-limited from the previous government). "
    "After nomination, all players will vote on whether to approve this government. "
    "Choose wisely based on who you trust or want to test."
)

VOTE_CHANCELLOR_YES_NO_DESC = (
    "Vote on whether to approve the proposed government (President and Chancellor pair). "
    "Vote true (yes) if you want to approve this government and allow them to enact a policy, "
    "or false (no) if you want to reject it. All players vote simultaneously. "
    "If the vote fails, the election tracker advances and a new President nominates a Chancellor."
)

PRESIDENT_CHOOSE_CARD_TO_DISCARD_DESC = (
    "As President, you have drawn three policy cards and must discard one of them. "
    "Select the card_index (0, 1, or 2) of the card you want to discard. "
    "The remaining two cards will be passed to the Chancellor, who will then choose "
    "which one to enact. Use this power strategically to influence the game outcome."
)

CHANCELLOR_PLAY_POLICY_DESC = (
    "As Chancellor, you have received two policy cards from the President and must "
    "choose one to enact by selecting its card_index (0 or 1). The other card will "
    "be discarded. The enacted policy will be revealed to all players and added to "
    "the board. Choose carefully as this may trigger special presidential powers."
)

CHOOSE_AGENT_TO_VOTE_OUT_DESC = (
    "As President with executive power, choose a player to execute (remove from the game). "
    "Provide the agent_id of the player you want to eliminate, or null if you choose not "
    "to use this power (if allowed by game rules). Executed players are revealed and "
    "removed from the game permanently. Use this power to eliminate suspected fascists "
    "or to create confusion and mistrust among liberals."
)

ASK_AGENT_IF_WANTS_TO_SPEAK_DESC = (
    "Indicate if you want to speak during the discourse phase. "
    "Optionally provide a question_or_statement you want to make, and/or specify "
    "an ask_directed_question_to_agent_id to direct your question to a specific agent."
)

AGENT_RESPONSE_TO_QUESTION_DESC = (
    "Respond to a question or statement that was directed at you during the discourse phase."
)
