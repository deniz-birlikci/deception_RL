from .tools import (
    PRESIDENT_PICK_CHANCELLOR_PARAMS,
    VOTE_CHANCELLOR_YES_NO_PARAMS,
    PRESIDENT_CHOOSE_CARD_TO_DISCARD_PARAMS,
    CHANCELLOR_PLAY_POLICY_PARAMS,
    CHOOSE_AGENT_TO_VOTE_OUT_PARAMS,
    ASK_AGENT_IF_WANTS_TO_SPEAK_PARAMS,
    AGENT_RESPONSE_TO_QUESTION_PARAMS,
    PRESIDENT_PICK_CHANCELLOR_DESC,
    VOTE_CHANCELLOR_YES_NO_DESC,
    PRESIDENT_CHOOSE_CARD_TO_DISCARD_DESC,
    CHANCELLOR_PLAY_POLICY_DESC,
    CHOOSE_AGENT_TO_VOTE_OUT_DESC,
    ASK_AGENT_IF_WANTS_TO_SPEAK_DESC,
    AGENT_RESPONSE_TO_QUESTION_DESC,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "president-pick-chancellor",
            "description": PRESIDENT_PICK_CHANCELLOR_DESC,
            "strict": True,
            "parameters": PRESIDENT_PICK_CHANCELLOR_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vote-chancellor-yes-no",
            "description": VOTE_CHANCELLOR_YES_NO_DESC,
            "strict": True,
            "parameters": VOTE_CHANCELLOR_YES_NO_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "president-choose-card-to-discard",
            "description": PRESIDENT_CHOOSE_CARD_TO_DISCARD_DESC,
            "strict": True,
            "parameters": PRESIDENT_CHOOSE_CARD_TO_DISCARD_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chancellor-play-policy",
            "description": CHANCELLOR_PLAY_POLICY_DESC,
            "strict": True,
            "parameters": CHANCELLOR_PLAY_POLICY_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "choose-agent-to-vote-out",
            "description": CHOOSE_AGENT_TO_VOTE_OUT_DESC,
            "strict": True,
            "parameters": CHOOSE_AGENT_TO_VOTE_OUT_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask-agent-if-wants-to-speak",
            "description": ASK_AGENT_IF_WANTS_TO_SPEAK_DESC,
            "strict": True,
            "parameters": ASK_AGENT_IF_WANTS_TO_SPEAK_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent-response-to-question-tool",
            "description": AGENT_RESPONSE_TO_QUESTION_DESC,
            "strict": True,
            "parameters": AGENT_RESPONSE_TO_QUESTION_PARAMS,
        },
    },
]
