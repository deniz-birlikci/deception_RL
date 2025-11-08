from src.models import Agent


def get_base_game_rules_prompt(
    num_players: int,
    fascist_policies_to_win: int,
    liberal_policies_to_win: int,
    hitler_election_threshold: int,
    num_liberal_cards: int,
    num_fascist_cards: int,
) -> str:
    return f"""SECRET HITLER - GAME RULES

=== OVERVIEW ===
Secret Hitler is a social deduction game where players are secretly assigned to either the Liberal team or the Fascist team. One Fascist player is Hitler.

=== ROLES ({num_players} PLAYERS) ===
- {num_players - 2} Liberals
- 1 Fascist (not Hitler)
- 1 Hitler

=== TEAM KNOWLEDGE ===
- Liberals: Know only their own role
- Fascists: Know each other and know who Hitler is
- Hitler: Knows who the other Fascist is

=== WIN CONDITIONS ===
Liberals win if:
- {liberal_policies_to_win} Liberal Policies are enacted
- Hitler is assassinated (if assassination power is implemented)

Fascists win if:
- {fascist_policies_to_win} Fascist Policies are enacted
- Hitler is elected Chancellor after {hitler_election_threshold} or more Fascist Policies have been enacted

=== GAME FLOW ===
Each round consists of:
1. Election Phase
2. Legislative Session
3. Executive Action (if applicable)

=== ELECTION PHASE ===
1. Presidential Candidacy passes clockwise to the next player
2. President nominates a Chancellor (any player except themselves and the previous Chancellor)
3. Discourse: All players may speak and ask/answer questions
4. All players vote Yes (Ja!) or No (Nein) on the proposed government
5. If majority votes Yes: Government is elected, proceed to Legislative Session
6. If majority votes No or tie: Government fails, Election Tracker advances by 1

TERM LIMITS:
- The last elected Chancellor cannot be nominated as Chancellor again
- The last elected President can be nominated as Chancellor
- Anyone can be President

FAILED ELECTIONS:
- After 3 consecutive failed elections, the top Policy card is automatically enacted
- No Presidential Power is granted from auto-enacted policies
- Election Tracker resets to 0
- All term limits are cleared

=== LEGISLATIVE SESSION ===
1. President draws 3 Policy cards from the deck
2. President discards 1 card face-down and passes 2 cards to Chancellor
3. Chancellor discards 1 card face-down and enacts the remaining card face-up
4. The enacted Policy is placed on the appropriate track (Liberal or Fascist)
5. Discourse: All players may speak and ask/answer questions

IMPORTANT:
- President and Chancellor may lie about what cards they saw
- Discarded cards are never revealed
- If deck has fewer than 3 cards, shuffle discard pile into deck before drawing

=== EXECUTIVE ACTIONS ===
After certain Fascist Policies are enacted, the President gains a one-time power:
- These powers vary based on the Fascist track configuration
- Powers must be used before the next round begins
- Even Liberal Presidents must use Fascist powers

=== POLICY DECK ===
- {num_liberal_cards} Liberal Policy cards
- {num_fascist_cards} Fascist Policy cards
- Cards are shuffled at game start
- When deck runs out, shuffle discard pile to create new deck

=== HITLER ELECTED ===
If Hitler is elected Chancellor after {hitler_election_threshold} or more Fascist Policies have been enacted, Fascists immediately win the game.

=== LYING ===
Players may lie about hidden information (cards seen, investigation results, etc.). The only exception is Hitler must reveal if assassinated or if elected Chancellor after {hitler_election_threshold} Fascist Policies.
"""
