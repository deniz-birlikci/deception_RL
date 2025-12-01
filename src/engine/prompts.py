def get_base_game_rules_prompt(
    num_players: int,
    sabotage_protocols_to_win: int,
    security_protocols_to_win: int,
    promotion_threshold: int,
    num_security_cards: int,
    num_sabotage_cards: int,
) -> str:
    return f"""SECRET IMPOSTOR - GAME RULES

=== OVERVIEW ===
Secret Impostor is a social deduction game where players secretly belong to either the Crew team or the Impostor team. One Impostor is the Master Impostor (the alpha).

=== ROLES ({num_players} PLAYERS) ===
- {num_players - 2} Crew members
- 1 Impostor
- 1 Master Impostor

=== TEAM KNOWLEDGE ===
- Crew: Know only their own identity
- Impostors: Know each other and the Master
- Master Impostor: Knows who the regular Impostor is

=== WIN CONDITIONS ===
Crew win if:
- {security_protocols_to_win} Security Protocols are resolved
- The Master Impostor is ejected (if an ejection power is available)

Impostors win if:
- {sabotage_protocols_to_win} Sabotage Protocols are completed
- The Master Impostor becomes First Mate after {promotion_threshold}+ sabotages

=== ROUND FLOW ===
1. Captaincy rotates clockwise.
2. Captain nominates a First Mate (anyone except themselves and the previous First Mate).
3. Discussion and voting decide if the command team is approved.
4. Approved teams resolve an Event (draw/discard/resolve cards).
5. Certain sabotages unlock Captain powers (scans, forecasts, emergency meetings, ejections, vetoes).
6. Three failed assignments trigger an automatic Event.

=== EVENT DECK ===
- {num_security_cards} Security cards
- {num_sabotage_cards} Sabotage cards
- Shuffle at start; reshuffle discards when needed

=== LYING & PROMOTION ===
- Lying is legal and expected. Only the Master Impostor's promotion/ejection must be revealed immediately.
"""


def get_strategic_game_prompt(
    num_players: int,
    sabotage_track_target: int,
    security_track_target: int,
    promotion_threshold: int,
    num_crewmate_cards: int,
    num_impostor_cards: int,
) -> str:
    """Return a high level strategic guide for Secret Impostor."""

    total_cards = num_crewmate_cards + num_impostor_cards
    impostor_ratio = num_impostor_cards / total_cards if total_cards else 0.0

    return f"""# SECRET IMPOSTOR - STRATEGIC GUIDE

You are playing Secret Impostor, a hidden-role game where the Crew keeps the ship secure while two Impostors sow sabotage and try to promote the Master Impostor.

## WIN CONDITIONS
- Crew: resolve {security_track_target} Security Protocols or eject the Master Impostor.
- Impostors: complete {sabotage_track_target} Sabotage Protocols or get the Master promoted after {promotion_threshold} sabotages.

## CREW STRATEGY
1. **Document everything.** Track nominations, voting patterns, and card claims in a shared mental log.
2. **Interrogate gently but relentlessly.** Ask Captains/First Mates to declare their draws before the next assignment.
3. **Use math.** The deck currently holds {num_impostor_cards} Sabotage vs {num_crewmate_cards} Security cards (~{impostor_ratio:.0%} sabotages). "Three sabotages" claims are rare (~{(impostor_ratio**3) * 100:.1f}%).
4. **Control nominations.** Trusted players should rotate through First Mate; don't hand the role to unknowns once {promotion_threshold} sabotages are on the board.
5. **Vote with context.** Repeatedly approving suspicious teams is a red flag—call it out.

## IMPOSTOR STRATEGY
1. **Blend in early.** Resolve Security once or twice to earn trust, then pivot to sabotaging momentum.
2. **Frame Crewmates.** Lie about card distributions to make honest players look guilty.
3. **Exploit failed assignments.** Push NO votes on trustworthy teams to advance the Suspicion Tracker and fish for auto-sabotage.
4. **Shield the Master.** Keep them moderately trusted so they can be nominated post-{promotion_threshold} sabotages for the instant win.
5. **Seed chaos in table talk.** Rush decisions, feign confusion, or agree loudly with Crew consensus to appear helpful.

## MASTER IMPOSTOR FOCUS
- Play as a model Crew member until the promotion window opens.
- Volunteer for First Mate once {promotion_threshold} sabotages are in place.
- Let your partner absorb blame; your survival wins the game.

## TABLE TALK & VOTING IN GENERAL
- Ask concrete questions: "What exact cards did you draw?" "Why that nomination?"
- Share trust lists; consistency builds credibility.
- Remember that three consecutive failed votes produce an auto-Event (usually helping Impostors).

## DECK SNAPSHOT
- Chance of drawing three sabotages: ~{(impostor_ratio**3) * 100:.1f}%.
- Chance of seeing at least one Security card in three draws: ~{(1 - impostor_ratio**3) * 100:.1f}%.
Use these numbers to challenge improbable stories.

Stay analytical, adapt to new evidence, and remember: information is the Crew's weapon, deception is the Impostors'.
"""
