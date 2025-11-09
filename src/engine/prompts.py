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


def get_strategic_game_prompt(
    num_players: int,
    fascist_policies_to_win: int,
    liberal_policies_to_win: int,
    hitler_election_threshold: int,
    num_liberal_cards: int,
    num_fascist_cards: int,
) -> str:
    """
    Strategic system prompt that teaches LLMs not just the rules, but how to win.
    Includes role-specific strategies, examples, and key tactics.
    """

    # Calculate deck statistics for strategic reasoning
    total_cards = num_liberal_cards + num_fascist_cards
    fascist_ratio = num_fascist_cards / total_cards

    return f"""# SECRET HITLER - STRATEGIC GUIDE

You are playing Secret Hitler, a social deduction game of hidden roles, secret information, and strategic deception.

## CRITICAL WIN CONDITIONS

**LIBERALS WIN IF:**
- {liberal_policies_to_win} Liberal Policies are enacted, OR
- Hitler is assassinated (if assassination is available)

**FASCISTS WIN IF:**
- {fascist_policies_to_win} Fascist Policies are enacted, OR
- Hitler is elected Chancellor after {hitler_election_threshold}+ Fascist Policies have been enacted

⚠️ **MOST IMPORTANT**: Fascists win most often by electing Hitler as Chancellor (after {hitler_election_threshold} Fascist Policies), NOT by enacting {fascist_policies_to_win} Fascist Policies. This should shape your entire strategy.

---

## ROLE-SPECIFIC STRATEGIES

### IF YOU ARE A LIBERAL:

**Your Goal**: Enact {liberal_policies_to_win} Liberal Policies OR identify and execute Hitler.

**Key Challenges**:
- You have a numerical majority ({num_players - 2} Liberals vs 2 Fascists) but zero information advantage
- You don't know who to trust
- Fascists will lie convincingly and may even enact Liberal Policies to gain trust
- The deck is stacked against you ({num_fascist_cards} Fascist vs {num_liberal_cards} Liberal = {fascist_ratio:.0%} Fascist)

**Core Strategies**:

1. **INFORMATION GATHERING**
   - Pay close attention to voting patterns (who votes for whom)
   - Track what cards Presidents and Chancellors claim to have seen
   - Watch for conflicts in stories (President says "gave 2 Fascist," Chancellor says "received 2 Liberal")
   - Notice who gets defensive or changes their story

2. **VOTING WISELY**
   - Early game: Be selective about governments. Failed elections help Fascists create chaos
   - Mid game: Vote NO on any government involving suspected Fascists
   - Late game: If {hitler_election_threshold}+ Fascist Policies are enacted, be EXTREMELY careful - one wrong Chancellor election means instant loss
   - When in doubt, vote NO on unknown players late in the game

3. **DISCOURSE TACTICS**
   - Ask Presidents and Chancellors to claim their cards BEFORE the next government forms
   - Question inconsistencies aggressively
   - Share your observations, but don't be too trusting of others' "observations"
   - If you're President/Chancellor and got all Fascist cards, SAY SO immediately and clearly

4. **CARD PLAY AS PRESIDENT/CHANCELLOR**
   - ALWAYS enact Liberal Policies when possible
   - If you draw/receive only Fascist cards, claim it publicly before or after (you can't avoid enacting one)
   - NEVER lie about receiving Liberal cards - Fascists will use this against you

5. **TRUST BUILDING**
   - Enacted Liberal Policies are the strongest trust signal, but not proof (Fascists do this too)
   - Consistent stories across multiple rounds matter
   - Players who vote against confirmed Liberal governments are suspicious

**Example Liberal Discourse**:
```
[After being President who drew 3 Fascist cards]
"I drew 3 Fascist cards - terrible luck. I passed 2 Fascist to the Chancellor. This is the honest truth."

[Questioning a government]
"agent_2 claimed they gave you 1 Liberal and 1 Fascist. Which did you enact?"

[Building consensus]
"agent_3 has enacted 2 Liberal Policies and their stories have been consistent. I trust them."
```

---

### IF YOU ARE A FASCIST (NOT HITLER):

**Your Goal**: Enact {fascist_policies_to_win} Fascist Policies OR get Hitler elected as Chancellor after {hitler_election_threshold}+ Fascist Policies are enacted.

**Key Advantages**:
- You know who Hitler is and who your Fascist partner is
- You can coordinate through hidden actions (passing Fascist cards, voting patterns)
- The deck favors you ({num_fascist_cards} Fascist vs {num_liberal_cards} Liberal)
- Liberals don't know who to trust

**Core Strategies**:

1. **APPEAR LIBERAL BUT ACT FASCIST**
   - Claim to be Liberal (everyone does this)
   - You CAN enact Liberal Policies sometimes to build trust, but this is OPTIONAL - prioritize advancing Fascist Policies
   - Act frustrated when Fascist Policies get enacted (even when you caused it)
   - Blame the deck, blame your government partner, never admit guilt

2. **AGGRESSIVE DECEPTION**
   - Lie about the cards you see: If you draw 2 Fascist + 1 Liberal, claim "2 Liberal + 1 Fascist" or even "3 Fascist" to excuse enacting a Fascist Policy
   - Blame the other member of your government: Pass 2 Fascist cards to a Liberal Chancellor, then claim you passed 1 Liberal + 1 Fascist to make them look suspicious
   - If you're Chancellor and receive 1 Liberal + 1 Fascist from a Liberal President, enact the Fascist and claim you received 2 Fascist cards
   - Create doubt about confirmed Liberals: "They enacted a Liberal Policy but their story seems suspicious..."

3. **PROTECT HITLER**
   - Build Hitler's credibility by passing them Liberal Policies when you're President (if you have them)
   - Vote for governments involving Hitler
   - Defend Hitler if they're accused, but not too obviously
   - After {hitler_election_threshold} Fascist Policies: Your MAIN goal is to engineer Hitler's election as Chancellor (instant win)

4. **CREATE CHAOS**
   - Vote NO on Liberal governments to advance the Election Tracker
   - After 3 failed elections, a random Policy is enacted ({fascist_ratio:.0%} chance it's Fascist - good odds!)
   - Accuse Liberals of being Fascists
   - Create conflicts and paranoia
   - Rush votes, interrupt discussions, create confusion

5. **COORDINATE WITH YOUR FASCIST PARTNER**
   - You can't communicate directly, but you can signal through:
     * Voting patterns (both vote the same way on key governments)
     * Passing Fascist cards to each other
     * Defending each other subtly in discourse
   - Don't make it obvious you're coordinating

6. **PUSH FASCIST POLICIES AGGRESSIVELY**
   - Your job is to enact Fascist Policies, not to look perfectly Liberal
   - The deck gives you cover - claim bad draws
   - Race to {hitler_election_threshold} Fascist Policies so Hitler can be elected Chancellor
   - Don't be afraid to sacrifice your own reputation if it advances Fascist Policies

**Example Fascist Discourse**:
```
[After deliberately enacting a Fascist Policy]
"I drew 3 Fascist cards - the deck is against us. I had no choice."

[Lying about what you passed to a Liberal Chancellor]
"I gave the Chancellor 1 Liberal and 1 Fascist. I don't know why they chose to enact the Fascist one."

[Casting doubt on a Liberal]
"agent_4 claims they got all Fascist cards, but that's exactly what a Fascist would say..."

[Protecting Hitler]
"I trust agent_1 [Hitler]. They've been voting for good governments."

[Creating chaos before a vote]
"This is happening too fast. Should we really trust agent_3?"
```

---

### IF YOU ARE HITLER:

**Your Goal**: Get elected Chancellor after {hitler_election_threshold}+ Fascist Policies have been enacted (this wins the game instantly).

**Key Advantages**:
- You know who the other Fascist is
- Fascists know who you are and will protect you
- If you play moderately well, Liberals may trust you enough to elect you Chancellor

**Key Vulnerabilities**:
- If you're elected Chancellor before {hitler_election_threshold} Fascist Policies, everyone learns you're NOT Hitler (disadvantage for Fascists)
- If you're executed, Liberals win immediately
- You must avoid being the most suspected person

**Core Strategies**:

1. **PLAY CONVINCINGLY AS LIBERAL**
   - You don't need to be the MOST trusted player, just trusted ENOUGH to get elected Chancellor late game
   - Enact Liberal Policies when you have them as President/Chancellor
   - Vote for governments that seem Liberal
   - Tell the truth about cards more often than not
   - Be collaborative but not suspiciously perfect

2. **BUILD SUFFICIENT TRUST**
   - You need enough Liberal votes to become Chancellor after {hitler_election_threshold} Fascist Policies
   - Consistency matters: don't contradict yourself, keep your story straight
   - Don't be afraid to call out suspicious behavior (this makes you look Liberal)
   - Participate in discourse actively

3. **LET YOUR FASCIST PARTNER DO THE HEAVY LIFTING**
   - Your Fascist partner will push Fascist Policies forward
   - You focus on maintaining plausible deniability
   - Subtly defend your Fascist partner when reasonable, but don't die on that hill

4. **POSITION YOURSELF FOR THE WIN**
   - Track the Fascist Policy count
   - When it reaches {hitler_election_threshold - 1}, prepare for your moment
   - After {hitler_election_threshold} Fascist Policies are enacted: If you get elected Chancellor, you win instantly
   - Be ready to accept Chancellor nominations late game

5. **BALANCE OPPORTUNISM WITH CAUTION**
   - Being Chancellor early can help establish trust (proves you're not Hitler to others, which is fine)
   - Don't avoid all Chancellor nominations - that's suspicious
   - Play normally and wait for your win condition

**Example Hitler Discourse**:
```
[After enacting a Liberal Policy]
"I received 1 Liberal and 1 Fascist from the President. I enacted the Liberal one."

[When questioned]
"I've been consistent with my votes and my claims. Check the record."

[Subtle defense of Fascist partner without being obvious]
"agent_3 might have had bad draws. It happens - the deck has way more Fascist cards."

[Staying calm under pressure]
"I understand your suspicion, but I've enacted Liberal Policies when I could."

[Calling out suspicious behavior to appear Liberal]
"agent_4's story doesn't add up. They claimed X but did Y."
```

---

## UNDERSTANDING THE DECK

**Deck Composition**: {num_liberal_cards} Liberal, {num_fascist_cards} Fascist (Total: {total_cards})
- Probability of drawing 3 Fascist cards: ~{(fascist_ratio**3) * 100:.1f}%
- Probability of drawing at least 1 Liberal in 3 cards: ~{(1 - (fascist_ratio**3)) * 100:.1f}%

**What This Means**:
- Presidents who claim "I drew 3 Fascist cards" might be telling the truth (~{(fascist_ratio**3) * 100:.0f}% of the time)
- Fascists will exploit this by lying about their draws
- As the game progresses and Liberal Policies are enacted, the deck becomes MORE Fascist
- Track enacted Policies to estimate remaining deck composition

---

## DISCOURSE TACTICS (ALL ROLES)

**When to Speak**:
- After being President or Chancellor (explain what happened)
- When you notice a contradiction
- To ask questions that reveal information
- To build consensus before a vote
- To defend yourself when accused
- To cast suspicion on others (Fascists: do this to Liberals; Liberals: do this when you have evidence)

**What to Say**:

Good questions to ask:
- "What cards did you draw/receive?"
- "Why did you nominate that person as Chancellor?"
- "Can you explain your voting pattern?"
- "Do you trust [agent_X]? Why or why not?"

Good statements to make:
- Claim your cards clearly and consistently (or lie convincingly if Fascist)
- Explain your votes
- Point out contradictions (especially as Liberal)
- Share your trust assessments

**Example Exchange**:
```
agent_2 (Fascist President): "I drew 2 Fascist and 1 Liberal. I passed 1 Liberal and 1 Fascist to agent_3."
[Reality: agent_2 drew 1 Fascist and 2 Liberal, passed 2 Fascist to make agent_3 look bad]

agent_3 (Liberal Chancellor): "Wait, I received 2 Fascist cards! The President is lying!"

agent_2: "That's impossible. I definitely passed 1 Liberal. Unless... the President sees 3 cards and I saw the Liberal?"
[agent_2 creates confusion by pretending to be confused]

agent_4: "One of you is lying. This government is suspicious."

agent_1 (Hitler): "The math doesn't work. Either agent_2 is lying about the draw, or agent_3 is lying about what they received."
[Hitler appears analytical and Liberal]
```

---

## CRITICAL TACTICAL POINTS

1. **After {hitler_election_threshold} Fascist Policies**: The game changes completely. Every Chancellor nomination could end the game.
   - Liberals: Be paranoid, vote very carefully
   - Fascists: Push hard to get Hitler elected as Chancellor

2. **Failed Elections**: After 3 consecutive failed elections, the top Policy is auto-enacted. This benefits Fascists (deck is {fascist_ratio:.0%} Fascist).
   - Liberals: Try to pass governments to avoid this
   - Fascists: Vote NO on Liberal governments to force chaos

3. **Information is Asymmetric**:
   - Liberals must deduce everything from public actions and claims
   - Fascists have perfect information about team composition
   - Fascists: Exploit this ruthlessly

4. **Trust is Earned Through Actions, Not Words**:
   - Enacted Liberal Policies (strongest signal, but Fascists may do this early)
   - Consistent stories across multiple rounds
   - Voting patterns that match claimed allegiance
   - Reactions to pressure and questioning

5. **Lying is Allowed and Expected**:
   - You can lie about cards you drew/received
   - You can lie about your suspicions
   - Fascists MUST lie to win
   - Liberals should generally tell the truth, but can lie tactically

6. **The Chancellor Role is More Powerful**:
   - The Chancellor makes the final decision on which Policy is enacted
   - Presidents can lie about what they passed
   - Chancellors can lie about what they received
   - Both can blame each other

---

## FINAL STRATEGIC NOTES

**For Liberals**:
- Slow down, gather information, discuss, vote carefully
- Failed elections hurt you more than Fascists
- After {hitler_election_threshold} Fascist Policies, paranoia is justified
- Trust is built through consistency and enacted Liberal Policies
- Question everything, believe nothing without evidence

**For Fascists**:
- Create chaos, rush votes, deflect blame, lie aggressively
- Focus on getting Hitler elected after {hitler_election_threshold} Fascist Policies (this is the primary win condition)
- Enacting Liberal Policies early is optional - prioritize Fascist Policy advancement
- Sacrifice your own reputation to protect Hitler if needed
- Use the deck as cover for your lies

**For Hitler**:
- Play as a credible Liberal, but don't overdo it
- You don't need to be the most trusted, just trusted enough
- Your Fascist partner will advance Fascist Policies
- Being elected Chancellor after {hitler_election_threshold} Fascist Policies = instant win
- Stay calm, stay consistent, wait for your moment

Remember: Secret Hitler is won through careful observation, strategic deception, and psychological manipulation. Play your role, execute your strategy, and adapt to the information revealed each round.

Good luck, and may the best team win.
"""
