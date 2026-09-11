<div align="center">

# 🃏 Sneak Peek Hold'em — IIT Pokerbots 2026

**A production-grade, ML-augmented poker bot for a novel hold'em variant with information-theoretic auction mechanics.**

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Competition](https://img.shields.io/badge/IIT%20Pokerbots-2026-gold?style=for-the-badge&logo=academia)](https://pokerbots.org)
[![XGBoost](https://img.shields.io/badge/Engine-Bitwise%20Evaluator-red?style=for-the-badge&logo=apache&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Game Overview — Sneak Peek Hold'em

**Sneak Peek Hold'em** is a heads-up variant of **No-Limit Texas Hold'em** with one critical twist: a post-flop, **sealed-bid, second-price auction**.

### Standard Flow

| Street | Event |
|--------|-------|
| Pre-flop | Both players post blinds (SB = 10, BB = 20). Betting round. |
| Flop | Three community cards are dealt. |
| **Auction** | **Sealed-bid second-price auction for opponent hole-card intelligence.** |
| Turn | Fourth community card. Betting round. |
| River | Fifth community card. Final betting round + showdown. |

### The Auction Mechanic

After the flop, both players **simultaneously** submit a chip bid. The rules:

- The **higher bidder wins** the auction.
- The winner **pays the lower bid** into the pot (second-price / Vickrey logic).
- As a reward, the winner is shown **one of the opponent's hole cards**, chosen **uniformly at random**.

This mechanic creates a rich information-theoretic sub-game layered on top of standard poker:
- **Bid too little**: opponent gains a free information advantage.
- **Bid too much**: you overpay and erode your chip stack.
- **Optimal bid**: proportional to the informational *value* of seeing an opponent card, not the card's absolute strength.

The bot exploits this through its **GTO Auction Engine** (see Architecture below).

---

## ⚙️ Competition Constraints

| Parameter | Value |
|-----------|-------|
| Total rounds per match | **1,000** |
| Starting chips per round | **5,000** |
| Per-query time limit | **2 seconds** |
| Total time budget (all rounds) | **20 seconds** |

The strict timing budget drives several architectural decisions: pre-computed lookup tables for pre-flop equity, time-guarded Monte Carlo loops (abort after 15 ms per trial), and memoised river enumeration.

---

## 📁 Directory Structure

```
bot-engine-2026/
│
├── bot.py              # ← Primary bot — full decision engine (this submission)
├── config.py           # Engine configuration (bot paths, Python command, log folder)
├── engine.py           # Official match engine provided by organisers
├── requirements.txt    # Python dependencies (eval7, pyparsing, future)
│
├── pkbot/              # Official bot SDK package
│   ├── actions.py      # Action types: ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
│   ├── base.py         # BaseBot abstract class
│   ├── runner.py       # CLI runner and socket communication
│   └── states.py       # GameInfo, PokerState data structures
│
└── logs/               # Auto-generated match logs (created at runtime)
```

---

## 🏗️ Architecture & Strategy — Deep Dive

### High-Level Pipeline

```
get_move() called
     │
     ├─ Stage 1: State Parsing
     │   • Convert card strings → integer deck indices
     │   • Detect auction intel (secured / leaked)
     │   • Update opponent profiling metrics (VPIP, AF, hyperbet ratio)
     │
     ├─ Stage 2: Equity Computation  (street-adaptive)
     │   ├─ pre-flop  → O(1) baseline table lookup
     │   ├─ flop/turn → Weighted Monte Carlo (60–250 iterations)
     │   └─ river     → Full deterministic enumeration (memoised)
     │
     ├─ Stage 3: Archetype Equity Modifiers
     │   • Applied after hand 100 once opponent archetype is classified
     │   • Equity shifts of ±0.05 to ±0.25 based on opponent profile
     │
     ├─ Stage 4: Required Equity Calculation
     │   • Minimum call equity = cost / (pot + cost)
     │   • + buffer (3–5 pp) + leverage penalty + overbet penalty
     │
     └─ Stage 5: Street-Specific Action Selection
         ├─ pre-flop  → Position-aware opening / 3-bet / defence
         ├─ auction   → GTO bid formula with archetype overrides
         └─ flop/turn/river → Raise / call / fold via equity thresholds
```

---

### Layer 1 — Bitwise Hand Evaluator

The hand evaluator is a **custom, zero-dependency, bitwise evaluator** inspired by Cactus Kev, capable of ranking any 5–7 card combination in **microseconds** using pure integer arithmetic.

#### Card Encoding

All 52 cards are assigned a unique integer index `0–51` via:

```python
index = (rank_value - 2) * 4 + suit_index
# e.g. Ace of Hearts = (14-2)*4 + 0 = 48
```

#### Compressed Bitmask Structures

| Structure | Format | Purpose |
|-----------|--------|---------|
| `rank_count` (52-bit int) | 4-bit nibble per rank | Pair/trip/quad detection via bit-shift |
| `suit_count` (16-bit int) | 4-bit nibble per suit | Flush detection: nibble ≥ 5 |
| `bit_mask` (int) | OR-mask of all rank bits | Straight detection via 5-bit window scan |
| `flush_masks[4]` (int×4) | Per-suit rank bitmasks | Straight-flush detection |

#### Evaluation Priority

```
Straight Flush ▶ Four of a Kind ▶ Full House ▶ Flush
▶ Straight ▶ Three of a Kind ▶ Two Pair ▶ Pair ▶ High Card
```

All rank comparisons use **lexicographic tuple ordering**, so `(8, 14) > (7, 14, 13)` correctly prefers a Royal Flush over Quad Aces with a King kicker.

The **wheel straight** (A-2-3-4-5) is handled by duplicating the Ace bit at position 1 in the `bit_mask`.

---

### Layer 2 — Equity Engine (Street-Adaptive)

#### Pre-flop: O(1) Baseline Table
A hand-coded 169-entry dictionary maps every canonical hand signature (`'AKs'`, `'72o'`, `'JJ'`, etc.) to its **heads-up win rate** in [0.0, 1.0]. Look-up is O(1) and costs zero CPU time.

```python
hand_key = _generate_signature(my_hand)  # → e.g. 'AKs'
equity   = _hole_card_baselines[hand_key] # → e.g. 0.670
```

#### Flop & Turn: Range-Weighted Monte Carlo

For each trial:
1. Sample a random opponent hand from the live deck (cards not in my hand or on board).
2. **Weight** the hand by its range-density given the opponent's betting pattern.
3. Draw the remaining runout cards.
4. Evaluate both hands via the bitwise evaluator.
5. Accumulate weighted wins / ties.

**Weighted equity:**

$$E = \frac{\sum_i w_i \cdot \mathbf{1}[\text{win}_i] + 0.5 \cdot w_i \cdot \mathbf{1}[\text{tie}_i]}{\sum_i w_i}$$

**Iteration count** is dynamically scaled by pot leverage:
| Leverage | MC Iterations |
|----------|--------------|
| > 40% of effective stack | 250 |
| > 15% of effective stack | 120 |
| Otherwise | 60 |

**Time guard**: every 20 iterations, elapsed time is checked. Loop aborts after 15 ms to stay within the 2-second budget.

#### River: Full Deterministic Enumeration

At the river, all 5 community cards are known. The bot iterates over the full C(N, 2) space of opponent hand combinations (where N = remaining deck size ≈ 44), weights each combo, and accumulates. **Results are memoised** in `_terminal_cache` so repeat queries within the same decision point are O(1).

---

### Layer 3 — Range Density Model

The **range density model** is the bridge between observed opponent betting and opponent hand distribution. It maps a hand bucket label to a probability weight under a given betting pattern.

#### Action Vector → Betting Profile

| Opponent Bet Size (pot%) | Action Signal | Strategic Interpretation |
|--------------------------|---------------|--------------------------|
| 0 (check) | `capped_passive` | Balanced/capped range |
| 1–74% | `merged_linear` | Merged hands, thin value |
| 75–119% | `standard_polar` | Polarised: value + bluffs |
| ≥ 120% | `hyper_polar` | Hyper-polarised overbets |

#### Range Density Table

```
                 nut    strong   medium   weak    draw    air
hyper_polar      1.00   0.20     0.10     0.03    0.02    0.01
standard_polar   1.00   0.60     0.20     0.05    0.10    0.02
merged_linear    1.00   0.90     0.60     0.20    0.30    0.10
capped_passive   0.80   0.75     0.70     0.55    0.50    0.30
```

Hand bucket assignment is driven by the bitwise evaluator's category rank (0–8) and board texture (paired / flush-possible).

---

### Layer 4 — Opponent Profiling & Archetype Classifier

The bot maintains a running statistical profile of the opponent updated on every hand:

| Metric | What It Measures |
|--------|-----------------|
| `adversary_vpip_ratio` | Voluntary pre-flop investment rate |
| `opp_pfr_events` | Pre-flop raise frequency |
| `opp_raises / opp_calls` | Aggression factor (AF) |
| `opp_folds_to_raise` | Fold-to-raise frequency |
| `adversary_hyperbet_ratio` | Overbet (>75% pot) frequency |
| `adversary_avg_bid` | Mean auction bid |
| `opp_3bet_events` | 3-bet frequency |
| `opp_went_to_showdown` | WTSD rate |

After **100 hands**, `_classify_opponent()` assigns one of **13 archetype labels**:

| Archetype | Key Pattern | Counter-Strategy |
|-----------|-------------|-----------------|
| `MAX_BIDDER` | Avg bid > 500 chips | Never bid — let them waste chips |
| `STATIC_BIDDER` | Fixed auction bid | Snipe with `bid + 1` |
| `INFO_BLEEDER` | Avg bid < 6 | Ignore auction, outplay post-flop |
| `3BET_MANIAC` | 3-bet rate > 15% | 4-bet trap with top 8% |
| `FIT_OR_FOLD` | Fold-to-raise > 65% | Bet every flop aggressively |
| `LAG_MANIAC` | VPIP > 60%, AF > 1.8 | Call down wider, value-bet thin |
| `CALLING_STATION` | VPIP > 50%, AF < 0.8 | Polarise bets, no bluffs |
| `NIT_PASSIVE` | VPIP < 25%, AF < 0.8 | Steal aggressively pre-flop |
| `NIT_AGGRESSIVE` | VPIP < 35%, AF > 1.5 | Call down value hands vs. bluffs |
| `PREFLOP_LIMPER` | VPIP > 60%, PFR < 10% | Isolate with raises |
| `OVERBET_ABUSER` | Hyperbet ratio > 30% | Tighten calling thresholds |
| `ABSTRACTION_STRONG/WEAK` | Fixed bet-size clusters | Exploit predicted sizing |
| `BALANCED_CFR` | No clear pattern | GTO baseline |

Archetype-specific **equity modifiers** are then applied in Stage 3 of `get_move()`, shifting the estimated win-rate by ±5–25 percentage points to express the exploitative counter-strategy.

**Positional archetypes** (`opp_sb_archetype`, `opp_bb_archetype`) are independently tracked with ≥30 hands per position.

---

### Layer 5 — GTO Auction Bidding Engine

The central innovation is the **equity-advantage bid formula**:

$$\text{bid}_{\text{GTO}} = \alpha \times 2.5 \times \max(0,\; E - 0.5) \times \text{pot}$$

- **Why `E - 0.5` rather than `E`?** A naive `E × pot` formula bids ~108% of pot at exactly 50% equity — paying a huge price for **zero informational edge**. The advantage formulation correctly bids **0 at a coin flip** and scales linearly with actual edge.
- **`α` (SPR scalar)**: At deep stacks (SPR > 5), `α = 1.2` — bid aggressively because any information advantage has many streets to be applied. At shallow stacks (SPR < 2), `α = 0.5` — the game is nearly over.

**Indifference threshold** (maximum rational bid):

$$b_{\max} = \frac{(E + 0.10) - (E - 0.06)}{1 - (E + 0.10)} \times \text{pot}$$

This is the bid at which the opponent is mathematically indifferent between winning and conceding the auction.

#### Auction Archetype Branches

| Archetype | Strategy |
|-----------|----------|
| `STATIC` | Snipe: predict opponent fixed bid via mode of history, bid `+1` |
| `HYPER_AGGRESSIVE` | Trap: all-in bid on E > 75%, GTO otherwise, concede on weak hands |
| `GTO_TRUTHFUL` | Full formula with nut/strong/marginal sub-branches |

**Gaussian noise** (`±5% of final_bid`) is added to prevent opponents from back-calculating our exact equity from bid amounts.

---

### Layer 6 — Pre-flop Position Logic

The pre-flop branch implements **position-aware opening ranges** calibrated against 169 hand percentiles from the baseline table.

#### Small Blind (SB) Opening

| Situation | Condition | Action |
|-----------|-----------|--------|
| Open vs BB `FOLDER` | Percentile ≥ 40% | Raise to 2.5× BB |
| Open vs BB `CALLING_STATION` | Percentile ≥ 55% | Raise to 3× BB |
| Open vs BB `3BET_DEFENDER` | Percentile ≥ 92% | Raise to 2.5× BB |
| Standard open | Percentile > 32% | Raise to 2.5× BB |
| SB vs 3-bet (BB re-raises) | Percentile ≥ 95% | 4-bet shove |

#### Big Blind (BB) Defence

| Opponent SB Style | 3-bet Conditions | Call-wide Threshold |
|-------------------|-----------------|---------------------|
| `SERIAL_THIEF` (raises > 65% hands) | Top 8% value, 20% suited-connector bluffs | Call top 55% |
| `SELECTIVE_AGG` | Top 5% only | Call top 45% |
| `LIMPER` | Never | Pot-odds + 3% |

---

### Layer 7 — Post-flop Defence & Attack

#### Raise Sizing

All raise targets pass through `_get_safe_raise()`:
```python
target += N(0, max(1, target × 0.05))   # Gaussian obfuscation
return clamp(target, min_raise, max_raise)
```

Bet sizes are calibrated to **pot×scalar** expressions:
- Standard value bet: 0.75–1.05× pot
- Nut hand river bet: 1.35–1.60× pot (leverage-adjusted)
- Underbet probe (near-nuts): 0.20× pot (induces bluff-raises)
- Bluff bet: 0.60–0.75× pot

#### River Brakes

A `river_brakes_active` flag blocks aggressive raises on the river when:
- Pot is very large (> 500 chips) and hand rank < 5 (below straight)
- Opponent has escalated twice (multiple raises on river street)
- Hand rank < 4 and equity < 90%

**Override**: if we have < 750 chips remaining, the brakes are lifted (pot-committed, push).

#### Minimum Defence Frequency (MDF)

Against aggressive opponents (`is_exploitative_opponent`), a **Minimum Defence Frequency** check prevents over-folding:

```
MDF = pot_before_bet / pot_after_bet × 0.85
# We defend the fraction of our range needed to make opponent's bluffs indifferent
```

Hands in the top `(1 - MDF)` percentile of our range always call regardless of other conditions.

---

## 🚀 Setup & Running Locally

### 1. Install Dependencies

```bash
cd bot-engine-2026
pip install -r requirements.txt
```

### 2. Configure Bots

Edit `config.py` to set the bot file paths:

```python
BOT_1_FILE = './bot.py'     # Your bot
BOT_2_FILE = './bot.py'     # Opponent (can be a copy or a different bot)
```

On macOS / Linux, you may need:
```python
PYTHON_CMD = "python3"
```

### 3. Run a Match

```bash
python engine.py
```

The engine will log each round's results to the `./logs/` folder.

### 4. Connect via Socket (Competition Mode)

```bash
python bot.py --host HOST --port PORT
```

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built for the IIT Pokerbots 2026 Competition · Engineered for performance, information theory, and exploitative adaptation.</sub>
</div>
