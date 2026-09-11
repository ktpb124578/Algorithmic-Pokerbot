"""bot.py — Primary decision-making engine for the IIT Pokerbots Competition 2026 submission.

This module implements the ``Player`` class, a highly sophisticated poker bot designed
for *Sneak Peek Hold'em* — a variant of No-Limit Texas Hold'em that incorporates a
post-flop sealed-bid, second-price auction for partial hole-card information.

Architecture Overview:
    * **Bitwise Hand Evaluator**: A compact, Cactus-Kev-inspired evaluator that operates
      entirely on integer bit-masks to rank any 5–7 card combination in microseconds,
      avoiding external library overhead.
    * **Weighted Monte Carlo Equity Engine**: A stochastic simulator that samples opponent
      hand distributions weighted by a *range density* model inferred from opponent
      betting patterns (VPIP, aggression factor, bet-sizing buckets).
    * **Deterministic River Evaluator**: A full enumeration pass over all remaining opponent
      hand combos at the river where the reduced card space makes exhaustive calculation
      feasible within the 2-second per-query time budget.
    * **Opponent Profiling & Archetype Classifier**: Tracks VPIP, PFR, AF, fold-to-raise,
      WTSD, 3-bet frequency, bid history, and positional tendencies to assign one of
      twelve opponent archetypes (e.g. ``LAG_MANIAC``, ``FIT_OR_FOLD``, ``STATIC_BIDDER``)
      after 100 hands, then applies archetype-specific equity modifiers.
    * **GTO Auction Bidding Engine**: Computes sealed bids using an equity-advantage
      formula (``E − 0.5``) rather than raw equity, ensuring the bot never overpays at
      coin-flip spots. Adapts bidding strategy per auction archetype.
    * **Positional Pre-flop Logic**: Position-aware opening ranges (SB/BB), 3-bet/fold
      responses, and BB-defence adjustments.
    * **Gaussian Bet-Size Masking**: Adds calibrated Gaussian noise to raise sizes to
      prevent opponents from exploiting fixed bet-size abstraction patterns.

Performance Budget:
    * ≤ 2 s per ``get_move`` call.
    * ≤ 20 s cumulative across all 1 000 rounds.
    * Internal time guards abort Monte Carlo loops after 15 ms.

Usage:
    Run directly via the engine runner::

        python bot.py --host HOST --port PORT
"""
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random
import time
from bisect import bisect_right
from itertools import combinations, islice

# ── Obfuscated Constants & Mappings ─────────────────────────────────────────────
_VAL_MAPPING = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
_STE_MAPPING = {'h': 0, 'd': 1, 'c': 2, 's': 3}

_R_INDICES = [i // 4 + 2 for i in range(52)]
_S_INDICES = [i % 4 for i in range(52)]

_DECK_MASTER_INTS = tuple((_VAL_MAPPING[r] - 2) * 4 + _STE_MAPPING[s] for r in '23456789TJQKA' for s in 'hdcs')
_CHAR_TO_INT = {r + s: (_VAL_MAPPING[r] - 2) * 4 + _STE_MAPPING[s] for r in '23456789TJQKA' for s in 'hdcs'}
_INT_TO_CHAR = {v: k for k, v in _CHAR_TO_INT.items()}

# ── Range Density Arrays ───────────────────────────────────────────────────────
_RANGE_DENSITIES = {
    'hyper_polar': {
        'nut': 1.00, 'strong': 0.20, 'medium': 0.10,
        'weak': 0.03, 'draw': 0.02, 'air': 0.01
    },
    'standard_polar': {
        'nut': 1.00, 'strong': 0.60, 'medium': 0.20,
        'weak': 0.05, 'draw': 0.10, 'air': 0.02
    },
    'merged_linear': {
        'nut': 1.00, 'strong': 0.90, 'medium': 0.60,
        'weak': 0.20, 'draw': 0.30, 'air': 0.10
    },
    'capped_passive': {
        'nut': 0.80, 'strong': 0.75, 'medium': 0.70,
        'weak': 0.55, 'draw': 0.50, 'air': 0.30
    },
}

def _compress_ranks(rank_array):
    """Pack a 13-element rank-count array into a single 52-bit integer.

    Each rank occupies a 4-bit nibble (bits ``idx*4 .. idx*4+3``), allowing
    counts 0–15 per rank.  This compressed form is used as the primary key
    for the bitwise hand evaluator so that pair/trip/quad detection reduces
    to single-integer bit-shift operations.

    Args:
        rank_array (list[int]): A length-13 list where ``rank_array[i]`` is
            the number of cards with rank ``i+2`` (2 → index 0, A → index 12)
            currently held in the combined hand+board.

    Returns:
        int: A 52-bit integer with each rank's count in its corresponding nibble.
    """
    val = 0
    for idx, count in enumerate(rank_array):
        val |= (count & 0xF) << (idx * 4)
    return val

def _compress_suits(suit_array):
    """Pack a 4-element suit-count array into a single 16-bit integer.

    Each suit occupies a 4-bit nibble in the order *hearts, diamonds, clubs,
    spades* (indices 0–3 as defined by ``_STE_MAPPING``).  A suit count ≥ 5
    in any nibble indicates a flush is present or possible.

    Args:
        suit_array (list[int]): A length-4 list where ``suit_array[s]`` is
            the number of cards of suit ``s`` in the current hand+board.

    Returns:
        int: A 16-bit integer with each suit's card-count in its nibble.
    """
    return (suit_array[0] & 0xF) | ((suit_array[1] & 0xF) << 4) | ((suit_array[2] & 0xF) << 8) | ((suit_array[3] & 0xF) << 12)

def _build_bitmask(cards):
    """Pre-compute all bitwise bookkeeping structures for a set of cards.

    This is the entry-point for building the four-tuple representation used
    throughout the evaluator.  It is called once per Monte Carlo trial for the
    *partial* board state and once for the *my_hand + board* state; subsequent
    incremental card additions use ``_evaluate_hand_bitwise`` directly to avoid
    re-scanning already-processed cards.

    Args:
        cards (tuple[int]): A tuple of card integers from ``_DECK_MASTER_INTS``.

    Returns:
        tuple: A 4-tuple ``(rank_compressed, suit_compressed, bit_mask, flush_masks)``
            where:
            * ``rank_compressed`` (int): Compressed rank-count nibble vector.
            * ``suit_compressed`` (int): Compressed suit-count nibble vector.
            * ``bit_mask`` (int): OR-mask of all card rank bits (ace = bit 14).
            * ``flush_masks`` (tuple[int, int, int, int]): Per-suit rank bitmasks
              used for straight-flush detection.
    """
    r_counts = [0] * 13; s_counts = [0] * 4; bit_mask = 0; flush_masks = [0, 0, 0, 0]
    for c in cards:
        r = _R_INDICES[c]; s = _S_INDICES[c]
        r_counts[r - 2] += 1; s_counts[s] += 1; bit_mask |= (1 << r); flush_masks[s] |= (1 << r)
    return _compress_ranks(r_counts), _compress_suits(s_counts), bit_mask, tuple(flush_masks)

def _evaluate_hand_bitwise(prc, psc, pmask, pfmasks, new_cards):
    """Incrementally evaluate a poker hand by merging new cards into existing bitmask state.

    Accepts the pre-computed bitmask structures for an existing partial hand and
    adds one or more ``new_cards``, then classifies the resulting best 5-card hand.
    The evaluator handles all standard hand rankings from High Card (0) to
    Straight Flush (8), including the Ace-low (wheel) straight A-2-3-4-5.

    Hand Rank Table:
        +----+---------------------+
        | 8  | Straight Flush      |
        +----+---------------------+
        | 7  | Four of a Kind      |
        +----+---------------------+
        | 6  | Full House          |
        +----+---------------------+
        | 5  | Flush               |
        +----+---------------------+
        | 4  | Straight            |
        +----+---------------------+
        | 3  | Three of a Kind     |
        +----+---------------------+
        | 2  | Two Pair            |
        +----+---------------------+
        | 1  | One Pair            |
        +----+---------------------+
        | 0  | High Card           |
        +----+---------------------+

    The return tuple is lexicographically comparable: ``(8, A) > (7, K, Q)`` correctly
    ranks a royal flush over quad kings with a queen kicker.

    Args:
        prc (int): Pre-existing compressed rank-count vector (from ``_build_bitmask``).
        psc (int): Pre-existing compressed suit-count vector.
        pmask (int): Pre-existing rank bitmask.
        pfmasks (tuple[int, int, int, int]): Per-suit rank bitmasks for SF detection.
        new_cards (tuple[int]): Additional card integers to fold into the evaluation.
            Pass an empty tuple ``()`` to evaluate ``prc/psc/pmask/pfmasks`` as-is.

    Returns:
        tuple[int, ...]: A lexicographically-sortable hand-rank tuple whose first
            element is the hand category (0–8) followed by tiebreaker ranks in
            descending order of significance.
    """
    rc = prc; sc = psc; mask = pmask
    fm0, fm1, fm2, fm3 = pfmasks

    for c in new_cards:
        r = _R_INDICES[c]; s = _S_INDICES[c]
        shift = (r - 2) * 4
        cnt = (rc >> shift) & 0xF
        rc = (rc & ~(0xF << shift)) | ((cnt + 1) << shift)
        sshift = s * 4
        scnt = (sc >> sshift) & 0xF
        sc = (sc & ~(0xF << sshift)) | ((scnt + 1) << sshift)
        mask |= (1 << r)
        if s == 0:   fm0 |= (1 << r)
        elif s == 1: fm1 |= (1 << r)
        elif s == 2: fm2 |= (1 << r)
        else:        fm3 |= (1 << r)

    flush_idx = -1
    if   (sc & 0xF)         >= 5: flush_idx = 0
    elif (sc >> 4  & 0xF)   >= 5: flush_idx = 1
    elif (sc >> 8  & 0xF)   >= 5: flush_idx = 2
    elif (sc >> 12 & 0xF)   >= 5: flush_idx = 3

    wheel_check = mask | (2 if mask & (1 << 14) else 0)

    if flush_idx >= 0:
        fm = (fm0, fm1, fm2, fm3)[flush_idx]
        fm_al = fm | (2 if fm & (1 << 14) else 0)
        for h in range(14, 4, -1):
            if (fm_al >> (h - 4)) & 0x1F == 0x1F: return (8, h)

    q_idx = t_idx = p1_idx = p2_idx = -1
    for i in range(12, -1, -1):
        v = (rc >> (i * 4)) & 0xF; r = i + 2
        if v >= 4 and q_idx < 0: q_idx = r
        elif v == 3 and t_idx < 0: t_idx = r
        elif v >= 2:
            if p1_idx < 0: p1_idx = r
            elif p2_idx < 0: p2_idx = r

    if q_idx >= 0:
        kicker = next(i + 2 for i in range(12, -1, -1) if ((rc >> (i * 4)) & 0xF) > 0 and i + 2 != q_idx)
        return (7, q_idx, kicker)
    if t_idx >= 0 and p1_idx >= 0: return (6, t_idx, p1_idx)

    if flush_idx >= 0:
        fm = (fm0, fm1, fm2, fm3)[flush_idx]
        t1=t2=t3=t4=t5=-1
        for h in range(14, 1, -1):
            if fm & (1 << h):
                if t1 < 0: t1 = h
                elif t2 < 0: t2 = h
                elif t3 < 0: t3 = h
                elif t4 < 0: t4 = h
                elif t5 < 0: t5 = h; break
        return (5, t1, t2, t3, t4, t5)

    for h in range(14, 4, -1):
        if (wheel_check >> (h - 4)) & 0x1F == 0x1F: return (4, h)

    if t_idx >= 0:
        k1 = k2 = -1
        for i in range(12, -1, -1):
            if ((rc >> (i * 4)) & 0xF) > 0 and (i + 2) != t_idx:
                if k1 < 0: k1 = i + 2
                else: k2 = i + 2; break
        return (3, t_idx, k1, k2)

    if p1_idx >= 0 and p2_idx >= 0:
        kicker = next(i + 2 for i in range(12, -1, -1) if ((rc >> (i * 4)) & 0xF) > 0 and i + 2 not in (p1_idx, p2_idx))
        return (2, p1_idx, p2_idx, kicker)

    if p1_idx >= 0:
        k1 = k2 = k3 = -1
        for i in range(12, -1, -1):
            if ((rc >> (i * 4)) & 0xF) > 0 and (i + 2) != p1_idx:
                if k1 < 0: k1 = i + 2
                elif k2 < 0: k2 = i + 2
                else: k3 = i + 2; break
        return (1, p1_idx, k1, k2, k3)

    t1=t2=t3=t4=t5=-1
    for i in range(12, -1, -1):
        if (rc >> (i * 4)) & 0xF:
            if t1 < 0: t1 = i + 2
            elif t2 < 0: t2 = i + 2
            elif t3 < 0: t3 = i + 2
            elif t4 < 0: t4 = i + 2
            elif t5 < 0: t5 = i + 2; break
    return (0, t1, t2, t3, t4, t5)

def _is_paired_board(prc: int) -> bool:
    """Determine whether the board contains at least one pair.

    Scans the compressed rank-count vector for any nibble ≥ 2, indicating
    two or more cards of the same rank on the board.  Used to discount
    trip and full-house draws from the opponent range classification.

    Args:
        prc (int): Compressed rank-count integer produced by ``_build_bitmask``
            for the community cards only (not including hole cards).

    Returns:
        bool: ``True`` if the board has at least one pair, ``False`` otherwise.
    """
    for i in range(13):
        if (prc >> (i * 4)) & 0xF >= 2: return True
    return False

def _is_flush_active(psc: int) -> bool:
    """Check whether any suit has three or more cards on the board.

    Three suited board cards means both a flush draw and a made flush are
    possible depending on hole cards.  This flag is used during range
    classification to determine whether an opponent holding a flush draw
    should be upgraded from 'weak' to 'medium'.

    Args:
        psc (int): Compressed suit-count integer produced by ``_build_bitmask``
            for the community cards only.

    Returns:
        bool: ``True`` if at least one suit appears three or more times, else ``False``.
    """
    return ((psc & 0xF) >= 3 or (psc >> 4 & 0xF) >= 3 or
            (psc >> 8 & 0xF) >= 3 or (psc >> 12 & 0xF) >= 3)

def _extract_suit_vol(psc: int, s: int) -> int:
    """Extract the card count for a single suit from the compressed suit vector.

    Args:
        psc (int): Compressed suit-count integer from ``_build_bitmask``.
        s (int): Suit index (0=hearts, 1=diamonds, 2=clubs, 3=spades).

    Returns:
        int: Number of cards of suit ``s`` present in the encoded hand/board.
    """
    return (psc >> (s * 4)) & 0xF

class Player(BaseBot):
    """Primary bot implementation for the IIT Pokerbots 2026 *Sneak Peek Hold'em* competition.

    Inherits from ``BaseBot`` and overrides three lifecycle hooks:

    * ``__init__``: Game-level initialisation of all tracking state.
    * ``on_hand_start``: Per-hand reset and position detection.
    * ``on_hand_end``: Post-hand statistics update and archetype classification.
    * ``get_move``: Core decision engine invoked on every action request.

    Key design pillars:
        1. **Adaptive equity estimation** — switches between bitwise preflop baselines,
           stochastic MC simulation (flop/turn), and full deterministic enumeration
           (river) based on computational budget.
        2. **Range-weighted opponent modelling** — Monte Carlo weights are modulated by
           a six-bucket range density function (nut/strong/medium/weak/draw/air) inferred
           from the opponent's betting action vector.
        3. **Sealed-bid auction logic** — GTO auction bids are computed using the
           *equity advantage* formulation: ``bid ≈ α × 2.5 × (E − 0.5) × pot``, ensuring
           zero bid at coin-flip spots and scaling linearly with informational edge.
        4. **Opponent archetype system** — 12 labelled archetypes selected after 100 hands
           each with tailored equity modifiers and bet-sizing responses.
        5. **Gaussian bet-size obfuscation** — all raise targets are perturbed with
           ``N(0, 0.05 × target)`` noise to defeat abstraction-based exploitation.
    """

    def __init__(self) -> None:
        """Initialise all game-level state counters, profiling arrays, and static lookup tables.

        This method is called exactly once at the start of a 1 000-round game.
        No game state is available at this point; all attributes here are either
        Bayesian priors (e.g. ``adversary_vpip_ratio = 0.50``) or empty containers
        that accumulate data as hands are played.

        Attributes initialised:
            total_encounters (int): Running count of completed hands.
            adversary_vpip_ratio (float): Estimated fraction of hands where opponent
                voluntarily invested chips pre-flop (initialised to 0.50 prior).
            adversary_hyperbet_ratio (float): Estimated overbet (>75 % pot) frequency.
            opponent_archetype (str): Current archetype label; starts as ``'PROFILING'``.
            _hole_card_baselines (dict[str, float]): Pre-computed 169-combo head-up
                equity table used for O(1) pre-flop equity look-up.
            _sorted_baselines (list[float]): Sorted list of baseline equities used
                for percentile-rank computation via ``bisect_right``.
        """
        self.total_encounters = 0

        self.adversary_vpip_events = 0
        self.adversary_vpip_ratio = 0.50
        self.adversary_wager_count = 0
        self.adversary_hyperbet_events = 0
        self.adversary_hyperbet_ratio = 0.15

        self.secured_intel_this_round = False
        self.leaked_intel_this_round = False
        self.initiated_raise = False
        self._terminal_cache: dict = {}

        self.adversary_terminal_escalations = 0
        self._prior_terminal_fee = 0
        self.adversary_aggression_vector = 0

        self.adversary_bid_volume = 0
        self.adversary_auction_participation = 0
        self.adversary_avg_bid = 5.0
        self.pot_memory_auction = 0
        self.auction_evaluated = False
        
        self._last_street_tracked = None
        self._last_cost_tracked = 0
        self._vpip_tracked = False
        self._last_auction_bid = 0
        
        # --- Advanced Exploitative Tracking ---
        self.opponent_bid_history = []
        self.opp_calls = 0
        self.opp_raises = 0
        
        # --- AUCTION PROFILING ARRAYS ---
        self._opp_bid_ratios = []        # opponent bid/pot (normalized)
        self._auction_pot_history = []   # pot size at each auction
        
        # --- MASTER TOGGLE FOR GOD MODE PATCHES ---
        self.use_god_mode_patches = True
        self.consecutive_auction_losses = 0

        # --- NEW: DEEP PROFILING ENGINE STATE ---
        self.opp_pfr_events = 0
        self.opponent_archetype = 'PROFILING' # Phase 1 state
        self.opp_folds_to_raise = 0
        self.opp_faces_raise = 0
        self.opp_went_to_showdown = 0
        self.opp_3bet_events = 0

        # --- PREFLOP POSITION PROFILING ---
        self.opp_sb_hands = 0
        self.opp_sb_raises = 0
        self.opp_sb_raise_sizes = []
        self.opp_sb_folds_to_3bet = 0
        self.opp_sb_3bet_opportunities = 0

        self.opp_bb_hands = 0
        self.opp_bb_3bets = 0
        self.opp_bb_folds = 0
        self.opp_bb_calls = 0

        self.opp_sb_archetype = 'PROFILING'
        self.opp_bb_archetype = 'PROFILING'

        self._my_position_this_round = None
        self._opp_raised_preflop = False
        self._i_opened_preflop = False
        self._3bet_sent = False
        self._bb_response_recorded = False

        # --- Static Threshold Abstraction Tracking ---
        self.opp_bet_ratios_history = []
        self.identified_opp_strong_bucket = -1.0
        self.identified_opp_weak_bucket = -1.0

        self._hole_card_baselines = {
            'AA': 0.853, 'KK': 0.824, 'QQ': 0.799, 'JJ': 0.775, 'TT': 0.751,
            '99': 0.721, '88': 0.691, '77': 0.662, '66': 0.633, '55': 0.603,
            '44': 0.570, '33': 0.537, '22': 0.503,
            'AKs': 0.670, 'AQs': 0.661, 'AJs': 0.654, 'ATs': 0.647, 'A9s': 0.630,
            'A8s': 0.621, 'A7s': 0.611, 'A6s': 0.600, 'A5s': 0.599, 'A4s': 0.589,
            'A3s': 0.580, 'A2s': 0.570,
            'KQs': 0.634, 'KJs': 0.626, 'KTs': 0.619, 'K9s': 0.600, 'K8s': 0.585,
            'K7s': 0.578, 'K6s': 0.568, 'K5s': 0.558, 'K4s': 0.547, 'K3s': 0.538,
            'K2s': 0.529,
            'QJs': 0.603, 'QTs': 0.595, 'Q9s': 0.579, 'Q8s': 0.562, 'Q7s': 0.545,
            'Q6s': 0.538, 'Q5s': 0.529, 'Q4s': 0.517, 'Q3s': 0.507, 'Q2s': 0.499,
            'JTs': 0.575, 'J9s': 0.558, 'J8s': 0.542, 'J7s': 0.524, 'J6s': 0.508,
            'J5s': 0.500, 'J4s': 0.490, 'J3s': 0.479, 'J2s': 0.471,
            'T9s': 0.543, 'T8s': 0.526, 'T7s': 0.510, 'T6s': 0.492, 'T5s': 0.472,
            'T4s': 0.464, 'T3s': 0.455, 'T2s': 0.447,
            '98s': 0.511, '97s': 0.495, '96s': 0.477, '95s': 0.459, '94s': 0.438,
            '93s': 0.432, '92s': 0.423,
            '87s': 0.482, '86s': 0.465, '85s': 0.448, '84s': 0.427, '83s': 0.408,
            '82s': 0.403,
            '76s': 0.457, '75s': 0.438, '74s': 0.418, '73s': 0.400, '72s': 0.381,
            '65s': 0.432, '64s': 0.414, '63s': 0.394, '62s': 0.375,
            '54s': 0.411, '53s': 0.393, '52s': 0.375,
            '43s': 0.380, '42s': 0.363, '32s': 0.351,
            'AKo': 0.654, 'AQo': 0.645, 'AJo': 0.636, 'ATo': 0.629, 'A9o': 0.609,
            'A8o': 0.601, 'A7o': 0.591, 'A6o': 0.578, 'A5o': 0.577, 'A4o': 0.564,
            'A3o': 0.556, 'A2o': 0.546,
            'KQo': 0.614, 'KJo': 0.606, 'KTo': 0.599, 'K9o': 0.580, 'K8o': 0.563,
            'K7o': 0.554, 'K6o': 0.543, 'K5o': 0.533, 'K4o': 0.521, 'K3o': 0.512,
            'K2o': 0.502,
            'QJo': 0.582, 'QTo': 0.574, 'Q9o': 0.555, 'Q8o': 0.538, 'Q7o': 0.519,
            'Q6o': 0.511, 'Q5o': 0.502, 'Q4o': 0.490, 'Q3o': 0.479, 'Q2o': 0.470,
            'JTo': 0.554, 'J9o': 0.534, 'J8o': 0.517, 'J7o': 0.499, 'J6o': 0.479,
            'J5o': 0.471, 'J4o': 0.461, 'J3o': 0.450, 'J2o': 0.440,
            'T9o': 0.517, 'T8o': 0.500, 'T7o': 0.482, 'T6o': 0.463, 'T5o': 0.442,
            'T4o': 0.434, 'T3o': 0.424, 'T2o': 0.415,
            '98o': 0.484, '97o': 0.467, '96o': 0.449, '95o': 0.429, '94o': 0.407,
            '93o': 0.399, '92o': 0.389,
            '87o': 0.455, '86o': 0.436, '85o': 0.417, '84o': 0.396, '83o': 0.375,
            '82o': 0.368,
            '76o': 0.427, '75o': 0.408, '74o': 0.386, '73o': 0.366, '72o': 0.318,
            '65o': 0.401, '64o': 0.380, '63o': 0.359, '62o': 0.340,
            '54o': 0.379, '53o': 0.358, '52o': 0.339,
            '43o': 0.344, '42o': 0.325, '32o': 0.312,
        }
        self._sorted_baselines = sorted(self._hole_card_baselines.values())

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        """Reset all per-hand state and detect positional assignment at the start of a round.

        Called exactly once per hand, before any ``get_move`` invocations for that hand.
        The method determines whether we are Small Blind (SB) or Big Blind (BB) by
        inspecting ``current_state.my_wager``:

        * ``my_wager == 10`` → we posted the small blind → ``_my_position_this_round = 'SB'``.
        * ``my_wager == 20`` → we posted the big blind  → ``_my_position_this_round = 'BB'``.

        The opponent's corresponding position counter is also incremented here so that
        positional archetype classifiers (``opp_sb_archetype``, ``opp_bb_archetype``)
        have accurate sample sizes.

        Args:
            game_info (GameInfo): Global game metadata (bankroll, time_bank, round_num).
            current_state (PokerState): Snapshot of the table at hand start, used
                solely for wager-based position detection.
        """
        self.total_encounters += 1
        self.secured_intel_this_round = False
        self.leaked_intel_this_round = False
        self.initiated_raise = False
        self.adversary_terminal_escalations = 0
        self._prior_terminal_fee = 0
        self.adversary_aggression_vector = 0

        self.pot_memory_auction = 0
        self.auction_evaluated = False
        
        self._last_street_tracked = None
        self._last_cost_tracked = 0
        self._vpip_tracked = False
        self._last_auction_bid = 0
        self._terminal_cache.clear()
        
        # Initialize traps
        self._preflop_trap_used = False
        self._postflop_offensive_trap_used = False
        
        # Reset per-hand trackers
        self._3bet_tracked_this_round = False
        self._my_position_this_round = None
        self._opp_raised_preflop = False
        self._i_opened_preflop = False
        self._3bet_sent = False
        self._bb_response_recorded = False

        # Detect position accurately at the start of the hand
        if current_state.my_wager == 10:
            self._my_position_this_round = 'SB'
            self.opp_bb_hands += 1
        elif current_state.my_wager == 20:
            self._my_position_this_round = 'BB'
            self.opp_sb_hands += 1

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        """Update cumulative opponent statistics after each hand concludes.

        Key updates performed (only when ``use_god_mode_patches`` is ``True``):

        1. **Fold-to-raise tracking**: If we raised during the hand
           (``initiated_raise``), checks whether the opponent's final wager is lower
           than ours (proxy for a fold) and increments ``opp_folds_to_raise``.
        2. **WTSD (Went-To-ShowDown)**: Detects river showdowns by checking that the
           hand reached 'river' street with equal wagers on both sides.
        3. **Call tracking**: Post-raise call detected when wagers are equal and both
           positive after we raised.
        4. **Archetype lock-in**: ``_classify_opponent()`` is triggered every 50 hands
           starting from hand 100 to progressively refine the opponent label as the
           sample grows.
        5. **Positional response tracking**: Records BB fold/call responses to SB
           opens, and SB fold responses to 3-bets, feeding the positional archetypes.

        Args:
            game_info (GameInfo): Global game metadata (bankroll, time_bank, round_num).
            current_state (PokerState): Final table state carrying ``payoff``,
                ``street``, ``my_wager``, and ``opp_wager``.
        """
        if getattr(self, 'use_god_mode_patches', True):
            # 1. Track Fold-to-Raise Frequency
            if self.initiated_raise:
                self.opp_faces_raise += 1
                # If our wager is higher at the end of the hand, it means they folded to us.
                if current_state.my_wager > current_state.opp_wager:
                    self.opp_folds_to_raise += 1

            # 2. Track Showdown Frequency (WTSD)
            if current_state.street == 'river' and current_state.my_wager == current_state.opp_wager:
                self.opp_went_to_showdown += 1

            # Track opponent calls
            if self.initiated_raise and current_state.my_wager == current_state.opp_wager and current_state.my_wager > 0:
                self.opp_calls += 1

            # 3. Lock in Archetype exactly at Hand 100
            if self.total_encounters >= 100 and self.total_encounters % 50 == 0:
                self._classify_opponent()

            # Track opponent BB response to my SB open (Folds vs Calls)
            if self._my_position_this_round == 'SB' and self._i_opened_preflop:
                if not getattr(self, '_bb_response_recorded', False):
                    if current_state.street == 'pre-flop':
                        self.opp_bb_folds += 1
                    else:
                        self.opp_bb_calls += 1

            # Track opponent SB fold to my 3-bet
            if self._my_position_this_round == 'BB' and self._3bet_sent:
                self.opp_sb_3bet_opportunities += 1
                if current_state.street == 'pre-flop':
                    self.opp_sb_folds_to_3bet += 1

    def _classify_opponent(self) -> None:
        """Assign a strategic archetype label to the opponent based on accumulated statistics.

        Decision tree evaluated in priority order:

        1. ``MAX_BIDDER``    — avg auction bid > 500 chips OR ≥10 consecutive auction losses.
        2. ``STATIC_BIDDER`` — last 10 bids are all identical (bot using a fixed bid script).
        3. ``INFO_BLEEDER``  — avg auction bid < 6 chips (ignores the auction mechanic).
        4. ``3BET_MANIAC``   — 3-bet frequency > 15 %.
        5. ``FIT_OR_FOLD``   — fold-to-raise > 65 % (exploitable with frequent c-bets).
        6. ``LAG_MANIAC``    — VPIP > 60 % AND aggression factor > 1.8.
        7. ``CALLING_STATION`` — VPIP > 50 %, AF < 0.8, fold-to-raise < 40 %.
        8. ``NIT_PASSIVE``   — VPIP < 25 %, AF < 0.8.
        9. ``NIT_AGGRESSIVE``— VPIP < 35 %, AF > 1.5.
        10. ``PREFLOP_LIMPER``— VPIP > 60 %, PFR < 10 % (enters lots but rarely raises).
        11. ``OVERBET_ABUSER``— hyperbet ratio > 30 %.
        12. ``ABSTRACTION_STRONG/WEAK`` — detected fixed bet-size clustering.
        13. ``BALANCED_CFR`` — default; no clear exploitative pattern detected.

        Positional archetypes for SB and BB are also updated independently after
        ≥ 30 hands in each position.

        Returns:
            None: Updates ``self.opponent_archetype``, ``self.opp_sb_archetype``,
                and ``self.opp_bb_archetype`` in-place.
        """
        vpip = self.adversary_vpip_ratio
        pfr = self.opp_pfr_events / max(1.0, float(self.total_encounters))
        af = self.opp_raises / max(1.0, float(self.opp_calls))
        avg_bid = self.adversary_avg_bid
        
        # New deep profiling metrics
        fold_to_raise = self.opp_folds_to_raise / max(1.0, float(self.opp_faces_raise))
        three_bet_rate = self.opp_3bet_events / max(1.0, float(self.total_encounters))

        if avg_bid > 500 or getattr(self, 'consecutive_auction_losses', 0) >= 10:
            self.opponent_archetype = 'MAX_BIDDER'
        elif len(set(self.opponent_bid_history[-10:])) == 1 and len(self.opponent_bid_history) >= 10:
            self.opponent_archetype = 'STATIC_BIDDER'
        elif avg_bid < 6.0:
            self.opponent_archetype = 'INFO_BLEEDER'
        elif three_bet_rate > 0.15:
            self.opponent_archetype = '3BET_MANIAC'
        elif fold_to_raise > 0.65:
            self.opponent_archetype = 'FIT_OR_FOLD'
        elif vpip > 0.60 and af > 1.8:
            self.opponent_archetype = 'LAG_MANIAC'
        elif vpip > 0.50 and af < 0.8 and fold_to_raise < 0.40:
            self.opponent_archetype = 'CALLING_STATION'
        elif vpip < 0.25 and af < 0.8:
            self.opponent_archetype = 'NIT_PASSIVE'
        elif vpip < 0.35 and af > 1.5:
            self.opponent_archetype = 'NIT_AGGRESSIVE'
        elif vpip > 0.60 and pfr < 0.10:
            self.opponent_archetype = 'PREFLOP_LIMPER'
        elif self.adversary_hyperbet_ratio > 0.30:
            self.opponent_archetype = 'OVERBET_ABUSER'
        elif self.identified_opp_strong_bucket > 0:
            self.opponent_archetype = 'ABSTRACTION_STRONG'
        elif self.identified_opp_weak_bucket > 0:
            self.opponent_archetype = 'ABSTRACTION_WEAK'
        else:
            self.opponent_archetype = 'BALANCED_CFR'
        # SB Archetype Classification
        if self.opp_sb_hands >= 30:
            sb_raise_rate = self.opp_sb_raises / self.opp_sb_hands
            if sb_raise_rate > 0.65:
                self.opp_sb_archetype = 'SERIAL_THIEF'
            elif 0.35 <= sb_raise_rate <= 0.65:
                self.opp_sb_archetype = 'SELECTIVE_AGG'
            elif sb_raise_rate < 0.20:
                self.opp_sb_archetype = 'LIMPER'
            else:
                self.opp_sb_archetype = 'BALANCED'

        # BB Archetype Classification
        if self.opp_bb_hands >= 30:
            total_bb = self.opp_bb_3bets + self.opp_bb_folds + self.opp_bb_calls
            if total_bb > 0:
                bb_3bet_rate = self.opp_bb_3bets / total_bb
                bb_fold_rate = self.opp_bb_folds / total_bb
                bb_call_rate = self.opp_bb_calls / total_bb
                if bb_3bet_rate > 0.20:
                    self.opp_bb_archetype = '3BET_DEFENDER'
                elif bb_fold_rate > 0.55:
                    self.opp_bb_archetype = 'FOLDER'
                elif bb_call_rate > 0.60 and bb_3bet_rate < 0.10:
                    self.opp_bb_archetype = 'CALLING_STATION_BB'
                else:
                    self.opp_bb_archetype = 'BALANCED'

    def _generate_signature(self, hand) -> str:
        """Canonicalise a two-card hole hand into a standard hand-range key.

        Converts a pair of card strings (e.g. ``['Ah', 'Kd']``) into the
        169-class canonical representation used as keys in ``_hole_card_baselines``:

        * Pocket pairs → ``"XX"`` (e.g. ``"AA"``, ``"72"``)
        * Suited hands → ``"XYs"`` (higher rank first, e.g. ``"AKs"``)
        * Offsuit hands → ``"XYo"`` (e.g. ``"AKo"``)

        Args:
            hand (list[str]): Two-element list of card strings in the format
                ``"<rank><suit>"`` (e.g. ``['Ah', 'Kd']``).

        Returns:
            str: Canonicalised hand signature string (e.g. ``'AKs'``, ``'72o'``).
        """
        r1, r2 = hand[0][0], hand[1][0]
        s1, s2 = hand[0][1], hand[1][1]
        if _VAL_MAPPING[r1] < _VAL_MAPPING[r2]: r1, r2 = r2, r1
        if r1 == r2: return r1 + r2
        elif s1 == s2: return r1 + r2 + 's'
        return r1 + r2 + 'o'

    def _find_equity_strata(self, eq: float) -> float:
        """Return the percentile rank of an equity value within the 169-hand preflop baseline.

        Uses binary search (``bisect_right``) on the pre-sorted ``_sorted_baselines``
        array for O(log 169) look-up.  The result represents the fraction of all
        starting hands weaker than or equal to the given equity, giving a position-agnostic
        hand-strength percentile used for range-construction heuristics.

        Example:
            An equity of 0.75 might return 0.90, meaning the hand is stronger than
            90 % of all starting hand combos in heads-up play.

        Args:
            eq (float): Win-rate equity in ``[0.0, 1.0]``.

        Returns:
            float: Percentile rank in ``[0.0, 1.0]``.
        """
        return bisect_right(self._sorted_baselines, eq) / len(self._sorted_baselines)

    def _get_safe_raise(self, target: int, min_r: int, max_r: int) -> int:
        """Clamp a desired raise amount to legal bounds with Gaussian noise obfuscation.

        Applies ``N(0, σ)`` noise where ``σ = max(1, target × 0.05)`` to the target
        raise before clamping to ``[min_r, max_r]``.  The perturbation prevents the
        opponent from inferring exact hand strength from repeated fixed bet sizes
        (a form of cost abstraction exploitation common in competition bots).

        Args:
            target (int): Desired raise amount in chips before noise or clamping.
            min_r (int): Engine-enforced minimum legal raise.
            max_r (int): Engine-enforced maximum legal raise (typically remaining stack).

        Returns:
            int: A legal raise amount in ``[min_r, max_r]`` with added noise.
        """
        if min_r > max_r:
            min_r = max_r
        # Gaussian Strategy Masking to prevent Abstraction Exploitation
        target = int(target + random.gauss(0, max(1, target * 0.05)))
        return min(max_r, max(min_r, int(target)))

    def _classify_enemy_range(self, opp_hand_i: tuple, board_i: tuple, board_partial, board_paired: bool, flush_possible: bool) -> str:
        """Map a specific opponent hand to one of six range-density buckets.

        Buckets are: ``'nut'``, ``'strong'``, ``'medium'``, ``'weak'``, ``'draw'``, ``'air'``.
        The function serves as the weight-generation oracle for Monte Carlo trials: a
        hand labelled ``'nut'`` receives a density (probability the opponent would
        actually hold that hand given observed betting) proportional to the current
        ``signal`` (action vector), while ``'air'`` receives minimal weight.

        Pre-flop (no board cards):
            Falls back to the 169-combo equity baseline table.  Buckets are assigned
            at fixed equity thresholds: ≥0.72 → nut, ≥0.58 → strong, etc.

        Post-flop:
            Uses the bitwise evaluator's category rank (0–8) as the primary discriminant,
            then adjusts for board texture:
            * Trips on a paired board are downgraded from ``'nut'`` to ``'strong'`` because
              the board pair reduces the relative nuts available to the opponent.
            * A flush on a flush-possible board is only ``'strong'`` (may not be the nut flush).
            * Pairs / high cards plus active draws are upgraded from ``'weak'`` to ``'medium'``.

        Args:
            opp_hand_i (tuple[int, int]): Two-card integer representation of the simulated
                opponent hand.
            board_i (tuple[int, ...]): Community cards as integers (empty on pre-flop).
            board_partial (tuple): Pre-computed bitmask 4-tuple for the board alone.
            board_paired (bool): Whether the board has a pair (from ``_is_paired_board``).
            flush_possible (bool): Whether three+ suited cards are on the board.

        Returns:
            str: One of ``{'nut', 'strong', 'medium', 'weak', 'draw', 'air'}``.
        """
        if not board_i:
            eq = self._hole_card_baselines.get(self._generate_signature((_INT_TO_CHAR[opp_hand_i[0]], _INT_TO_CHAR[opp_hand_i[1]])), 0.5)
            if eq >= 0.72: return 'nut'
            elif eq >= 0.58: return 'strong'
            elif eq >= 0.47: return 'medium'
            elif eq >= 0.38: return 'weak'
            elif eq >= 0.32: return 'draw'
            return 'air'

        rank_val = _evaluate_hand_bitwise(*board_partial, opp_hand_i)[0]

        is_flush_draw = False
        if flush_possible:
            for s in range(4):
                opp_suited = sum(1 for c in opp_hand_i if _S_INDICES[c] == s)
                board_suited = sum(1 for c in board_i if _S_INDICES[c] == s)
                if board_suited + opp_suited == 4:
                    is_flush_draw = True
                    break

        is_oesd = False
        combined_mask = 0
        for c in board_i + opp_hand_i:
            combined_mask |= (1 << _R_INDICES[c])
        if combined_mask & (1 << 14):
            combined_mask |= (1 << 1)
        for h in range(14, 4, -1):
            window = (combined_mask >> (h - 4)) & 0x1F
            if bin(window).count('1') >= 4:
                is_oesd = True
                break

        has_draw = is_flush_draw or is_oesd

        if rank_val >= 7: return 'nut'
        elif rank_val == 6: return 'nut'
        elif rank_val == 5: return 'strong' if flush_possible else 'nut'
        elif rank_val == 4: return 'strong'
        elif rank_val == 3: return 'nut' if not board_paired else 'strong'
        elif rank_val == 2: return 'medium'
        elif rank_val == 1: return 'weak' if not has_draw else 'medium'
        else:
            if has_draw: return 'draw'
            return 'air'

    def _extract_density(self, opp_hand_i: tuple, board_i: tuple, board_partial, signal: str, board_paired: bool, flush_possible: bool) -> float:
        """Return the range-density weight for a single opponent hand given an action signal.

        Acts as a thin pipeline from hand → bucket → density, combining
        ``_classify_enemy_range`` and the ``_RANGE_DENSITIES`` lookup table.
        The weight represents how likely the opponent is to hold ``opp_hand_i`` given
        that they made the observed bet sizing (encoded in ``signal``).

        Args:
            opp_hand_i (tuple[int, int]): Two-card integer opponent hand.
            board_i (tuple[int, ...]): Community cards as integers.
            board_partial (tuple): Pre-computed bitmask 4-tuple for the board.
            signal (str): Action vector key matching a ``_RANGE_DENSITIES`` row;
                one of ``'hyper_polar'``, ``'standard_polar'``, ``'merged_linear'``,
                ``'capped_passive'``.
            board_paired (bool): Whether the board has a pair.
            flush_possible (bool): Whether a flush draw exists on board.

        Returns:
            float: A non-negative density weight in ``[0.0, 1.0]``.
        """
        bucket = self._classify_enemy_range(opp_hand_i, board_i, board_partial, board_paired, flush_possible)
        return _RANGE_DENSITIES[signal][bucket]

    def _determine_action_vector(self, bet_ratio: float) -> str:
        """Map the opponent's observed bet-to-pot ratio to a range density profile.

        The mapping encodes a simplified GTO range theory:

        * ``0``           → ``'capped_passive'``  — check/limp implies a capped, balanced range.
        * ``[0, 0.75)``   → ``'merged_linear'``   — small bets are merged/linear ranges.
        * ``[0.75, 1.2)`` → ``'standard_polar'``  — pot-sized bets are typically polarised.
        * ``≥ 1.2``       → ``'hyper_polar'``     — overbets are hyper-polarised (nuts or bluffs).

        Args:
            bet_ratio (float): ``cost_to_call / actual_pot`` for the current street.

        Returns:
            str: A key into ``_RANGE_DENSITIES``; one of ``'capped_passive'``,
                ``'merged_linear'``, ``'standard_polar'``, or ``'hyper_polar'``.
        """
        if bet_ratio == 0: return 'capped_passive'
        if bet_ratio >= 1.2: return 'hyper_polar'
        if bet_ratio >= 0.75: return 'standard_polar'
        return 'merged_linear'

    def _stochastic_simulation(self, my_hand_i: tuple, board_i: tuple, opp_revealed_i: tuple, signal: str, iterations: int) -> float:
        """Estimate win-rate equity via range-weighted Monte Carlo simulation.

        For each of ``iterations`` trials, a random opponent hand is sampled
        uniformly from the live deck (cards not held by us or on the board).
        The hand is then *weighted* by its range-density (how likely the opponent
        would make the observed bet with that holding), and the remaining board
        runout cards are drawn to determine the winner.

        The weighted win-rate is:

        .. math::

            E = \\frac{\\sum_i w_i \\cdot \\mathbb{1}[\\text{win}_i] + 0.5 \\cdot w_i \\cdot \\mathbb{1}[\\text{tie}_i]}{\\sum_i w_i}

        If the opponent revealed a hole card (via auction win), ``opp_revealed_i``
        fixes that card and samples only the second opponent card, narrowing the
        distribution to the reduced opponent range.

        Time Guard:
            Every 20 iterations the elapsed time is checked (``self.move_start_time``)
            and the loop is aborted after 15 ms to stay within the per-move budget.

        Args:
            my_hand_i (tuple[int, int]): Bot's own hole cards as deck integers.
            board_i (tuple[int, ...]): Current community cards as deck integers.
                Empty tuple on pre-flop; this triggers an O(1) table look-up instead.
            opp_revealed_i (tuple): One-card tuple if an opponent card was revealed
                via auction, else empty tuple.
            signal (str): Range density profile key (from ``_determine_action_vector``).
            iterations (int): Maximum number of MC trials (adaptive: 60–250 based
                on pot leverage).

        Returns:
            float: Weighted win-rate in ``[0.0, 1.0]``; 0.5 on degenerate empty weight.
        """
        if not board_i:
            return self._hole_card_baselines.get(self._generate_signature((_INT_TO_CHAR[my_hand_i[0]], _INT_TO_CHAR[my_hand_i[1]])), 0.500)

        dead = set(my_hand_i + board_i)
        if opp_revealed_i: dead.update(opp_revealed_i)
        deck = tuple(c for c in _DECK_MASTER_INTS if c not in dead)
        dl = len(deck)
        cards_needed = 5 - len(board_i)

        my_partial = _build_bitmask(my_hand_i + board_i)
        board_partial = _build_bitmask(board_i)

        board_paired = _is_paired_board(board_partial[0])
        flush_possible = _is_flush_active(board_partial[1])

        rand_range = random.randrange
        weighted_wins = 0.0
        total_weight = 0.0
        weight_cache = {}

        for i in range(iterations):
            if i % 20 == 0 and hasattr(self, 'move_start_time') and (time.time() - self.move_start_time) > 0.015:
                break

            if opp_revealed_i:
                oi2 = rand_range(dl)
                opp_i = (opp_revealed_i[0], deck[oi2])
                skip1 = oi2; skip2 = -1
            else:
                oi1 = rand_range(dl)
                oi2 = rand_range(dl - 1)
                if oi2 >= oi1: oi2 += 1
                opp_i = (deck[oi1], deck[oi2])
                skip1, skip2 = (oi1, oi2) if oi1 < oi2 else (oi2, oi1)

            opp_key = opp_i if opp_i[0] < opp_i[1] else (opp_i[1], opp_i[0])
            if opp_key in weight_cache:
                weight = weight_cache[opp_key]
            else:
                weight = self._extract_density(opp_i, board_i, board_partial, signal, board_paired, flush_possible)
                weight_cache[opp_key] = weight

            if cards_needed == 1:
                if skip2 == -1:
                    j = rand_range(dl - 1)
                    if j >= skip1: j += 1
                else:
                    j = rand_range(dl - 2)
                    if j >= skip1: j += 1
                    if j >= skip2: j += 1
                sim = (deck[j],)
            elif cards_needed == 2:
                if skip2 == -1:
                    j1 = rand_range(dl - 1); j2 = rand_range(dl - 2)
                    if j2 >= j1: j2 += 1
                    r1 = j1 + (1 if j1 >= skip1 else 0)
                    r2 = j2 + (1 if j2 >= skip1 else 0)
                else:
                    j1 = rand_range(dl - 2); j2 = rand_range(dl - 3)
                    if j2 >= j1: j2 += 1
                    r1 = j1 + (1 if j1 >= skip1 else 0); r1 += (1 if r1 >= skip2 else 0)
                    r2 = j2 + (1 if j2 >= skip1 else 0); r2 += (1 if r2 >= skip2 else 0)
                sim = (deck[r1], deck[r2])
            else:
                sim = ()

            mr = _evaluate_hand_bitwise(*my_partial, sim)
            or_ = _evaluate_hand_bitwise(*board_partial, opp_i + sim)

            if mr > or_: weighted_wins += weight
            elif mr == or_: weighted_wins += weight * 0.5
            total_weight += weight

        return weighted_wins / total_weight if total_weight > 0 else 0.5

    def _deterministic_river(self, my_hand_i: tuple, board_i: tuple, opp_revealed_i: tuple, signal: str, step: int) -> float:
        """Compute range-weighted win-rate by full enumeration on the river.

        At the river, all five community cards are known, so no runout sampling is
        needed.  The method iterates over all valid opponent two-card combos in the
        live deck, weights each by range density, and accumulates wins/ties.

        When ``step == 1`` the full combo space is evaluated (C(N,2) where N is
        the remaining deck size after removing dead cards).  A step > 1 subsamples
        via ``islice`` to accelerate evaluation when time is limited.

        Caching:
            Results are memoised in ``self._terminal_cache`` keyed by the full
            ``(my_hand_i, board_i, opp_revealed_i, signal, step)`` tuple so that
            repeated calls within the same hand (e.g. a raise/call sequence) are O(1).

        Time Guard:
            The combo enumeration loop checks elapsed time every 20 iterations and
            exits early after 15 ms to respect the per-move latency budget.

        Args:
            my_hand_i (tuple[int, int]): Bot's hole cards as deck integers.
            board_i (tuple[int, ...]): All five community cards as deck integers.
            opp_revealed_i (tuple): One-card tuple if opponent card was revealed;
                empty tuple otherwise.
            signal (str): Range density profile key.
            step (int): Stride for combo enumeration; 1 = full enumeration.

        Returns:
            float: Weighted win-probability in ``[0.0, 1.0]``.
        """
        cache_key = (my_hand_i, board_i, opp_revealed_i, signal, step)
        if cache_key in self._terminal_cache: return self._terminal_cache[cache_key]

        dead = set(my_hand_i + board_i)
        if opp_revealed_i: dead.update(opp_revealed_i)
        deck = tuple(c for c in _DECK_MASTER_INTS if c not in dead)

        if opp_revealed_i: combos = ((opp_revealed_i[0], c) for c in deck)
        else: combos = islice(combinations(deck, 2), 0, None, step)

        my_partial = _build_bitmask(my_hand_i + board_i)
        my_rank = _evaluate_hand_bitwise(*my_partial, ())
        board_partial = _build_bitmask(board_i)

        board_paired = _is_paired_board(board_partial[0])
        flush_possible = _is_flush_active(board_partial[1])

        weighted_wins = 0.0
        total_weight = 0.0
        weight_cache = {}

        for idx, opp_hand_i in enumerate(combos):
            if idx % 20 == 0 and hasattr(self, 'move_start_time') and (time.time() - self.move_start_time) > 0.015:
                break

            opp_key = opp_hand_i if opp_hand_i[0] < opp_hand_i[1] else (opp_hand_i[1], opp_hand_i[0])
            if opp_key in weight_cache:
                weight = weight_cache[opp_key]
            else:
                weight = self._extract_density(opp_hand_i, board_i, board_partial, signal, board_paired, flush_possible)
                weight_cache[opp_key] = weight

            opp_rank = _evaluate_hand_bitwise(*board_partial, opp_hand_i)
            if my_rank > opp_rank: weighted_wins += weight
            elif my_rank == opp_rank: weighted_wins += weight * 0.5
            total_weight += weight

        result = weighted_wins / total_weight if total_weight > 0 else 0.5
        self._terminal_cache[cache_key] = result
        return result

    def _holds_nut_blocker(self, my_hand_i: tuple, board_i: tuple) -> bool:
        """Check whether we hold the nut-flush blocker for any active flush suit.

        A *nut blocker* here is defined as holding the Ace of a suit where three or
        more board cards share that suit.  Holding the nut-flush blocker has two
        strategic implications:

        * On the river when facing an overbet, the blocker reduces the probability
          that the opponent holds the nut flush, making a call more defensible.
        * When choosing to bluff, holding the blocker improves bluff credibility
          because our range plausibly contains the nut flush.

        Args:
            my_hand_i (tuple[int, int]): Bot's hole cards as deck integers.
            board_i (tuple[int, ...]): Community cards as deck integers.

        Returns:
            bool: ``True`` if we hold the Ace of any suit with ≥3 board cards.
        """
        board_partial = _build_bitmask(board_i)
        psc = board_partial[1]
        for s in range(4):
            if _extract_suit_vol(psc, s) >= 3:
                ace_of_suit = (_VAL_MAPPING['A'] - 2) * 4 + s
                if ace_of_suit in my_hand_i:
                    return True
        return False

    def _get_spr_alpha(self, pot, current_chips, opp_chips):
        """Compute the Stack-to-Pot Ratio (SPR) scalar for auction bid sizing.

        The SPR is computed on the *effective* stack (the lesser of both players'
        remaining chips) to avoid over-committing against a shorter stack.  The
        returned ``alpha`` linearly scales the GTO bid formula:

        * **SPR > 5** (deep stacks): ``alpha = 1.2`` — bid aggressively; the risk
          of over-investing is low relative to stack depth, and information gained
          from seeing an opponent card has a long horizon to be exploited.
        * **SPR < 2** (shallow / near all-in): ``alpha = 0.5`` — bid conservatively;
          pot-commitment effects dominate and the information advantage has little
          time to be applied before all chips are in.
        * **2 ≤ SPR ≤ 5**: Linear interpolation between 0.5 and 1.2.

        Args:
            pot (int): Current pot size in chips at the start of the auction phase.
            current_chips (int): Bot's remaining chip count.
            opp_chips (int): Opponent's remaining chip count.

        Returns:
            float: Scaling factor ``alpha`` in ``[0.5, 1.2]`` for the GTO bid formula.
        """
        eff_stack = min(current_chips, opp_chips)
        spr = eff_stack / max(1, pot)
        if spr > 5:   return 1.2
        elif spr < 2: return 0.5
        else:         return 0.5 + (spr - 2.0) / 3.0 * 0.7

    def _get_fold_equity_factor(self):
        """Estimate the opponent's fold-to-raise frequency with a sample-size guard.

        Returns the fraction of times the opponent folds when facing a raise, clamped
        to ``[0.05, 0.85]`` to avoid degenerate extremes.  If fewer than 10 data
        points have been recorded, a conservative prior of 0.40 is returned instead
        of using a potentially noisy small-sample estimate.

        Returns:
            float: Estimated fold frequency in ``[0.05, 0.85]``.
        """
        if self.opp_faces_raise > 10:
            fold_freq = self.opp_folds_to_raise / self.opp_faces_raise
        else:
            fold_freq = 0.40
        return max(0.05, min(0.85, fold_freq))

    def _classify_auction_archetype(self):
        """Classify the opponent's auction bidding style into one of three profiles.

        Analyses the rolling bid history window to identify stereotyped bidding patterns
        that allow deterministic or near-deterministic counter-strategies:

        * ``'STATIC'``          — Bid variance < 15 over the last 15 auctions AND ≥5
          samples available.  Opponent uses a scripted constant or near-constant bid,
          enabling exact snipe via ``_get_static_bid_prediction``.
        * ``'HYPER_AGGRESSIVE'``— Average bid/pot ratio > 0.65 over the last 10 auctions.
          Opponent almost always overbids; counter-strategy is to trap with nuts and
          concede cheaply on marginal hands.
        * ``'GTO_TRUTHFUL'``    — Default; no exploitable pattern detected; use the
          GTO equity-advantage formula.

        Returns:
            str: One of ``'UNKNOWN'``, ``'STATIC'``, ``'HYPER_AGGRESSIVE'``,
                or ``'GTO_TRUTHFUL'``.
        """
        n = len(self.opponent_bid_history)
        if n < 3: return 'UNKNOWN'
        window = self.opponent_bid_history[-min(15, n):]
        mean_b = sum(window) / len(window)
        variance = sum((b - mean_b) ** 2 for b in window) / len(window)
        if variance < 15.0 and n >= 5: return 'STATIC'
        ratios = self._opp_bid_ratios[-min(10, len(self._opp_bid_ratios)):]
        if len(ratios) >= 8 and sum(ratios) / len(ratios) > 0.65: return 'HYPER_AGGRESSIVE'
        return 'GTO_TRUTHFUL'

    def _get_static_bid_prediction(self):
        """Predict the opponent's next auction bid when a ``'STATIC'`` pattern is detected.

        Uses a mode (most-frequent-value) heuristic on the last 10 observed bids to
        predict the opponent's fixed bid value, then adds 1 chip to guarantee a win
        in the second-price auction while minimising the price paid.

        Returns:
            int: Predicted bid that beats the opponent by exactly 1 chip, or ``-1``
                if no bid history is available.
        """
        if not self.opponent_bid_history: return -1
        window = self.opponent_bid_history[-min(10, len(self.opponent_bid_history)):]
        return max(set(window), key=window.count) + 1

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        """Core decision engine — called by the game runner on every action request.

        This is the single most time-critical entry point.  It must return an action
        within the 2-second per-query wall-clock limit.  The method executes a
        deterministic pipeline of stages in strict order:

        Stage 1 — State Parsing:
            Convert card strings to integer deck indices, detect auction intelligence
            status (``secured_intel_this_round``, ``leaked_intel_this_round``), and update
            opponent profiling metrics (VPIP, hyperbet ratio, aggression vector).

        Stage 2 — Equity Computation:
            Select the appropriate evaluator:
            * **River** → ``_deterministic_river`` (full enumeration over all opponent combos).
            * **Flop/Turn** → ``_stochastic_simulation`` (range-weighted Monte Carlo).
            * **Pre-flop** → ``_hole_card_baselines`` table look-up (O(1)).
            The number of MC iterations is dynamically set by pot leverage:
            high-leverage pots (> 40 % of effective stack) run 250 iterations;
            low-leverage pots run 60.

        Stage 3 — Archetype Equity Modifiers:
            After hand 100, archetype-specific equity shifts are applied.  Examples:
            * ``CALLING_STATION``: equity ± 0.15 to discourage thin calls and promote
              value-betting.
            * ``FIT_OR_FOLD``: equity +0.25 when facing a check (free bluff equity).

        Stage 4 — Required Equity Calculation:
            The minimum equity to profitably call is: ``cost / (pot + cost)``.  A
            ``buffer`` (3–5 pp) and an optional ``leverage_penalty`` (5 pp when the pot
            exceeds 35 % of effective stack) are added to form ``effective_required``.
            Overbets further increase ``effective_required`` by an ``overbet_penalty``.

        Stage 5 — Street-Specific Action Selection:
            Dedicated logic branches for:
            * ``'pre-flop'``: Position-aware opening/3-bet/defence strategy.
            * ``'auction'``: GTO bid formula with archetype-specific overrides.
            * ``'flop'`` / ``'turn'`` / ``'river'``: Raise/call/fold decisions using
              ``safe_equity`` vs ``effective_required``, with river-specific escalation
              detection and bluff-frequency controls.

        Args:
            game_info (GameInfo): Global game metadata (bankroll, time_bank, round_num).
            current_state (PokerState): Full table snapshot including hole cards, board,
                pot, cost-to-call, raise bounds, chip counts, and revealed opponent cards.

        Returns:
            ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
                The chosen action for this decision point.
        """
        self.move_start_time = time.time()
        
        phase = current_state.street
        pot = current_state.pot
        cost = current_state.cost_to_call
        current_chips = current_state.my_chips

        if phase == 'pre-flop' and current_state.opp_wager > 20:
            self.opp_pfr_events += 1

        # --- ADDED: 3-BET TRACKING FOR ARCHETYPES ---
        if phase == 'pre-flop' and current_state.opp_wager > 30 and current_state.my_wager > 0:
            if not getattr(self, '_3bet_tracked_this_round', False):
                self.opp_3bet_events += 1
                self._3bet_tracked_this_round = True

        my_hand_i = tuple(_CHAR_TO_INT[c] for c in current_state.my_hand)
        board_i = tuple(_CHAR_TO_INT[c] for c in current_state.board)
        opp_rev_i = tuple(_CHAR_TO_INT[c] for c in current_state.opp_revealed_cards) if current_state.opp_revealed_cards else ()

        if not self.secured_intel_this_round and not self.leaked_intel_this_round and phase not in ('pre-flop', 'auction'):
            if opp_rev_i:
                self.secured_intel_this_round = True
                auction_added_chips = pot - self.pot_memory_auction
                if auction_added_chips == 2 * self._last_auction_bid and self._last_auction_bid > 0:
                    self.leaked_intel_this_round = True
            else:
                self.leaked_intel_this_round = True

        if not self.auction_evaluated and phase not in ('pre-flop', 'auction'):
            self.auction_evaluated = True
            if self.pot_memory_auction > 0:
                if getattr(self, 'use_god_mode_patches', True):
                    paid_into_pot = pot - current_state.opp_wager - current_state.my_wager - self.pot_memory_auction
                    ref_pot = self._auction_pot_history[-1] if self._auction_pot_history else max(1, pot)
                    if self.secured_intel_this_round and not self.leaked_intel_this_round:
                        actual_opp_bid = paid_into_pot
                        self.opponent_bid_history.append(actual_opp_bid)
                        self._opp_bid_ratios.append(actual_opp_bid / max(1, ref_pot))
                        self.adversary_auction_participation += 1
                        self.adversary_bid_volume += actual_opp_bid
                        self.adversary_avg_bid = self.adversary_bid_volume / self.adversary_auction_participation
                        self.consecutive_auction_losses = 0
                    elif self.secured_intel_this_round and self.leaked_intel_this_round:
                        half = paid_into_pot // 2
                        self.opponent_bid_history.append(half)
                        self._opp_bid_ratios.append(half / max(1, ref_pot))
                        self.consecutive_auction_losses = 0
                    else:
                        lb = self._last_auction_bid + 1
                        self.opponent_bid_history.append(lb)
                        self._opp_bid_ratios.append(lb / max(1, ref_pot))
                        self.consecutive_auction_losses += 1
                else:
                    if self.secured_intel_this_round:
                        paid = pot - current_state.opp_wager - current_state.my_wager - self.pot_memory_auction
                        self.adversary_auction_participation += 1
                        self.adversary_bid_volume += max(0, paid)
                        self.adversary_avg_bid = self.adversary_bid_volume / self.adversary_auction_participation
                        self.opponent_bid_history.append(paid)
                    else:
                        lower_bound = self._last_auction_bid + 1
                        if lower_bound > self.adversary_avg_bid:
                            self.adversary_avg_bid = (self.adversary_avg_bid * 0.8) + (lower_bound * 0.2)
                        self.opponent_bid_history.append(self._last_auction_bid + 1)

        if phase == 'pre-flop' and current_state.opp_wager > 20 and not self._vpip_tracked:
            self.adversary_vpip_events += 1
            self._vpip_tracked = True
        self.adversary_vpip_ratio = (self.adversary_vpip_events + 2.5) / (self.total_encounters + 5.0)

        actual_pot = max(1, pot - cost)
        bet_ratio = cost / actual_pot if actual_pot > 0 else 0

        if cost > 0 and phase != 'pre-flop' and phase != 'auction':
            if self._last_street_tracked != phase or cost > self._last_cost_tracked:
                self.opp_bet_ratios_history.append(round(bet_ratio, 2))
                if len(self.opp_bet_ratios_history) > 10:
                    recent_ratios = self.opp_bet_ratios_history[-10:]
                    ratio_counts = {r: recent_ratios.count(r) for r in set(recent_ratios)}
                    for r, count in ratio_counts.items():
                        if count >= 4:
                            if r >= 0.8: self.identified_opp_strong_bucket = r
                            elif r <= 0.5: self.identified_opp_weak_bucket = r

        if cost > 0 and phase not in ('pre-flop', 'auction'):
            if self._last_street_tracked != phase or cost > self._last_cost_tracked:
                self.opp_raises += 1
                if self._last_street_tracked != phase:
                    self.adversary_wager_count += 1
                    if bet_ratio > 0.75: self.adversary_hyperbet_events += 1
                    self.adversary_hyperbet_ratio = (self.adversary_hyperbet_events + 1.0) / (self.adversary_wager_count + 6.0)
                    self.adversary_aggression_vector += 1
                
                self._last_street_tracked = phase
                self._last_cost_tracked = cost

        wager_freq = self.adversary_wager_count / max(1.0, float(self.total_encounters))
        is_exploitative_opponent = (wager_freq > 0.35) and (self.adversary_hyperbet_ratio > 0.20)

        signal = self._determine_action_vector(bet_ratio)

        my_rank_category = -1
        is_board_paired = False
        if phase not in ('pre-flop', 'auction'):
            my_partial_eval = _build_bitmask(my_hand_i + board_i)
            my_rank_category = _evaluate_hand_bitwise(*my_partial_eval, ())[0]
            board_partial_eval = _build_bitmask(board_i)
            is_board_paired = _is_paired_board(board_partial_eval[0])

        leverage = pot / max(1, current_chips + pot)
        if leverage > 0.40 or pot > 200: mc_iters = 250; river_step = 1
        elif leverage > 0.15 or pot > 80: mc_iters = 120; river_step = 1
        else: mc_iters = 60; river_step = 1

        if phase == 'river':
            eval_equity = self._deterministic_river(my_hand_i, board_i, opp_rev_i, signal, river_step)
        else:
            eval_equity = self._stochastic_simulation(my_hand_i, board_i, opp_rev_i, signal, mc_iters)

        if phase == 'pre-flop':
            hand_key = self._generate_signature(current_state.my_hand)
            base_eq = self._hole_card_baselines.get(hand_key, eval_equity)
            pressure = min(0.25, pot / max(1, current_chips + 1))
            eval_equity = max(0.30, eval_equity - pressure * 0.20)
        else:
            base_eq = eval_equity

        # --- EXPLOITATIVE ARCHETYPE MODIFIERS (Applied after hand 100) ---
        if getattr(self, 'use_god_mode_patches', True) and self.total_encounters > 100:
            if self.opponent_archetype == 'CALLING_STATION':
                if eval_equity < 0.60: eval_equity -= 0.15
                elif eval_equity >= 0.60: eval_equity += 0.15
            elif self.opponent_archetype == 'FIT_OR_FOLD':
                if cost == 0: eval_equity += 0.25
                elif cost > 0: eval_equity -= 0.15
            elif self.opponent_archetype == 'NIT_PASSIVE':
                if cost == 0: eval_equity += 0.20
                elif cost > 0: eval_equity -= 0.20
            elif self.opponent_archetype == 'LAG_MANIAC':
                if cost > 0 and (my_rank_category >= 1 or phase == 'pre-flop'): eval_equity += 0.15
            elif self.opponent_archetype == 'PREFLOP_LIMPER' and phase == 'pre-flop':
                base_eq += 0.15
            elif self.opponent_archetype == '3BET_MANIAC' and phase == 'pre-flop':
                if cost > 0 and base_eq > 0.65: eval_equity += 0.20
        # -----------------------------------------------------------------

        required_equity = cost / (pot + cost) if (pot + cost) > 0 else 0.0
        buffer = 0.05 if phase == 'flop' else 0.03 if phase == 'turn' else 0.05
        safe_equity = eval_equity - buffer

        leverage_penalty = 0.0
        if leverage > 0.35 and phase != 'pre-flop': leverage_penalty = 0.05
        effective_required = required_equity + leverage_penalty

        if bet_ratio > 0.90 and phase != 'pre-flop':
            overbet_penalty = max(1.0, min(1.12, 1.15 - (self.adversary_hyperbet_ratio * 0.25)))
            effective_required *= overbet_penalty

        river_brakes_active = False

        if phase == 'river':
            if cost > self._prior_terminal_fee and cost > 0: self.adversary_terminal_escalations += 1
            self._prior_terminal_fee = cost

            if not self.secured_intel_this_round:
                if bet_ratio > 1.2 and cost > 0 and not self._holds_nut_blocker(my_hand_i, board_i):
                    base_downgrade = 0.08 if self.adversary_aggression_vector >= 3 else 0.05
                    eval_equity -= base_downgrade
                    safe_equity -= base_downgrade

                if self.adversary_aggression_vector >= 3 and bet_ratio >= 1.0 and cost > 0:
                    eval_equity -= 0.05
                    safe_equity -= 0.05

                if is_board_paired and my_rank_category <= 1 and cost > 0:
                    eval_equity -= 0.05
                    safe_equity -= 0.05

            required_equity = cost / (pot + cost) if (pot + cost) > 0 else 0.0
            effective_required = required_equity + leverage_penalty

            if getattr(self, 'use_god_mode_patches', True):
                if (pot + cost) > 500 and my_rank_category < 5 and eval_equity < 0.95:
                    river_brakes_active = True
            else:
                if (pot + cost) > 500 and my_rank_category < 6 and eval_equity < 0.95:
                    river_brakes_active = True

            if self.adversary_terminal_escalations >= 1 and my_rank_category < 4 and eval_equity < 0.90:
                river_brakes_active = True

            # SPR pot-commitment override — if nearly all-in, brakes are irrelevant
            if current_chips < 750:
                river_brakes_active = False  

        # --- POST-FLOP OFFENSIVE TRAP ---
        if phase in ('flop', 'turn', 'river') and cost == 0:
            if not getattr(self, '_postflop_offensive_trap_used', False):
                self._postflop_offensive_trap_used = True
                if current_state.can_act(ActionRaise):
                    target_raise = int(pot * random.uniform(1.8, 2.2))
                    min_raise, max_raise = current_state.raise_bounds
                    self.initiated_raise = True
                    return ActionRaise(self._get_safe_raise(target_raise, min_raise, max_raise))
        # --------------------------------

        if phase in ('flop', 'turn', 'river') and cost > 0:
            if getattr(self, 'use_god_mode_patches', True):
                if my_rank_category == 0 and bet_ratio > 0.40 and eval_equity < 0.45:
                    if current_state.can_act(ActionFold): return ActionFold()
                if my_rank_category <= 1 and bet_ratio > 0.85 and eval_equity < 0.65:
                    if current_state.can_act(ActionFold): return ActionFold()
                if self.adversary_terminal_escalations >= 1 and my_rank_category < 3 and bet_ratio > 0.70 and eval_equity < 0.80:
                    if current_state.can_act(ActionFold): return ActionFold()

        if phase == 'pre-flop':
            _pf_chunk = self._find_equity_strata(base_eq)
            opp_wager = current_state.opp_wager
            my_wager  = current_state.my_wager
            bb_size = 20
            sb_size = 10

            # --- PRE-FLOP TRAP ---
            if not getattr(self, '_preflop_trap_used', False):
                self._preflop_trap_used = True
                if opp_wager <= 40 and current_state.can_act(ActionRaise):
                    target = random.randint(120, 150)
                    min_raise, max_raise = current_state.raise_bounds
                    if min_raise <= target:
                        self._i_opened_preflop = True
                        self.initiated_raise = True
                        return ActionRaise(self._get_safe_raise(target, min_raise, max_raise))
            # ---------------------

            # Track opponent SB raise (when I am BB)
            if self._my_position_this_round == 'BB' and not self._opp_raised_preflop:
                if opp_wager > bb_size:
                    self._opp_raised_preflop = True
                    self.opp_sb_raises += 1
                    self.opp_sb_raise_sizes.append(opp_wager)
            
            # Track BB 3-bet response mid-preflop (when I am SB)
            if (self._my_position_this_round == 'SB' 
                    and self._i_opened_preflop 
                    and not getattr(self, '_bb_response_recorded', False)):
                if opp_wager > my_wager:
                    self.opp_bb_3bets += 1
                    self._bb_response_recorded = True

            is_sb_open = (self._my_position_this_round == 'SB' and cost > 0 and cost <= sb_size and current_state.can_act(ActionRaise) and not self._i_opened_preflop)
            if is_sb_open:
                min_raise, max_raise = current_state.raise_bounds
                if self.total_encounters > 100 and self.opp_bb_archetype == 'FOLDER':
                    if _pf_chunk >= 0.40:
                        self._i_opened_preflop = True
                        self.initiated_raise = True
                        return ActionRaise(self._get_safe_raise(int(bb_size * 2.5), min_raise, max_raise))
                    elif _pf_chunk >= 0.15:
                        if current_state.can_act(ActionCall): return ActionCall()
                    return ActionFold()
                elif self.total_encounters > 100 and self.opp_bb_archetype == 'CALLING_STATION_BB':
                    if _pf_chunk >= 0.55:
                        self._i_opened_preflop = True
                        self.initiated_raise = True
                        return ActionRaise(self._get_safe_raise(int(bb_size * 3), min_raise, max_raise))
                    elif _pf_chunk >= 0.25:
                        if current_state.can_act(ActionCall): return ActionCall()
                    return ActionFold()
                elif self.total_encounters > 100 and self.opp_bb_archetype == '3BET_DEFENDER':
                    if _pf_chunk >= 0.92:
                        self._i_opened_preflop = True
                        self.initiated_raise = True
                        return ActionRaise(self._get_safe_raise(int(bb_size * 2.5), min_raise, max_raise))
                    elif _pf_chunk >= 0.60:
                        self._i_opened_preflop = True
                        self.initiated_raise = True
                        return ActionRaise(self._get_safe_raise(min_raise, min_raise, max_raise))
                    elif _pf_chunk >= 0.25:
                        if current_state.can_act(ActionCall): return ActionCall()
                    return ActionFold()
                else:
                    if _pf_chunk > 0.32:
                        self._i_opened_preflop = True
                        self.initiated_raise = True
                        return ActionRaise(self._get_safe_raise(int(bb_size * 2.5), min_raise, max_raise))
                    elif _pf_chunk > 0.25:
                        if random.random() < 0.60:
                            self._i_opened_preflop = True
                            self.initiated_raise = True
                            return ActionRaise(self._get_safe_raise(int(bb_size * 2.5), min_raise, max_raise))
                        else: return ActionCall()
                    elif _pf_chunk > 0.12:
                        if current_state.can_act(ActionCall): return ActionCall()
                        return ActionFold()
                    else:
                        return ActionFold()

            is_sb_vs_3bet = (my_wager > bb_size and cost > 0)
            if is_sb_vs_3bet:
                open_size = max(1, my_wager)
                three_bet_multiple = (my_wager + cost) / open_size
                if three_bet_multiple < 2.5: call_threshold = 0.30
                elif three_bet_multiple < 4.0: call_threshold = 0.50
                else: call_threshold = 0.70
                if _pf_chunk >= 0.95:
                    if current_state.can_act(ActionRaise):
                        min_raise, max_raise = current_state.raise_bounds
                        return ActionRaise(self._get_safe_raise(int(pot * 2.5), min_raise, max_raise))
                if _pf_chunk >= call_threshold:
                    if current_state.can_act(ActionCall): return ActionCall()
                return ActionFold()

            is_bb_vs_limp = (my_wager == bb_size and cost == 0 and opp_wager > 0 and opp_wager <= bb_size and current_state.can_act(ActionCheck) and current_state.can_act(ActionRaise))
            if is_bb_vs_limp:
                min_raise, max_raise = current_state.raise_bounds
                if _pf_chunk > 0.65:
                    return ActionRaise(self._get_safe_raise(int(pot * 2.5), min_raise, max_raise))
                elif _pf_chunk > 0.25:
                    if current_state.can_act(ActionCheck): return ActionCheck()
                    if current_state.can_act(ActionCall): return ActionCall()
                else:
                    if current_state.can_act(ActionCheck): return ActionCheck()
                    if current_state.can_act(ActionFold): return ActionFold()
            else:
                if cost == 0:
                    if current_state.can_act(ActionCheck): return ActionCheck()
                pot_odds_threshold = cost / (pot + cost) if (pot + cost) > 0 else 0.0

                if self.total_encounters > 100 and self.opp_sb_archetype == 'SERIAL_THIEF':
                    # 3-bet value: top 8%
                    if _pf_chunk >= 0.92 and current_state.can_act(ActionRaise):
                        min_raise, max_raise = current_state.raise_bounds
                        self.initiated_raise = True
                        self._3bet_sent = True
                        return ActionRaise(self._get_safe_raise(cost * 3, min_raise, max_raise))
                    # 3-bet bluff: suited connectors range ~20%
                    if 0.55 <= _pf_chunk <= 0.70 and current_state.can_act(ActionRaise):
                        if random.random() < 0.20:
                            min_raise, max_raise = current_state.raise_bounds
                            self.initiated_raise = True
                            self._3bet_sent = True
                            return ActionRaise(self._get_safe_raise(cost * 3, min_raise, max_raise))
                    # Call wide with top 55% — punish steal frequency
                    if _pf_chunk >= 0.45:
                        if current_state.can_act(ActionCall): return ActionCall()
                    return ActionFold()
                
                elif self.total_encounters > 100 and self.opp_sb_archetype == 'SELECTIVE_AGG':
                    if _pf_chunk >= 0.95 and current_state.can_act(ActionRaise):
                        min_raise, max_raise = current_state.raise_bounds
                        self.initiated_raise = True
                        self._3bet_sent = True
                        return ActionRaise(self._get_safe_raise(cost * 3, min_raise, max_raise))
                    if _pf_chunk >= 0.55:
                        if current_state.can_act(ActionCall): return ActionCall()
                    return ActionFold()

                elif self.total_encounters > 100 and self.opp_sb_archetype == 'LIMPER':
                    req_eq = pot_odds_threshold + 0.03
                    if base_eq >= req_eq:
                        if current_state.can_act(ActionCall): return ActionCall()
                    return ActionFold()
                else:
                    req_eq = pot_odds_threshold + 0.05
                    req_eq = min(req_eq, 0.52)
                    if base_eq >= req_eq:
                        if current_state.can_act(ActionCall): return ActionCall()
                    return ActionFold()

        if phase == 'auction':
            if self.pot_memory_auction == 0:
                self.pot_memory_auction = pot
                self._auction_pot_history.append(pot)

                E_raw = eval_equity

                # Ensure hand rank is available
                if my_rank_category < 0 and board_i:
                    my_partial_eval = _build_bitmask(my_hand_i + board_i)
                    my_rank_category = _evaluate_hand_bitwise(*my_partial_eval, ())[0]

                # GTO Wizard baseline: use equity ADVANTAGE (E_raw - 0.5), not raw equity.
                # Raw equity bids 108% of pot at 50% equity — paying huge for zero edge.
                # Advantage correctly bids 0 at coinflip and scales with actual edge.
                alpha = self._get_spr_alpha(pot, current_chips, current_state.opp_chips)
                equity_advantage = max(0.0, E_raw - 0.50)
                _raw_gto = int(alpha * 2.5 * equity_advantage * pot)
                _info_floor = int(pot * 0.10) if E_raw >= 0.50 else 0
                GTO_bid = max(_info_floor, _raw_gto)

                # Indifference threshold — on nut hands this approaches the stack limit
                E_info = min(0.98, E_raw + 0.10)
                E_loss = max(0.02, E_raw - 0.06)
                b_max = int(pot * (E_info - E_loss) / max(0.001, 1.0 - E_info))

                archetype = self._classify_auction_archetype()

                # BRANCH A — Static bidder: snipe from round 3 onwards, always win cheaply
                if archetype == 'STATIC':
                    static_bid = self._get_static_bid_prediction()
                    proposed_bid = static_bid if 0 < static_bid <= int(current_chips * 0.50) else GTO_bid

                # BRANCH B — Hyper-aggressive: Trap at nut, GTO baseline otherwise
                elif archetype == 'HYPER_AGGRESSIVE':
                    if E_raw > 0.75:
                        proposed_bid = int(min(current_chips, current_state.opp_chips) * 0.95)
                    elif E_raw >= 0.50:
                        proposed_bid = GTO_bid  # still contest, just don't overpay
                    else:
                        proposed_bid = 0

                # BRANCH C — GTO/Unknown
                else:
                    if board_i:
                        bp = _build_bitmask(board_i)
                        wet = _is_flush_active(bp[1]) or _is_paired_board(bp[0])
                    else:
                        wet = False

                    # NUT HAND: hyper-bid — cost approaches zero, force them to pay your price
                    if E_raw > 0.90:
                        proposed_bid = b_max

                    # STRONG HAND: bid aggressively above GTO baseline
                    elif E_raw > 0.65:
                        proposed_bid = int(GTO_bid * 1.25)

                    # MARGINAL on wet board: minimal bid — don't pay much but don't gift free info
                    elif my_rank_category <= 1 and wet and E_raw < 0.45:
                        proposed_bid = int(pot * 0.05)

                    else:
                        proposed_bid = GTO_bid

                # Apply b_max ceiling only to non-nut, non-trap bids
                if proposed_bid > 0 and archetype != 'HYPER_AGGRESSIVE' and E_raw <= 0.90:
                    proposed_bid = min(proposed_bid, b_max)

                # Safety wrapper: max(0, min(B, my_chips, opp_chips))
                final_bid = max(0, min(int(proposed_bid), current_chips, current_state.opp_chips))
                if final_bid > 5:
                    final_bid = max(0, min(final_bid + random.choice([-1, 0, 0, 1]),
                                          current_chips, current_state.opp_chips))

                self._last_auction_bid = final_bid

            return ActionBid(self._last_auction_bid)

        if self.leaked_intel_this_round and not self.secured_intel_this_round and phase in ('flop', 'turn', 'river'):
            _leak_penalty = 0.08 if bet_ratio >= 1.5 else 0.05
            eval_equity = eval_equity - _leak_penalty
            safe_equity = safe_equity - _leak_penalty

        if getattr(self, 'use_god_mode_patches', True) and self.secured_intel_this_round and phase in ('flop', 'turn'):
            if current_state.can_act(ActionRaise) and not river_brakes_active:
                min_raise, max_raise = current_state.raise_bounds
                if eval_equity >= 0.85:
                    self.initiated_raise = True
                    return ActionRaise(self._get_safe_raise(int(pot * 1.0) + cost, min_raise, max_raise))
                elif eval_equity >= 0.65 and cost == 0:
                    self.initiated_raise = True
                    return ActionRaise(self._get_safe_raise(int(pot * 0.60), min_raise, max_raise))

        if self.secured_intel_this_round and phase == 'river':
            if current_state.can_act(ActionRaise) and not river_brakes_active:
                min_raise, max_raise = current_state.raise_bounds

                if eval_equity >= 0.92:
                    multiplier = 1.35 if leverage > 0.50 else 1.60
                    self.initiated_raise = True
                    return ActionRaise(self._get_safe_raise(int(pot * multiplier) + cost, min_raise, max_raise))
                elif eval_equity >= 0.85:
                    self.initiated_raise = True
                    return ActionRaise(self._get_safe_raise(int(pot * 1.35) + cost, min_raise, max_raise))
                elif eval_equity >= 0.75:
                    self.initiated_raise = True
                    return ActionRaise(self._get_safe_raise(int(pot * 0.75) + cost, min_raise, max_raise))
                elif eval_equity >= 0.55:
                    self.initiated_raise = True
                    return ActionRaise(self._get_safe_raise(int(pot * 0.50) + cost, min_raise, max_raise))

            if eval_equity >= 0.45:
                if cost == 0 and current_state.can_act(ActionCheck): return ActionCheck()
                if current_state.can_act(ActionCall): return ActionCall()
            elif 0.30 <= eval_equity < 0.45:
                board_partial = _build_bitmask(board_i)
                board_paired = _is_paired_board(board_partial[0])
                flush_possible = _is_flush_active(board_partial[1])
                has_blocker = self._holds_nut_blocker(my_hand_i, board_i)

                bluff_freq = 0.15 if has_blocker else 0.07
                if (board_paired or flush_possible) and random.random() < bluff_freq and not river_brakes_active:
                    if current_state.can_act(ActionRaise):
                        min_raise, max_raise = current_state.raise_bounds
                        self.initiated_raise = True
                        return ActionRaise(self._get_safe_raise(int(pot * 0.65), min_raise, max_raise))
                if current_state.can_act(ActionCheck) and cost == 0: return ActionCheck()
                if current_state.can_act(ActionFold): return ActionFold()
                return ActionCheck()
            else:
                if current_state.can_act(ActionCheck) and cost == 0: return ActionCheck()
                if current_state.can_act(ActionFold): return ActionFold()
                return ActionCheck()

        if current_state.can_act(ActionRaise):
            min_raise, max_raise = current_state.raise_bounds
            raise_eq_threshold = 0.65 + (0.50 - self.adversary_vpip_ratio) * 0.10
            medium_raises_allowed = leverage <= 0.35

            if phase == 'river' and eval_equity >= 0.96 and pot > 50 and not river_brakes_active:
                self.initiated_raise = True
                underbet_target = int(pot * 0.20)
                if getattr(self, 'use_god_mode_patches', True):
                    underbet_target += random.choice([1, 3, 7, 11])
                    return ActionRaise(self._get_safe_raise(underbet_target, min_raise, max_raise))
                else:
                    return ActionRaise(self._get_safe_raise(underbet_target, min_raise, max_raise))

            if not is_exploitative_opponent and phase in ('turn', 'river') and cost == 0 and not river_brakes_active:
                if 0.45 <= eval_equity <= 0.70 and my_rank_category >= 1:
                    if random.random() < 0.25: 
                        self.initiated_raise = True
                        return ActionRaise(self._get_safe_raise(int(pot * 1.05), min_raise, max_raise))

            if phase == 'river' and eval_equity >= 0.93 and self.adversary_vpip_ratio > 0.50 and pot > 120 and not river_brakes_active:
                self.initiated_raise = True
                return ActionRaise(self._get_safe_raise(int(pot * 1.20 * 0.85) + cost, min_raise, max_raise))

            if medium_raises_allowed and eval_equity > 0.72 and safe_equity > effective_required + 0.08:
                scale = 0.50 + (safe_equity - raise_eq_threshold) * 2.0
                scale *= 0.85
                self.initiated_raise = True
                return ActionRaise(self._get_safe_raise(int(pot * random.uniform(scale - 0.1, scale + 0.1)) + cost, min_raise, max_raise))

            if eval_equity > 0.70 and cost == 0:
                self.initiated_raise = True
                return ActionRaise(self._get_safe_raise(int(pot * 0.75 * 0.85) + cost, min_raise, max_raise))

            if phase == 'turn' and cost == 0 and signal == 'capped_passive' and pot > 40 and eval_equity > 0.55:
                hand_chunk = self._find_equity_strata(eval_equity)
                if hand_chunk > 0.65:
                    self.initiated_raise = True
                    return ActionRaise(self._get_safe_raise(int(pot * random.uniform(0.55, 0.65) * 0.85), min_raise, max_raise))

            if phase == 'river' and cost == 0 and 0.68 <= eval_equity <= 0.74 and self.adversary_vpip_ratio > 0.55 and medium_raises_allowed and not river_brakes_active:
                self.initiated_raise = True
                return ActionRaise(self._get_safe_raise(int(pot * 0.50 * 0.85), min_raise, max_raise))

            if phase == 'river' and cost == 0 and 0.30 <= eval_equity <= 0.48 and signal != 'standard_polar' and medium_raises_allowed and not river_brakes_active:
                board_partial = _build_bitmask(board_i)
                board_paired = _is_paired_board(board_partial[0])
                flush_possible = _is_flush_active(board_partial[1])
                if (board_paired or flush_possible) and random.random() < 0.07:
                    self.initiated_raise = True
                    return ActionRaise(self._get_safe_raise(int(pot * random.uniform(0.60, 0.75) * 0.85), min_raise, max_raise))
            
            if self.secured_intel_this_round and cost == 0 and eval_equity > 0.55 and medium_raises_allowed:
                self.initiated_raise = True
                return ActionRaise(self._get_safe_raise(int(pot * random.uniform(0.45, 0.65)), min_raise, max_raise))

        if phase == 'flop' and cost > 0:
            # --- EXPLOITATIVE FLOP TRAP LOGIC ---
            if bet_ratio >= 1.5:
                req_odds = cost / (pot + cost) if (pot + cost) > 0 else 0
                if eval_equity >= req_odds or my_rank_category <= 1:
                    if current_state.can_act(ActionCall): return ActionCall()
                else:
                    if current_state.can_act(ActionFold): return ActionFold()
            # ------------------------------------
            if bet_ratio >= 0.45 and eval_equity < 0.50:
                if current_state.can_act(ActionFold): return ActionFold()
            if cost >= 50 and eval_equity < 0.60:
                if current_state.can_act(ActionFold): return ActionFold()
                
        if phase in ('turn', 'river') and cost > 0:
            if bet_ratio >= 2.0:
                if eval_equity < 0.82:
                    if current_state.can_act(ActionFold): return ActionFold()
                    
            if bet_ratio >= 1.5 and my_rank_category < 6 and eval_equity < 0.95:
                if current_state.can_act(ActionFold): return ActionFold()
                
            if is_board_paired and my_rank_category <= 2 and bet_ratio >= 0.60 and eval_equity < 0.65:
                if current_state.can_act(ActionFold): return ActionFold()
                
            if is_exploitative_opponent:
                actual_pot_before_bet = max(1, pot - cost)
                mdf = actual_pot_before_bet / max(1, pot)
                adjusted_mdf = mdf * 0.85
                hand_percentile = self._find_equity_strata(eval_equity)
                is_mdf_defense = hand_percentile >= (1.0 - adjusted_mdf)
                
                if bet_ratio >= 1.5 and eval_equity < 0.85:
                    is_mdf_defense = False
                    
                if bet_ratio >= 0.70 and my_rank_category == 0:
                    is_mdf_defense = False
                elif bet_ratio >= 1.0 and eval_equity < 0.75:
                    is_mdf_defense = False
                elif cost > 40 and eval_equity < 0.65:
                    is_mdf_defense = False
                
                if is_mdf_defense:
                    pass 
                else:
                    if bet_ratio >= 0.70 and eval_equity < 0.65:
                        if current_state.can_act(ActionFold): return ActionFold()
                    if bet_ratio >= 1.0 and eval_equity < 0.80:
                        if current_state.can_act(ActionFold): return ActionFold()
            else:
                if bet_ratio > 0.50 and my_rank_category == 0:
                    if current_state.can_act(ActionFold): return ActionFold()
                if bet_ratio > 0.70 and my_rank_category == 1 and self.adversary_aggression_vector >= 2:
                    if current_state.can_act(ActionFold): return ActionFold()

                is_weak_two_pair = (my_rank_category == 2 and is_board_paired)
                if bet_ratio >= 1.0 and (my_rank_category < 2 or is_weak_two_pair):
                    if current_state.can_act(ActionFold): return ActionFold()

                if bet_ratio >= 1.2 and eval_equity < 0.85:
                    if current_state.can_act(ActionFold): return ActionFold()
                if bet_ratio >= 0.9 and eval_equity < 0.65:
                    if current_state.can_act(ActionFold): return ActionFold()

        if cost > 0 and phase in ('flop', 'turn', 'river'):
            current_rounded_ratio = round(bet_ratio, 2)
            if self.identified_opp_strong_bucket > 0 and abs(current_rounded_ratio - self.identified_opp_strong_bucket) < 0.05:
                if eval_equity < 0.85 and current_state.can_act(ActionFold):
                    return ActionFold()
            
            if self.identified_opp_weak_bucket > 0 and abs(current_rounded_ratio - self.identified_opp_weak_bucket) < 0.05:
                if eval_equity > 0.60 and current_state.can_act(ActionRaise):
                    min_raise, max_raise = current_state.raise_bounds
                    self.initiated_raise = True
                    return ActionRaise(self._get_safe_raise(int(pot * 2.0) + cost, min_raise, max_raise))

        if safe_equity >= effective_required:
            if cost == 0 and current_state.can_act(ActionCheck): return ActionCheck()
            if current_state.can_act(ActionCall): return ActionCall()

        if current_state.can_act(ActionCheck) and cost == 0: return ActionCheck()
        if current_state.can_act(ActionFold): return ActionFold()

        if current_state.can_act(ActionCheck): return ActionCheck()
        if current_state.can_act(ActionCall): return ActionCall()
        if current_state.can_act(ActionFold): return ActionFold()
        return ActionCall()

if __name__ == '__main__':
    run_bot(Player(), parse_args())