"""engine.py — Official IIT Pokerbots 2026 match engine (v1.0.0).

.. warning::
    **DO NOT REMOVE, RENAME, OR EDIT THIS FILE.**
    This is the authoritative match engine distributed by the IIT Pokerbots
    competition organisers.  Only documentation (comments and docstrings) has
    been added; all functional logic is unchanged.

The engine orchestrates a complete 1 000-round heads-up Sneak Peek Hold'em
match between two bots.  Each bot runs as a **separate OS subprocess**,
communicating with the engine over a local TCP socket using a compact
text protocol.  The engine is the single source of truth for game state;
both bots maintain independent local reconstructions of the game tree via
``pkbot/runner.py`` and ``pkbot/states.py``.

High-Level Match Flow:
    1. ``PokerMatch.run()`` spawns two ``BotProcess`` subprocesses and
       establishes socket connections.
    2. For each of the 1 000 rounds, ``PokerMatch.play_hand()`` drives a
       single hand to completion.
    3. Within each hand, the ``GameState`` tree is traversed: the engine
       queries the active bot via ``BotProcess.query()``, validates the
       returned action, applies it to the local ``GameState``, and broadcasts
       the action to both bots.
    4. The auction (post-flop sealed bid) is resolved in ``apply_action()``
       for ``ActionBid``: the higher bidder pays the lower bid and receives a
       randomly chosen opponent hole card.
    5. At showdown, ``eval7.evaluate()`` ranks both hands and the engine
       computes the chip transfer.
    6. Per-round logs are broadcast to both bots (``'D<delta>'``) and written
       to the ``GAME_LOG_FOLDER`` directory.

Time Enforcement:
    Each bot has a shared ``GAME_CLOCK`` budget (30 s) across all 1 000 rounds.
    Response time for each ``BotProcess.query()`` call is deducted from the
    bot's ``time_bank``.  A bot that exhausts its time bank is forced to
    ``ActionCheck`` / ``ActionFold`` for the remainder of the match.

Format Utilities:
    ``CCARDS``, ``PCARDS``, ``PVALUE``, ``STATUS`` — lambda helpers for
    compact human-readable log formatting.

Usage::

    python engine.py [--small_log]
"""
'''
1.0.0 IIT-POKERBOTS GAME ENGINE
DO NOT REMOVE, RENAME, OR EDIT THIS FILE
'''
from collections import namedtuple
import eval7
import argparse
import json
import os
from queue import Queue
import subprocess
import socket
import sys
from threading import Thread
import time
from datetime import datetime
import traceback
import random

sys.path.append(os.getcwd())

from config import *

PLAYER_LOG_SIZE_LIMIT = 524288

GAME_CLOCK = 30.0
BUILD_TIMEOUT = 10.0
CONNECT_TIMEOUT = 10.0

NUM_ROUNDS = 1000
STARTING_STACK = 5000
BIG_BLIND = 20
SMALL_BLIND = 10

# Format Utils ---------------------------------------------------------------------------------------
CCARDS = lambda cards: ','.join(map(str, cards))
PCARDS = lambda cards: '[{}]'.format(' '.join(map(str, cards)))
PVALUE = lambda name, value: ', {} ({})'.format(name, value)
STATUS = lambda players: ''.join([PVALUE(p.name, p.bankroll) for p in players])
STREET_LABELS = ['Flop', 'Turn', 'River']

# Actions --------------------------------------------------------------------------------------------
ActionFold = namedtuple('ActionFold', [])
ActionCall = namedtuple('ActionCall', [])
ActionCheck = namedtuple('ActionCheck', [])
ActionRaise = namedtuple('ActionRaise', ['amount'])
ActionBid = namedtuple('ActionBid', ['amount'])

DECODE_ACTION = {
    'F': ActionFold,
    'C': ActionCall,
    'K': ActionCheck,
    'R': ActionRaise,
    'A': ActionBid,
}

# States ---------------------------------------------------------------------------------------------
HandResult = namedtuple('HandResult', ['payoffs', 'bids', 'parent_state'])

class GameState(
            namedtuple(
                '_GameState',
                ['dealer', 'street', 'auction', 'bids', 'wagers', 'chips', 'hands', 'opp_hands', 'deck', 'parent_state']
            )
    ):
    """Represents the complete state of the table at a single decision point.

    ``GameState`` is an immutable namedtuple subclass.  Every action applied
    via ``apply_action()`` returns a **new** ``GameState`` (or ``HandResult``),
    keeping the full game tree intact for logging and backtracking.

    Attributes:
        dealer (int): Action counter; ``dealer % 2`` is the index of the active
            player (0 = SB, 1 = BB).
        street (int): Current street as an integer (0=pre-flop, 3=flop,
            4=turn, 5=river).  During the auction phase ``auction=True``
            while street remains 3.
        auction (bool): ``True`` during the sealed-bid auction phase.
        bids (list[int | None, int | None]): Mutable bid tracking.  ``None``
            until a player submits their bid via ``ActionBid``.
        wagers (list[int, int]): Per-street chip commitments for each player.
            Resets to ``[0, 0]`` at the start of every new street.
        chips (list[int, int]): Remaining chip stacks.  Decrements as wagers
            are placed and auction bids are paid.
        hands (list[list[Card], list[Card]]): Each player's private hole cards.
        opp_hands (list[list[Card], list[Card]]): Cards revealed via the
            auction.  Populated after the auction resolves.
        deck (eval7.Deck): Card deck used to deal community cards via
            ``deck.peek(n)``.
        parent_state (GameState | None): Previous node in the game tree;
            ``None`` for the root.
    """

    def calculate_result(self):
        """Determine the winner via hand evaluation and compute chip payoffs.

        Uses ``eval7.evaluate()`` to score each player's best 5-card hand
        from their two hole cards and the five community cards (``deck.peek(5)``).
        In the case of a tie, the pot is split evenly (odd chip goes to neither
        player — truncating integer division).

        Returns:
            HandResult: Terminal node with ``payoffs[i]`` being the chip delta
                for player ``i``, and ``bids`` carrying the final auction bids.
        """
        score0 = eval7.evaluate(self.deck.peek(5) + self.hands[0])
        score1 = eval7.evaluate(self.deck.peek(5) + self.hands[1])
        if score0 > score1:
            delta = STARTING_STACK - self.chips[1]
        elif score0 < score1:
            delta = self.chips[0] - STARTING_STACK
        else:  # equal split the pot
            delta = (self.chips[0] - self.chips[1]) // 2
        return HandResult([delta, -delta], self.auction, self)

    def get_valid_actions(self):
        """Return the set of action classes available to the currently active player.

        During the **auction phase** (``self.auction == True``), only
        ``{ActionBid}`` is legal for both players.

        During a **betting round**:
            * If ``cost_to_call == 0``: ``{ActionCheck}`` (and ``{ActionRaise}``
              if neither player is all-in).
            * If ``cost_to_call > 0``: ``{ActionFold, ActionCall}`` (and
              ``{ActionRaise}`` if re-raising is possible).

        Returns:
            set[type]: A set of action *classes* legal for the active player.
        """
        if self.auction:
            return {ActionBid}

        active_idx = self.dealer % 2
        cost_to_call = self.wagers[1-active_idx] - self.wagers[active_idx]
        
        if cost_to_call == 0:
            # Check or Raise allowed, unless all-in
            cannot_bet = (self.chips[0] == 0 or self.chips[1] == 0)
            return {ActionCheck} if cannot_bet else {ActionCheck, ActionRaise}
        
        # Must Call or Fold (or Raise if possible)
        cannot_raise = (cost_to_call == self.chips[active_idx] or self.chips[1-active_idx] == 0)
        return {ActionFold, ActionCall} if cannot_raise else {ActionFold, ActionCall, ActionRaise}

    def get_raise_limits(self):
        """Compute the minimum and maximum legal raise amounts for the active player.

        The minimum raise is the larger of the previous bet and ``BIG_BLIND``,
        added on top of the cost to call, and capped at the active player's
        remaining chips.  The maximum raise is capped at the effective stack
        (i.e., the maximum the opponent can call).

        Returns:
            tuple[int, int]: ``(min_total_wager, max_total_wager)`` — the total
                wager amount (inclusive of existing commitment) that constitutes
                a legal raise.  Use directly as ``ActionRaise.amount``.
        """
        active_idx = self.dealer % 2
        cost = self.wagers[1-active_idx] - self.wagers[active_idx]
        max_bet = min(self.chips[active_idx], self.chips[1-active_idx] + cost)
        min_bet = min(max_bet, cost + max(cost, BIG_BLIND))
        return (self.wagers[active_idx] + min_bet, self.wagers[active_idx] + max_bet)

    def get_bid_limits(self):
        """Return the legal bid range for the active player during the auction phase.

        Bids must be non-negative integers no greater than the active player's
        remaining chip count.  A bid of ``0`` is always legal and concedes the
        auction to the opponent (assuming a non-zero opponent bid).

        Returns:
            tuple[int, int]: ``(min_bid, max_bid)`` where ``min_bid = 0`` and
                ``max_bid = chips[active]``.
        """
        active_idx = self.dealer % 2
        max_bid = self.chips[active_idx]
        min_bid = 0
        return (min_bid, max_bid)
    
    def next_street(self):
        """Advance to the next betting round or trigger a showdown.

        Called automatically when both players' actions have concluded on a
        street (two checks, or a call/check sequence).

        Transitions:
            * ``street == 5`` (river) → ``calculate_result()``.
            * ``street == 0`` (pre-flop) → auction phase
              (``auction=True``, wagers reset to ``[0, 0]``).
            * Otherwise → ``street + 1``, wagers reset, ``auction=False``.

        Returns:
            GameState | HandResult: Next game-tree node.
        """
        if self.street == 5:
            return self.calculate_result()
        if self.street == 0:
            return GameState(1, 3, True, self.bids, [0, 0], self.chips, self.hands, self.opp_hands, self.deck, self)
        # new_street = 3 if self.street == 0 else self.street + 1
        return GameState(1, self.street+1, False, self.bids, [0, 0], self.chips, self.hands, self.opp_hands, self.deck, self)
    
    def apply_action(self, action):
        """Transition the game tree by applying one player action.

        All five action types are handled.  The method is functionally pure;
        it always returns a **new** ``GameState`` or ``HandResult`` without
        mutating ``self`` (with the exception of ``self.bids`` which is a
        mutable list shared for auction tracking).

        Action semantics:
            * **``ActionFold``**: Creates a ``HandResult`` using the folding
              player's chip deficit relative to ``STARTING_STACK``.
            * **``ActionCall``**: Special case for SB pre-flop limp (``dealer==0``).
              Otherwise matches the bet and advances via ``next_street()``.
            * **``ActionCheck``**: Advances when both players have acted.
            * **``ActionBid``**: Records the bid.  When both bids are set,
              resolves the second-price auction: the winner pays the *lower*
              bid, receives a randomly chosen opponent hole card, and play
              continues to the turn.  On a tie, both players pay and each
              sees one of the opponent's cards.
            * **``ActionRaise``**: Updates wager and deducts chips for the
              active player.

        Args:
            action: One of the five action namedtuples.

        Returns:
            GameState | HandResult: The resulting game-tree node.
        """
        active = self.dealer % 2
        
        if isinstance(action, ActionFold):
            delta = self.chips[0] - STARTING_STACK if active == 0 else STARTING_STACK - self.chips[1]
            return HandResult([delta, -delta], self.bids, self)
            
        if isinstance(action, ActionCall):
            if self.dealer == 0:  # SB calls BB
                return GameState(1, 0, self.auction, self.bids, [BIG_BLIND] * 2, [STARTING_STACK - BIG_BLIND] * 2, self.hands, self.opp_hands, self.deck, self)
            
            # Match the bet
            next_wagers = list(self.wagers)
            next_chips = list(self.chips)
            amt_to_call = next_wagers[1-active] - next_wagers[active]
            next_chips[active] -= amt_to_call
            next_wagers[active] += amt_to_call
            
            state = GameState(self.dealer + 1, self.street, self.auction, self.bids, next_wagers, next_chips, self.hands, self.opp_hands, self.deck, self)
            return state.next_street()
            
        if isinstance(action, ActionCheck):
            if (self.street == 0 and self.dealer > 0) or self.dealer > 1:
                return self.next_street()
            return GameState(self.dealer + 1, self.street, self.auction, self.bids, self.wagers, self.chips, self.hands, self.opp_hands, self.deck, self)
        
        if isinstance(action, ActionBid):
            self.bids[active] = action.amount

            if None not in self.bids: 
                if self.bids[0] == self.bids[1]:
                    rv_card_0 = random.choice(self.hands[0])
                    rv_card_1 = random.choice(self.hands[1])
                    self.opp_hands[0].append(rv_card_1)
                    self.opp_hands[1].append(rv_card_0)

                    new_chips = list(self.chips)
                    new_chips[0] -= self.bids[0]
                    new_chips[1] -= self.bids[1]
                    state = GameState(1, self.street, False, self.bids, self.wagers, new_chips, self.hands, self.opp_hands, self.deck, self)

                else:
                    winner = self.bids.index(max(self.bids))
                    revealed_card = random.choice(self.hands[1 - winner])
                    self.opp_hands[winner].append(revealed_card)

                    new_chips = list(self.chips)
                    new_chips[winner] -= self.bids[1 - winner]
                    state = GameState(1, self.street, False, self.bids, self.wagers, new_chips, self.hands, self.opp_hands, self.deck, self)
                return state
            
            else:
                return GameState(self.dealer + 1, self.street, True, self.bids, self.wagers, self.chips, self.hands, self.opp_hands, self.deck, self)

        # ActionRaise
        next_wagers = list(self.wagers)
        next_chips = list(self.chips)
        added = action.amount - next_wagers[active]
        next_chips[active] -= added
        next_wagers[active] += added
        return GameState(self.dealer + 1, self.street, self.auction, self.bids, next_wagers, next_chips, self.hands, self.opp_hands, self.deck, self)


# BotWrapper --------------------------------------------------------------------------------------
class BotProcess:
    """Manages the subprocess and socket connection for a single competing bot.

    ``BotProcess`` launches the bot's Python script as a child process, waits
    for it to connect on a dynamically allocated port, and subsequently handles
    all communication over a read/write socket file.  It also collects
    per-bot statistics (win rate, auction metrics, response times) and writes
    bot stdout to a per-player log file at the end of the match.

    Attributes:
        name (str): Display name of the bot (from ``config.py``).
        file_path (str): Absolute path to the bot entry-point script.
        time_bank (float): Remaining seconds in the bot's global time budget.
            Starts at ``GAME_CLOCK`` (30 s) and decrements with each query.
        bankroll (int): Cumulative chip gain/loss across all completed hands.
        proc (subprocess.Popen | None): Handle to the bot subprocess.
        socketfile: Read/write text file wrapping the TCP socket; ``None``
            until ``run()`` connects successfully.
        bytes_queue (Queue): Thread-safe queue collecting the bot's stdout
            bytes for log writing.
        query_times (list[float]): Per-query wall-clock response times in
            seconds, used for post-match statistics.
        hand_response_times (dict[int, float]): Total response time per round
            (key = round number).
        wins (int): Count of hands won (positive payoff).
        auction_wins (int): Count of auctions won (higher bid submitted).
        auction_total (int): Total number of auctions contested.
        bids (list[int]): History of all bids submitted by this bot.
    """

    def __init__(self, name, file_path):
        self.name = name
        self.file_path = file_path
        self.time_bank = GAME_CLOCK
        self.bankroll = 0
        self.proc = None
        self.socketfile = None
        self.bytes_queue = Queue()
        self.query_times = []
        self.hand_response_times = {}
        self.wins = 0
        self.auction_wins = 0
        self.auction_total = 0
        self.bids = []

    def run(self):
        """Spawn the bot subprocess and establish the TCP socket connection.

        Binds a server socket on an ephemeral port, launches the bot script
        as a subprocess passing the port as a CLI argument, and waits for the
        bot to connect within ``CONNECT_TIMEOUT`` seconds.  A daemon thread
        continuously drains the bot's stdout into ``bytes_queue`` so the pipe
        buffer never fills.

        After this method returns (successfully or not), ``self.socketfile``
        is either a live read/write handle or ``None`` (on failure).
        Subsequent calls to ``query()`` guard against a ``None`` socketfile.

        Raises:
            Prints error messages on ``TypeError``, ``ValueError``,
            ``OSError``, and ``socket.timeout``; does not re-raise.
        """
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with server_socket:
                server_socket.bind(('', 0))
                server_socket.settimeout(CONNECT_TIMEOUT)
                server_socket.listen()
                port = server_socket.getsockname()[1]

                proc = subprocess.Popen(
                    [PYTHON_CMD, self.file_path, str(port)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(self.file_path))
                self.proc = proc
                # function for bot listening
                def enqueue_output(out, queue):
                    try:
                        for line in out:
                            queue.put(line)
                    except ValueError:
                        pass
                # start a separate bot listening thread which dies with the program
                Thread(target=enqueue_output, args=(proc.stdout, self.bytes_queue), daemon=True).start()
                # block until we timeout or the player connects
                client_socket, _ = server_socket.accept()
                with client_socket:
                    client_socket.settimeout(CONNECT_TIMEOUT)
                    sock = client_socket.makefile('rw')
                    self.socketfile = sock
                    print(self.name, 'connected successfully')
        except (TypeError, ValueError):
            print(self.name, 'run command misformatted')
        except OSError as e:
            print(self.name, ' timed out or failed to connect.')
            self.bytes_queue.put(traceback.format_exc().encode())
        except socket.timeout:
            print('Timed out waiting for', self.name, 'to connect')

    def stop(self):
        """Gracefully shut down the bot subprocess and flush its log file.

        Sends the ``'Q'`` quit signal over the socket, closes the connection,
        and waits for the child process to terminate (with a timeout).  All
        buffered stdout bytes from ``bytes_queue`` are written to a
        ``<name>.plog`` file in ``GAME_LOG_FOLDER``, capped at
        ``PLAYER_LOG_SIZE_LIMIT`` bytes.

        Returns:
            None
        """
        if self.socketfile is not None:
            try:
                self.socketfile.write('Q\n')
                self.socketfile.close()
            except socket.timeout:
                print('Timed out waiting for', self.name, 'to disconnect')
            except OSError:
                print('Could not close socket connection with', self.name)
        if self.proc is not None:
            try:
                outs, _ = self.proc.communicate(timeout=CONNECT_TIMEOUT)
                self.bytes_queue.put(outs)
            except subprocess.TimeoutExpired:
                print('Timed out waiting for', self.name, 'to quit')
                self.proc.kill()
                outs, _ = self.proc.communicate()
                self.bytes_queue.put(outs)
        os.makedirs(GAME_LOG_FOLDER, exist_ok=True)
        with open(os.path.join(GAME_LOG_FOLDER, self.name + '.plog'), 'wb') as log_file:
            bytes_written = 0
            for output in self.bytes_queue.queue:
                try:
                    bytes_written += log_file.write(output)
                    if bytes_written >= PLAYER_LOG_SIZE_LIMIT:
                        break
                except TypeError:
                    pass

    def query(self, state, player_message, game_log, round_num):
        """Request one action from the bot and validate the response.

        Sends the accumulated ``player_message`` (including the current time
        budget) to the bot, reads its response, deducts elapsed time from
        ``time_bank``, and validates the returned action against the legal
        action set.  Returns a safe default (``ActionCheck`` or ``ActionFold``)
        if the bot submits an illegal action, disconnects, or times out.

        Validation rules:
            * ``ActionRaise`` / ``ActionBid``: must not contain a decimal point
              (integer-only), and the amount must fall within the legal bounds
              from ``get_raise_limits()`` / ``get_bid_limits()``.
            * All other actions: must appear in ``get_valid_actions()``.

        Fallback defaults:
            * Auction phase (``ActionBid`` required): ``ActionBid(0)``.
            * Otherwise: ``ActionCheck`` if legal, else ``ActionFold()``.

        Args:
            state (GameState | HandResult): Current game-tree node used to
                determine valid actions and raise/bid limits.
            player_message (list[str]): Mutable list of message clauses to
                send.  The time-budget clause is prepended as ``player_message[0]``
                and redundant history clauses beyond index 1 are discarded.
            game_log (list[str]): Mutable game log list; illegal-action messages
                are appended here.
            round_num (int): Current round number for per-hand log attribution.

        Returns:
            ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
                A validated action ready to apply to the game tree.
        """
        valid_actions = state.get_valid_actions() if isinstance(state, GameState) else {ActionCheck}
        if self.socketfile is not None and self.time_bank > 0.:
            clause = ''
            try:
                player_message[0] = 'T{:.3f}'.format(self.time_bank)
                message = ' '.join(player_message) + '\n'
                del player_message[1:]  # do not send redundant action history
                start_time = time.perf_counter()
                self.socketfile.write(message)
                self.socketfile.flush()
                clause = self.socketfile.readline().strip()
                end_time = time.perf_counter()
                response_time = end_time - start_time
                self.time_bank -= response_time
                self.query_times.append(response_time)
                self.hand_response_times[round_num] = self.hand_response_times.get(round_num, 0) + response_time
                if self.time_bank <= 0.:
                    raise socket.timeout
                action = DECODE_ACTION[clause[0]]
                if action in valid_actions:
                    if clause[0] == 'R':
                        if '.' in clause[1:]:
                            game_log.append(self.name + ' attempted illegal ActionRaise({}) with decimal'.format(clause[1:]))
                            self.bytes_queue.put(f"[Round#{round_num}] Tried to raise with decimal amount: {clause[1:]}\n".encode())
                            return ActionCheck() if ActionCheck in valid_actions else ActionFold()
                        amount = int(clause[1:])
                        min_raise, max_raise = state.get_raise_limits()
                        if min_raise <= amount <= max_raise:
                            return action(amount)
                    elif clause[0] == 'A':
                        if '.' in clause[1:]:
                            game_log.append(self.name + ' attempted illegal bid with decimal')
                            self.bytes_queue.put(f"[Round#{round_num}] Tried to bid with decimal amount: {clause[1:]}\n".encode())
                            return ActionCheck() if ActionCheck in valid_actions else ActionFold()
                        amount = int(clause[1:])
                        min_bid, max_bid = state.get_bid_limits()
                        if min_bid <= amount <= max_bid:
                            return action(amount)
                    else:
                        return action()
                
                if clause[0] in ('R', 'A'):
                    game_log.append(self.name + ' attempted illegal ' + action.__name__ + ' with amount ' + str(int(clause[1:])))
                else:
                    game_log.append(self.name + ' attempted illegal ' + action.__name__)

            except socket.timeout:
                error_message = self.name + ' ran out of time'
                game_log.append(error_message)
                print(error_message)
                self.time_bank = 0.
            except OSError:
                error_message = self.name + ' disconnected'
                game_log.append(error_message)
                print(error_message)
                self.time_bank = 0.
            except (IndexError, KeyError, ValueError) as e:
                game_log.append(self.name + ' response misformatted: ' + str(clause))
        # set a base bid action of 0 if pokerbot fails to submit legal bid action
        if ActionBid in valid_actions: 
            return ActionBid(0)
        
        return ActionCheck() if ActionCheck in valid_actions else ActionFold()

# PokerMatch -------------------------------------------------------------------------------------------------
class PokerMatch():
    """Orchestrates a full 1 000-round heads-up Sneak Peek Hold'em match.

    ``PokerMatch`` owns the game log, coordinates player messages, and runs
    the complete match via ``run()``.  It alternates blinds each round
    (SB/BB positions swap after every hand) and accumulates statistics printed
    to stdout on completion.

    Attributes:
        small_log (bool): When ``True``, log entries use a compressed format
            (card shorthand and ``+/-`` payoffs) to reduce file size.  When
            ``False`` (default), full human-readable phrasing is used.
        timestamp (datetime): Match start time, used for log file naming.
        log (list[str]): Ordered list of log entry strings.  Written to a
            ``.glog`` file in ``GAME_LOG_FOLDER`` at the end of the match.
        player_messages (list[list[str], list[str]]): Per-player accumulation
            of pending message clauses.  Sent to each bot via
            ``BotProcess.query()`` and partially cleared after each query.
    """

    def __init__(self, small_log=False):
        """Initialise the match log and player message buffers.

        Args:
            small_log (bool): If ``True``, use compressed log format
                (``name: [Ah Kd]`` instead of ``name received [Ah Kd]``,
                and ``+100`` instead of ``name awarded 100``).  Defaults
                to ``False``.
        """
        self.small_log = small_log
        self.timestamp = datetime.now()
        self.log = [self.timestamp.strftime('%Y-%m-%d %H:%M:%S ') + BOT_1_NAME + ' vs ' + BOT_2_NAME]
        self.player_messages = [[], []]

    def log_state(self, players, state: GameState):
        """Update the game log and player message queues from a ``GameState``.

        Called at each decision point before querying the active player.
        Handles three types of state transitions:

        1. **Post-auction** (``street == 3, auction == False, dealer == 1``):
           Logs who won the auction and what card was revealed.  Sends the
           ``'N<chips>_<bids>_<revealed_card>'`` clause to both bots.
        2. **Start of pre-flop** (``street == 0, dealer == 0``):
           Logs blind postings and hole card deals.  Initialises
           ``player_messages`` with the hole card ``'H<cards>'`` clause.
        3. **Start of a post-flop street** (``street > 0, dealer == 1``):
           Logs the new board cards.  Appends the ``'B<board>'`` clause.

        Args:
            players (list[BotProcess, BotProcess]): Active player list
                (index 0 = current SB).
            state (GameState): Game-tree node to log.
        """
        if state.street == 3 and state.auction is False and state.dealer == 1:
            for i in range(2):
                if len(state.opp_hands[i]) == 1:
                    self.log.append('{} won the auction and was revealed {}'.format(players[i].name, PCARDS(state.opp_hands[i])))
            
            self.player_messages[0].append('P0')
            self.player_messages[0].append('N' + ','.join([str(x) for x in state.chips]) + '_' + ','.join([str(x) for x in state.bids]) + '_' + CCARDS(state.opp_hands[0]))
            self.player_messages[1].append('P1')
            self.player_messages[1].append('N' + ','.join([str(x) for x in state.chips]) + '_' + ','.join([str(x) for x in state.bids]) + '_' + CCARDS(state.opp_hands[1]))

    
        if state.street == 0 and state.dealer == 0:
            if not self.small_log:
                self.log.append('{} posts blind: {}'.format(players[0].name, SMALL_BLIND))
                self.log.append('{} posts blind: {}'.format(players[1].name, BIG_BLIND))
                self.log.append('{} received {}'.format(players[0].name, PCARDS(state.hands[0])))
                self.log.append('{} received {}'.format(players[1].name, PCARDS(state.hands[1])))
            else:
                self.log.append('{}: {}'.format(players[0].name, PCARDS(state.hands[0])))
                self.log.append('{}: {}'.format(players[1].name, PCARDS(state.hands[1])))
            self.player_messages[0] = ['T0.', 'P0', 'H' + CCARDS(state.hands[0])]
            self.player_messages[1] = ['T0.', 'P1', 'H' + CCARDS(state.hands[1])]
        elif state.street > 0 and state.dealer == 1:
            board = state.deck.peek(state.street)
            self.log.append(STREET_LABELS[state.street - 3] + ' ' + PCARDS(board) +
                            PVALUE(players[0].name, STARTING_STACK-state.chips[0]) +
                            PVALUE(players[1].name, STARTING_STACK-state.chips[1]))
            compressed_board = 'B' + CCARDS(board)
            self.player_messages[0].append(compressed_board)
            self.player_messages[1].append(compressed_board)

    def log_action(self, name, action, bet_override):
        """Append an action event to the game log and broadcast it to both bots.

        Translates an action namedtuple into a human-readable phrase (for the
        ``.glog`` file) and its wire-format code (for bot messages).  The
        ``bet_override`` flag distinguishes a *bet* (first bet on a street,
        wagers would be ``[0, 0]``) from a *raise* in the log phrasing.

        Args:
            name (str): Display name of the player who took the action.
            action: One of the five action namedtuples.
            bet_override (bool): ``True`` when the action is the first bet
                on a street (logs ``'bets X'`` instead of ``'raises to X'``).
        """
        if isinstance(action, ActionFold):
            phrasing = ' folds'
            code = 'F'
        elif isinstance(action, ActionCall):
            phrasing = ' calls'
            code = 'C'
        elif isinstance(action, ActionCheck):
            phrasing = ' checks'
            code = 'K'
        elif isinstance(action, ActionBid):
            phrasing = ' bids ' + str(action.amount)
            code = 'A' + str(action.amount)
        else:  # isinstance(action, ActionRaise)
            phrasing = (' bets ' if bet_override else ' raises to ') + str(action.amount)
            code = 'R' + str(action.amount)
        if self.small_log:
            self.log.append(name + ' ' + code)
        else:
            self.log.append(name + phrasing)
        self.player_messages[0].append(code)
        self.player_messages[1].append(code)

    def log_result(self, players, result):
        """Append the hand result to the game log and send payoff messages to bots.

        If the hand went to showdown (both players' wagers were equal at the
        river), logs each player's shown hand and appends the ``'O<cards>'``
        reveal clause to the opponent's message queue.  Always logs the chip
        payoffs and appends ``'D<delta>'`` to each player's message queue.

        Args:
            players (list[BotProcess, BotProcess]): Active player list.
            result (HandResult): The terminal game-tree node with payoff info.
        """
        prev = result.parent_state
        if prev.wagers[0] == prev.wagers[1]:
            self.log.append('{} shows {}'.format(players[0].name, PCARDS(prev.hands[0])))
            self.log.append('{} shows {}'.format(players[1].name, PCARDS(prev.hands[1])))
            self.player_messages[0].append('O' + CCARDS(prev.hands[1]))
            self.player_messages[1].append('O' + CCARDS(prev.hands[0]))
        if self.small_log:
            self.log.append('{}: {:+d}'.format(players[0].name, result.payoffs[0]))
            self.log.append('{}: {:+d}'.format(players[1].name, result.payoffs[1]))
        else:
            self.log.append('{} awarded {}'.format(players[0].name, result.payoffs[0]))
            self.log.append('{} awarded {}'.format(players[1].name, result.payoffs[1]))
        self.player_messages[0].append('D' + str(result.payoffs[0]))
        self.player_messages[1].append('D' + str(result.payoffs[1]))

    def play_hand(self, players, round_num):
        """Run a single complete hand (round) of Sneak Peek Hold'em.

        Deals two hole cards to each player, initialises the ``GameState`` at
        the root, and drives the game tree forward by alternately querying the
        active player and applying their action until a ``HandResult`` is
        reached.  After each ``ActionBid`` resolution, per-bot auction
        statistics are updated.

        Args:
            players (list[BotProcess, BotProcess]): Current [SB, BB] player list.
                The list is reversed between rounds to rotate blind positions.
            round_num (int): 1-indexed current round number for log attribution.
        """
        deck = eval7.Deck()
        deck.shuffle()
        hands = [deck.deal(2), deck.deal(2)]
        wagers = [SMALL_BLIND, BIG_BLIND]
        chips = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        state = GameState(0, 0, False, [None, None], wagers, chips, hands, [[], []], deck, None)
        
        while not isinstance(state, HandResult):
            self.log_state(players, state)
            active = state.dealer % 2
            player = players[active]
            action = player.query(state, self.player_messages[active], self.log, round_num)
            bet_override = (state.wagers == [0, 0])
            self.log_action(player.name, action, bet_override)
            previous_auction = state.auction
            state = state.apply_action(action)
            if previous_auction and not isinstance(state, HandResult) and not state.auction:
                players[0].auction_total += 1
                players[1].auction_total += 1
                players[0].bids.append(state.bids[0])
                players[1].bids.append(state.bids[1])
                if state.bids[0] > state.bids[1]:
                    players[0].auction_wins += 1
                elif state.bids[1] > state.bids[0]:
                    players[1].auction_wins += 1
            
        self.log_result(players, state)
        for player, player_message, delta in zip(players, self.player_messages, state.payoffs):
            player.query(state, player_message, self.log, round_num)
            player.bankroll += delta
            if delta > 0:
                player.wins += 1

    def run(self):
        """Execute the complete 1 000-round match and write the match log.

        Procedure:
            1. Print the IIT Pokerbots ASCII banner (unless ``small_log=True``).
            2. Instantiate and connect both ``BotProcess`` objects.
            3. Loop for ``NUM_ROUNDS`` rounds, calling ``play_hand()`` each
               time and swapping the blind positions (player list reversal).
            4. Print game-end statistics to stdout for each bot (win rate,
               payoff, auction metrics, response times).
            5. Call ``stop()`` on each bot to flush logs and close sockets.
            6. Write the full ``.glog`` match log to ``GAME_LOG_FOLDER``.

        Returns:
            None
        """
        start_time = time.perf_counter()
        if not self.small_log:
            print('██ ██ ████████     ██████   ██████  ██   ██ ███████ ██████  ██████   ██████  ████████ ███████ ')
            print('██ ██    ██        ██   ██ ██    ██ ██  ██  ██      ██   ██ ██   ██ ██    ██    ██    ██      ')
            print('██ ██    ██        ██████  ██    ██ █████   █████   ██████  ██████  ██    ██    ██    ███████ ')
            print('██ ██    ██        ██      ██    ██ ██  ██  ██      ██   ██ ██   ██ ██    ██    ██         ██ ')
            print('██ ██    ██        ██       ██████  ██   ██ ███████ ██   ██ ██████   ██████     ██    ███████ ')
            print()
        print('Initializing Game Engine...')
        players = [
            BotProcess(BOT_1_NAME, BOT_1_FILE),
            BotProcess(BOT_2_NAME, BOT_2_FILE)
        ]
        all_bots = list(players)
        for player in players:
            player.run()
        for round_num in range(1, NUM_ROUNDS + 1):
            self.log.append('')
            self.log.append('Round #' + str(round_num) + STATUS(players))
            self.play_hand(players, round_num)
            players = players[::-1]
        self.log.append('')
        self.log.append('Final' + STATUS(players))

        print("\n=== Game Stats ===")
        for bot in all_bots:
            print(f"\nStats for {bot.name}:")
            total_queries = len(bot.query_times)
            avg_query = sum(bot.query_times) / total_queries if total_queries > 0 else 0.0
            max_query = max(bot.query_times) if total_queries > 0 else 0.0
            avg_hand_time = sum(bot.hand_response_times.values()) / NUM_ROUNDS
            win_rate = bot.wins / NUM_ROUNDS
            avg_payoff = bot.bankroll / NUM_ROUNDS
            auction_rate = bot.auction_wins / bot.auction_total if bot.auction_total > 0 else 0.0
            
            if bot.bids:
                avg_bid = sum(bot.bids) / len(bot.bids)
                var_bid = sum((x - avg_bid) ** 2 for x in bot.bids) / len(bot.bids)
            else:
                avg_bid = 0.0
                var_bid = 0.0
            
            print(f"  Total Bankroll: {bot.bankroll}")
            print(f"------------------------------------------------------------")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Avg Payoff/Hand: {avg_payoff:.2f}")
            print(f"------------------------------------------------------------")
            print(f"  Auction Win Rate: {auction_rate:.1%}")
            print(f"  Avg Bid Amount (Mean, Var): ({avg_bid:.2f}, {var_bid:.2f})")
            print(f"------------------------------------------------------------")
            print(f"  Avg Response Time (Query): {avg_query:.5f}s")
            print(f"  Avg Response Time (Hand): {avg_hand_time:.5f}s")
            print(f"  Max Response Time: {max_query:.5f}s")

        print(f"\nTotal Match Time: {time.perf_counter() - start_time:.3f}s")
        for player in players:
            player.stop()

        name = f"{self.timestamp.strftime('%Y%m%d-%H%M%S-%f')}.glog"
        print('Writing game log to', name)
        os.makedirs(GAME_LOG_FOLDER, exist_ok=True)
        with open(os.path.join(GAME_LOG_FOLDER, name), 'w') as log_file:
            log_file.write('\n'.join(self.log))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_log', action='store_true', help='Use compressed logging format')
    args = parser.parse_args()
    PokerMatch(small_log=args.small_log).run()