#!/usr/bin/env python3
"""
GCG parser: replays a GCG file and returns board state (CGP) at each turn.

The CGP produced here is the offline ground truth — it never goes through OCR.
"""

import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Board layout
# ---------------------------------------------------------------------------

STANDARD_PREMIUM = [
    "TW.....TW.....TW",
    ".DW...DL...DL.DW",
    "..DW...DL.DL..DW",  # only 15 cols but matches rows below
    # ... full 15x15 premium square map omitted for brevity;
    # for CGP output we only need to know which cells are occupied and
    # what letter is there.
]

COLS = "ABCDEFGHIJKLMNO"  # 15 standard columns

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    letter: str = ""   # uppercase = regular; lowercase = blank playing that letter
    is_blank: bool = False

@dataclass
class GameState:
    board: List[List[Cell]] = field(default_factory=lambda: [[Cell() for _ in range(15)] for _ in range(15)])
    racks: List[str] = field(default_factory=lambda: ["", ""])
    scores: List[int] = field(default_factory=lambda: [0, 0])
    turn: int = 0  # 0-indexed
    on_turn: int = 0  # 0 or 1, which player is on turn
    players: List[str] = field(default_factory=lambda: ["Player1", "Player2"])
    lexicon: str = ""
    game_id: str = ""

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def to_cgp(self, rack: str = None, lexicon: str = None) -> str:
        """Serialize board to CGP format."""
        rows = []
        for r in range(15):
            row = ""
            empty = 0
            for c in range(15):
                cell = self.board[r][c]
                if not cell.letter:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += cell.letter
            if empty:
                row += str(empty)
            rows.append(row)
        board_str = "/".join(rows)
        rack_str = rack if rack is not None else self.racks[self.on_turn]
        lex = lexicon or self.lexicon or "NWL23"
        s0, s1 = self.scores
        return f"{board_str} {rack_str}/ {s0} {s1} lex {lex};"

# ---------------------------------------------------------------------------
# Coordinate parsing
# ---------------------------------------------------------------------------

def parse_coord(coord: str) -> Tuple[int, int, bool]:
    """
    Parse a move coordinate like '8H' (row 8, col H, horizontal) or
    'H8' (col H, row 8, vertical).
    Returns (row, col, horizontal) — both 0-indexed.
    """
    coord = coord.upper()
    # Horizontal: digit(s) + letter(s), e.g. "8H", "12A", "8HI"
    m = re.match(r'^(\d+)([A-O]+)$', coord)
    if m:
        row = int(m.group(1)) - 1
        col = COLS.index(m.group(2)[0])
        return row, col, True
    # Vertical: letter(s) + digit(s), e.g. "H8", "A12"
    m = re.match(r'^([A-O]+)(\d+)$', coord)
    if m:
        col = COLS.index(m.group(1)[0])
        row = int(m.group(2)) - 1
        return row, col, False
    raise ValueError(f"Cannot parse coordinate: {coord}")

# ---------------------------------------------------------------------------
# Place a word on the board
# ---------------------------------------------------------------------------

def place_word(board: List[List[Cell]], word: str, row: int, col: int, horizontal: bool):
    """
    Place a word on the board. '.' means the tile is already there (through-play).
    Lowercase = blank playing that letter.
    """
    r, c = row, col
    for ch in word:
        if ch != '.':
            is_blank = ch.islower()
            board[r][c] = Cell(letter=ch, is_blank=is_blank)
        if horizontal:
            c += 1
        else:
            r += 1

def remove_word(board: List[List[Cell]], word: str, row: int, col: int, horizontal: bool):
    """Remove tiles placed by a word (for phony withdrawal). '.' = pre-existing, skip."""
    r, c = row, col
    for ch in word:
        if ch != '.':
            board[r][c] = Cell()
        if horizontal:
            c += 1
        else:
            r += 1

# ---------------------------------------------------------------------------
# GCG replay
# ---------------------------------------------------------------------------

def parse_gcg(gcg_text: str) -> List[GameState]:
    """
    Replay a GCG and return a list of GameState at each turn (after each move).
    Index 0 = state after move 1, etc.
    The list starts with the initial empty board (index -1 if you want that).
    """
    lines = gcg_text.strip().split("\n")

    state = GameState()
    # Track which player maps to player index
    # GCG #player lines: #player1 nickname full_name
    # Move lines use nicknames: >nickname:
    nicknames = {}  # nickname -> player_index

    states = []
    # Initial state (empty board)
    states.append(state.copy())

    pending_phony = None  # (row, col, horizontal, word) — waiting to see if challenged

    for line in lines:
        line = line.strip()

        # Header lines
        if line.startswith("#lexicon"):
            state.lexicon = line.split(None, 1)[1].strip()
        elif line.startswith("#id"):
            parts = line.split()
            if len(parts) >= 3:
                state.game_id = parts[2]
        elif line.startswith("#player1"):
            parts = line.split(None, 2)
            nickname = parts[1] if len(parts) >= 2 else "Player1"
            fullname = parts[2].strip() if len(parts) >= 3 else nickname
            state.players[0] = fullname
            nicknames[nickname] = 0     # move lines use nickname
            nicknames[fullname] = 0     # some GCGs use full name in moves
        elif line.startswith("#player2"):
            parts = line.split(None, 2)
            nickname = parts[1] if len(parts) >= 2 else "Player2"
            fullname = parts[2].strip() if len(parts) >= 3 else nickname
            state.players[1] = fullname
            nicknames[nickname] = 1
            nicknames[fullname] = 1
        elif not line.startswith(">"):
            continue

        if not line.startswith(">"):
            continue

        # Move line: >PlayerName: RACK COORD WORD +score total
        # Strip leading ">"
        line = line[1:]
        colon = line.index(":")
        player_name = line[:colon].strip()
        rest = line[colon+1:].strip().split()

        if len(rest) < 4:
            continue

        rack = rest[0]
        coord = rest[1]
        word = rest[2]
        score_str = rest[3]  # e.g. "+48" or "-29"
        total_str = rest[4] if len(rest) > 4 else "0"

        # Determine player index by nickname (most reliable)
        pidx = nicknames.get(player_name)
        if pidx is None:
            # Fallback: use on_turn (alternating moves)
            pidx = state.on_turn

        # Update rack
        state.racks[pidx] = rack

        try:
            score_delta = int(score_str.lstrip("+"))
            total = int(total_str)
        except ValueError:
            score_delta = 0
            total = state.scores[pidx]

        # Handle different move types
        if coord == "--":
            # Phony withdrawal format: >player: RACK -- DELTA TOTAL
            # rest[2]=delta (negative), rest[3]=reverted total — no "word" field
            try:
                total = int(rest[3]) if len(rest) > 3 else state.scores[pidx]
            except (ValueError, IndexError):
                total = state.scores[pidx]
            state.scores[pidx] = total  # score reverts
            if pending_phony:
                pr, pc, ph, pw = pending_phony
                remove_word(state.board, pw, pr, pc, ph)
                pending_phony = None
            # Player stays on turn (same player replays)
            states.append(state.copy())
            continue

        # Clear pending phony (previous move wasn't challenged)
        pending_phony = None

        if word.startswith("-") and word[1:]:
            # Exchange: -TILES (tiles exchanged from rack)
            state.scores[pidx] = total
            state.on_turn = 1 - pidx
            states.append(state.copy())
            continue

        if coord in ("(challenge)", "(time)") or not coord[0].isalnum():
            # Special event: challenge bonus, time penalty, etc.
            state.scores[pidx] = total
            states.append(state.copy())
            continue

        if word.startswith("(") and word.endswith(")"):
            # End-of-game rack subtraction: opponent's remaining tiles
            state.scores[pidx] = total
            states.append(state.copy())
            continue

        # Regular play or pass
        try:
            row, col, horizontal = parse_coord(coord)
        except (ValueError, IndexError):
            # Unknown coord format (pass, etc.)
            state.scores[pidx] = total
            state.on_turn = 1 - pidx
            states.append(state.copy())
            continue

        # Place tiles
        place_word(state.board, word, row, col, horizontal)
        state.scores[pidx] = total

        # Remember this play in case it gets challenged
        pending_phony = (row, col, horizontal, word)

        state.on_turn = 1 - pidx
        state.turn += 1
        states.append(state.copy())

    return states


def states_to_test_cases(states: List[GameState], game_id: str):
    """
    Convert replay states to (name, cgp) pairs suitable for testdata/.
    Skips the initial empty board.
    """
    cases = []
    for i, state in enumerate(states[1:], 1):  # skip index 0 (empty board)
        name = f"{game_id}_t{i:03d}"
        cgp = state.to_cgp()
        cases.append((name, cgp))
    return cases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: gcg_parser.py <file.gcg> [--turn N]")
        sys.exit(1)

    gcg_file = sys.argv[1]
    target_turn = None
    if "--turn" in sys.argv:
        idx = sys.argv.index("--turn")
        target_turn = int(sys.argv[idx + 1])

    with open(gcg_file, encoding="utf-8") as f:
        gcg_text = f.read()

    states = parse_gcg(gcg_text)
    print(f"Replayed {len(states)-1} moves")

    if target_turn is not None:
        s = states[target_turn]
        print(s.to_cgp())
    else:
        for i, s in enumerate(states[1:], 1):
            print(f"Turn {i}: {s.to_cgp()[:80]}...")
