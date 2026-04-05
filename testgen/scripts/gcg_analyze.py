#!/usr/bin/env python3
"""
Analyze a GCG file and output JSON with exchange turns, blank turns,
and player names.  Used by gen_augment.ts.

Usage: gcg_analyze.py <file.gcg>
Output: JSON to stdout
"""

import json
import sys
from gcg_parser import parse_gcg, patch_forward_racks, move_index_map


def analyze(gcg_text: str) -> dict:
    states = parse_gcg(gcg_text)
    patch_forward_racks(states)
    mmap = move_index_map(states)

    players = states[0].players if states else ["Player1", "Player2"]
    lexicon = states[0].lexicon if states else ""

    # Find exchanges: states where the board didn't change but it's a real move
    exchanges = []
    for url_turn in range(1, len(mmap)):
        si = mmap[url_turn]
        if si < 1:
            continue
        prev_si = mmap[url_turn - 1] if url_turn > 0 else 0
        s = states[si]
        prev = states[prev_si]

        # Check if board is identical (exchange doesn't change board)
        board_same = True
        for r in range(15):
            for c in range(15):
                if s.board[r][c].letter != prev.board[r][c].letter:
                    board_same = False
                    break
            if not board_same:
                break

        if board_same and url_turn > 0 and si > 0:
            # The player who just moved (on_turn flipped, so previous on_turn acted)
            actor = 1 - s.on_turn
            # Find the exchanged tiles from the GCG (stored in rack diff)
            # Actually, we need to parse the original GCG lines for exchange info
            pass

    # Simpler approach: scan GCG text directly for exchange lines
    exchanges = []
    lines = gcg_text.strip().split("\n")
    nicknames = {}
    for line in lines:
        if line.startswith("#player1"):
            parts = line.split(None, 2)
            nicknames[parts[1]] = 0 if len(parts) >= 2 else None
        elif line.startswith("#player2"):
            parts = line.split(None, 2)
            nicknames[parts[1]] = 1 if len(parts) >= 2 else None

    # Find exchanges by scanning states: exchange states have is_event=True
    # (because coord starts with "-", parsed as special event) and the board
    # is unchanged from the previous state.  The banner appears at the NEXT
    # URL turn after the exchange.
    gcg_move_idx = 0
    for line in lines:
        if not line.startswith(">"):
            continue
        gcg_move_idx += 1

        rest = line[1:]
        colon = rest.index(":")
        nickname = rest[:colon].strip()
        parts = rest[colon+1:].strip().split()
        if len(parts) < 2:
            continue

        coord_or_exch = parts[1]

        if coord_or_exch.startswith("-") and len(coord_or_exch) > 1 and coord_or_exch != "--":
            tiles = coord_or_exch[1:]
            # Exchange is state[gcg_move_idx] with is_event=True.
            # Find the next URL turn (first mmap entry > gcg_move_idx).
            next_url_turn = None
            for ut in range(1, len(mmap)):
                if mmap[ut] > gcg_move_idx:
                    next_url_turn = ut
                    break
            if next_url_turn is not None:
                cgp = states[mmap[next_url_turn]].to_cgp()
                exchanges.append({
                    "nickname": nickname,
                    "tiles": tiles,
                    "urlTurn": next_url_turn,
                    "cgp": cgp,
                })

    # Find turns where blanks are on the board
    blank_turns = []
    for url_turn in range(1, len(mmap)):
        si = mmap[url_turn]
        s = states[si]
        has_blank = False
        for r in range(15):
            for c in range(15):
                if s.board[r][c].is_blank:
                    has_blank = True
                    break
            if has_blank:
                break
        if has_blank:
            cgp = s.to_cgp()
            blank_turns.append({
                "urlTurn": url_turn,
                "cgp": cgp,
            })

    return {
        "players": players,
        "nicknames": list(nicknames.keys()),
        "lexicon": lexicon,
        "exchanges": exchanges,
        "blankTurns": blank_turns,
        "totalUrlTurns": len(mmap) - 1,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: gcg_analyze.py <file.gcg>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        gcg_text = f.read()

    result = analyze(gcg_text)
    print(json.dumps(result))
