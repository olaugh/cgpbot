#!/usr/bin/env python3
"""
Match a real Woogles screenshot to its source game in the Woogles database.

Given:
  - approximate OCR output (board occupancy, scores, player names)

Returns:
  - exact game ID
  - exact turn number
  - golden CGP ground truth

Usage:
    python3 match_game.py --cgp "15/15/..." --players cesar BestBot --scores 306 367
    python3 match_game.py --image path/to/screenshot.png  # uses cgpbot server for OCR
"""

import argparse
import json
import os
import sys
import time
import requests
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fetch_games import get_recent_games, get_gcg, save_game
from gcg_parser import parse_gcg, GameState

API = "https://woogles.io/api/game_service.GameMetadataService"
GCG_DIR = os.path.join(os.path.dirname(__file__), "..", "gcg")
META_DIR = os.path.join(os.path.dirname(__file__), "..", "meta")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")

# ---------------------------------------------------------------------------
# Index-based fast lookup
# ---------------------------------------------------------------------------

_index_cache = None

def _load_index():
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    occ_path = os.path.join(INDEX_DIR, "occ.npy")
    if not os.path.exists(occ_path):
        return None
    occ     = np.load(os.path.join(INDEX_DIR, "occ.npy"),     mmap_mode="r")
    letters = np.load(os.path.join(INDEX_DIR, "letters.npy"), mmap_mode="r")
    scores  = np.load(os.path.join(INDEX_DIR, "scores.npy"),  mmap_mode="r")

    # Fast binary meta arrays (preferred) or fallback to JSONL/JSON
    game_ids_path = os.path.join(INDEX_DIR, "meta_game_ids.npy")
    turns_path = os.path.join(INDEX_DIR, "meta_turns.npy")
    n_occ_path = os.path.join(INDEX_DIR, "n_occ.npy")
    if os.path.exists(game_ids_path) and os.path.exists(turns_path):
        meta_game_ids = np.load(game_ids_path, mmap_mode="r")
        meta_turns = np.load(turns_path, mmap_mode="r")
        n_occ = np.load(n_occ_path, mmap_mode="r") if os.path.exists(n_occ_path) else None
        _index_cache = (occ, letters, scores, None, meta_game_ids, meta_turns, n_occ)
    else:
        # Fallback: parse JSONL or JSON
        jsonl_path = os.path.join(INDEX_DIR, "meta.jsonl")
        json_path = os.path.join(INDEX_DIR, "meta.json")
        if os.path.exists(jsonl_path):
            meta = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        meta.append(json.loads(line))
        elif os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
        else:
            return None
        _index_cache = (occ, letters, scores, meta, None, None, None)
    return _index_cache


def _query_to_arrays(cgp: str):
    """Convert a CGP string to (occ, letters) uint8 arrays of length 225."""
    board_str = cgp.split()[0]
    rows = board_str.split("/")
    occ = np.zeros(225, dtype=np.uint8)
    letters = np.zeros(225, dtype=np.uint16)
    for r, row in enumerate(rows[:15]):
        c = 0
        i = 0
        while i < len(row) and c < 15:
            ch = row[i]
            if ch.isdigit():
                # multi-digit run length
                n = 0
                while i < len(row) and row[i].isdigit():
                    n = n * 10 + int(row[i])
                    i += 1
                c += n
            else:
                idx = r * 15 + c
                occ[idx] = 1
                letters[idx] = ord(ch)
                c += 1
                i += 1
    return occ, letters


def match_screenshot_indexed(
    ocr_cgp: str,
    ocr_scores: tuple = None,
    min_similarity: float = 0.85,
    tolerance: int = 5,
):
    """
    Fast indexed lookup: vectorized Jaccard over all ~11M board states.
    Returns same dict as match_screenshot, or None.
    """
    t_start = time.time()
    index = _load_index()
    if index is None:
        print("  No index found — run build_index.py first")
        return None
    t_load = time.time()

    occ_idx, letters_idx, scores_idx, meta, meta_game_ids, meta_turns, precomputed_n_occ = index
    N = len(occ_idx)

    q_occ, q_letters = _query_to_arrays(ocr_cgp)
    q_n = int(q_occ.sum())
    if q_n == 0:
        return None

    # Pre-filter by tile count: for Jaccard >= 0.90, the state's tile count
    # must satisfy 0.9 * q_n <= n_state <= q_n / 0.9.  This avoids reading
    # the full 2.5GB occ array for states that can't possibly match.
    if precomputed_n_occ is not None:
        lo = int(q_n * 0.9) - 1  # small margin
        hi = int(q_n / 0.9) + 1
        count_mask = (precomputed_n_occ >= lo) & (precomputed_n_occ <= hi)
        subset_idx = np.where(count_mask)[0]
        n_subset = len(subset_idx)
        t_prefilter = time.time()

        if n_subset == 0:
            print(f"  Index: {N} states, 0 passed count pre-filter [{lo},{hi}]")
            return None

        # Vectorized Jaccard on the subset only
        occ_sub = occ_idx[subset_idx]
        intersection_sub = occ_sub.dot(q_occ.astype(np.int32)).astype(np.int32)
        n_occ_sub = precomputed_n_occ[subset_idx]
        union_sub = n_occ_sub + q_n - intersection_sub
        valid_sub = union_sub > 0
        occ_sim_sub = np.where(valid_sub, intersection_sub / union_sub.clip(1), 0.0)

        # Map back to global indices
        cand_local = np.where(occ_sim_sub >= 0.90)[0]
        candidates = subset_idx[cand_local]

        # Build full-size occ_sim array only for candidates (sparse)
        occ_sim = np.zeros(N, dtype=np.float64)
        occ_sim[candidates] = occ_sim_sub[cand_local]
    else:
        t_prefilter = t_load
        # Full scan fallback
        intersection = occ_idx.dot(q_occ.astype(np.int32)).astype(np.int32)
        n_occ = occ_idx.sum(axis=1).astype(np.int32)
        union = n_occ + q_n - intersection
        valid = union > 0
        occ_sim = np.where(valid, intersection / union.clip(1), 0.0)
        candidates = np.where(occ_sim >= 0.90)[0]

    t_jaccard = time.time()
    print(f"  Index: {N} states, {len(candidates)} candidates (occ>=0.90)"
          + (f" [{len(subset_idx)} passed count filter]" if precomputed_n_occ is not None else ""))

    if len(candidates) == 0:
        print(f"  No confident match found (best occ_sim={occ_sim.max():.3f})")
        return None

    best_sim = 0.0
    best_idx = -1
    best_occ_sim = 0.0
    # Track all scored candidates for occupancy_locked check
    scored = []  # list of (combined, occ_sim, idx)

    for ci in candidates:
        o_sim = float(occ_sim[ci])

        # Letter accuracy over common cells
        common_mask = (occ_idx[ci] == 1) & (q_occ == 1)
        n_common = int(common_mask.sum())
        if n_common == 0:
            continue
        letter_acc = float(np.sum((letters_idx[ci] == q_letters) & common_mask)) / n_common

        # Score bonus
        score_bonus = 0.0
        if ocr_scores is not None:
            t_s0, t_s1 = int(scores_idx[ci, 0]), int(scores_idx[ci, 1])
            o_s0, o_s1 = ocr_scores
            diff = min(abs(t_s0 - o_s0) + abs(t_s1 - o_s1),
                       abs(t_s0 - o_s1) + abs(t_s1 - o_s0))
            if diff <= tolerance:
                score_bonus = 0.1 * (1 - diff / (tolerance * 10))

        combined = o_sim * 0.6 + letter_acc * 0.3 + score_bonus
        scored.append((combined, o_sim, ci))
        if combined > best_sim:
            best_sim = combined
            best_idx = ci
            best_occ_sim = o_sim
            if combined >= 0.98:
                break  # confident match

    if best_idx < 0 or best_sim < min_similarity:
        print(f"  No confident match found (best combined={best_sim:.3f})")
        return None

    # Get game_id and turn from binary arrays or legacy meta dicts
    if meta_game_ids is not None:
        game_id = str(meta_game_ids[best_idx])
        turn_idx = int(meta_turns[best_idx])
    else:
        m = meta[best_idx]
        game_id = m["game_id"]
        turn_idx = m["turn"]

    # Determine occupancy_locked: high occ_sim and all close candidates agree
    # on game_id.  This means we're confident the occupancy from the golden
    # CGP is correct and can be trusted over local is_tile() results.
    occupancy_locked = False
    if best_occ_sim >= 0.95:
        top_ids = set()
        for (csim, osim, ci) in scored:
            if csim >= best_sim - 0.02:
                if meta_game_ids is not None:
                    top_ids.add(str(meta_game_ids[ci]))
                else:
                    top_ids.add(meta[ci]["game_id"])
        occupancy_locked = len(top_ids) == 1

    # Load the single matching GCG to get the exact CGP and player info
    gcg_path = os.path.join(GCG_DIR, f"{game_id}.gcg")
    try:
        with open(gcg_path, encoding="utf-8") as f:
            gcg_text = f.read()
        states = parse_gcg(gcg_text)
        state = states[turn_idx] if turn_idx < len(states) else states[-1]

        # Disambiguate identical boards (e.g. exchange) using rack
        query_rack = rack_from_cgp(ocr_cgp)
        if query_rack:
            turn_idx, state = _disambiguate_by_rack(states, turn_idx, query_rack)

        golden_cgp = state.to_cgp()
        players = state.players
    except Exception as e:
        print(f"  Warning: could not load GCG for {game_id}: {e}")
        golden_cgp = None
        players = []

    t_end = time.time()
    print(f"  Best match: {game_id} turn {turn_idx} (similarity={best_sim:.3f}, occ_sim={best_occ_sim:.3f}, locked={occupancy_locked})")
    print(f"  Timing: load={t_load-t_start:.2f}s jaccard={t_jaccard-t_load:.2f}s total={t_end-t_start:.2f}s")
    return {
        "game_id": game_id,
        "turn": turn_idx,
        "golden_cgp": golden_cgp,
        "similarity": best_sim,
        "occupancy_locked": occupancy_locked,
        "players": players,
        "lookup_time": round(t_end - t_start, 2),
    }

# ---------------------------------------------------------------------------
# Board occupancy comparison
# ---------------------------------------------------------------------------

def board_occupancy(cgp: str) -> frozenset:
    """Return set of (row, col) positions that have tiles, from a CGP string."""
    board_str = cgp.split()[0]
    rows = board_str.split("/")
    occupied = set()
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                occupied.add((r, c))
                c += 1
    return frozenset(occupied)

def board_letters(cgp: str) -> dict:
    """Return dict of (row, col) -> letter from a CGP string."""
    board_str = cgp.split()[0]
    rows = board_str.split("/")
    letters = {}
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                letters[(r, c)] = ch.upper()
                c += 1
    return letters

def occupancy_similarity(cgp_a: str, cgp_b: str) -> float:
    """Jaccard similarity of occupied cells between two boards."""
    occ_a = board_occupancy(cgp_a)
    occ_b = board_occupancy(cgp_b)
    if not occ_a and not occ_b:
        return 1.0
    intersection = len(occ_a & occ_b)
    union = len(occ_a | occ_b)
    return intersection / union if union > 0 else 0.0

def letter_accuracy(cgp_ocr: str, cgp_truth: str) -> float:
    """Fraction of occupied cells with matching letters."""
    occ_a = board_occupancy(cgp_ocr)
    occ_b = board_occupancy(cgp_truth)
    common = occ_a & occ_b
    if not common:
        return 0.0
    letters_a = board_letters(cgp_ocr)
    letters_b = board_letters(cgp_truth)
    matches = sum(1 for pos in common if letters_a.get(pos) == letters_b.get(pos))
    return matches / len(common)

# ---------------------------------------------------------------------------
# Rack matching
# ---------------------------------------------------------------------------

def rack_from_cgp(cgp: str) -> str:
    """Extract rack string from a CGP.  Format: BOARD RACK/ S1 S2 lex LEX;"""
    parts = cgp.split()
    if len(parts) >= 2:
        return parts[1].rstrip("/").upper()
    return ""

def rack_similarity(rack_a: str, rack_b: str) -> float:
    """Similarity between two racks based on sorted letter overlap.
    Returns fraction of letters in common (Jaccard on multisets)."""
    if not rack_a or not rack_b:
        return 0.0
    sa = sorted(rack_a.replace("?", ""))
    sb = sorted(rack_b.replace("?", ""))
    if not sa and not sb:
        return 1.0
    # Multiset intersection
    common = 0
    j = 0
    for c in sa:
        while j < len(sb) and sb[j] < c:
            j += 1
        if j < len(sb) and sb[j] == c:
            common += 1
            j += 1
    total = max(len(sa), len(sb))
    return common / total if total > 0 else 0.0

def _disambiguate_by_rack(states, turn_idx, query_rack):
    """Fix the golden CGP rack using the query rack to disambiguate.

    Handles two cases:
    1. Adjacent turns with identical boards (e.g. exchange followed by
       opponent move that gets challenged off).
    2. Stale rack after an exchange — the GCG only reveals the post-exchange
       rack on the player's *next* move line.  When the current rack doesn't
       match the query, look forward for the on-turn player's next move
       and use that rack instead.

    Returns the best (turn_idx, state), possibly with a corrected rack.
    """
    if not query_rack or turn_idx < 0 or turn_idx >= len(states):
        return turn_idx, states[turn_idx] if turn_idx < len(states) else None

    state = states[turn_idx]
    base_board = state.to_cgp().split()[0]
    current_rack = rack_from_cgp(state.to_cgp())
    current_rsim = rack_similarity(query_rack, current_rack)

    # Case 1: check adjacent turns with identical boards
    best_ti, best_state, best_rsim = turn_idx, state, current_rsim
    for delta in (-1, 1, -2, 2):
        ni = turn_idx + delta
        if 0 < ni < len(states):
            nb = states[ni].to_cgp().split()[0]
            if nb == base_board:
                r = rack_from_cgp(states[ni].to_cgp())
                rsim = rack_similarity(query_rack, r)
                if rsim > best_rsim:
                    best_ti, best_state, best_rsim = ni, states[ni], rsim

    if best_ti != turn_idx:
        print(f"  Rack disambiguation (identical board): turn {turn_idx} -> {best_ti}")
        return best_ti, best_state

    # Case 2: stale rack after exchange — look forward for on-turn player's
    # next move to get the real rack.  Always try this and pick whichever
    # rack matches the query better.
    on_turn = state.on_turn
    for ni in range(turn_idx + 1, min(turn_idx + 4, len(states))):
        future = states[ni]
        # The on-turn player's rack gets updated when they make a move,
        # which flips on_turn away from them.
        if future.on_turn != on_turn:
            # Player 'on_turn' just played at state[ni], so
            # future.racks[on_turn] is their pre-move rack (the real one).
            forward_rack = future.racks[on_turn]
            rsim = rack_similarity(query_rack, forward_rack)
            if rsim > best_rsim:
                # Patch the rack into the matched state
                import copy
                patched = copy.deepcopy(state)
                patched.racks[on_turn] = forward_rack
                print(f"  Rack fix-up: {current_rack} -> {forward_rack} "
                      f"(from turn {ni}, sim {rsim:.2f})")
                return turn_idx, patched
            break  # only check the first future move by this player

    return best_ti, best_state

# ---------------------------------------------------------------------------
# Score matching
# ---------------------------------------------------------------------------

def scores_from_cgp(cgp: str):
    """Extract (score1, score2) from a CGP string."""
    import re
    m = re.search(r'/\s*(\d+)\s+(\d+)', cgp)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

# ---------------------------------------------------------------------------
# Main matching logic
# ---------------------------------------------------------------------------

def find_matching_turn(ocr_cgp: str, states: list, ocr_scores=None, tolerance=5):
    """
    Find the turn in a game that best matches the OCR'd board.
    Returns (turn_index, similarity_score, state).
    """
    best = (0, 0.0, None)
    for i, state in enumerate(states[1:], 1):  # skip initial empty board
        truth_cgp = state.to_cgp()

        # Occupancy must be very close (>= 0.92 Jaccard)
        occ_sim = occupancy_similarity(ocr_cgp, truth_cgp)
        if occ_sim < 0.90:
            continue

        # Letter accuracy
        letter_acc = letter_accuracy(ocr_cgp, truth_cgp)

        # Score proximity (if provided)
        score_bonus = 0.0
        if ocr_scores:
            t_s0, t_s1 = scores_from_cgp(truth_cgp)
            o_s0, o_s1 = ocr_scores
            if t_s0 is not None and o_s0 is not None:
                # Check both orderings
                diff_a = abs(t_s0 - o_s0) + abs(t_s1 - o_s1)
                diff_b = abs(t_s0 - o_s1) + abs(t_s1 - o_s0)
                diff = min(diff_a, diff_b)
                if diff <= tolerance:
                    score_bonus = 0.1 * (1 - diff / (tolerance * 10))

        combined = occ_sim * 0.6 + letter_acc * 0.3 + score_bonus
        if combined > best[1]:
            best = (i, combined, state)

    # Disambiguate identical boards (e.g. exchange) using rack
    if best[2] is not None:
        query_rack = rack_from_cgp(ocr_cgp)
        if query_rack:
            new_ti, new_state = _disambiguate_by_rack(states, best[0], query_rack)
            best = (new_ti, best[1], new_state)

    return best

def match_screenshot(
    ocr_cgp: str,
    player_names: list = None,
    ocr_scores: tuple = None,
    min_similarity: float = 0.85,
):
    """
    Given OCR output for a screenshot, find the matching Woogles game and turn.

    Uses the prebuilt index (build_index.py) for fast vectorized search.
    If player_names are given, also fetches their recent games from the API
    in case they're not yet in the index.

    Returns dict with:
        game_id, turn, golden_cgp, similarity, players
    Or None if no match found.
    """
    players_to_search = player_names or []

    # Fast path: indexed search
    result = match_screenshot_indexed(ocr_cgp, ocr_scores, min_similarity)
    if result:
        return result

    # If player names given, fetch their recent games (may not be in index yet)
    # and search those directly.
    if not players_to_search:
        return None

    extra_ids = set()
    indexed_ids = set()
    index = _load_index()
    if index:
        _, _, _, meta, meta_game_ids, _, _ = index
        if meta_game_ids is not None:
            indexed_ids = set(meta_game_ids)
        elif meta:
            indexed_ids = {m["game_id"] for m in meta}

    for player in players_to_search:
        print(f"  Fetching recent games for {player}...")
        try:
            games = get_recent_games(player, num=50)
            for g in games:
                gid = g["game_id"]
                if gid not in indexed_ids:
                    extra_ids.add(gid)
        except Exception as e:
            print(f"  Warning: could not fetch games for {player}: {e}")
        time.sleep(0.3)

    if not extra_ids:
        return None

    print(f"  Scanning {len(extra_ids)} un-indexed games for {players_to_search}...")
    best_match = None
    best_sim = 0.0

    for game_id in extra_ids:
        gcg_path = os.path.join(GCG_DIR, f"{game_id}.gcg")
        if not os.path.exists(gcg_path):
            try:
                gcg = get_gcg(game_id)
                if gcg:
                    save_game(game_id, gcg)
                    time.sleep(0.2)
            except Exception:
                continue
        if not os.path.exists(gcg_path):
            continue
        try:
            with open(gcg_path, encoding="utf-8") as f:
                gcg_text = f.read()
            states = parse_gcg(gcg_text)
        except Exception:
            continue

        turn_idx, sim, state = find_matching_turn(ocr_cgp, states, ocr_scores)
        if sim > best_sim:
            best_sim = sim
            best_match = {
                "game_id": game_id,
                "turn": turn_idx,
                "golden_cgp": state.to_cgp() if state else None,
                "similarity": sim,
                "occupancy_locked": False,
                "players": state.players if state else [],
            }
            if sim >= 0.98:
                break

    if best_match and best_sim >= min_similarity:
        print(f"  Best match (un-indexed): {best_match['game_id']} turn {best_match['turn']} "
              f"(similarity={best_sim:.3f})")
        return best_match

    print(f"  No confident match found (best similarity={best_sim:.3f})")
    return None

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match screenshot to Woogles game")
    parser.add_argument("--cgp", help="Approximate CGP from OCR")
    parser.add_argument("--players", nargs="+", help="Player names from the screenshot")
    parser.add_argument("--scores", nargs=2, type=int, help="Scores from screenshot")
    parser.add_argument("--image", help="Screenshot image (use cgpbot OCR)")
    parser.add_argument("--server", default="http://localhost:8080",
                        help="cgpbot server URL for OCR")
    args = parser.parse_args()

    ocr_cgp = args.cgp
    ocr_scores = tuple(args.scores) if args.scores else None
    players = args.players or []

    if args.image:
        # Use cgpbot server to OCR the image
        print(f"OCR-ing {args.image} via {args.server}...")
        with open(args.image, "rb") as f:
            resp = requests.post(f"{args.server}/analyze-gemini",
                files={"image": f}, timeout=120)
        # Parse NDJSON stream
        for line in resp.text.strip().split("\n"):
            try:
                d = json.loads(line)
                if d.get("cgp"):
                    ocr_cgp = d["cgp"]
                    print(f"OCR CGP: {ocr_cgp[:80]}...")
                    # Extract player names and scores if available
                    m = __import__("re").search(r'/\s*(\d+)\s+(\d+)', ocr_cgp)
                    if m and not ocr_scores:
                        ocr_scores = (int(m.group(1)), int(m.group(2)))
            except Exception:
                pass

    if not ocr_cgp:
        print("Error: provide --cgp or --image")
        sys.exit(1)

    result = match_screenshot(ocr_cgp, players, ocr_scores)
    if result:
        print(json.dumps(result, indent=2))
    else:
        sys.exit(1)
