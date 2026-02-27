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

sys.path.insert(0, os.path.dirname(__file__))
from fetch_games import get_recent_games, get_gcg, save_game
from gcg_parser import parse_gcg, GameState

API = "https://woogles.io/api/game_service.GameMetadataService"
GCG_DIR = os.path.join(os.path.dirname(__file__), "..", "gcg")
META_DIR = os.path.join(os.path.dirname(__file__), "..", "meta")

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

    return best

def match_screenshot(
    ocr_cgp: str,
    player_names: list = None,
    ocr_scores: tuple = None,
    min_similarity: float = 0.85,
):
    """
    Given OCR output for a screenshot, find the matching Woogles game and turn.

    Returns dict with:
        game_id, turn, golden_cgp, similarity, players
    Or None if no match found.
    """
    players_to_search = player_names or []

    # Collect candidate game IDs
    candidate_ids = set()

    # Check already-downloaded GCGs first
    if os.path.exists(GCG_DIR):
        for fname in os.listdir(GCG_DIR):
            if fname.endswith(".gcg"):
                candidate_ids.add(fname[:-4])

    # Also fetch recent games for mentioned players
    for player in players_to_search:
        print(f"  Fetching recent games for {player}...")
        try:
            games = get_recent_games(player, num=50)
            for g in games:
                candidate_ids.add(g["game_id"])
        except Exception as e:
            print(f"  Warning: could not fetch games for {player}: {e}")
        time.sleep(0.3)

    print(f"  Searching {len(candidate_ids)} candidate games...")

    best_match = None
    best_sim = 0.0

    for game_id in candidate_ids:
        gcg_path = os.path.join(GCG_DIR, f"{game_id}.gcg")

        # Fetch GCG if not cached
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

        with open(gcg_path, encoding="utf-8") as f:
            gcg_text = f.read()

        try:
            states = parse_gcg(gcg_text)
        except Exception as e:
            print(f"  Warning: parse error for {game_id}: {e}")
            continue

        turn_idx, sim, state = find_matching_turn(ocr_cgp, states, ocr_scores)

        if sim > best_sim:
            best_sim = sim
            best_match = {
                "game_id": game_id,
                "turn": turn_idx,
                "golden_cgp": state.to_cgp() if state else None,
                "similarity": sim,
                "players": state.players if state else [],
            }
            if sim >= 0.98:
                break  # confident match, stop searching

    if best_match and best_sim >= min_similarity:
        print(f"  Best match: {best_match['game_id']} turn {best_match['turn']} "
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
