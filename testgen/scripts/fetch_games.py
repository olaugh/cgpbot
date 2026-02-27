#!/usr/bin/env python3
"""
Fetch Woogles games and save GCG files to testgen/gcg/.

Usage:
    python3 fetch_games.py                    # fetch curated set
    python3 fetch_games.py --player cesar     # fetch from specific player
    python3 fetch_games.py --game AmSAGtDHDU  # fetch single game
"""

import argparse
import json
import os
import sys
import time
import requests

API = "https://woogles.io/api/game_service.GameMetadataService"
GCG_DIR = os.path.join(os.path.dirname(__file__), "..", "gcg")
META_DIR = os.path.join(os.path.dirname(__file__), "..", "meta")

os.makedirs(GCG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Woogles API helpers
# ---------------------------------------------------------------------------

def get_recent_games(username, num=50, offset=0):
    r = requests.post(f"{API}/GetRecentGames",
        headers={"Content-Type": "application/json"},
        json={"username": username, "numGames": num, "offset": offset},
        timeout=30)
    r.raise_for_status()
    return r.json().get("game_info", [])

def get_gcg(game_id):
    r = requests.post(f"{API}/GetGCG",
        headers={"Content-Type": "application/json"},
        json={"game_id": game_id},
        timeout=30)
    r.raise_for_status()
    return r.json().get("gcg", "")

# ---------------------------------------------------------------------------
# GCG analysis helpers
# ---------------------------------------------------------------------------

def analyze_gcg(gcg_text):
    """Extract metadata from a GCG string."""
    lines = gcg_text.strip().split("\n")
    meta = {
        "lexicon": None,
        "players": [],
        "move_count": 0,
        "has_phony": False,
        "has_blank_on_board": False,
        "has_exchange": False,
        "final_scores": [],
        "game_type": "classic",
    }
    for line in lines:
        if line.startswith("#lexicon"):
            meta["lexicon"] = line.split(None, 1)[1].strip()
        elif line.startswith("#player"):
            parts = line.split(None, 2)
            if len(parts) >= 3:
                meta["players"].append(parts[2].strip())
        elif line.startswith("#game-type"):
            meta["game_type"] = line.split(None, 1)[1].strip()
        elif line.startswith(">"):
            # Move line: >player: RACK COORD WORD +score total
            meta["move_count"] += 1
            parts = line.split()
            if len(parts) >= 4:
                rack = parts[1] if len(parts) > 1 else ""
                word = parts[3] if len(parts) > 3 else ""
                score_part = parts[4] if len(parts) > 4 else ""
                # Phony: score is negative (withdrawal)
                if score_part.startswith("-"):
                    meta["has_phony"] = True
                # Exchange: word starts with - (exchange notation)
                if word.startswith("-") and len(word) > 1 and word[1:].isalpha():
                    meta["has_exchange"] = True
                # Blank on board: lowercase letter in word (not in rack)
                if word and any(c.islower() for c in word if c.isalpha()):
                    meta["has_blank_on_board"] = True
        # Final scores from last move
    # Parse final scores from last >player line with total
    for line in reversed(lines):
        if line.startswith(">"):
            parts = line.split()
            if len(parts) >= 5:
                try:
                    meta["final_scores"].append(int(parts[-1]))
                except ValueError:
                    pass
            break
    return meta

# ---------------------------------------------------------------------------
# Save a game
# ---------------------------------------------------------------------------

def save_game(game_id, gcg_text, game_info=None):
    """Save GCG + metadata sidecar for a game."""
    gcg_path = os.path.join(GCG_DIR, f"{game_id}.gcg")
    meta_path = os.path.join(META_DIR, f"{game_id}.json")

    if not gcg_text:
        print(f"  [skip] {game_id}: empty GCG")
        return False

    # GCG file
    with open(gcg_path, "w", encoding="utf-8") as f:
        f.write(gcg_text)

    # Metadata sidecar
    meta = analyze_gcg(gcg_text)
    if game_info:
        meta["source"] = "woogles"
        meta["game_id"] = game_id
        meta["players_info"] = [
            {"nickname": p.get("nickname"), "is_bot": p.get("is_bot", False)}
            for p in game_info.get("players", [])
        ]
        meta["created_at"] = game_info.get("created_at")
        meta["scores"] = game_info.get("scores")
        meta["challenge_rule"] = (
            game_info.get("game_request", {}).get("challenge_rule", "")
        )
        meta["rating_mode"] = (
            game_info.get("game_request", {}).get("rating_mode", "")
        )
        meta["board_layout"] = (
            game_info.get("game_request", {})
            .get("rules", {})
            .get("board_layout_name", "CrosswordGame")
        )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  [saved] {game_id}: {meta.get('lexicon','?')} "
          f"moves={meta['move_count']} "
          f"phony={'Y' if meta['has_phony'] else 'N'} "
          f"blank={'Y' if meta['has_blank_on_board'] else 'N'}")
    return True

# ---------------------------------------------------------------------------
# Curated game list (from the test generation plan)
# ---------------------------------------------------------------------------

CURATED_GAMES = [
    # Phonies
    "AmSAGtDHDU",  # cesar vs BestBot, NWL23, DOUBLE, 1 phony
    "vih65vN7zw",  # BestBot vs cesar, NWL23, 2 phonies
    "x92DfARyWc",  # cesar vs BestBot, NWL23, 2 phonies
    "We9r93HF6q",  # BestBot vs cesar, NWL23, 2 phonies
    "Ju6yzk8ugd",  # tarkovsky7 vs BestBot, NWL23, 2 phonies
    "CTDVDRDQkM",  # BestBot vs magrathean, CSW24, 2 phonies
    "UyNCs9cPuf",  # CarlSagan121520 vs BestBot, NWL23, 2 phonies
    "h3c2z3Uykv",  # magrathean vs budak, CSW24, phony + challenge bonus
    "DoM4tL9vH2",  # thisguy vs squush, human vs human, phonies
    "BzAQ3Gifu9",  # Heron vs thisguy, human vs human, phonies
]

# ---------------------------------------------------------------------------
# Fetch players for game discovery
# ---------------------------------------------------------------------------

DISCOVERY_PLAYERS = [
    "cesar", "BestBot", "magrathean", "thisguy", "josh",
    "tarkovsky7", "squush", "woogie",
]

def fetch_player_games(username, pages=2, delay=0.5):
    """Fetch and return all games for a player (paginated)."""
    all_games = []
    for page in range(pages):
        games = get_recent_games(username, 50, page * 50)
        if not games:
            break
        all_games.extend(games)
        if len(games) < 50:
            break
        time.sleep(delay)
    return all_games

def categorize_games(games_with_gcg):
    """Sort games into categories for balanced test coverage."""
    categories = {
        "phony": [],
        "endgame": [],
        "blank": [],
        "sparse": [],    # < 18 moves
        "medium": [],    # 18-24 moves
        "dense": [],     # 25+ moves
        "csw": [],
        "exchange": [],
    }
    for g in games_with_gcg:
        meta = g["_meta"]
        gid = g["game_id"]
        if meta["has_phony"]:
            categories["phony"].append(gid)
        if meta["has_blank_on_board"]:
            categories["blank"].append(gid)
        if meta.get("lexicon", "").startswith("CSW"):
            categories["csw"].append(gid)
        if meta["has_exchange"]:
            categories["exchange"].append(gid)
        mc = meta["move_count"]
        if mc < 18:
            categories["sparse"].append(gid)
        elif mc < 25:
            categories["medium"].append(gid)
        else:
            categories["dense"].append(gid)
    return categories

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Woogles games")
    parser.add_argument("--game", help="Fetch a single game by ID")
    parser.add_argument("--player", help="Fetch recent games for a player")
    parser.add_argument("--curated", action="store_true", default=False,
                        help="Fetch the curated game list only")
    parser.add_argument("--discover", action="store_true", default=False,
                        help="Fetch games from all discovery players")
    parser.add_argument("--pages", type=int, default=2,
                        help="Pages per player (50 games/page)")
    args = parser.parse_args()

    # Default: fetch curated list
    if not any([args.game, args.player, args.curated, args.discover]):
        args.curated = True

    if args.game:
        print(f"Fetching game {args.game}...")
        gcg = get_gcg(args.game)
        save_game(args.game, gcg)
        return

    if args.player:
        print(f"Fetching games for {args.player}...")
        games = fetch_player_games(args.player, args.pages)
        print(f"  {len(games)} games found, fetching GCGs...")
        for g in games:
            gid = g["game_id"]
            if os.path.exists(os.path.join(GCG_DIR, f"{gid}.gcg")):
                print(f"  [skip] {gid}: already exists")
                continue
            gcg = get_gcg(gid)
            save_game(gid, gcg, g)
            time.sleep(0.3)
        return

    if args.curated:
        print(f"Fetching {len(CURATED_GAMES)} curated games...")
        for gid in CURATED_GAMES:
            if os.path.exists(os.path.join(GCG_DIR, f"{gid}.gcg")):
                print(f"  [skip] {gid}: already exists")
                continue
            gcg = get_gcg(gid)
            save_game(gid, gcg)
            time.sleep(0.3)
        return

    if args.discover:
        print(f"Discovering games from {len(DISCOVERY_PLAYERS)} players...")
        all_game_ids = set()
        for player in DISCOVERY_PLAYERS:
            print(f"\nFetching {player}...")
            games = fetch_player_games(player, args.pages)
            for g in games:
                gid = g["game_id"]
                if gid in all_game_ids:
                    continue
                all_game_ids.add(gid)
                if os.path.exists(os.path.join(GCG_DIR, f"{gid}.gcg")):
                    print(f"  [skip] {gid}")
                    continue
                gcg = get_gcg(gid)
                save_game(gid, gcg, g)
                time.sleep(0.3)
        print(f"\nTotal unique games: {len(all_game_ids)}")

if __name__ == "__main__":
    main()
