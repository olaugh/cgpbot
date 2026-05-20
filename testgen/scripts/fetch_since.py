#!/usr/bin/env python3
"""
Fetch all Woogles games since a given date using BFS through the player graph.

Usage:
    python3 fetch_since.py --since 2026-02-22

Strategy:
  1. Start with seed bots (they play against nearly everyone).
  2. For each bot game fetched, collect unique human opponents.
  3. Fetch each human opponent's games — collecting their opponents too.
  4. Repeat until no new players are discovered.

This efficiently covers the full reachable game graph without needing
a global game-listing API (which Woogles doesn't expose).
Run periodically to keep the local GCG database current.
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(__file__))
from fetch_games import get_gcg, save_game

API = "https://woogles.io/api/game_service.GameMetadataService"
GCG_DIR = os.path.join(os.path.dirname(__file__), "..", "gcg")
META_DIR = os.path.join(os.path.dirname(__file__), "..", "meta")

os.makedirs(GCG_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# Seed bots — start here because they play the most opponents.
SEED_BOTS = [
    "BestBot", "HastyBot", "BasicBot", "BetterBot", "STEEBot", "BeginnerBot",
]

# Known bot usernames — don't add these to the human BFS queue
# (their games are already covered by the seed step).
BOT_NAMES = {
    "BestBot", "HastyBot", "BasicBot", "BetterBot", "STEEBot", "BeginnerBot",
    "Best459", "best4591", "Computer", "Maximize",
}


def get_recent_games(username, num=50, offset=0):
    r = requests.post(f"{API}/GetRecentGames",
        headers={"Content-Type": "application/json"},
        json={"username": username, "numGames": num, "offset": offset},
        timeout=30)
    r.raise_for_status()
    return r.json().get("game_info", [])


def parse_ts(ts_str):
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def opponents_from_game(game_info):
    """Return set of player nicknames from a game_info dict."""
    names = set()
    for p in game_info.get("players", []):
        nick = p.get("nickname", "").strip()
        if nick:
            names.add(nick)
    return names


def fetch_player_since(username, since_dt, delay=0.3):
    """
    Fetch all games for username since since_dt.
    Returns (fetched_count, skipped_count, new_opponents_set).
    """
    fetched = 0
    skipped = 0
    new_opponents = set()
    offset = 0

    while True:
        try:
            games = get_recent_games(username, 50, offset)
        except Exception as e:
            print(f"    Warning: API error for {username}: {e}")
            break
        if not games:
            break

        done = False
        for g in games:
            created = parse_ts(g.get("created_at"))
            if created and created < since_dt:
                done = True
                break

            # Collect opponents regardless of whether we already have the GCG
            new_opponents |= opponents_from_game(g)

            gid = g["game_id"]
            gcg_path = os.path.join(GCG_DIR, f"{gid}.gcg")
            if os.path.exists(gcg_path):
                skipped += 1
                continue

            try:
                gcg = get_gcg(gid)
                if save_game(gid, gcg, g):
                    fetched += 1
                time.sleep(delay)
            except Exception as e:
                print(f"    Warning: could not fetch GCG {gid}: {e}")

        if done or len(games) < 50:
            break
        offset += 50
        time.sleep(delay)

    return fetched, skipped, new_opponents


def main():
    parser = argparse.ArgumentParser(description="Fetch Woogles games since a date (BFS)")
    parser.add_argument("--since", required=True,
                        help="Fetch games on or after this date (YYYY-MM-DD)")
    parser.add_argument("--seeds", nargs="+",
                        help="Seed players to start BFS from (default: bots)")
    args = parser.parse_args()

    try:
        since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"Invalid date: {args.since}  (use YYYY-MM-DD)")
        sys.exit(1)

    seeds = args.seeds or SEED_BOTS
    print(f"Fetching games since {args.since} via BFS starting from: {', '.join(seeds)}")
    print(f"GCG cache: {os.path.abspath(GCG_DIR)}")
    print()

    visited = set()
    queue = deque(seeds)
    total_fetched = 0
    total_skipped = 0
    wave = 0

    while queue:
        # Show wave boundaries (bots = wave 0, their opponents = wave 1, etc.)
        if wave == 0:
            print(f"=== Wave 0: seed bots ({len(seeds)}) ===")
        username = queue.popleft()

        if username in visited:
            continue
        visited.add(username)

        is_bot = username in BOT_NAMES
        print(f"  {'[bot] ' if is_bot else ''}{username}...", end=" ", flush=True)

        fetched, skipped, opponents = fetch_player_since(username, since_dt)
        print(f"+{fetched} new, {skipped} cached")
        total_fetched += fetched
        total_skipped += skipped

        # After finishing bots, announce start of human wave
        if is_bot and not any(b in queue for b in BOT_NAMES):
            new_humans = {p for p in opponents if p not in BOT_NAMES and p not in visited}
            # Queue all human opponents discovered so far
            human_queue = sorted(new_humans)
            if human_queue:
                wave += 1
                print(f"\n=== Wave {wave}: {len(human_queue)} human opponents discovered ===")
                queue.extend(human_queue)
        elif not is_bot:
            # Queue newly discovered players not yet visited
            new_players = {p for p in opponents if p not in visited and p not in queue}
            for p in sorted(new_players):
                queue.append(p)

    gcg_count = len([f for f in os.listdir(GCG_DIR) if f.endswith(".gcg")])
    print(f"\nBFS complete. Visited {len(visited)} players.")
    print(f"Fetched {total_fetched} new games, {total_skipped} already cached.")
    print(f"Total GCG cache: {gcg_count} games")


if __name__ == "__main__":
    main()
