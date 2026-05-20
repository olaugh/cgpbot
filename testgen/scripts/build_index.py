#!/usr/bin/env python3
"""
Build a fast lookup index from all cached GCG files.

Produces testgen/index/ containing:
  occ.npy            uint8  (N, 225) — 1=occupied, 0=empty per cell
  letters.npy        uint16 (N, 225) — 0=empty, ord(letter) for occupied cells
                                        (uppercase regular, lowercase blank)
  scores.npy         int16 (N, 2)    — [score1, score2] at each turn
  meta.jsonl         NDJSON — {"game_id":..., "turn":..., "lexicon":...}
  meta_game_ids.npy  U12 (N,) — game IDs as fixed-width strings
  meta_turns.npy     uint16 (N,) — turn numbers
  n_occ.npy          int32 (N,) — pre-computed row sums of occ
  indexed_games.txt  one game_id per line (for fast incremental skip)

N is the total number of board states across all games and turns.

Uses chunked writes to raw binary files during build so RAM stays
under ~100MB regardless of how many games are indexed.

Supports incremental updates: only new GCG files are parsed and appended.
Use --full to force a complete rebuild.

Run after fetching new GCG files:
    python3 build_index.py              # incremental (fast)
    python3 build_index.py --full       # from scratch (~2h)
"""

import argparse
import json
import os
import re
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gcg_parser import parse_gcg

GCG_DIR = os.path.join(os.path.dirname(__file__), "..", "gcg")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")

CHUNK_SIZE = 10000  # flush to disk every 10K states


def _write_npy_header(f, dtype, shape):
    """Write a .npy v2 header. Returns header size (data starts right after)."""
    header_dict = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (
        np.dtype(dtype).str, repr(tuple(shape)))
    header_bytes = header_dict.encode('latin1')
    # v2 header: 6 magic + 2 version + 4 header_len + header + padding
    # Pad to 64-byte alignment
    base = 12 + len(header_bytes)
    pad = (64 - base % 64) % 64
    if pad == 0:
        pad = 64
    header_bytes += b' ' * (pad - 1) + b'\n'
    f.write(b'\x93NUMPY')
    f.write(struct.pack('BB', 2, 0))
    f.write(struct.pack('<I', len(header_bytes)))
    f.write(header_bytes)
    return f.tell()


def _read_npy_shape(path):
    """Read the shape from a .npy header without loading data."""
    with open(path, 'rb') as f:
        magic = f.read(6)
        major, minor = struct.unpack('BB', f.read(2))
        if major == 1:
            header_len = struct.unpack('<H', f.read(2))[0]
        else:
            header_len = struct.unpack('<I', f.read(4))[0]
        header = f.read(header_len).decode('latin1').strip()
        m = re.search(r"'shape':\s*\(([^)]+)\)", header)
        return tuple(int(x.strip()) for x in m.group(1).split(',') if x.strip())


def _rewrite_npy_shape(path, new_shape):
    """Rewrite the shape in a .npy header in-place (no data copy)."""
    with open(path, 'r+b') as f:
        magic = f.read(6)
        major, minor = struct.unpack('BB', f.read(2))
        if major == 1:
            old_header_len = struct.unpack('<H', f.read(2))[0]
            header_start = 10
        else:
            old_header_len = struct.unpack('<I', f.read(4))[0]
            header_start = 12
        data_start = header_start + old_header_len

        # Read existing header to get dtype
        f.seek(header_start)
        old_header = f.read(old_header_len).decode('latin1').strip()
        m = re.search(r"'descr':\s*'([^']+)'", old_header)
        descr = m.group(1)

        new_dict = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (
            descr, repr(tuple(new_shape)))
        new_bytes = new_dict.encode('latin1')
        avail = old_header_len
        if len(new_bytes) > avail - 1:
            raise ValueError("New header too large for in-place rewrite")
        new_bytes += b' ' * (avail - len(new_bytes) - 1) + b'\n'
        f.seek(header_start)
        f.write(new_bytes)

        # Truncate file to actual data
        dt = np.dtype(descr)
        data_bytes = int(np.prod(new_shape)) * dt.itemsize
        f.truncate(data_start + data_bytes)


def _process_gcg_files(gcg_files, gcg_dir, f_occ, f_let, f_sco, f_meta):
    """Parse GCG files and write board states to open file handles.

    Returns (n_states, n_errors, new_game_ids).
    """
    buf_occ = np.zeros((CHUNK_SIZE, 225), dtype=np.uint8)
    buf_let = np.zeros((CHUNK_SIZE, 225), dtype=np.uint16)
    buf_sco = np.zeros((CHUNK_SIZE, 2), dtype=np.int16)
    buf_idx = 0
    total_idx = 0
    errors = 0
    new_game_ids = []
    n_files = len(gcg_files)
    t0 = time.time()

    def flush_chunk():
        nonlocal buf_idx
        if buf_idx == 0:
            return
        f_occ.write(buf_occ[:buf_idx].tobytes())
        f_let.write(buf_let[:buf_idx].tobytes())
        f_sco.write(buf_sco[:buf_idx].tobytes())
        buf_occ[:buf_idx] = 0
        buf_let[:buf_idx] = 0
        buf_sco[:buf_idx] = 0
        buf_idx = 0

    for i, fname in enumerate(gcg_files):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            remaining = (n_files - i) / rate
            print(f"  {i}/{n_files}  ({rate:.0f}/s, ~{remaining/60:.0f}m left)  "
                  f"states={total_idx}", flush=True)

        game_id = fname[:-4]
        gcg_path = os.path.join(gcg_dir, fname)
        try:
            with open(gcg_path, encoding="utf-8") as f:
                gcg_text = f.read()
            states = parse_gcg(gcg_text)
        except Exception:
            errors += 1
            continue

        new_game_ids.append(game_id)
        lexicon = states[0].lexicon if states else ""

        for turn_idx, state in enumerate(states[1:], 1):
            has_tile = False
            for r in range(15):
                for c in range(15):
                    cell = state.board[r][c]
                    if cell.letter:
                        has_tile = True
                        k = r * 15 + c
                        buf_occ[buf_idx, k] = 1
                        buf_let[buf_idx, k] = ord(cell.letter)
            if not has_tile:
                continue
            buf_sco[buf_idx, 0] = state.scores[0]
            buf_sco[buf_idx, 1] = state.scores[1]
            f_meta.write(f'{{"game_id":"{game_id}","turn":{turn_idx},"lexicon":"{lexicon}"}}\n')
            buf_idx += 1
            total_idx += 1

            if buf_idx >= CHUNK_SIZE:
                flush_chunk()

    flush_chunk()
    return total_idx, errors, new_game_ids


def _build_fast_arrays(index_dir, meta_path, occ_path, total_rows):
    """Build binary meta arrays and pre-computed n_occ from existing files."""
    print("Building binary meta arrays...", flush=True)
    game_ids = []
    turns = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                game_ids.append(d["game_id"])
                turns.append(d["turn"])
    np.save(os.path.join(index_dir, "meta_game_ids.npy"),
            np.array(game_ids, dtype="U12"))
    np.save(os.path.join(index_dir, "meta_turns.npy"),
            np.array(turns, dtype=np.uint16))

    print("Computing n_occ...", flush=True)
    occ_mm = np.load(occ_path, mmap_mode="r")
    n_occ_arr = np.zeros(total_rows, dtype=np.int32)
    chunk_sz = 500000
    for ci in range(0, total_rows, chunk_sz):
        end = min(ci + chunk_sz, total_rows)
        n_occ_arr[ci:end] = occ_mm[ci:end].sum(axis=1)
    np.save(os.path.join(index_dir, "n_occ.npy"), n_occ_arr)
    del occ_mm


def _print_index_size(index_dir):
    files = ["occ.npy", "letters.npy", "scores.npy", "meta.jsonl",
             "meta_game_ids.npy", "meta_turns.npy", "n_occ.npy"]
    size_mb = sum(
        os.path.getsize(os.path.join(index_dir, fn))
        for fn in files if os.path.exists(os.path.join(index_dir, fn))
    ) / 1e6
    print(f"Index saved to {index_dir}  ({size_mb:.1f} MB)", flush=True)


# ---------------------------------------------------------------------------
# Full rebuild
# ---------------------------------------------------------------------------

def build_index(gcg_dir, index_dir):
    os.makedirs(index_dir, exist_ok=True)

    gcg_files = sorted(f for f in os.listdir(gcg_dir) if f.endswith(".gcg"))
    n_files = len(gcg_files)

    print(f"Full build: indexing {n_files} GCG files from {gcg_dir}", flush=True)

    # Open raw binary files with .npy headers
    estimated = n_files * 30  # headroom for header rewrite
    occ_path = os.path.join(index_dir, "occ.npy")
    letters_path = os.path.join(index_dir, "letters.npy")
    scores_path = os.path.join(index_dir, "scores.npy")
    meta_path = os.path.join(index_dir, "meta.jsonl")

    f_occ = open(occ_path, 'wb')
    f_let = open(letters_path, 'wb')
    f_sco = open(scores_path, 'wb')

    _write_npy_header(f_occ, np.uint8, (estimated, 225))
    _write_npy_header(f_let, np.uint16, (estimated, 225))
    _write_npy_header(f_sco, np.int16, (estimated, 2))

    f_meta = open(meta_path, 'w')

    t0 = time.time()
    total_idx, errors, all_game_ids = _process_gcg_files(
        gcg_files, gcg_dir, f_occ, f_let, f_sco, f_meta)

    f_occ.close()
    f_let.close()
    f_sco.close()
    f_meta.close()

    elapsed = time.time() - t0
    print(f"\nParsed {n_files - errors} games ({errors} errors) in {elapsed:.1f}s",
          flush=True)
    print(f"Total board states: {total_idx}", flush=True)

    # Fix .npy headers to actual shape
    print(f"Fixing array headers ({estimated} -> {total_idx})...", flush=True)
    _rewrite_npy_shape(occ_path, (total_idx, 225))
    _rewrite_npy_shape(letters_path, (total_idx, 225))
    _rewrite_npy_shape(scores_path, (total_idx, 2))

    # Remove old meta.json if present
    old_meta = os.path.join(index_dir, "meta.json")
    if os.path.exists(old_meta):
        os.remove(old_meta)

    # Write indexed game list
    indexed_path = os.path.join(index_dir, "indexed_games.txt")
    with open(indexed_path, 'w') as f:
        for gid in all_game_ids:
            f.write(gid + '\n')

    _build_fast_arrays(index_dir, meta_path, occ_path, total_idx)
    _print_index_size(index_dir)


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def update_index(gcg_dir, index_dir):
    """Append only new games to the existing index."""
    occ_path = os.path.join(index_dir, "occ.npy")
    letters_path = os.path.join(index_dir, "letters.npy")
    scores_path = os.path.join(index_dir, "scores.npy")
    meta_path = os.path.join(index_dir, "meta.jsonl")
    indexed_path = os.path.join(index_dir, "indexed_games.txt")

    if not os.path.exists(occ_path):
        print("No existing index found — doing full build.", flush=True)
        build_index(gcg_dir, index_dir)
        return

    # Load set of already-indexed game IDs
    if os.path.exists(indexed_path):
        with open(indexed_path) as f:
            indexed = set(line.strip() for line in f if line.strip())
    else:
        # Rebuild from meta.jsonl (slower but works for old indices)
        print("No indexed_games.txt — scanning meta.jsonl...", flush=True)
        indexed = set()
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    indexed.add(json.loads(line)["game_id"])
        # Write it out for next time
        with open(indexed_path, 'w') as f:
            for gid in sorted(indexed):
                f.write(gid + '\n')

    # Find new GCG files
    all_gcgs = sorted(f for f in os.listdir(gcg_dir) if f.endswith(".gcg"))
    new_gcgs = [f for f in all_gcgs if f[:-4] not in indexed]

    if not new_gcgs:
        print(f"Index is up to date ({len(indexed)} games, "
              f"{len(all_gcgs)} GCG files).", flush=True)
        return

    print(f"Incremental update: {len(new_gcgs)} new games "
          f"({len(indexed)} already indexed)", flush=True)

    # Read existing row count
    existing_rows = _read_npy_shape(occ_path)[0]
    print(f"Existing index: {existing_rows} board states", flush=True)

    # Open .npy files in append mode (seek to end of data)
    f_occ = open(occ_path, 'r+b')
    f_occ.seek(0, 2)  # seek to EOF
    f_let = open(letters_path, 'r+b')
    f_let.seek(0, 2)
    f_sco = open(scores_path, 'r+b')
    f_sco.seek(0, 2)
    f_meta = open(meta_path, 'a')

    t0 = time.time()
    new_states, errors, new_game_ids = _process_gcg_files(
        new_gcgs, gcg_dir, f_occ, f_let, f_sco, f_meta)

    f_occ.close()
    f_let.close()
    f_sco.close()
    f_meta.close()

    elapsed = time.time() - t0
    total_rows = existing_rows + new_states
    print(f"\nAdded {new_states} states from {len(new_gcgs) - errors} games "
          f"({errors} errors) in {elapsed:.1f}s", flush=True)
    print(f"Total board states: {total_rows}", flush=True)

    # Update .npy headers
    _rewrite_npy_shape(occ_path, (total_rows, 225))
    _rewrite_npy_shape(letters_path, (total_rows, 225))
    _rewrite_npy_shape(scores_path, (total_rows, 2))

    # Append to indexed_games.txt
    with open(indexed_path, 'a') as f:
        for gid in new_game_ids:
            f.write(gid + '\n')

    # Rebuild binary meta arrays and n_occ (fast — reads existing files)
    _build_fast_arrays(index_dir, meta_path, occ_path, total_rows)
    _print_index_size(index_dir)


def main():
    parser = argparse.ArgumentParser(description="Build Woogles board state index")
    parser.add_argument("--gcg-dir", default=GCG_DIR)
    parser.add_argument("--index-dir", default=INDEX_DIR)
    parser.add_argument("--full", action="store_true",
                        help="Force full rebuild (default: incremental)")
    args = parser.parse_args()
    if args.full:
        build_index(args.gcg_dir, args.index_dir)
    else:
        update_index(args.gcg_dir, args.index_dir)


if __name__ == "__main__":
    main()
