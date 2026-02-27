#!/usr/bin/env python3
"""
Woogles game lookup — thin stdin/stdout wrapper around match_game.py.

Reads a JSON blob from stdin:
  {"cgp": "...", "players": ["p1", "p2"], "scores": [s1, s2]}

Writes the JSON match result (or null) to stdout.
All diagnostic print() output is redirected to stderr to keep stdout clean.
"""

import json
import os
import sys
import builtins

# Redirect print() to stderr so stdout stays clean for JSON output.
_real_print = builtins.print
def _stderr_print(*args, **kwargs):
    kwargs['file'] = sys.stderr
    _real_print(*args, **kwargs)
builtins.print = _stderr_print

sys.path.insert(0, os.path.dirname(__file__))
from match_game import match_screenshot

data = json.loads(sys.stdin.read())
cgp = data.get('cgp', '')
players = data.get('players', [])
scores = data.get('scores')
if scores:
    scores = tuple(scores)

result = match_screenshot(cgp, players, scores)

# Restore stdout print for the result
builtins.print = _real_print
print(json.dumps(result) if result else 'null')
