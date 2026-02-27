# Test Case Generation

Generates Scrabble board screenshots with known ground truth for measuring OCR pipeline quality.

## Quick start

```bash
# Fetch games from Woogles
python3 scripts/fetch_games.py

# (Phase 4+) Generate screenshots with Playwright
npm install
npx playwright test
```

## Directory structure

```
testgen/
  gcg/          # downloaded GCG game files (keyed by Woogles game ID)
  scripts/      # Python scripts for game fetching and GCG parsing
  screenshots/  # generated screenshots (gitignored, large)
  package.json  # Playwright + TypeScript
```

Generated `.png` + `.cgp` pairs land in `../testdata/`.
