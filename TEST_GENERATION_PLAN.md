# Test Case Generation Plan

Generate hundreds of Scrabble board screenshots with known ground truth for
measuring OCR pipeline quality.

## Architecture

```
cgpbot/
  liwords/          # git submodule (domino14/liwords or woogles-io/liwords)
  testgen/
    playwright.config.ts
    generate.ts      # main test generation script
    gcg/             # GCG game files (input)
    testdata/        # output: .png + .cgp pairs (symlinked or copied to ../testdata)
```

## Data Sources

### Source 1: Curated Woogles Games (primary source)

The primary test data comes from **real games played on Woogles.io**, fetched
via their public Connect RPC API. No authentication required.

**API base:** `https://woogles.io/api/game_service.GameMetadataService/`

```bash
# Fetch recent games for a player (paginated)
curl -s -X POST 'https://woogles.io/api/game_service.GameMetadataService/GetRecentGames' \
  -H 'Content-Type: application/json' \
  -d '{"username":"cesar","numGames":50,"offset":0}'

# Fetch GCG for a completed game
curl -s -X POST 'https://woogles.io/api/game_service.GameMetadataService/GetGCG' \
  -H 'Content-Type: application/json' \
  -d '{"gameId":"AmSAGtDHDU"}'

# Fetch full structured game document (protobuf events)
curl -s -X POST 'https://woogles.io/api/game_service.GameMetadataService/GetGameDocument' \
  -H 'Content-Type: application/json' \
  -d '{"gameId":"AmSAGtDHDU"}'
```

The `GetRecentGames` response includes rich metadata for filtering:
`challenge_rule` (VOID/DOUBLE/FIVE_POINT), `lexicon`, `is_bot`, `scores`,
`game_end_reason`, `game_mode`, `rating_mode`. There is no server-side
search/filter — filtering must be done client-side after fetching.

#### Curated game categories (~50 games)

Each game replayed at every turn yields ~20 board positions, so 50 games
produce ~1000 turn positions spanning empty through dense boards.

**Category 1: Games with phony plays (10-12 games)**

Phonies are plays that were challenged off the board. In GCG, they appear as:
```
>cesar: AADMNSU I7 D.MAN +29 61     ← cesar plays DAMAN
>cesar: AADMNSU -- -29 32            ← challenged off, score reverts
```
In the structured API: `PHONY_TILES_RETURNED` event with `lost_score` field.

Games with phonies are critical because the board state changes mid-turn —
tiles appear then get removed, which tests whether we capture the correct
post-challenge board state.

| Game ID | Players | Phonies | Notes |
|---------|---------|---------|-------|
| `AmSAGtDHDU` | cesar vs BestBot | 1 | NWL23, DOUBLE challenge |
| `vih65vN7zw` | BestBot vs cesar | 2 | NWL23 |
| `x92DfARyWc` | cesar vs BestBot | 2 | NWL23 |
| `We9r93HF6q` | BestBot vs cesar | 2 | NWL23 |
| `Ju6yzk8ugd` | tarkovsky7 vs BestBot | 2 | NWL23 |
| `CTDVDRDQkM` | BestBot vs magrathean | 2 | CSW24 |
| `UyNCs9cPuf` | CarlSagan121520 vs BestBot | 2 | NWL23 |
| `h3c2z3Uykv` | magrathean vs budak | both | CSW24, phony + challenge bonus |
| `DoM4tL9vH2` | thisguy vs squush | yes | Human vs human |
| `BzAQ3Gifu9` | Heron vs thisguy | yes | Human vs human |

**Category 2: Endgame situations (5-7 games)**

Endgame = bag is empty, players playing out remaining tiles. Dense boards
with 80+ tiles. In GCG, end-of-game appears as:
```
>bnjy: ADEI K8 IDEA +14 450         ← final play-out
>bnjy: (ETU) +6 456                 ← winner gets opponent's remaining tiles
```

Detect endgame by tracking tile count: start with 100 tiles (English), when
`tiles_on_board + tiles_on_racks >= 100`, bag is empty. Typically the last
4-6 moves of a game. Select games with 25+ moves for rich endgame positions.

**Category 3: Blanks on board (8-10 games)**

97% of Woogles games have blanks. Select games where blanks appear in
different positions (early, mid, late) and as different letters. Blanks
render differently in liwords (no point value indicator).

**Category 4: Different board densities (10-12 games)**

| Density | Move count | Examples |
|---------|------------|----------|
| Sparse (14-18 moves) | Short/resigned | `bpX9qVgNKy`, `HfdgNM5v4i` |
| Medium (20-24 moves) | Typical game | `nSStvcU8Du`, `A8Y8jkW87S` |
| Dense (25-28 moves) | Long endgame | `cWodMqmUWF`, `cZxGeijR6W` |

**Category 5: Different lexicons and languages (8-10 games)**

| Lexicon | Language | Letter Distribution | Notes |
|---------|----------|---------------------|-------|
| NWL23 | English (NA) | `english` | Most cesar games |
| CSW24 | English (World) | `english` | `CTDVDRDQkM`, `YB6rxJBVGU` |
| ECWL | English (Common) | `english` | Beginner-friendly |
| RD29 | German | `german` | Ä, Ö, Ü tiles |
| FRA24 | French | `french` | Accented characters (É, È, etc.) |
| OSPS50 | Polish | `polish` | Ą, Ć, Ę, Ł, Ń, Ó, Ś, Ź, Ż tiles |
| NSF25 | Norwegian | `norwegian` | Æ, Ø, Å tiles |
| FILE2017 | Spanish | `spanish` | Ñ, CH, LL, RR digraphs |
| DISC2 | Catalan | `catalan` | Ç, L·L, NY tiles |

Non-English games are important because they have different alphabets,
accented characters, and tile distributions that the OCR pipeline must
handle correctly. Find these by searching for active players in each
lexicon on Woogles.

**Category 6: 21x21 Super boards (3-5 games)**

The super crossword game uses a 21x21 board with the `english_super` letter
distribution. In liwords, this renders with the `.zomgboard` CSS class and
`--dim: 21` CSS variable. Tiles are smaller and there are more bonus squares.

Find these by filtering `GetRecentGames` for games with
`rules.board_layout_name` containing "Super" or by looking at games with
`letter_distribution_name: "english_super"`.

**Category 7: Special end conditions (3-4 games)**

- Time penalty games: `YB6rxJBVGU` (has `(time) -10` notation)
- Resigned games: `bpX9qVgNKy`
- Games with exchanges (40% of games have them)
- FIVE_POINT challenge rule: `YB6rxJBVGU`, `ytJoe9Fd9n`

#### Game fetching script

```python
import requests, json, os

API = "https://woogles.io/api/game_service.GameMetadataService"

def get_recent_games(username, num=50, offset=0):
    r = requests.post(f"{API}/GetRecentGames",
        json={"username": username, "numGames": num, "offset": offset})
    return r.json().get("game_info", [])

def get_gcg(game_id):
    r = requests.post(f"{API}/GetGCG", json={"gameId": game_id})
    return r.json().get("gcg", "")

def fetch_and_filter(username, pages=3):
    """Fetch games and categorize them."""
    all_games = []
    for page in range(pages):
        games = get_recent_games(username, 50, page * 50)
        all_games.extend(games)

    for g in all_games:
        gcg = get_gcg(g["game_id"])
        g["_gcg"] = gcg
        g["_has_phony"] = "-- -" in gcg
        g["_has_blank"] = any("?" in line for line in gcg.split("\n")
                              if line.startswith(">"))
        g["_has_exchange"] = any(line.split()[-3].startswith("-")
                                 for line in gcg.split("\n")
                                 if line.startswith(">") and " -" in line)
        g["_move_count"] = sum(1 for line in gcg.split("\n")
                               if line.startswith(">"))
    return all_games

# Fetch from known active players
for player in ["cesar", "BestBot", "magrathean", "thisguy", "josh"]:
    games = fetch_and_filter(player)
    phony_games = [g for g in games if g["_has_phony"]]
    # Save GCGs to testgen/gcg/
    for g in games:
        with open(f"testgen/gcg/{g['game_id']}.gcg", "w") as f:
            f.write(g["_gcg"])
```

### Source 2: Drive liwords locally with Playwright

liwords runs locally via `docker compose up` (Go API + React frontend +
PostgreSQL + NATS + Redis). The frontend is React 19 + react-router v7 +
antd + Zustand. Key routes:

| Route | Purpose |
|-------|---------|
| `/game/:gameID` | Active game view |
| `/anno/:gameID` | Annotated game review |
| `/editor` | Board editor — load arbitrary positions |

**The `/editor` route is the key entry point.** It lets you set up an
arbitrary board position without needing to play through a full game. We
can also replay GCG files via the annotated game viewer.

**Pipeline:**
1. Start liwords via Docker Compose
2. Use the game fetching script to download GCG files from Woogles
3. Import GCG files into local liwords via the `ImportGCG` RPC
4. Playwright navigates to `/editor/:gameID` (or `/anno/:gameID`)
5. Step through turns to reach desired board position
6. Apply variation settings (see "Variation Matrix" below)
7. Take a screenshot
8. Save the screenshot + the known CGP as a test pair

### Source 3: Match real screenshots to Woogles database

For screenshots taken from real Woogles games (the existing 6 test cases and
any future user submissions):

1. **OCR the screenshot** with the current pipeline to get an approximate
   board state, player names, and scores
2. **Query the Woogles API** to search for the game:
   ```bash
   # Fetch recent games for a player
   curl -s -X POST 'https://woogles.io/api/game_service.GameMetadataService/GetRecentGames' \
     -H 'Content-Type: application/json' \
     -d '{"username":"PlayerName","numGames":50,"offset":0}'

   # Fetch GCG for a known game ID
   curl -s -X POST 'https://woogles.io/api/game_service.GameMetadataService/GetGCG' \
     -H 'Content-Type: application/json' \
     -d '{"gameId":"GdTkgTga"}'
   ```
   - Match on: player names + approximate scores + board pattern
3. **If found**, replay the GCG to reconstruct the exact board state at the
   turn matching the screenshot — this is the golden ground truth
4. **Replay the game** to every turn position and take screenshots at each
   turn, multiplying one game into ~20 test cases

The occupancy grid alone (225 bits for 15x15, or 441 bits for 21x21) is
almost always unique to a specific turn in a specific game, so even
imperfect OCR should match correctly.

This is a stretch goal — it requires OCR to already be decent — but it
creates a virtuous cycle: better OCR → better matching → more ground truth →
even better OCR.

### Source 4: GCG game archives

GCG (Game Commentary and Grammar) files encode full Scrabble games. Format
spec: https://www.poslfit.com/scrabble/gcg/

```
#player1 Brian Brian Cappelletto
#player2 Peter Peter Morris
#title 1991 Worlds Finals Game 3
>Brian: CEGINOU 8D CUEING +24 24
>Peter: IINORRW -IINORRW +0 0
>Brian: AEEEOSY 7E YEA +17 41
>Peter: AFIIOTU 6D OAF +34 34
```

Key notation:
- Coordinates: `8D` = row 8, column D, horizontal; `D8` = column D, row 8, vertical
- Lowercase in words = blank tile (e.g., `InHALER` means blank played as N)
- `?` in rack = blank tile in hand
- `-TILES` = tile exchange
- `--` = phony withdrawal (challenged off); same player, same rack, negative score
- `(challenge) +5` = successful challenge defense (bonus under FIVE_POINT rules)
- `(TILES)` = end-of-game rack subtraction (opponent's remaining tiles)
- `(time) -10` = time penalty

Example of a phony play followed by challenge:
```
>cesar: AADMNSU I7 D.MAN +29 61
>cesar: AADMNSU -- -29 32
```

Example of endgame play-out:
```
>bnjy: ADEI K8 IDEA +14 450
>bnjy: (ETU) +6 456
```

Additional GCG sources (beyond Woogles API):
- **cross-tables.com** — ~48,600 annotated tournament games with download links
  - Browse: https://www.cross-tables.com/annolistself.php
  - Direct GCG URLs follow patterns like:
    `https://cross-tables.com/annotated/selfgcg/171/anno17123.gcg`
- **Kaggle: Raw woogles.io games** — monthly-updated dataset by Meg Risdal
  https://www.kaggle.com/datasets/mrisdal/raw-wooglesio-games
  (Note: only covers BasicBot games with VOID challenge — no phonies)
- **macondo** (liwords dependency) — 26 test GCG files in the repo
- **Quackle** — open-source Scrabble AI, natively reads/writes GCG

Each GCG file can be replayed to generate a screenshot + CGP at every turn
position, so a single game yields ~20 test cases across different board
densities (early game with few tiles through endgame with many).

## Variation Matrix

The key insight: **the same board position should be photographed under many
different conditions** to test the OCR pipeline's robustness. For each board
state, cycle through combinations of:

### Theme / Color Mode
- **Dark mode** (`.mode--dark` class on body; SCSS `$modes: dark`)
- **Light mode** (default `.mode--default`; SCSS `$modes: default`)

Zustand store (`useUIStore`) has `themeMode: "light" | "dark"` with
`toggleTheme()`. Persisted to localStorage key `darkMode`.

Set via Playwright:
```js
await page.evaluate(() => localStorage.setItem('darkMode', 'true'));
await page.reload();
// Or directly: document.body.classList.replace('mode--default', 'mode--dark');
```

### Tile Style
- liwords has `tile_modes.scss` — different tile visual styles
- localStorage key: `userTile`; body class: `tile--*`
- Toggle between available tile themes

### Board Style
- liwords has `board_modes.scss` — different board visual styles
- localStorage key: `userBoard`; body class: `board--*`
- Special: `bnjyMode` localStorage key for BNJY tile mode
- Toggle between available board themes

### Viewport Size (device simulation)

| Device | Viewport | DPR |
|--------|----------|-----|
| iPhone SE | 375x667 | 2 |
| iPhone 14 Pro | 393x852 | 3 |
| iPhone 14 Pro Max | 430x932 | 3 |
| iPad Mini | 768x1024 | 2 |
| iPad Pro 12.9" | 1024x1366 | 2 |
| Android phone (small) | 360x640 | 2 |
| Android phone (large) | 412x915 | 3.5 |
| Desktop 1080p | 1920x1080 | 1 |
| Desktop 1440p | 2560x1440 | 1 |
| Desktop 4K | 3840x2160 | 1 |
| MacBook Air 13" | 1470x956 | 2 |
| MacBook Pro 16" | 1728x1117 | 2 |

Set via Playwright: `page.setViewportSize({ width, height })` and
`browserContext({ deviceScaleFactor })`.

### Crop Margins

Real user screenshots vary in how tightly they're cropped. Simulate this:

| Crop | Description |
|------|-------------|
| Full page | Include all browser chrome, sidebars, chat |
| Board + rack + scores | Tight crop around the game area |
| Board only | Just the 15x15 grid |
| Board + generous margin | Board with 50-100px padding (typical phone screenshot) |
| Off-center crop | Board not centered, partial sidebar visible |
| Slightly rotated | 1-3 degree rotation (simulating phone camera angle) |

Implement by:
- Playwright's `page.screenshot({ clip: { x, y, width, height } })`
- For rotation: apply CSS `transform: rotate(Xdeg)` to the body before capture
- For margins: compute board element bounds and add/subtract pixel offsets

### Browser Zoom / Font Scale
- 90%, 100%, 110%, 125% zoom levels
- Set via: `page.evaluate(() => document.body.style.zoom = '1.1')`

### Board Dimensions
- **Standard 15x15**: Default CrosswordGame layout (CSS `--dim: 15`)
- **Super 21x21**: Super crossword game (CSS `--dim: 21`, `.zomgboard` class)

The 21x21 board has smaller tiles at every viewport size, more bonus squares,
and uses the `english_super` letter distribution (more tiles in the bag).

### Language / Tile Set
- **English** (NWL23 / CSW24): Standard A-Z tiles
- **German** (RD29): Includes Ä, Ö, Ü tiles
- **French** (FRA24): Includes É, È, Ê, Ë, etc.
- **Polish** (OSPS50): Includes Ą, Ć, Ę, Ł, Ń, Ó, Ś, Ź, Ż
- **Norwegian** (NSF25): Includes Æ, Ø, Å
- **Spanish** (FILE2017): Includes Ñ, CH, LL, RR
- **Catalan** (DISC2): Includes Ç, L·L, NY

Different languages affect:
- The characters on tiles (accented/special characters)
- Tile point values and distributions
- The number of tiles in the bag
- Lexicon badge display

### Game State Density
- **Early game**: 1-3 words, mostly empty board (5-15 tiles)
- **Mid game**: moderate coverage (30-60 tiles)
- **Late game / endgame**: dense board (60-80+ tiles), bag empty
- **Post-phony**: board state after a word was challenged off
- **With blanks on board**: at least one blank tile (displayed without points)
- **With blanks on rack**: blank tile visible in rack
- **Recently played tiles highlighted**: last move shown in orange/gold

### Score / UI Variations
- Low scores vs high scores (affects digit count in score display)
- Different lexicon badges (NWL23, CSW24, ECWL, RD29, FRA24, etc.)
- Tile bag visible vs hidden
- Timer visible vs hidden
- Challenge rule indicator (VOID, DOUBLE, FIVE_POINT)

## Combinatorial Explosion

Rough count:
- ~1000 board positions (from ~50 Woogles games, each replayed at ~20 turns)
- 2 color modes (light, dark)
- 5 board themes (Cheery, Forest, Aflame, Vintage, Mahogany)
- 4 tile themes (Charcoal, Whitish, Balsa, Tealish)
- 2 board sizes (15x15, 21x21)
- 9 languages
- 12 viewport sizes
- 6 crop styles
- 4 zoom levels

The full cross-product is astronomically large. Use stratified sampling:

1. **Canonical set** (~500 cases): Core positions covering all game
   categories (phonies, endgames, blanks, each language, 21x21) x
   {dark, light} x {mobile, desktop}. Full pipeline eval runs against this.
2. **Stress set** (~1000 cases): Random sample from the full matrix, biased
   toward edge cases (tiny viewports, extreme zoom, tight crops, rotation,
   non-English languages, post-phony board states).
3. **Regression set** (~50-100 cases): Hand-picked cases that previously
   failed. Add new cases when bugs are found.

Priority for initial generation:
1. English games with phonies and endgames (highest immediate value)
2. Each non-English language with at least 5 positions
3. 21x21 super board with at least 10 positions
4. Variation matrix (themes, viewports, crops) applied to a subset

## Playwright Script Outline

```typescript
import { chromium, Browser, Page } from 'playwright';
import * as fs from 'fs';
import * as path from 'path';

interface TestConfig {
  gcgFile: string;      // path to GCG game file
  turnNumber: number;   // which turn to screenshot
  viewport: { width: number; height: number };
  deviceScaleFactor: number;
  darkMode: boolean;
  crop: 'full' | 'board-rack' | 'board-only' | 'margin-50' | 'off-center';
  zoom: number;         // 0.9, 1.0, 1.1, 1.25
}

async function generateTestCase(browser: Browser, config: TestConfig) {
  const context = await browser.newContext({
    viewport: config.viewport,
    deviceScaleFactor: config.deviceScaleFactor,
  });
  const page = await context.newPage();

  // Navigate to liwords and load position
  await page.goto('http://liwords.localhost/editor');
  // ... load GCG, advance to turn, apply settings ...

  // Toggle dark mode
  if (config.darkMode) {
    await page.evaluate(() => {
      document.body.classList.add('mode--dark');
    });
  }

  // Set zoom
  if (config.zoom !== 1.0) {
    await page.evaluate((z) => {
      (document.body.style as any).zoom = String(z);
    }, config.zoom);
  }

  // Wait for board to render
  await page.waitForSelector('.board-container', { state: 'visible' });

  // Take screenshot with appropriate crop
  const clip = await computeClip(page, config.crop);
  const screenshot = await page.screenshot({
    type: 'png',
    clip: clip || undefined,
  });

  // Extract ground truth CGP from the loaded game state
  const cgp = await page.evaluate(() => {
    // Access the game state from the app's store
    // (implementation depends on how liwords exposes this)
    return (window as any).__GAME_CGP__;
  });

  // Save test pair
  const name = buildTestName(config);
  fs.writeFileSync(`testdata/${name}.png`, screenshot);
  fs.writeFileSync(`testdata/${name}.cgp`, cgp);

  await context.close();
}
```

## Ground Truth Extraction

The CGP ground truth comes from the **game state in liwords**, not from OCR.
Since we control the input (GCG file + turn number), we know the exact board
position. The CGP can be derived from:

1. **liwords editor/game state** — extract from the React app's Zustand store
   via `page.evaluate()`
2. **GCG replay** — parse the GCG file offline and compute the board state at
   each turn (this is deterministic and doesn't require the browser at all)
3. **macondo library** — the Go game engine can replay GCG and output board
   state

Option 2 is the most reliable — compute ground truth offline, use Playwright
only for screenshot generation. This way the CGP is never contaminated by UI
bugs.

## Matching Real Screenshots to Woogles Games

For user-submitted screenshots (not generated by us):

```
Screenshot → OCR pipeline → approximate {board, scores, players}
                                  ↓
                          Woogles API query
                                  ↓
                    Exact game state (golden CGP)
```

Matching algorithm:
1. **OCR player names** from the screenshot
2. **Query `GetRecentGames`** for each player name
3. For each candidate game, **fetch its GCG** via `GetGCG`
4. **Replay the GCG** turn by turn, computing board state + scores at each turn
5. **Match**: find the turn where the board occupancy pattern and scores
   best match the OCR'd screenshot (occupancy grid + score pair)
6. The matched turn's exact board state is the golden CGP

The occupancy grid alone (225 bits) is almost always unique to a specific
turn in a specific game, so even imperfect OCR should match correctly.

This is a stretch goal — it requires OCR to already be decent — but it
creates a virtuous cycle: better OCR → better matching → more ground truth →
even better OCR.

## Implementation Steps

### Phase 1: Scrape Woogles games
1. Write the game fetching script (Python) to download GCG files from Woogles
2. Fetch games from known active players: cesar, BestBot, magrathean,
   thisguy, josh, tarkovsky7, squush, woogie
3. Filter and categorize: phonies, endgames, blanks, different lexicons
4. Find non-English games by searching for players in each lexicon
5. Find 21x21 super games
6. Save curated GCG files to `testgen/gcg/` with metadata JSON sidecar
7. Target: ~50 games across all categories

### Phase 2: Submodule + Docker setup
1. Add liwords as a git submodule
2. Write a `testgen/docker-compose.override.yml` if needed
3. Verify `docker compose up` works and `/editor` is accessible
4. Document the setup in README

### Phase 3: GCG parser + ground truth
1. Write a GCG parser (TypeScript or Go) that computes board state at each
   turn, correctly handling:
   - Phony withdrawals (tiles placed then removed)
   - Blank tiles (lowercase = blank designating that letter)
   - Tile exchanges
   - End-of-game rack subtraction
2. Output CGP for each turn position — this is the offline ground truth
3. Test parser against known game states

### Phase 4: Basic Playwright generation
1. Install Playwright in `testgen/`
2. Import a GCG file into local liwords via the `ImportGCG` RPC
3. Navigate to the editor, step through turns
4. Take one screenshot per turn in default settings
5. Pair with offline-computed CGP ground truth
6. Verify test pairs work with `cgptest --test`

### Phase 5: Variation matrix
1. Add theme cycling (dark/light, board themes, tile themes)
2. Add viewport/device presets
3. Add crop logic (full page, board-only, margin, off-center)
4. Add zoom variations
5. Add non-English language game rendering
6. Add 21x21 board rendering
7. Sample from the full matrix to build canonical + stress + regression sets

### Phase 6: Woogles game matching (stretch)
1. Build a matching script that takes OCR output + player names and queries
   Woogles for the source game
2. Extract golden ground truth for matched games
3. Add matched games to the regression set

## Integration with Existing Test Runner

The generated files land in `testdata/` as `.png` + `.cgp` pairs, which is
exactly what `run_tests_cli()` in `testapp.cpp` expects (lines 365-464).
No changes needed to the test runner.

Consider adding metadata to each test case (a `.json` sidecar) recording:
- Source game ID (Woogles game ID for traceability)
- Turn number within the game
- Source (generated vs real screenshot)
- Device profile
- Color mode, board theme, tile theme
- Board dimensions (15x15 or 21x21)
- Language / lexicon
- Board density (tile count)
- Whether blanks are present on board
- Whether this turn follows a phony withdrawal
- Whether this is an endgame position (bag empty)
- Challenge rule

This enables **sliced metrics**: "What's our accuracy on mobile dark mode
screenshots?" / "How do we perform on German games?" / "Do we handle
post-phony board states correctly?" / "Is 21x21 accuracy comparable to
15x15?"
