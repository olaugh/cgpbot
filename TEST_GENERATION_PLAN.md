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

### Source 1: Drive liwords locally with Playwright (synthetic screenshots)

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
2. Playwright navigates to `/editor` (or `/anno/:gameID`)
3. Load a board position (paste CGP, import GCG, or use the API)
4. Apply variation settings (see "Variation Matrix" below)
5. Take a fullscreen screenshot
6. Save the screenshot + the known CGP as a test pair

### Source 2: Match real screenshots to Woogles database

For screenshots taken from real Woogles games (the existing 6 test cases and
any future user submissions):

1. **OCR the screenshot** with the current pipeline to get an approximate
   board state, player names, and scores
2. **Query the Woogles Twirp RPC API** to search for the game:
   ```bash
   # Fetch recent games for a player
   curl -X POST https://woogles.io/twirp/game_service.GameMetadataService/GetRecentGames \
     -H 'Content-Type: application/json' \
     -d '{"username":"PlayerName","numGames":20,"offset":0}'

   # Fetch GCG for a known game ID
   curl -X POST https://woogles.io/twirp/game_service.GameMetadataService/GetGCG \
     -H 'Content-Type: application/json' \
     -d '{"gameId":"GdTkgTga"}'

   # Fetch full game document
   curl -X POST https://woogles.io/twirp/game_service.GameMetadataService/GetGameDocument \
     -H 'Content-Type: application/json' \
     -d '{"gameId":"GdTkgTga"}'
   ```
   - Match on: player names + approximate scores + board pattern
3. **If found**, replay the GCG to reconstruct the exact board state at the
   turn matching the screenshot — this is the golden ground truth
4. **Replay the game** to every turn position and take screenshots at each
   turn, multiplying one game into ~20 test cases

Game IDs on Woogles are short alphanumeric strings (e.g., `GdTkgTga`).
Game URLs follow the pattern `https://woogles.io/game/<GameID>`.

### Source 3: GCG game archives

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
- `--` = phoney withdrawal (challenged off)

Sources of GCG files:
- **cross-tables.com** — ~48,600 annotated tournament games with download links
  - Browse: https://www.cross-tables.com/annolistself.php
  - Direct GCG URLs follow patterns like:
    `https://cross-tables.com/annotated/selfgcg/171/anno17123.gcg`
- **Woogles.io API** — fetch any completed game as GCG:
  `POST /twirp/game_service.GameMetadataService/GetGCG {"gameId":"..."}`
- **Kaggle: Raw woogles.io games** — monthly-updated dataset by Meg Risdal
  https://www.kaggle.com/datasets/mrisdal/raw-wooglesio-games
- **macondo** (liwords dependency) — test GCG files in the repo
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

### Game State Density
- **Early game**: 1-3 words, mostly empty board (5-15 tiles)
- **Mid game**: moderate coverage (30-60 tiles)
- **Late game**: dense board (60-80+ tiles)
- **With blanks on board**: at least one lowercase-letter tile
- **With blanks on rack**: blank tile visible in rack
- **Recently played tiles highlighted**: last move shown in orange/gold

### Score / UI Variations
- Low scores vs high scores (affects digit count in score display)
- Different lexicon badges (NWL23, CSW21, ECWL, etc.)
- Tile bag visible vs hidden
- Timer visible vs hidden

## Combinatorial Explosion

Rough count:
- 50 board positions (from ~3 games, each replayed at ~17 turns)
- 2 color modes
- 2 tile styles
- 12 viewport sizes
- 4 crop styles
- 3 zoom levels

**50 x 2 x 2 x 12 x 4 x 3 = 28,800 test cases**

That's way too many. Use stratified sampling:

1. **Canonical set** (~200 cases): Every board position x {dark, light} x
   {mobile, desktop} = 50 x 2 x 2. Full pipeline eval runs against this.
2. **Stress set** (~500 cases): Random sample from the full matrix, biased
   toward edge cases (tiny viewports, extreme zoom, tight crops, rotation).
3. **Regression set** (~50 cases): Hand-picked cases that previously failed.
   Add new cases when bugs are found.

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

Concrete API calls:
```bash
# Step 1: Get recent games for player "exampleuser"
curl -s -X POST https://woogles.io/twirp/game_service.GameMetadataService/GetRecentGames \
  -H 'Content-Type: application/json' \
  -d '{"username":"exampleuser","numGames":50,"offset":0}'

# Step 2: For each game ID, fetch the GCG
curl -s -X POST https://woogles.io/twirp/game_service.GameMetadataService/GetGCG \
  -H 'Content-Type: application/json' \
  -d '{"gameId":"GdTkgTga"}'
```

This is a stretch goal — it requires OCR to already be decent — but it
creates a virtuous cycle: better OCR → better matching → more ground truth →
even better OCR.

## Implementation Steps

### Phase 1: Submodule + Docker setup
1. Add liwords as a git submodule
2. Write a `testgen/docker-compose.override.yml` if needed
3. Verify `docker compose up` works and `/editor` is accessible
4. Document the setup in README

### Phase 2: Basic Playwright generation
1. Install Playwright in `testgen/`
2. Write script to load a hardcoded position in the editor
3. Take one screenshot in dark mode, one in light mode
4. Extract CGP from the known position
5. Verify the test pair works with `cgptest --test`

### Phase 3: GCG replay
1. Write a GCG parser (TypeScript or Go) that computes board state at each turn
2. Feed positions into the editor via Playwright
3. Generate test pairs for every turn of a game

### Phase 4: Variation matrix
1. Add viewport/device presets
2. Add crop logic
3. Add zoom variations
4. Sample from the full matrix to build canonical + stress sets

### Phase 5: Woogles game matching (stretch)
1. Research the Woogles Connect RPC API for game search
2. Build a matching script that takes a CGP + player names and queries Woogles
3. Extract golden ground truth for matched games

## Integration with Existing Test Runner

The generated files land in `testdata/` as `.png` + `.cgp` pairs, which is
exactly what `run_tests_cli()` in `testapp.cpp` expects (lines 365-464).
No changes needed to the test runner.

Consider adding metadata to each test case (a `.json` sidecar) recording:
- Source (generated vs real screenshot)
- Device profile
- Color mode
- Board density (tile count)
- Whether blanks are present

This enables **sliced metrics**: "What's our accuracy on mobile dark mode
screenshots with >60 tiles on the board?"
