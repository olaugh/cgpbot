#!/usr/bin/env tsx
/**
 * gen_augment.ts — Generate augmented training data for three gap areas:
 *
 *   1. Exchange banners: desktop screenshots at turns after exchanges,
 *      showing the "{nickname} exchanged {tiles}" notification banner
 *      that partially occludes row 1.
 *
 *   2. Memento blanks: memento (server-rendered) screenshots of turns
 *      where blank tiles are visible on the board.
 *
 *   3. Near-tile tooltips: desktop screenshots with tooltips hovering
 *      over empty premium squares adjacent to occupied cells.
 *
 * Usage:
 *   cd testgen && npx tsx gen_augment.ts [--dry-run] [--phase=exchange|memento-blanks|near-tooltip]
 *   --max-games N   Limit number of GCG files scanned (default: 200)
 */

import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { chromium } from "playwright";
import sharp from "sharp";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const ROOT = path.resolve(__dirname, "..");
const TESTDATA = path.join(ROOT, "testdata");
const GCG_DIR = path.join(__dirname, "gcg");
const ANALYZER = path.join(__dirname, "scripts", "gcg_analyze.py");

const DELAY_BETWEEN_PAGES_MS = 1500;
const MEMENTO_DELAY_MS = 300;
const MAX_RETRIES = 3;
const RETRY_BACKOFF = [5000, 15000, 45000];

type Theme = "light" | "dark" | "mahogany";
const THEMES: Theme[] = ["light", "dark", "mahogany"];

// ---------------------------------------------------------------------------
// Seeded PRNG
// ---------------------------------------------------------------------------

let _seed = 98765;
function seededRandom(): number {
  _seed = (_seed * 1664525 + 1013904223) & 0x7fffffff;
  return _seed / 0x7fffffff;
}

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(seededRandom() * arr.length)];
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(seededRandom() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ---------------------------------------------------------------------------
// Premium squares and CGP board parsing (from gen_tooltip.ts)
// ---------------------------------------------------------------------------

type PremiumType = "DLS" | "DWS" | "TLS" | "TWS";

interface PremiumSquare {
  row: number;
  col: number;
  type: PremiumType;
}

const PREMIUM_SQUARES: PremiumSquare[] = [
  ...[
    [0, 0], [0, 7], [0, 14], [7, 0], [7, 14], [14, 0], [14, 7], [14, 14],
  ].map(([r, c]) => ({ row: r, col: c, type: "TWS" as PremiumType })),
  ...[
    [1, 1], [1, 13], [2, 2], [2, 12], [3, 3], [3, 11], [4, 4], [4, 10],
    [7, 7], [10, 4], [10, 10], [11, 3], [11, 11], [12, 2], [12, 12],
    [13, 1], [13, 13],
  ].map(([r, c]) => ({ row: r, col: c, type: "DWS" as PremiumType })),
  ...[
    [1, 5], [1, 9], [5, 1], [5, 5], [5, 9], [5, 13], [9, 1], [9, 5],
    [9, 9], [9, 13], [13, 5], [13, 9],
  ].map(([r, c]) => ({ row: r, col: c, type: "TLS" as PremiumType })),
  ...[
    [0, 3], [0, 11], [2, 6], [2, 8], [3, 0], [3, 7], [3, 14], [6, 2],
    [6, 6], [6, 8], [6, 12], [7, 3], [7, 11], [8, 2], [8, 6], [8, 8],
    [8, 12], [11, 0], [11, 7], [11, 14], [12, 6], [12, 8], [14, 3],
    [14, 11],
  ].map(([r, c]) => ({ row: r, col: c, type: "DLS" as PremiumType })),
];

function parseCgpBoard(cgp: string): boolean[][] {
  const boardStr = cgp.split(" ")[0];
  const rows = boardStr.split("/");
  const grid: boolean[][] = [];
  for (const row of rows) {
    const cells: boolean[] = [];
    for (let i = 0; i < row.length; i++) {
      const ch = row[i];
      if (ch >= "0" && ch <= "9") {
        let numStr = ch;
        while (i + 1 < row.length && row[i + 1] >= "0" && row[i + 1] <= "9") {
          numStr += row[++i];
        }
        for (let j = 0; j < parseInt(numStr); j++) cells.push(false);
      } else {
        cells.push(true);
      }
    }
    grid.push(cells);
  }
  return grid;
}

function findNearTilePremiumSquares(cgp: string): PremiumSquare[] {
  const grid = parseCgpBoard(cgp);
  return PREMIUM_SQUARES.filter((sq) => {
    if (grid[sq.row]?.[sq.col]) return false; // occupied
    // Check all 8 neighbors for occupied cells
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (dr === 0 && dc === 0) continue;
        const nr = sq.row + dr, nc = sq.col + dc;
        if (nr >= 0 && nr < 15 && nc >= 0 && nc < 15 && grid[nr]?.[nc]) {
          return true;
        }
      }
    }
    return false;
  });
}

// ---------------------------------------------------------------------------
// GCG analysis
// ---------------------------------------------------------------------------

interface GcgAnalysis {
  players: string[];
  nicknames: string[];
  exchanges: { nickname: string; tiles: string; urlTurn: number; cgp: string }[];
  blankTurns: { urlTurn: number; cgp: string }[];
  totalUrlTurns: number;
}

function analyzeGcg(gameId: string): GcgAnalysis | null {
  const gcgPath = path.join(GCG_DIR, `${gameId}.gcg.gz`);
  if (!fs.existsSync(gcgPath)) return null;

  try {
    const tmpFile = `/tmp/cgpbot_gcg_${gameId}.gcg`;
    execSync(`gzip -d -c "${gcgPath}" > "${tmpFile}"`, { stdio: "pipe" });
    const result = execSync(
      `python3 "${ANALYZER}" "${tmpFile}"`,
      { cwd: path.join(__dirname, "scripts"), stdio: "pipe", maxBuffer: 10 * 1024 * 1024 }
    );
    fs.unlinkSync(tmpFile);
    return JSON.parse(result.toString());
  } catch (err) {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Playwright helpers
// ---------------------------------------------------------------------------

async function setupThemeContext(browser: any, theme: Theme) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  });

  const darkMode = theme === "dark" ? "true" : "false";
  const boardStyle = theme === "mahogany" ? "mahogany" : "";
  await context.addInitScript(
    ({ dm, bs }: { dm: string; bs: string }) => {
      localStorage.setItem("darkMode", dm);
      if (bs) {
        localStorage.setItem("userBoard", bs);
        localStorage.setItem("userTile", bs);
      } else {
        localStorage.removeItem("userBoard");
        localStorage.removeItem("userTile");
      }
    },
    { dm: darkMode, bs: boardStyle }
  );

  return context;
}

async function navigateAndWait(page: any, gameId: string, turn: number) {
  const url = `https://woogles.io/game/${gameId}?turn=${turn}`;
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });

  // Dismiss modal
  try {
    const closeBtn = page.locator(".ant-modal-close");
    if (await closeBtn.isVisible({ timeout: 1000 })) {
      await closeBtn.click();
      await page.waitForTimeout(500);
    }
  } catch {}

  await page.waitForSelector(".board-spaces-container", { timeout: 20000 });
  await page.waitForSelector(".board-spaces-container .tile", { timeout: 15000 });
  await page.waitForTimeout(1500);
}

async function cropAndSave(page: any, filename: string, extraTopPadding = 0) {
  const layout = (await page.evaluate(`(() => {
    var sx = window.scrollX, sy = window.scrollY;
    var abs = function(sel) {
      var el = document.querySelector(sel);
      if (!el) return null;
      var r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) return null;
      return { x: r.x + sx, y: r.y + sy, width: r.width, height: r.height };
    };
    return {
      dpr: window.devicePixelRatio || 1,
      board: abs('.board-spaces-container'),
      rack: abs('.rack'),
      cards: abs('.player-cards'),
    };
  })()`)) as any;

  const screenshotBuf = await page.screenshot({ type: "png", fullPage: true });
  const imgMeta = await sharp(screenshotBuf).metadata();
  const imgW = imgMeta.width!;
  const imgH = imgMeta.height!;

  const dpr = layout.dpr;
  const scale = (b: any) =>
    b ? { x: b.x * dpr, y: b.y * dpr, width: b.width * dpr, height: b.height * dpr } : null;
  const board = scale(layout.board);
  const rack = scale(layout.rack);
  const cards = scale(layout.cards);

  if (!board) {
    fs.writeFileSync(path.join(TESTDATA, `${filename}.png`), screenshotBuf);
    return;
  }

  let rx = board.x, ry = board.y;
  let rr = board.x + board.width, rb = board.y + board.height;
  const expand = (b: any) => { if (!b) return; rx = Math.min(rx, b.x); ry = Math.min(ry, b.y); rr = Math.max(rr, b.x + b.width); rb = Math.max(rb, b.y + b.height); };
  expand(rack);
  expand(cards);

  const pad = 40;
  rx = Math.max(0, rx - pad);
  ry = Math.max(0, ry - pad - extraTopPadding);
  rr = Math.min(imgW, rr + pad);
  rb = Math.min(imgH, rb + pad);

  const crop = {
    left: Math.round(rx),
    top: Math.round(ry),
    width: Math.round(rr - rx),
    height: Math.round(rb - ry),
  };

  const cropped = await sharp(screenshotBuf).extract(crop).png().toBuffer();
  fs.writeFileSync(path.join(TESTDATA, `${filename}.png`), cropped);
  console.log(`  Saved ${filename}.png (${crop.width}x${crop.height})`);
}

// ---------------------------------------------------------------------------
// Phase 1: Exchange Banner Screenshots
// ---------------------------------------------------------------------------

interface ExchangeCase {
  gameId: string;
  turn: number;
  theme: Theme;
  cgp: string;
  nickname: string;
  tiles: string;
  filename: string;
}

async function runExchangePhase(dryRun: boolean, maxGames: number) {
  console.log("\n=== Phase 1: Exchange Banner Screenshots ===\n");

  const gcgFiles = fs.readdirSync(GCG_DIR).filter((f) => f.endsWith(".gcg.gz"));
  const shuffled = shuffle(gcgFiles).slice(0, maxGames);

  const cases: ExchangeCase[] = [];
  let scanned = 0;

  for (const f of shuffled) {
    const gameId = f.replace(".gcg.gz", "");
    const analysis = analyzeGcg(gameId);
    scanned++;
    if (scanned % 50 === 0) console.log(`  Scanned ${scanned}/${shuffled.length} GCGs...`);
    if (!analysis || analysis.exchanges.length === 0) continue;

    // Pick up to 2 exchanges per game
    const exch = analysis.exchanges.slice(0, 2);
    for (const ex of exch) {
      for (const theme of THEMES) {
        const turnStr = String(ex.urlTurn).padStart(2, "0");
        const filename = `${gameId}_t${turnStr}_${theme}_desktop_exchange`;
        cases.push({
          gameId,
          turn: ex.urlTurn,
          theme,
          cgp: ex.cgp,
          nickname: ex.nickname,
          tiles: ex.tiles,
          filename,
        });
      }
    }

    if (cases.length >= 150) break; // enough exchange cases
  }

  console.log(`Found ${cases.length} exchange banner cases from ${scanned} GCGs`);

  if (dryRun) {
    for (const tc of cases.slice(0, 20)) {
      console.log(`  ${tc.filename}  "${tc.nickname} exchanged ${tc.tiles}"`);
    }
    if (cases.length > 20) console.log(`  ... and ${cases.length - 20} more`);
    return;
  }

  const browser = await chromium.launch({ headless: true });
  let success = 0, failed = 0;

  for (const tc of cases) {
    const context = await setupThemeContext(browser, tc.theme);
    const page = await context.newPage();
    const idx = success + failed + 1;
    console.log(`\n[${idx}/${cases.length}] ${tc.filename}`);
    console.log(`  ${tc.nickname} exchanged ${tc.tiles} (turn ${tc.turn})`);

    try {
      await navigateAndWait(page, tc.gameId, tc.turn);

      // Inject exchange banner via Ant Design's message API
      const bannerText = `${tc.nickname} exchanged ${tc.tiles}`;
      await page.evaluate((text: string) => {
        // Use Ant Design's message API if available
        const antMsg = (window as any).antd?.message || (window as any).message;
        if (antMsg?.info) {
          antMsg.info({
            content: text,
            className: "board-hud-message",
            key: "board-messages",
            duration: 30, // long duration to survive screenshot
          });
        } else {
          // Fallback: inject DOM directly matching Ant Design v5 style
          const wrapper = document.createElement("div");
          wrapper.style.cssText = "position:fixed;top:8px;left:0;right:0;z-index:1010;text-align:center;pointer-events:none;";
          const notice = document.createElement("div");
          notice.style.cssText = "display:inline-block;padding:9px 12px;background:#e6f4ff;border-radius:8px;box-shadow:0 6px 16px rgba(0,0,0,0.08);font-size:14px;color:rgba(0,0,0,0.88);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;";
          notice.innerHTML = '<span style="color:#1677ff;margin-right:8px;font-size:16px;">&#9432;</span>' + text;
          wrapper.appendChild(notice);
          document.body.appendChild(wrapper);
        }
      }, bannerText);
      await page.waitForTimeout(500);

      await cropAndSave(page, tc.filename, 40); // extra top padding for banner

      fs.writeFileSync(path.join(TESTDATA, `${tc.filename}.cgp`), tc.cgp + "\n");
      success++;
    } catch (err) {
      console.error(`  FAILED: ${err}`);
      failed++;
    }

    await context.close();
    await new Promise((r) => setTimeout(r, DELAY_BETWEEN_PAGES_MS));
  }

  await browser.close();
  console.log(`\nExchange phase: ${success}/${cases.length} saved (${failed} failed)`);
}

// ---------------------------------------------------------------------------
// Phase 2: Memento Blank Tile Screenshots
// ---------------------------------------------------------------------------

interface MementoBlanksCase {
  gameId: string;
  turn: number;
  cgp: string;
  filename: string;
}

async function runMementoBlanksPhase(dryRun: boolean, maxGames: number) {
  console.log("\n=== Phase 2: Memento Blank Tile Screenshots ===\n");

  const gcgFiles = fs.readdirSync(GCG_DIR).filter((f) => f.endsWith(".gcg.gz"));
  const shuffled = shuffle(gcgFiles).slice(0, maxGames);

  const cases: MementoBlanksCase[] = [];
  let scanned = 0;

  for (const f of shuffled) {
    const gameId = f.replace(".gcg.gz", "");
    const analysis = analyzeGcg(gameId);
    scanned++;
    if (scanned % 50 === 0) console.log(`  Scanned ${scanned}/${shuffled.length} GCGs...`);
    if (!analysis || analysis.blankTurns.length === 0) continue;

    // Pick 2-3 turns with blanks per game
    const picks = shuffle(analysis.blankTurns).slice(0, 3);
    for (const bt of picks) {
      const turnStr = String(bt.urlTurn).padStart(2, "0");
      const filename = `${gameId}_t${turnStr}_memento`;
      // Skip if already exists in testdata
      if (fs.existsSync(path.join(TESTDATA, `${filename}.png`))) continue;

      cases.push({
        gameId,
        turn: bt.urlTurn,
        cgp: bt.cgp,
        filename,
      });
    }

    if (cases.length >= 100) break;
  }

  console.log(`Found ${cases.length} memento blank cases from ${scanned} GCGs`);

  if (dryRun) {
    for (const tc of cases.slice(0, 20)) {
      console.log(`  ${tc.filename}`);
    }
    if (cases.length > 20) console.log(`  ... and ${cases.length - 20} more`);
    return;
  }

  let success = 0, failed = 0;

  for (const tc of cases) {
    const idx = success + failed + 1;
    console.log(`[${idx}/${cases.length}] ${tc.filename}`);

    const url = `https://woogles.io/gameimg/${tc.gameId}-v2-${tc.turn}.png`;
    let buf: Buffer | null = null;

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        buf = Buffer.from(await resp.arrayBuffer());
        break;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.warn(`  Attempt ${attempt + 1}/${MAX_RETRIES} failed: ${msg}`);
        if (attempt < MAX_RETRIES - 1) {
          await new Promise((r) => setTimeout(r, RETRY_BACKOFF[attempt]));
        }
      }
    }

    if (buf) {
      fs.writeFileSync(path.join(TESTDATA, `${tc.filename}.png`), buf);
      fs.writeFileSync(path.join(TESTDATA, `${tc.filename}.cgp`), tc.cgp + "\n");
      console.log(`  Saved ${tc.filename}.png (${buf.length} bytes)`);
      success++;
    } else {
      console.error(`  SKIPPING ${tc.filename}`);
      failed++;
    }

    await new Promise((r) => setTimeout(r, MEMENTO_DELAY_MS));
  }

  console.log(`\nMemento blanks phase: ${success}/${cases.length} saved (${failed} failed)`);
}

// ---------------------------------------------------------------------------
// Phase 3: Near-Tile Tooltip Screenshots
// ---------------------------------------------------------------------------

interface NearTooltipCase {
  gameId: string;
  turn: number;
  theme: Theme;
  cgp: string;
  tooltipSquare: PremiumSquare;
  filename: string;
}

function discoverGameTurns(): { gameId: string; turn: number; cgp: string }[] {
  const files = fs.readdirSync(TESTDATA).filter((f) => f.endsWith(".cgp"));
  const seen = new Map<string, string>();
  for (const f of files) {
    const m = f.match(/^([A-Za-z0-9]{10})_t(\d+)_/);
    if (!m) continue;
    const key = `${m[1]}_t${m[2]}`;
    if (seen.has(key)) continue;
    seen.set(key, fs.readFileSync(path.join(TESTDATA, f), "utf-8").trim());
  }
  const result: { gameId: string; turn: number; cgp: string }[] = [];
  for (const [key, cgp] of seen) {
    const m = key.match(/^(.+)_t(\d+)$/);
    if (m) result.push({ gameId: m[1], turn: parseInt(m[2]), cgp });
  }
  return result.sort((a, b) => a.gameId.localeCompare(b.gameId) || a.turn - b.turn);
}

async function runNearTooltipPhase(dryRun: boolean) {
  console.log("\n=== Phase 3: Near-Tile Tooltip Screenshots ===\n");

  const gameTurns = discoverGameTurns();
  const cases: NearTooltipCase[] = [];
  const skipped: string[] = [];

  for (const gt of gameTurns) {
    const nearPremiums = findNearTilePremiumSquares(gt.cgp);
    if (nearPremiums.length === 0) {
      skipped.push(`${gt.gameId}_t${gt.turn}`);
      continue;
    }

    const sq = pickRandom(nearPremiums);
    for (const theme of THEMES) {
      const turnStr = String(gt.turn).padStart(2, "0");
      const filename = `${gt.gameId}_t${turnStr}_${theme}_desktop_neartooltip`;
      // Skip if already exists
      if (fs.existsSync(path.join(TESTDATA, `${filename}.png`))) continue;

      cases.push({
        gameId: gt.gameId,
        turn: gt.turn,
        theme,
        cgp: gt.cgp,
        tooltipSquare: sq,
        filename,
      });
    }
  }

  console.log(
    `Plan: ${cases.length} near-tooltip screenshots ` +
    `(${gameTurns.length - skipped.length} game/turns, ${skipped.length} skipped)`
  );

  if (dryRun) {
    for (const tc of cases.slice(0, 20)) {
      console.log(
        `  ${tc.filename}  hover=r${tc.tooltipSquare.row}c${tc.tooltipSquare.col} (${tc.tooltipSquare.type})`
      );
    }
    if (cases.length > 20) console.log(`  ... and ${cases.length - 20} more`);
    return;
  }

  const browser = await chromium.launch({ headless: true });
  let success = 0, failed = 0;

  for (const tc of cases) {
    const context = await setupThemeContext(browser, tc.theme);
    const page = await context.newPage();
    const idx = success + failed + 1;
    console.log(`\n[${idx}/${cases.length}] ${tc.filename}`);
    console.log(`  hover r${tc.tooltipSquare.row}c${tc.tooltipSquare.col} (${tc.tooltipSquare.type})`);

    try {
      await navigateAndWait(page, tc.gameId, tc.turn);

      // Hover over the target square
      const boardBox = await page.locator(".board-spaces-container").boundingBox();
      if (!boardBox) throw new Error("Board element not found");
      const cellW = boardBox.width / 15;
      const cellH = boardBox.height / 15;
      const hoverX = boardBox.x + (tc.tooltipSquare.col + 0.5) * cellW;
      const hoverY = boardBox.y + (tc.tooltipSquare.row + 0.5) * cellH;
      await page.mouse.move(hoverX, hoverY);
      await page.waitForTimeout(800);

      await cropAndSave(page, tc.filename);
      fs.writeFileSync(path.join(TESTDATA, `${tc.filename}.cgp`), tc.cgp + "\n");
      success++;
    } catch (err) {
      console.error(`  FAILED: ${err}`);
      failed++;
    }

    await context.close();
    await new Promise((r) => setTimeout(r, DELAY_BETWEEN_PAGES_MS));
  }

  await browser.close();
  console.log(`\nNear-tooltip phase: ${success}/${cases.length} saved (${failed} failed)`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const dryRun = process.argv.includes("--dry-run");
  const phaseArg = process.argv.find((a) => a.startsWith("--phase="))?.split("=")[1];
  const maxGamesArg = process.argv.find((a) => a.startsWith("--max-games="))?.split("=")[1];
  const maxGames = maxGamesArg ? parseInt(maxGamesArg) : 200;

  fs.mkdirSync(TESTDATA, { recursive: true });

  if (!phaseArg || phaseArg === "exchange") {
    await runExchangePhase(dryRun, maxGames);
  }

  if (!phaseArg || phaseArg === "memento-blanks") {
    await runMementoBlanksPhase(dryRun, maxGames);
  }

  if (!phaseArg || phaseArg === "near-tooltip") {
    await runNearTooltipPhase(dryRun);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
