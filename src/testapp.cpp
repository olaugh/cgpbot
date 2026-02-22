#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

#include <httplib.h>

#include "board.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Minimal KWG (DAWG) word checker — reads MAGPIE .kwg binary files.
// Format: array of uint32_t (little-endian). Each node:
//   bits 0-21:  arc index (child node index)
//   bit 22:     is_end (last sibling in list)
//   bit 23:     accepts (word terminates here)
//   bits 24-31: tile (machine letter, 1=A .. 26=Z)
// Node[0] arc_index = DAWG root; Node[1] arc_index = GADDAG root.
// ---------------------------------------------------------------------------
class KWGChecker {
    std::vector<uint32_t> nodes_;

    static constexpr uint32_t ARC_MASK    = 0x3FFFFF;
    static constexpr uint32_t IS_END      = 0x400000;
    static constexpr uint32_t ACCEPTS     = 0x800000;
    static constexpr int      TILE_SHIFT  = 24;

    uint32_t arc_index(uint32_t node) const { return node & ARC_MASK; }
    bool is_end(uint32_t node) const { return (node & IS_END) != 0; }
    bool accepts(uint32_t node) const { return (node & ACCEPTS) != 0; }
    uint32_t tile(uint32_t node) const { return node >> TILE_SHIFT; }

public:
    bool loaded() const { return !nodes_.empty(); }

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return false;
        auto sz = f.tellg();
        if (sz <= 0 || sz % 4 != 0) return false;
        size_t count = static_cast<size_t>(sz) / 4;
        nodes_.resize(count);
        f.seekg(0);
        f.read(reinterpret_cast<char*>(nodes_.data()), sz);
        return f.good();
    }

    // Check if a word (uppercase A-Z string) is valid.
    bool is_valid(const std::string& word) const {
        if (word.size() < 2 || nodes_.empty()) return false;
        // DAWG root = arc_index of node[0]
        uint32_t idx = arc_index(nodes_[0]);
        if (idx == 0) return false;

        int lidx = 0;
        uint32_t node = nodes_[idx];
        int wlen = static_cast<int>(word.size());
        while (true) {
            if (lidx > wlen - 1) return false;
            uint32_t ml = static_cast<uint32_t>(word[lidx] - 'A' + 1);
            if (tile(node) == ml) {
                if (lidx == wlen - 1) return accepts(node);
                idx = arc_index(node);
                if (idx == 0) return false;
                node = nodes_[idx];
                lidx++;
            } else {
                if (is_end(node)) return false;
                idx++;
                node = nodes_[idx];
            }
        }
    }
};

// Global KWG checker, loaded once at startup or on first use.
static KWGChecker g_kwg;
static std::string g_kwg_lexicon;  // which lexicon is loaded

static bool ensure_kwg_loaded(const std::string& lexicon) {
    if (g_kwg.loaded() && g_kwg_lexicon == lexicon) return true;
    // Try magpie/data/lexica/<LEXICON>.kwg
    std::string path = "magpie/data/lexica/" + lexicon + ".kwg";
    if (!fs::exists(path)) {
        // Fallback: try testdata
        path = "magpie/testdata/lexica/" + lexicon + ".kwg";
    }
    if (!fs::exists(path)) return false;
    if (g_kwg.load(path)) {
        g_kwg_lexicon = lexicon;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Word extraction from a 15x15 board.
// ---------------------------------------------------------------------------
struct BoardWord {
    std::string word;                     // uppercase letters
    std::vector<std::pair<int,int>> cells; // (row, col) of each letter
    bool horizontal;                       // true=row, false=column
};

static std::vector<BoardWord> extract_words(const CellResult board[15][15]) {
    std::vector<BoardWord> words;
    // Horizontal
    for (int r = 0; r < 15; r++) {
        int c = 0;
        while (c < 15) {
            if (board[r][c].letter != 0) {
                BoardWord bw;
                bw.horizontal = true;
                while (c < 15 && board[r][c].letter != 0) {
                    char ch = static_cast<char>(std::toupper(
                        static_cast<unsigned char>(board[r][c].letter)));
                    bw.word += ch;
                    bw.cells.push_back({r, c});
                    c++;
                }
                if (bw.word.size() >= 2) words.push_back(std::move(bw));
            } else c++;
        }
    }
    // Vertical
    for (int c = 0; c < 15; c++) {
        int r = 0;
        while (r < 15) {
            if (board[r][c].letter != 0) {
                BoardWord bw;
                bw.horizontal = false;
                while (r < 15 && board[r][c].letter != 0) {
                    char ch = static_cast<char>(std::toupper(
                        static_cast<unsigned char>(board[r][c].letter)));
                    bw.word += ch;
                    bw.cells.push_back({r, c});
                    r++;
                }
                if (bw.word.size() >= 2) words.push_back(std::move(bw));
            } else r++;
        }
    }
    return words;
}

// ---------------------------------------------------------------------------
// Last uploaded/fetched image (for saving test cases).
// ---------------------------------------------------------------------------
static std::shared_ptr<std::vector<uint8_t>> g_last_image;
static std::mutex g_last_image_mutex;

// ---------------------------------------------------------------------------
// Minimal JSON-safe escaping (CGP has no quotes/backslashes, but be safe).
// ---------------------------------------------------------------------------
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            default:   out += c;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Base64 encoder for debug image.
// ---------------------------------------------------------------------------
static const char B64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const std::vector<uint8_t>& data) {
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    size_t i = 0;
    while (i < data.size()) {
        uint32_t a = data[i++];
        uint32_t b = (i < data.size()) ? data[i++] : 0;
        uint32_t c = (i < data.size()) ? data[i++] : 0;
        uint32_t triple = (a << 16) | (b << 8) | c;
        out += B64[(triple >> 18) & 0x3F];
        out += B64[(triple >> 12) & 0x3F];
        out += (i > data.size() + 1) ? '=' : B64[(triple >> 6) & 0x3F];
        out += (i > data.size()) ? '=' : B64[triple & 0x3F];
    }
    return out;
}

// ---------------------------------------------------------------------------
// .env file loader — reads KEY=VALUE pairs and sets as environment variables.
// ---------------------------------------------------------------------------
static void load_dotenv() {
    std::ifstream f(".env");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        // Strip surrounding quotes from value
        if (val.size() >= 2 &&
            ((val.front() == '"' && val.back() == '"') ||
             (val.front() == '\'' && val.back() == '\''))) {
            val = val.substr(1, val.size() - 2);
        }
        setenv(key.c_str(), val.c_str(), 0);  // 0 = don't overwrite existing
    }
}

// ---------------------------------------------------------------------------
// Build JSON response from DebugResult.
// ---------------------------------------------------------------------------
static std::string make_json_response(const DebugResult& dr) {
    std::string json = "{\"cgp\":\"" + json_escape(dr.cgp) + "\"";

    // Per-cell detail array for the UI (letter, confidence, subscript, blank)
    json += ",\"cells\":[";
    for (int r = 0; r < 15; r++) {
        if (r > 0) json += ",";
        json += "[";
        for (int c = 0; c < 15; c++) {
            if (c > 0) json += ",";
            const auto& cell = dr.cells[r][c];
            if (cell.letter == 0) {
                json += "null";
            } else {
                char ltr = static_cast<char>(std::toupper(
                    static_cast<unsigned char>(cell.letter)));
                json += "{\"l\":\"";
                json += (ltr >= 'A' && ltr <= 'Z') ? ltr : '?';
                json += "\",\"c\":";
                json += std::to_string(static_cast<int>(cell.confidence * 100));
                json += ",\"s\":";
                json += std::to_string(cell.subscript);
                json += ",\"b\":";
                json += cell.is_blank ? "true" : "false";
                // Top-5 candidates
                json += ",\"cands\":[";
                for (int k = 0; k < 5; k++) {
                    if (cell.cand_letters[k] == 0) break;
                    if (k > 0) json += ",";
                    json += "{\"l\":\"";
                    json += cell.cand_letters[k];
                    json += "\",\"c\":";
                    json += std::to_string(
                        static_cast<int>(cell.cand_scores[k] * 100));
                    json += "}";
                }
                json += "]";
                json += "}";
            }
        }
        json += "]";
    }
    json += "]";

    if (!dr.debug_png.empty()) {
        json += ",\"debug_image\":\"data:image/png;base64,"
              + base64_encode(dr.debug_png) + "\"";
    }
    if (!dr.log.empty()) {
        json += ",\"log\":\"" + json_escape(dr.log) + "\"";
    }
    json += "}";
    return json;
}

// ---------------------------------------------------------------------------
// Log Gemini requests and responses to /tmp/gemini_log/ for debugging.
// ---------------------------------------------------------------------------
static int g_gemini_log_counter = 0;

static void gemini_log(const std::string& label,
                        const std::string& prompt_text,
                        const std::string& response_text) {
    std::filesystem::create_directories("/tmp/gemini_log");
    int n = ++g_gemini_log_counter;
    std::string base = "/tmp/gemini_log/" + std::to_string(n) + "_" + label;

    // Write prompt (text only, no base64 images)
    {
        std::ofstream f(base + "_prompt.txt");
        f << prompt_text;
    }
    // Write response
    {
        std::ofstream f(base + "_response.txt");
        f << response_text;
    }
}

// ---------------------------------------------------------------------------
// URL safety check for /fetch-url proxy.
// ---------------------------------------------------------------------------
static bool is_safe_url(const std::string& url) {
    if (url.size() < 9 || url.substr(0, 8) != "https://") return false;
    for (char c : url) {
        if (c == '\'' || c == ';' || c == '&' || c == '|' || c == '$'
            || c == '`' || c == '(' || c == ')' || c == '{' || c == '}') {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Parse CGP board portion into 15x15 character array.
// ---------------------------------------------------------------------------
static std::array<std::array<char, 15>, 15> parse_cgp_board(const std::string& cgp) {
    std::array<std::array<char, 15>, 15> board = {};
    std::string board_str = cgp.substr(0, cgp.find(' '));

    int r = 0;
    size_t pos = 0;
    while (r < 15 && pos <= board_str.size()) {
        size_t slash = board_str.find('/', pos);
        if (slash == std::string::npos) slash = board_str.size();
        std::string row_str = board_str.substr(pos, slash - pos);

        int c = 0;
        for (size_t i = 0; i < row_str.size() && c < 15;) {
            if (row_str[i] >= '0' && row_str[i] <= '9') {
                int n = 0;
                while (i < row_str.size() && row_str[i] >= '0' && row_str[i] <= '9') {
                    n = n * 10 + (row_str[i] - '0');
                    i++;
                }
                c += n;
            } else {
                board[r][c] = row_str[i];
                c++;
                i++;
            }
        }
        r++;
        pos = slash + 1;
    }

    return board;
}

// ---------------------------------------------------------------------------
// Run all test cases from testdata/ directory (CLI mode).
// ---------------------------------------------------------------------------
static int run_tests_cli() {
    if (!fs::exists("testdata")) {
        std::cout << "No testdata/ directory found.\n";
        return 1;
    }

    int total_cases = 0, passed_cases = 0;
    int total_tiles = 0, correct_tiles = 0;
    int total_occ_expected = 0, total_occ_correct = 0, total_occ_false_pos = 0;
    int total_letter_attempts = 0, total_letter_correct = 0;

    std::vector<fs::directory_entry> entries;
    for (const auto& e : fs::directory_iterator("testdata"))
        if (e.path().extension() == ".cgp") entries.push_back(e);
    std::sort(entries.begin(), entries.end());

    for (const auto& entry : entries) {
        std::string name = entry.path().stem().string();
        std::string img_path;
        for (const char* ext : {".png", ".jpg", ".jpeg"}) {
            std::string p = "testdata/" + name + ext;
            if (fs::exists(p)) { img_path = p; break; }
        }
        if (img_path.empty()) continue;

        std::string expected_cgp;
        {
            std::ifstream ifs(entry.path());
            std::getline(ifs, expected_cgp);
        }

        std::vector<uint8_t> img_data;
        {
            std::ifstream ifs(img_path, std::ios::binary);
            img_data.assign(std::istreambuf_iterator<char>(ifs),
                            std::istreambuf_iterator<char>());
        }

        DebugResult dr = process_board_image_debug(img_data);

        auto expected = parse_cgp_board(expected_cgp);
        auto got = parse_cgp_board(dr.cgp);

        int case_total = 0, case_correct = 0;
        int occ_expected = 0, occ_correct = 0, occ_false_pos = 0;
        int letter_attempts = 0, letter_correct = 0;
        for (int r = 0; r < 15; r++) {
            for (int c = 0; c < 15; c++) {
                bool exp_occ = (expected[r][c] != 0);
                bool got_occ = (got[r][c] != 0);
                if (exp_occ) {
                    occ_expected++;
                    if (got_occ) {
                        occ_correct++;
                        letter_attempts++;
                        if (got[r][c] == expected[r][c]) letter_correct++;
                    }
                } else if (got_occ) {
                    occ_false_pos++;
                }
                if (exp_occ || got_occ) {
                    case_total++;
                    if (expected[r][c] == got[r][c]) case_correct++;
                }
            }
        }

        total_cases++;
        total_tiles += case_total;
        correct_tiles += case_correct;
        total_occ_expected += occ_expected;
        total_occ_correct += occ_correct;
        total_occ_false_pos += occ_false_pos;
        total_letter_attempts += letter_attempts;
        total_letter_correct += letter_correct;

        double occ_pct = occ_expected > 0 ? (100.0 * occ_correct / occ_expected) : 100.0;
        double let_pct = letter_attempts > 0 ? (100.0 * letter_correct / letter_attempts) : 0.0;
        std::printf("%-25s occ %3d/%3d (%.0f%%) +%d fp  letters %3d/%3d (%.0f%%)\n",
                    name.c_str(), occ_correct, occ_expected, occ_pct,
                    occ_false_pos, letter_correct, letter_attempts, let_pct);

        if (case_correct == case_total) passed_cases++;
    }

    if (total_cases == 0) {
        std::cout << "No test cases found in testdata/.\n";
        return 1;
    }

    double occ_overall = total_occ_expected > 0 ? (100.0 * total_occ_correct / total_occ_expected) : 100.0;
    double let_overall = total_letter_attempts > 0 ? (100.0 * total_letter_correct / total_letter_attempts) : 0.0;
    std::printf("\n%d test case(s), %d passed\n", total_cases, passed_cases);
    std::printf("Occupancy: %d/%d (%.1f%%) +%d false positives\n",
                total_occ_correct, total_occ_expected, occ_overall, total_occ_false_pos);
    std::printf("Letters:   %d/%d (%.1f%%)\n",
                total_letter_correct, total_letter_attempts, let_overall);

    return (passed_cases == total_cases) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Embedded single-page UI.
// ---------------------------------------------------------------------------
static const char* HTML = R"html(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CGP Bot — Test Bench</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  background:#1a1a2e;color:#e0e0e0;min-height:100vh;padding:24px;
}
h1{font-size:1.4rem;margin-bottom:20px;color:#fff;font-weight:600}
.layout{display:grid;grid-template-columns:1fr 1fr;gap:20px;max-width:1400px;margin:0 auto}
.panel{background:#16213e;border-radius:12px;padding:20px;border:1px solid #2a2a4a}
.panel h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.1em;color:#888;margin-bottom:12px}

/* drop zone */
#drop-zone{
  border:2px dashed #444;border-radius:12px;padding:60px 20px;
  text-align:center;cursor:pointer;transition:all .2s;background:#1a1a2e;
}
#drop-zone.dragover{border-color:#6c63ff;background:#1e1e3f}
#drop-zone p{color:#666}
#preview{max-width:100%;max-height:400px;border-radius:8px;display:none;margin-top:12px}
#debug-img{max-width:100%;max-height:400px;border-radius:8px;display:none;margin-top:12px}
#debug-log{
  max-height:200px;overflow-y:auto;font-size:.75rem;color:#8b8;
  font-family:'SF Mono','Fira Code',monospace;margin-top:8px;
  background:#0d1117;border-radius:8px;padding:8px;display:none;
  white-space:pre-wrap;
}

/* cgp field */
#cgp-output{
  width:100%;background:#0d1117;border:1px solid #333;border-radius:8px;
  color:#58a6ff;font-family:'SF Mono','Fira Code',monospace;font-size:.85rem;
  padding:12px;resize:vertical;min-height:60px;
}
.btn-row{display:flex;gap:8px;justify-content:flex-end;margin-bottom:8px;flex-wrap:wrap}
.btn{
  background:#333;border:1px solid #555;color:#ccc;padding:4px 12px;
  border-radius:4px;cursor:pointer;font-size:.75rem;
}
.btn:hover{background:#444}

/* board grid */
.board-wrapper{display:inline-grid;grid-template-columns:28px auto;grid-template-rows:auto auto;gap:0}
.col-labels{display:grid;grid-template-columns:repeat(15,36px);gap:1px;padding-left:0;margin-left:0}
.col-labels span{text-align:center;font-size:.7rem;color:#666;line-height:22px}
.row-labels-and-board{display:grid;grid-template-columns:28px auto;gap:0}
.row-labels{display:grid;grid-template-rows:repeat(15,36px);gap:1px}
.row-labels span{
  display:flex;align-items:center;justify-content:flex-end;
  padding-right:4px;font-size:.7rem;color:#666;
}
.board{display:grid;grid-template-columns:repeat(15,36px);grid-template-rows:repeat(15,36px);gap:1px;background:#333;border:2px solid #333;border-radius:4px}
.cell{
  display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:.9rem;position:relative;cursor:pointer;
}
.cell.tile{background:#f5deb3;color:#1a1a1a}
.cell.blank-tile{background:#c8b888;color:#555}
.cell.tw{background:#c0392b;color:rgba(255,255,255,.55)}
.cell.dw{background:#e88b8b;color:rgba(255,255,255,.55)}
.cell.tl{background:#2980b9;color:rgba(255,255,255,.55)}
.cell.dl{background:#7ec8e3;color:rgba(255,255,255,.55)}
.cell.normal{background:#1b7a3d}
.cell.center{background:#e88b8b;color:rgba(255,255,255,.55)}
.cell .lbl{font-size:.5rem;font-weight:400}
.cell .sub{position:absolute;bottom:1px;right:2px;font-size:.45rem;font-weight:400;opacity:.7}
.cell.has-tip{cursor:pointer}
.cell.selected{outline:2px solid #ffeb3b;outline-offset:-2px;z-index:1}
.cell.edited{box-shadow:inset 0 0 0 2px rgba(255,165,0,0.6)}
#tip{
  display:none;position:fixed;background:#0d1117;color:#c9d1d9;
  border:1px solid #444;border-radius:6px;padding:8px 10px;
  font-size:.75rem;font-family:'SF Mono','Fira Code',monospace;
  white-space:pre;pointer-events:none;z-index:100;max-width:320px;
  line-height:1.4;
}

/* rack */
.rack{display:flex;gap:4px;margin-top:16px;justify-content:center}
.rack-tile{
  width:40px;height:40px;background:#f5deb3;color:#1a1a1a;
  display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:1.1rem;border-radius:4px;
}
.rack-tile.blank{background:#c8b888;color:#555}

/* test results */
.test-results{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:8px}
.test-results th,.test-results td{padding:4px 8px;border:1px solid #333;text-align:left}
.test-results th{background:#1a1a2e;color:#888}
.test-results td{color:#ccc}
.test-diff{font-size:.75rem;color:#f88;margin-top:4px;font-family:'SF Mono','Fira Code',monospace}

#status{margin-top:8px;font-size:.8rem;color:#666}
</style>
</head>
<body>
<h1>CGP Bot &mdash; Test Bench</h1>
<div class="layout">
  <div>
    <div class="panel">
      <h2>Input</h2>
      <label style="display:flex;align-items:center;gap:8px;margin-bottom:12px;cursor:pointer;font-size:.85rem">
        <input type="checkbox" id="use-gemini" checked> Use Gemini Flash
      </label>
      <div id="drop-zone">
        <p>Drop a board screenshot here</p>
        <p style="font-size:.8rem;margin-top:8px">or click to select a file &mdash; also accepts image URLs (e.g. from Discord)</p>
      </div>
      <img id="preview">
      <p id="status"></p>
    </div>
    <div class="panel" style="margin-top:20px">
      <h2>Debug</h2>
      <img id="debug-img">
      <pre id="debug-log"></pre>
      <div id="crops-area" style="display:none;margin-top:12px">
        <h3 style="font-size:.85rem;color:#aaa;margin-bottom:8px">Verification Crops</h3>
        <div id="crops-container" style="display:flex;flex-wrap:wrap;gap:8px"></div>
      </div>
    </div>
  </div>
  <div>
    <div class="panel">
      <h2>CGP Output</h2>
      <div class="btn-row">
        <button class="btn" onclick="renderFromField()">Render</button>
        <button class="btn" onclick="copyCGP()">Copy</button>
        <button class="btn" onclick="saveTest()">Save Test</button>
        <button class="btn" onclick="runTests()">Run Tests</button>
      </div>
      <textarea id="cgp-output" rows="3" spellcheck="false"></textarea>
    </div>
    <div class="panel" style="margin-top:20px">
      <h2>Board</h2>
      <div id="board-area"></div>
    </div>
    <div id="test-results-panel" class="panel" style="margin-top:20px;display:none">
      <h2>Test Results</h2>
      <div id="test-results"></div>
    </div>
  </div>
</div>
<div id="tip"></div>
<input type="file" id="file-input" accept="image/*" hidden>
<script>
const PREMIUM=[
  [4,0,0,1,0,0,0,4,0,0,0,1,0,0,4],
  [0,3,0,0,0,2,0,0,0,2,0,0,0,3,0],
  [0,0,3,0,0,0,1,0,1,0,0,0,3,0,0],
  [1,0,0,3,0,0,0,1,0,0,0,3,0,0,1],
  [0,0,0,0,3,0,0,0,0,0,3,0,0,0,0],
  [0,2,0,0,0,2,0,0,0,2,0,0,0,2,0],
  [0,0,1,0,0,0,1,0,1,0,0,0,1,0,0],
  [4,0,0,1,0,0,0,5,0,0,0,1,0,0,4],
  [0,0,1,0,0,0,1,0,1,0,0,0,1,0,0],
  [0,2,0,0,0,2,0,0,0,2,0,0,0,2,0],
  [0,0,0,0,3,0,0,0,0,0,3,0,0,0,0],
  [1,0,0,3,0,0,0,1,0,0,0,3,0,0,1],
  [0,0,3,0,0,0,1,0,1,0,0,0,3,0,0],
  [0,3,0,0,0,2,0,0,0,2,0,0,0,3,0],
  [4,0,0,1,0,0,0,4,0,0,0,1,0,0,4]
];
const PCLS=['normal','dl','tl','dw','tw','center'];
const PLBL=['','DL','TL','DW','TW','\u2605'];

// Point value -> valid letters (for tooltip display)
const PTS_LETTERS={1:'AEILNORSTU',2:'DG',3:'BCMP',4:'FHVWY',5:'K',8:'JX',10:'QZ'};
// Letter -> point value
const LETTER_PTS={};
for(const[p,ls]of Object.entries(PTS_LETTERS))for(const c of ls)LETTER_PTS[c]=+p;

// Per-cell detail from the last analysis (15x15, null for empty)
let cellData=null;

// Editable board state
let boardLetters=Array.from({length:15},()=>Array(15).fill(''));
let ocrLetters=null;   // snapshot after OCR, for showing edited indicators
let selectedCell=null; // {r, c} or null
let currentRack='';
let currentLexicon='';
let currentBag='';
let currentRackWarning='';
let currentInvalidWords=null;
let currentOccupancy=null;

const dropZone=document.getElementById('drop-zone');
const preview=document.getElementById('preview');
const fileInput=document.getElementById('file-input');
const cgpOut=document.getElementById('cgp-output');
const status=document.getElementById('status');
const boardArea=document.getElementById('board-area');
const debugImg=document.getElementById('debug-img');
const debugLog=document.getElementById('debug-log');

dropZone.addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',e=>{if(e.target.files.length)handleFile(e.target.files[0])});
dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('dragover')});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop',e=>{
  e.preventDefault();dropZone.classList.remove('dragover');
  if(e.dataTransfer.files.length){
    handleFile(e.dataTransfer.files[0]);
  }else{
    const url=e.dataTransfer.getData('text/uri-list')||e.dataTransfer.getData('text/plain')||'';
    if(url.startsWith('http')){
      handleURL(url.trim());
    }
  }
});

function showDebug(data){
  if(data.debug_image){
    debugImg.src=data.debug_image;
    debugImg.style.display='block';
  }else{
    debugImg.style.display='none';
  }
  if(data.log){
    debugLog.textContent=data.log;
    debugLog.style.display='block';
  }else{
    debugLog.style.display='none';
  }
}

async function processStream(res){
  const reader=res.body.getReader();
  const decoder=new TextDecoder();
  let buf='';
  while(true){
    const{done,value}=await reader.read();
    if(done) break;
    buf+=decoder.decode(value,{stream:true});
    let idx;
    while((idx=buf.indexOf('\n'))!==-1){
      const line=buf.slice(0,idx).trim();
      buf=buf.slice(idx+1);
      if(!line) continue;
      try{
        const data=JSON.parse(line);
        if(data.status) status.textContent=data.status+'\u2026';
        if(data.debug_image){debugImg.src=data.debug_image;debugImg.style.display='block';}
        if(data.log){debugLog.textContent=data.log;debugLog.style.display='block';}
        if(data.crops){
          const ca=document.getElementById('crops-area');
          const cc=document.getElementById('crops-container');
          cc.innerHTML='';
          for(const crop of data.crops){
            const d=document.createElement('div');
            d.style.cssText='text-align:center;background:#1a1a2e;border-radius:6px;padding:6px;min-width:60px';
            d.innerHTML=`<img src="${crop.img}" style="width:48px;height:48px;image-rendering:pixelated;border-radius:4px"><div style="font-size:.7rem;color:#ccc;margin-top:4px">${crop.pos}: ${crop.cur}</div>`;
            cc.appendChild(d);
          }
          ca.style.display='block';
        }
        if(data.cgp){
          cgpOut.value=data.cgp;
          cellData=data.cells||null;
          currentBag=data.bag||'';
          currentRackWarning=data.rack_warning||'';
          currentInvalidWords=data.invalid_words||null;
          currentOccupancy=data.occupancy||null;
          selectedCell=null;
          renderBoard(data.cgp);
          ocrLetters=boardLetters.map(row=>[...row]);
          if(!data.status) status.textContent='Done.';
        }
      }catch(e){}
    }
  }
}

async function handleFile(file){
  preview.src=URL.createObjectURL(file);
  preview.style.display='block';
  status.textContent='Uploading\u2026';
  const form=new FormData();
  form.append('image',file);
  try{
    const endpoint=document.getElementById('use-gemini').checked?'/analyze-gemini':'/analyze';
    const res=await fetch(endpoint,{method:'POST',body:form});
    await processStream(res);
  }catch(err){status.textContent='Error: '+err.message}
}

async function handleURL(url){
  preview.src=url;
  preview.style.display='block';
  status.textContent='Fetching URL\u2026';
  const useGemini=document.getElementById('use-gemini').checked;
  try{
    const res=await fetch('/fetch-url',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url,method:useGemini?'gemini':'template'})
    });
    await processStream(res);
  }catch(err){status.textContent='Error: '+err.message}
}

function parseCGP(cgp){
  const parts=cgp.trim().split(/\s+/);
  const boardStr=parts[0]||'';
  const rackStr=parts[1]||'';
  const racks=rackStr.split('/');
  const rack=racks[0]||'';

  const rows=boardStr.split('/');
  const board=[];
  for(const row of rows){
    const cells=[];
    let i=0;
    while(i<row.length){
      if(/[0-9]/.test(row[i])){
        let n='';
        while(i<row.length&&/[0-9]/.test(row[i])){n+=row[i];i++}
        for(let j=0;j<parseInt(n);j++)cells.push('');
      }else{cells.push(row[i]);i++}
    }
    while(cells.length<15)cells.push('');
    board.push(cells);
  }
  while(board.length<15)board.push(Array(15).fill(''));
  // Extract lexicon: "lex NWL23;" at end of CGP
  let lexicon='';
  const lexMatch=cgp.match(/lex\s+(\S+?);/);
  if(lexMatch) lexicon=lexMatch[1];
  return{board,rack,lexicon};
}

const tipEl=document.getElementById('tip');

function cellTooltip(r,c){
  if(!cellData||!cellData[r]||!cellData[r][c]) return null;
  const d=cellData[r][c];
  const lines=[];
  lines.push(`[${String.fromCharCode(65+c)}${r+1}]  Letter: ${d.l}  (${d.c}%)`);
  if(d.b) lines.push('Blank tile (no point value)');
  if(d.cands&&d.cands.length>0){
    lines.push('Candidates:');
    for(const cd of d.cands){
      const pts=LETTER_PTS[cd.l]||0;
      const bar='\u2588'.repeat(Math.max(1,Math.round(cd.c/10)));
      lines.push(`  ${cd.l} (${pts}pt) ${bar} ${cd.c}%`);
    }
  }
  return lines.join('\n');
}

function setupTips(){
  boardArea.addEventListener('mouseover',e=>{
    const cell=e.target.closest('.has-tip');
    if(!cell) return;
    const r=+cell.dataset.r, c=+cell.dataset.c;
    const text=cellTooltip(r,c);
    if(!text){tipEl.style.display='none';return;}
    tipEl.textContent=text;
    tipEl.style.display='block';
  });
  boardArea.addEventListener('mousemove',e=>{
    if(tipEl.style.display==='block'){
      tipEl.style.left=(e.clientX+12)+'px';
      tipEl.style.top=(e.clientY+12)+'px';
    }
  });
  boardArea.addEventListener('mouseout',e=>{
    const cell=e.target.closest('.has-tip');
    if(cell&&!cell.contains(e.relatedTarget)) tipEl.style.display='none';
  });
}
setupTips();

// --- Board editing ---

function boardToCGP(letters){
  let rows=[];
  for(let r=0;r<15;r++){
    let row='';
    let empty=0;
    for(let c=0;c<15;c++){
      if(!letters[r][c]){
        empty++;
      }else{
        if(empty>0){row+=empty;empty=0;}
        row+=letters[r][c];
      }
    }
    if(empty>0) row+=empty;
    rows.push(row);
  }
  return rows.join('/');
}

function syncCGP(){
  const old=cgpOut.value.trim();
  const parts=old.split(/\s+/);
  parts[0]=boardToCGP(boardLetters);
  cgpOut.value=parts.join(' ');
}

function renderBoard(cgp){
  const{board,rack,lexicon}=parseCGP(cgp);
  for(let r=0;r<15;r++)
    for(let c=0;c<15;c++)
      boardLetters[r][c]=board[r][c];
  currentRack=rack;
  currentLexicon=lexicon;
  renderBoardUI();
}

function renderBoardUI(){
  let h='<div class="board-wrapper"><div style="width:28px"></div><div class="col-labels">';
  for(let c=0;c<15;c++)h+=`<span>${String.fromCharCode(65+c)}</span>`;
  h+='</div><div class="row-labels-and-board"><div class="row-labels">';
  for(let r=0;r<15;r++)h+=`<span>${r+1}</span>`;
  h+='</div><div class="board">';
  for(let r=0;r<15;r++){
    for(let c=0;c<15;c++){
      const ch=boardLetters[r][c];
      const sel=selectedCell&&selectedCell.r===r&&selectedCell.c===c;
      const selCls=sel?' selected':'';
      const edited=ocrLetters&&ocrLetters[r][c]!==boardLetters[r][c];
      const editCls=edited?' edited':'';
      if(!ch){
        const p=PREMIUM[r][c];
        h+=`<div class="cell ${PCLS[p]}${selCls}${editCls}" data-r="${r}" data-c="${c}"><span class="lbl">${PLBL[p]}</span></div>`;
      }else{
        const blank=ch===ch.toLowerCase();
        const cls=blank?'blank-tile':'tile';
        const cd=cellData&&cellData[r]&&cellData[r][c];
        const hasTip=cd?'has-tip':'';
        let sub='';
        if(cd&&cd.s>0) sub=`<span class="sub">${cd.s}</span>`;
        else if(cd&&!cd.b){
          const ep=LETTER_PTS[ch.toUpperCase()];
          if(ep) sub=`<span class="sub" style="opacity:.35">${ep}</span>`;
        }
        h+=`<div class="cell ${cls} ${hasTip}${selCls}${editCls}" data-r="${r}" data-c="${c}">${ch.toUpperCase()}${sub}</div>`;
      }
    }
  }
  h+='</div></div></div>';
  if(currentRack){
    h+='<div class="rack">';
    for(const ch of currentRack){
      if(ch==='?')h+='<div class="rack-tile blank">?</div>';
      else h+=`<div class="rack-tile">${ch.toUpperCase()}</div>`;
    }
    h+='</div>';
  }
  if(currentLexicon){
    h+=`<div style="text-align:center;margin-top:8px;font-size:.75rem;color:#888">Lexicon: ${currentLexicon}</div>`;
  }
  if(currentBag){
    h+=`<div style="text-align:center;margin-top:6px;font-size:.75rem;color:#888">Bag: ${currentBag}</div>`;
  }
  if(currentRackWarning){
    h+=`<div style="text-align:center;margin-top:6px;font-size:.75rem;color:#e44;font-weight:bold">\u26a0 ${currentRackWarning}</div>`;
  }
  if(currentInvalidWords&&currentInvalidWords.length>0){
    h+=`<div style="text-align:center;margin-top:8px;padding:8px;background:#2a1515;border:1px solid #e44;border-radius:6px">`;
    h+=`<div style="font-size:.75rem;color:#e44;font-weight:bold;margin-bottom:4px">\u26a0 Invalid words</div>`;
    for(const iw of currentInvalidWords){
      h+=`<div style="font-size:.8rem;color:#f88;font-family:'SF Mono','Fira Code',monospace">${iw.word} <span style="color:#888">(${iw.pos})</span></div>`;
    }
    h+=`</div>`;
  }
  if(currentOccupancy){
    h+=`<div style="margin-top:12px;text-align:center"><div style="font-size:.7rem;color:#888;margin-bottom:4px">Occupancy Mask</div>`;
    h+=`<div style="display:inline-grid;grid-template-columns:repeat(15,12px);gap:1px">`;
    for(let r=0;r<15;r++)for(let c=0;c<15;c++){
      const occ=currentOccupancy[r]&&currentOccupancy[r][c];
      const bg=occ?'#4a8':'#333';
      h+=`<div style="width:12px;height:12px;background:${bg};border-radius:1px"></div>`;
    }
    h+=`</div></div>`;
  }
  boardArea.innerHTML=h;
}

// Click to select a cell
boardArea.addEventListener('click',e=>{
  const cell=e.target.closest('[data-r]');
  if(!cell){selectedCell=null;renderBoardUI();return;}
  selectedCell={r:+cell.dataset.r,c:+cell.dataset.c};
  renderBoardUI();
});

// Keyboard editing
document.addEventListener('keydown',e=>{
  if(!selectedCell) return;
  if(document.activeElement===cgpOut) return;
  const{r,c}=selectedCell;

  if(e.key==='Escape'){
    selectedCell=null;
    renderBoardUI();
    return;
  }
  if(e.key==='ArrowUp'){e.preventDefault();if(r>0)selectedCell.r--;renderBoardUI();return;}
  if(e.key==='ArrowDown'){e.preventDefault();if(r<14)selectedCell.r++;renderBoardUI();return;}
  if(e.key==='ArrowLeft'){e.preventDefault();if(c>0)selectedCell.c--;renderBoardUI();return;}
  if(e.key==='ArrowRight'){e.preventDefault();if(c<14)selectedCell.c++;renderBoardUI();return;}

  if(e.key==='Backspace'||e.key==='Delete'){
    e.preventDefault();
    boardLetters[r][c]='';
    if(c>0) selectedCell.c--;
    syncCGP();renderBoardUI();return;
  }
  if(e.key==='.'){
    e.preventDefault();
    boardLetters[r][c]='';
    if(c<14) selectedCell.c++;
    syncCGP();renderBoardUI();return;
  }
  // Shift+letter = regular tile (uppercase)
  if(/^[A-Z]$/.test(e.key)){
    e.preventDefault();
    boardLetters[r][c]=e.key;
    if(c<14) selectedCell.c++;
    syncCGP();renderBoardUI();return;
  }
  // Plain letter = blank tile (lowercase)
  if(/^[a-z]$/.test(e.key)){
    e.preventDefault();
    boardLetters[r][c]=e.key;
    if(c<14) selectedCell.c++;
    syncCGP();renderBoardUI();return;
  }
});

function renderFromField(){
  selectedCell=null;
  cellData=null;
  ocrLetters=null;
  renderBoard(cgpOut.value);
}
function copyCGP(){navigator.clipboard.writeText(cgpOut.value)}

// --- Save test case ---
async function saveTest(){
  const cgp=cgpOut.value.trim();
  if(!cgp){status.textContent='No CGP to save.';return;}
  status.textContent='Saving test case\u2026';
  try{
    const res=await fetch('/save-test',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({cgp})
    });
    const data=await res.json();
    if(data.error){status.textContent='Error: '+data.error;return;}
    status.textContent='Saved: '+data.saved;
  }catch(err){status.textContent='Error: '+err.message}
}

// --- Run tests ---
async function runTests(){
  status.textContent='Running tests\u2026';
  try{
    const res=await fetch('/run-tests');
    const data=await res.json();
    if(data.error){status.textContent='Error: '+data.error;return;}
    displayTestResults(data);
    status.textContent='Tests complete.';
  }catch(err){status.textContent='Error: '+err.message}
}

function displayTestResults(results){
  const panel=document.getElementById('test-results-panel');
  const container=document.getElementById('test-results');
  if(!results.length){container.innerHTML='<p style="color:#888">No test cases found.</p>';panel.style.display='block';return;}
  let h='<table class="test-results"><tr><th>Test</th><th>Total</th><th>Correct</th><th>Wrong</th><th>Accuracy</th></tr>';
  for(const r of results){
    const pct=r.total>0?(r.correct/r.total*100).toFixed(1):'0.0';
    const color=r.wrong===0?'#4c4':'#f88';
    h+=`<tr><td>${r.name}</td><td>${r.total}</td><td>${r.correct}</td><td style="color:${color}">${r.wrong}</td><td>${pct}%</td></tr>`;
  }
  h+='</table>';
  for(const r of results){
    if(r.diffs&&r.diffs.length>0){
      h+=`<div class="test-diff"><strong>${r.name} diffs:</strong><br>`;
      for(const d of r.diffs){
        h+=`${d.pos}: expected &lsquo;${d.exp}&rsquo; got &lsquo;${d.got}&rsquo;<br>`;
      }
      h+='</div>';
    }
  }
  container.innerHTML=h;
  panel.style.display='block';
}

renderBoard('15/15/15/15/15/15/15/15/15/15/15/15/15/15/15');
</script>
</body>
</html>)html";

// ---------------------------------------------------------------------------
// Build a progress NDJSON line (no cells/cgp, just status + log + image).
// ---------------------------------------------------------------------------
static std::string make_progress_line(const char* status,
                                       const std::string& log_text,
                                       const std::vector<uint8_t>& debug_png) {
    std::string json = "{\"status\":\"";
    json += json_escape(status);
    json += "\"";
    if (!log_text.empty())
        json += ",\"log\":\"" + json_escape(log_text) + "\"";
    if (!debug_png.empty())
        json += ",\"debug_image\":\"data:image/png;base64,"
              + base64_encode(debug_png) + "\"";
    json += "}\n";
    return json;
}

// ---------------------------------------------------------------------------
// Stream processing results as NDJSON (newline-delimited JSON).
// ---------------------------------------------------------------------------
static void stream_analyze(const std::vector<uint8_t>& buf,
                            httplib::DataSink& sink) {
    DebugResult dr = process_board_image_debug(buf,
        [&sink](const char* status, const std::string& log_text,
                const std::vector<uint8_t>& debug_png) {
            auto line = make_progress_line(status, log_text, debug_png);
            sink.write(line.data(), line.size());
        });

    // Final result line (includes cgp, cells, etc.)
    std::string final_json = make_json_response(dr);
    final_json += "\n";
    sink.write(final_json.data(), final_json.size());
    sink.done();
}

// ---------------------------------------------------------------------------
// Extract a JSON string value by key (simple, no nesting).
// ---------------------------------------------------------------------------
static std::string json_extract_string(const std::string& body,
                                        const char* key) {
    auto pos = body.find(std::string("\"") + key + "\"");
    if (pos == std::string::npos) return {};
    auto colon = body.find(':', pos);
    if (colon == std::string::npos) return {};
    auto q1 = body.find('"', colon + 1);
    if (q1 == std::string::npos) return {};
    auto q2 = body.find('"', q1 + 1);
    if (q2 == std::string::npos) return {};
    return body.substr(q1 + 1, q2 - q1 - 1);
}

// ---------------------------------------------------------------------------
// Extract the "text" string value from a Gemini API JSON response.
// Handles escaped characters within the string value.
// ---------------------------------------------------------------------------
static std::string extract_gemini_text(const std::string& json) {
    // Find the LAST "text" key — Gemini 2.5 models may have thinking parts
    // before the actual response, each with their own "text" key.
    size_t last_text = std::string::npos;
    size_t search = 0;
    while (true) {
        size_t found = json.find("\"text\"", search);
        if (found == std::string::npos) break;
        last_text = found;
        search = found + 6;
    }
    if (last_text == std::string::npos) return {};

    size_t pos = json.find(':', last_text + 6);
    if (pos == std::string::npos) return {};
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'
           || json[pos] == '\n' || json[pos] == '\r'))
        pos++;
    if (pos >= json.size() || json[pos] != '"') return {};
    pos++; // skip opening quote

    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                default:   result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        pos++;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Parse Gemini's 15x15 JSON array response into CellResult grid.
// Expected: [["A","B",null,...], ...] — uppercase=tile, lowercase=blank, null=empty
// ---------------------------------------------------------------------------
static bool parse_gemini_board(const std::string& text,
                                CellResult cells[15][15]) {
    std::string s = text;

    // Strip markdown code fences if present
    size_t fence_start = s.find("```");
    if (fence_start != std::string::npos) {
        size_t line_end = s.find('\n', fence_start);
        if (line_end != std::string::npos)
            s = s.substr(line_end + 1);
        size_t fence_end = s.rfind("```");
        if (fence_end != std::string::npos)
            s = s.substr(0, fence_end);
    }

    // Initialize cells to empty
    for (int r = 0; r < 15; r++)
        for (int c = 0; c < 15; c++)
            cells[r][c] = {};

    size_t pos = 0;
    auto skip_ws = [&]() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t'
               || s[pos] == '\n' || s[pos] == '\r'))
            pos++;
    };

    // Find the board array start — look for "board" key first, fallback to first [
    size_t board_start;
    auto board_key = s.find("\"board\"");
    if (board_key != std::string::npos)
        board_start = s.find('[', board_key);
    else
        board_start = s.find('[');
    if (board_start == std::string::npos) return false;
    pos = board_start + 1; // skip outer [

    for (int r = 0; r < 15; r++) {
        skip_ws();
        if (r > 0) {
            if (pos < s.size() && s[pos] == ',') pos++;
            skip_ws();
        }
        if (pos >= s.size() || s[pos] != '[') return false;
        pos++; // skip inner [

        for (int c = 0; c < 15; c++) {
            skip_ws();
            if (c > 0) {
                if (pos < s.size() && s[pos] == ',') pos++;
                skip_ws();
            }

            if (pos < s.size() && s[pos] == '"') {
                pos++; // skip opening quote
                if (pos < s.size()) {
                    char ch = s[pos];
                    pos++;
                    if (pos < s.size() && s[pos] == '"') pos++; // skip closing quote

                    cells[r][c].letter = ch;
                    cells[r][c].confidence = 1.0f;
                    if (ch >= 'a' && ch <= 'z')
                        cells[r][c].is_blank = true;
                }
            } else if (pos + 4 <= s.size() && s.substr(pos, 4) == "null") {
                pos += 4;
                // cell stays empty (default)
            }
        }

        skip_ws();
        if (pos < s.size() && s[pos] == ']') pos++; // skip inner ]
    }

    return true;
}

// ---------------------------------------------------------------------------
// Build CGP board string from CellResult grid.
// ---------------------------------------------------------------------------
static std::string cells_to_cgp(const CellResult cells[15][15]) {
    std::string result;
    for (int r = 0; r < 15; r++) {
        if (r > 0) result += '/';
        int empty = 0;
        for (int c = 0; c < 15; c++) {
            if (cells[r][c].letter == 0) {
                empty++;
            } else {
                if (empty > 0) {
                    result += std::to_string(empty);
                    empty = 0;
                }
                result += cells[r][c].letter;
            }
        }
        if (empty > 0)
            result += std::to_string(empty);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Parse board rectangle from OpenCV pipeline log.
// ---------------------------------------------------------------------------
static bool parse_board_rect_from_log(const std::string& log,
                                       int& bx, int& by, int& cell_sz) {
    auto pos = log.find("Final: rect=");
    if (pos == std::string::npos) return false;
    int bw, bh;
    return sscanf(log.c_str() + pos, "Final: rect=%d,%d %dx%d cell=%d",
                  &bx, &by, &bw, &bh, &cell_sz) == 5;
}

// ---------------------------------------------------------------------------
// Detect whether the board is in light mode or dark mode.
// Dark mode: dark green board background (V < 100)
// Light mode: white/cream board background (V > 150)
// ---------------------------------------------------------------------------
static bool detect_board_mode(const std::vector<uint8_t>& image_data,
                               int bx, int by, int cell_sz) {
    cv::Mat raw(1, static_cast<int>(image_data.size()), CV_8UC1,
                const_cast<uint8_t*>(image_data.data()));
    cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
    if (img.empty()) return false;

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // Sample several cells near corners (likely to be empty premium squares)
    // Use center of each cell to avoid edges
    int sample_cells[][2] = {{0,0}, {0,14}, {14,0}, {14,14}, {7,0}, {7,14}};
    double total_v = 0;
    int count = 0;
    for (auto& rc : sample_cells) {
        int r = rc[0], c = rc[1];
        int cx = bx + c * cell_sz + cell_sz / 2;
        int cy = by + r * cell_sz + cell_sz / 2;
        if (cx >= 0 && cx < img.cols && cy >= 0 && cy < img.rows) {
            cv::Vec3b pixel = hsv.at<cv::Vec3b>(cy, cx);
            total_v += pixel[2];
            count++;
        }
    }
    if (count == 0) return false;
    double mean_v = total_v / count;
    return mean_v > 150;  // light mode if bright background
}

// ---------------------------------------------------------------------------
// Color-based tile detection. Tiles are beige/tan (warm hue, low saturation,
// bright) and have internal contrast from letter text + subscript.
// ---------------------------------------------------------------------------
static void detect_tiles_by_color(const std::vector<uint8_t>& image_data,
                                   int bx, int by, int cell_sz,
                                   bool occupied[15][15],
                                   bool is_light_mode = false) {
    cv::Mat raw(1, static_cast<int>(image_data.size()), CV_8UC1,
                const_cast<uint8_t*>(image_data.data()));
    cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
    if (img.empty()) {
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++)
                occupied[r][c] = false;
        return;
    }

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    int inset = std::max(2, cell_sz / 4);

    for (int r = 0; r < 15; r++) {
        for (int c = 0; c < 15; c++) {
            int x0 = std::clamp(bx + c * cell_sz + inset, 0, img.cols - 1);
            int y0 = std::clamp(by + r * cell_sz + inset, 0, img.rows - 1);
            int x1 = std::clamp(bx + (c + 1) * cell_sz - inset, 1, img.cols);
            int y1 = std::clamp(by + (r + 1) * cell_sz - inset, 1, img.rows);

            if (x1 <= x0 || y1 <= y0) {
                occupied[r][c] = false;
                continue;
            }

            cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
            cv::Scalar mh = cv::mean(hsv(roi));
            double h = mh[0], s = mh[1], v = mh[2]; // H:0-180 S:0-255 V:0-255

            // Brightness variance — tiles have dark letter text on light bg
            cv::Mat gray;
            cv::cvtColor(img(roi), gray, cv::COLOR_BGR2GRAY);
            cv::Scalar gm, gs;
            cv::meanStdDev(gray, gm, gs);

            if (is_light_mode) {
                // Light mode: tiles are blue/purple squares on white bg
                bool is_blue_tile = (h >= 100 && h <= 140 && s > 40 &&
                                     v >= 40 && v <= 200);
                // Blank tiles: green circle with italic letter
                bool is_green_blank = (h >= 40 && h <= 85 && s > 40 && v > 60);
                // Recently played: orange/gold highlight
                bool is_orange = (h >= 10 && h <= 30 && s > 80 && v > 150);
                // Text contrast on colored tile background
                bool has_text = (gs[0] > 15 && v < 180 && s > 30);

                occupied[r][c] = is_blue_tile || is_green_blank ||
                                 is_orange || has_text;
            } else {
                // Dark mode: tiles are beige/tan on dark green board
                // Tile: warm beige (H ~10-35, low-medium S, bright V)
                bool is_beige = (h >= 8 && h <= 38 && s < 140 && v > 130);

                // Blank tile: purple circle (H ~130-155 in OpenCV = 260-310°)
                bool is_purple = (h >= 120 && h <= 160 && s > 30 && v > 60);

                // High internal contrast on a light, desaturated background
                bool has_text = (gs[0] > 20 && gm[0] > 110 && s < 120);

                occupied[r][c] = is_beige || is_purple || has_text;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Detect rack tiles below the board by scanning for beige/tan rectangles.
// Returns cropped images + bounding rects for each rack tile found.
// ---------------------------------------------------------------------------
struct RackTile {
    cv::Rect rect;
    std::vector<uint8_t> png;
    bool is_blank;  // true if tile appears to have no letter (blank)
};

static std::vector<RackTile> detect_rack_tiles(
    const std::vector<uint8_t>& image_data,
    int bx, int by, int cell_sz,
    bool is_light_mode = false)
{
    std::vector<RackTile> tiles;
    cv::Mat raw(1, static_cast<int>(image_data.size()), CV_8UC1,
                const_cast<uint8_t*>(image_data.data()));
    cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
    if (img.empty()) return tiles;

    // Rack is below the board. Search from board bottom (small gap)
    int board_bottom = by + 15 * cell_sz;
    int search_top = board_bottom + cell_sz / 4;  // start closer to board
    int search_bottom = std::min(img.rows, board_bottom + 5 * cell_sz);
    if (search_top >= img.rows) return tiles;

    // Convert to HSV for color detection
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // Create mask for tile color — beige in dark mode, blue/purple in light mode
    cv::Mat beige_mask;
    if (is_light_mode) {
        cv::inRange(hsv, cv::Scalar(90, 30, 30), cv::Scalar(150, 255, 220), beige_mask);
    } else {
        cv::inRange(hsv, cv::Scalar(5, 10, 100), cv::Scalar(45, 200, 255), beige_mask);
    }

    // Only look in the rack search area
    cv::Mat search_mask = cv::Mat::zeros(beige_mask.size(), beige_mask.type());
    int x_left = std::max(0, bx - cell_sz);
    int x_right = std::min(img.cols, bx + 15 * cell_sz + cell_sz);
    cv::Rect search_roi(x_left, search_top,
                         x_right - x_left,
                         search_bottom - search_top);
    search_roi &= cv::Rect(0, 0, img.cols, img.rows);
    beige_mask(search_roi).copyTo(search_mask(search_roi));

    // Morphological close to fill gaps in tile detection
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(search_mask, search_mask, cv::MORPH_CLOSE, kernel);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(search_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Filter by size — rack tiles should be roughly cell_sz x cell_sz
    int min_dim = cell_sz / 2;
    int max_dim = cell_sz * 2;
    for (const auto& contour : contours) {
        cv::Rect br = cv::boundingRect(contour);
        if (br.width < min_dim || br.height < min_dim) continue;
        if (br.width > max_dim || br.height > max_dim) continue;

        // Check if tile has text (letter) or is blank
        cv::Mat gray;
        cv::cvtColor(img(br), gray, cv::COLOR_BGR2GRAY);
        cv::Scalar gm, gs;
        cv::meanStdDev(gray, gm, gs);
        bool is_blank_tile = (gs[0] < 15); // very low variance = no letter

        // Crop with small padding
        int p = 2;
        int cx = std::max(0, br.x - p);
        int cy = std::max(0, br.y - p);
        int cw = std::min(br.width + 2*p, img.cols - cx);
        int ch = std::min(br.height + 2*p, img.rows - cy);
        cv::Mat crop = img(cv::Rect(cx, cy, cw, ch));
        std::vector<uint8_t> png_buf;
        cv::imencode(".png", crop, png_buf);

        tiles.push_back({cv::Rect(cx, cy, cw, ch), std::move(png_buf), is_blank_tile});
    }

    // Sort left to right
    std::sort(tiles.begin(), tiles.end(),
              [](const RackTile& a, const RackTile& b) {
                  return a.rect.x < b.rect.x;
              });

    return tiles;
}

// ---------------------------------------------------------------------------
// Draw rack tile detections on the debug image.
// ---------------------------------------------------------------------------
static void draw_rack_debug(std::vector<uint8_t>& debug_png,
                             const std::vector<RackTile>& rack_tiles)
{
    if (rack_tiles.empty() || debug_png.empty()) return;
    cv::Mat raw(1, static_cast<int>(debug_png.size()), CV_8UC1,
                debug_png.data());
    cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
    if (img.empty()) return;

    for (const auto& rt : rack_tiles) {
        cv::Scalar color = rt.is_blank
            ? cv::Scalar(255, 0, 255)   // magenta for blanks
            : cv::Scalar(0, 255, 255);  // yellow for regular
        cv::rectangle(img, rt.rect, color, 2);
    }

    std::vector<uint8_t> out;
    cv::imencode(".png", img, out);
    debug_png = std::move(out);
}

// ---------------------------------------------------------------------------
// Gemini Flash analysis with color-based occupancy assist.
// 1. Run OpenCV for board rect, then color-based tile detection
// 2. Show occupancy mask on board while waiting for Gemini
// 3. Include occupancy grid in Gemini prompt
// 4. Enforce occupancy mask on Gemini results
// ---------------------------------------------------------------------------
static void stream_analyze_gemini(const std::vector<uint8_t>& buf,
                                   httplib::DataSink& sink) {
    // Step 1: Run OpenCV pipeline for board detection
    {
        std::string msg = "{\"status\":\"Detecting board layout...\"}\n";
        sink.write(msg.data(), msg.size());
    }

    DebugResult opencv_dr;
    bool have_opencv = false;
    try {
        opencv_dr = process_board_image_debug(buf);
        have_opencv = true;
    } catch (...) {}

    // Step 2: Color-based tile detection
    bool color_occupied[15][15] = {};
    bool have_color = false;

    std::vector<RackTile> rack_tiles;
    bool is_light_mode = false;
    if (have_opencv) {
        int bx, by, cell_sz;
        if (parse_board_rect_from_log(opencv_dr.log, bx, by, cell_sz)) {
            is_light_mode = detect_board_mode(buf, bx, by, cell_sz);
            detect_tiles_by_color(buf, bx, by, cell_sz, color_occupied,
                                  is_light_mode);
            have_color = true;
            rack_tiles = detect_rack_tiles(buf, bx, by, cell_sz,
                                           is_light_mode);
            // Draw rack detections on debug image
            draw_rack_debug(opencv_dr.debug_png, rack_tiles);
            // Report rack detection
            int blank_ct = 0;
            for (const auto& rt : rack_tiles) if (rt.is_blank) blank_ct++;
            std::string rack_msg = "{\"status\":\"Detected " +
                std::to_string(rack_tiles.size()) + " rack tile(s)";
            if (blank_ct > 0) rack_msg += " (" + std::to_string(blank_ct) + " blank)";
            int board_bottom = by + 15 * cell_sz;
            rack_msg += ", search y=" + std::to_string(board_bottom + cell_sz/2)
                + "-" + std::to_string(std::min(static_cast<int>(849), board_bottom + 6*cell_sz));
            rack_msg += "\"}\n";
            sink.write(rack_msg.data(), rack_msg.size());
            // Report detected mode
            std::string mode_msg = std::string("{\"status\":\"Board mode: ") +
                (is_light_mode ? "light" : "dark") + "\"}\n";
            sink.write(mode_msg.data(), mode_msg.size());
        }
    }

    // Build occupancy grid for Gemini prompt
    std::string occupancy_grid;
    auto get_occupied = [&](int r, int c) -> bool {
        if (have_color) return color_occupied[r][c];
        if (have_opencv) return opencv_dr.cells[r][c].letter != 0;
        return false;
    };

    if (have_opencv) {
        for (int r = 0; r < 15; r++) {
            if (r > 0) occupancy_grid += "\\n";
            for (int c = 0; c < 15; c++)
                occupancy_grid += get_occupied(r, c) ? 'X' : '.';
        }
    }

    // Show occupancy mask on the board (empty tiles, no letters)
    if (have_opencv) {
        DebugResult mask_dr;
        mask_dr.debug_png = opencv_dr.debug_png;
        mask_dr.log = opencv_dr.log;
        for (int r = 0; r < 15; r++) {
            for (int c = 0; c < 15; c++) {
                if (get_occupied(r, c)) {
                    // Show as blank tile placeholder (no letter, just tile color)
                    mask_dr.cells[r][c].letter = '?';
                    mask_dr.cells[r][c].confidence = 0.0f;
                }
            }
        }
        mask_dr.cgp = cells_to_cgp(mask_dr.cells) + " / 0 0";
        std::string intermediate = make_json_response(mask_dr);
        intermediate = "{\"status\":\"Board detected \\u2014 calling Gemini Flash...\","
                      + intermediate.substr(1) + "\n";
        sink.write(intermediate.data(), intermediate.size());
    }

    // Step 3: Call Gemini Flash
    {
        size_t img_kb = buf.size() / 1024;
        std::string msg = "{\"status\":\"Calling Gemini Flash ("
            + std::to_string(img_kb) + " KB image)...\"}\n";
        sink.write(msg.data(), msg.size());
    }

    const char* api_key = std::getenv("GEMINI_API_KEY");
    if (!api_key || !api_key[0]) {
        std::string err = "{\"status\":\"Error: GEMINI_API_KEY not set in .env\"}\n";
        sink.write(err.data(), err.size());
        sink.done();
        return;
    }

    std::string b64 = base64_encode(buf);

    // Build prompt — include occupancy grid if available
    std::string prompt = "Look at this Scrabble board screenshot. Read every cell "
        "in the 15x15 grid.";

    if (!occupancy_grid.empty()) {
        prompt += "\\n\\nI have already detected which cells contain tiles using "
            "computer vision. Here is the occupancy grid where X = tile present "
            "and . = empty cell:\\n" + occupancy_grid +
            "\\n\\nUse this grid to know EXACTLY which cells have tiles and which "
            "are empty. Cells marked '.' MUST be null in your response. "
            "Every cell marked 'X' MUST have a letter — do NOT return null "
            "for any X cell.";
    }

    // Add rack tile detection info
    if (!rack_tiles.empty()) {
        int blank_count = 0;
        for (const auto& rt : rack_tiles)
            if (rt.is_blank) blank_count++;
        prompt += "\\n\\nComputer vision detected " +
            std::to_string(rack_tiles.size()) + " tiles on the rack";
        if (blank_count > 0)
            prompt += ", " + std::to_string(blank_count) +
                " of which appear to be BLANK (no letter, low contrast)."
                " These blanks should be reported as '?' in the rack string";
        prompt += ".";
    }

    prompt += "\\n\\nIMPORTANT — tile types in this Scrabble app (Woogles.io):";
    if (is_light_mode) {
        prompt +=
            "\\n- REGULAR tiles: BLUE/PURPLE SQUARES with WHITE upright letters "
            "and a small white subscript number (point value) in the bottom-right."
            "\\n- BLANK tiles on the BOARD: GREEN CIRCLES with italic/left-tilting "
            "letters and NO subscript."
            "\\n- Recently played tiles: ORANGE/GOLD highlighted squares."
            "\\n- BLANK tiles on the RACK: BLUE/PURPLE tiles with NO letter and NO "
            "subscript number. Any rack tile that has no visible letter is a blank (?). "
            "\\n  IMPORTANT: Count ALL blue/purple squares in the rack row, including "
            "empty-looking ones. If the bag has tiles, the rack MUST have exactly 7 "
            "tiles. If you only see 6 letters, one tile is a blank (?).";
    } else {
        prompt +=
            "\\n- REGULAR tiles: beige/tan SQUARES with upright letters and a small "
            "subscript number (point value) in the bottom-right corner."
            "\\n- BLANK tiles on the BOARD: PURPLE CIRCLES with italic/left-tilting "
            "letters and NO subscript."
            "\\n- BLANK tiles on the RACK: plain BEIGE tiles with NO letter and NO "
            "subscript number — they look like empty beige squares. Any rack tile "
            "that has no visible letter on it is a blank (?). "
            "\\n  IMPORTANT: Count ALL beige squares in the rack row, including "
            "empty-looking ones. If the bag has tiles, the rack MUST have exactly 7 "
            "tiles. If you only see 6 letters, one tile is a blank (?).";
    }
    prompt += "\\n\\nAlso read:"
        "\\n- The current player's RACK (row of tiles below the board). "
        "Use uppercase for regular tiles, '?' for blank tiles (tiles with no letter)."
        "\\n- The LEXICON shown in the game info area (e.g. \\\"NWL23\\\", \\\"CSW21\\\")."
        "\\n- The TILE TRACKING section (\\\"tiles in bag\\\" area) — read the letters "
        "listed there. They show remaining unseen tiles. Transcribe exactly, e.g. "
        "\\\"A E II O U B C D L N S TT X\\\"."
        "\\n- The SCORES for both players. The current player (whose rack is shown) "
        "has a colored score bar below their name. Read both scores as integers."
        "\\n\\nReturn ONLY a JSON object with these fields:"
        "\\n{"
        "\\n  \\\"board\\\": [[...], ...],  // 15x15 array"
        "\\n  \\\"rack\\\": \\\"ABCDE?F\\\",  // current player rack (? = blank)"
        "\\n  \\\"lexicon\\\": \\\"NWL23\\\",  // lexicon name"
        "\\n  \\\"bag\\\": \\\"A E II O U B C D L N S TT X\\\",  // tile tracking text"
        "\\n  \\\"scores\\\": [329, 351]  // [current player score, opponent score]"
        "\\n}";
    prompt += "\\n\\nBoard array elements:";
    if (is_light_mode) {
        prompt +=
            "\\n- Uppercase letter (e.g. \\\"A\\\") for regular tiles (blue/purple squares)"
            "\\n- Lowercase letter (e.g. \\\"s\\\") for BLANK tiles (green circles, italic)";
    } else {
        prompt +=
            "\\n- Uppercase letter (e.g. \\\"A\\\") for regular tiles (beige, with subscript)"
            "\\n- Lowercase letter (e.g. \\\"s\\\") for BLANK tiles (purple circles, italic)";
    }
    prompt += "\\n- null for empty cells"
        "\\n\\nSanity check: board tiles + rack tiles + bag tiles + opponent rack "
        "should total 100 tiles (standard English Scrabble distribution). "
        "If a rack tile is a plain ";
    prompt += is_light_mode ? "blue/purple" : "beige";
    prompt += " square with no letter, it is a blank (?)."
        "\\n\\nReturn ONLY the JSON object, no other text.";

    std::string payload = "{\"contents\":[{\"parts\":["
        "{\"text\":\"" + prompt + "\"},"
        "{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":\"" + b64 + "\"}}"
        "]}]}";

    // Write payload to temp file (too large for command-line arg)
    char tmppath[] = "/tmp/gemini_XXXXXX";
    int fd = mkstemp(tmppath);
    if (fd < 0) {
        std::string err = "{\"status\":\"Error: failed to create temp file\"}\n";
        sink.write(err.data(), err.size());
        sink.done();
        return;
    }
    ssize_t written = 0;
    size_t total = payload.size();
    while (written < static_cast<ssize_t>(total)) {
        ssize_t n = write(fd, payload.data() + written, total - written);
        if (n <= 0) break;
        written += n;
    }
    close(fd);

    std::string url = "https://generativelanguage.googleapis.com/v1beta/models/"
                      "gemini-2.5-flash:generateContent?key=";
    url += api_key;

    std::string cmd = "curl -s --max-time 60 -X POST "
        "-H 'Content-Type: application/json' "
        "-d @" + std::string(tmppath) + " "
        "'" + url + "'";

    auto t0 = std::chrono::steady_clock::now();
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        unlink(tmppath);
        std::string err = "{\"status\":\"Error: failed to execute curl\"}\n";
        sink.write(err.data(), err.size());
        sink.done();
        return;
    }

    std::string response;
    char chunk[8192];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), pipe)) > 0) {
        response.append(chunk, n);
    }
    pclose(pipe);
    unlink(tmppath);

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    {
        std::string msg = "{\"status\":\"Gemini responded ("
            + std::to_string(ms) + " ms, "
            + std::to_string(response.size() / 1024) + " KB). Parsing...\"}\n";
        sink.write(msg.data(), msg.size());
    }

    // Parse Gemini response
    gemini_log("main", prompt, response);  // always log raw response
    std::string text = extract_gemini_text(response);
    if (text.empty()) {
        std::string err_msg = json_extract_string(response, "message");
        if (err_msg.empty()) {
            // Show first 500 chars of raw response for debugging
            std::string preview = response.substr(0, 500);
            err_msg = "Failed to parse Gemini response. Raw: " + preview;
        }
        std::string err = "{\"status\":\"Error: " + json_escape(err_msg) + "\"}\n";
        sink.write(err.data(), err.size());
        sink.done();
        return;
    }

    // Extract rack, lexicon, and bag from the JSON response
    std::string gemini_rack, gemini_lexicon, gemini_bag;
    int gemini_score1 = 0, gemini_score2 = 0;
    {
        auto extract_str = [&](const char* key) -> std::string {
            auto kpos = text.find(std::string("\"") + key + "\"");
            if (kpos == std::string::npos) return {};
            auto colon = text.find(':', kpos);
            if (colon == std::string::npos) return {};
            auto q1 = text.find('"', colon + 1);
            if (q1 == std::string::npos) return {};
            auto q2 = text.find('"', q1 + 1);
            if (q2 == std::string::npos) return {};
            return text.substr(q1 + 1, q2 - q1 - 1);
        };
        gemini_rack = extract_str("rack");
        gemini_lexicon = extract_str("lexicon");
        gemini_bag = extract_str("bag");

        // Extract scores: find "scores" : [ N, N ]
        auto scores_pos = text.find("\"scores\"");
        if (scores_pos != std::string::npos) {
            auto br = text.find('[', scores_pos);
            if (br != std::string::npos) {
                int s1 = 0, s2 = 0;
                if (sscanf(text.c_str() + br, "[%d,%d]", &s1, &s2) == 2 ||
                    sscanf(text.c_str() + br, "[ %d , %d ]", &s1, &s2) == 2) {
                    gemini_score1 = s1;
                    gemini_score2 = s2;
                }
            }
        }
    }

    // Parse the 15x15 board (find first [[ in the text)
    DebugResult dr = {};
    if (!parse_gemini_board(text, dr.cells)) {
        std::string preview = text.substr(0, 300);
        std::string err = "{\"status\":\"Error: Failed to parse board\","
            "\"log\":\"" + json_escape("Gemini text:\n" + preview) + "\"}\n";
        sink.write(err.data(), err.size());
        sink.done();
        return;
    }

    // Step 3.5: Row realignment — Gemini often returns correct letters
    // but shifted horizontally from the occupancy mask positions.
    // For each row, find contiguous blocks in both mask and Gemini output.
    // If blocks match in count and relative spacing, realign Gemini's
    // letters to the mask positions.
    if (have_color || have_opencv) {
        std::string realign_log;
        for (int r = 0; r < 15; r++) {
            // Find contiguous blocks in mask
            struct Block { int start, len; };
            std::vector<Block> mask_blocks, gem_blocks;
            {
                int c = 0;
                while (c < 15) {
                    if (get_occupied(r, c)) {
                        int s = c;
                        while (c < 15 && get_occupied(r, c)) c++;
                        mask_blocks.push_back({s, c - s});
                    } else c++;
                }
            }
            {
                int c = 0;
                while (c < 15) {
                    if (dr.cells[r][c].letter != 0) {
                        int s = c;
                        while (c < 15 && dr.cells[r][c].letter != 0) c++;
                        gem_blocks.push_back({s, c - s});
                    } else c++;
                }
            }
            // Try to match blocks: same number and same lengths
            if (mask_blocks.size() != gem_blocks.size() ||
                mask_blocks.empty()) continue;
            bool lengths_match = true;
            for (size_t i = 0; i < mask_blocks.size(); i++) {
                if (mask_blocks[i].len != gem_blocks[i].len) {
                    lengths_match = false;
                    break;
                }
            }
            if (!lengths_match) continue;
            // Check if any block is shifted
            bool any_shifted = false;
            for (size_t i = 0; i < mask_blocks.size(); i++) {
                if (mask_blocks[i].start != gem_blocks[i].start) {
                    any_shifted = true;
                    break;
                }
            }
            if (!any_shifted) continue;
            // Realign: save letters, clear row, place at mask positions
            CellResult saved[15];
            for (int c = 0; c < 15; c++) saved[c] = dr.cells[r][c];
            for (int c = 0; c < 15; c++) dr.cells[r][c] = {};
            for (size_t i = 0; i < mask_blocks.size(); i++) {
                for (int j = 0; j < mask_blocks[i].len; j++) {
                    dr.cells[r][mask_blocks[i].start + j] =
                        saved[gem_blocks[i].start + j];
                }
                if (mask_blocks[i].start != gem_blocks[i].start) {
                    // Collect letters for log
                    std::string letters;
                    for (int j = 0; j < mask_blocks[i].len; j++)
                        letters += static_cast<char>(std::toupper(
                            static_cast<unsigned char>(
                                saved[gem_blocks[i].start + j].letter)));
                    realign_log += "Row " + std::to_string(r + 1)
                        + ": shifted " + letters + " from cols "
                        + std::to_string(gem_blocks[i].start + 1) + "-"
                        + std::to_string(gem_blocks[i].start + gem_blocks[i].len)
                        + " to " + std::to_string(mask_blocks[i].start + 1)
                        + "-" + std::to_string(mask_blocks[i].start + mask_blocks[i].len)
                        + "\n";
                }
            }
        }
        if (!realign_log.empty()) {
            std::string msg = "{\"status\":\"Realigned shifted rows: "
                + json_escape(realign_log) + "\"}\n";
            sink.write(msg.data(), msg.size());
        }
    }

    // Step 4: Enforce occupancy mask — clear non-occupied cells
    if (have_color || have_opencv) {
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++)
                if (!get_occupied(r, c) && dr.cells[r][c].letter != 0)
                    dr.cells[r][c] = {};
    }

    // Step 5: Re-query Gemini for occupied cells it missed (crop + retry)
    struct MissPos { int r, c; };
    std::vector<MissPos> missing;
    if (have_color || have_opencv) {
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++)
                if (get_occupied(r, c) && dr.cells[r][c].letter == 0)
                    missing.push_back({r, c});
    }

    if (!missing.empty() && have_opencv) {
        std::string msg = "{\"status\":\"Re-querying "
            + std::to_string(missing.size()) + " unclear cell(s)...\"}\n";
        sink.write(msg.data(), msg.size());

        int bx, by, cell_sz;
        if (parse_board_rect_from_log(opencv_dr.log, bx, by, cell_sz)) {
            cv::Mat raw(1, static_cast<int>(buf.size()), CV_8UC1,
                        const_cast<uint8_t*>(buf.data()));
            cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);

            if (!img.empty()) {
                // Build multi-image retry with cropped cells
                std::string retry_prompt =
                    "Here are cropped images of individual Scrabble tiles. "
                    "For each image, identify the letter on the tile. "
                    "Regular tiles are beige with a subscript (return UPPERCASE). "
                    "Blank tiles are purple circles with italic letters "
                    "(return lowercase). "
                    "Return ONLY a JSON array of single letters, one per image. "
                    "Example: [\\\"F\\\", \\\"s\\\"]";

                std::string retry_payload = "{\"contents\":[{\"parts\":["
                    "{\"text\":\"" + retry_prompt + "\"}";

                int pad = cell_sz / 8;
                for (const auto& mc : missing) {
                    int cx = std::max(0, bx + mc.c * cell_sz - pad);
                    int cy = std::max(0, by + mc.r * cell_sz - pad);
                    int cw = std::min(cell_sz + 2 * pad, img.cols - cx);
                    int ch = std::min(cell_sz + 2 * pad, img.rows - cy);
                    cv::Mat cell_img = img(cv::Rect(cx, cy, cw, ch));
                    std::vector<uint8_t> png_buf;
                    cv::imencode(".png", cell_img, png_buf);
                    retry_payload += ",{\"inlineData\":{\"mimeType\":\"image/png\","
                        "\"data\":\"" + base64_encode(png_buf) + "\"}}";
                }
                retry_payload += "]}]}";

                char tmppath2[] = "/tmp/gemini2_XXXXXX";
                int fd2 = mkstemp(tmppath2);
                if (fd2 >= 0) {
                    ssize_t w2 = 0;
                    size_t t2 = retry_payload.size();
                    while (w2 < static_cast<ssize_t>(t2)) {
                        ssize_t nn = write(fd2, retry_payload.data() + w2, t2 - w2);
                        if (nn <= 0) break;
                        w2 += nn;
                    }
                    close(fd2);

                    std::string cmd2 = "curl -s --max-time 30 -X POST "
                        "-H 'Content-Type: application/json' "
                        "-d @" + std::string(tmppath2) + " '" + url + "'";

                    FILE* pipe2 = popen(cmd2.c_str(), "r");
                    if (pipe2) {
                        std::string resp2;
                        char ch2[8192];
                        size_t nn;
                        while ((nn = fread(ch2, 1, sizeof(ch2), pipe2)) > 0)
                            resp2.append(ch2, nn);
                        pclose(pipe2);

                        std::string text2 = extract_gemini_text(resp2);
                        gemini_log("retry_missing", retry_prompt, text2.empty() ? resp2 : text2);
                        if (!text2.empty()) {
                            // Strip markdown fences
                            std::string s2 = text2;
                            auto f1 = s2.find("```");
                            if (f1 != std::string::npos) {
                                auto nl = s2.find('\n', f1);
                                if (nl != std::string::npos) s2 = s2.substr(nl + 1);
                                auto f2 = s2.rfind("```");
                                if (f2 != std::string::npos) s2 = s2.substr(0, f2);
                            }
                            // Parse ["F", "s", ...]
                            size_t idx = 0, mi = 0;
                            while (idx < s2.size() && s2[idx] != '[') idx++;
                            idx++;
                            while (idx < s2.size() && mi < missing.size()) {
                                while (idx < s2.size() && s2[idx] != '"'
                                       && s2[idx] != ']') idx++;
                                if (idx >= s2.size() || s2[idx] == ']') break;
                                idx++; // skip opening "
                                if (idx < s2.size()) {
                                    char ltr = s2[idx];
                                    auto& cell = dr.cells[missing[mi].r][missing[mi].c];
                                    cell.letter = ltr;
                                    cell.confidence = 0.8f;
                                    cell.is_blank = (ltr >= 'a' && ltr <= 'z');
                                    mi++;
                                    idx++;
                                    if (idx < s2.size() && s2[idx] == '"') idx++;
                                }
                            }
                        }
                    }
                    unlink(tmppath2);
                }
            }
        }
    }

    // Any cells still unresolved get '?' placeholder
    if (have_color || have_opencv) {
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++)
                if (get_occupied(r, c) && dr.cells[r][c].letter == 0) {
                    dr.cells[r][c].letter = '?';
                    dr.cells[r][c].confidence = 0.0f;
                }
    }

    // Step 6: Bag-math validation — re-query cells with over-counted letters
    if (have_opencv && !gemini_bag.empty()) {
        int full_bag[27] = {};
        auto init_bag = [&](char ch, int n) { full_bag[ch - 'A'] = n; };
        init_bag('A',9); init_bag('B',2); init_bag('C',2); init_bag('D',4);
        init_bag('E',12); init_bag('F',2); init_bag('G',3); init_bag('H',2);
        init_bag('I',9); init_bag('J',1); init_bag('K',1); init_bag('L',4);
        init_bag('M',2); init_bag('N',6); init_bag('O',8); init_bag('P',2);
        init_bag('Q',1); init_bag('R',6); init_bag('S',4); init_bag('T',6);
        init_bag('U',4); init_bag('V',2); init_bag('W',2); init_bag('X',1);
        init_bag('Y',2); init_bag('Z',1); full_bag[26] = 2;

        // Subtract board tiles
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++) {
                char ch = dr.cells[r][c].letter;
                if (ch == 0 || ch == '?') continue;
                if (dr.cells[r][c].is_blank) full_bag[26]--;
                else {
                    int idx = std::toupper(static_cast<unsigned char>(ch)) - 'A';
                    if (idx >= 0 && idx < 26) full_bag[idx]--;
                }
            }
        // Subtract bag tiles
        for (size_t i = 0; i < gemini_bag.size(); i++) {
            char ch = gemini_bag[i];
            if (ch == '?' || ch == '_') full_bag[26]--;
            else if (std::isalpha(static_cast<unsigned char>(ch))) {
                int idx = std::toupper(static_cast<unsigned char>(ch)) - 'A';
                if (idx >= 0 && idx < 26) full_bag[idx]--;
            }
        }
        // Subtract rack tiles
        for (char ch : gemini_rack) {
            if (ch == '?') full_bag[26]--;
            else {
                int idx = std::toupper(static_cast<unsigned char>(ch)) - 'A';
                if (idx >= 0 && idx < 26) full_bag[idx]--;
            }
        }

        // full_bag[i] < 0 means we used more of letter i than exist
        // Find over-counted letters on the board and re-query those cells
        bool overcount[26] = {};
        bool has_overcount = false;
        for (int i = 0; i < 26; i++) {
            if (full_bag[i] < 0) {
                overcount[i] = true;
                has_overcount = true;
            }
        }

        if (has_overcount) {
            struct SuspectPos { int r, c; };
            std::vector<SuspectPos> suspects;
            for (int r = 0; r < 15; r++)
                for (int c = 0; c < 15; c++) {
                    char ch = dr.cells[r][c].letter;
                    if (ch == 0 || ch == '?' || dr.cells[r][c].is_blank) continue;
                    int idx = std::toupper(static_cast<unsigned char>(ch)) - 'A';
                    if (idx >= 0 && idx < 26 && overcount[idx])
                        suspects.push_back({r, c});
                }

            if (!suspects.empty()) {
                // Build over-counted letter list for the prompt
                std::string over_letters;
                for (int i = 0; i < 26; i++)
                    if (overcount[i]) {
                        if (!over_letters.empty()) over_letters += ", ";
                        over_letters += static_cast<char>('A' + i);
                    }

                int bx, by, cell_sz;
                if (parse_board_rect_from_log(opencv_dr.log, bx, by, cell_sz)) {
                    cv::Mat raw_v(1, static_cast<int>(buf.size()), CV_8UC1,
                                const_cast<uint8_t*>(buf.data()));
                    cv::Mat img_v = cv::imdecode(raw_v, cv::IMREAD_COLOR);

                    if (!img_v.empty()) {
                        // Build status with crop images so user can inspect
                        std::string crop_status = "{\"status\":\"Verifying "
                            + std::to_string(suspects.size()) + " cell(s) ("
                            + over_letters + " over-counted)...\",\"crops\":[";

                        std::string vfy_prompt =
                            "These are cropped Scrabble tile images. The tile "
                            "counts don't add up \\u2014 too many of: " + over_letters +
                            ". For each image, carefully identify the letter. "
                            "Pay close attention to similar-looking letters "
                            "(H vs I vs N, O vs Q vs D, etc). "
                            "Regular tiles are beige with a subscript (UPPERCASE). "
                            "Blank tiles are purple circles with italic letters "
                            "(lowercase). "
                            "Return ONLY a JSON array of single letters. "
                            "Example: [\\\"I\\\", \\\"H\\\"]";

                        std::string vfy_payload = "{\"contents\":[{\"parts\":["
                            "{\"text\":\"" + vfy_prompt + "\"}";

                        int pad = cell_sz / 8;
                        for (size_t si = 0; si < suspects.size(); si++) {
                            const auto& sp = suspects[si];
                            int cx = std::max(0, bx + sp.c * cell_sz - pad);
                            int cy = std::max(0, by + sp.r * cell_sz - pad);
                            int cw = std::min(cell_sz + 2*pad, img_v.cols - cx);
                            int ch_ = std::min(cell_sz + 2*pad, img_v.rows - cy);
                            cv::Mat cell_img = img_v(cv::Rect(cx, cy, cw, ch_));
                            std::vector<uint8_t> png_buf;
                            cv::imencode(".png", cell_img, png_buf);
                            std::string b64_crop = base64_encode(png_buf);
                            vfy_payload += ",{\"inlineData\":{\"mimeType\":\"image/png\","
                                "\"data\":\"" + b64_crop + "\"}}";
                            // Add to status crops
                            if (si > 0) crop_status += ",";
                            char pos_label = static_cast<char>(
                                std::toupper(static_cast<unsigned char>(
                                    dr.cells[sp.r][sp.c].letter)));
                            crop_status += "{\"pos\":\"" +
                                std::string(1, static_cast<char>('A' + sp.c)) +
                                std::to_string(sp.r + 1) + "\",\"cur\":\"" +
                                std::string(1, pos_label) +
                                "\",\"img\":\"data:image/png;base64," +
                                b64_crop + "\"}";
                        }
                        vfy_payload += "]}]}";

                        // Send status with crop previews
                        crop_status += "]}\n";
                        sink.write(crop_status.data(), crop_status.size());

                        char tmpv[] = "/tmp/gemini_v_XXXXXX";
                        int fdv = mkstemp(tmpv);
                        if (fdv >= 0) {
                            ssize_t wv = 0;
                            size_t tv = vfy_payload.size();
                            while (wv < static_cast<ssize_t>(tv)) {
                                ssize_t nn = write(fdv, vfy_payload.data() + wv, tv - wv);
                                if (nn <= 0) break;
                                wv += nn;
                            }
                            close(fdv);

                            std::string cmdv = "curl -s --max-time 30 -X POST "
                                "-H 'Content-Type: application/json' "
                                "-d @" + std::string(tmpv) + " '" + url + "'";
                            FILE* pipev = popen(cmdv.c_str(), "r");
                            if (pipev) {
                                std::string respv;
                                char chv[8192];
                                size_t nnv;
                                while ((nnv = fread(chv, 1, sizeof(chv), pipev)) > 0)
                                    respv.append(chv, nnv);
                                pclose(pipev);

                                std::string textv = extract_gemini_text(respv);
                                gemini_log("verify_board", vfy_prompt, textv.empty() ? respv : textv);
                                if (!textv.empty()) {
                                    std::string sv = textv;
                                    auto f1 = sv.find("```");
                                    if (f1 != std::string::npos) {
                                        auto nl = sv.find('\n', f1);
                                        if (nl != std::string::npos) sv = sv.substr(nl+1);
                                        auto f2 = sv.rfind("```");
                                        if (f2 != std::string::npos) sv = sv.substr(0, f2);
                                    }
                                    size_t vi = 0, si = 0;
                                    int corrections = 0;
                                    std::string corr_detail;
                                    while (vi < sv.size() && sv[vi] != '[') vi++;
                                    vi++;
                                    while (vi < sv.size() && si < suspects.size()) {
                                        while (vi < sv.size() && sv[vi] != '"'
                                               && sv[vi] != ']') vi++;
                                        if (vi >= sv.size() || sv[vi] == ']') break;
                                        vi++;
                                        if (vi < sv.size()) {
                                            char ltr = sv[vi];
                                            auto& cell = dr.cells[suspects[si].r][suspects[si].c];
                                            if (ltr != cell.letter) {
                                                if (!corr_detail.empty()) corr_detail += ", ";
                                                corr_detail +=
                                                    std::string(1, static_cast<char>('A' + suspects[si].c))
                                                    + std::to_string(suspects[si].r + 1) + ": "
                                                    + static_cast<char>(std::toupper(
                                                        static_cast<unsigned char>(cell.letter)))
                                                    + " -> "
                                                    + static_cast<char>(std::toupper(
                                                        static_cast<unsigned char>(ltr)));
                                                cell.letter = ltr;
                                                cell.confidence = 0.85f;
                                                cell.is_blank = (ltr >= 'a' && ltr <= 'z');
                                                corrections++;
                                            }
                                            si++;
                                            vi++;
                                            if (vi < sv.size() && sv[vi] == '"') vi++;
                                        }
                                    }
                                    // Report corrections
                                    std::string vfy_result;
                                    if (corrections > 0)
                                        vfy_result = "{\"status\":\"Corrected "
                                            + std::to_string(corrections) + " cell(s): "
                                            + json_escape(corr_detail) + "\"}\n";
                                    else
                                        vfy_result = "{\"status\":\"Verification confirmed all "
                                            + std::to_string(suspects.size()) + " cell(s)\"}\n";
                                    sink.write(vfy_result.data(), vfy_result.size());
                                }
                            }
                            unlink(tmpv);
                        }
                    }
                }
            }
        }
    }

    // Step 7: Rack verification — send rack crops to Gemini if available
    if (!rack_tiles.empty() && rack_tiles.size() <= 7 &&
        (static_cast<int>(gemini_rack.size()) != static_cast<int>(rack_tiles.size())
         || std::any_of(rack_tiles.begin(), rack_tiles.end(),
                        [](const RackTile& rt) { return rt.is_blank; }))) {
        std::string rack_msg = "{\"status\":\"Verifying rack ("
            + std::to_string(rack_tiles.size()) + " tiles detected)...\",\"crops\":[";
        std::string rack_prompt =
            "These are cropped images of Scrabble rack tiles, in order left to right. "
            "Identify each tile's letter. "
            "Regular tiles are beige with a letter and a subscript number (return UPPERCASE). "
            "BLANK tiles are beige with NO letter and NO subscript \\u2014 return '?' for these. "
            "Return ONLY a JSON array of strings, one per image. "
            "Example: [\\\"B\\\", \\\"I\\\", \\\"?\\\"]";
        std::string rack_payload = "{\"contents\":[{\"parts\":["
            "{\"text\":\"" + rack_prompt + "\"}";
        for (size_t ri = 0; ri < rack_tiles.size(); ri++) {
            std::string b64r = base64_encode(rack_tiles[ri].png);
            rack_payload += ",{\"inlineData\":{\"mimeType\":\"image/png\","
                "\"data\":\"" + b64r + "\"}}";
            if (ri > 0) rack_msg += ",";
            rack_msg += "{\"pos\":\"R" + std::to_string(ri + 1) +
                "\",\"cur\":\"" +
                (ri < gemini_rack.size()
                    ? std::string(1, gemini_rack[ri])
                    : std::string("?")) +
                "\",\"img\":\"data:image/png;base64," + b64r + "\"}";
        }
        rack_payload += "]}]}";
        rack_msg += "]}\n";
        sink.write(rack_msg.data(), rack_msg.size());

        // Send to Gemini
        char tmpr[] = "/tmp/gemini_r_XXXXXX";
        int fdr = mkstemp(tmpr);
        if (fdr >= 0) {
            ssize_t wr = 0;
            size_t tr = rack_payload.size();
            while (wr < static_cast<ssize_t>(tr)) {
                ssize_t nn = write(fdr, rack_payload.data() + wr, tr - wr);
                if (nn <= 0) break;
                wr += nn;
            }
            close(fdr);

            std::string cmdr = "curl -s --max-time 30 -X POST "
                "-H 'Content-Type: application/json' "
                "-d @" + std::string(tmpr) + " '" + url + "'";
            FILE* piper = popen(cmdr.c_str(), "r");
            if (piper) {
                std::string respr;
                char chr[8192];
                size_t nnr;
                while ((nnr = fread(chr, 1, sizeof(chr), piper)) > 0)
                    respr.append(chr, nnr);
                pclose(piper);

                std::string textr = extract_gemini_text(respr);
                gemini_log("verify_rack", rack_prompt, textr.empty() ? respr : textr);
                if (!textr.empty()) {
                    // Parse ["B", "I", "?", ...]
                    std::string new_rack;
                    size_t ri = 0;
                    while (ri < textr.size() && textr[ri] != '[') ri++;
                    ri++;
                    while (ri < textr.size()) {
                        while (ri < textr.size() && textr[ri] != '"'
                               && textr[ri] != ']') ri++;
                        if (ri >= textr.size() || textr[ri] == ']') break;
                        ri++;
                        if (ri < textr.size()) {
                            new_rack += textr[ri];
                            ri++;
                            if (ri < textr.size() && textr[ri] == '"') ri++;
                        }
                    }
                    if (!new_rack.empty() && new_rack != gemini_rack) {
                        std::string rmsg = "{\"status\":\"Rack corrected: "
                            + json_escape(gemini_rack) + " -> "
                            + json_escape(new_rack) + "\"}\n";
                        sink.write(rmsg.data(), rmsg.size());
                        gemini_rack = new_rack;
                    }
                }
            }
            unlink(tmpr);
        }
    }

    // Build CGP: <board> <rack>/ <scores> lex <lexicon>;
    std::string rack_str = gemini_rack.empty() ? "" : gemini_rack;
    std::string lex_str = gemini_lexicon.empty() ? "" : gemini_lexicon;
    dr.cgp = cells_to_cgp(dr.cells) + " " + rack_str + "/ "
           + std::to_string(gemini_score1) + " " + std::to_string(gemini_score2);
    if (!lex_str.empty())
        dr.cgp += " lex " + lex_str + ";";
    dr.log = have_color ? "Color occupancy + Gemini Flash OCR"
           : have_opencv ? "OpenCV occupancy + Gemini Flash OCR"
           : "Gemini Flash analysis";

    if (have_opencv)
        dr.debug_png = std::move(opencv_dr.debug_png);

    // --- Rack validation & auto-correction ---
    // full bag - board tiles - bag tiles = expected rack
    std::string rack_warning;
    {
        // Standard Scrabble tile distribution (100 tiles)
        int full_bag[27] = {}; // A-Z = 0-25, blank = 26
        auto set_bag = [&](char ch, int n) {
            full_bag[ch - 'A'] = n;
        };
        set_bag('A',9); set_bag('B',2); set_bag('C',2); set_bag('D',4);
        set_bag('E',12); set_bag('F',2); set_bag('G',3); set_bag('H',2);
        set_bag('I',9); set_bag('J',1); set_bag('K',1); set_bag('L',4);
        set_bag('M',2); set_bag('N',6); set_bag('O',8); set_bag('P',2);
        set_bag('Q',1); set_bag('R',6); set_bag('S',4); set_bag('T',6);
        set_bag('U',4); set_bag('V',2); set_bag('W',2); set_bag('X',1);
        set_bag('Y',2); set_bag('Z',1); full_bag[26] = 2; // blanks

        // Subtract board tiles
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++) {
                char ch = dr.cells[r][c].letter;
                if (ch == 0 || ch == '?') continue;
                if (dr.cells[r][c].is_blank) {
                    full_bag[26]--;
                } else {
                    int idx = std::toupper(static_cast<unsigned char>(ch)) - 'A';
                    if (idx >= 0 && idx < 26) full_bag[idx]--;
                }
            }

        // Subtract bag/tracking tiles (parse "A E II O U B C D" format)
        if (!gemini_bag.empty()) {
            for (size_t i = 0; i < gemini_bag.size(); i++) {
                char ch = gemini_bag[i];
                if (ch == '?' || ch == '_') {
                    full_bag[26]--;
                } else if (std::isalpha(static_cast<unsigned char>(ch))) {
                    int idx = std::toupper(static_cast<unsigned char>(ch)) - 'A';
                    if (idx >= 0 && idx < 26) full_bag[idx]--;
                }
            }
        }

        // full_bag now holds: expected unseen tiles (rack + opponent rack)
        // Build the expected rack from what's left, compare to Gemini's rack
        int rack_counts[27] = {};
        for (char ch : gemini_rack) {
            if (ch == '?') rack_counts[26]++;
            else {
                int idx = std::toupper(static_cast<unsigned char>(ch)) - 'A';
                if (idx >= 0 && idx < 26) rack_counts[idx]++;
            }
        }

        // Count total expected unseen tiles
        int total_unseen = 0;
        for (int i = 0; i < 27; i++) total_unseen += full_bag[i];
        int rack_size = static_cast<int>(gemini_rack.size());

        // Auto-correct: if Gemini gave fewer tiles than expected rack size
        // (bag not empty → rack should be 7, bag empty → rack = total_unseen)
        // The bag text tells us tiles in the bag; if bag text exists and has
        // tiles, rack should be 7
        bool has_bag_tiles = false;
        if (!gemini_bag.empty()) {
            for (char ch : gemini_bag)
                if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '?') {
                    has_bag_tiles = true; break;
                }
        }
        int expected_rack_size = has_bag_tiles ? 7 : total_unseen;
        // Clamp: can't have more than total unseen
        if (expected_rack_size > total_unseen)
            expected_rack_size = total_unseen;

        if (rack_size < expected_rack_size && !gemini_bag.empty()) {
            // A missing rack tile is almost always a blank (?) on Woogles —
            // it's a plain beige square with no letter that Gemini skips.
            // If OpenCV detected a blank rack tile, or the rack is 1 short,
            // prefer adding '?' over inferred letters.
            bool cv_saw_blank = false;
            for (const auto& rt : rack_tiles)
                if (rt.is_blank) { cv_saw_blank = true; break; }

            std::string corrected_rack = gemini_rack;
            int to_add = expected_rack_size - rack_size;

            // If CV saw a blank tile or blanks are available in the bag,
            // fill missing slots with '?' first
            if (cv_saw_blank || full_bag[26] > 0) {
                int blanks_to_add = std::min(to_add,
                    std::max(full_bag[26] - rack_counts[26], 0));
                if (cv_saw_blank && blanks_to_add == 0) blanks_to_add = 1;
                blanks_to_add = std::min(blanks_to_add, to_add);
                for (int j = 0; j < blanks_to_add; j++)
                    corrected_rack += '?';
            }

            // Fill any remaining from bag math (letters)
            for (int i = 0; i < 26 && (int)corrected_rack.size() < expected_rack_size; i++) {
                int deficit = full_bag[i] - rack_counts[i];
                for (int j = 0; j < deficit && (int)corrected_rack.size() < expected_rack_size; j++) {
                    corrected_rack += static_cast<char>('A' + i);
                }
            }
            if (corrected_rack != gemini_rack) {
                rack_warning = "Auto-corrected rack: " + gemini_rack
                             + " -> " + corrected_rack;
                gemini_rack = corrected_rack;
                // Rebuild CGP with corrected rack
                dr.cgp = cells_to_cgp(dr.cells) + " " + gemini_rack + "/ "
                       + std::to_string(gemini_score1) + " "
                       + std::to_string(gemini_score2);
                if (!lex_str.empty())
                    dr.cgp += " lex " + lex_str + ";";
            }
        } else {
            // Just report mismatches if any
            std::string mismatches;
            for (int i = 0; i < 27; i++) {
                if (full_bag[i] != rack_counts[i]) {
                    char label = (i < 26) ? static_cast<char>('A' + i) : '?';
                    if (!mismatches.empty()) mismatches += ", ";
                    mismatches += label;
                    mismatches += ": expected " + std::to_string(full_bag[i])
                               + " got " + std::to_string(rack_counts[i]);
                }
            }
            if (!mismatches.empty())
                rack_warning = "Rack mismatch: " + mismatches;
        }
    }

    // Step 8: Dictionary validation — check all words against KWG
    std::string invalid_words_json;  // JSON array string for UI
    {
        std::string lex = gemini_lexicon;
        if (lex.empty()) lex = "NWL23";  // default
        bool dict_ok = ensure_kwg_loaded(lex);
        if (!dict_ok && lex != "NWL23") dict_ok = ensure_kwg_loaded("NWL23");
        if (!dict_ok && lex != "CSW21") dict_ok = ensure_kwg_loaded("CSW21");

        if (dict_ok) {
            auto all_words = extract_words(dr.cells);
            struct InvalidWord {
                std::string word;
                std::string position;  // e.g. "row 1" or "col G"
                std::vector<std::pair<int,int>> cells;
                bool horizontal;
            };
            std::vector<InvalidWord> invalid;
            for (const auto& bw : all_words) {
                if (!g_kwg.is_valid(bw.word)) {
                    InvalidWord iw;
                    iw.word = bw.word;
                    iw.cells = bw.cells;
                    iw.horizontal = bw.horizontal;
                    if (bw.horizontal)
                        iw.position = "row " + std::to_string(bw.cells[0].first + 1);
                    else
                        iw.position = "col " + std::string(1, static_cast<char>('A' + bw.cells[0].second));
                    invalid.push_back(std::move(iw));
                }
            }

            if (!invalid.empty()) {
                // Identify suspect cells — cells in invalid words but not in
                // any valid word
                std::set<std::pair<int,int>> valid_cells, invalid_cells;
                for (const auto& bw : all_words) {
                    bool is_valid = g_kwg.is_valid(bw.word);
                    for (const auto& p : bw.cells) {
                        if (is_valid) valid_cells.insert(p);
                        else invalid_cells.insert(p);
                    }
                }
                std::vector<std::pair<int,int>> suspects;
                for (const auto& p : invalid_cells) {
                    if (valid_cells.find(p) == valid_cells.end())
                        suspects.push_back(p);
                }

                // Build status message listing invalid words
                std::string iw_list;
                for (const auto& iw : invalid) {
                    if (!iw_list.empty()) iw_list += ", ";
                    iw_list += iw.word + " (" + iw.position + ")";
                }
                {
                    std::string msg = "{\"status\":\"Invalid words ["
                        + json_escape(g_kwg_lexicon) + "]: "
                        + json_escape(iw_list) + "\"}\n";
                    sink.write(msg.data(), msg.size());
                }

                // Re-query suspect cells via Gemini
                if (!suspects.empty() && have_opencv) {
                    int bx, by, cell_sz;
                    if (parse_board_rect_from_log(opencv_dr.log, bx, by, cell_sz)) {
                        cv::Mat raw_d(1, static_cast<int>(buf.size()), CV_8UC1,
                                    const_cast<uint8_t*>(buf.data()));
                        cv::Mat img_d = cv::imdecode(raw_d, cv::IMREAD_COLOR);

                        if (!img_d.empty()) {
                            // Build crop status for UI debug
                            std::string crop_status =
                                "{\"status\":\"Re-querying "
                                + std::to_string(suspects.size())
                                + " suspect cell(s) from invalid words...\""
                                + ",\"crops\":[";

                            // Build prompt listing current letters and the
                            // invalid words they participate in
                            std::string dict_prompt =
                                "These are cropped Scrabble tile images. "
                                "The current OCR reads formed words that are "
                                "NOT in the " + g_kwg_lexicon + " dictionary. "
                                "Invalid words: " + iw_list + ". "
                                "For each cropped tile, carefully re-identify "
                                "the letter. Pay close attention to similar "
                                "letters (H/N/I, O/Q/D, E/F, S/Z, etc). "
                                "Regular tiles are beige with a subscript "
                                "(return UPPERCASE). Blank tiles are purple "
                                "circles (return lowercase). "
                                "Return ONLY a JSON array of single letters. "
                                "Example: [\\\"I\\\", \\\"H\\\"]";

                            std::string dict_payload =
                                "{\"contents\":[{\"parts\":["
                                "{\"text\":\"" + dict_prompt + "\"}";

                            int pad = cell_sz / 8;
                            for (size_t si = 0; si < suspects.size(); si++) {
                                const auto& sp = suspects[si];
                                int cx = std::max(0, bx + sp.second * cell_sz - pad);
                                int cy = std::max(0, by + sp.first * cell_sz - pad);
                                int cw = std::min(cell_sz + 2*pad, img_d.cols - cx);
                                int ch_ = std::min(cell_sz + 2*pad, img_d.rows - cy);
                                cv::Mat cell_img = img_d(cv::Rect(cx, cy, cw, ch_));
                                std::vector<uint8_t> png_buf;
                                cv::imencode(".png", cell_img, png_buf);
                                std::string b64_crop = base64_encode(png_buf);
                                dict_payload +=
                                    ",{\"inlineData\":{\"mimeType\":\"image/png\","
                                    "\"data\":\"" + b64_crop + "\"}}";
                                if (si > 0) crop_status += ",";
                                char cur_ltr = static_cast<char>(std::toupper(
                                    static_cast<unsigned char>(
                                        dr.cells[sp.first][sp.second].letter)));
                                crop_status += "{\"pos\":\""
                                    + std::string(1, static_cast<char>('A' + sp.second))
                                    + std::to_string(sp.first + 1)
                                    + "\",\"cur\":\""
                                    + std::string(1, cur_ltr)
                                    + "\",\"img\":\"data:image/png;base64,"
                                    + b64_crop + "\"}";
                            }
                            dict_payload += "]}]}";
                            crop_status += "]}\n";
                            sink.write(crop_status.data(), crop_status.size());

                            // Call Gemini
                            char tmpd[] = "/tmp/gemini_d_XXXXXX";
                            int fdd = mkstemp(tmpd);
                            if (fdd >= 0) {
                                ssize_t wd = 0;
                                size_t td = dict_payload.size();
                                while (wd < static_cast<ssize_t>(td)) {
                                    ssize_t nn = write(fdd,
                                        dict_payload.data() + wd, td - wd);
                                    if (nn <= 0) break;
                                    wd += nn;
                                }
                                close(fdd);

                                std::string cmdd =
                                    "curl -s --max-time 30 -X POST "
                                    "-H 'Content-Type: application/json' "
                                    "-d @" + std::string(tmpd)
                                    + " '" + url + "'";
                                FILE* piped = popen(cmdd.c_str(), "r");
                                if (piped) {
                                    std::string respd;
                                    char chd[8192];
                                    size_t nnd;
                                    while ((nnd = fread(chd, 1, sizeof(chd),
                                                        piped)) > 0)
                                        respd.append(chd, nnd);
                                    pclose(piped);

                                    std::string textd =
                                        extract_gemini_text(respd);
                                    gemini_log("dict_requery", dict_prompt,
                                        textd.empty() ? respd : textd);
                                    if (!textd.empty()) {
                                        std::string sd = textd;
                                        auto f1 = sd.find("```");
                                        if (f1 != std::string::npos) {
                                            auto nl = sd.find('\n', f1);
                                            if (nl != std::string::npos)
                                                sd = sd.substr(nl + 1);
                                            auto f2 = sd.rfind("```");
                                            if (f2 != std::string::npos)
                                                sd = sd.substr(0, f2);
                                        }
                                        size_t di = 0, si2 = 0;
                                        int corrections = 0;
                                        std::string corr_detail;
                                        while (di < sd.size() && sd[di] != '[')
                                            di++;
                                        di++;
                                        while (di < sd.size()
                                               && si2 < suspects.size()) {
                                            while (di < sd.size()
                                                   && sd[di] != '"'
                                                   && sd[di] != ']')
                                                di++;
                                            if (di >= sd.size()
                                                || sd[di] == ']')
                                                break;
                                            di++;
                                            if (di < sd.size()) {
                                                char ltr = sd[di];
                                                auto& cell = dr.cells
                                                    [suspects[si2].first]
                                                    [suspects[si2].second];
                                                if (ltr != cell.letter) {
                                                    if (!corr_detail.empty())
                                                        corr_detail += ", ";
                                                    corr_detail += std::string(1,
                                                        static_cast<char>('A' + suspects[si2].second))
                                                        + std::to_string(suspects[si2].first + 1)
                                                        + ": "
                                                        + static_cast<char>(std::toupper(
                                                            static_cast<unsigned char>(cell.letter)))
                                                        + " -> "
                                                        + static_cast<char>(std::toupper(
                                                            static_cast<unsigned char>(ltr)));
                                                    cell.letter = ltr;
                                                    cell.confidence = 0.85f;
                                                    cell.is_blank =
                                                        (ltr >= 'a' && ltr <= 'z');
                                                    corrections++;
                                                }
                                                si2++;
                                                di++;
                                                if (di < sd.size()
                                                    && sd[di] == '"')
                                                    di++;
                                            }
                                        }
                                        if (corrections > 0) {
                                            std::string vmsg =
                                                "{\"status\":\"Dict corrections: "
                                                + json_escape(corr_detail)
                                                + "\"}\n";
                                            sink.write(vmsg.data(),
                                                       vmsg.size());
                                        }
                                    }
                                }
                                unlink(tmpd);
                            }
                        }
                    }
                }

                // Rebuild CGP after any corrections
                dr.cgp = cells_to_cgp(dr.cells) + " "
                       + (gemini_rack.empty() ? "" : gemini_rack) + "/ "
                       + std::to_string(gemini_score1) + " "
                       + std::to_string(gemini_score2);
                if (!lex_str.empty())
                    dr.cgp += " lex " + lex_str + ";";

                // Re-validate after corrections for final report
                auto final_words = extract_words(dr.cells);
                invalid.clear();
                for (const auto& bw : final_words) {
                    if (!g_kwg.is_valid(bw.word)) {
                        InvalidWord iw;
                        iw.word = bw.word;
                        iw.cells = bw.cells;
                        iw.horizontal = bw.horizontal;
                        if (bw.horizontal)
                            iw.position = "row " + std::to_string(bw.cells[0].first + 1);
                        else
                            iw.position = "col " + std::string(1, static_cast<char>('A' + bw.cells[0].second));
                        invalid.push_back(std::move(iw));
                    }
                }
            }

            // Build JSON for UI display
            if (!invalid.empty()) {
                invalid_words_json = "[";
                for (size_t i = 0; i < invalid.size(); i++) {
                    if (i > 0) invalid_words_json += ",";
                    invalid_words_json += "{\"word\":\""
                        + json_escape(invalid[i].word)
                        + "\",\"pos\":\""
                        + json_escape(invalid[i].position) + "\"}";
                }
                invalid_words_json += "]";
            }
        }
    }

    std::string final_json = make_json_response(dr);
    // Inject extra fields before the closing }
    std::string extra;
    if (!gemini_bag.empty())
        extra += ",\"bag\":\"" + json_escape(gemini_bag) + "\"";
    if (!rack_warning.empty())
        extra += ",\"rack_warning\":\"" + json_escape(rack_warning) + "\"";
    if (!invalid_words_json.empty())
        extra += ",\"invalid_words\":" + invalid_words_json;
    // Include occupancy grid so UI can show it
    if (have_color || have_opencv) {
        extra += ",\"occupancy\":[";
        for (int r = 0; r < 15; r++) {
            if (r > 0) extra += ",";
            extra += "[";
            for (int c = 0; c < 15; c++) {
                if (c > 0) extra += ",";
                extra += get_occupied(r, c) ? "1" : "0";
            }
            extra += "]";
        }
        extra += "]";
    }
    if (!extra.empty())
        final_json.insert(final_json.size() - 1, extra); // insert before }
    final_json += "\n";
    sink.write(final_json.data(), final_json.size());
    sink.done();
}

// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    load_dotenv();

    // CLI test mode
    if (argc > 1 && std::string(argv[1]) == "--test") {
        return run_tests_cli();
    }

    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(HTML, "text/html");
    });

    svr.Post("/analyze", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("image")) {
            res.status = 400;
            res.set_content(R"({"error":"no image field"})", "application/json");
            return;
        }
        const auto& file = req.get_file_value("image");
        auto buf = std::make_shared<std::vector<uint8_t>>(
            file.content.begin(), file.content.end());

        // Store for test case saving
        {
            std::lock_guard<std::mutex> lk(g_last_image_mutex);
            g_last_image = buf;
        }

        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_chunked_content_provider(
            "application/x-ndjson",
            [buf](size_t /*offset*/, httplib::DataSink& sink) {
                stream_analyze(*buf, sink);
                return false;
            });
    });

    svr.Post("/analyze-gemini", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("image")) {
            res.status = 400;
            res.set_content(R"({"error":"no image field"})", "application/json");
            return;
        }
        const auto& file = req.get_file_value("image");
        auto buf = std::make_shared<std::vector<uint8_t>>(
            file.content.begin(), file.content.end());

        // Store for test case saving
        {
            std::lock_guard<std::mutex> lk(g_last_image_mutex);
            g_last_image = buf;
        }

        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_chunked_content_provider(
            "application/x-ndjson",
            [buf](size_t /*offset*/, httplib::DataSink& sink) {
                stream_analyze_gemini(*buf, sink);
                return false;
            });
    });

    svr.Post("/fetch-url", [](const httplib::Request& req, httplib::Response& res) {
        std::string url = json_extract_string(req.body, "url");
        if (url.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"missing url field"})", "application/json");
            return;
        }

        if (!is_safe_url(url)) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid or unsafe URL (must be https)\"}",
                            "application/json");
            return;
        }

        // Download via curl
        std::string cmd = "curl -sL --max-time 10 '" + url + "'";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            res.status = 500;
            res.set_content(R"({"error":"failed to execute curl"})", "application/json");
            return;
        }

        std::vector<uint8_t> buf;
        uint8_t chunk[8192];
        size_t n;
        while ((n = fread(chunk, 1, sizeof(chunk), pipe)) > 0) {
            buf.insert(buf.end(), chunk, chunk + n);
        }
        int rc = pclose(pipe);

        if (rc != 0 || buf.empty()) {
            res.status = 502;
            res.set_content(R"({"error":"failed to download URL"})", "application/json");
            return;
        }

        auto buf_ptr = std::make_shared<std::vector<uint8_t>>(std::move(buf));
        bool use_gemini = json_extract_string(req.body, "method") == "gemini";

        // Store for test case saving
        {
            std::lock_guard<std::mutex> lk(g_last_image_mutex);
            g_last_image = buf_ptr;
        }

        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_chunked_content_provider(
            "application/x-ndjson",
            [buf_ptr, use_gemini](size_t /*offset*/, httplib::DataSink& sink) {
                if (use_gemini)
                    stream_analyze_gemini(*buf_ptr, sink);
                else
                    stream_analyze(*buf_ptr, sink);
                return false;
            });
    });

    svr.Post("/save-test", [](const httplib::Request& req, httplib::Response& res) {
        std::string cgp = json_extract_string(req.body, "cgp");
        if (cgp.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"missing cgp field"})", "application/json");
            return;
        }

        // Check we have an image
        std::shared_ptr<std::vector<uint8_t>> img;
        {
            std::lock_guard<std::mutex> lk(g_last_image_mutex);
            img = g_last_image;
        }
        if (!img || img->empty()) {
            res.status = 400;
            res.set_content(R"({"error":"no image loaded - upload or fetch an image first"})",
                            "application/json");
            return;
        }

        // Create testdata directory
        fs::create_directories("testdata");

        // Generate filename from timestamp
        time_t now = time(nullptr);
        struct tm tm;
        localtime_r(&now, &tm);
        char name[64];
        snprintf(name, sizeof(name), "test_%04d%02d%02d_%02d%02d%02d",
                 tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                 tm.tm_hour, tm.tm_min, tm.tm_sec);

        // Save image
        std::string img_path = std::string("testdata/") + name + ".png";
        {
            std::ofstream ofs(img_path, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(img->data()),
                      static_cast<std::streamsize>(img->size()));
        }

        // Save CGP
        std::string cgp_path = std::string("testdata/") + name + ".cgp";
        {
            std::ofstream ofs(cgp_path);
            ofs << cgp;
        }

        res.set_content(std::string("{\"saved\":\"") + name + "\"}",
                        "application/json");
    });

    svr.Get("/run-tests", [](const httplib::Request&, httplib::Response& res) {
        if (!fs::exists("testdata")) {
            res.set_content("[]", "application/json");
            return;
        }

        std::vector<fs::directory_entry> entries;
        for (const auto& e : fs::directory_iterator("testdata"))
            if (e.path().extension() == ".cgp") entries.push_back(e);
        std::sort(entries.begin(), entries.end());

        std::string json = "[";
        bool first = true;

        for (const auto& entry : entries) {
            std::string name = entry.path().stem().string();
            std::string img_path = "testdata/" + name + ".png";
            if (!fs::exists(img_path)) continue;

            std::string expected_cgp;
            {
                std::ifstream ifs(entry.path());
                std::getline(ifs, expected_cgp);
            }

            std::vector<uint8_t> img_data;
            {
                std::ifstream ifs(img_path, std::ios::binary);
                img_data.assign(std::istreambuf_iterator<char>(ifs),
                                std::istreambuf_iterator<char>());
            }

            DebugResult dr;
            try {
                dr = process_board_image_debug(img_data);
            } catch (const std::exception& ex) {
                // If OCR fails, report as all-wrong
                if (!first) json += ",";
                first = false;
                json += "{\"name\":\"" + json_escape(name) + "\"";
                json += ",\"total\":0,\"correct\":0,\"wrong\":0";
                json += ",\"error\":\"" + json_escape(ex.what()) + "\"";
                json += ",\"diffs\":[]}";
                continue;
            }

            auto expected = parse_cgp_board(expected_cgp);
            auto got = parse_cgp_board(dr.cgp);

            int case_total = 0, case_correct = 0;
            std::string diffs = "[";
            bool dfirst = true;

            for (int r = 0; r < 15; r++) {
                for (int c = 0; c < 15; c++) {
                    if (expected[r][c] != 0 || got[r][c] != 0) {
                        case_total++;
                        if (expected[r][c] == got[r][c]) {
                            case_correct++;
                        } else {
                            if (!dfirst) diffs += ",";
                            dfirst = false;
                            std::string pos;
                            pos += static_cast<char>('A' + c);
                            pos += std::to_string(r + 1);
                            std::string exp_s = expected[r][c]
                                ? std::string(1, expected[r][c]) : ".";
                            std::string got_s = got[r][c]
                                ? std::string(1, got[r][c]) : ".";
                            diffs += "{\"pos\":\"" + pos
                                   + "\",\"exp\":\"" + exp_s
                                   + "\",\"got\":\"" + got_s + "\"}";
                        }
                    }
                }
            }
            diffs += "]";

            if (!first) json += ",";
            first = false;
            json += "{\"name\":\"" + json_escape(name) + "\"";
            json += ",\"total\":" + std::to_string(case_total);
            json += ",\"correct\":" + std::to_string(case_correct);
            json += ",\"wrong\":" + std::to_string(case_total - case_correct);
            json += ",\"diffs\":" + diffs + "}";
        }

        json += "]";
        res.set_content(json, "application/json");
    });

    const char* port_env = std::getenv("PORT");
    int port = port_env ? std::atoi(port_env) : 8080;

    std::cout << "CGP test bench -> http://localhost:" << port << "\n";

    if (!svr.listen("127.0.0.1", port)) {
        std::cerr << "Failed to bind to port " << port << "\n";
        return 1;
    }
}
