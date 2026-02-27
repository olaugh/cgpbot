#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
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
        if (idx == 0 || idx >= nodes_.size()) return false;

        int lidx = 0;
        uint32_t node = nodes_[idx];
        int wlen = static_cast<int>(word.size());
        while (true) {
            if (lidx > wlen - 1) return false;
            uint32_t ml = static_cast<uint32_t>(word[lidx] - 'A' + 1);
            if (ml < 1 || ml > 26) return false; // not A-Z
            if (tile(node) == ml) {
                if (lidx == wlen - 1) return accepts(node);
                idx = arc_index(node);
                if (idx == 0 || idx >= nodes_.size()) return false;
                node = nodes_[idx];
                lidx++;
            } else {
                if (is_end(node)) return false;
                idx++;
                if (idx >= nodes_.size()) return false;
                node = nodes_[idx];
            }
        }
    }
};

// Global KWG checker, loaded once at startup or on first use.
static KWGChecker g_kwg;
static std::string g_kwg_lexicon;  // which lexicon is loaded

static bool load_kwg_from_path(KWGChecker& kwg, const std::string& lexicon) {
    std::string path = "magpie/data/lexica/" + lexicon + ".kwg";
    if (!fs::exists(path)) {
        path = "magpie/testdata/lexica/" + lexicon + ".kwg";
    }
    if (!fs::exists(path)) return false;
    return kwg.load(path);
}

static bool ensure_kwg_loaded(const std::string& lexicon) {
    if (g_kwg.loaded() && g_kwg_lexicon == lexicon) return true;
    if (load_kwg_from_path(g_kwg, lexicon)) {
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
// Lexicon inference: if Gemini didn't detect a lexicon from the screenshot,
// check board words against NWL23 and CSW24. If any word is valid in CSW24
// but not NWL23, the game is likely using CSW24.
// ---------------------------------------------------------------------------
static std::string infer_lexicon(const CellResult board[15][15],
                                 std::vector<std::string>* csw_only_out = nullptr) {
    KWGChecker nwl, csw;
    if (!load_kwg_from_path(nwl, "NWL23")) return "NWL23";
    if (!load_kwg_from_path(csw, "CSW24")) return "NWL23";

    auto words = extract_words(board);
    std::vector<std::string> csw_only;
    for (const auto& bw : words) {
        if (bw.word.find('?') != std::string::npos) continue;
        if (csw.is_valid(bw.word) && !nwl.is_valid(bw.word))
            csw_only.push_back(bw.word);
    }
    if (csw_only_out) *csw_only_out = csw_only;
    if (!csw_only.empty()) return "CSW24";
    return "NWL23";
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

// Forward declaration — defined in gemini_parse.h, included later.
static inline std::string extract_gemini_text(const std::string& json);

// ---------------------------------------------------------------------------
// Call Gemini API with automatic retry on empty response.
// Writes payload to temp file, calls curl, reads response, extracts text.
// Retries up to max_retries times (with 1s delay) if extract_gemini_text
// returns empty.
// ---------------------------------------------------------------------------
struct GeminiCallResult {
    std::string raw_response;
    std::string text;       // extract_gemini_text result
    int attempts;           // how many attempts were made
    bool cached;            // true if served from cache
};

// ---------------------------------------------------------------------------
// Gemini response cache — keyed on exact payload string.
// Cache files stored in .gemini_cache/ directory.
// ---------------------------------------------------------------------------
static const std::string GEMINI_CACHE_DIR = ".gemini_cache";

static std::string cache_key(const std::string& payload) {
    std::size_t h = std::hash<std::string>{}(payload);
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%016zx", h);
    return std::string(buf);
}

static bool gemini_cache_lookup(const std::string& payload, std::string& raw_response) {
    std::string path = GEMINI_CACHE_DIR + "/" + cache_key(payload);
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    raw_response.assign(std::istreambuf_iterator<char>(f),
                        std::istreambuf_iterator<char>());
    return !raw_response.empty();
}

static void gemini_cache_store(const std::string& payload, const std::string& raw_response) {
    fs::create_directories(GEMINI_CACHE_DIR);
    std::string path = GEMINI_CACHE_DIR + "/" + cache_key(payload);
    std::ofstream f(path, std::ios::binary);
    if (f) f.write(raw_response.data(), raw_response.size());
}

static GeminiCallResult call_gemini(const std::string& url,
                                    const std::string& payload,
                                    const std::string& log_label,
                                    const std::string& log_prompt,
                                    int timeout_sec = 30,
                                    int max_retries = 1) {
    GeminiCallResult result;
    result.attempts = 0;
    result.cached = false;

    // Check cache first
    if (gemini_cache_lookup(payload, result.raw_response)) {
        result.text = extract_gemini_text(result.raw_response);
        if (!result.text.empty()) {
            result.cached = true;
            result.attempts = 0;
            gemini_log(log_label + " [CACHED]", log_prompt, result.text);
            return result;
        }
    }

    for (int attempt = 0; attempt <= max_retries; attempt++) {
        result.attempts = attempt + 1;

        if (attempt > 0) {
            // Brief delay before retry
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        // Write payload to temp file
        char tmppath[] = "/tmp/gemini_call_XXXXXX";
        int fd = mkstemp(tmppath);
        if (fd < 0) continue;

        ssize_t written = 0;
        size_t total = payload.size();
        while (written < static_cast<ssize_t>(total)) {
            ssize_t n = write(fd, payload.data() + written, total - written);
            if (n <= 0) break;
            written += n;
        }
        close(fd);

        std::string cmd = "curl -s --max-time " + std::to_string(timeout_sec)
            + " -X POST -H 'Content-Type: application/json' -d @"
            + std::string(tmppath) + " '" + url + "'";

        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            unlink(tmppath);
            continue;
        }

        result.raw_response.clear();
        char chunk[8192];
        size_t n;
        while ((n = fread(chunk, 1, sizeof(chunk), pipe)) > 0)
            result.raw_response.append(chunk, n);
        pclose(pipe);
        unlink(tmppath);

        result.text = extract_gemini_text(result.raw_response);

        std::string suffix = (attempt > 0)
            ? " (retry " + std::to_string(attempt) + ")" : "";
        gemini_log(log_label + suffix, log_prompt,
            result.text.empty() ? result.raw_response : result.text);

        if (!result.text.empty()) {
            // Store successful response in cache
            gemini_cache_store(payload, result.raw_response);
            break;
        }
    }

    return result;
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

// Forward declarations for run_tests_cli.
static bool parse_board_rect_from_log(const std::string& log,
                                       int& bx, int& by, int& cell_sz,
                                       int* board_w = nullptr,
                                       int* board_h = nullptr);
static bool detect_board_mode(const std::vector<uint8_t>& image_data,
                               int bx, int by, int cell_sz);
// ---------------------------------------------------------------------------
// Run all test cases from testdata/ directory (CLI mode).
// ---------------------------------------------------------------------------
static int run_tests_cli() {
    if (!fs::exists("testdata")) {
        std::cout << "No testdata/ directory found.\n";
        return 1;
    }

    int total_cases = 0, passed_cases = 0;
    int total_occ_expected = 0, total_occ_correct = 0, total_occ_false_pos = 0;

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

        int occ_expected = 0, occ_correct = 0, occ_false_pos = 0;
        std::string fp_cells, fn_cells;
        for (int r = 0; r < 15; r++) {
            for (int c = 0; c < 15; c++) {
                bool exp_occ = (expected[r][c] != 0);
                bool got_occ = (dr.cells[r][c].letter != 0);
                if (exp_occ) {
                    occ_expected++;
                    if (got_occ) occ_correct++;
                    else {
                        if (!fn_cells.empty()) fn_cells += " ";
                        fn_cells += static_cast<char>('A' + c);
                        fn_cells += std::to_string(r + 1);
                    }
                } else if (got_occ) {
                    occ_false_pos++;
                    if (!fp_cells.empty()) fp_cells += " ";
                    fp_cells += static_cast<char>('A' + c);
                    fp_cells += std::to_string(r + 1);
                }
            }
        }

        total_cases++;
        total_occ_expected += occ_expected;
        total_occ_correct += occ_correct;
        total_occ_false_pos += occ_false_pos;

        double occ_pct = occ_expected > 0 ? (100.0 * occ_correct / occ_expected) : 100.0;
        int fn = occ_expected - occ_correct;
        std::printf("%-20s occ %3d/%3d (%.0f%%) +%d fp -%d fn",
                    name.c_str(), occ_correct, occ_expected, occ_pct,
                    occ_false_pos, fn);
        if (!fp_cells.empty()) std::printf("  FP:[%s]", fp_cells.c_str());
        if (!fn_cells.empty()) std::printf("  FN:[%s]", fn_cells.c_str());
        std::printf("\n");

        if (occ_correct == occ_expected && occ_false_pos == 0) passed_cases++;
    }

    if (total_cases == 0) {
        std::cout << "No test cases found in testdata/.\n";
        return 1;
    }

    double occ_overall = total_occ_expected > 0 ? (100.0 * total_occ_correct / total_occ_expected) : 100.0;
    std::printf("\n%d test case(s), %d perfect occupancy\n", total_cases, passed_cases);
    std::printf("Occupancy: %d/%d (%.1f%%) +%d fp -%d fn\n",
                total_occ_correct, total_occ_expected, occ_overall,
                total_occ_false_pos, total_occ_expected - total_occ_correct);

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
.layout{display:grid;grid-template-columns:180px 1fr 1fr;gap:20px;max-width:1600px;margin:0 auto}
.sidebar{display:flex;flex-direction:column;gap:12px}
.sidebar-panel{background:#16213e;border-radius:12px;padding:14px;border:1px solid #2a2a4a}
.sidebar-panel h2{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:#888;margin-bottom:10px}
.test-list{list-style:none}
.test-list li{padding:4px 6px;border-radius:4px;cursor:pointer;font-size:.75rem;color:#aaa;display:flex;align-items:center;gap:6px}
.test-list li:hover{background:#1e2d50;color:#fff}
.test-list li.active{background:#1e2d50;color:#58a6ff}
.test-list li.running{background:#1a1630;color:#aaf}
.test-list .dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;background:#555}
.test-list li.pass .dot{background:#4c4}
.test-list li.fail .dot{background:#f44}
.test-list li.running .dot{width:auto;height:auto;background:none;border-radius:0;color:#88f;font-size:.85rem;line-height:1}
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
.cell.diff-wrong{outline:2px solid #f44;outline-offset:-2px;z-index:1}
.cell.diff-wrong .diff-exp{position:absolute;top:1px;left:2px;font-size:.45rem;color:#f88;font-weight:700}
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
  <div class="sidebar">
    <div class="sidebar-panel">
      <h2>Test Cases</h2>
      <ul class="test-list" id="test-list"></ul>
    </div>
    <div class="sidebar-panel">
      <h2>Last Eval</h2>
      <div id="eval-summary-bar">
        <div id="eval-summary-acc" style="font-size:.85rem;font-weight:600;color:#4c4;margin-bottom:2px"></div>
        <div id="eval-summary-time" style="font-size:.7rem;color:#666;margin-bottom:6px"></div>
        <a href="/eval" target="_blank" style="font-size:.72rem;color:#58a6ff;text-decoration:none">details &rarr;</a>
        &nbsp;<button class="btn" style="font-size:.65rem;padding:2px 6px" onclick="evalAllGemini()">re-run</button>
      </div>
      <div id="eval-summary-none" style="font-size:.72rem;color:#555">
        No eval yet.<br>
        <button class="btn" style="font-size:.65rem;padding:2px 6px;margin-top:6px" onclick="evalAllGemini()">Run eval</button>
      </div>
    </div>
  </div>
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
      <div id="transposed-area" style="display:none;margin-top:16px">
        <h3 style="font-size:.85rem;color:#aaa;margin-bottom:8px">Transposed Board OCR (columns as rows)</h3>
        <img id="transposed-img" style="max-width:100%;border-radius:6px;image-rendering:pixelated">
        <div id="transposed-disagree" style="margin-top:8px;font-size:.75rem;line-height:2"></div>
      </div>
      <div id="trail-area" style="display:none;margin-top:16px">
        <h3 style="font-size:.85rem;color:#aaa;margin-bottom:6px">OCR Trail</h3>
        <div id="trail-raw-cgp" style="font-family:monospace;font-size:.7rem;color:#888;word-break:break-all;margin-bottom:8px;background:#111;padding:6px;border-radius:4px"></div>
        <div id="trail-table"></div>
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
        <button class="btn" onclick="evalAllGemini()">Eval All</button>
        <button class="btn" id="eval-stop-btn" onclick="evalStop()" style="display:none;background:#a33">Stop Eval</button>
      </div>
      <textarea id="cgp-output" rows="3" spellcheck="false"></textarea>
      <div id="woogles-area" style="display:none;margin-top:8px;padding:8px 12px;background:#0f1f0f;border:1px solid #2a3a2a;border-radius:6px;font-size:.82rem"></div>
    </div>
    <div class="panel" style="margin-top:20px">
      <h2>Board</h2>
      <div id="board-area"></div>
      <div id="diff-summary" style="display:none;margin-top:10px;font-size:.75rem;font-family:'SF Mono','Fira Code',monospace"></div>
    </div>
    <div id="eval-panel" class="panel" style="margin-top:20px;display:none">
      <h2>Gemini Eval</h2>
      <div id="eval-results"></div>
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

// Test case tracking
let currentTestCase=null;   // name of currently loaded test case, or null
let expectedBoard=null;     // 15x15 array of expected letters (from testdata CGP), or null

// Braille spinner for eval progress
const BRAILLE='\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f';
let _spinInterval=null,_spinFrame=0;
function startSpinner(name){
  stopSpinner();
  _spinFrame=0;
  for(const li of document.querySelectorAll('#test-list li')){
    li.classList.toggle('running',li.dataset.name===name);
    if(li.dataset.name===name){const d=li.querySelector('.dot');if(d)d.textContent=BRAILLE[0];}
  }
  _spinInterval=setInterval(()=>{
    _spinFrame=(_spinFrame+1)%BRAILLE.length;
    const li=document.querySelector('#test-list li.running');
    if(li){const d=li.querySelector('.dot');if(d)d.textContent=BRAILLE[_spinFrame];}
  },100);
}
function stopSpinner(){
  if(_spinInterval){clearInterval(_spinInterval);_spinInterval=null;}
  for(const li of document.querySelectorAll('#test-list li.running')){
    li.classList.remove('running');
    const d=li.querySelector('.dot');if(d)d.textContent='';
  }
}

// Parse CGP board string -> 15x15 array of chars ('' = empty)
function parseCGPBoard(cgp){
  const board=Array.from({length:15},()=>Array(15).fill(''));
  const boardStr=cgp.split(' ')[0];
  const rows=boardStr.split('/');
  for(let r=0;r<Math.min(rows.length,15);r++){
    let c=0,i=0;
    while(i<rows[r].length&&c<15){
      const ch=rows[r][i];
      if(ch>='0'&&ch<='9'){
        let n=0;
        while(i<rows[r].length&&rows[r][i]>='0'&&rows[r][i]<='9')n=n*10+parseInt(rows[r][i++]);
        c+=n;
      }else{board[r][c++]=ch;i++;}
    }
  }
  return board;
}
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
            const lbl=crop.initial&&crop.initial!==crop.cur
              ?`${crop.initial}<span style="color:#fa8">\u2192${crop.cur}</span>`
              :crop.cur;
            d.innerHTML=`<img src="${crop.img}" style="width:48px;height:48px;image-rendering:pixelated;border-radius:4px"><div style="font-size:.7rem;color:#ccc;margin-top:4px">${crop.pos}: ${lbl}</div>`;
            cc.appendChild(d);
          }
          ca.style.display='block';
        }
        if(data.transposed_image){
          document.getElementById('transposed-img').src=data.transposed_image;
          document.getElementById('transposed-area').style.display='block';
        }
        if(data.transposed_disagree){
          const td=document.getElementById('transposed-disagree');
          if(data.transposed_disagree.length===0){
            td.innerHTML='<span style="color:#8f8">&#10003; Transposed OCR agrees with main OCR on all occupied cells.</span>';
          }else{
            td.innerHTML='Raw OCR disagreements (main\u2192trans): '+data.transposed_disagree.map(
              x=>`<span style="background:#2a1010;border:1px solid #a33;padding:1px 6px;border-radius:3px;margin:2px;display:inline-block">${x.pos}: <b>${x.orig}</b>&rarr;<b style="color:#f88">${x.trans}</b></span>`
            ).join(' ');
          }
        }
        if(data.raw_main_cgp){
          document.getElementById('trail-raw-cgp').textContent='Raw main OCR: '+data.raw_main_cgp;
          document.getElementById('trail-area').style.display='block';
        }
        if(data.ocr_trail&&data.ocr_trail.length>0){
          const ta=document.getElementById('trail-area');
          ta.style.display='block';
          const tbl=document.getElementById('trail-table');
          let h='<table style="font-size:.72rem;border-collapse:collapse;width:100%">';
          h+='<tr style="color:#888"><th style="text-align:left;padding:2px 6px">Cell</th><th style="padding:2px 6px">Raw Main</th><th style="padding:2px 6px">Trans OCR</th><th style="padding:2px 6px">Final</th><th style="text-align:left;padding:2px 6px">Note</th></tr>';
          for(const e of data.ocr_trail){
            const raw=e.raw||'\u2014';
            const trans=e.trans||'\u2014';
            const fin=e.final||'\u2014';
            let note='corrected',bg='#1a0a1a',nc='#d8a';
            if(e.trans&&e.trans===e.final&&e.raw!==e.final){note='trans helped';bg='#0a1f0a';nc='#8f8';}
            else if(e.raw===e.final&&e.trans&&e.trans!==e.final){note='trans wrong';bg='#1f1a00';nc='#bb8';}
            h+=`<tr style="background:${bg}"><td style="padding:2px 8px;font-weight:bold">${e.pos}</td><td style="text-align:center;padding:2px 8px">${raw}</td><td style="text-align:center;padding:2px 8px">${trans}</td><td style="text-align:center;padding:2px 8px;font-weight:bold">${fin}</td><td style="padding:2px 8px;color:${nc}">${note}</td></tr>`;
          }
          h+='</table>';
          tbl.innerHTML=h;
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
          if(currentTestCase) updateDiffSummary(data.cgp);
          // Clear previous woogles result while lookup runs
          const wa=document.getElementById('woogles-area');
          wa.style.display='none'; wa.innerHTML='';
        }
        if('woogles' in data){
          const w=data.woogles;
          const wa=document.getElementById('woogles-area');
          if(w&&w.game_id){
            const url=`https://woogles.io/game/${w.game_id}?turn=${w.turn}`;
            const ps=(w.players||[]).join(' vs ');
            const lex=w.lexicon?` &middot; ${w.lexicon}`:'';
            const sim=w.similarity!=null?` &middot; sim&nbsp;${w.similarity.toFixed(3)}`:'';
            wa.innerHTML=`<span style="color:#8f8">&#9654;</span>&nbsp;<a href="${url}" target="_blank" style="color:#58a6ff;font-weight:600">${w.game_id} turn&nbsp;${w.turn}</a>`
              +(ps?`&nbsp;&nbsp;<span style="color:#aaa">${ps}</span>`:'')
              +lex+sim;
          }else{
            wa.innerHTML='<span style="color:#666">No Woogles match found.</span>';
          }
          wa.style.display='block';
          if(!document.getElementById('status').textContent.startsWith('Error'))
            status.textContent='Done.';
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
      const expCh=expectedBoard&&expectedBoard[r]&&expectedBoard[r][c]||'';
      const diffWrong=expCh&&(expCh!==ch); // expected has something that differs from got
      const diffCls=diffWrong?' diff-wrong':'';
      if(!ch){
        const p=PREMIUM[r][c];
        const expLabel=diffWrong?`<span class="diff-exp">${expCh.toUpperCase()}</span>`:'';
        h+=`<div class="cell ${PCLS[p]}${selCls}${editCls}${diffCls}" data-r="${r}" data-c="${c}">${expLabel}<span class="lbl">${PLBL[p]}</span></div>`;
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
        const expLabel=diffWrong?`<span class="diff-exp">${expCh.toUpperCase()}</span>`:'';
        h+=`<div class="cell ${cls} ${hasTip}${selCls}${editCls}${diffCls}" data-r="${r}" data-c="${c}">${expLabel}${ch.toUpperCase()}${sub}</div>`;
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

function relTime(ts){
  const d=Math.floor((Date.now()/1000)-ts);
  if(d<60)return 'just now';
  if(d<3600)return Math.round(d/60)+'m ago';
  if(d<86400)return Math.round(d/3600)+'h ago';
  return Math.round(d/86400)+'d ago';
}

let testCaseNames=[];

// --- Load test cases into sidebar list ---
async function loadTestList(){
  try{
    const res=await fetch('/testdata-list');
    const cases=await res.json();
    testCaseNames=cases.map(c=>c.name);
    const list=document.getElementById('test-list');
    list.innerHTML='';
    for(const tc of cases){
      const li=document.createElement('li');
      li.dataset.name=tc.name;
      li.innerHTML=`<span class="dot"></span>${tc.name}`;
      li.onclick=()=>loadTestCase(tc.name);
      list.appendChild(li);
    }
  }catch(e){}
  loadEvalSummary();
}

// --- Load and display compact eval summary ---
async function loadEvalSummary(){
  try{
    const r=await fetch('/eval-summary');
    if(!r.ok)throw new Error();
    const d=await r.json();
    const pct=(d.correct/d.total_cells*100).toFixed(1);
    const wrong=d.total_cells-d.correct;
    document.getElementById('eval-summary-acc').textContent=
      pct+'% ('+d.correct+'/'+d.total_cells+(wrong?' \u2022 '+wrong+' wrong':'')+')';
    document.getElementById('eval-summary-time').textContent=relTime(d.timestamp);
    document.getElementById('eval-summary-bar').style.display='block';
    document.getElementById('eval-summary-none').style.display='none';
    // apply pass/fail dots if case results available
    if(d.cases){
      const byName={};
      for(const c of d.cases)byName[c.name]=c;
      for(const li of document.querySelectorAll('#test-list li')){
        const c=byName[li.dataset.name];
        if(c){li.classList.toggle('pass',c.wrong===0);li.classList.toggle('fail',c.wrong>0);}
      }
    }
  }catch(e){
    document.getElementById('eval-summary-bar').style.display='none';
    document.getElementById('eval-summary-none').style.display='block';
  }
}

// --- Load and submit a test case image ---
async function loadTestCase(name){
  if(!name)return;
  currentTestCase=name;
  expectedBoard=null;
  document.getElementById('diff-summary').style.display='none';
  // highlight active in sidebar
  for(const li of document.querySelectorAll('#test-list li')){
    li.classList.toggle('active',li.dataset.name===name);
  }
  // Fetch expected CGP
  try{
    const r=await fetch('/testdata-cgp/'+encodeURIComponent(name));
    if(r.ok){const t=await r.text();expectedBoard=parseCGPBoard(t.trim());}
  }catch(e){}
  // Fetch image and submit
  try{
    const r=await fetch('/testdata-image/'+encodeURIComponent(name));
    if(!r.ok){status.textContent='Error loading test image.';return;}
    const blob=await r.blob();
    const file=new File([blob],name+'.png',{type:'image/png'});
    preview.src=URL.createObjectURL(blob);
    preview.style.display='block';
    handleFile(file);
  }catch(e){status.textContent='Error: '+e.message;}
}

// --- Update diff summary after a result comes in ---
function updateDiffSummary(gotCGP){
  const ds=document.getElementById('diff-summary');
  if(!expectedBoard||!gotCGP){ds.style.display='none';return;}
  const got=parseCGPBoard(gotCGP);
  const diffs=[];
  for(let r=0;r<15;r++)for(let c=0;c<15;c++){
    const e=expectedBoard[r][c]||'';
    const g=got[r][c]||'';
    if(e||g){
      if(e!==g){
        const pos=String.fromCharCode(65+c)+(r+1);
        diffs.push(`${pos}:exp=${e||'.'} got=${g||'.'}`);
      }
    }
  }
  if(diffs.length===0){
    ds.innerHTML='<span style="color:#4c4">\u2713 Board matches expected ('+currentTestCase+')</span>';
  }else{
    ds.innerHTML=`<span style="color:#f88">\u2717 ${diffs.length} diff(s) vs ${currentTestCase}: </span>`
      +diffs.map(d=>`<span style="color:#f88">${d}</span>`).join(' ');
  }
  ds.style.display='block';
}

// --- Eval helpers ---
function stageAccFromBoards(cgpStr,expBoard){
  if(!cgpStr||!expBoard)return null;
  const got=parseCGPBoard(cgpStr);
  let sc=0,st=0;
  for(let r=0;r<15;r++)for(let c=0;c<15;c++){
    const e=expBoard[r][c]||'',g=got[r][c]||'';
    if(e||g){st++;if(e===g)sc++;}
  }
  return st?((sc/st)*100).toFixed(1):null;
}
function occAccFromBoards(occCGP,expBoard){
  if(!occCGP||!expBoard)return null;
  const occ=parseCGPBoard(occCGP);
  let os=0;
  for(let r=0;r<15;r++)for(let c=0;c<15;c++)
    if(!!(expBoard[r][c])===!!(occ[r][c]))os++;
  return ((os/225)*100).toFixed(1);
}
// Shared braille spinner for eval table cells
let _evalSpinIv=null,_evalSpinF=0;
function startEvalSpinner(){
  if(_evalSpinIv)return;
  _evalSpinF=0;
  _evalSpinIv=setInterval(()=>{
    _evalSpinF=(_evalSpinF+1)%BRAILLE.length;
    for(const el of document.querySelectorAll('.eval-spin'))
      el.textContent=BRAILLE[_evalSpinF];
  },100);
}
function stopEvalSpinner(){
  if(_evalSpinIv){clearInterval(_evalSpinIv);_evalSpinIv=null;}
  for(const el of document.querySelectorAll('.eval-spin'))el.textContent='—';
}
const EVAL_SCOLS=['occ','raw','realigned','trans','retry','wc'];
function evalSetCell(n,col,val){
  const el=document.getElementById('ec-'+n+'-'+col);
  if(el)el.innerHTML=val!=null?(val+'%'):'—';
}

// --- Eval stop ---
let _evalStopped=false;
function evalStop(){_evalStopped=true;document.getElementById('eval-stop-btn').style.display='none';}

// --- Eval all test cases sequentially via Gemini ---
async function evalAllGemini(){
  _evalStopped=false;
  document.getElementById('eval-stop-btn').style.display='';
  const panel=document.getElementById('eval-panel');
  const results=document.getElementById('eval-results');
  panel.style.display='block';

  let cases=[];
  try{const r=await fetch('/testdata-list');cases=await r.json();}
  catch(e){results.innerHTML='<div style="color:#f88">Error fetching test list.</div>';return;}
  if(!cases.length){results.innerHTML='<div style="color:#888">No test cases found.</div>';return;}

  const TH=`<tr><th>Case</th><th>Cells</th><th>Correct</th><th>Wrong</th><th>Board%</th><th style="color:#68a">Occ%</th><th style="color:#68a">Raw%</th><th style="color:#68a">Align%</th><th style="color:#68a">Trans%</th><th style="color:#68a">Retry%</th><th style="color:#68a">WC%</th><th>Exp scores</th><th>Got scores</th><th>&#9654;</th></tr>`;
  results.innerHTML=`<table class="test-results" id="eval-table">${TH}</table><div id="eval-running" style="color:#888;margin-top:6px"></div>`;
  const tbl=document.getElementById('eval-table');
  startEvalSpinner();

  let totalCells=0,totalCorrect=0,totalWrong=0,scoresCorrect=0,scoresTotal=0;
  const caseResults=[];
  const SP=`<span class="eval-spin" style="color:#88f">${BRAILLE[0]}</span>`;

  for(const tc of cases){
    if(_evalStopped){document.getElementById('eval-running').textContent='Stopped.';break;}
    document.getElementById('eval-running').textContent='Running: '+tc.name+'...';
    startSpinner(tc.name);

    let expCGP='',expBoard=null,expScores=null;
    if(tc.has_expected){
      try{const r=await fetch('/testdata-cgp/'+encodeURIComponent(tc.name));expCGP=(await r.text()).trim();}catch(e){}
      if(expCGP){
        expBoard=parseCGPBoard(expCGP);
        const sm=expCGP.match(/\/\s*(\d+)\s+(\d+)/);
        if(sm)expScores=[parseInt(sm[1]),parseInt(sm[2])];
      }
    }

    // Pre-insert row with spinners
    const n=tc.name;
    const stCols=EVAL_SCOLS.map(col=>`<td id="ec-${n}-${col}">${SP}</td>`).join('');
    tbl.insertAdjacentHTML('beforeend',
      `<tr id="eval-row-${n}"><td>${n}</td>`
      +`<td id="ec-${n}-cells">${SP}</td><td id="ec-${n}-correct">${SP}</td>`
      +`<td id="ec-${n}-wrong">${SP}</td><td id="ec-${n}-pct">${SP}</td>`
      +stCols
      +`<td id="ec-${n}-exps" style="font-family:monospace">${expScores?expScores[0]+' '+expScores[1]:'—'}</td>`
      +`<td id="ec-${n}-gots" style="font-family:monospace">${SP}</td>`
      +`<td id="ec-${n}-scok">—</td></tr>`);

    // Stream NDJSON and update cells as stages arrive
    let gotCGP='',stageCGPs={};
    try{
      const ir=await fetch('/testdata-image/'+encodeURIComponent(n));
      if(ir.ok){
        const blob=await ir.blob();
        const form=new FormData();
        form.append('image',new File([blob],n+'.png',{type:'image/png'}));
        form.append('skip_woogles','1');
        const resp=await fetch('/analyze-gemini',{method:'POST',body:form});
        const reader=resp.body.getReader(),dec=new TextDecoder();
        let buf='';
        while(true){
          const {done,value}=await reader.read();
          if(done)break;
          buf+=dec.decode(value,{stream:true});
          let nl;
          while((nl=buf.indexOf('\n'))>=0){
            const line=buf.slice(0,nl).trim();buf=buf.slice(nl+1);
            if(!line)continue;
            try{const d=JSON.parse(line);
              if(d.cgp)gotCGP=d.cgp;
              if(d.raw_main_cgp){stageCGPs.raw=d.raw_main_cgp;evalSetCell(n,'raw',stageAccFromBoards(d.raw_main_cgp,expBoard));}
              if(d.stage==='occupancy'&&d.occupancy_cgp){stageCGPs.occupancy=d.occupancy_cgp;evalSetCell(n,'occ',occAccFromBoards(d.occupancy_cgp,expBoard));}
              if(d.stage&&d.stage_cgp){
                stageCGPs[d.stage]=d.stage_cgp;
                const m={realigned:'realigned',trans:'trans',retry:'retry',wc:'wc'};
                if(m[d.stage])evalSetCell(n,m[d.stage],stageAccFromBoards(d.stage_cgp,expBoard));
              }
            }catch(e){}
          }
        }
      }
    }catch(e){}

    // Final board comparison
    let caseCells=0,caseCorrect=0,caseWrong=0;const diffs=[];
    if(expBoard&&gotCGP){
      const got=parseCGPBoard(gotCGP);
      for(let r=0;r<15;r++)for(let c=0;c<15;c++){
        const e=expBoard[r][c]||'',g=got[r][c]||'';
        if(e||g){caseCells++;if(e===g)caseCorrect++;
          else{caseWrong++;diffs.push(String.fromCharCode(65+c)+(r+1)+':'+e+'\u2192'+(g||'.'));}}
      }
    }
    const pct=caseCells?((caseCorrect/caseCells)*100).toFixed(1)+'%':'—';
    const stageAccs={};
    const ov=occAccFromBoards(stageCGPs.occupancy,expBoard);if(ov)stageAccs.occ=ov;
    ['raw','realigned','trans','retry','wc'].forEach(sn=>{
      const v=stageAccFromBoards(sn==='raw'?stageCGPs.raw:stageCGPs[sn],expBoard);
      if(v!=null)stageAccs[sn]=v;
    });

    // Fill in main accuracy columns
    const wc=caseWrong>0?'color:#f88':'color:#4c4';
    document.getElementById(`ec-${n}-cells`).textContent=caseCells||'—';
    document.getElementById(`ec-${n}-correct`).textContent=caseCorrect||'—';
    document.getElementById(`ec-${n}-wrong`).innerHTML=`<span style="${wc}">${caseWrong||'0'}</span>`;
    document.getElementById(`ec-${n}-pct`).textContent=pct;
    // Clear any remaining spinners in stage cells
    EVAL_SCOLS.forEach(col=>{const el=document.getElementById(`ec-${n}-${col}`);if(el&&el.querySelector('.eval-spin'))el.textContent='—';});
    if(diffs.length)document.getElementById(`eval-row-${n}`)
      .insertAdjacentHTML('afterend',`<tr><td colspan="14" style="color:#f88;font-family:'SF Mono',monospace;font-size:.7rem;padding:2px 8px">&nbsp;&nbsp;${diffs.join('  ')}</td></tr>`);

    // Scores
    let gotSc='—',scOk='—';
    if(expScores){
      scoresTotal++;
      const sm=gotCGP.match(/\/\s*(\d+)\s+(\d+)/);
      const gs=sm?[parseInt(sm[1]),parseInt(sm[2])]:null;
      gotSc=gs?gs[0]+' '+gs[1]:'?';
      if(gs&&gs[0]===expScores[0]&&gs[1]===expScores[1]){scOk='✓';scoresCorrect++;}
      else scOk='<span style="color:#f88">✗</span>';
    }
    document.getElementById(`ec-${n}-gots`).textContent=gotSc;
    document.getElementById(`ec-${n}-scok`).innerHTML=scOk;

    totalCells+=caseCells;totalCorrect+=caseCorrect;totalWrong+=caseWrong;
    const gsm=gotCGP?gotCGP.match(/\/\s*(\d+)\s+(\d+)/):null;
    caseResults.push({name:n,cells:caseCells,correct:caseCorrect,wrong:caseWrong,diffs,
      exp_cgp:caseWrong>0?expCGP.split(' ')[0]:'',got_cgp:caseWrong>0?gotCGP.split(' ')[0]:'',
      exp_scores:expScores?expScores[0]+' '+expScores[1]:null,
      got_scores:gsm?gsm[1]+' '+gsm[2]:null,stage_accs:stageAccs});
    stopSpinner();
    for(const li of document.querySelectorAll('#test-list li'))
      if(li.dataset.name===n){li.classList.toggle('pass',caseWrong===0);li.classList.toggle('fail',caseWrong>0);}
  }

  stopEvalSpinner();
  document.getElementById('eval-stop-btn').style.display='none';
  document.getElementById('eval-running').textContent='';
  const totPct=totalCells?((totalCorrect/totalCells)*100).toFixed(1)+'%':'—';
  tbl.insertAdjacentHTML('beforeend',
    `<tr style="font-weight:bold;border-top:2px solid #555"><td>TOTAL</td><td>${totalCells}</td><td>${totalCorrect}</td><td style="color:${totalWrong?'#f88':'#4c4'}">${totalWrong}</td><td>${totPct}</td><td colspan="6"></td><td colspan="2" style="color:#ccc">${scoresCorrect}/${scoresTotal} scores ✓</td><td></td></tr>`);

  try{await fetch('/eval-save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({total_cells:totalCells,correct:totalCorrect,
      scores_correct:scoresCorrect,scores_total:scoresTotal,cases:caseResults})});}catch(e){}
  loadEvalSummary();
}

loadTestList();
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

// Gemini response parsing (shared with unit tests)
#include "gemini_parse.h"

// ---------------------------------------------------------------------------
// Parse board rectangle from OpenCV pipeline log.
// ---------------------------------------------------------------------------
static bool parse_board_rect_from_log(const std::string& log,
                                       int& bx, int& by, int& cell_sz,
                                       int* board_w,
                                       int* board_h) {
    auto pos = log.find("Final: rect=");
    if (pos == std::string::npos) return false;
    int bw, bh;
    bool ok = sscanf(log.c_str() + pos, "Final: rect=%d,%d %dx%d cell=%d",
                     &bx, &by, &bw, &bh, &cell_sz) == 5;
    if (ok && board_w) *board_w = bw;
    if (ok && board_h) *board_h = bh;
    return ok;
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
// Color-based tile detection using corner-sampled HSV with subpixel cell
// coordinates.  Corners (top-left, top-right, bottom-left — skip bottom-right
// where subscript lives) give the background color without letter
// interference, avoiding mis-detection of heavy letters like W/M.
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
// Word-crop OCR helpers: crop each word run and send to Gemini.
// ---------------------------------------------------------------------------
struct WordRun {
    bool horizontal;
    int r;      // row (horiz), -1 for vert
    int c;      // col (vert),  -1 for horiz
    int start;  // start col (horiz) or start row (vert)
    int end;    // end col/row (inclusive)
    std::string label;  // e.g. "H11H" (horiz row 11, col H) or "VH12" (vert col H, row 12)
    int length() const { return end - start + 1; }
};

static std::vector<WordRun> find_word_runs(
        const std::function<bool(int,int)>& occ) {
    static const char COL[] = "ABCDEFGHIJKLMNO";
    std::vector<WordRun> runs;
    for (int r = 0; r < 15; r++) {
        for (int c = 0; c < 15; ) {
            if (!occ(r, c)) { c++; continue; }
            int c2 = c;
            while (c2 < 15 && occ(r, c2)) c2++;
            WordRun wr; wr.horizontal = true; wr.r = r; wr.c = -1;
            wr.start = c; wr.end = c2 - 1;
            wr.label = "H" + std::to_string(r + 1) + COL[c];
            runs.push_back(wr); c = c2;
        }
    }
    for (int c = 0; c < 15; c++) {
        for (int r = 0; r < 15; ) {
            if (!occ(r, c)) { r++; continue; }
            int r2 = r;
            while (r2 < 15 && occ(r2, c)) r2++;
            WordRun wr; wr.horizontal = false; wr.c = c; wr.r = -1;
            wr.start = r; wr.end = r2 - 1;
            wr.label = "V" + std::string(1, COL[c]) + std::to_string(r + 1);
            runs.push_back(wr); r = r2;
        }
    }
    return runs;
}

// img: original board image (for horizontal words).
// transposed: cell-level transposed image where col C of board becomes row C
//             (for vertical words — letters stay upright, no rotation needed).
// cs_t: uniform cell size (pixels) used in the transposed image.
static std::vector<uint8_t> crop_word_run(
        const cv::Mat& img, const cv::Mat& transposed, int cs_t,
        const WordRun& wr,
        int bx, int by, double cw, double ch) {
    cv::Mat crop;
    if (wr.horizontal) {
        auto cx = [&](int col) { return bx + (int)std::round(col * cw); };
        auto cy = [&](int row) { return by + (int)std::round(row * ch); };
        int x0 = cx(wr.start), y0 = cy(wr.r);
        int pw = cx(wr.end + 1) - x0, ph = cy(wr.r + 1) - y0;
        x0 = std::max(0, x0); y0 = std::max(0, y0);
        if (x0 + pw > img.cols) pw = img.cols - x0;
        if (y0 + ph > img.rows) ph = img.rows - y0;
        if (pw <= 0 || ph <= 0) return {};
        crop = img(cv::Rect(x0, y0, pw, ph)).clone();
    } else {
        // Vertical word: crop from transposed image where board column C is row C.
        // Letters are upright — no rotation needed.
        if (transposed.empty() || cs_t <= 0) return {};
        int x0 = wr.start * cs_t, y0 = wr.c * cs_t;
        int pw = wr.length() * cs_t, ph = cs_t;
        x0 = std::max(0, x0); y0 = std::max(0, y0);
        if (x0 + pw > transposed.cols) pw = transposed.cols - x0;
        if (y0 + ph > transposed.rows) ph = transposed.rows - y0;
        if (pw <= 0 || ph <= 0) return {};
        crop = transposed(cv::Rect(x0, y0, pw, ph)).clone();
    }
    // Normalize to 80px tall per cell for compact, consistent payloads
    const int STD_H = 80;
    if (crop.rows != STD_H) {
        double scale = (double)STD_H / crop.rows;
        cv::resize(crop, crop, cv::Size((int)(crop.cols * scale), STD_H),
                   0, 0, cv::INTER_AREA);
    }
    std::vector<uint8_t> png;
    cv::imencode(".png", crop, png);
    return png;
}

// Parse {"key":"value",...} JSON — string values only.
static std::map<std::string, std::string> parse_json_str_map(
        const std::string& text) {
    std::map<std::string, std::string> result;
    size_t p = text.find('{');
    if (p == std::string::npos) return result;
    p++;
    while (p < text.size()) {
        while (p < text.size() && std::isspace((unsigned char)text[p])) p++;
        if (p >= text.size() || text[p] == '}') break;
        if (text[p] != '"') { p++; continue; }
        p++;
        size_t ks = p;
        while (p < text.size() && text[p] != '"') { if (text[p]=='\\') p++; p++; }
        std::string key = text.substr(ks, p - ks);
        if (p < text.size()) p++;
        while (p < text.size() && text[p] != ':') p++;
        if (p < text.size()) p++;
        while (p < text.size() && std::isspace((unsigned char)text[p])) p++;
        if (p >= text.size()) break;
        if (text[p] == '"') {
            p++;
            size_t vs = p;
            while (p < text.size() && text[p] != '"') { if (text[p]=='\\') p++; p++; }
            result[key] = text.substr(vs, p - vs);
            if (p < text.size()) p++;
        } else {
            while (p < text.size() && text[p] != ',' && text[p] != '}') p++;
        }
        while (p < text.size() && std::isspace((unsigned char)text[p])) p++;
        if (p < text.size() && text[p] == ',') p++;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Gemini Flash analysis with OpenCV occupancy + per-word crop OCR.
// ---------------------------------------------------------------------------
static void stream_analyze_gemini(const std::vector<uint8_t>& buf,
                                   httplib::DataSink& sink,
                                   bool is_memento = false,
                                   bool skip_woogles = false) {
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

    // Step 2: Board mode + rack detection
    std::vector<RackTile> rack_tiles;
    bool is_light_mode = false;
    if (have_opencv) {
        int bx, by, cell_sz, board_w = 0;
        if (parse_board_rect_from_log(opencv_dr.log, bx, by, cell_sz, &board_w)) {
            is_light_mode = detect_board_mode(buf, bx, by, cell_sz);
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
                (is_light_mode ? "light" : "dark") +
                (is_memento ? " (memento share image)" : "") + "\"}\n";
            sink.write(mode_msg.data(), mode_msg.size());
        }
    }

    // Build occupancy grid for Gemini prompt
    std::string occupancy_grid;
    auto get_occupied = [&](int r, int c) -> bool {
        return have_opencv && opencv_dr.cells[r][c].letter != 0;
    };

    if (have_opencv) {
        for (int r = 0; r < 15; r++) {
            if (r > 0) occupancy_grid += "\\n";
            for (int c = 0; c < 15; c++)
                occupancy_grid += get_occupied(r, c) ? 'X' : '.';
        }
    }

    // Build per-word crop images for focused Gemini OCR.
    std::vector<WordRun> word_runs;
    std::vector<std::string> wc_payloads;   // one payload per batch
    std::vector<std::future<GeminiCallResult>> wc_futs;

    if (have_opencv) {
        int bx_w, by_w, cs_w, bw_w = 0, bh_w = 0;
        if (parse_board_rect_from_log(opencv_dr.log, bx_w, by_w, cs_w, &bw_w, &bh_w)) {
            cv::Mat raw_w(1, static_cast<int>(buf.size()), CV_8UC1,
                          const_cast<uint8_t*>(buf.data()));
            cv::Mat img_w = cv::imdecode(raw_w, cv::IMREAD_COLOR);
            if (!img_w.empty()) {
                double cw_w = bw_w > 0 ? bw_w / 15.0 : (double)cs_w;
                double ch_w = bh_w > 0 ? bh_w / 15.0 : (double)cs_w;
                word_runs = find_word_runs(get_occupied);

                // Build transposed image: original column C becomes row C,
                // so vertical words appear as horizontal strips with upright letters.
                int cs_t = (int)std::round(std::min(cw_w, ch_w));
                cv::Mat timg;
                if (cs_t > 0) {
                    int tsz = 15 * cs_t;
                    timg = cv::Mat(tsz, tsz, CV_8UC3, cv::Scalar(30, 30, 30));
                    auto cell_x = [&](int c) { return bx_w + (int)std::round(c * cw_w); };
                    auto cell_y = [&](int r) { return by_w + (int)std::round(r * ch_w); };
                    for (int tr = 0; tr < 15; tr++) {
                        for (int tc = 0; tc < 15; tc++) {
                            int sx = cell_x(tr), sw = cell_x(tr+1) - sx;
                            int sy = cell_y(tc), sh = cell_y(tc+1) - sy;
                            int dx = tc * cs_t, dy = tr * cs_t;
                            if (sx < 0 || sy < 0 || sw <= 0 || sh <= 0 ||
                                sx+sw > img_w.cols || sy+sh > img_w.rows) continue;
                            cv::Mat src_roi = img_w(cv::Rect(sx, sy, sw, sh));
                            cv::Mat dst_roi = timg(cv::Rect(dx, dy, cs_t, cs_t));
                            if (sw == cs_t && sh == cs_t)
                                src_roi.copyTo(dst_roi);
                            else
                                cv::resize(src_roi, dst_roi, dst_roi.size(),
                                           0, 0, cv::INTER_LANCZOS4);
                        }
                    }
                }

                std::string wc_prompt =
                    "Read the letters on Scrabble tiles in each labeled image. "
                    "Images labeled H* are horizontal words; V* are vertical words "
                    "(presented with upright letters reading left to right).\\n"
                    "Tile rules:\\n";
                if (is_memento) {
                    wc_prompt +=
                        "- Regular tiles: PURPLE or GOLD squares with a letter and small "
                        "subscript number \\u2014 return UPPERCASE.\\n"
                        "- Blank tiles: CIRCLES with italic letters, NO subscript "
                        "\\u2014 return lowercase.\\n";
                } else if (is_light_mode) {
                    wc_prompt +=
                        "- Regular tiles: BLUE/PURPLE squares with a letter and small "
                        "subscript number \\u2014 return UPPERCASE.\\n"
                        "- Blank tiles: GREEN CIRCLES with italic letters, NO subscript "
                        "\\u2014 return lowercase.\\n";
                } else {
                    wc_prompt +=
                        "- Regular tiles: beige squares with a letter and small subscript "
                        "number \\u2014 return UPPERCASE.\\n"
                        "- Blank tiles: PURPLE CIRCLES with italic letters, NO subscript "
                        "\\u2014 return lowercase.\\n";
                }
                wc_prompt +=
                    "The small subscript number is just the point value \\u2014 ignore it.\\n"
                    "Reply ONLY as JSON: {\\\"LABEL\\\": \\\"LETTERS\\\", ...}\\n"
                    "Example: {\\\"H11H\\\": \\\"WORMER\\\", \\\"VH8\\\": \\\"WAVES\\\"}";

                // Build batched payloads: split at 200 KB to avoid Gemini timeouts
                const size_t BATCH_LIMIT = 200 * 1024;
                std::string cur_payload = "{\"contents\":[{\"parts\":["
                    "{\"text\":\"" + wc_prompt + "\"}";
                int n_crops = 0;
                for (const auto& wr : word_runs) {
                    auto png = crop_word_run(img_w, timg, cs_t, wr, bx_w, by_w, cw_w, ch_w);
                    if (png.empty()) continue;
                    std::string b64 = base64_encode(png);
                    std::string part =
                        ",{\"text\":\"" + wr.label + ":\"}"
                        ",{\"inlineData\":{\"mimeType\":\"image/png\","
                        "\"data\":\"" + b64 + "\"}}";
                    // Seal current batch and start a new one if over limit
                    if (n_crops > 0 && cur_payload.size() + part.size() > BATCH_LIMIT) {
                        wc_payloads.push_back(cur_payload + "]}]}");
                        cur_payload = "{\"contents\":[{\"parts\":["
                            "{\"text\":\"" + wc_prompt + "\"}";
                    }
                    cur_payload += part;
                    n_crops++;
                }
                if (n_crops > 0) wc_payloads.push_back(cur_payload + "]}]}");
                size_t total_kb = 0;
                for (const auto& p : wc_payloads) total_kb += p.size() / 1024;
                std::string wc_msg = "{\"status\":\"Built " + std::to_string(n_crops)
                    + " word crop images (" + std::to_string(total_kb) + " KB"
                    + (wc_payloads.size() > 1
                        ? ", " + std::to_string(wc_payloads.size()) + " batches"
                        : "")
                    + ", 2.0+2.5-flash in parallel)\"}\n";
                sink.write(wc_msg.data(), wc_msg.size());
            }
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

    // Stage snapshot: occupancy mask (which cells are detected as occupied)
    if (have_opencv) {
        CellResult occ_cells[15][15] = {};
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++)
                if (get_occupied(r, c)) occ_cells[r][c].letter = '?';
        std::string occ_cgp = cells_to_cgp(occ_cells);
        std::string smsg = "{\"stage\":\"occupancy\",\"occupancy_cgp\":\""
            + json_escape(occ_cgp) + "\"}\n";
        sink.write(smsg.data(), smsg.size());
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
            "\\n\\nCells marked '.' are likely empty, but if you can clearly see "
            "a tile there, include it. "
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

    if (is_memento) {
        // --- Memento (server-rendered share image) prompt ---
        prompt += "\\n\\nIMPORTANT — this is a Woogles.io SHARE IMAGE (server-rendered), "
            "NOT a browser screenshot. The layout is:"
            "\\n- \\\"Woogles.io\\\" header with logo at top-left"
            "\\n- Two SCORE BOXES at top-right: one PURPLE, one GOLD/YELLOW"
            "\\n- 15x15 board below the header"
            "\\n- Player RACK below the board"
            "\\n\\nTile colors indicate which PLAYER owns them:"
            "\\n- One player's tiles are PURPLE with white letters + subscript number"
            "\\n- Other player's tiles are GOLD/YELLOW with dark letters + subscript number"
            "\\n- BLANK tiles on the board: CIRCLES (not squares) with italic letters, NO subscript"
            "\\n- BLANK tiles on the rack: tiles with NO letter and NO subscript number — "
            "report as '?'";
        prompt += "\\n\\nAlso read:"
            "\\n- The current player's RACK (row of tiles below the board). "
            "Use uppercase for regular tiles, '?' for blank tiles (tiles with no letter)."
            "\\n- The SCORES for both players. To determine who is ON TURN:"
            "\\n  Look at the RACK tiles below the board — they are either PURPLE or GOLD."
            "\\n  The rack color tells you which player is ON TURN."
            "\\n  Match the rack color to the score box of the SAME color at the top-right."
            "\\n  That score is the ON-TURN player's score."
            "\\n  The OTHER score box is the waiting player's score."
            "\\n  List ON-TURN player's score FIRST, waiting player SECOND."
            "\\n- The PLAYER USERNAMES shown in the score boxes (e.g. \\\"cesar\\\", \\\"BestBot\\\")."
            "\\n  player1 = on-turn player, player2 = waiting player."
            "\\n\\nReturn ONLY a JSON object with these fields:"
            "\\n{"
            "\\n  \\\"board\\\": [[...], ...],  // 15x15 array"
            "\\n  \\\"rack\\\": \\\"ABCDE?F\\\",  // current player rack (? = blank)"
            "\\n  \\\"scores\\\": [241, 198],  // [ON-TURN player score, waiting player score]"
            "\\n  \\\"player1\\\": \\\"on_turn_username\\\",  // optional, if visible"
            "\\n  \\\"player2\\\": \\\"waiting_username\\\"   // optional, if visible"
            "\\n}";
        prompt += "\\n\\nBoard array elements:"
            "\\n- Uppercase letter (e.g. \\\"A\\\") for regular tiles (purple or gold squares)"
            "\\n- Lowercase letter (e.g. \\\"s\\\") for BLANK tiles (circles, italic, no subscript)"
            "\\n- null for empty cells";
        prompt += "\\n\\nSanity check: board tiles + rack tiles should be consistent with "
            "a standard 100-tile English Scrabble distribution."
            "\\n\\nReturn ONLY the JSON object, no other text.";
    } else {
        // --- Browser screenshot prompt ---
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
                "\\n- REGULAR tiles: beige/tan SQUARES with upright letters AND a small "
                "subscript number (point value) in the bottom-right corner. "
                "The subscript IS the key indicator — if you see a small number, "
                "it is ALWAYS a regular tile."
                "\\n- BLANK tiles on the BOARD: PURPLE CIRCLES (not squares!) with "
                "italic letters and NO subscript number. The tile SHAPE changes from "
                "square to circle for blanks."
                "\\n- IMPORTANT: If the tile is a SQUARE (not circle), it is ALWAYS "
                "a regular tile — return UPPERCASE even if the letter looks italic."
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
            "\\n- The SCORES for both players. To determine who is ON TURN:"
            "\\n  STEP 1 — Look at the TURN HISTORY list. Find the MOST RECENT move "
            "(the last entry at the bottom of the visible list). "
            "Each history entry shows a small avatar/icon next to it matching the player's avatar "
            "in the player panel — use this to identify which player just moved. "
            "IMPORTANT: the history may be scrolled — verify the last visible entry's cumulative "
            "score matches the current score shown in that player's panel. "
            "That player just finished their turn — they are WAITING. "
            "The OTHER player is ON TURN."
            "\\n  STEP 2 — Confirm: the ON-TURN player's score box usually has a GREEN background."
            "\\n  STEP 3 — Confirm: the rack tiles shown belong to the ON-TURN player."
            "\\nList the ON-TURN player's score FIRST, waiting player SECOND."
            "\\n- The PLAYER USERNAMES from the player panels (the text labels showing each "
            "player's Woogles username, e.g. \\\"cesar\\\", \\\"BestBot\\\")."
            "\\n  player1 = on-turn player, player2 = waiting player."
            "\\n\\nReturn ONLY a JSON object with these fields:"
            "\\n{"
            "\\n  \\\"board\\\": [[...], ...],  // 15x15 array"
            "\\n  \\\"rack\\\": \\\"ABCDE?F\\\",  // current player rack (? = blank)"
            "\\n  \\\"lexicon\\\": \\\"NWL23\\\",  // lexicon name"
            "\\n  \\\"bag\\\": \\\"A E II O U B C D L N S TT X\\\",  // tile tracking text"
            "\\n  \\\"scores\\\": [241, 198],  // [ON-TURN player score, waiting player score]"
            "\\n  \\\"player1\\\": \\\"on_turn_username\\\",  // optional, if visible"
            "\\n  \\\"player2\\\": \\\"waiting_username\\\"   // optional, if visible"
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
    }

    std::string payload = "{\"contents\":[{\"parts\":["
        "{\"text\":\"" + prompt + "\"},"
        "{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":\"" + b64 + "\"}}"
        "]}]}";


    std::string url = "https://generativelanguage.googleapis.com/v1beta/models/"
                      "gemini-2.5-flash:generateContent?key=";
    url += api_key;
    // Cheaper/faster model for word crops: no thinking overhead, lower latency.
    std::string wc_url_20 = "https://generativelanguage.googleapis.com/v1beta/models/"
                            "gemini-2.0-flash:generateContent?key=";
    wc_url_20 += api_key;

    auto t0 = std::chrono::steady_clock::now();
    // Launch main OCR and word-crop OCR in parallel
    auto main_fut = std::async(std::launch::async,
        [&url, &payload, &prompt]() {
            return call_gemini(url, payload, "main", prompt, 60, 2);
        });
    // For each batch, query both gemini-2.0-flash (fast, no thinking) and
    // gemini-2.5-flash (thinking disabled via thinkingBudget=0) in parallel.
    // Results are merged — whichever model reads more words wins for each cell.
    for (const auto& wcp : wc_payloads) {
        // 2.0-flash: fast, no thinking by default
        wc_futs.push_back(std::async(std::launch::async,
            [&wc_url_20, p = wcp]() {
                return call_gemini(wc_url_20, p, "wc_2.0", "wc", 30, 1);
            }));
        // 2.5-flash with thinking disabled: strip final } and inject thinkingBudget=0
        std::string wcp_nt = wcp.substr(0, wcp.size() - 1)
            + ",\"generationConfig\":{\"thinkingConfig\":{\"thinkingBudget\":0}}}";
        wc_futs.push_back(std::async(std::launch::async,
            [&url, p = wcp_nt]() {
                return call_gemini(url, p, "wc_2.5", "wc", 45, 1);
            }));
    }
    auto gcr = main_fut.get();
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    {
        std::string retry_note = gcr.attempts > 1
            ? " after " + std::to_string(gcr.attempts) + " attempts" : "";
        std::string msg = "{\"status\":\"Gemini responded ("
            + std::to_string(ms) + " ms, "
            + std::to_string(gcr.raw_response.size() / 1024) + " KB"
            + retry_note + "). Parsing...\"}\n";
        sink.write(msg.data(), msg.size());
    }

    std::string text = gcr.text;
    if (text.empty()) {
        std::string err_msg = json_extract_string(gcr.raw_response, "message");
        if (err_msg.empty()) {
            std::string preview = gcr.raw_response.substr(0, 500);
            err_msg = "Failed to parse Gemini response. Raw: " + preview;
        }
        std::string err = "{\"status\":\"Error: " + json_escape(err_msg) + "\"}\n";
        sink.write(err.data(), err.size());
        sink.done();
        return;
    }

    // Extract rack, lexicon, bag, player names from the JSON response
    std::string gemini_rack, gemini_lexicon, gemini_bag;
    std::string gemini_player1, gemini_player2;
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
        gemini_player1 = extract_str("player1");
        gemini_player2 = extract_str("player2");

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

    // Join all word-crop OCR batch futures (launched in parallel with main OCR).
    // Runs here so it overlaps with main board parsing and rack detection above.
    std::map<std::string, std::string> word_crop_map;
    bool have_word_crop_ocr = false;
    for (auto& fut : wc_futs) {
        auto gcr_wc = fut.get();
        if (!gcr_wc.text.empty()) {
            auto batch_map = parse_json_str_map(gcr_wc.text);
            word_crop_map.insert(batch_map.begin(), batch_map.end());
        }
    }
    if (!wc_futs.empty()) {
        have_word_crop_ocr = !word_crop_map.empty();
        std::string wmsg = "{\"status\":\"Word crop OCR "
            + std::string(have_word_crop_ocr
                ? "parsed (" + std::to_string(word_crop_map.size()) + " words)."
                : "failed to parse.")
            + "\"}\n";
        sink.write(wmsg.data(), wmsg.size());
    }

    // Save raw OCR snapshot (before any corrections) for trail comparison.
    CellResult raw_main_cells[15][15];
    std::memcpy(raw_main_cells, dr.cells, sizeof(raw_main_cells));

    // Stream the raw main OCR board immediately so the user can see it
    // while corrections are being computed.
    {
        std::string raw_cgp = cells_to_cgp(raw_main_cells);
        // Build minimal board-cells JSON for the raw board update
        std::string cells_json = "[";
        for (int r = 0; r < 15; r++) {
            if (r > 0) cells_json += ",";
            cells_json += "[";
            for (int c = 0; c < 15; c++) {
                if (c > 0) cells_json += ",";
                char ch = raw_main_cells[r][c].letter;
                if (ch == 0) cells_json += "null";
                else cells_json += "\"" + std::string(1, ch) + "\"";
            }
            cells_json += "]";
        }
        cells_json += "]";
        std::string rmsg = "{\"status\":\"Raw Gemini OCR — applying corrections...\","
            "\"raw_main_cgp\":\"" + json_escape(raw_cgp) + "\","
            "\"raw_cells\":" + cells_json + "}\n";
        sink.write(rmsg.data(), rmsg.size());
    }

    // Transposed OCR corrections (Cases 1 and 3) are applied AFTER realignment
    // and occupancy enforcement below — see comment there.


    // Launch rack verification asynchronously — runs concurrently with
    // board post-processing (realignment, retry-missing, bag-math).
    std::future<std::string> rack_verify_future;
    bool rack_verify_launched = false;
    if (!rack_tiles.empty() && rack_tiles.size() <= 7 &&
        (static_cast<int>(gemini_rack.size()) != static_cast<int>(rack_tiles.size())
         || std::any_of(rack_tiles.begin(), rack_tiles.end(),
                        [](const RackTile& rt) { return rt.is_blank; }))) {
        // Build payload on main thread (needs rack_tiles data)
        std::string rack_prompt =
            "These are cropped images of Scrabble rack tiles, in order left to right. "
            "Identify each tile's letter. "
            "Regular tiles are beige with a letter and a subscript number (return UPPERCASE). "
            "BLANK tiles are beige with NO letter and NO subscript \\u2014 return '?' for these. "
            "Return ONLY a JSON array of strings, one per image. "
            "Example: [\\\"B\\\", \\\"I\\\", \\\"?\\\"]";
        std::string rack_payload = "{\"contents\":[{\"parts\":["
            "{\"text\":\"" + rack_prompt + "\"}";
        std::string rack_msg = "{\"status\":\"Verifying rack ("
            + std::to_string(rack_tiles.size()) + " tiles detected)...\",\"crops\":[";
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

        // Launch async Gemini call
        rack_verify_launched = true;
        rack_verify_future = std::async(std::launch::async,
            [rack_payload = std::move(rack_payload),
             rack_prompt = std::move(rack_prompt),
             url]() -> std::string {
                auto gcr = call_gemini(url, rack_payload, "verify_rack",
                                       rack_prompt, 30, 1);
                std::string result;
                if (!gcr.text.empty()) {
                    std::string textr = gcr.text;
                    size_t ri = 0;
                    while (ri < textr.size() && textr[ri] != '[') ri++;
                    ri++;
                    while (ri < textr.size()) {
                        while (ri < textr.size() && textr[ri] != '"'
                               && textr[ri] != ']') ri++;
                        if (ri >= textr.size() || textr[ri] == ']') break;
                        ri++;
                        if (ri < textr.size()) {
                            result += textr[ri];
                            ri++;
                            if (ri < textr.size() && textr[ri] == '"') ri++;
                        }
                    }
                }
                return result;
            });
    }

    // Step 3.5: Row realignment — Gemini often returns correct letters
    // but shifted horizontally from the occupancy mask positions.
    // For each row, find contiguous blocks in both mask and Gemini output.
    // If blocks match in count and relative spacing, realign Gemini's
    // letters to the mask positions.
    if (have_opencv) {
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
            if (mask_blocks.empty()) continue;

            // Compute total tiles in each
            int mask_total = 0, gem_total = 0;
            for (const auto& b : mask_blocks) mask_total += b.len;
            for (const auto& b : gem_blocks) gem_total += b.len;

            // Strategy 1: block-by-block realignment (same count, same lengths)
            bool do_block_realign = false;
            if (mask_blocks.size() == gem_blocks.size()) {
                bool lengths_match = true;
                for (size_t i = 0; i < mask_blocks.size(); i++) {
                    if (mask_blocks[i].len != gem_blocks[i].len) {
                        lengths_match = false;
                        break;
                    }
                }
                if (lengths_match) {
                    bool any_shifted = false;
                    for (size_t i = 0; i < mask_blocks.size(); i++) {
                        if (mask_blocks[i].start != gem_blocks[i].start) {
                            any_shifted = true;
                            break;
                        }
                    }
                    do_block_realign = any_shifted;
                }
            }

            // Strategy 2: flat realignment — total tile counts match but
            // blocks differ (Gemini merged gaps or returned wrong row length).
            // Map Gemini tiles left-to-right into mask positions.
            bool do_flat_realign = !do_block_realign
                && mask_total == gem_total && gem_total > 0
                && mask_blocks.size() != gem_blocks.size();

            if (!do_block_realign && !do_flat_realign) continue;

            CellResult saved[15];
            for (int c = 0; c < 15; c++) saved[c] = dr.cells[r][c];
            for (int c = 0; c < 15; c++) dr.cells[r][c] = {};

            if (do_block_realign) {
                for (size_t i = 0; i < mask_blocks.size(); i++) {
                    for (int j = 0; j < mask_blocks[i].len; j++) {
                        dr.cells[r][mask_blocks[i].start + j] =
                            saved[gem_blocks[i].start + j];
                    }
                    if (mask_blocks[i].start != gem_blocks[i].start) {
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
            } else {
                // Flat realign: collect all Gemini tiles, place into mask slots
                std::vector<CellResult> gem_tiles;
                for (const auto& b : gem_blocks)
                    for (int j = 0; j < b.len; j++)
                        gem_tiles.push_back(saved[b.start + j]);
                int ti = 0;
                for (const auto& b : mask_blocks)
                    for (int j = 0; j < b.len && ti < (int)gem_tiles.size(); j++)
                        dr.cells[r][b.start + j] = gem_tiles[ti++];

                std::string letters;
                for (const auto& t : gem_tiles)
                    letters += static_cast<char>(std::toupper(
                        static_cast<unsigned char>(t.letter)));
                realign_log += "Row " + std::to_string(r + 1)
                    + ": flat-realigned " + letters + " ("
                    + std::to_string(gem_blocks.size()) + " blocks -> "
                    + std::to_string(mask_blocks.size()) + " blocks)\n";
            }
        }
        if (!realign_log.empty()) {
            std::string msg = "{\"status\":\"Realigned shifted rows: "
                + json_escape(realign_log) + "\"}\n";
            sink.write(msg.data(), msg.size());
        }
    }

    // Stage snapshot: after realignment (before occupancy enforcement)
    { std::string s="{\"stage\":\"realigned\",\"stage_cgp\":\""+json_escape(cells_to_cgp(dr.cells))+"\"}\n"; sink.write(s.data(),s.size()); }

    // Step 4: Strict occupancy enforcement — clear all non-occupied cells.
    // Save Gemini's original readings for disputed cells (Gemini found a
    // letter but color detection says empty) so we can re-verify with crops.
    struct DisputedCell { int r, c; CellResult orig; };
    std::vector<DisputedCell> disputed;
    if (have_opencv) {
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++)
                if (!get_occupied(r, c) && dr.cells[r][c].letter != 0) {
                    // Save before clearing — we'll re-verify with crops
                    disputed.push_back({r, c, dr.cells[r][c]});
                    dr.cells[r][c] = {};
                }
    }

    // Apply word-crop OCR results: per-word Gemini reads of tight rectangular
    // crops, which are immune to toast/UI overlays that confuse full-board OCR.
    // Only accept a run's reading if it forms a valid lexicon word — this guards
    // against Gemini confusing labels when processing many images at once.
    if (have_word_crop_ocr) {
        struct WCReading { char letter; int run_len; bool is_horiz; };
        std::map<std::pair<int,int>, WCReading> cell_readings;

        for (const auto& wr : word_runs) {
            // Skip single-cell runs — not enough context for reliable OCR.
            if (wr.length() < 2) continue;

            auto it = word_crop_map.find(wr.label);
            if (it == word_crop_map.end()) continue;
            const std::string& letters = it->second;
            if ((int)letters.size() != wr.length()) continue;

            // Validate: all chars must be ASCII letters
            bool all_letters = true;
            for (char ch : letters)
                if (!std::isalpha((unsigned char)ch)) { all_letters = false; break; }
            if (!all_letters) continue;

            // Only accept if the reading forms a valid lexicon word.
            if (!g_kwg_lexicon.empty()) {
                std::string upper = letters;
                for (char& ch : upper)
                    ch = static_cast<char>(std::toupper((unsigned char)ch));
                if (!g_kwg.is_valid(upper)) continue;
            }

            for (int i = 0; i < wr.length(); i++) {
                int r = wr.horizontal ? wr.r : (wr.start + i);
                int c = wr.horizontal ? (wr.start + i) : wr.c;
                if (!get_occupied(r, c)) continue;
                char ch = letters[i];
                auto key = std::make_pair(r, c);
                auto existing = cell_readings.find(key);
                bool prefer_new = existing == cell_readings.end()
                    || wr.length() > existing->second.run_len
                    || (wr.length() == existing->second.run_len && wr.horizontal);
                if (prefer_new)
                    cell_readings[key] = {ch, wr.length(), wr.horizontal};
            }
        }

        std::string wc_fill_detail, wc_fix_detail;
        for (const auto& [pos, rd] : cell_readings) {
            int r = pos.first, c = pos.second;
            char new_ch = rd.letter;
            if (new_ch == 0) continue;
            char old_ch = dr.cells[r][c].letter;
            char old_u = old_ch ? static_cast<char>(std::toupper(
                static_cast<unsigned char>(old_ch))) : 0;
            char new_u = static_cast<char>(std::toupper(
                static_cast<unsigned char>(new_ch)));
            if (old_ch == 0) {
                dr.cells[r][c].letter = new_ch;
                dr.cells[r][c].confidence = 0.8f;
                dr.cells[r][c].is_blank = (new_ch >= 'a' && new_ch <= 'z');
                if (!wc_fill_detail.empty()) wc_fill_detail += ", ";
                wc_fill_detail += std::string(1, static_cast<char>('A' + c))
                    + std::to_string(r + 1) + "=" + new_u;
            } else if (old_u != new_u) {
                dr.cells[r][c].letter = new_ch;
                dr.cells[r][c].confidence = 0.8f;
                dr.cells[r][c].is_blank = (new_ch >= 'a' && new_ch <= 'z');
                if (!wc_fix_detail.empty()) wc_fix_detail += ", ";
                wc_fix_detail += std::string(1, static_cast<char>('A' + c))
                    + std::to_string(r + 1) + ":" + old_u + "->" + new_u;
            }
        }
        if (!wc_fill_detail.empty()) {
            std::string msg = "{\"status\":\"Word crop OCR filled: "
                + json_escape(wc_fill_detail) + "\"}\n";
            sink.write(msg.data(), msg.size());
        }
        if (!wc_fix_detail.empty()) {
            std::string msg = "{\"status\":\"Word crop OCR corrected: "
                + json_escape(wc_fix_detail) + "\"}\n";
            sink.write(msg.data(), msg.size());
        }
    }

    // Stage snapshot: after word-crop OCR corrections
    { std::string s="{\"stage\":\"trans\",\"stage_cgp\":\""+json_escape(cells_to_cgp(dr.cells))+"\"}\n"; sink.write(s.data(),s.size()); }

    // Step 5: Re-query Gemini for cells needing verification (crop + retry)
    // Two categories:
    //   (a) occupied cells Gemini missed
    //   (b) disputed cells (Gemini found letter, color says empty)
    std::vector<RetryCell> retry_cells;
    if (have_opencv) {
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++)
                if (get_occupied(r, c) && dr.cells[r][c].letter == 0)
                    retry_cells.push_back({r, c, false});
    }
    for (const auto& d : disputed)
        retry_cells.push_back({d.r, d.c, true});

    if (!retry_cells.empty() && have_opencv) {
        int n_missing = 0, n_disputed = 0;
        for (const auto& rc : retry_cells)
            if (rc.is_disputed) n_disputed++; else n_missing++;
        std::string msg = "{\"status\":\"Re-querying "
            + std::to_string(n_missing) + " missed + "
            + std::to_string(n_disputed)
            + " disputed cell(s)...\"}\n";
        sink.write(msg.data(), msg.size());

        int bx, by, cell_sz;
        if (parse_board_rect_from_log(opencv_dr.log, bx, by, cell_sz)) {
            cv::Mat raw(1, static_cast<int>(buf.size()), CV_8UC1,
                        const_cast<uint8_t*>(buf.data()));
            cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);

            if (!img.empty()) {
                // Build multi-image retry with cropped cells
                std::string retry_prompt =
                    "Here are cropped images of individual Scrabble board cells. "
                    "Some cells have tiles, some may be EMPTY (no tile). "
                    "For each image: "
                    "- If there IS a tile: return the letter. "
                    "Regular tiles are beige SQUARES with a small subscript "
                    "number in the bottom-right (return UPPERCASE). "
                    "Blank tiles are PURPLE CIRCLES (round, not square) with "
                    "italic letters and NO subscript (return lowercase). "
                    "If the tile is SQUARE with a subscript, it is ALWAYS "
                    "regular — return UPPERCASE. "
                    "- If the cell is EMPTY (board background, no tile): "
                    "return null. "
                    "Return ONLY a JSON array. "
                    "Example: [\\\"F\\\", null, \\\"s\\\", \\\"A\\\"]";

                std::string retry_payload = "{\"contents\":[{\"parts\":["
                    "{\"text\":\"" + retry_prompt + "\"}";

                int pad = cell_sz / 8;
                for (const auto& rc : retry_cells) {
                    int cx = std::max(0, bx + rc.c * cell_sz - pad);
                    int cy = std::max(0, by + rc.r * cell_sz - pad);
                    int cw = std::min(cell_sz + 2 * pad, img.cols - cx);
                    int ch = std::min(cell_sz + 2 * pad, img.rows - cy);
                    cv::Mat cell_img = img(cv::Rect(cx, cy, cw, ch)).clone();
                    // Mask out the subscript (point value) in the bottom-right
                    // corner — it can mislead Gemini into reading the wrong letter.
                    int sx = cell_img.cols * 6 / 8;
                    int sy = cell_img.rows * 6 / 8;
                    cell_img(cv::Rect(sx, sy, cell_img.cols - sx,
                                     cell_img.rows - sy)) = cv::Scalar(128, 128, 128);
                    std::vector<uint8_t> png_buf;
                    cv::imencode(".png", cell_img, png_buf);
                    retry_payload += ",{\"inlineData\":{\"mimeType\":\"image/png\","
                        "\"data\":\"" + base64_encode(png_buf) + "\"}}";
                }
                retry_payload += "]}]}";

                auto gcr2 = call_gemini(url, retry_payload,
                    "retry_verify", retry_prompt, 30, 1);
                if (!gcr2.text.empty()) {
                    auto results = parse_retry_response(gcr2.text, retry_cells);
                    for (size_t mi = 0; mi < results.size(); mi++) {
                        if (results[mi].letter != 0) {
                            dr.cells[retry_cells[mi].r]
                                    [retry_cells[mi].c] = results[mi];
                        }
                    }
                }
            }
        }
    }

    // Any cells still unresolved get '?' placeholder
    if (have_opencv) {
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++)
                if (get_occupied(r, c) && dr.cells[r][c].letter == 0) {
                    dr.cells[r][c].letter = '?';
                    dr.cells[r][c].confidence = 0.0f;
                }
    }

    // Step 5.5: Board connectivity check — find islands not connected to center
    if (have_opencv) {
        auto conn = check_board_connectivity(dr.cells);
        if (conn.center_empty || !conn.islands.empty()) {
            // Build status message
            std::string conn_msg = "{\"status\":\"Connectivity: ";
            if (conn.center_empty)
                conn_msg += "center empty, ";
            if (!conn.islands.empty())
                conn_msg += std::to_string(conn.islands.size()) + " island(s), ";
            conn_msg += std::to_string(conn.bridge_candidates.size())
                + " bridge candidate(s)\"}\n";
            sink.write(conn_msg.data(), conn_msg.size());

            // Build requery list: bridge candidates + island tiles (disputed)
            std::vector<RetryCell> conn_retry;
            for (const auto& [br, bc] : conn.bridge_candidates)
                conn_retry.push_back({br, bc, false});
            for (const auto& island : conn.islands)
                for (const auto& [ir, ic] : island)
                    conn_retry.push_back({ir, ic, true});

            if (!conn_retry.empty()) {
                int bx, by, cell_sz;
                if (parse_board_rect_from_log(opencv_dr.log, bx, by, cell_sz)) {
                    cv::Mat raw_c(1, static_cast<int>(buf.size()), CV_8UC1,
                                const_cast<uint8_t*>(buf.data()));
                    cv::Mat img_c = cv::imdecode(raw_c, cv::IMREAD_COLOR);

                    if (!img_c.empty()) {
                        std::string conn_prompt =
                            "Here are cropped images of individual Scrabble board cells. "
                            "Some cells have tiles, some may be EMPTY (no tile). "
                            "For each image: "
                            "- If there IS a tile: return the letter. "
                            "Regular tiles are beige SQUARES with a small subscript "
                            "number in the bottom-right (return UPPERCASE). "
                            "Blank tiles are PURPLE CIRCLES (round, not square) with "
                            "italic letters and NO subscript (return lowercase). "
                            "If the tile is SQUARE with a subscript, it is ALWAYS "
                            "regular — return UPPERCASE. "
                            "- If the cell is EMPTY (board background, no tile): "
                            "return null. "
                            "Return ONLY a JSON array. "
                            "Example: [\\\"F\\\", null, \\\"s\\\", \\\"A\\\"]";

                        std::string conn_payload = "{\"contents\":[{\"parts\":["
                            "{\"text\":\"" + conn_prompt + "\"}";

                        int pad = cell_sz / 8;
                        for (const auto& rc : conn_retry) {
                            int cx = std::max(0, bx + rc.c * cell_sz - pad);
                            int cy = std::max(0, by + rc.r * cell_sz - pad);
                            int cw = std::min(cell_sz + 2 * pad, img_c.cols - cx);
                            int ch = std::min(cell_sz + 2 * pad, img_c.rows - cy);
                            cv::Mat cell_img = img_c(cv::Rect(cx, cy, cw, ch)).clone();
                            int sx = cell_img.cols * 6 / 8;
                            int sy = cell_img.rows * 6 / 8;
                            cell_img(cv::Rect(sx, sy, cell_img.cols - sx,
                                             cell_img.rows - sy)) = cv::Scalar(128, 128, 128);
                            std::vector<uint8_t> png_buf;
                            cv::imencode(".png", cell_img, png_buf);
                            conn_payload += ",{\"inlineData\":{\"mimeType\":\"image/png\","
                                "\"data\":\"" + base64_encode(png_buf) + "\"}}";
                        }
                        conn_payload += "]}]}";

                        auto gcrc = call_gemini(url, conn_payload,
                            "connectivity", conn_prompt, 30, 1);
                        if (!gcrc.text.empty()) {
                            auto results = parse_retry_response(gcrc.text, conn_retry);
                            int filled = 0;
                            for (size_t mi = 0; mi < results.size(); mi++) {
                                if (results[mi].letter != 0) {
                                    dr.cells[conn_retry[mi].r]
                                            [conn_retry[mi].c] = results[mi];
                                    filled++;
                                }
                            }
                            if (filled > 0) {
                                std::string fmsg = "{\"status\":\"Connectivity: filled "
                                    + std::to_string(filled) + " cell(s)\"}\n";
                                sink.write(fmsg.data(), fmsg.size());
                            }
                        }
                    }
                }
            }

            // Re-check connectivity after fixes
            auto conn2 = check_board_connectivity(dr.cells);
            if (!conn2.islands.empty()) {
                std::string wmsg = "{\"status\":\"Warning: "
                    + std::to_string(conn2.islands.size())
                    + " island(s) still disconnected\"}\n";
                sink.write(wmsg.data(), wmsg.size());
            }
        }
    }

    // Stage snapshot: after all retry crops + connectivity
    { std::string s="{\"stage\":\"retry\",\"stage_cgp\":\""+json_escape(cells_to_cgp(dr.cells))+"\"}\n"; sink.write(s.data(),s.size()); }

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
                            "Regular tiles are beige SQUARES with a small subscript "
                            "number (UPPERCASE). Blank tiles are PURPLE CIRCLES "
                            "(round, not square) with NO subscript (lowercase). "
                            "If SQUARE with subscript, ALWAYS return UPPERCASE. "
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
                            char cur_lbl = static_cast<char>(std::toupper(
                                static_cast<unsigned char>(dr.cells[sp.r][sp.c].letter)));
                            char raw_lbl = raw_main_cells[sp.r][sp.c].letter
                                ? static_cast<char>(std::toupper(static_cast<unsigned char>(
                                    raw_main_cells[sp.r][sp.c].letter))) : 0;
                            crop_status += "{\"pos\":\"" +
                                std::string(1, static_cast<char>('A' + sp.c)) +
                                std::to_string(sp.r + 1) + "\",\"cur\":\"" +
                                std::string(1, cur_lbl) + "\"" +
                                (raw_lbl && raw_lbl != cur_lbl
                                    ? ",\"initial\":\"" + std::string(1, raw_lbl) + "\""
                                    : "") +
                                ",\"img\":\"data:image/png;base64," +
                                b64_crop + "\"}";
                        }
                        vfy_payload += "]}]}";

                        // Send status with crop previews
                        crop_status += "]}\n";
                        sink.write(crop_status.data(), crop_status.size());

                        auto gcrv = call_gemini(url, vfy_payload,
                            "verify_board", vfy_prompt, 30, 1);
                        if (!gcrv.text.empty()) {
                            std::string sv = gcrv.text;
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
                }
            }
        }
    }

    // Step 7: Join rack verification (launched asynchronously before Step 3.5)
    if (rack_verify_launched) {
        std::string new_rack = rack_verify_future.get();
        if (!new_rack.empty() && new_rack != gemini_rack) {
            std::string rmsg = "{\"status\":\"Rack corrected: "
                + json_escape(gemini_rack) + " -> "
                + json_escape(new_rack) + "\"}\n";
            sink.write(rmsg.data(), rmsg.size());
            gemini_rack = new_rack;
        }
    }

    // Build CGP: <board> <rack>/ <scores> lex <lexicon>;
    std::string rack_str = gemini_rack.empty() ? "" : gemini_rack;
    std::string lex_str = gemini_lexicon.empty() ? "" : gemini_lexicon;
    dr.cgp = cells_to_cgp(dr.cells) + " " + rack_str + "/ "
           + std::to_string(gemini_score1) + " " + std::to_string(gemini_score2);
    if (!lex_str.empty())
        dr.cgp += " lex " + lex_str + ";";
    dr.log = have_opencv ? "OpenCV occupancy + Gemini Flash OCR"
           : "Gemini Flash analysis";

    if (have_opencv)
        dr.debug_png = std::move(opencv_dr.debug_png);

    // Rack validation runs after word corrections (see below).
    std::string rack_warning;

    // Step 7.5: Lexicon inference — always check board words regardless of
    // what Gemini returned. If CSW-only words are present, that's definitive
    // evidence for CSW24 and overrides Gemini's claim (Gemini sometimes says
    // NWL23 for CSW24 games).
    {
        bool have_valid_gemini_lex = !gemini_lexicon.empty()
            && fs::exists("magpie/data/lexica/" + gemini_lexicon + ".kwg");

        std::vector<std::string> csw_only;
        std::string inferred = infer_lexicon(dr.cells, &csw_only);

        if (!csw_only.empty()) {
            // Found CSW-only words — override whatever Gemini returned.
            std::string wlist;
            for (const auto& w : csw_only) {
                if (!wlist.empty()) wlist += ", ";
                wlist += w;
            }
            std::string note = (have_valid_gemini_lex && gemini_lexicon != "CSW24")
                ? " (overrides Gemini's " + gemini_lexicon + ")" : "";
            gemini_lexicon = "CSW24";
            std::string msg = "{\"status\":\"Inferred lexicon CSW24 (CSW-only words: "
                + json_escape(wlist) + json_escape(note) + ")\"}\n";
            sink.write(msg.data(), msg.size());
        } else if (!have_valid_gemini_lex) {
            // Gemini gave no valid lexicon and no CSW-only words found.
            gemini_lexicon = "NWL23";
            std::string msg = "{\"status\":\"Inferred lexicon NWL23 (no CSW-only words found)\"}\n";
            sink.write(msg.data(), msg.size());
        }
        // If Gemini gave a valid lexicon and no CSW-only words, keep it as-is.

        // Update CGP with confirmed lexicon
        dr.cgp = cells_to_cgp(dr.cells) + " "
               + (gemini_rack.empty() ? "" : gemini_rack) + "/ "
               + std::to_string(gemini_score1) + " " + std::to_string(gemini_score2)
               + " lex " + gemini_lexicon + ";";
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

                // Gap filling: check empty cells at endpoints of invalid words
                // Only look at the cell immediately before/after each invalid
                // word — if it's empty and has letters on the other side, it
                // may be a missing tile that would extend the word.
                if (have_opencv) {
                    std::set<std::pair<int,int>> gap_set;
                    for (const auto& iw : invalid) {
                        if (iw.cells.size() < 2) continue;
                        if (iw.horizontal) {
                            int r = iw.cells.front().first;
                            int c0 = iw.cells.front().second;
                            int c1 = iw.cells.back().second;
                            // Check left of word
                            if (c0 > 0 && dr.cells[r][c0-1].letter == 0
                                && c0 >= 2 && dr.cells[r][c0-2].letter != 0)
                                gap_set.insert({r, c0-1});
                            // Check right of word
                            if (c1 < 14 && dr.cells[r][c1+1].letter == 0
                                && c1 <= 12 && dr.cells[r][c1+2].letter != 0)
                                gap_set.insert({r, c1+1});
                        } else {
                            int c = iw.cells.front().second;
                            int r0 = iw.cells.front().first;
                            int r1 = iw.cells.back().first;
                            // Check above word
                            if (r0 > 0 && dr.cells[r0-1][c].letter == 0
                                && r0 >= 2 && dr.cells[r0-2][c].letter != 0)
                                gap_set.insert({r0-1, c});
                            // Check below word
                            if (r1 < 14 && dr.cells[r1+1][c].letter == 0
                                && r1 <= 12 && dr.cells[r1+2][c].letter != 0)
                                gap_set.insert({r1+1, c});
                        }
                    }

                    if (!gap_set.empty()) {
                        std::string gmsg = "{\"status\":\"Found "
                            + std::to_string(gap_set.size())
                            + " gap(s) at invalid word endpoints, re-querying...\"}\n";
                        sink.write(gmsg.data(), gmsg.size());

                        std::vector<RetryCell> gap_retry;
                        for (const auto& [gr, gc] : gap_set)
                            gap_retry.push_back({gr, gc, false});

                        int bx, by, cell_sz;
                        if (parse_board_rect_from_log(opencv_dr.log, bx, by, cell_sz)) {
                            cv::Mat raw_g(1, static_cast<int>(buf.size()), CV_8UC1,
                                        const_cast<uint8_t*>(buf.data()));
                            cv::Mat img_g = cv::imdecode(raw_g, cv::IMREAD_COLOR);

                            if (!img_g.empty()) {
                                std::string gap_prompt =
                                    "Here are cropped images of individual Scrabble board cells. "
                                    "Some cells have tiles, some may be EMPTY (no tile). "
                                    "For each image: "
                                    "- If there IS a tile: return the letter. "
                                    "Regular tiles are beige SQUARES with a small subscript "
                                    "number in the bottom-right (return UPPERCASE). "
                                    "Blank tiles are PURPLE CIRCLES (round, not square) with "
                                    "italic letters and NO subscript (return lowercase). "
                                    "If the tile is SQUARE with a subscript, it is ALWAYS "
                                    "regular — return UPPERCASE. "
                                    "- If the cell is EMPTY (board background, no tile): "
                                    "return null. "
                                    "Return ONLY a JSON array. "
                                    "Example: [\\\"F\\\", null, \\\"s\\\", \\\"A\\\"]";

                                std::string gap_payload = "{\"contents\":[{\"parts\":["
                                    "{\"text\":\"" + gap_prompt + "\"}";

                                int pad = cell_sz / 8;
                                for (const auto& rc : gap_retry) {
                                    int cx = std::max(0, bx + rc.c * cell_sz - pad);
                                    int cy = std::max(0, by + rc.r * cell_sz - pad);
                                    int cw = std::min(cell_sz + 2 * pad, img_g.cols - cx);
                                    int ch = std::min(cell_sz + 2 * pad, img_g.rows - cy);
                                    cv::Mat cell_img = img_g(cv::Rect(cx, cy, cw, ch));
                                    std::vector<uint8_t> png_buf;
                                    cv::imencode(".png", cell_img, png_buf);
                                    gap_payload += ",{\"inlineData\":{\"mimeType\":\"image/png\","
                                        "\"data\":\"" + base64_encode(png_buf) + "\"}}";
                                }
                                gap_payload += "]}]}";

                                auto gcrg = call_gemini(url, gap_payload,
                                    "gap_fill", gap_prompt, 30, 1);
                                if (!gcrg.text.empty()) {
                                    auto gap_results = parse_retry_response(gcrg.text, gap_retry);
                                    int gap_filled = 0;
                                    std::string gap_detail;
                                    for (size_t gi = 0; gi < gap_results.size(); gi++) {
                                        if (gap_results[gi].letter != 0) {
                                            // Tentatively fill the gap
                                            int gr = gap_retry[gi].r, gc = gap_retry[gi].c;
                                            dr.cells[gr][gc] = gap_results[gi];

                                            // Check if the resulting word(s) are valid
                                            auto test_words = extract_words(dr.cells);
                                            bool all_valid = true;
                                            for (const auto& tw : test_words) {
                                                bool touches_gap = false;
                                                for (const auto& [wr, wc] : tw.cells)
                                                    if (wr == gr && wc == gc) { touches_gap = true; break; }
                                                if (touches_gap && !g_kwg.is_valid(tw.word)) {
                                                    all_valid = false;
                                                    break;
                                                }
                                            }

                                            if (all_valid) {
                                                gap_filled++;
                                                if (!gap_detail.empty()) gap_detail += ", ";
                                                gap_detail += std::string(1, static_cast<char>('A' + gc))
                                                    + std::to_string(gr + 1) + "="
                                                    + static_cast<char>(std::toupper(
                                                        static_cast<unsigned char>(gap_results[gi].letter)));
                                            } else {
                                                // Reject — leave empty
                                                dr.cells[gr][gc] = {};
                                            }
                                        }
                                    }
                                    if (gap_filled > 0) {
                                        std::string gfmsg = "{\"status\":\"Gap fill: "
                                            + json_escape(gap_detail) + "\"}\n";
                                        sink.write(gfmsg.data(), gfmsg.size());

                                        // Re-extract words and rebuild invalid list
                                        all_words = extract_words(dr.cells);
                                        invalid.clear();
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
                                        // Rebuild suspects
                                        valid_cells.clear();
                                        invalid_cells.clear();
                                        for (const auto& bw : all_words) {
                                            bool is_valid = g_kwg.is_valid(bw.word);
                                            for (const auto& p : bw.cells) {
                                                if (is_valid) valid_cells.insert(p);
                                                else invalid_cells.insert(p);
                                            }
                                        }
                                        suspects.clear();
                                        for (const auto& p : invalid_cells) {
                                            if (valid_cells.find(p) == valid_cells.end())
                                                suspects.push_back(p);
                                        }

                                        // Update status
                                        iw_list.clear();
                                        for (const auto& iw : invalid) {
                                            if (!iw_list.empty()) iw_list += ", ";
                                            iw_list += iw.word + " (" + iw.position + ")";
                                        }
                                        if (!invalid.empty()) {
                                            std::string umsg = "{\"status\":\"After gap fill, invalid: "
                                                + json_escape(iw_list) + "\"}\n";
                                            sink.write(umsg.data(), umsg.size());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Word completion: for invalid words with 1-2 suspect cells,
                // try all letter combinations and apply the unique one (if any)
                // that makes all words touching those cells valid.
                // This is deterministic and avoids an extra Gemini round-trip.
                // 1 suspect: 26 tries. 2 suspects: 26*26=676 tries.
                if (!suspects.empty()) {
                    std::set<std::pair<int,int>> suspect_set(
                        suspects.begin(), suspects.end());
                    std::vector<std::pair<int,int>> wc_fixed;
                    std::string wc_detail;

                    // Helper: check if all words touching a set of cells are valid
                    auto all_touching_valid = [&](
                            const std::vector<std::pair<int,int>>& sps) -> bool {
                        auto tw_list = extract_words(dr.cells);
                        for (const auto& tw : tw_list) {
                            bool touches = false;
                            for (const auto& sp2 : sps)
                                for (const auto& [wr, wc] : tw.cells)
                                    if (wr == sp2.first && wc == sp2.second)
                                        { touches = true; break; }
                            if (touches && !g_kwg.is_valid(tw.word)) return false;
                        }
                        return true;
                    };

                    for (const auto& iw : invalid) {
                        // Find suspect cells in this invalid word
                        std::vector<std::pair<int,int>> word_suspects;
                        for (const auto& pos : iw.cells) {
                            if (suspect_set.count(pos))
                                word_suspects.push_back(pos);
                        }
                        if (word_suspects.size() < 1 || word_suspects.size() > 2)
                            continue;

                        // Don't change blank tiles' face letters
                        bool has_blank = false;
                        for (const auto& sp2 : word_suspects)
                            if (dr.cells[sp2.first][sp2.second].is_blank)
                                { has_blank = true; break; }
                        if (has_blank) continue;

                        if (word_suspects.size() == 1) {
                            auto sp = word_suspects[0];
                            auto orig = dr.cells[sp.first][sp.second];
                            char orig_upper = static_cast<char>(std::toupper(
                                static_cast<unsigned char>(orig.letter)));

                            char found = 0;
                            int found_count = 0;
                            for (int li = 0; li < 26 && found_count <= 1; li++) {
                                char l = static_cast<char>('A' + li);
                                if (l == orig_upper) continue;
                                dr.cells[sp.first][sp.second].letter = l;
                                dr.cells[sp.first][sp.second].is_blank = false;
                                if (all_touching_valid({sp}))
                                    { found = l; found_count++; }
                            }
                            dr.cells[sp.first][sp.second] = orig;
                            if (found_count == 0 || found_count > 1) continue;

                            // If the original letter was agreed on by both raw
                            // main OCR and word-crop (current == raw_main),
                            // don't clobber it — defer to visual dict_requery.
                            char raw_upper = 0;
                            if (raw_main_cells[sp.first][sp.second].letter)
                                raw_upper = static_cast<char>(std::toupper(
                                    static_cast<unsigned char>(
                                        raw_main_cells[sp.first][sp.second].letter)));
                            if (raw_upper && raw_upper == orig_upper) {
                                // Both agreed: leave the cell alone, keep it
                                // in suspects so dict_requery can crop-verify.
                                continue;
                            }

                            if (!wc_detail.empty()) wc_detail += ", ";
                            wc_detail +=
                                std::string(1, static_cast<char>('A' + sp.second))
                                + std::to_string(sp.first + 1) + ": "
                                + orig_upper + " -> " + found;
                            dr.cells[sp.first][sp.second].letter = found;
                            dr.cells[sp.first][sp.second].confidence = 0.95f;
                            dr.cells[sp.first][sp.second].is_blank = false;
                            wc_fixed.push_back(sp);
                            suspect_set.erase(sp);

                        } else { // 2 suspects
                            auto sp0 = word_suspects[0], sp1 = word_suspects[1];
                            auto orig0 = dr.cells[sp0.first][sp0.second];
                            auto orig1 = dr.cells[sp1.first][sp1.second];
                            char u0 = static_cast<char>(std::toupper(
                                static_cast<unsigned char>(orig0.letter)));
                            char u1 = static_cast<char>(std::toupper(
                                static_cast<unsigned char>(orig1.letter)));

                            char found0 = 0, found1 = 0;
                            int found_count = 0;
                            for (int li0 = 0; li0 < 26 && found_count <= 1; li0++) {
                                char l0 = static_cast<char>('A' + li0);
                                dr.cells[sp0.first][sp0.second].letter = l0;
                                dr.cells[sp0.first][sp0.second].is_blank = false;
                                for (int li1 = 0; li1 < 26 && found_count <= 1; li1++) {
                                    char l1 = static_cast<char>('A' + li1);
                                    if (l0 == u0 && l1 == u1) continue;
                                    dr.cells[sp1.first][sp1.second].letter = l1;
                                    dr.cells[sp1.first][sp1.second].is_blank = false;
                                    if (all_touching_valid({sp0, sp1}))
                                        { found0 = l0; found1 = l1; found_count++; }
                                }
                            }
                            dr.cells[sp0.first][sp0.second] = orig0;
                            dr.cells[sp1.first][sp1.second] = orig1;
                            if (found_count == 0 || found_count > 1) continue;

                            // If either cell was agreed on by raw OCR and
                            // word-crop, defer both to dict_requery.
                            auto raw_upper2 = [&](const std::pair<int,int>& sp) {
                                char r = raw_main_cells[sp.first][sp.second].letter;
                                return r ? static_cast<char>(std::toupper(
                                    static_cast<unsigned char>(r))) : (char)0;
                            };
                            char r0 = raw_upper2(sp0), r1 = raw_upper2(sp1);
                            if ((r0 && r0 == u0) || (r1 && r1 == u1)) continue;

                            if (!wc_detail.empty()) wc_detail += ", ";
                            wc_detail +=
                                std::string(1, static_cast<char>('A' + sp0.second))
                                + std::to_string(sp0.first + 1) + ": "
                                + u0 + " -> " + found0 + ", "
                                + std::string(1, static_cast<char>('A' + sp1.second))
                                + std::to_string(sp1.first + 1) + ": "
                                + u1 + " -> " + found1;
                            dr.cells[sp0.first][sp0.second].letter = found0;
                            dr.cells[sp0.first][sp0.second].confidence = 0.95f;
                            dr.cells[sp0.first][sp0.second].is_blank = false;
                            dr.cells[sp1.first][sp1.second].letter = found1;
                            dr.cells[sp1.first][sp1.second].confidence = 0.95f;
                            dr.cells[sp1.first][sp1.second].is_blank = false;
                            wc_fixed.push_back(sp0);
                            wc_fixed.push_back(sp1);
                            suspect_set.erase(sp0);
                            suspect_set.erase(sp1);
                        }
                    }

                    if (!wc_fixed.empty()) {
                        std::string msg = "{\"status\":\"Word completion: "
                            + json_escape(wc_detail) + "\"}\n";
                        sink.write(msg.data(), msg.size());

                        // Remove fixed cells from suspects
                        suspects.erase(
                            std::remove_if(suspects.begin(), suspects.end(),
                                [&](const auto& p) {
                                    return suspect_set.find(p)
                                           == suspect_set.end();
                                }),
                            suspects.end());

                        // Rebuild invalid/iw_list for dict_requery prompt
                        all_words = extract_words(dr.cells);
                        invalid.clear();
                        for (const auto& bw : all_words) {
                            if (!g_kwg.is_valid(bw.word)) {
                                InvalidWord iw2;
                                iw2.word = bw.word;
                                iw2.cells = bw.cells;
                                iw2.horizontal = bw.horizontal;
                                if (bw.horizontal)
                                    iw2.position = "row "
                                        + std::to_string(bw.cells[0].first + 1);
                                else
                                    iw2.position = "col "
                                        + std::string(1, static_cast<char>(
                                            'A' + bw.cells[0].second));
                                invalid.push_back(std::move(iw2));
                            }
                        }
                        iw_list.clear();
                        for (const auto& iw2 : invalid) {
                            if (!iw_list.empty()) iw_list += ", ";
                            iw_list += iw2.word + " (" + iw2.position + ")";
                        }
                    }
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
                                "Regular tiles are beige SQUARES with a small "
                                "subscript number (return UPPERCASE). Blank tiles "
                                "are PURPLE CIRCLES (round, not square) with NO "
                                "subscript (return lowercase). If SQUARE with "
                                "subscript, ALWAYS return UPPERCASE. "
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
                                char raw_ltr = raw_main_cells[sp.first][sp.second].letter
                                    ? static_cast<char>(std::toupper(static_cast<unsigned char>(
                                        raw_main_cells[sp.first][sp.second].letter))) : 0;
                                crop_status += "{\"pos\":\""
                                    + std::string(1, static_cast<char>('A' + sp.second))
                                    + std::to_string(sp.first + 1)
                                    + "\",\"cur\":\""
                                    + std::string(1, cur_ltr) + "\""
                                    + (raw_ltr && raw_ltr != cur_ltr
                                        ? ",\"initial\":\"" + std::string(1, raw_ltr) + "\""
                                        : "")
                                    + ",\"img\":\"data:image/png;base64,"
                                    + b64_crop + "\"}";
                            }
                            dict_payload += "]}]}";
                            crop_status += "]}\n";
                            sink.write(crop_status.data(), crop_status.size());

                            // Call Gemini
                            auto gcrd = call_gemini(url, dict_payload,
                                "dict_requery", dict_prompt, 30, 1);
                            if (!gcrd.text.empty()) {
                                std::string sd = gcrd.text;
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

    // Stage snapshot: after word completion + dict requery
    { std::string s="{\"stage\":\"wc\",\"stage_cgp\":\""+json_escape(cells_to_cgp(dr.cells))+"\"}\n"; sink.write(s.data(),s.size()); }

    // --- Rack validation & auto-correction ---
    // Runs after word corrections so the warning reflects the final board state.
    // full bag - board tiles - bag tiles = expected rack
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


    std::string final_json = make_json_response(dr);
    // Inject extra fields before the closing }
    std::string extra;
    if (!gemini_bag.empty())
        extra += ",\"bag\":\"" + json_escape(gemini_bag) + "\"";
    if (!rack_warning.empty())
        extra += ",\"rack_warning\":\"" + json_escape(rack_warning) + "\"";
    if (!invalid_words_json.empty())
        extra += ",\"invalid_words\":" + invalid_words_json;
    // OCR trail: per-cell comparison of raw main / transposed / final
    {
        std::string trail = "[";
        bool first_t = true;
        for (int r = 0; r < 15; r++) {
            for (int c = 0; c < 15; c++) {
                char raw_ch = raw_main_cells[r][c].letter;
                char fin_ch = dr.cells[r][c].letter;
                if (raw_ch == 0 && fin_ch == 0) continue;
                char raw_u = raw_ch ? static_cast<char>(std::toupper(
                    static_cast<unsigned char>(raw_ch))) : 0;
                char fin_u = fin_ch ? static_cast<char>(std::toupper(
                    static_cast<unsigned char>(fin_ch))) : 0;
                // Skip if raw==final
                if (raw_u == fin_u) continue;
                char trans_u = 0;
                if (!first_t) trail += ",";
                first_t = false;
                std::string pos = std::string(1, static_cast<char>('A' + c))
                    + std::to_string(r + 1);
                trail += "{\"pos\":\"" + pos + "\","
                    "\"raw\":\"" + (raw_u ? std::string(1, raw_u) : "") + "\","
                    "\"trans\":\"" + (trans_u ? std::string(1, trans_u) : "") + "\","
                    "\"final\":\"" + (fin_u ? std::string(1, fin_u) : "") + "\"}";
            }
        }
        trail += "]";
        extra += ",\"ocr_trail\":" + trail;
        extra += ",\"raw_main_cgp\":\"" + json_escape(cells_to_cgp(raw_main_cells)) + "\"";
    }
    // Include occupancy grid so UI can show it
    if (have_opencv) {
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
    // Include player names in extra fields if available
    if (!gemini_player1.empty())
        extra += ",\"player1\":\"" + json_escape(gemini_player1) + "\"";
    if (!gemini_player2.empty())
        extra += ",\"player2\":\"" + json_escape(gemini_player2) + "\"";
    if (!extra.empty())
        final_json.insert(final_json.size() - 1, extra); // insert before }
    final_json += "\n";
    sink.write(final_json.data(), final_json.size());

    // --- Woogles game lookup ---
    // Run after emitting the CGP result so the user sees it immediately.
    // Skipped during eval runs (skip_woogles=true) to avoid slowdown.
    if (!skip_woogles && !dr.cgp.empty()) {
        {
            std::string s = "{\"status\":\"Looking up game in Woogles database...\"}\n";
            sink.write(s.data(), s.size());
        }
        // Build JSON input for woogles_lookup.py
        std::string lookup_input = "{\"cgp\":\"" + json_escape(dr.cgp) + "\"";
        lookup_input += ",\"players\":[";
        bool had_player = false;
        if (!gemini_player1.empty()) {
            lookup_input += "\"" + json_escape(gemini_player1) + "\"";
            had_player = true;
        }
        if (!gemini_player2.empty()) {
            if (had_player) lookup_input += ",";
            lookup_input += "\"" + json_escape(gemini_player2) + "\"";
        }
        lookup_input += "]";
        if (gemini_score1 != 0 || gemini_score2 != 0)
            lookup_input += ",\"scores\":["
                + std::to_string(gemini_score1) + ","
                + std::to_string(gemini_score2) + "]";
        lookup_input += "}";

        // Write to a temp file to avoid shell-escaping issues
        char tmp_buf[] = "/tmp/cgpbot_wlu_XXXXXX";
        int tmpfd = mkstemp(tmp_buf);
        if (tmpfd >= 0) {
            const char* d = lookup_input.data();
            size_t rem = lookup_input.size();
            while (rem > 0) {
                ssize_t n = ::write(tmpfd, d, rem);
                if (n <= 0) break;
                d += n; rem -= n;
            }
            ::close(tmpfd);

            std::string cmd = "python3 testgen/scripts/woogles_lookup.py < ";
            cmd += tmp_buf;
            cmd += " 2>/dev/null";
            FILE* pipe = popen(cmd.c_str(), "r");
            std::string result;
            if (pipe) {
                char buf[8192];
                while (fgets(buf, sizeof(buf), pipe)) result += buf;
                pclose(pipe);
            }
            std::remove(tmp_buf);

            // Trim trailing whitespace
            while (!result.empty() &&
                   (result.back() == '\n' || result.back() == '\r' ||
                    result.back() == ' '))
                result.pop_back();

            if (!result.empty()) {
                std::string woogles_line = "{\"woogles\":" + result + "}\n";
                sink.write(woogles_line.data(), woogles_line.size());
            } else {
                std::string woogles_line = "{\"woogles\":null}\n";
                sink.write(woogles_line.data(), woogles_line.size());
            }
        }
    }
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

        // Detect memento (server-rendered share image) from filename
        bool is_memento = file.filename.find("_memento") != std::string::npos;
        // skip_woogles=true skips the Woogles lookup (used during eval runs)
        bool skip_woogles = req.has_file("skip_woogles");

        // Store for test case saving
        {
            std::lock_guard<std::mutex> lk(g_last_image_mutex);
            g_last_image = buf;
        }

        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_chunked_content_provider(
            "application/x-ndjson",
            [buf, is_memento, skip_woogles](size_t /*offset*/, httplib::DataSink& sink) {
                stream_analyze_gemini(*buf, sink, is_memento, skip_woogles);
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

    // POST /eval-save — save eval results to testdata/last_eval.json
    svr.Post("/eval-save", [](const httplib::Request& req, httplib::Response& res) {
        fs::create_directories("testdata");
        // Inject timestamp
        std::string body = req.body;
        // Insert timestamp field after opening {
        auto pos = body.find('{');
        if (pos != std::string::npos) {
            time_t now = time(nullptr);
            body.insert(pos + 1, "\"timestamp\":" + std::to_string(now) + ",");
        }
        std::ofstream ofs("testdata/last_eval.json");
        ofs << body;
        res.set_content("{\"ok\":true}", "application/json");
    });

    // GET /eval-summary — compact summary from last_eval.json
    svr.Get("/eval-summary", [](const httplib::Request&, httplib::Response& res) {
        std::string path = "testdata/last_eval.json";
        if (!fs::exists(path)) { res.status = 404; return; }
        std::ifstream ifs(path);
        std::string body((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());
        res.set_content(body, "application/json");
    });

    // GET /eval — full eval results page
    svr.Get("/eval", [](const httplib::Request&, httplib::Response& res) {
        std::string path = "testdata/last_eval.json";
        std::string json;
        if (fs::exists(path)) {
            std::ifstream ifs(path);
            json.assign(std::istreambuf_iterator<char>(ifs),
                        std::istreambuf_iterator<char>());
        }
        // Escape for embedding in JS
        std::string json_esc;
        for (char c : json) {
            if (c == '\\') json_esc += "\\\\";
            else if (c == '`') json_esc += "\\`";
            else if (c == '$') json_esc += "\\$";
            else json_esc += c;
        }
        std::string html = R"html(<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>CGP Bot &mdash; Eval Results</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  background:#1a1a2e;color:#e0e0e0;min-height:100vh;padding:24px}
h1{font-size:1.2rem;margin-bottom:6px;color:#fff}
.back{font-size:.8rem;color:#58a6ff;text-decoration:none;display:inline-block;margin-bottom:20px}
.summary{background:#16213e;border-radius:8px;padding:14px 20px;margin-bottom:20px;border:1px solid #2a2a4a;display:flex;gap:32px;align-items:center;flex-wrap:wrap}
.stat-val{font-size:1.4rem;font-weight:700;color:#4c4}
.stat-lbl{font-size:.7rem;color:#888;text-transform:uppercase;letter-spacing:.05em}
table{width:100%;border-collapse:collapse;font-size:.82rem;margin-bottom:20px}
th,td{padding:6px 10px;border:1px solid #2a2a4a;text-align:left}
th{background:#16213e;color:#888;font-size:.72rem;text-transform:uppercase}
tr:hover td{background:#1a2540}
.pass{color:#4c4}.fail{color:#f44}
.ts{font-size:.72rem;color:#666}
.case-boards{display:flex;gap:24px;flex-wrap:wrap;margin:16px 0 8px}
.board-box{text-align:center}
.board-lbl{font-size:.7rem;color:#888;margin-bottom:4px}
.mini-board{display:grid;grid-template-columns:repeat(15,13px);grid-template-rows:repeat(15,13px);gap:1px;background:#222;border:1px solid #333;border-radius:3px}
.mb{display:flex;align-items:center;justify-content:center;font-size:6px;font-weight:700;color:#111}
.mb.tile{background:#f5deb3;color:#111}
.mb.blank-tile{background:#c8b888;color:#555}
.mb.tw{background:#c0392b}.mb.dw{background:#e88b8b}.mb.tl{background:#2980b9}.mb.dl{background:#7ec8e3}
.mb.center{background:#e88b8b}.mb.normal{background:#1b7a3d}
.mb.wrong-exp{outline:2px solid #f44;outline-offset:-1px;z-index:1}
.mb.wrong-got{outline:2px solid #f88;outline-offset:-1px;z-index:1}
.case-section{background:#16213e;border-radius:8px;padding:16px;margin-bottom:16px;border:1px solid #2a2a4a}
.case-title{font-size:.85rem;font-weight:600;margin-bottom:4px}
.diff-text{font-size:.72rem;color:#f88;font-family:'SF Mono','Fira Code',monospace;margin-bottom:8px}
</style></head><body>
<a href="/" class="back">&larr; Back to test bench</a>
<h1>Gemini Eval Results</h1>
<div id="root"><p style="color:#666">No eval data found.</p></div>
<script>
const PREMIUM=[
  [4,0,0,1,0,0,0,4,0,0,0,1,0,0,4],[0,3,0,0,0,2,0,0,0,2,0,0,0,3,0],
  [0,0,3,0,0,0,1,0,1,0,0,0,3,0,0],[1,0,0,3,0,0,0,1,0,0,0,3,0,0,1],
  [0,0,0,0,3,0,0,0,0,0,3,0,0,0,0],[0,2,0,0,0,2,0,0,0,2,0,0,0,2,0],
  [0,0,1,0,0,0,1,0,1,0,0,0,1,0,0],[4,0,0,1,0,0,0,5,0,0,0,1,0,0,4],
  [0,0,1,0,0,0,1,0,1,0,0,0,1,0,0],[0,2,0,0,0,2,0,0,0,2,0,0,0,2,0],
  [0,0,0,0,3,0,0,0,0,0,3,0,0,0,0],[1,0,0,3,0,0,0,1,0,0,0,3,0,0,1],
  [0,0,3,0,0,0,1,0,1,0,0,0,3,0,0],[0,3,0,0,0,2,0,0,0,2,0,0,0,3,0],
  [4,0,0,1,0,0,0,4,0,0,0,1,0,0,4]];
const PCLS=['normal','dl','tl','dw','tw','center'];
function parseCGPBoard(cgp){
  const b=Array.from({length:15},()=>Array(15).fill(''));
  const rows=(cgp.split(' ')[0]).split('/');
  for(let r=0;r<Math.min(rows.length,15);r++){
    let c=0,i=0;
    while(i<rows[r].length&&c<15){
      const ch=rows[r][i];
      if(ch>='0'&&ch<='9'){let n=0;while(i<rows[r].length&&rows[r][i]>='0'&&rows[r][i]<='9')n=n*10+parseInt(rows[r][i++]);c+=n;}
      else{b[r][c++]=ch;i++;}
    }
  }
  return b;
}
function renderMiniBoard(board,wrongSet,cls){
  let h=`<div class="mini-board">`;
  for(let r=0;r<15;r++)for(let c=0;c<15;c++){
    const ch=board[r][c];
    const key=String.fromCharCode(65+c)+(r+1);
    const wrong=wrongSet&&wrongSet.has(key);
    if(!ch){const p=PREMIUM[r][c];h+=`<div class="mb ${PCLS[p]}${wrong?' '+cls:''}"></div>`;}
    else{const bl=ch===ch.toLowerCase();h+=`<div class="mb ${bl?'blank-tile':'tile'}${wrong?' '+cls:''}">${ch.toUpperCase()}</div>`;}
  }
  h+=`</div>`;
  return h;
}
function relTime(ts){
  const d=Math.floor(Date.now()/1000-ts);
  if(d<60)return 'just now';if(d<3600)return Math.round(d/60)+'m ago';
  if(d<86400)return Math.round(d/3600)+'h ago';return Math.round(d/86400)+'d ago';
}
const DATA=)html" + (json.empty() ? "null" : json_esc) + R"html(;
if(DATA){
  const pct=(DATA.correct/DATA.total_cells*100).toFixed(1);
  const wrong=DATA.total_cells-DATA.correct;
  const ts=new Date(DATA.timestamp*1000).toLocaleString();
  let h=`<div class="summary">
    <div><div class="stat-val">${pct}%</div><div class="stat-lbl">Board accuracy</div></div>
    <div><div class="stat-val" style="color:${wrong?'#f88':'#4c4'}">${wrong}</div><div class="stat-lbl">Wrong cells</div></div>
    <div><div class="stat-val">${DATA.scores_correct}/${DATA.scores_total}</div><div class="stat-lbl">Scores correct</div></div>
    <div style="margin-left:auto"><div class="ts">${ts}</div><div class="ts">${relTime(DATA.timestamp)}</div></div>
  </div>`;
  // Summary table
  h+=`<table><tr><th>Case</th><th>Cells</th><th>Correct</th><th>Wrong</th><th>Board%</th><th style="color:#68a">Occ%</th><th style="color:#68a">Raw%</th><th style="color:#68a">Align%</th><th style="color:#68a">Trans%</th><th style="color:#68a">Retry%</th><th style="color:#68a">WC%</th><th>Exp scores</th><th>Got scores</th><th>&#9654;</th></tr>`;
  for(const c of DATA.cases||[]){
    const cp=c.cells?((c.correct/c.cells)*100).toFixed(1)+'%':'—';
    const scOk=c.exp_scores&&c.got_scores?(c.exp_scores===c.got_scores?'<span class="pass">&#10003;</span>':'<span class="fail">&#10007;</span>'):'—';
    const expSc=c.exp_scores||'—';const gotSc=c.got_scores||'—';
    const scMismatch=c.exp_scores&&c.got_scores&&c.exp_scores!==c.got_scores;
    const sa=c.stage_accs||{};
    function s(k){return sa[k]!=null?sa[k]+'%':'—';}
    const stageCols=`<td>${s('occ')}</td><td>${s('raw')}</td><td>${s('realigned')}</td><td>${s('trans')}</td><td>${s('retry')}</td><td>${s('wc')}</td>`;
    h+=`<tr><td>${c.name}</td><td>${c.cells||'—'}</td><td>${c.correct||'—'}</td><td class="${c.wrong>0?'fail':'pass'}">${c.wrong||'0'}</td><td>${cp}</td>${stageCols}<td style="font-family:monospace;${scMismatch?'color:#f88':''}">${expSc}</td><td style="font-family:monospace;${scMismatch?'color:#f88':''}">${gotSc}</td><td>${scOk}</td></tr>`;
    if(c.diffs&&c.diffs.length)h+=`<tr><td colspan="14" style="color:#f88;font-family:'SF Mono',monospace;font-size:.7rem;padding:2px 8px">&nbsp;&nbsp;${c.diffs.join('&nbsp;&nbsp;')}</td></tr>`;
  }
  h+=`<tr style="font-weight:bold;border-top:2px solid #444">
    <td>TOTAL</td><td>${DATA.total_cells}</td><td>${DATA.correct}</td>
    <td class="${wrong?'fail':'pass'}">${wrong}</td><td>${pct}%</td>
    <td colspan="6"></td>
    <td colspan="3" style="color:#ccc">${DATA.scores_correct}/${DATA.scores_total} scores &#10003;</td></tr></table>`;
  // Failing case boards
  for(const c of DATA.cases||[]){
    if(!c.wrong||!c.exp_cgp||!c.got_cgp)continue;
    const expBoard=parseCGPBoard(c.exp_cgp);
    const gotBoard=parseCGPBoard(c.got_cgp);
    // build sets of wrong positions for each board
    const expWrong=new Set(),gotWrong=new Set();
    for(const d of c.diffs||[]){
      const pos=d.split(':')[0];expWrong.add(pos);gotWrong.add(pos);
    }
    h+=`<div class="case-section">
      <div class="case-title">${c.name} &mdash; ${c.wrong} wrong cell${c.wrong>1?'s':''}</div>
      <div class="diff-text">${(c.diffs||[]).join('&nbsp;&nbsp;')}</div>
      <div class="case-boards">
        <div class="board-box"><div class="board-lbl">Expected</div>${renderMiniBoard(expBoard,expWrong,'wrong-exp')}</div>
        <div class="board-box"><div class="board-lbl">Got</div>${renderMiniBoard(gotBoard,gotWrong,'wrong-got')}</div>
      </div>
    </div>`;
  }
  document.getElementById('root').innerHTML=h;
}
</script></body></html>)html";
        res.set_content(html, "text/html");
    });

    // GET /testdata-list -> [{name, has_expected, has_image}]
    svr.Get("/testdata-list", [](const httplib::Request&, httplib::Response& res) {
        std::string json = "[";
        bool first = true;
        if (fs::exists("testdata")) {
            std::vector<fs::directory_entry> entries;
            for (const auto& e : fs::directory_iterator("testdata"))
                if (e.path().extension() == ".cgp") entries.push_back(e);
            std::sort(entries.begin(), entries.end());
            for (const auto& entry : entries) {
                std::string name = entry.path().stem().string();
                // find image with any common extension
                std::string img_path;
                for (auto& ext : std::vector<std::string>{".png", ".jpg", ".jpeg"}) {
                    std::string p = "testdata/" + name + ext;
                    if (fs::exists(p)) { img_path = p; break; }
                }
                if (img_path.empty()) continue;
                if (!first) json += ",";
                first = false;
                json += "{\"name\":\"" + json_escape(name) + "\""
                    ",\"has_expected\":true"
                    ",\"has_image\":true}";
            }
        }
        json += "]";
        res.set_content(json, "application/json");
    });

    // GET /testdata-image/:name -> serve the image file
    svr.Get("/testdata-image/(.*)", [](const httplib::Request& req, httplib::Response& res) {
        std::string name = req.matches[1];
        // sanitize: no path traversal
        if (name.find('/') != std::string::npos || name.find("..") != std::string::npos) {
            res.status = 400; return;
        }
        std::string img_path;
        std::string mime;
        for (auto& [ext, m] : std::vector<std::pair<std::string,std::string>>{
                {".png","image/png"},{".jpg","image/jpeg"},{".jpeg","image/jpeg"}}) {
            std::string p = "testdata/" + name + ext;
            if (fs::exists(p)) { img_path = p; mime = m; break; }
        }
        if (img_path.empty()) { res.status = 404; return; }
        std::ifstream ifs(img_path, std::ios::binary);
        std::vector<char> buf((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());
        res.set_content(std::string(buf.begin(), buf.end()), mime);
    });

    // GET /testdata-cgp/:name -> serve the expected CGP text
    svr.Get("/testdata-cgp/(.*)", [](const httplib::Request& req, httplib::Response& res) {
        std::string name = req.matches[1];
        if (name.find('/') != std::string::npos || name.find("..") != std::string::npos) {
            res.status = 400; return;
        }
        std::string path = "testdata/" + name + ".cgp";
        if (!fs::exists(path)) { res.status = 404; return; }
        std::ifstream ifs(path);
        std::string cgp;
        std::getline(ifs, cgp);
        res.set_content(cgp, "text/plain");
    });

    const char* port_env = std::getenv("PORT");
    int port = port_env ? std::atoi(port_env) : 8080;

    std::cout << "CGP test bench -> http://localhost:" << port << "\n";

    if (!svr.listen("127.0.0.1", port)) {
        std::cerr << "Failed to bind to port " << port << "\n";
        return 1;
    }
}
