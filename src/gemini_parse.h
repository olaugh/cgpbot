#pragma once
// Gemini response parsing utilities — shared between testapp.cpp and tests.

#include <queue>
#include <string>
#include <utility>
#include <vector>
#include "board.h"

// ---------------------------------------------------------------------------
// Extract the text content from a Gemini API JSON response.
// Finds the LAST "text" key — Gemini 2.5 models may have thinking parts
// before the actual response, each with their own "text" key.
// ---------------------------------------------------------------------------
static inline std::string extract_gemini_text(const std::string& json) {
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
// Handles rows with fewer or more than 15 elements (Gemini sometimes returns
// 14 or 16 elements per row).
// ---------------------------------------------------------------------------
static inline bool parse_gemini_board(const std::string& text,
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

    int rows_parsed = 0;
    for (int r = 0; r < 15; r++) {
        skip_ws();
        if (r > 0) {
            if (pos < s.size() && s[pos] == ',') pos++;
            skip_ws();
        }
        if (pos >= s.size() || s[pos] != '[') break;
        pos++; // skip inner [

        // Read all elements until ']', storing up to 15 in cells
        int c = 0;
        bool first = true;
        while (pos < s.size()) {
            skip_ws();
            if (pos < s.size() && s[pos] == ']') { pos++; break; }
            if (!first) {
                if (pos < s.size() && s[pos] == ',') pos++;
                skip_ws();
            }
            first = false;

            // Parse one element: "X", null, or skip unknown
            CellResult cr = {};
            if (pos < s.size() && s[pos] == '"') {
                pos++; // skip opening quote
                if (pos < s.size()) {
                    char ch = s[pos];
                    pos++;
                    if (pos < s.size() && s[pos] == '"') pos++;
                    cr.letter = ch;
                    cr.confidence = 1.0f;
                    if (ch >= 'a' && ch <= 'z') cr.is_blank = true;
                }
            } else if (pos + 4 <= s.size() && s.substr(pos, 4) == "null") {
                pos += 4;
            } else if (pos < s.size()) {
                pos++; // unknown token — advance to avoid infinite loop
            }

            if (c < 15)
                cells[r][c] = cr;
            c++;
        }
        rows_parsed++;
    }

    return rows_parsed >= 10; // accept even if a few rows are weird
}

// ---------------------------------------------------------------------------
// Parse a retry/revalidation response: JSON array of letters and nulls.
// Input: Gemini text like '["F", null, "s", "A"]' or with code fences.
// Returns vector of CellResult — letter=0 for null/empty elements.
// Confidence: 0.75 for disputed cells, 0.8 for missing cells.
// ---------------------------------------------------------------------------
struct RetryCell {
    int r, c;
    bool is_disputed;  // true=disputed (Gemini found, color says empty)
};

static inline std::vector<CellResult> parse_retry_response(
        const std::string& text, const std::vector<RetryCell>& positions) {
    std::vector<CellResult> results(positions.size());

    std::string s = text;
    // Strip markdown code fences if present
    auto f1 = s.find("```");
    if (f1 != std::string::npos) {
        auto nl = s.find('\n', f1);
        if (nl != std::string::npos) s = s.substr(nl + 1);
        auto f2 = s.rfind("```");
        if (f2 != std::string::npos) s = s.substr(0, f2);
    }

    // Find opening [
    size_t idx = 0;
    while (idx < s.size() && s[idx] != '[') idx++;
    if (idx >= s.size()) return results; // no array found
    idx++; // skip [

    size_t mi = 0;
    while (idx < s.size() && mi < positions.size()) {
        // Skip whitespace and commas
        while (idx < s.size()
               && (s[idx] == ' ' || s[idx] == ','
                   || s[idx] == '\n' || s[idx] == '\r'
                   || s[idx] == '\t'))
            idx++;
        if (idx >= s.size() || s[idx] == ']') break;

        if (s[idx] == '"') {
            idx++; // skip opening "
            if (idx < s.size()) {
                char ltr = s[idx];
                CellResult cr = {};
                cr.letter = ltr;
                cr.confidence = positions[mi].is_disputed ? 0.75f : 0.8f;
                cr.is_blank = (ltr >= 'a' && ltr <= 'z');
                results[mi] = cr;
                idx++;
                if (idx < s.size() && s[idx] == '"') idx++;
            }
        } else if (idx + 4 <= s.size() && s.substr(idx, 4) == "null") {
            idx += 4;
            // results[mi] stays default (letter=0) — confirmed empty
        }
        mi++;
    }

    return results;
}

// ---------------------------------------------------------------------------
// Build CGP board string from CellResult grid.
// ---------------------------------------------------------------------------
static inline std::string cells_to_cgp(const CellResult cells[15][15]) {
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
        if (empty > 0) result += std::to_string(empty);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Board connectivity check — find islands not connected to center (7,7).
// ---------------------------------------------------------------------------
struct ConnectivityResult {
    bool center_empty = false;
    std::vector<std::pair<int,int>> main_component;
    std::vector<std::vector<std::pair<int,int>>> islands;
    std::vector<std::pair<int,int>> bridge_candidates;
};

static inline ConnectivityResult check_board_connectivity(
        const CellResult cells[15][15]) {
    ConnectivityResult result;

    // Count tiles
    int tile_count = 0;
    for (int r = 0; r < 15; r++)
        for (int c = 0; c < 15; c++)
            if (cells[r][c].letter != 0) tile_count++;
    if (tile_count == 0) return result;

    // Check center
    if (cells[7][7].letter == 0) result.center_empty = true;

    // BFS from (7,7) over occupied cells
    bool visited[15][15] = {};
    static constexpr int DR[] = {-1, 1, 0, 0};
    static constexpr int DC[] = {0, 0, -1, 1};

    if (cells[7][7].letter != 0) {
        std::queue<std::pair<int,int>> q;
        q.push({7, 7});
        visited[7][7] = true;
        while (!q.empty()) {
            auto [r, c] = q.front(); q.pop();
            result.main_component.push_back({r, c});
            for (int d = 0; d < 4; d++) {
                int nr = r + DR[d], nc = c + DC[d];
                if (nr >= 0 && nr < 15 && nc >= 0 && nc < 15
                    && !visited[nr][nc] && cells[nr][nc].letter != 0) {
                    visited[nr][nc] = true;
                    q.push({nr, nc});
                }
            }
        }
    }

    // Find islands — occupied cells not visited
    for (int r = 0; r < 15; r++)
        for (int c = 0; c < 15; c++) {
            if (cells[r][c].letter != 0 && !visited[r][c]) {
                // BFS this island
                std::vector<std::pair<int,int>> island;
                std::queue<std::pair<int,int>> q;
                q.push({r, c});
                visited[r][c] = true;
                while (!q.empty()) {
                    auto [ir, ic] = q.front(); q.pop();
                    island.push_back({ir, ic});
                    for (int d = 0; d < 4; d++) {
                        int nr = ir + DR[d], nc = ic + DC[d];
                        if (nr >= 0 && nr < 15 && nc >= 0 && nc < 15
                            && !visited[nr][nc] && cells[nr][nc].letter != 0) {
                            visited[nr][nc] = true;
                            q.push({nr, nc});
                        }
                    }
                }
                result.islands.push_back(std::move(island));
            }
        }

    // Bridge candidates — empty cells adjacent to island tiles
    bool bridge_added[15][15] = {};
    for (const auto& island : result.islands) {
        for (const auto& [r, c] : island) {
            for (int d = 0; d < 4; d++) {
                int nr = r + DR[d], nc = c + DC[d];
                if (nr >= 0 && nr < 15 && nc >= 0 && nc < 15
                    && cells[nr][nc].letter == 0 && !bridge_added[nr][nc]) {
                    result.bridge_candidates.push_back({nr, nc});
                    bridge_added[nr][nc] = true;
                }
            }
        }
    }

    // If center is empty, add (7,7) as a bridge candidate
    if (result.center_empty && !bridge_added[7][7]) {
        result.bridge_candidates.push_back({7, 7});
    }

    return result;
}

// ---------------------------------------------------------------------------
// Find single-cell gaps between letter sequences (potential missing tiles).
// ---------------------------------------------------------------------------
struct GapCandidate {
    int r, c;
    bool horizontal;
};

static inline std::vector<GapCandidate> find_gap_candidates(
        const CellResult cells[15][15]) {
    std::vector<GapCandidate> gaps;

    // Horizontal: scan each row for [2+ letters][empty][2+ letters]
    for (int r = 0; r < 15; r++) {
        for (int c = 1; c < 14; c++) {
            if (cells[r][c].letter == 0
                && cells[r][c-1].letter != 0
                && cells[r][c+1].letter != 0) {
                // Require 2+ letters on the left side
                bool left_ok = (c >= 2 && cells[r][c-2].letter != 0);
                // Require 2+ letters on the right side
                bool right_ok = (c <= 12 && cells[r][c+2].letter != 0);
                if (left_ok && right_ok)
                    gaps.push_back({r, c, true});
            }
        }
    }

    // Vertical: scan each column for [2+ letters][empty][2+ letters]
    for (int c = 0; c < 15; c++) {
        for (int r = 1; r < 14; r++) {
            if (cells[r][c].letter == 0
                && cells[r-1][c].letter != 0
                && cells[r+1][c].letter != 0) {
                // Require 2+ letters above
                bool above_ok = (r >= 2 && cells[r-2][c].letter != 0);
                // Require 2+ letters below
                bool below_ok = (r <= 12 && cells[r+2][c].letter != 0);
                if (above_ok && below_ok)
                    gaps.push_back({r, c, false});
            }
        }
    }

    return gaps;
}
