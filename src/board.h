#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// Per-cell OCR result.
struct CellResult {
    char letter = 0;      // 0 = empty, A-Z = tile, a-z = blank tile
    float confidence = 0;  // match confidence [0,1]
    bool is_blank = false; // blank tile (no point value)
    int subscript = 0;    // detected point-value subscript (1-10), 0 = unread
    // Top-5 match candidates (best first, for debugging/tooltips)
    char cand_letters[5] = {};
    float cand_scores[5] = {};
};

// Full board state from vision pipeline.
struct BoardState {
    cv::Rect board_rect;
    int cell_size = 0;
    CellResult cells[15][15] = {};
    std::string cgp;
    std::string log;
};

// Get the valid letters for a given Scrabble point value (0 = unknown).
const char* scrabble_letters_for_points(int pts);

// Debug output bundle.
struct DebugResult {
    std::string cgp;
    std::vector<uint8_t> debug_png;
    std::string log;
    CellResult cells[15][15] = {};
};

// Progress callback: (status_message, log_so_far, debug_png_so_far).
// Called after each processing stage to report incremental progress.
using ProgressCallback = std::function<void(const char* status,
                                             const std::string& log,
                                             const std::vector<uint8_t>& debug_png)>;

// Process a board screenshot and return a CGP string.
std::string process_board_image(const std::vector<uint8_t>& image_data);

// Process with debug overlay image and log. Optional progress callback.
DebugResult process_board_image_debug(const std::vector<uint8_t>& image_data,
                                       ProgressCallback on_progress = nullptr);
