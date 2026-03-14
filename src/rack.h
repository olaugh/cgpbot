#pragma once

#include "board.h"

#include <string>
#include <vector>

#include <opencv2/core.hpp>

// Rack tile detection result.
struct RackTile {
    cv::Rect rect;
    std::vector<uint8_t> png;
    bool is_blank;
};

// Scrabble tile distribution (max count per letter in a standard game).
extern const int RACK_TILE_DIST[26];

// Parse rack from CGP string. Rack is the token after the board rows:
// "board_rows RACK/ scores lex ...;" -> returns rack string (e.g. "CEJOTX?")
std::string parse_cgp_rack(const std::string& cgp);

// Sort a rack string for comparison (order doesn't matter).
std::string sort_rack(const std::string& rack);

// Detect whether the board is in light mode or dark mode.
bool detect_board_mode(const std::vector<uint8_t>& image_data,
                       int bx, int by, int cell_sz);

// Detect rack tiles below the board.
std::vector<RackTile> detect_rack_tiles(
    const std::vector<uint8_t>& image_data,
    int bx, int by, int cell_sz, bool is_light_mode);

// Classify a rack tile: decode PNG, trim bottom 15%, center-crop to square,
// classify with CNN. Returns full CellResult (including top-5 candidates).
CellResult classify_rack_tile_full(const RackTile& rt);

// Refine rack classification using remaining tile pool constraints.
void refine_rack(CellResult rack_results[], int n_tiles,
                 const CellResult board_cells[15][15]);

// Alphagram tiebreaker: prefer top-5 candidates that maintain sorted order.
void alphagram_tiebreak(CellResult rack_results[], int n_tiles);

// Draw rack tile detections on the debug image.
void draw_rack_debug(std::vector<uint8_t>& debug_png,
                     const std::vector<RackTile>& rack_tiles);
