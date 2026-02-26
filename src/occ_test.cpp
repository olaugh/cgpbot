// Occupancy mask accuracy: compare is_tile() results against ground truth CGP
#include "board.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// Parse CGP board section into a 15x15 occupancy grid
// Returns true if cell has a tile (letter A-Z or a-z or ?)
static void parse_cgp_occupancy(const std::string& cgp, bool occ[15][15]) {
    std::memset(occ, 0, sizeof(bool) * 225);
    // Board is the first token (before first space)
    auto sp = cgp.find(' ');
    std::string board = (sp != std::string::npos) ? cgp.substr(0, sp) : cgp;

    int row = 0, col = 0;
    for (size_t i = 0; i < board.size() && row < 15; i++) {
        char ch = board[i];
        if (ch == '/') {
            row++;
            col = 0;
        } else if (ch >= '0' && ch <= '9') {
            // Could be multi-digit number
            int n = ch - '0';
            while (i + 1 < board.size() && board[i+1] >= '0' && board[i+1] <= '9') {
                n = n * 10 + (board[++i] - '0');
            }
            col += n;
        } else if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')) {
            if (row < 15 && col < 15)
                occ[row][col] = true;
            col++;
        }
    }
}

int main(int argc, char* argv[]) {
    std::setbuf(stdout, nullptr);  // unbuffered output
    if (argc < 2) {
        std::cerr << "Usage: occ_test <testdata_dir> [filter]\n";
        return 1;
    }
    std::string dir = argv[1];
    std::string filter = argc >= 3 ? argv[2] : "";

    struct Result {
        std::string name;
        int total_tiles, false_pos, false_neg;
    };
    std::vector<Result> results;

    int grand_fp = 0, grand_fn = 0, grand_tiles = 0, grand_empty = 0;
    int n_files = 0;

    for (auto& entry : fs::directory_iterator(dir)) {
        std::string path = entry.path().string();
        std::string name = entry.path().stem().string();
        std::string ext = entry.path().extension().string();

        if (ext != ".png" && ext != ".jpg") continue;
        if (!filter.empty() && name.find(filter) == std::string::npos) continue;

        // Find matching CGP file
        std::string cgp_path = dir + "/" + name + ".cgp";
        if (!fs::exists(cgp_path)) continue;

        // Read CGP
        std::ifstream cgp_ifs(cgp_path);
        std::string cgp_line;
        std::getline(cgp_ifs, cgp_line);

        bool gt_occ[15][15];
        parse_cgp_occupancy(cgp_line, gt_occ);

        // Run board detection
        std::ifstream ifs(path, std::ios::binary);
        std::vector<uint8_t> data(std::istreambuf_iterator<char>(ifs), {});
        auto dr = process_board_image_debug(data);

        // Compare occupancy
        int fp = 0, fn = 0, tiles = 0, empty = 0;
        for (int r = 0; r < 15; r++) {
            for (int c = 0; c < 15; c++) {
                bool detected = (dr.cells[r][c].letter != 0);
                bool actual = gt_occ[r][c];
                if (actual) tiles++;
                else empty++;
                if (detected && !actual) fp++;
                if (!detected && actual) fn++;
            }
        }

        grand_fp += fp;
        grand_fn += fn;
        grand_tiles += tiles;
        grand_empty += empty;
        n_files++;

        std::fprintf(stderr, "\r%d files processed...", n_files);
        if (fp > 0 || fn > 0) {
            std::printf("%-45s tiles=%3d  FP=%2d FN=%2d", name.c_str(), tiles, fp, fn);
            // Print which cells are wrong
            if (fp + fn <= 10) {
                std::printf("  ");
                for (int r = 0; r < 15; r++)
                    for (int c = 0; c < 15; c++) {
                        bool detected = (dr.cells[r][c].letter != 0);
                        bool actual = gt_occ[r][c];
                        if (detected && !actual)
                            std::printf(" +%c%d", 'A'+c, r+1);
                        if (!detected && actual)
                            std::printf(" -%c%d", 'A'+c, r+1);
                    }
            }
            std::printf("\n");
            results.push_back({name, tiles, fp, fn});
        }
    }

    std::printf("\n=== Summary (%d files) ===\n", n_files);
    std::printf("Total tiles: %d  Total empty: %d\n", grand_tiles, grand_empty);
    std::printf("False positives: %d (%.2f%% of empty)\n", grand_fp,
        grand_empty > 0 ? 100.0 * grand_fp / grand_empty : 0.0);
    std::printf("False negatives: %d (%.2f%% of tiles)\n", grand_fn,
        grand_tiles > 0 ? 100.0 * grand_fn / grand_tiles : 0.0);
    std::printf("Perfect masks: %d / %d\n", n_files - (int)results.size(), n_files);
}
