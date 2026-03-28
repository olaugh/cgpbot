// Extract labeled rack tile crops from testdata (image + CGP pairs) for CNN training.
// Uses detect_rack_tiles to find tile positions, prepare_rack_crop to normalize,
// and CGP rack field for ground truth labels.
//
// Output: rack_training_data/<LETTER>/<theme>_<stem>_t<I>.png
//         rack_training_data/?/<theme>_<stem>_t<I>.png  (blank tiles)
#include "board.h"
#include "rack.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

static void parse_cgp_letters(const std::string& cgp, char letters[15][15]) {
    std::memset(letters, 0, 15 * 15);
    auto sp = cgp.find(' ');
    std::string board = (sp != std::string::npos) ? cgp.substr(0, sp) : cgp;
    int row = 0, col = 0;
    for (size_t i = 0; i < board.size() && row < 15; i++) {
        char ch = board[i];
        if (ch == '/') { row++; col = 0; }
        else if (ch >= '0' && ch <= '9') {
            int n = ch - '0';
            while (i + 1 < board.size() && board[i+1] >= '0' && board[i+1] <= '9')
                n = n * 10 + (board[++i] - '0');
            col += n;
        } else if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')) {
            if (row < 15 && col < 15) letters[row][col] = ch;
            col++;
        }
    }
}

static std::string classify_theme(const std::string& name) {
    if (name.find("_memento") != std::string::npos) return "memento";
    if (name.find("_mahogany_desktop") != std::string::npos) return "mahogany_desk";
    if (name.find("_mahogany_mobile") != std::string::npos) return "mahogany_mob";
    if (name.find("_light_desktop") != std::string::npos) return "light_desk";
    if (name.find("_dark_desktop") != std::string::npos) return "dark_desk";
    if (name.find("_light_mobile") != std::string::npos) return "light_mob";
    if (name.find("_dark_mobile") != std::string::npos) return "dark_mob";
    // Check for jpeg/lowjpeg variants
    std::string base = name;
    if (base.find("_lowjpeg") != std::string::npos) {
        base = base.substr(0, base.find("_lowjpeg"));
        return classify_theme(base) + "_lowjpeg";
    }
    if (base.find("_jpeg") != std::string::npos) {
        base = base.substr(0, base.find("_jpeg"));
        return classify_theme(base) + "_jpeg";
    }
    return "original";
}

// Prepare rack crop for saving: apply prepare_rack_crop logic
// (adaptive bottom trim capped at 25%, then square via center-crop or padding).
// This matches what classify_rack_tile_full does before CNN inference.
static cv::Mat prepare_rack_crop_for_training(const cv::Mat& crop) {
    cv::Mat gray;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);

    int trim_bot = crop.rows * 15 / 100;
    int max_trim = crop.rows / 4;
    for (int y = crop.rows - 1; y > crop.rows / 2; y--) {
        cv::Mat row_data = gray.row(y);
        cv::Scalar m, s;
        cv::meanStdDev(row_data, m, s);
        if (s[0] < 15) {
            int extra_trim = crop.rows - y;
            if (extra_trim > trim_bot && extra_trim <= max_trim)
                trim_bot = extra_trim;
        } else {
            break;
        }
    }
    int new_h = std::max(1, crop.rows - trim_bot);
    int new_w = crop.cols;

    int x_off = 0;
    if (new_w > new_h) {
        x_off = (new_w - new_h) / 2;
        new_w = new_h;
    }
    cv::Rect letter_roi(x_off, 0, new_w, new_h);
    letter_roi &= cv::Rect(0, 0, crop.cols, crop.rows);
    cv::Mat sq = crop(letter_roi);

    if (new_h > new_w * 5 / 4) {
        int target = new_h;
        int pad_left = (target - new_w) / 2;
        int pad_right = target - new_w - pad_left;
        cv::Mat padded;
        cv::copyMakeBorder(sq, padded, 0, 0, pad_left, pad_right,
                           cv::BORDER_REPLICATE);
        return padded;
    }
    return sq.clone();
}

int main(int argc, char* argv[]) {
    std::setbuf(stdout, nullptr);
    if (argc < 2) {
        std::cerr << "Usage: extract_rack_crops <testdata_dir> [output_dir]\n";
        return 1;
    }
    std::string dir = argv[1];
    std::string out_dir = (argc >= 3) ? argv[2] : "rack_training_data";

    // Create output directories for each letter + blank
    for (int i = 0; i < 26; i++) {
        fs::create_directories(out_dir + "/" + std::string(1, 'A' + i));
    }
    fs::create_directories(out_dir + "/_blank");

    int n_files = 0, n_tiles = 0, n_blanks = 0, n_skipped = 0;
    int n_matched = 0, n_unmatched = 0;

    for (auto& entry : fs::directory_iterator(dir)) {
        std::string path = entry.path().string();
        std::string name = entry.path().stem().string();
        std::string ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpg") continue;

        std::string cgp_path = dir + "/" + name + ".cgp";
        if (!fs::exists(cgp_path)) continue;

        // Read CGP ground truth
        std::ifstream cgp_ifs(cgp_path);
        std::string cgp_line;
        std::getline(cgp_ifs, cgp_line);

        // Parse expected rack
        std::string expected_rack = parse_cgp_rack(cgp_line);
        if (expected_rack.empty()) continue;
        std::string sorted_rack = sort_rack(expected_rack);

        std::string theme = classify_theme(name);

        // Read image
        std::ifstream ifs(path, std::ios::binary);
        std::vector<uint8_t> imgdata(std::istreambuf_iterator<char>(ifs), {});

        // Run board detection to get board rect + cell size
        auto dr = process_board_image_debug(imgdata);
        if (dr.cell_size <= 0) {
            n_skipped++;
            continue;
        }

        // Detect rack tiles
        bool is_light = detect_board_mode(imgdata,
            dr.board_rect.x, dr.board_rect.y, dr.cell_size);
        auto rack_tiles = detect_rack_tiles(imgdata,
            dr.board_rect.x, dr.board_rect.y, dr.cell_size, is_light);

        int n_rt = static_cast<int>(rack_tiles.size());
        if (n_rt == 0 || n_rt > 7) {
            n_skipped++;
            continue;
        }

        // Classify rack tiles to check if they match expected
        CellResult rack_cr[7] = {};
        for (int i = 0; i < n_rt && i < 7; i++)
            rack_cr[i] = classify_rack_tile_full(rack_tiles[i]);
        refine_rack(rack_cr, std::min(n_rt, 7), dr.cells);
        alphagram_tiebreak(rack_cr, std::min(n_rt, 7));

        std::string got_rack;
        for (int i = 0; i < n_rt && i < 7; i++) {
            char ch = rack_cr[i].letter;
            got_rack += (ch >= 'A' && ch <= 'Z') ? ch : '?';
        }
        std::string got_sorted = sort_rack(got_rack);

        // Only extract crops when the classifier already gets the rack right.
        // This ensures labels are correct (bootstrapping from known-good matches).
        if (sorted_rack != got_sorted) {
            n_unmatched++;
            continue;
        }

        // Label each tile by what the classifier identified it as.
        // Since sorted_rack == got_sorted, every classification is correct.
        for (int i = 0; i < n_rt && i < 7; i++) {
            char label = rack_cr[i].letter;

            // Decode the tile crop
            cv::Mat raw(1, static_cast<int>(rack_tiles[i].png.size()), CV_8UC1,
                        const_cast<uint8_t*>(rack_tiles[i].png.data()));
            cv::Mat crop = cv::imdecode(raw, cv::IMREAD_COLOR);
            if (crop.empty()) continue;

            // Apply prepare_rack_crop (same as inference pipeline)
            cv::Mat prepared = prepare_rack_crop_for_training(crop);

            std::string fname = theme + "_" + name + "_t" + std::to_string(i) + ".png";

            if (label == '?') {
                cv::imwrite(out_dir + "/_blank/" + fname, prepared);
                n_blanks++;
            } else {
                char upper = static_cast<char>(
                    std::toupper(static_cast<unsigned char>(label)));
                cv::imwrite(out_dir + "/" + std::string(1, upper) + "/" + fname, prepared);
                n_tiles++;
            }
        }

        n_matched++;
        n_files++;
        std::fprintf(stderr, "\r%d files, %d tiles, %d blanks, %d skipped...",
                     n_files, n_tiles, n_blanks, n_skipped);
    }
    std::fprintf(stderr, "\nDone: %d files, %d tiles, %d blanks, %d skipped, %d unmatched\n",
                 n_files, n_tiles, n_blanks, n_skipped, n_unmatched);

    // Print per-letter counts
    std::printf("\nPer-letter rack crop counts:\n");
    for (int i = 0; i < 26; i++) {
        int count = 0;
        std::string ldir = out_dir + "/" + std::string(1, 'A' + i);
        if (fs::exists(ldir))
            for (auto& e : fs::directory_iterator(ldir)) { (void)e; count++; }
        std::printf("  %c: %d\n", 'A' + i, count);
    }
}
