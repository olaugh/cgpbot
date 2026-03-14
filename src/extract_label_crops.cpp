// Extract labeled column/row label crops from testdata for CNN training.
// Follows the pattern of extract_crops.cpp: iterate testdata images with
// matching .cgp files, run board detection, crop label regions.
//
// Column labels: A-O above the board (classes 0-14)
// Row labels: 1-15 left of the board (classes 15-29)
//
// Output: label_training_data/col_A/ ... col_O/, row_01/ ... row_15/
#include "board.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

// Parse CGP board section into a 15x15 letter grid.
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

// Classify board theme from filename.
static std::string classify_theme(const std::string& name) {
    if (name.find("_memento") != std::string::npos) return "memento";
    if (name.find("_mahogany_desktop") != std::string::npos) return "mahogany_desk";
    if (name.find("_mahogany_mobile") != std::string::npos) return "mahogany_mob";
    if (name.find("_light_desktop") != std::string::npos) return "light_desk";
    if (name.find("_dark_desktop") != std::string::npos) return "dark_desk";
    if (name.find("_light_mobile") != std::string::npos) return "light_mob";
    if (name.find("_dark_mobile") != std::string::npos) return "dark_mob";
    return "original";
}

int main(int argc, char* argv[]) {
    std::setbuf(stdout, nullptr);
    if (argc < 2) {
        std::cerr << "Usage: extract_label_crops <testdata_dir> [output_dir]\n";
        return 1;
    }
    std::string dir = argv[1];
    std::string out_dir = (argc >= 3) ? argv[2] : "label_training_data";

    // Create output directories: col_A..col_O, row_01..row_15
    for (int i = 0; i < 15; i++) {
        std::string col_dir = out_dir + "/col_" + std::string(1, 'A' + i);
        fs::create_directories(col_dir);
        char buf[16];
        std::snprintf(buf, sizeof(buf), "/row_%02d", i + 1);
        fs::create_directories(out_dir + buf);
    }

    int n_files = 0, n_col_crops = 0, n_row_crops = 0, n_skipped = 0;

    for (auto& entry : fs::directory_iterator(dir)) {
        std::string path = entry.path().string();
        std::string name = entry.path().stem().string();
        std::string ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpg") continue;

        std::string cgp_path = dir + "/" + name + ".cgp";
        if (!fs::exists(cgp_path)) continue;

        std::string theme = classify_theme(name);

        // Skip memento theme (no labels)
        if (theme == "memento") {
            n_skipped++;
            continue;
        }

        // Read CGP ground truth
        std::ifstream cgp_ifs(cgp_path);
        std::string cgp_line;
        std::getline(cgp_ifs, cgp_line);
        char gt_letters[15][15];
        parse_cgp_letters(cgp_line, gt_letters);

        // Read image and run board detection
        std::ifstream ifs(path, std::ios::binary);
        std::vector<uint8_t> imgdata(std::istreambuf_iterator<char>(ifs), {});
        auto dr = process_board_image_debug(imgdata);

        // Parse board rect from debug log
        int bx = 0, by = 0, bw = 0, bh = 0;
        auto pos = dr.log.find("Final: rect=");
        if (pos != std::string::npos)
            std::sscanf(dr.log.c_str() + pos,
                        "Final: rect=%d,%d %dx%d", &bx, &by, &bw, &bh);
        if (bw == 0) {
            std::fprintf(stderr, "  SKIP %s (no board rect)\n", name.c_str());
            n_skipped++;
            continue;
        }

        // Check occupancy agreement between detection and CGP
        int mismatches = 0;
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++) {
                bool detected = (dr.cells[r][c].letter != 0);
                bool actual = (gt_letters[r][c] != 0);
                if (detected != actual) mismatches++;
            }
        if (mismatches > 2) {
            std::fprintf(stderr, "  SKIP %s (occ mismatches=%d)\n",
                         name.c_str(), mismatches);
            n_skipped++;
            continue;
        }

        // Decode image and extract label crops
        cv::Mat img = cv::imdecode(imgdata, cv::IMREAD_COLOR);
        double cw = static_cast<double>(bw) / 15.0;
        double ch = static_cast<double>(bh) / 15.0;
        double crop_size = 0.8 * std::min(cw, ch);
        int crop_px = std::max(8, static_cast<int>(crop_size));

        // Column labels: A-O, centered above each column
        for (int c = 0; c < 15; c++) {
            double cx = bx + (c + 0.5) * cw;
            double cy = by - 0.4 * ch;

            int x0 = static_cast<int>(cx - crop_px / 2.0);
            int y0 = static_cast<int>(cy - crop_px / 2.0);
            int x1 = x0 + crop_px;
            int y1 = y0 + crop_px;

            // Clamp to image bounds
            x0 = std::max(0, x0);
            y0 = std::max(0, y0);
            x1 = std::min(img.cols, x1);
            y1 = std::min(img.rows, y1);

            if (x1 - x0 < 4 || y1 - y0 < 4) continue;

            cv::Mat crop = img(cv::Rect(x0, y0, x1 - x0, y1 - y0));
            std::string col_name = "col_" + std::string(1, 'A' + c);
            std::string fname = theme + "_" + name + "_c" + std::to_string(c) + ".png";
            cv::imwrite(out_dir + "/" + col_name + "/" + fname, crop);
            n_col_crops++;
        }

        // Row labels: 1-15, centered left of each row
        for (int r = 0; r < 15; r++) {
            double cx = bx - 0.5 * cw;
            double cy = by + (r + 0.5) * ch;

            int x0 = static_cast<int>(cx - crop_px / 2.0);
            int y0 = static_cast<int>(cy - crop_px / 2.0);
            int x1 = x0 + crop_px;
            int y1 = y0 + crop_px;

            x0 = std::max(0, x0);
            y0 = std::max(0, y0);
            x1 = std::min(img.cols, x1);
            y1 = std::min(img.rows, y1);

            if (x1 - x0 < 4 || y1 - y0 < 4) continue;

            cv::Mat crop = img(cv::Rect(x0, y0, x1 - x0, y1 - y0));
            char buf[16];
            std::snprintf(buf, sizeof(buf), "row_%02d", r + 1);
            std::string fname = theme + "_" + name + "_r" + std::to_string(r) + ".png";
            cv::imwrite(out_dir + "/" + buf + "/" + fname, crop);
            n_row_crops++;
        }

        n_files++;
        std::fprintf(stderr, "\r%d files, %d col crops, %d row crops, %d skipped...",
                     n_files, n_col_crops, n_row_crops, n_skipped);
    }
    std::fprintf(stderr, "\nDone: %d files, %d col crops, %d row crops, %d skipped\n",
                 n_files, n_col_crops, n_row_crops, n_skipped);

    // Print per-class counts
    std::printf("\nColumn label crop counts:\n");
    for (int i = 0; i < 15; i++) {
        int count = 0;
        std::string d = out_dir + "/col_" + std::string(1, 'A' + i);
        if (fs::exists(d))
            for (auto& e : fs::directory_iterator(d)) { (void)e; count++; }
        std::printf("  %c: %d\n", 'A' + i, count);
    }
    std::printf("\nRow label crop counts:\n");
    for (int i = 0; i < 15; i++) {
        int count = 0;
        char buf[16];
        std::snprintf(buf, sizeof(buf), "row_%02d", i + 1);
        std::string d = out_dir + "/" + buf;
        if (fs::exists(d))
            for (auto& e : fs::directory_iterator(d)) { (void)e; count++; }
        std::printf("  %2d: %d\n", i + 1, count);
    }
}
