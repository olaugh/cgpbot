// Sample brightness/contrast/HSV at every cell across all test images,
// grouped by (board_style, premium_type, tile_vs_empty).
// Only uses images where the board rect is correct (occupancy errors <= 2).
#include "board.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

static const int PREMIUM[15][15] = {
    {4,0,0,1,0,0,0,4,0,0,0,1,0,0,4},
    {0,3,0,0,0,2,0,0,0,2,0,0,0,3,0},
    {0,0,3,0,0,0,1,0,1,0,0,0,3,0,0},
    {1,0,0,3,0,0,0,1,0,0,0,3,0,0,1},
    {0,0,0,0,3,0,0,0,0,0,3,0,0,0,0},
    {0,2,0,0,0,2,0,0,0,2,0,0,0,2,0},
    {0,0,1,0,0,0,1,0,1,0,0,0,1,0,0},
    {4,0,0,1,0,0,0,5,0,0,0,1,0,0,4},
    {0,0,1,0,0,0,1,0,1,0,0,0,1,0,0},
    {0,2,0,0,0,2,0,0,0,2,0,0,0,2,0},
    {0,0,0,0,3,0,0,0,0,0,3,0,0,0,0},
    {1,0,0,3,0,0,0,1,0,0,0,3,0,0,1},
    {0,0,3,0,0,0,1,0,1,0,0,0,3,0,0},
    {0,3,0,0,0,2,0,0,0,2,0,0,0,3,0},
    {4,0,0,1,0,0,0,4,0,0,0,1,0,0,4},
};
static const char* PREM_NAME[] = {"norm","DL","TL","DW","TW","ctr"};

static void parse_cgp_occupancy(const std::string& cgp, bool occ[15][15]) {
    std::memset(occ, 0, sizeof(bool) * 225);
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
            if (row < 15 && col < 15) occ[row][col] = true;
            col++;
        }
    }
}

enum Style { LIGHT_DESK, DARK_DESK, LIGHT_MOB, DARK_MOB, MEMENTO, ORIGINAL };
static const char* STYLE_NAME[] = {"light_desk","dark_desk","light_mob","dark_mob","memento","original"};

static Style classify_style(const std::string& name) {
    if (name.find("_memento") != std::string::npos) return MEMENTO;
    if (name.find("_light_desktop") != std::string::npos) return LIGHT_DESK;
    if (name.find("_dark_desktop") != std::string::npos) return DARK_DESK;
    if (name.find("_light_mobile") != std::string::npos) return LIGHT_MOB;
    if (name.find("_dark_mobile") != std::string::npos) return DARK_MOB;
    return ORIGINAL;
}

struct Sample {
    float brightness, contrast, h, s, v;
};

int main(int argc, char* argv[]) {
    std::setbuf(stdout, nullptr);
    if (argc < 2) {
        std::cerr << "Usage: color_survey <testdata_dir>\n";
        return 1;
    }
    std::string dir = argv[1];

    // [style][premium_type][tile=1/empty=0] -> samples
    std::vector<Sample> data[6][6][2];
    int n_files = 0;

    for (auto& entry : fs::directory_iterator(dir)) {
        std::string path = entry.path().string();
        std::string name = entry.path().stem().string();
        std::string ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpg") continue;

        std::string cgp_path = dir + "/" + name + ".cgp";
        if (!fs::exists(cgp_path)) continue;

        std::ifstream cgp_ifs(cgp_path);
        std::string cgp_line;
        std::getline(cgp_ifs, cgp_line);

        bool gt_occ[15][15];
        parse_cgp_occupancy(cgp_line, gt_occ);

        Style style = classify_style(name);

        // Read image and run board detection
        std::ifstream ifs(path, std::ios::binary);
        std::vector<uint8_t> imgdata(std::istreambuf_iterator<char>(ifs), {});
        auto dr = process_board_image_debug(imgdata);

        // Check board is reasonably detected (few occupancy errors)
        int errors = 0;
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++) {
                bool detected = (dr.cells[r][c].letter != 0);
                bool actual = gt_occ[r][c];
                if (detected != actual) errors++;
            }
        // Skip badly detected boards
        if (errors > 15) {
            std::fprintf(stderr, "  SKIP %s (occ errors=%d)\n", name.c_str(), errors);
            continue;
        }

        // Decode image and extract cell stats
        cv::Mat img = cv::imdecode(imgdata, cv::IMREAD_COLOR);

        // Parse board rect from debug log
        int bx = 0, by = 0, bw = 0;
        auto pos = dr.log.find("Final: rect=");
        if (pos != std::string::npos)
            sscanf(dr.log.c_str() + pos, "Final: rect=%d,%d %dx%d", &bx, &by, &bw, &bw);
        if (bw == 0) continue;

        double cw = bw / 15.0;
        int inset_x = (int)(cw * 0.2), inset_y = (int)(cw * 0.2);
        int inner_w = (int)(cw * 0.6), inner_h = (int)(cw * 0.6);

        for (int r = 0; r < 15; r++) {
            for (int c = 0; c < 15; c++) {
                int x0 = bx + (int)(c * cw) + inset_x;
                int y0 = by + (int)(r * cw) + inset_y;
                if (x0 < 0 || y0 < 0 || x0 + inner_w > img.cols || y0 + inner_h > img.rows)
                    continue;

                cv::Mat cell = img(cv::Rect(x0, y0, inner_w, inner_h));
                cv::Mat gray;
                cv::cvtColor(cell, gray, cv::COLOR_BGR2GRAY);
                cv::Scalar gm, gs;
                cv::meanStdDev(gray, gm, gs);

                cv::Mat hsv;
                cv::cvtColor(cell, hsv, cv::COLOR_BGR2HSV);
                cv::Scalar hm = cv::mean(hsv);

                int prem = PREMIUM[r][c];
                int is_tile = gt_occ[r][c] ? 1 : 0;
                data[style][prem][is_tile].push_back({
                    (float)gm[0], (float)gs[0],
                    (float)hm[0], (float)hm[1], (float)hm[2]
                });
            }
        }

        n_files++;
        std::fprintf(stderr, "\r%d files...", n_files);
    }
    std::fprintf(stderr, "\n");

    // Print summary stats
    for (int st = 0; st < 6; st++) {
        bool has_data = false;
        for (int p = 0; p < 6; p++)
            for (int t = 0; t < 2; t++)
                if (!data[st][p][t].empty()) has_data = true;
        if (!has_data) continue;

        std::printf("\n=== %s ===\n", STYLE_NAME[st]);
        std::printf("%-5s %-6s %5s  %6s %6s  %6s %6s %6s\n",
            "prem", "type", "n", "bright", "contr", "H", "S", "V");

        for (int p = 0; p < 6; p++) {
            for (int t = 0; t < 2; t++) {
                auto& v = data[st][p][t];
                if (v.empty()) continue;

                float sb=0,sc=0,sh=0,ss=0,sv=0;
                float min_b=999,max_b=0,min_c=999,max_c=0;
                for (auto& s : v) {
                    sb += s.brightness; sc += s.contrast;
                    sh += s.h; ss += s.s; sv += s.v;
                    min_b = std::min(min_b, s.brightness);
                    max_b = std::max(max_b, s.brightness);
                    min_c = std::min(min_c, s.contrast);
                    max_c = std::max(max_c, s.contrast);
                }
                int n = (int)v.size();
                std::printf("%-5s %-6s %5d  %6.1f %6.1f  %6.1f %6.1f %6.1f  bri[%.0f-%.0f] con[%.0f-%.0f]\n",
                    PREM_NAME[p], t ? "TILE" : "empty", n,
                    sb/n, sc/n, sh/n, ss/n, sv/n,
                    min_b, max_b, min_c, max_c);
            }
        }
    }
}
