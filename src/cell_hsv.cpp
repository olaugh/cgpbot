// Print mean HSV + brightness + contrast for specific cells in an image.
// Usage: cell_hsv <image.png> r1,c1 r2,c2 ...
// Rows 1-15, cols 1-15 (1-indexed)
#include "board.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: cell_hsv <image.png> r,c [r,c ...]\n";
        return 1;
    }

    std::string path = argv[1];
    std::ifstream ifs(path, std::ios::binary);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());

    auto dr = process_board_image_debug(buf);

    // Parse board rect from log
    int bx = 0, by = 0, bw = 0, bh = 0;
    auto pos = dr.log.find("Final: rect=");
    if (pos != std::string::npos)
        sscanf(dr.log.c_str() + pos, "Final: rect=%d,%d %dx%d", &bx, &by, &bw, &bh);

    if (bw == 0) {
        std::cerr << "Board not found in " << path << "\n";
        std::cerr << "Log:\n" << dr.log.substr(0, 500) << "\n";
        return 1;
    }

    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) { std::cerr << "Cannot load image\n"; return 1; }

    std::cout << "Board rect: " << bx << "," << by << " " << bw << "x" << bh << "\n";

    // Print relevant log lines (is_light, cell detections)
    {
        auto& log = dr.log;
        size_t p = 0;
        while (p < log.size()) {
            size_t nl = log.find('\n', p);
            if (nl == std::string::npos) nl = log.size();
            std::string line = log.substr(p, nl - p);
            if (line.find("light") != std::string::npos ||
                line.find("Final") != std::string::npos)
                std::cout << "LOG: " << line << "\n";
            p = nl + 1;
        }
    }
    std::cout << "\n";

    double cw = bw / 15.0;
    double ch = bh / 15.0;

    for (int i = 2; i < argc; i++) {
        int r, c;
        if (sscanf(argv[i], "%d,%d", &r, &c) != 2) {
            std::cerr << "Bad position: " << argv[i] << "\n";
            continue;
        }
        r--; c--;  // to 0-indexed
        if (r < 0 || r >= 15 || c < 0 || c >= 15) {
            std::cerr << "Position out of range: " << argv[i] << "\n";
            continue;
        }

        // Use same 8% inset as extract_cells in board.cpp
        const double inset = 0.08;
        int x0 = bx + (int)(c * cw + cw * inset);
        int y0 = by + (int)(r * ch + ch * inset);
        int iw = (int)((1.0 - 2*inset) * cw), ih = (int)((1.0 - 2*inset) * ch);
        if (x0 + iw > img.cols) iw = img.cols - x0;
        if (y0 + ih > img.rows) ih = img.rows - y0;
        if (iw <= 0 || ih <= 0) continue;

        cv::Mat cell = img(cv::Rect(x0, y0, iw, ih));

        // Center 60% region (same as is_tile)
        int ccx = iw / 5, ccy = ih / 5;
        int ccw = iw * 3 / 5, cch = ih * 3 / 5;
        cv::Mat center = cell(cv::Rect(ccx, ccy, ccw, cch));

        cv::Mat gray, hsv;
        cv::cvtColor(center, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(center, hsv, cv::COLOR_BGR2HSV);

        cv::Scalar mean_g, std_g;
        cv::meanStdDev(gray, mean_g, std_g);
        cv::Scalar mean_hsv = cv::mean(hsv);
        cv::Scalar mean_bgr = cv::mean(center);

        // Bottom-right quadrant (blank tile indicator region)
        int qx = iw / 2, qy = ih / 2;
        int qw = iw - qx, qh = ih - qy;
        cv::Mat quad = cell(cv::Rect(qx, qy, qw, qh));
        cv::Mat quad_gray;
        cv::cvtColor(quad, quad_gray, cv::COLOR_BGR2GRAY);
        cv::Scalar q_mean, q_std;
        cv::meanStdDev(quad_gray, q_mean, q_std);

        char row_letter = 'A' + r;
        std::cout << row_letter << (c + 1)
                  << " (row=" << r+1 << ",col=" << c+1 << ")"
                  << "  H=" << (int)mean_hsv[0]
                  << " S=" << (int)mean_hsv[1]
                  << " V=" << (int)mean_hsv[2]
                  << "  bri=" << (int)mean_g[0]
                  << " con=" << (int)std_g[0]
                  << "  BR_quad_bri=" << (int)q_mean[0]
                  << " BR_quad_con=" << (int)q_std[0]
                  << "  BGR=(" << (int)mean_bgr[2] << "," << (int)mean_bgr[1] << "," << (int)mean_bgr[0] << ")"
                  << "  detected=" << (dr.cells[r][c].letter ? std::string(1, dr.cells[r][c].letter) : "empty")
                  << "\n";
    }

    return 0;
}
