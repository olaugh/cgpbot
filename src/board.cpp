#include "board.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <ft2build.h>
#include FT_FREETYPE_H

#ifdef HAS_TESSERACT
#include <tesseract/baseapi.h>
#include <allheaders.h>
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// Known premium square layout: 0=normal, 1=DL, 2=TL, 3=DW, 4=TW, 5=center
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 1: Board region detection via premium-pattern grid search
// ═══════════════════════════════════════════════════════════════════════════════

struct BoardRegion {
    cv::Rect rect;
    int cell_size;
    bool found;
    bool is_light;
};

// Compute mean HSV in a small block around (cx, cy).
static cv::Vec3f mean_hsv_block(const cv::Mat& hsv, int cx, int cy, int radius) {
    int x0 = std::max(0, cx - radius);
    int y0 = std::max(0, cy - radius);
    int x1 = std::min(hsv.cols, cx + radius + 1);
    int y1 = std::min(hsv.rows, cy + radius + 1);
    if (x1 <= x0 || y1 <= y0) return cv::Vec3f(0, 0, 0);
    cv::Scalar m = cv::mean(hsv(cv::Rect(x0, y0, x1 - x0, y1 - y0)));
    return cv::Vec3f(static_cast<float>(m[0]), static_cast<float>(m[1]),
                     static_cast<float>(m[2]));
}

// Score how well a candidate rect aligns with the known premium pattern.
// Uses area-mean HSV (not single pixel) for robustness.
// Corner TW squares are weighted very heavily since they're almost never
// covered by tiles.
static double score_premium(const cv::Mat& hsv, cv::Rect r,
                            bool is_light = false) {
    double cw = r.width / 15.0;
    double ch = r.height / 15.0;
    int sample_r = std::max(2, static_cast<int>(cw * 0.15));
    double score = 0;

    for (int row = 0; row < 15; row++) {
        for (int col = 0; col < 15; col++) {
            int cx = r.x + static_cast<int>((col + 0.5) * cw);
            int cy = r.y + static_cast<int>((row + 0.5) * ch);

            cv::Vec3f v = mean_hsv_block(hsv, cx, cy, sample_r);
            float h = v[0], s = v[1], val = v[2];
            int prem = PREMIUM[row][col];

            if (is_light) {
                // Light mode: skip blue/purple tiles
                if (h >= 100 && h <= 140 && s > 40 && val >= 40 && val <= 200)
                    continue;
                // Skip orange/gold recently-played tiles
                if (h >= 10 && h <= 30 && s > 80 && val > 150)
                    continue;
                // Skip very dark (outside board)
                if (val < 25) { score -= 0.5; continue; }

                bool white   = (s < 30 && val > 180);
                bool red     = ((h < 12 || h > 162) && s > 50 && val > 35);
                bool pink    = ((h < 15 || h > 158) && s > 15 && s < 160 && val > 100);
                bool blue    = (h >= 85 && h <= 130 && s > 35 && val > 35);
                bool ltblue  = (h >= 75 && h <= 125 && s > 10 && val > 100);

                bool is_corner = ((row == 0 || row == 14) &&
                                  (col == 0 || col == 14));

                if (prem == 0) {
                    if (white) score += 1.0;
                    else if (red || blue) score -= 2.0;
                } else if (prem == 4 || prem == 5) {
                    if (red || pink) score += (is_corner ? 10.0 : 4.0);
                    else if (white) score -= (is_corner ? 8.0 : 2.0);
                } else if (prem == 3) {
                    if (pink) score += 2.5;
                    else if (white) score -= 0.3;
                } else if (prem == 2) {
                    if (blue) score += 3.0;
                    else if (white) score -= 0.3;
                } else if (prem == 1) {
                    if (ltblue) score += 2.0;
                }
            } else {
                // Dark mode: skip beige/tan tiles
                if (h >= 8 && h <= 42 && s >= 12 && s <= 150 && val > 130)
                    continue;
                // Skip very dark (likely outside board)
                if (val < 25) { score -= 0.5; continue; }

                bool green  = (h >= 35 && h <= 90 && s > 30 && val > 25);
                bool red    = ((h < 12 || h > 162) && s > 50 && val > 35);
                bool pink   = ((h < 15 || h > 158) && s > 15 && s < 160 && val > 100);
                bool blue   = (h >= 85 && h <= 130 && s > 35 && val > 35);
                bool ltblue = (h >= 75 && h <= 125 && s > 10 && val > 100);

                bool is_corner = ((row == 0 || row == 14) &&
                                  (col == 0 || col == 14));

                if (prem == 0) {
                    if (green) score += 1.0;
                    else if (red || blue) score -= 2.0;
                } else if (prem == 4 || prem == 5) {  // TW or center
                    if (red || pink) score += (is_corner ? 10.0 : 4.0);
                    else if (green) score -= (is_corner ? 8.0 : 2.0);
                } else if (prem == 3) {  // DW
                    if (pink) score += 2.5;
                    else if (green) score -= 0.3;
                } else if (prem == 2) {  // TL
                    if (blue) score += 3.0;
                    else if (green) score -= 0.3;
                } else if (prem == 1) {  // DL
                    if (ltblue) score += 2.0;
                }
            }
        }
    }
    return score;
}

// Precision offset scoring for light mode: sample near cell EDGES to detect
// premium color spillover.  When correctly aligned, each cell's edges show
// only that cell's color.  When misaligned, premium colors bleed into
// adjacent normal cells.  Much more sensitive to small offsets than
// center-based scoring (score_premium samples cell centers, which are always
// well inside the cell for offsets of a few pixels).
static double score_edges_light(const cv::Mat& hsv, cv::Rect r) {
    double cw = r.width / 15.0;
    double ch = r.height / 15.0;
    double score = 0;

    for (int row = 0; row < 15; row++) {
        for (int col = 0; col < 15; col++) {
            int prem = PREMIUM[row][col];
            bool is_corner = ((row == 0 || row == 14) &&
                              (col == 0 || col == 14));

            // Sample 4 points near cell edges (12% inward from boundary)
            double offsets[4][2] = {
                {col + 0.12, row + 0.5},   // near left edge
                {col + 0.88, row + 0.5},   // near right edge
                {col + 0.5,  row + 0.12},  // near top edge
                {col + 0.5,  row + 0.88},  // near bottom edge
            };

            for (int i = 0; i < 4; i++) {
                int sx = r.x + static_cast<int>(offsets[i][0] * cw);
                int sy = r.y + static_cast<int>(offsets[i][1] * ch);
                if (sx < 0 || sy < 0 || sx >= hsv.cols || sy >= hsv.rows)
                    continue;

                cv::Vec3f v = mean_hsv_block(hsv, sx, sy, 2);
                float h = v[0], s = v[1], val = v[2];

                bool white    = (s < 25 && val > 180);
                bool red_pink = ((h < 15 || h > 158) && s > 20 && val > 100);
                bool blue     = (h >= 85 && h <= 130 && s > 35 && val > 35);
                bool ltblue   = (h >= 75 && h <= 125 && s > 10 && val > 100);

                double w = is_corner ? 3.0 : 1.0;

                if (prem == 0) {
                    if (white) score += 0.5;
                    // Any non-white color in a normal cell = spillover.
                    // Catches premium colors AND tile/purple slivers.
                    // Tiles on normal cells cause equal penalty at any offset
                    // (tile fills cell regardless), so this correctly drives
                    // the search toward minimal spillover.
                    if (!white && s > 20 && val > 60) score -= 2.0;
                } else if (prem == 4 || prem == 5) {
                    if (red_pink) score += 2.0 * w;
                    if (white) score -= 1.0 * w;
                } else if (prem == 3) {
                    if (red_pink) score += 1.5;
                } else if (prem == 2) {
                    if (blue) score += 1.5;
                } else if (prem == 1) {
                    if (ltblue) score += 1.0;
                }
            }
        }
    }
    return score;
}

// Find board grid position using Sobel edge projections (for light mode).
// Light mode boards have prominent dark grid lines on white background.
// We project vertical/horizontal edge magnitudes onto x/y axes, then search
// for the (cell_size, origin_x, origin_y) that best aligns 16 evenly-spaced
// grid lines with the projection peaks.
static cv::Rect find_board_gridlines(const cv::Mat& gray, cv::Rect search,
                                      std::ostringstream& log) {
    cv::Mat sobel_x, sobel_y;
    cv::Sobel(gray, sobel_x, CV_32F, 1, 0, 3);  // vertical edges
    cv::Sobel(gray, sobel_y, CV_32F, 0, 1, 3);  // horizontal edges

    int sx0 = search.x, sy0 = search.y;
    int sx1 = std::min(search.x + search.width, gray.cols);
    int sy1 = std::min(search.y + search.height, gray.rows);

    // Column-wise sum of |Sobel_x| -> peaks at vertical grid lines
    std::vector<double> vproj(gray.cols, 0);
    for (int y = sy0; y < sy1; y++) {
        const float* row = sobel_x.ptr<float>(y);
        for (int x = sx0; x < sx1; x++)
            vproj[x] += std::abs(row[x]);
    }

    // Row-wise sum of |Sobel_y| -> peaks at horizontal grid lines
    std::vector<double> hproj(gray.rows, 0);
    for (int y = sy0; y < sy1; y++) {
        const float* row = sobel_y.ptr<float>(y);
        for (int x = sx0; x < sx1; x++)
            hproj[y] += std::abs(row[x]);
    }

    // Search over cell_size and origin to maximize gridline alignment.
    // A 15-cell board has 16 grid lines (boundaries). We look for the
    // cell_size in a range around search_width/15.
    double approx_cs = search.width / 15.0;
    double min_cs = approx_cs * 0.75;
    double max_cs = approx_cs * 1.05;
    double cs_step = 0.25;

    double best_total = -1;
    int best_ox = sx0, best_oy = sy0;
    double best_cs = approx_cs;

    for (double cs = min_cs; cs <= max_cs; cs += cs_step) {
        int board_sz = static_cast<int>(cs * 15);
        if (board_sz < 50) continue;

        // Find best x-origin for this cell_size
        double best_v = -1;
        int bx = sx0;
        int ox_max = std::min(sx1 - board_sz, gray.cols - board_sz);
        for (int ox = sx0; ox <= ox_max; ox++) {
            double v = 0;
            for (int k = 0; k <= 15; k++) {
                int gx = ox + static_cast<int>(k * cs);
                if (gx >= 0 && gx < gray.cols) {
                    v += vproj[gx];
                    if (gx > 0) v += vproj[gx - 1] * 0.5;
                    if (gx + 1 < gray.cols) v += vproj[gx + 1] * 0.5;
                }
            }
            if (v > best_v) { best_v = v; bx = ox; }
        }

        // Find best y-origin for this cell_size
        double best_h = -1;
        int by = sy0;
        int oy_max = std::min(sy1 - board_sz, gray.rows - board_sz);
        for (int oy = sy0; oy <= oy_max; oy++) {
            double h = 0;
            for (int k = 0; k <= 15; k++) {
                int gy = oy + static_cast<int>(k * cs);
                if (gy >= 0 && gy < gray.rows) {
                    h += hproj[gy];
                    if (gy > 0) h += hproj[gy - 1] * 0.5;
                    if (gy + 1 < gray.rows) h += hproj[gy + 1] * 0.5;
                }
            }
            if (h > best_h) { best_h = h; by = oy; }
        }

        double total = best_v + best_h;
        if (total > best_total) {
            best_total = total;
            best_ox = bx;
            best_oy = by;
            best_cs = cs;
        }
    }

    int board_size = static_cast<int>(best_cs * 15);
    if (best_ox + board_size > gray.cols)
        best_ox = gray.cols - board_size;
    if (best_oy + board_size > gray.rows)
        best_oy = gray.rows - board_size;
    best_ox = std::max(0, best_ox);
    best_oy = std::max(0, best_oy);

    log << "Gridline detection: cell=" << best_cs
        << " origin=" << best_ox << "," << best_oy
        << " size=" << board_size << "\n";

    return cv::Rect(best_ox, best_oy, board_size, board_size);
}

static BoardRegion find_board_region(const cv::Mat& img, std::ostringstream& log) {
    // ── Step 1: Contour to get approximate search area ──────────────────
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    cv::Mat dilated;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(edges, dilated, kernel, cv::Point(-1, -1), 2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect search;
    double best_area = 0;
    int img_area = img.cols * img.rows;
    for (const auto& c : contours) {
        cv::Rect r = cv::boundingRect(c);
        double area = r.area();
        double aspect = static_cast<double>(r.width) / r.height;
        if (area > best_area && area > img_area * 0.04 &&
            aspect > 0.6 && aspect < 1.6) {
            best_area = area;
            search = r;
        }
    }
    // For tall (mobile) images, contour detection often picks up non-board
    // UI elements.  Use a generous search area covering the upper portion.
    bool is_mobile = (img.rows > img.cols * 3 / 2);
    if (best_area == 0 || is_mobile) {
        if (is_mobile) {
            // Board is in the upper portion, spans most of the width.
            int h = img.rows * 3 / 4;
            search = cv::Rect(0, 0, img.cols, h);
        } else {
            int side = std::min(img.cols * 2 / 3, img.rows);
            search = cv::Rect(0, 0, side, side);
        }
    }
    log << "Search area: " << search.x << "," << search.y
        << " " << search.width << "x" << search.height << "\n";

    // ── Step 2: Coarse grid search using premium pattern scoring ────────
    // The board is inside the search area. Labels (A-O, 1-15) may consume
    // up to ~20% on top and left. Board size is 60-100% of search area.
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // Detect light vs dark mode.  Sample 4 corner quadrants of the search
    // area (less likely covered by tiles) + the center.  Light mode boards
    // have white background (high V, low S); dark mode has green (moderate S).
    bool is_light = false;
    {
        int r = std::min(search.width, search.height) / 10;
        int margin = r * 2;
        struct { int x, y; } pts[5] = {
            {search.x + margin,                search.y + margin},                 // TL
            {search.x + search.width - margin, search.y + margin},                 // TR
            {search.x + margin,                search.y + search.height - margin},  // BL
            {search.x + search.width - margin, search.y + search.height - margin},  // BR
            {search.x + search.width / 2,      search.y + search.height / 2},       // center
        };
        float total_s = 0, total_v = 0;
        for (auto& p : pts) {
            cv::Vec3f v = mean_hsv_block(hsv, p.x, p.y, r);
            total_s += v[1];
            total_v += v[2];
        }
        float avg_s = total_s / 5, avg_v = total_v / 5;
        // Light mode: brightness alone separates modes (~108 dark vs ~212 light)
        is_light = (avg_v > 170);
        log << "Board mode: " << (is_light ? "light" : "dark")
            << " (avg_V=" << avg_v << " avg_S=" << avg_s << ")\n";
    }

    // ── Steps 2-4: premium-pattern scoring + gridline refinement ────────
    // Both light and dark modes use the same pipeline; score_premium
    // already handles color differences via is_light.

    int max_x_offset, max_y_offset, min_size, max_size;
    if (is_mobile) {
        // Mobile: board is ~90% of width, Y position varies (after status/player info)
        max_x_offset = img.cols / 8;
        max_y_offset = img.rows / 2;
        min_size = img.cols * 80 / 100;
        max_size = std::min(img.cols, img.rows * 2 / 3);
    } else {
        max_x_offset = search.width / 3;
        max_y_offset = max_x_offset;
        min_size = search.width * 55 / 100;
        max_size = std::min({search.width, search.height,
                             img.cols - search.x, img.rows - search.y});
    }

    int coarse_x_step = std::max(3, max_x_offset / (is_mobile ? 15 : 20));
    int coarse_y_step = std::max(3, max_y_offset / (is_mobile ? 40 : 20));
    int coarse_size_step = std::max(3, (max_size - min_size) / 15);

    cv::Rect best_rect(search.x, search.y, max_size, max_size);
    double best_score = -1e9;

    for (int size = min_size; size <= max_size; size += coarse_size_step) {
        for (int dy = 0; dy <= max_y_offset && search.y + dy + size <= img.rows;
             dy += coarse_y_step) {
            for (int dx = 0; dx <= max_x_offset && search.x + dx + size <= img.cols;
                 dx += coarse_x_step) {
                cv::Rect trial(search.x + dx, search.y + dy, size, size);
                double s = score_premium(hsv, trial, is_light);
                if (s > best_score) {
                    best_score = s;
                    best_rect = trial;
                }
            }
        }
    }
    log << "Coarse: score=" << best_score << " rect=" << best_rect.x
        << "," << best_rect.y << " " << best_rect.width
        << "x" << best_rect.height << "\n";

    // ── Step 3: Fine grid search around the best coarse result ──────────
    int coarse_step_for_fine = is_mobile ? std::max(coarse_x_step, coarse_y_step)
                                         : coarse_x_step;  // desktop: x=y
    int fine_pos = coarse_step_for_fine * 2;
    int fine_pos_step = std::max(1, coarse_step_for_fine / 3);
    int fine_size = coarse_size_step * 2;
    int fine_size_step = std::max(1, coarse_size_step / 3);

    cv::Rect coarse_best = best_rect;
    for (int size = coarse_best.width - fine_size;
         size <= coarse_best.width + fine_size;
         size += fine_size_step) {
        if (size < 50) continue;
        for (int dy = -fine_pos; dy <= fine_pos; dy += fine_pos_step) {
            for (int dx = -fine_pos; dx <= fine_pos; dx += fine_pos_step) {
                int x = coarse_best.x + dx;
                int y = coarse_best.y + dy;
                if (x < 0 || y < 0 || x + size > img.cols || y + size > img.rows)
                    continue;
                cv::Rect trial(x, y, size, size);
                double s = score_premium(hsv, trial, is_light);
                if (s > best_score) {
                    best_score = s;
                    best_rect = trial;
                }
            }
        }
    }

    // ── Step 4a: Pixel-precise offset + size search ────────────────────
    // All modes benefit from sub-step position refinement.  Light mode
    // uses edge-spillover scoring; dark mode uses premium-center scoring.
    {
        int half_cell = best_rect.width / 30;
        cv::Rect prec_best = best_rect;
        double prec_score = is_light ? score_edges_light(hsv, best_rect)
                                     : score_premium(hsv, best_rect, false);

        int size_range = is_mobile ? 15 : 5;
        {
            int n_threads = std::max(1u, std::thread::hardware_concurrency());
            struct ThreadResult { cv::Rect rect; double score; };
            std::vector<ThreadResult> results(n_threads, {prec_best, prec_score});
            std::vector<std::thread> threads(n_threads);

            for (int t = 0; t < n_threads; t++) {
                threads[t] = std::thread([&, t]() {
                    cv::Rect local_best = results[t].rect;
                    double local_score = results[t].score;
                    for (int ds = -size_range + t; ds <= size_range;
                         ds += n_threads) {
                        int sz = best_rect.width + ds;
                        if (sz < 100) continue;
                        for (int dy = -half_cell; dy <= half_cell; dy++) {
                            for (int dx = -half_cell; dx <= half_cell; dx++) {
                                int x = best_rect.x + dx;
                                int y = best_rect.y + dy;
                                if (x < 0 || y < 0 ||
                                    x + sz > img.cols || y + sz > img.rows)
                                    continue;
                                cv::Rect trial(x, y, sz, sz);
                                double s = is_light
                                    ? score_edges_light(hsv, trial)
                                    : score_premium(hsv, trial, false);
                                if (s > local_score) {
                                    local_score = s;
                                    local_best = trial;
                                }
                            }
                        }
                    }
                    results[t] = {local_best, local_score};
                });
            }
            for (auto& th : threads) th.join();
            for (int t = 0; t < n_threads; t++) {
                if (results[t].score > prec_score) {
                    prec_score = results[t].score;
                    prec_best = results[t].rect;
                }
            }
        }
        log << "Precision offset: rect=" << prec_best.x << "," << prec_best.y
            << " " << prec_best.width << "x" << prec_best.height
            << " score=" << prec_score << "\n";
        best_rect = prec_best;
    }

    // ── Step 4b: Gridline refinement via Sobel edge projections ────────
    // All modes have visible grid lines.  Project |Sobel_x| and |Sobel_y|
    // onto x/y axes, then search (cell_size, origin_x, origin_y) to
    // maximize edge magnitude at the 16 expected grid line positions.
    // X and Y origins are searched independently for each cell_size,
    // which is both faster and more accurate than a joint 3D search.
    {
        // Compute Sobel edge projections within a padded region.
        int pad = best_rect.width / 10;
        int rx0 = std::max(0, best_rect.x - pad);
        int ry0 = std::max(0, best_rect.y - pad);
        int rx1 = std::min(img.cols, best_rect.x + best_rect.width + pad);
        int ry1 = std::min(img.rows, best_rect.y + best_rect.height + pad);

        cv::Mat sobel_x, sobel_y;
        cv::Sobel(gray, sobel_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray, sobel_y, CV_32F, 0, 1, 3);

        // Column-wise sum of |Sobel_x| → peaks at vertical grid lines
        std::vector<double> vproj(img.cols, 0);
        for (int y = ry0; y < ry1; y++) {
            const float* row = sobel_x.ptr<float>(y);
            for (int x = rx0; x < rx1; x++)
                vproj[x] += std::abs(row[x]);
        }
        // Row-wise sum of |Sobel_y| → peaks at horizontal grid lines
        std::vector<double> hproj(img.rows, 0);
        for (int y = ry0; y < ry1; y++) {
            const float* row = sobel_y.ptr<float>(y);
            for (int x = rx0; x < rx1; x++)
                hproj[y] += std::abs(row[x]);
        }

        double approx_cs = best_rect.width / 15.0;
        int pos_range = std::max(3, static_cast<int>(approx_cs / 3));

        // Search cell_size at 0.1px precision in ±5% range.
        int min_cell_10 = static_cast<int>(approx_cs * 9.5);
        int max_cell_10 = static_cast<int>(approx_cs * 10.5) + 1;

        double best_total = -1;
        double best_cs = approx_cs;
        int best_ox = best_rect.x, best_oy = best_rect.y;

        for (int cell_10 = min_cell_10; cell_10 <= max_cell_10; cell_10++) {
            double cs = cell_10 / 10.0;
            int board_sz = static_cast<int>(std::round(cs * 15));

            // Best x-origin for this cell_size
            double best_v = -1;
            int bx = best_rect.x;
            for (int ox = best_rect.x - pos_range; ox <= best_rect.x + pos_range; ox++) {
                if (ox < 0 || ox + board_sz > img.cols) continue;
                double v = 0;
                for (int k = 0; k <= 15; k++) {
                    int gx = ox + static_cast<int>(k * cs);
                    if (gx >= 0 && gx < img.cols) {
                        v += vproj[gx];
                        if (gx > 0) v += vproj[gx - 1] * 0.5;
                        if (gx + 1 < img.cols) v += vproj[gx + 1] * 0.5;
                    }
                }
                if (v > best_v) { best_v = v; bx = ox; }
            }

            // Best y-origin for this cell_size
            double best_h = -1;
            int by = best_rect.y;
            for (int oy = best_rect.y - pos_range; oy <= best_rect.y + pos_range; oy++) {
                if (oy < 0 || oy + board_sz > img.rows) continue;
                double h = 0;
                for (int k = 0; k <= 15; k++) {
                    int gy = oy + static_cast<int>(k * cs);
                    if (gy >= 0 && gy < img.rows) {
                        h += hproj[gy];
                        if (gy > 0) h += hproj[gy - 1] * 0.5;
                        if (gy + 1 < img.rows) h += hproj[gy + 1] * 0.5;
                    }
                }
                if (h > best_h) { best_h = h; by = oy; }
            }

            double total = best_v + best_h;
            if (total > best_total) {
                best_total = total;
                best_cs = cs;
                best_ox = bx;
                best_oy = by;
            }
        }

        int gl_size = static_cast<int>(std::round(best_cs * 15));
        log << "Grid-line refine: cell=" << best_cs
            << " (was " << approx_cs << ") pos=" << best_ox
            << "," << best_oy << " size=" << gl_size << "\n";
        best_rect = cv::Rect(best_ox, best_oy, gl_size, gl_size);
    }

    int cell_size = best_rect.width / 15;
    log << "Final: rect=" << best_rect.x << "," << best_rect.y
        << " " << best_rect.width << "x" << best_rect.height
        << " cell=" << cell_size << "\n";
    return {best_rect, cell_size, true, is_light};
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 2: Cell extraction
// ═══════════════════════════════════════════════════════════════════════════════

using CellImages = cv::Mat[15][15];

static void extract_cells(const cv::Mat& img, const BoardRegion& region,
                          CellImages& cells, std::ostringstream& log) {
    double cw = static_cast<double>(region.rect.width) / 15.0;
    double ch = static_cast<double>(region.rect.height) / 15.0;
    double inset_frac = 0.08;

    for (int r = 0; r < 15; r++) {
        for (int c = 0; c < 15; c++) {
            int x0 = region.rect.x + static_cast<int>(c * cw + cw * inset_frac);
            int y0 = region.rect.y + static_cast<int>(r * ch + ch * inset_frac);
            int x1 = region.rect.x + static_cast<int>((c + 1) * cw - cw * inset_frac);
            int y1 = region.rect.y + static_cast<int>((r + 1) * ch - ch * inset_frac);

            x0 = std::max(0, std::min(x0, img.cols - 1));
            y0 = std::max(0, std::min(y0, img.rows - 1));
            x1 = std::max(x0 + 1, std::min(x1, img.cols));
            y1 = std::max(y0 + 1, std::min(y1, img.rows));

            cells[r][c] = img(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
        }
    }
    log << "Extracted 15x15 cells (inset=" << static_cast<int>(inset_frac * 100) << "%)\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 3: Cell classification
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef HAS_TESSERACT
static tesseract::TessBaseAPI* get_tess() {
    static tesseract::TessBaseAPI* api = nullptr;
    if (!api) {
        api = new tesseract::TessBaseAPI();
        // OEM_DEFAULT lets Tesseract pick the best engine. The legacy
        // engine respects tessedit_char_whitelist; LSTM_ONLY ignores it,
        // which causes 0% confidence ghost letters.
        if (api->Init(nullptr, "eng", tesseract::OEM_DEFAULT) != 0) {
            delete api;
            api = nullptr;
            return nullptr;
        }
        api->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
        api->SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
    }
    return api;
}
#endif

static bool is_tile(const cv::Mat& cell, bool is_light, int row, int col,
                    std::ostringstream& /*log*/) {
    int cx = cell.cols / 5, cy = cell.rows / 5;
    int cw = cell.cols * 3 / 5, ch = cell.rows * 3 / 5;
    if (cw <= 0 || ch <= 0) return false;
    cv::Mat center = cell(cv::Rect(cx, cy, cw, ch));

    cv::Mat gray;
    if (center.channels() == 3)
        cv::cvtColor(center, gray, cv::COLOR_BGR2GRAY);
    else
        gray = center;

    cv::Scalar mean_val, stddev_val;
    cv::meanStdDev(gray, mean_val, stddev_val);
    double brightness = mean_val[0];
    double contrast = stddev_val[0];

    if (brightness < 80 || contrast < 8) return false;

    if (center.channels() == 3) {
        cv::Mat hsv;
        cv::cvtColor(center, hsv, cv::COLOR_BGR2HSV);
        cv::Scalar hmean = cv::mean(hsv);
        double h = hmean[0], s = hmean[1], v = hmean[2];

        if (is_light) {
            // Reject pink DW/TW squares (even with tooltip/"2x word" text)
            bool is_pink = ((h < 12 || h > 155) && s > 25 && v > 100);
            if (is_pink) return false;

            // Normal tiles: beige/tan
            bool is_beige = (h >= 8 && h <= 40 && s >= 15 && s <= 140 && v > 140);
            // Recently committed move: gold/orange ~(H14-19,S141-200,V173-240).
            // H starts at 8 (not 15) to catch borderline gold tiles.
            bool is_gold = (h >= 8 && h <= 45 && s > 100 && v > 160);
            if ((is_beige || is_gold) && contrast > 15) return true;

            // Older recently-played tiles: blue/purple tint (H ~80-150).
            // Observed: H=78-98 for cyan-blue (V up to 213), H=116-146 for purple.
            // Empty premium squares have contrast=0; tiles have 30-85+.
            // Exclude DW/TW tooltips: low saturation + very bright (S<70, V>210).
            bool is_played = (h >= 78 && h <= 150 && s > 30 && v > 80);
            if (is_played && contrast > 30 && !(s < 70 && v > 210)) {
                // Reject empty TLS/DL squares that look like played tiles.
                // Empty TLS: V >= 168, played tiles: V <= 159. Use 163 as
                // threshold (middle of the 9-unit gap).
                int prem = PREMIUM[row][col];
                if ((prem == 1 || prem == 2) && v >= 163) return false;
                return true;
            }

            return false;
        }

        bool is_beige = (h >= 8 && h <= 40 && s >= 15 && s <= 140 && v > 140);
        bool is_cream = (s < 30 && v > 180);
        bool is_gold = (h >= 15 && h <= 45 && s > 100 && v > 160);

        if ((is_beige || is_cream || is_gold) && contrast > 15) return true;
        // Dark mode recently-played tiles: low-saturation blue/purple tint
        // (H~120, S~14, V~150, contrast 41-54). Empty cells all have contrast=0.
        if (contrast > 40) return true;
    } else {
        if (brightness > 170 && contrast > 20) return true;
    }
    return false;
}

static bool is_blank_tile(const cv::Mat& cell) {
    int qx = cell.cols / 2, qy = cell.rows / 2;
    int qw = cell.cols - qx, qh = cell.rows - qy;
    if (qw <= 0 || qh <= 0) return false;
    cv::Mat quad = cell(cv::Rect(qx, qy, qw, qh));

    cv::Mat gray;
    if (quad.channels() == 3)
        cv::cvtColor(quad, gray, cv::COLOR_BGR2GRAY);
    else
        gray = quad;

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    return stddev[0] < 12;
}

// Scrabble point values → valid letter sets
const char* scrabble_letters_for_points(int pts) {
    switch (pts) {
        case 1:  return "AEILNORSTU";
        case 2:  return "DG";
        case 3:  return "BCMP";
        case 4:  return "FHVWY";
        case 5:  return "K";
        case 8:  return "JX";
        case 10: return "QZ";
        default: return nullptr;
    }
}

static int point_value_of(char ch) {
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    switch (ch) {
        case 'A': case 'E': case 'I': case 'L': case 'N':
        case 'O': case 'R': case 'S': case 'T': case 'U': return 1;
        case 'D': case 'G': return 2;
        case 'B': case 'C': case 'M': case 'P': return 3;
        case 'F': case 'H': case 'V': case 'W': case 'Y': return 4;
        case 'K': return 5;
        case 'J': case 'X': return 8;
        case 'Q': case 'Z': return 10;
        default: return 0;
    }
}

// Prepare a grayscale cell crop for Tesseract OCR.
static cv::Mat prepare_ocr_image(const cv::Mat& crop, int target_size) {
    cv::Mat upscaled;
    cv::resize(crop, upscaled, cv::Size(target_size, target_size), 0, 0, cv::INTER_CUBIC);

    cv::Mat gray;
    if (upscaled.channels() == 3)
        cv::cvtColor(upscaled, gray, cv::COLOR_BGR2GRAY);
    else
        gray = upscaled.clone();

    cv::Scalar m = cv::mean(gray);
    if (m[0] < 128) cv::bitwise_not(gray, gray);

    cv::adaptiveThreshold(gray, gray, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 15, 8);

    cv::Mat morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(gray, gray, cv::MORPH_OPEN, morph_kernel);

    cv::Mat padded;
    cv::copyMakeBorder(gray, padded, 20, 20, 20, 20,
                       cv::BORDER_CONSTANT, cv::Scalar(255));
    return padded;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Template matching: pre-render A-Z tiles (letter + subscript) via FreeType
// ═══════════════════════════════════════════════════════════════════════════════

static const int TMPL_SIZE = 128;

struct TileTemplates {
    cv::Mat tiles[26];   // A-Z, each TMPL_SIZE x TMPL_SIZE, includes subscript
    bool valid = false;
};

// Blit a FreeType bitmap onto a grayscale image (black on white, alpha blend).
static void blit_glyph(cv::Mat& img, const FT_Bitmap& bmp,
                        int ox, int oy, int bitmap_top) {
    int dy = oy - bitmap_top;
    for (unsigned r = 0; r < bmp.rows; r++) {
        for (unsigned c = 0; c < bmp.width; c++) {
            int px = ox + static_cast<int>(c);
            int py = dy + static_cast<int>(r);
            if (px < 0 || px >= img.cols || py < 0 || py >= img.rows) continue;
            uint8_t alpha = bmp.buffer[r * bmp.pitch + c];
            uint8_t cur = img.at<uint8_t>(py, px);
            img.at<uint8_t>(py, px) = static_cast<uint8_t>(
                std::min(static_cast<int>(cur), 255 - static_cast<int>(alpha)));
        }
    }
}

// Render a complete tile: letter centered in upper area, subscript bottom-right.
static cv::Mat render_tile(FT_Face face, char letter) {
    cv::Mat img(TMPL_SIZE, TMPL_SIZE, CV_8UC1, cv::Scalar(255));
    int pts = point_value_of(letter);

    // ── Main letter: centered in upper ~80% of tile ──
    int letter_sz = TMPL_SIZE * 58 / 100;  // font pixel size
    FT_Set_Pixel_Sizes(face, 0, letter_sz);
    FT_UInt gi = FT_Get_Char_Index(face, static_cast<FT_ULong>(letter));
    if (gi && !FT_Load_Glyph(face, gi, FT_LOAD_RENDER)) {
        FT_Bitmap& bmp = face->glyph->bitmap;
        if (bmp.width > 0 && bmp.rows > 0) {
            int asc = static_cast<int>(face->size->metrics.ascender >> 6);
            int desc = static_cast<int>(face->size->metrics.descender >> 6);
            int area_h = TMPL_SIZE * 80 / 100;
            int ox = (TMPL_SIZE - static_cast<int>(bmp.width)) / 2;
            int oy = (area_h + asc - desc) / 2;
            blit_glyph(img, bmp, ox, oy, face->glyph->bitmap_top);
        }
    }

    // ── Subscript: bottom-right ──
    if (pts > 0) {
        std::string sub = std::to_string(pts);
        int sub_sz = TMPL_SIZE * 16 / 100;
        FT_Set_Pixel_Sizes(face, 0, sub_sz);

        // Compute total advance width of subscript text
        int total_adv = 0;
        for (char ch : sub) {
            FT_UInt dgi = FT_Get_Char_Index(face, static_cast<FT_ULong>(ch));
            if (dgi && !FT_Load_Glyph(face, dgi, FT_LOAD_DEFAULT))
                total_adv += static_cast<int>(face->glyph->advance.x >> 6);
        }

        int sub_x = TMPL_SIZE * 92 / 100 - total_adv;
        int sub_baseline = TMPL_SIZE * 93 / 100;

        for (char ch : sub) {
            FT_UInt dgi = FT_Get_Char_Index(face, static_cast<FT_ULong>(ch));
            if (!dgi || FT_Load_Glyph(face, dgi, FT_LOAD_RENDER)) continue;
            FT_Bitmap& dbmp = face->glyph->bitmap;
            blit_glyph(img, dbmp,
                        sub_x + face->glyph->bitmap_left, sub_baseline,
                        face->glyph->bitmap_top);
            sub_x += static_cast<int>(face->glyph->advance.x >> 6);
        }
    }

    return img;
}

static const TileTemplates& get_templates() {
    static TileTemplates tmpl;
    if (tmpl.valid) return tmpl;

    FT_Library ft;
    if (FT_Init_FreeType(&ft)) return tmpl;

    FT_Face face;
    const char* font_paths[] = {
#ifdef FONT_PATH
        FONT_PATH,
#endif
        "fonts/RobotoMono-Bold.ttf",
        "/tmp/RobotoMono-Bold.ttf",
        nullptr
    };

    bool loaded = false;
    for (int i = 0; font_paths[i]; i++) {
        if (FT_New_Face(ft, font_paths[i], 0, &face) == 0) {
            loaded = true;
            break;
        }
    }
    if (!loaded) { FT_Done_FreeType(ft); return tmpl; }

    for (int i = 0; i < 26; i++) {
        cv::Mat tile = render_tile(face, 'A' + i);
        cv::GaussianBlur(tile, tile, cv::Size(3, 3), 1.0);
        tmpl.tiles[i] = tile;
    }

    FT_Done_Face(face);
    FT_Done_FreeType(ft);
    tmpl.valid = true;
    return tmpl;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scrabble tile distribution (for distribution-aware refinement)
// ═══════════════════════════════════════════════════════════════════════════════

static const int TILE_DIST[26] = {
//  A  B  C  D  E  F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z
    9, 2, 2, 4,12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1
};

// ═══════════════════════════════════════════════════════════════════════════════
// Cell classification with template matching
// ═══════════════════════════════════════════════════════════════════════════════

// Compute match scores for all 26 templates against a cell image.
// Cell must already be confirmed as a tile.
static void compute_scores(const cv::Mat& cell, const TileTemplates& tmpl,
                            float scores[26]) {
    cv::Mat resized;
    cv::resize(cell, resized, cv::Size(TMPL_SIZE, TMPL_SIZE), 0, 0, cv::INTER_CUBIC);

    cv::Mat gray;
    if (resized.channels() == 3)
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    else
        gray = resized.clone();

    // Normalize polarity: ensure light background
    cv::Scalar m = cv::mean(gray);
    if (m[0] < 128) cv::bitwise_not(gray, gray);

    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 1.0);

    // Same-size matching: 1×1 result per template
    for (int i = 0; i < 26; i++) {
        cv::Mat result_mat;
        cv::matchTemplate(gray, tmpl.tiles[i], result_mat, cv::TM_CCOEFF_NORMED);
        scores[i] = result_mat.at<float>(0, 0);
    }
}

// Pick the best letter from scores and populate top-5 candidates.
static void pick_best(const float scores[26], CellResult& cell) {
    // Sort indices by score descending
    int idx[26];
    for (int i = 0; i < 26; i++) idx[i] = i;
    std::sort(idx, idx + 26, [&](int a, int b) { return scores[a] > scores[b]; });

    // Store top-5 candidates
    for (int k = 0; k < 5 && k < 26; k++) {
        cell.cand_letters[k] = 'A' + idx[k];
        cell.cand_scores[k] = scores[idx[k]];
    }

    if (scores[idx[0]] >= 0.2) {
        cell.letter = 'A' + idx[0];
        cell.confidence = scores[idx[0]];
        cell.subscript = point_value_of(cell.letter);
    } else {
        cell.letter = '?';
        cell.confidence = std::max(0.0f, scores[idx[0]]);
    }
}

// Distribution-aware refinement: reassign letters that exceed tile limits.
// Uses two constraints:
//   1. Per-letter: at most TILE_DIST[i] + 1 (one blank per letter)
//   2. Global: total excess over TILE_DIST across all letters <= 2 (only 2 blanks exist)
// Weakest-confidence excess cells are reassigned first.
static void refine_distribution(CellResult cells[15][15],
                                 float all_scores[15][15][26],
                                 std::ostringstream& log) {
    struct Ref { int r, c, li; float conf; };

    for (int pass = 0; pass < 10; pass++) {
        // Count current assignments
        int counts[26] = {};
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++) {
                char ch = cells[r][c].letter;
                if (ch >= 'A' && ch <= 'Z') counts[ch - 'A']++;
                else if (ch >= 'a' && ch <= 'z') counts[ch - 'a']++;
            }

        // Total excess over base tile counts across all letters.
        int total_excess = 0;
        for (int i = 0; i < 26; i++)
            total_excess += std::max(0, counts[i] - TILE_DIST[i]);

        // Phase 1: enforce per-letter limit of TILE_DIST + 1
        // (at most one blank can plausibly represent any given letter)
        bool changed = false;
        for (int li = 0; li < 26; li++) {
            int limit = TILE_DIST[li] + 1;
            if (counts[li] <= limit) continue;

            std::vector<Ref> refs;
            for (int r = 0; r < 15; r++)
                for (int c = 0; c < 15; c++) {
                    char ch = std::toupper(static_cast<unsigned char>(
                        cells[r][c].letter));
                    if (ch == 'A' + li)
                        refs.push_back({r, c, li, cells[r][c].confidence});
                }
            std::sort(refs.begin(), refs.end(),
                      [](const Ref& a, const Ref& b) { return a.conf < b.conf; });

            int to_remove = counts[li] - limit;
            for (int i = 0; i < to_remove && i < static_cast<int>(refs.size()); i++) {
                float* sc = all_scores[refs[i].r][refs[i].c];
                int best_alt = -1;
                float best_val = -1;
                for (int j = 0; j < 26; j++) {
                    if (j == li) continue;
                    if (counts[j] >= TILE_DIST[j] + 1) continue;
                    if (sc[j] > best_val) { best_val = sc[j]; best_alt = j; }
                }
                if (best_alt >= 0 && best_val >= 0.05) {
                    bool was_blank = cells[refs[i].r][refs[i].c].is_blank;
                    cells[refs[i].r][refs[i].c].letter = was_blank
                        ? static_cast<char>('a' + best_alt)
                        : static_cast<char>('A' + best_alt);
                    cells[refs[i].r][refs[i].c].confidence = best_val;
                    cells[refs[i].r][refs[i].c].subscript =
                        point_value_of('A' + best_alt);
                    counts[li]--;
                    counts[best_alt]++;
                    changed = true;
                }
            }
        }

        // Recalculate total excess after per-letter phase.
        total_excess = 0;
        for (int i = 0; i < 26; i++)
            total_excess += std::max(0, counts[i] - TILE_DIST[i]);

        // Phase 2: enforce global blank budget of 2.
        // If total excess > 2, collect all excess cells across all letters,
        // keep the 2 most confident as blanks, reassign the rest.
        if (total_excess > 2) {
            std::vector<Ref> excess_cells;
            for (int li = 0; li < 26; li++) {
                int excess = counts[li] - TILE_DIST[li];
                if (excess <= 0) continue;
                std::vector<Ref> refs;
                for (int r = 0; r < 15; r++)
                    for (int c = 0; c < 15; c++) {
                        char ch = std::toupper(static_cast<unsigned char>(
                            cells[r][c].letter));
                        if (ch == 'A' + li)
                            refs.push_back({r, c, li, cells[r][c].confidence});
                    }
                std::sort(refs.begin(), refs.end(),
                    [](const Ref& a, const Ref& b) { return a.conf < b.conf; });
                for (int i = 0; i < excess && i < (int)refs.size(); i++)
                    excess_cells.push_back(refs[i]);
            }

            // Sort weakest first globally
            std::sort(excess_cells.begin(), excess_cells.end(),
                [](const Ref& a, const Ref& b) { return a.conf < b.conf; });

            // Keep 2 most confident excess cells as blanks, reassign the rest
            int to_reassign = std::max(0, (int)excess_cells.size() - 2);
            for (int i = 0; i < to_reassign; i++) {
                auto& ref = excess_cells[i];
                float* sc = all_scores[ref.r][ref.c];
                int best_alt = -1;
                float best_val = -1;
                // Prefer under-represented letters first
                for (int j = 0; j < 26; j++) {
                    if (j == ref.li) continue;
                    if (counts[j] >= TILE_DIST[j]) continue;
                    if (sc[j] > best_val) { best_val = sc[j]; best_alt = j; }
                }
                // Fall back to at-limit letters
                if (best_alt < 0) {
                    for (int j = 0; j < 26; j++) {
                        if (j == ref.li) continue;
                        if (counts[j] > TILE_DIST[j]) continue;
                        if (sc[j] > best_val) { best_val = sc[j]; best_alt = j; }
                    }
                }
                if (best_alt >= 0 && best_val >= 0.05) {
                    bool was_blank = cells[ref.r][ref.c].is_blank;
                    cells[ref.r][ref.c].letter = was_blank
                        ? static_cast<char>('a' + best_alt)
                        : static_cast<char>('A' + best_alt);
                    cells[ref.r][ref.c].confidence = best_val;
                    cells[ref.r][ref.c].subscript =
                        point_value_of('A' + best_alt);
                    counts[ref.li]--;
                    counts[best_alt]++;
                    changed = true;
                }
            }
        }

        if (!changed) break;
        log << "Distribution pass " << pass + 1 << ": reassigned tiles"
            << " (total excess was " << total_excess << ")\n";
    }
}

static void classify_cells(const CellImages& cell_imgs,
                           CellResult cells[15][15],
                           bool is_light,
                           std::ostringstream& log) {
    const auto& tmpl = get_templates();
    // Store all 26 scores per cell for distribution refinement
    static float all_scores[15][15][26];
    std::memset(all_scores, 0, sizeof(all_scores));

    int tile_count = 0, ocr_fail = 0;
    for (int r = 0; r < 15; r++) {
        for (int c = 0; c < 15; c++) {
            // Diagnostic: log HSV for every cell in light mode
            if (is_light) {
                const cv::Mat& ci = cell_imgs[r][c];
                int cx2 = ci.cols / 5, cy2 = ci.rows / 5;
                int cw2 = ci.cols * 3 / 5, ch2 = ci.rows * 3 / 5;
                if (cw2 > 0 && ch2 > 0 && ci.channels() == 3) {
                    cv::Mat ctr = ci(cv::Rect(cx2, cy2, cw2, ch2));
                    cv::Mat chsv;
                    cv::cvtColor(ctr, chsv, cv::COLOR_BGR2HSV);
                    cv::Scalar hm = cv::mean(chsv);
                    cv::Scalar gm, gs;
                    cv::Mat cg;
                    cv::cvtColor(ctr, cg, cv::COLOR_BGR2GRAY);
                    cv::meanStdDev(cg, gm, gs);
                    bool det = is_tile(ci, is_light, r, c, log);
                    // Temporary: log all cells for dark mode debugging
                    log << "  [" << r+1 << "," << (char)('A'+c) << "]"
                        << (det ? " TILE" : " skip")
                        << " H=" << (int)hm[0] << " S=" << (int)hm[1]
                        << " V=" << (int)hm[2]
                        << " bri=" << (int)gm[0]
                        << " con=" << (int)gs[0] << "\n";
                }
            }
            if (!is_tile(cell_imgs[r][c], is_light, r, c, log)) continue;

            tile_count++;
            if (tmpl.valid) {
                compute_scores(cell_imgs[r][c], tmpl, all_scores[r][c]);
                pick_best(all_scores[r][c], cells[r][c]);
            } else {
                cells[r][c].letter = '?';
            }

            // Blank tile detection
            if (cells[r][c].letter != '?' && cells[r][c].letter != 0 &&
                is_blank_tile(cell_imgs[r][c])) {
                cells[r][c].is_blank = true;
                cells[r][c].letter = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(cells[r][c].letter)));
            }

            if (cells[r][c].letter == '?') ocr_fail++;
        }
    }
    log << "Classified: " << tile_count << " tiles, " << ocr_fail << " OCR failures\n";

    // Distribution-aware refinement
    if (tmpl.valid && tile_count > 0)
        refine_distribution(cells, all_scores, log);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 4: CGP formatting
// ═══════════════════════════════════════════════════════════════════════════════

static std::string format_cgp(const CellResult cells[15][15]) {
    std::string board;
    for (int r = 0; r < 15; r++) {
        if (r > 0) board += '/';
        int empty_run = 0;
        for (int c = 0; c < 15; c++) {
            char ch = cells[r][c].letter;
            if (ch == 0) {
                empty_run++;
            } else {
                if (empty_run > 0) {
                    board += std::to_string(empty_run);
                    empty_run = 0;
                }
                board += ch;
            }
        }
        if (empty_run > 0) board += std::to_string(empty_run);
    }
    board += " / 0/0 0 lex NWL23;";
    return board;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stage 5: Debug image
// ═══════════════════════════════════════════════════════════════════════════════

static std::vector<uint8_t> generate_debug_image(const cv::Mat& img,
                                                  const BoardRegion& region,
                                                  const CellResult cells[15][15]) {
    cv::Mat debug = img.clone();

    cv::rectangle(debug, region.rect, cv::Scalar(0, 255, 0), 2);

    double cw = static_cast<double>(region.rect.width) / 15.0;
    double ch = static_cast<double>(region.rect.height) / 15.0;

    for (int i = 0; i <= 15; i++) {
        int x = region.rect.x + static_cast<int>(i * cw);
        int y = region.rect.y + static_cast<int>(i * ch);
        cv::line(debug,
                 cv::Point(x, region.rect.y),
                 cv::Point(x, region.rect.y + region.rect.height),
                 cv::Scalar(255, 255, 0), 1);
        cv::line(debug,
                 cv::Point(region.rect.x, y),
                 cv::Point(region.rect.x + region.rect.width, y),
                 cv::Scalar(255, 255, 0), 1);
    }

    // No letter overlays — just the rectangle and grid lines are enough.
    (void)cells;

    std::vector<uint8_t> png;
    cv::imencode(".png", debug, png);
    return png;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Debug image helpers for intermediate stages
// ═══════════════════════════════════════════════════════════════════════════════

// Stage 1: just the green rectangle
static std::vector<uint8_t> debug_image_rect(const cv::Mat& img,
                                              const BoardRegion& region) {
    cv::Mat debug = img.clone();
    cv::rectangle(debug, region.rect, cv::Scalar(0, 255, 0), 2);
    std::vector<uint8_t> png;
    cv::imencode(".png", debug, png);
    return png;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Top-level API
// ═══════════════════════════════════════════════════════════════════════════════

DebugResult process_board_image_debug(const std::vector<uint8_t>& image_data,
                                       ProgressCallback on_progress) {
    DebugResult result;
    std::ostringstream log;

    cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
    if (img.empty()) {
        result.cgp = "[error: could not decode image]";
        result.log = "Failed to decode image data";
        return result;
    }
    log << "Image: " << img.cols << "x" << img.rows << "\n";

    // Stage 1: find board region via premium-pattern grid search
    BoardRegion region = find_board_region(img, log);

    if (on_progress) {
        auto dbg = debug_image_rect(img, region);
        on_progress("Board detected", log.str(), dbg);
    }

    // Stage 2: extract cells
    CellImages cell_imgs;
    extract_cells(img, region, cell_imgs, log);

    if (on_progress) {
        CellResult empty[15][15] = {};
        auto dbg = generate_debug_image(img, region, empty);
        on_progress("Cells extracted", log.str(), dbg);
    }

    // Stage 3: classify
    CellResult cells[15][15] = {};
    classify_cells(cell_imgs, cells, region.is_light, log);

    if (on_progress) {
        auto dbg = generate_debug_image(img, region, cells);
        on_progress("Classified", log.str(), dbg);
    }

    // If OCR is failing badly (>10 failures), the rect is probably wrong.
    // Re-search with the premium scorer using a much wider/finer grid.
    {
        int tiles = 0, failures = 0;
        for (auto& row : cells)
            for (auto& c : row) {
                if (c.letter != 0) tiles++;
                if (c.letter == '?') failures++;
            }
        if (tiles > 3 && failures * 2 > tiles) {
            log << "OCR failures=" << failures << "/" << tiles << " > 50%, retrying detection...\n";

            if (on_progress)
                on_progress("Retrying detection...", log.str(), {});

            cv::Mat hsv;
            cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

            bool is_light = region.is_light;

            // Search a wide range around the current best
            int range = std::max(region.cell_size * 2, 60);
            int step = std::max(1, range / 20);
            int size_range = range;
            int size_step = std::max(1, size_range / 15);

            double best_score = -1e9;
            cv::Rect best_r = region.rect;

            for (int ds = -size_range; ds <= size_range; ds += size_step) {
                int side = region.rect.width + ds;
                if (side < 100) continue;
                for (int dy = -range; dy <= range; dy += step) {
                    for (int dx = -range; dx <= range; dx += step) {
                        int x = region.rect.x + dx;
                        int y = region.rect.y + dy;
                        if (x < 0 || y < 0 ||
                            x + side > img.cols || y + side > img.rows) continue;
                        cv::Rect trial(x, y, side, side);
                        double s = score_premium(hsv, trial, is_light);
                        if (s > best_score) {
                            best_score = s;
                            best_r = trial;
                        }
                    }
                }
            }

            region = {best_r, best_r.width / 15, true, is_light};
            log << "Retry: score=" << best_score << " rect=" << best_r.x
                << "," << best_r.y << " " << best_r.width
                << "x" << best_r.height << "\n";

            extract_cells(img, region, cell_imgs, log);
            std::memset(cells, 0, sizeof(cells));
            classify_cells(cell_imgs, cells, region.is_light, log);

            if (on_progress) {
                auto dbg = generate_debug_image(img, region, cells);
                on_progress("Retry classified", log.str(), dbg);
            }
        }
    }

    // Copy cell results to DebugResult
    std::memcpy(result.cells, cells, sizeof(cells));

    // Stage 4: format CGP
    result.cgp = format_cgp(cells);
    log << "CGP: " << result.cgp << "\n";

    // Stage 5: debug image
    result.debug_png = generate_debug_image(img, region, cells);
    log << "Debug image: " << result.debug_png.size() << " bytes\n";

    result.log = log.str();
    return result;
}

std::string process_board_image(const std::vector<uint8_t>& image_data) {
    return process_board_image_debug(image_data).cgp;
}
