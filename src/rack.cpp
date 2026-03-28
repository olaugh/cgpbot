#include "rack.h"

#include <algorithm>
#include <cstdio>
#include <cmath>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Scrabble tile distribution (max count per letter in a standard game).
//  A  B  C  D  E  F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z
const int RACK_TILE_DIST[26] = {
    9, 2, 2, 4,12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1
};

std::string parse_cgp_rack(const std::string& cgp) {
    auto sp = cgp.find(' ');
    if (sp == std::string::npos) return {};
    auto slash = cgp.find('/', sp + 1);
    if (slash == std::string::npos) return {};
    return cgp.substr(sp + 1, slash - sp - 1);
}

std::string sort_rack(const std::string& rack) {
    std::string s = rack;
    std::sort(s.begin(), s.end());
    return s;
}

bool detect_board_mode(const std::vector<uint8_t>& image_data,
                       int bx, int by, int cell_sz) {
    cv::Mat raw(1, static_cast<int>(image_data.size()), CV_8UC1,
                const_cast<uint8_t*>(image_data.data()));
    cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
    if (img.empty()) return false;

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

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
    return mean_v > 150;
}

// Prepare a rack tile crop for CNN classification:
// 1. Adaptive bottom trim (capped at 25%)
// 2. Square the crop: center-crop if wide, pad with border replication if tall
static cv::Mat prepare_rack_crop(const cv::Mat& crop) {
    cv::Mat gray;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);

    // Adaptive bottom trim: detect uniform-brightness bars at bottom
    // (e.g., memento score margin bar). Cap at 25% to avoid trimming letter content.
    int trim_bot = crop.rows * 15 / 100;
    int max_trim = crop.rows / 4;  // cap at 25%
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

    int x_off = 0, y_off = 0;
    if (new_w > new_h) {
        // Wide crop: center-crop horizontally to make square
        x_off = (new_w - new_h) / 2;
        new_w = new_h;
    }
    cv::Rect letter_roi(x_off, y_off, new_w, new_h);
    letter_roi &= cv::Rect(0, 0, crop.cols, crop.rows);
    cv::Mat sq = crop(letter_roi);

    if (new_h > new_w * 5 / 4) {
        // Tall/narrow crop: pad sides with border replication to make square.
        // This prevents stretching narrow crops (e.g., memento partial tiles).
        int target = new_h;
        int pad_left = (target - new_w) / 2;
        int pad_right = target - new_w - pad_left;
        cv::Mat padded;
        cv::copyMakeBorder(sq, padded, 0, 0, pad_left, pad_right,
                           cv::BORDER_REPLICATE);
        return padded;
    }
    return sq;
}

CellResult classify_rack_tile_full(const RackTile& rt) {
    CellResult cr = {};
    if (rt.is_blank) {
        cr.letter = '?';
        cr.is_blank = true;
        return cr;
    }
    cv::Mat raw(1, static_cast<int>(rt.png.size()), CV_8UC1,
                const_cast<uint8_t*>(rt.png.data()));
    cv::Mat crop = cv::imdecode(raw, cv::IMREAD_COLOR);
    if (crop.empty()) { cr.letter = '?'; return cr; }

    // Primary classification: standard crop
    cv::Mat letter_crop = prepare_rack_crop(crop);
    float scores_main[26] = {};
    cr = classify_single_tile_ex(letter_crop, 0, scores_main);
    if (cr.letter >= 'a' && cr.letter <= 'z')
        cr.letter = static_cast<char>(cr.letter - 32);

    // Multi-crop: try alternative crops when confidence is not very high.
    // This helps when the primary crop's bottom trim or squaring removes
    // critical letter features (e.g., J's descender hook, Q's tail).
    if (cr.confidence < 0.99f && crop.cols > 12 && crop.rows > 12) {
        float scores_alt[26] = {};
        int n_alt = 0;
        float scores_sum[26] = {};
        for (int i = 0; i < 26; i++) scores_sum[i] = scores_main[i];

        // Alt 1: no bottom trim — just center-crop to square
        {
            int w = crop.cols, h = crop.rows;
            int s = std::min(w, h);
            int x_off = (w > h) ? (w - h) / 2 : 0;
            int y_off = (h > w) ? (h - w) / 2 : 0;
            cv::Rect sq(x_off, y_off, s, s);
            sq &= cv::Rect(0, 0, w, h);
            if (sq.width > 8 && sq.height > 8) {
                classify_single_tile_ex(crop(sq), 0, scores_alt);
                for (int i = 0; i < 26; i++) scores_sum[i] += scores_alt[i];
                n_alt++;
            }
        }

        // Alt 2: inset crop (12.5% from each side)
        {
            int ix = crop.cols / 8, iy = crop.rows / 8;
            cv::Rect inset(ix, iy, crop.cols - 2 * ix, crop.rows - 2 * iy);
            inset &= cv::Rect(0, 0, crop.cols, crop.rows);
            if (inset.width > 8 && inset.height > 8) {
                cv::Mat isq = prepare_rack_crop(crop(inset));
                classify_single_tile_ex(isq, 0, scores_alt);
                for (int i = 0; i < 26; i++) scores_sum[i] += scores_alt[i];
                n_alt++;
            }
        }

        if (n_alt > 0) {
            float best_avg = 0;
            int best_idx = 0;
            for (int i = 0; i < 26; i++) {
                scores_sum[i] /= (1 + n_alt);
                if (scores_sum[i] > best_avg) { best_avg = scores_sum[i]; best_idx = i; }
            }
            char new_letter = 'A' + best_idx;
            if (new_letter != cr.letter) {
                std::fprintf(stderr, "  rack multi-crop: %c(%.3f)->%c(%.3f)\n",
                             cr.letter, cr.confidence, new_letter, best_avg);
                cr.letter = new_letter;
                cr.confidence = best_avg;
                int idx[26];
                for (int i = 0; i < 26; i++) idx[i] = i;
                std::sort(idx, idx + 26, [&](int a, int b) {
                    return scores_sum[a] > scores_sum[b]; });
                for (int k = 0; k < 5; k++) {
                    cr.cand_letters[k] = 'A' + idx[k];
                    cr.cand_scores[k] = scores_sum[idx[k]];
                }
            }
        }
    }
    return cr;
}

void refine_rack(CellResult rack_results[], int n_tiles,
                 const CellResult board_cells[15][15]) {
    if (n_tiles <= 0) return;

    int board_counts[26] = {};
    int blanks_on_board = 0;
    for (int r = 0; r < 15; r++)
        for (int c = 0; c < 15; c++) {
            char ch = board_cells[r][c].letter;
            if (ch >= 'A' && ch <= 'Z')
                board_counts[ch - 'A']++;
            else if (ch >= 'a' && ch <= 'z')
                blanks_on_board++;
        }

    int blanks_in_rack = 0;
    for (int i = 0; i < n_tiles; i++)
        if (rack_results[i].letter == '?' || rack_results[i].is_blank)
            blanks_in_rack++;

    int undetected_blanks = std::max(0, 2 - blanks_on_board - blanks_in_rack);

    int remaining[26] = {};
    for (int li = 0; li < 26; li++) {
        int slack = std::min(board_counts[li], undetected_blanks);
        remaining[li] = RACK_TILE_DIST[li] - board_counts[li] + slack;
    }

    for (int pass = 0; pass < 14; pass++) {
        int rack_counts[26] = {};
        for (int i = 0; i < n_tiles; i++) {
            char ch = rack_results[i].letter;
            if (ch >= 'A' && ch <= 'Z')
                rack_counts[ch - 'A']++;
        }

        int worst_li = -1;
        int worst_excess = 0;
        for (int li = 0; li < 26; li++) {
            int excess = rack_counts[li] - std::max(0, remaining[li]);
            if (excess > worst_excess) {
                worst_excess = excess;
                worst_li = li;
            }
        }
        if (worst_li < 0) break;

        int weakest_idx = -1;
        float weakest_conf = 2.0f;
        for (int i = 0; i < n_tiles; i++) {
            if (rack_results[i].letter == 'A' + worst_li) {
                if (rack_results[i].confidence < weakest_conf) {
                    weakest_conf = rack_results[i].confidence;
                    weakest_idx = i;
                }
            }
        }
        if (weakest_idx < 0) break;

        CellResult& cr = rack_results[weakest_idx];
        bool reassigned = false;
        for (int k = 1; k < 5; k++) {
            char cand = cr.cand_letters[k];
            if (cand < 'A' || cand > 'Z') continue;
            int cli = cand - 'A';
            if (cand == cr.letter) continue;
            int cand_used = rack_counts[cli];
            if (cand_used < std::max(0, remaining[cli])) {
                cr.letter = cand;
                cr.confidence = cr.cand_scores[k];
                reassigned = true;
                break;
            }
        }
        if (!reassigned) break;
    }
}

void alphagram_tiebreak(CellResult rack_results[], int n_tiles) {
    if (n_tiles <= 1) return;

    auto sortedness = [&]() {
        int ok = 0;
        for (int i = 0; i + 1 < n_tiles; i++) {
            char a = rack_results[i].letter;
            char b = rack_results[i + 1].letter;
            if (a == '?' || b == '?') { ok++; continue; }
            if (a <= b) ok++;
        }
        return ok;
    };

    int base_score = sortedness();
    if (base_score == n_tiles - 1) return;

    // Pass 1: conservative — only swap when confidence < 0.92 and gap < 0.15
    for (int i = 0; i < n_tiles; i++) {
        CellResult& cr = rack_results[i];
        if (cr.letter == '?' || cr.is_blank) continue;
        if (cr.confidence > 0.92f) continue;
        for (int k = 0; k < 5; k++) {
            char cand = cr.cand_letters[k];
            float score = cr.cand_scores[k];
            if (cand < 'A' || cand > 'Z') continue;
            if (cand == cr.letter) continue;
            if (cr.confidence - score > 0.15f) break;

            char orig = cr.letter;
            float orig_conf = cr.confidence;
            cr.letter = cand;
            cr.confidence = score;
            int new_score = sortedness();
            if (new_score > base_score) {
                std::fprintf(stderr, "  alphagram: tile %d: %c(%.2f)->%c(%.2f) sort %d->%d\n",
                             i, orig, orig_conf, cand, score, base_score, new_score);
                base_score = new_score;
                break;
            } else {
                cr.letter = orig;
                cr.confidence = orig_conf;
            }
        }
    }

    if (base_score == n_tiles - 1) return;

    // Pass 2: sort-violation override
    for (int i = 0; i < n_tiles; i++) {
        CellResult& cr = rack_results[i];
        if (cr.letter == '?' || cr.is_blank) continue;
        if (cr.confidence > 0.995f) continue;

        bool has_violation = false;
        if (i > 0) {
            char prev = rack_results[i - 1].letter;
            if (prev != '?' && prev > cr.letter) has_violation = true;
        }
        if (i + 1 < n_tiles) {
            char next = rack_results[i + 1].letter;
            if (next != '?' && cr.letter > next) has_violation = true;
        }
        if (!has_violation) continue;

        for (int k = 0; k < 5; k++) {
            char cand = cr.cand_letters[k];
            float score = cr.cand_scores[k];
            if (cand < 'A' || cand > 'Z') continue;
            if (cand == cr.letter) continue;
            if (score < 0.01f) break;

            char orig = cr.letter;
            float orig_conf = cr.confidence;
            cr.letter = cand;
            cr.confidence = score;
            int new_score = sortedness();
            if (new_score > base_score) {
                std::fprintf(stderr, "  alphagram-v2: tile %d: %c(%.2f)->%c(%.2f) sort %d->%d\n",
                             i, orig, orig_conf, cand, score, base_score, new_score);
                base_score = new_score;
                break;
            } else {
                cr.letter = orig;
                cr.confidence = orig_conf;
            }
        }
    }
}

std::vector<RackTile> detect_rack_tiles(
    const std::vector<uint8_t>& image_data,
    int bx, int by, int cell_sz,
    bool is_light_mode)
{
    std::vector<RackTile> tiles;
    cv::Mat raw(1, static_cast<int>(image_data.size()), CV_8UC1,
                const_cast<uint8_t*>(image_data.data()));
    cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
    if (img.empty()) return tiles;

    int board_bottom = by + 15 * cell_sz;
    int search_top = board_bottom + cell_sz / 3;
    int search_bottom = std::min(img.rows, board_bottom + cell_sz * 5 / 2);
    if (search_top >= img.rows) return tiles;

    int x_left = std::max(0, bx - cell_sz);
    int x_right = std::min(img.cols, bx + 15 * cell_sz + cell_sz);
    cv::Rect search_roi(x_left, search_top,
                         x_right - x_left,
                         search_bottom - search_top);
    search_roi &= cv::Rect(0, 0, img.cols, img.rows);
    if (search_roi.width <= 0 || search_roi.height <= 0) return tiles;

    cv::Mat region = img(search_roi);
    cv::Mat hsv;
    cv::cvtColor(region, hsv, cv::COLOR_BGR2HSV);

    cv::Mat v_chan;
    cv::extractChannel(hsv, v_chan, 2);
    int sample_sz = std::max(3, cell_sz / 4);
    auto sample_mean_v = [&](int sx, int sy) {
        cv::Rect sr(sx, sy,
            std::min(sample_sz, v_chan.cols - sx),
            std::min(sample_sz, v_chan.rows - sy));
        sr &= cv::Rect(0, 0, v_chan.cols, v_chan.rows);
        if (sr.width <= 0 || sr.height <= 0) return 128.0;
        return cv::mean(v_chan(sr))[0];
    };
    double bg_v = std::min({sample_mean_v(0, 0),
                            sample_mean_v(v_chan.cols - sample_sz, 0),
                            sample_mean_v(0, v_chan.rows - sample_sz),
                            sample_mean_v(v_chan.cols - sample_sz, v_chan.rows - sample_sz)});
    bool rack_bg_is_light = (bg_v > 150);

    // Blur V channel to reduce JPEG artifact noise in threshold masks
    cv::Mat v_blur;
    cv::GaussianBlur(v_chan, v_blur, cv::Size(3, 3), 0);
    cv::Mat s_chan;
    cv::extractChannel(hsv, s_chan, 1);

    cv::Mat mask;
    if (rack_bg_is_light) {
        cv::threshold(s_chan, mask, 26, 255, cv::THRESH_BINARY);
        cv::Mat not_dark;
        cv::threshold(v_blur, not_dark, 50, 255, cv::THRESH_BINARY);
        mask &= not_dark;
    } else {
        int thresh = std::max(80, (int)(bg_v + 40));
        cv::threshold(v_blur, mask, thresh, 255, cv::THRESH_BINARY);
        cv::Mat s_blur;
        cv::GaussianBlur(s_chan, s_blur, cv::Size(3, 3), 0);
        cv::Mat s_mask;
        cv::threshold(s_blur, s_mask, 40, 255, cv::THRESH_BINARY);
        cv::Mat v_above_bg;
        cv::threshold(v_blur, v_above_bg, (int)(bg_v + 10), 255, cv::THRESH_BINARY);
        s_mask &= v_above_bg;
        mask |= s_mask;
    }

    int close_sz = std::max(3, cell_sz / 8);
    cv::Mat k_close = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(close_sz, close_sz));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k_close);

    cv::Mat k_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k_open);

    cv::Mat row_sum;
    cv::reduce(mask, row_sum, 1, cv::REDUCE_SUM, CV_32S);

    int best_band_y = -1;
    double best_band_sum = 0;
    int band_h = std::max(1, cell_sz * 3 / 4);
    for (int y = 0; y <= mask.rows - band_h; y++) {
        double sum = 0;
        for (int dy = 0; dy < band_h; dy++)
            sum += row_sum.at<int>(y + dy, 0);
        if (sum > best_band_sum) {
            best_band_sum = sum;
            best_band_y = y;
        }
    }
    if (best_band_y < 0 || best_band_sum == 0) return tiles;

    int band_top = std::max(0, best_band_y - cell_sz / 8);
    int band_bot = std::min(mask.rows, best_band_y + band_h + cell_sz / 8);

    cv::Mat band_mask = mask(cv::Rect(0, band_top, mask.cols, band_bot - band_top));
    cv::Mat col_sum;
    cv::reduce(band_mask, col_sum, 0, cv::REDUCE_SUM, CV_32S);

    int col_thresh = (band_bot - band_top) * 255 * 3 / 10;
    std::vector<std::pair<int,int>> segments;
    bool in_seg = false;
    int seg_start = 0;
    for (int x = 0; x < col_sum.cols; x++) {
        int val = col_sum.at<int>(0, x);
        if (val >= col_thresh) {
            if (!in_seg) { seg_start = x; in_seg = true; }
        } else {
            if (in_seg) { segments.push_back({seg_start, x}); in_seg = false; }
        }
    }
    if (in_seg) segments.push_back({seg_start, col_sum.cols});

    struct TileRect { cv::Rect rect; };
    std::vector<TileRect> candidates;

    int abs_y = search_roi.y + band_top;
    int abs_h = band_bot - band_top;

    for (auto& [sx, ex] : segments) {
        int seg_w = ex - sx;
        if (seg_w < cell_sz / 3) continue;

        int abs_x = search_roi.x + sx;

        if (seg_w <= cell_sz * 3 / 2) {
            candidates.push_back({{abs_x, abs_y, seg_w, abs_h}});
        } else {
            int n_tiles_est = std::max(1, (seg_w + cell_sz / 3) / cell_sz);
            if (n_tiles_est > 7) n_tiles_est = 7;

            std::vector<int> split_points;
            int min_gap = cell_sz / 4;
            for (int x = min_gap; x < seg_w - min_gap; x++) {
                int v = col_sum.at<int>(0, sx + x);
                bool is_min = true;
                int win = std::max(3, cell_sz / 8);
                for (int dx = -win; dx <= win && is_min; dx++) {
                    int nx = sx + x + dx;
                    if (nx >= 0 && nx < col_sum.cols) {
                        if (col_sum.at<int>(0, nx) < v) is_min = false;
                    }
                }
                if (is_min && v < col_thresh) {
                    if (split_points.empty() || x - split_points.back() >= min_gap)
                        split_points.push_back(x);
                }
            }

            if ((int)split_points.size() >= n_tiles_est - 1) {
                if ((int)split_points.size() > n_tiles_est - 1) {
                    std::sort(split_points.begin(), split_points.end(),
                        [&](int a, int b) {
                            return col_sum.at<int>(0, sx + a) < col_sum.at<int>(0, sx + b);
                        });
                    split_points.resize(n_tiles_est - 1);
                    std::sort(split_points.begin(), split_points.end());
                }
                int prev = 0;
                for (int sp : split_points) {
                    int tw = sp - prev;
                    if (tw > 0) candidates.push_back({{abs_x + prev, abs_y, tw, abs_h}});
                    prev = sp;
                }
                int tw = seg_w - prev;
                if (tw > 0) candidates.push_back({{abs_x + prev, abs_y, tw, abs_h}});
            } else {
                int tile_w = seg_w / n_tiles_est;
                for (int i = 0; i < n_tiles_est; i++) {
                    int tx = abs_x + i * tile_w;
                    int tw = (i < n_tiles_est - 1) ? tile_w : (abs_x + seg_w - tx);
                    candidates.push_back({{tx, abs_y, tw, abs_h}});
                }
            }
        }
    }

    if (candidates.empty()) return tiles;

    // Filter by fill ratio
    {
        std::vector<TileRect> fill_ok;
        for (auto& c : candidates) {
            int mx = c.rect.x - search_roi.x;
            int my = c.rect.y - search_roi.y;
            cv::Rect mr(mx, my, c.rect.width, c.rect.height);
            mr &= cv::Rect(0, 0, mask.cols, mask.rows);
            if (mr.width <= 0 || mr.height <= 0) continue;
            int fg = cv::countNonZero(mask(mr));
            float ratio = float(fg) / float(mr.width * mr.height);
            if (ratio < 0.25f) continue;
            fill_ok.push_back(c);
        }
        candidates = fill_ok;
    }
    if (candidates.empty()) return tiles;

    // Drop button candidates
    int board_center_x = bx + 7 * cell_sz + cell_sz / 2;
    int rack_half = 5 * cell_sz;
    std::vector<TileRect> filtered;
    for (auto& c : candidates) {
        int cx = c.rect.x + c.rect.width / 2;
        if (c.rect.width <= cell_sz * 3 / 2) {
            if (std::abs(cx - board_center_x) > rack_half) continue;
        }
        filtered.push_back(c);
    }

    // Limit to 7 tiles max, prefer centrally located
    if (filtered.size() > 7) {
        std::sort(filtered.begin(), filtered.end(),
                  [board_center_x](const TileRect& a, const TileRect& b) {
                      int da = std::abs(a.rect.x + a.rect.width / 2 - board_center_x);
                      int db = std::abs(b.rect.x + b.rect.width / 2 - board_center_x);
                      return da < db;
                  });
        filtered.resize(7);
    }

    // Sort left to right
    std::sort(filtered.begin(), filtered.end(),
              [](const TileRect& a, const TileRect& b) {
                  return a.rect.x < b.rect.x;
              });

    // Gap-based clustering
    if (filtered.size() > 2) {
        int max_gap = cell_sz * 3 / 4;
        struct Cluster { int start; int len; };
        std::vector<Cluster> clusters;
        int cur_start = 0, cur_len = 1;
        for (int i = 1; i < (int)filtered.size(); i++) {
            int gap = filtered[i].rect.x - (filtered[i-1].rect.x + filtered[i-1].rect.width);
            if (gap <= max_gap) {
                cur_len++;
            } else {
                clusters.push_back({cur_start, cur_len});
                cur_start = i;
                cur_len = 1;
            }
        }
        clusters.push_back({cur_start, cur_len});
        if (clusters.size() > 1) {
            std::sort(clusters.begin(), clusters.end(),
                      [](const Cluster& a, const Cluster& b) { return a.len > b.len; });
            if (clusters[0].len > clusters[1].len) {
                int bs = clusters[0].start, bl = clusters[0].len;
                std::vector<TileRect> clustered(filtered.begin() + bs,
                                                filtered.begin() + bs + bl);
                filtered = clustered;
            } else if (clusters[0].len == 1) {
                auto& best = *std::min_element(filtered.begin(), filtered.end(),
                    [board_center_x](const TileRect& a, const TileRect& b) {
                        int da = std::abs(a.rect.x + a.rect.width / 2 - board_center_x);
                        int db = std::abs(b.rect.x + b.rect.width / 2 - board_center_x);
                        return da < db;
                    });
                filtered = {best};
            }
        }
    }

    // Ensure detected tiles are at least cell_sz*3/4 wide
    {
        int min_w = cell_sz * 3 / 4;
        for (auto& c : filtered) {
            if (c.rect.width < min_w) {
                int cx = c.rect.x + c.rect.width / 2;
                c.rect.x = cx - min_w / 2;
                c.rect.width = min_w;
                if (c.rect.x < 0) c.rect.x = 0;
                if (c.rect.x + c.rect.width > img.cols)
                    c.rect.x = img.cols - c.rect.width;
            }
        }
    }

    // --- Dimension sweep ---
    auto prep_and_classify = [](const cv::Mat& crop) -> CellResult {
        cv::Mat sq = prepare_rack_crop(crop);
        return classify_single_tile(sq, false);
    };

    int n_tiles_count = (int)filtered.size();
    std::vector<int> x_centers(n_tiles_count);
    std::vector<bool> is_blank(n_tiles_count, false);
    std::vector<float> baseline_conf(n_tiles_count, 1.0f);
    std::vector<char> baseline_letter(n_tiles_count, '?');
    float baseline_min_conf = 1.0f;

    for (int i = 0; i < n_tiles_count; i++) {
        cv::Rect r = filtered[i].rect & cv::Rect(0, 0, img.cols, img.rows);
        x_centers[i] = r.x + r.width / 2;
        if (r.width <= 0 || r.height <= 0) continue;
        cv::Mat gt;
        cv::cvtColor(img(r), gt, cv::COLOR_BGR2GRAY);
        cv::Scalar gm, gs;
        cv::meanStdDev(gt, gm, gs);
        is_blank[i] = (gs[0] < 8);
        if (!is_blank[i]) {
            int tb = r.height * 15 / 100;
            int bh2 = std::max(1, r.height - tb);
            int bw2 = r.width, bx2 = 0;
            if (bw2 > bh2) { bx2 = (bw2 - bh2) / 2; bw2 = bh2; }
            cv::Rect broi(r.x + bx2, r.y, bw2, bh2);
            broi &= cv::Rect(0, 0, img.cols, img.rows);
            CellResult cr = classify_single_tile(img(broi), true);
            if (cr.is_blank) {
                is_blank[i] = true;
            } else {
                baseline_conf[i] = cr.confidence;
                baseline_letter[i] = cr.letter;
                if (cr.confidence < baseline_min_conf)
                    baseline_min_conf = cr.confidence;
            }
        }
    }

    int y_mid = search_roi.y + (band_top + band_bot) / 2;
    bool use_sweep = false;
    int best_w = 0, best_h = 0, best_ymid = 0;

    if (baseline_min_conf < 0.95f) {
        int avg_seg_w = abs_h;
        {
            int sum_w = 0;
            for (int i = 0; i < n_tiles_count; i++)
                sum_w += filtered[i].rect.width;
            if (n_tiles_count > 0) avg_seg_w = sum_w / n_tiles_count;
        }

        auto evaluate_combo = [&](int sw, int sh, int sy) -> float {
            float sum_log = 0;
            for (int i = 0; i < n_tiles_count; i++) {
                if (is_blank[i]) continue;
                cv::Rect r(x_centers[i] - sw / 2, sy - sh / 2, sw, sh);
                r &= cv::Rect(0, 0, img.cols, img.rows);
                if (r.width < sw * 3 / 4 || r.height < sh * 3 / 4)
                    return -1e9f;
                int mx = r.x - search_roi.x;
                int my = r.y - search_roi.y;
                cv::Rect mr(mx, my, r.width, r.height);
                mr &= cv::Rect(0, 0, mask.cols, mask.rows);
                if (mr.width <= 0 || mr.height <= 0) return -1e9f;
                int fg = cv::countNonZero(mask(mr));
                float fill = float(fg) / float(mr.width * mr.height);
                if (fill < 0.20f) return -1e9f;

                CellResult cr = prep_and_classify(img(r));
                if (baseline_conf[i] >= 0.90f && cr.letter != baseline_letter[i])
                    return -1e9f;
                sum_log += std::log(std::max(cr.confidence, 0.01f));
            }
            return sum_log;
        };

        int coarse_step = std::max(2, cell_sz / 5);
        int fine_step = std::max(1, cell_sz / 10);
        float best_score = -1e9f;
        best_w = avg_seg_w; best_h = abs_h; best_ymid = y_mid;

        int w_lo = avg_seg_w * 4 / 5, w_hi = avg_seg_w * 6 / 5;
        int h_lo = abs_h * 4 / 5, h_hi = abs_h * 6 / 5;
        int dy_range = std::max(1, cell_sz / 6);

        for (int sw = w_lo; sw <= w_hi; sw += coarse_step) {
            for (int sh = h_lo; sh <= h_hi; sh += coarse_step) {
                for (int dy = -dy_range; dy <= dy_range; dy += coarse_step) {
                    float sc = evaluate_combo(sw, sh, y_mid + dy);
                    if (sc > best_score) {
                        best_score = sc;
                        best_w = sw; best_h = sh; best_ymid = y_mid + dy;
                    }
                }
            }
        }

        int fw0 = best_w, fh0 = best_h, fy0 = best_ymid;
        for (int sw = fw0 - coarse_step; sw <= fw0 + coarse_step; sw += fine_step) {
            for (int sh = fh0 - coarse_step; sh <= fh0 + coarse_step; sh += fine_step) {
                for (int dy = fy0 - coarse_step; dy <= fy0 + coarse_step; dy += fine_step) {
                    if (sw <= 0 || sh <= 0) continue;
                    float sc = evaluate_combo(sw, sh, dy);
                    if (sc > best_score) {
                        best_score = sc; best_w = sw; best_h = sh; best_ymid = dy;
                    }
                }
            }
        }

        float baseline_sum_log = 0;
        for (int i = 0; i < n_tiles_count; i++) {
            if (!is_blank[i])
                baseline_sum_log += std::log(std::max(baseline_conf[i], 0.01f));
        }
        use_sweep = (best_score > baseline_sum_log);
    }

    for (int i = 0; i < n_tiles_count; i++) {
        cv::Rect r = (use_sweep && !is_blank[i])
            ? cv::Rect(x_centers[i] - best_w / 2, best_ymid - best_h / 2,
                       best_w, best_h)
            : filtered[i].rect;
        r &= cv::Rect(0, 0, img.cols, img.rows);
        if (r.width <= 0 || r.height <= 0) continue;

        bool is_blank_tile = is_blank[i];

        int p = 2;
        int px = std::max(0, r.x - p);
        int py = std::max(0, r.y - p);
        int pw = std::min(r.width + 2 * p, img.cols - px);
        int ph = std::min(r.height + 2 * p, img.rows - py);
        cv::Mat crop = img(cv::Rect(px, py, pw, ph));
        std::vector<uint8_t> png_buf;
        cv::imencode(".png", crop, png_buf);

        tiles.push_back({cv::Rect(px, py, pw, ph), std::move(png_buf), is_blank_tile});
    }

    return tiles;
}

void draw_rack_debug(std::vector<uint8_t>& debug_png,
                     const std::vector<RackTile>& rack_tiles)
{
    if (rack_tiles.empty() || debug_png.empty()) return;
    cv::Mat raw(1, static_cast<int>(debug_png.size()), CV_8UC1,
                debug_png.data());
    cv::Mat img = cv::imdecode(raw, cv::IMREAD_COLOR);
    if (img.empty()) return;

    for (const auto& rt : rack_tiles) {
        cv::Scalar color = rt.is_blank
            ? cv::Scalar(255, 0, 255)
            : cv::Scalar(0, 255, 255);
        cv::rectangle(img, rt.rect, color, 2);
    }

    std::vector<uint8_t> out;
    cv::imencode(".png", img, out);
    debug_png = std::move(out);
}
