// Generate a diagnostic HTML page showing imperfect occupancy masks.
// For each imperfect case: debug image with board rect + FP/FN overlay + mask grid.
#include "board.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

static const char* PREM_NAME[] = {"norm","DL","TL","DW","TW","ctr"};

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

static std::string base64_encode(const std::vector<uint8_t>& data) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        uint32_t n = (uint32_t)data[i] << 16;
        if (i + 1 < data.size()) n |= (uint32_t)data[i+1] << 8;
        if (i + 2 < data.size()) n |= data[i+2];
        out += table[(n >> 18) & 63];
        out += table[(n >> 12) & 63];
        out += (i + 1 < data.size()) ? table[(n >> 6) & 63] : '=';
        out += (i + 2 < data.size()) ? table[n & 63] : '=';
    }
    return out;
}

int main(int argc, char* argv[]) {
    std::setbuf(stdout, nullptr);
    if (argc < 2) {
        std::cerr << "Usage: occ_diag <testdata_dir> [output.html]\n";
        return 1;
    }
    std::string dir = argv[1];
    std::string outpath = argc >= 3 ? argv[2] : "occ_diag.html";

    struct CellInfo {
        float bri, con, h, s, v;
        char letter;  // detected letter (0 = empty)
    };
    struct Case {
        std::string name;
        int tiles, fp, fn;
        std::string fp_cells, fn_cells;  // e.g. "+A1 +B2" / "-C3 -D4"
        std::vector<uint8_t> debug_png;
        bool det_occ[15][15];
        bool gt_occ[15][15];
        CellInfo cell_info[15][15];
    };
    std::vector<Case> cases;
    int n_files = 0, n_perfect = 0;

    // Collect sorted file list for deterministic order
    std::vector<std::string> files;
    for (auto& entry : fs::directory_iterator(dir)) {
        std::string ext = entry.path().extension().string();
        if (ext == ".png" || ext == ".jpg") files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());

    for (auto& path : files) {
        std::string name = fs::path(path).stem().string();
        std::string cgp_path = dir + "/" + name + ".cgp";
        if (!fs::exists(cgp_path)) continue;

        std::ifstream cgp_ifs(cgp_path);
        std::string cgp_line;
        std::getline(cgp_ifs, cgp_line);

        bool gt_occ[15][15];
        parse_cgp_occupancy(cgp_line, gt_occ);

        std::ifstream ifs(path, std::ios::binary);
        std::vector<uint8_t> data(std::istreambuf_iterator<char>(ifs), {});
        auto dr = process_board_image_debug(data);

        int fp = 0, fn = 0;
        std::string fp_cells, fn_cells;
        bool det_occ[15][15];
        for (int r = 0; r < 15; r++)
            for (int c = 0; c < 15; c++) {
                det_occ[r][c] = (dr.cells[r][c].letter != 0);
                bool actual = gt_occ[r][c];
                if (det_occ[r][c] && !actual) {
                    fp++;
                    fp_cells += std::string(" +") + (char)('A'+c) + std::to_string(r+1);
                }
                if (!det_occ[r][c] && actual) {
                    fn++;
                    fn_cells += std::string(" -") + (char)('A'+c) + std::to_string(r+1);
                }
            }

        n_files++;
        std::fprintf(stderr, "\r%d files...", n_files);

        if (fp == 0 && fn == 0) { n_perfect++; continue; }

        // Draw FP/FN overlays on the debug image
        cv::Mat dbg_img = cv::imdecode(dr.debug_png, cv::IMREAD_COLOR);
        if (!dbg_img.empty()) {
            int bx = 0, by = 0, bw = 0;
            auto pos = dr.log.find("Final: rect=");
            if (pos != std::string::npos)
                sscanf(dr.log.c_str() + pos, "Final: rect=%d,%d %dx%d", &bx, &by, &bw, &bw);
            if (bw > 0) {
                double cw = bw / 15.0;
                for (int r = 0; r < 15; r++) {
                    for (int c = 0; c < 15; c++) {
                        bool det = det_occ[r][c];
                        bool actual = gt_occ[r][c];
                        if (det == actual) continue;
                        int x0 = bx + (int)(c * cw);
                        int y0 = by + (int)(r * cw);
                        int x1 = bx + (int)((c+1) * cw);
                        int y1 = by + (int)((r+1) * cw);
                        cv::Scalar color = det ? cv::Scalar(0,0,255) : cv::Scalar(255,0,0);
                        // Semi-transparent overlay
                        cv::Mat roi = dbg_img(cv::Rect(
                            std::max(0, x0), std::max(0, y0),
                            std::min(x1, dbg_img.cols) - std::max(0, x0),
                            std::min(y1, dbg_img.rows) - std::max(0, y0)));
                        cv::Mat overlay(roi.size(), roi.type(), color);
                        cv::addWeighted(roi, 0.6, overlay, 0.4, 0, roi);
                        // Border
                        cv::rectangle(dbg_img, cv::Point(x0, y0), cv::Point(x1, y1),
                                      color, 2);
                    }
                }
            }
            std::vector<uint8_t> out_png;
            cv::imencode(".png", dbg_img, out_png);

            Case c;
            c.name = name;
            c.tiles = 0;
            for (int r = 0; r < 15; r++)
                for (int cc = 0; cc < 15; cc++)
                    if (gt_occ[r][cc]) c.tiles++;
            c.fp = fp; c.fn = fn;
            c.fp_cells = fp_cells; c.fn_cells = fn_cells;
            c.debug_png = std::move(out_png);
            std::memcpy(c.det_occ, det_occ, sizeof(det_occ));
            std::memcpy(c.gt_occ, gt_occ, sizeof(gt_occ));

            // Extract per-cell stats for mouseover tooltips
            std::memset(c.cell_info, 0, sizeof(c.cell_info));
            cv::Mat orig_img = cv::imdecode(data, cv::IMREAD_COLOR);
            if (!orig_img.empty()) {
                double cellw = bw / 15.0;
                const double inset = 0.08;
                for (int row = 0; row < 15; row++) {
                    for (int col = 0; col < 15; col++) {
                        int x0 = bx + (int)(col * cellw + cellw * inset);
                        int y0 = by + (int)(row * cellw + cellw * inset);
                        int iw = (int)((1.0 - 2*inset) * cellw);
                        int ih = iw;
                        if (x0 < 0 || y0 < 0) continue;
                        if (x0 + iw > orig_img.cols) iw = orig_img.cols - x0;
                        if (y0 + ih > orig_img.rows) ih = orig_img.rows - y0;
                        if (iw <= 0 || ih <= 0) continue;
                        cv::Mat cell = orig_img(cv::Rect(x0, y0, iw, ih));
                        // Center 60% (same as is_tile)
                        int cx2 = iw / 5, cy2 = ih / 5;
                        int ccw2 = iw * 3 / 5, cch2 = ih * 3 / 5;
                        if (ccw2 <= 0 || cch2 <= 0) continue;
                        cv::Mat center = cell(cv::Rect(cx2, cy2, ccw2, cch2));
                        cv::Mat gray, hsvmat;
                        cv::cvtColor(center, gray, cv::COLOR_BGR2GRAY);
                        cv::cvtColor(center, hsvmat, cv::COLOR_BGR2HSV);
                        cv::Scalar gm, gs;
                        cv::meanStdDev(gray, gm, gs);
                        cv::Scalar hm = cv::mean(hsvmat);
                        c.cell_info[row][col] = {
                            (float)gm[0], (float)gs[0],
                            (float)hm[0], (float)hm[1], (float)hm[2],
                            dr.cells[row][col].letter
                        };
                    }
                }
            }

            cases.push_back(std::move(c));
        }
    }
    std::fprintf(stderr, "\n%d files, %d perfect, %d imperfect\n",
        n_files, n_perfect, (int)cases.size());

    // Generate HTML
    std::ofstream html(outpath);
    html << R"(<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Occupancy Mask Diagnostics</title>
<style>
body { font-family: system-ui; background: #1a1a1a; color: #eee; padding: 20px; }
h1 { color: #4af; }
.case { border: 1px solid #444; border-radius: 8px; padding: 16px; margin: 16px 0;
        background: #222; display: flex; gap: 16px; align-items: flex-start; }
.case img { max-width: 500px; border-radius: 4px; }
.info { flex: 1; }
.info h3 { margin: 0 0 8px; color: #4af; }
.stats { color: #aaa; margin-bottom: 8px; }
.fp { color: #f44; font-weight: bold; }
.fn { color: #48f; font-weight: bold; }
.grid { display: inline-grid; grid-template-columns: 20px repeat(15, 22px);
        gap: 1px; font-size: 11px; font-family: monospace; }
.grid span { width: 22px; height: 22px; display: flex; align-items: center;
             justify-content: center; border-radius: 2px; cursor: default; }
.g-hdr { width: 20px; height: 22px; display: flex; align-items: center;
         justify-content: center; color: #666; font-size: 10px; }
.g-ok-tile { background: #363; }
.g-ok-empty { background: #333; }
.g-fp { background: #a22; color: #fff; font-weight: bold; }
.g-fn { background: #22a; color: #fff; font-weight: bold; }
.legend { margin: 12px 0; display: flex; gap: 12px; font-size: 13px; }
.legend span { padding: 2px 8px; border-radius: 3px; }
summary { cursor: pointer; color: #4af; }
#tt { position: fixed; background: #111; color: #eee; border: 1px solid #555;
      padding: 5px 8px; border-radius: 4px; font: 12px monospace;
      pointer-events: none; white-space: pre; display: none; z-index: 9999; }
</style></head><body>
<div id="tt"></div>
<script>
var tt = document.getElementById('tt');
document.addEventListener('mouseover', function(e) {
  var d = e.target.dataset.tip;
  if (d) { tt.textContent = d; tt.style.display = 'block'; }
  else tt.style.display = 'none';
});
document.addEventListener('mousemove', function(e) {
  tt.style.left = (e.clientX + 14) + 'px';
  tt.style.top  = (e.clientY + 14) + 'px';
});
document.addEventListener('mouseout', function(e) {
  if (!e.target.dataset.tip) return;
  tt.style.display = 'none';
});
</script>
<h1>Occupancy Mask Diagnostics</h1>
<p>)" << n_perfect << " / " << n_files << R"( perfect masks. Showing )"
       << cases.size() << R"( imperfect cases.</p>
<div class="legend">
  <span class="g-fp">Red = False Positive (detected tile, actually empty)</span>
  <span class="g-fn">Blue = False Negative (missed tile)</span>
</div>
)";

    for (auto& c : cases) {
        html << "<div class=\"case\">\n";
        html << "<img src=\"data:image/png;base64," << base64_encode(c.debug_png) << "\">\n";
        html << "<div class=\"info\">\n";
        html << "<h3>" << c.name << "</h3>\n";
        html << "<div class=\"stats\">Tiles: " << c.tiles;
        if (c.fp > 0) html << " | <span class=\"fp\">FP=" << c.fp << ":" << c.fp_cells << "</span>";
        if (c.fn > 0) html << " | <span class=\"fn\">FN=" << c.fn << ":" << c.fn_cells << "</span>";
        html << "</div>\n";

        // 15x15 grid with column/row headers and per-cell mouseover tooltips
        html << "<div class=\"grid\">\n";
        // Column header row
        html << "<span class=\"g-hdr\"></span>";
        for (int cc = 0; cc < 15; cc++)
            html << "<span class=\"g-hdr\">" << (char)('A'+cc) << "</span>";
        html << "\n";
        for (int r = 0; r < 15; r++) {
            // Row number header
            html << "<span class=\"g-hdr\">" << (r+1) << "</span>";
            for (int cc = 0; cc < 15; cc++) {
                bool det = c.det_occ[r][cc];
                bool gt = c.gt_occ[r][cc];
                const char* cls;
                std::string label;
                if (det && !gt) {
                    cls = "g-fp";
                    label = std::string(1, 'A'+cc) + std::to_string(r+1);
                } else if (!det && gt) {
                    cls = "g-fn";
                    label = std::string(1, 'A'+cc) + std::to_string(r+1);
                } else if (gt) {
                    cls = "g-ok-tile";
                } else {
                    cls = "g-ok-empty";
                }
                // Build tooltip from cell stats
                auto& ci = c.cell_info[r][cc];
                char tip[160];
                int prem = PREMIUM[r][cc];
                std::snprintf(tip, sizeof(tip),
                    "%c%d [%s] H=%d S=%d V=%d bri=%d con=%d det=%s",
                    'A'+cc, r+1, PREM_NAME[prem],
                    (int)ci.h, (int)ci.s, (int)ci.v,
                    (int)ci.bri, (int)ci.con,
                    ci.letter ? std::string(1, ci.letter).c_str() : "empty");
                html << "<span class=\"" << cls << "\" data-tip=\"" << tip << "\">"
                     << label << "</span>";
            }
            html << "\n";
        }
        html << "</div>\n";
        html << "</div></div>\n";
    }

    html << "</body></html>\n";
    html.close();
    std::printf("Wrote %s (%d cases)\n", outpath.c_str(), (int)cases.size());
}
