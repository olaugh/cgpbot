// Evaluate local OCR pipeline (template matching or CNN) against CGP ground truth.
// No Gemini, no server — just iterate testdata, run process_board_image_debug,
// compare output CGP against expected CGP per-cell.
//
// Usage: eval_local <testdata_dir> [--html <output.html>]
//   --html: generate a self-contained HTML debug page for non-perfect cases
#include "board.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

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

// ── Base64 encoder ──────────────────────────────────────────────────────────

static const char B64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const uint8_t* data, size_t len) {
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < len) v |= static_cast<uint32_t>(data[i+1]) << 8;
        if (i + 2 < len) v |= static_cast<uint32_t>(data[i+2]);
        out.push_back(B64[(v >> 18) & 0x3F]);
        out.push_back(B64[(v >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? B64[(v >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? B64[v & 0x3F] : '=');
    }
    return out;
}

static std::string base64_encode(const std::vector<uint8_t>& v) {
    return base64_encode(v.data(), v.size());
}

// ── HTML report generation ──────────────────────────────────────────────────

// Premium square map for board coloring
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

static const char* premium_bg(int p) {
    // Woogles dark mode default colors
    switch (p) {
        case 1: return "#6dadc9"; // DL - Woogles dark DLS
        case 2: return "#115d92"; // TL - Woogles dark TLS
        case 3: return "#a9545a"; // DW - Woogles dark DWS
        case 4: return "#6b2125"; // TW - Woogles dark TWS
        case 5: return "#a9545a"; // center star - same as DW
        default: return "#313131"; // regular - Woogles dark empty
    }
}

struct FailCase {
    std::string name;
    std::vector<uint8_t> orig_png;
    std::vector<uint8_t> debug_png;
    char gt[15][15];
    CellResult cells[15][15];
    std::string log;
    std::string got_cgp;
    int tiles, correct, occ_err;
    double ms;
};

static void html_escape(std::ostream& out, const std::string& s) {
    for (char c : s) {
        switch (c) {
            case '<': out << "&lt;"; break;
            case '>': out << "&gt;"; break;
            case '&': out << "&amp;"; break;
            case '"': out << "&quot;"; break;
            default:  out << c;
        }
    }
}

static void write_html_report(const std::string& path,
                               const std::vector<FailCase>& cases,
                               int total_files, int total_tiles, int total_correct,
                               int total_occ_errors, int perfect_cases) {
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "Cannot write HTML to %s\n", path.c_str());
        return;
    }

    out << R"(<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>eval_local — failing cases</title>
<style>
* { box-sizing: border-box; }
body { background: #111; color: #ccc; font-family: 'SF Mono', 'Fira Code', monospace;
       font-size: 13px; margin: 0; padding: 16px; }
h1 { color: #eee; font-size: 1.1rem; margin-bottom: 4px; }
.summary { color: #888; margin-bottom: 16px; font-size: .85rem; }
.case { background: #1a1a1a; border: 1px solid #333; border-radius: 6px;
        padding: 16px; margin-bottom: 24px; }
.case-title { color: #7af; font-size: 1rem; font-weight: bold; margin-bottom: 4px; }
.case-stats { color: #888; font-size: .8rem; margin-bottom: 12px; }
.layout { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
.img-col { flex: 0 0 auto; }
.img-col img { max-width: 480px; max-height: 540px; border: 1px solid #333;
               border-radius: 4px; display: block; margin-bottom: 6px; }
.img-label { color: #666; font-size: .72rem; margin-bottom: 2px; }
.board { border-collapse: collapse; font-size: 11px; }
.board th { color: #555; padding: 1px 3px; text-align: center; font-weight: normal; }
.board td { width: 22px; height: 22px; text-align: center; vertical-align: middle;
            border: 1px solid #2a2a2a; position: relative; cursor: default; }
.board td.tile { font-weight: bold; }
.board td.ok { color: #8f8; }
.board td.wrong { color: #f66; border: 2px solid #f44; }
.board td.occ-fp { color: #fa0; border: 2px solid #a80; } /* false positive: got tile, expected empty */
.board td.occ-fn { border: 2px solid #fa0; } /* false negative: got empty, expected tile */
.board td .exp { position: absolute; bottom: 0; right: 1px; font-size: 7px;
                 color: #fa0; line-height: 1; }
.board td .conf { position: absolute; top: 0; left: 1px; font-size: 6px;
                  color: #666; line-height: 1; }
.errors-list { margin-top: 8px; font-size: .8rem; }
.errors-list .err { display: inline-block; padding: 2px 8px; margin: 2px;
                    border-radius: 3px; background: #2a0a0a; color: #f88;
                    border: 1px solid #a33; }
.log-section { margin-top: 12px; }
.log-toggle { color: #7af; cursor: pointer; font-size: .8rem; }
.log-toggle:hover { text-decoration: underline; }
.log-content { background: #0a0a0a; border: 1px solid #222; border-radius: 4px;
               padding: 8px; font-size: .72rem; max-height: 300px; overflow-y: auto;
               color: #888; white-space: pre-wrap; display: none; margin-top: 4px; }
#tip { display: none; position: fixed; background: #222; border: 1px solid #555;
       border-radius: 4px; padding: 6px 10px; font-size: 12px; color: #ddd;
       pointer-events: none; z-index: 1000; max-width: 260px; line-height: 1.5;
       box-shadow: 0 2px 8px rgba(0,0,0,.6); }
#tip .tip-pos { color: #7af; font-weight: bold; }
#tip .tip-conf { color: #8f8; }
#tip .tip-err { color: #f88; }
#tip .tip-cands { color: #aaa; }
#tip .tip-cand-letter { color: #fff; font-weight: bold; }
#tip .tip-cand-pct { color: #888; }
</style>
<script>
document.addEventListener('DOMContentLoaded', function() {
  var tip = document.getElementById('tip');
  document.addEventListener('mouseover', function(e) {
    var td = e.target.closest('td[data-tip]');
    if (!td) { tip.style.display = 'none'; return; }
    tip.innerHTML = td.getAttribute('data-tip');
    tip.style.display = 'block';
    var r = td.getBoundingClientRect();
    var tx = r.right + 8, ty = r.top;
    if (tx + 260 > window.innerWidth) tx = r.left - 268;
    if (ty + tip.offsetHeight > window.innerHeight) ty = window.innerHeight - tip.offsetHeight - 4;
    tip.style.left = tx + 'px'; tip.style.top = ty + 'px';
  });
  document.addEventListener('mouseout', function(e) {
    if (e.target.closest('td[data-tip]')) tip.style.display = 'none';
  });
});
</script>
</head>
<body>
)";

    out << "<div id=\"tip\"></div>\n";
    out << "<h1>eval_local — failing cases</h1>\n";
    out << "<div class=\"summary\">"
        << cases.size() << " failing case(s) out of " << total_files
        << " total &mdash; " << total_correct << "/" << total_tiles
        << " tiles correct (" << std::fixed << std::setprecision(2)
        << (total_tiles > 0 ? 100.0 * total_correct / total_tiles : 0)
        << "%), " << total_occ_errors << " occupancy errors, "
        << perfect_cases << "/" << total_files << " perfect"
        << "</div>\n";

    const char* COL_LABELS = "ABCDEFGHIJKLMNO";

    for (size_t ci = 0; ci < cases.size(); ci++) {
        auto& fc = cases[ci];
        int wrong = fc.tiles - fc.correct;

        out << "<div class=\"case\">\n";
        out << "<div class=\"case-title\">" << fc.name << "</div>\n";
        out << "<div class=\"case-stats\">"
            << fc.correct << "/" << fc.tiles << " tiles correct, "
            << wrong << " wrong, " << fc.occ_err << " occ errors, "
            << std::fixed << std::setprecision(0) << fc.ms << "ms</div>\n";

        // Error summary chips
        out << "<div class=\"errors-list\">";
        for (int r = 0; r < 15; r++) {
            for (int c = 0; c < 15; c++) {
                char exp_ch = fc.gt[r][c];
                char got_ch = fc.cells[r][c].letter;
                char exp_upper = exp_ch ? static_cast<char>(
                    std::toupper(static_cast<unsigned char>(exp_ch))) : 0;
                char got_upper = got_ch ? static_cast<char>(
                    std::toupper(static_cast<unsigned char>(got_ch))) : 0;

                bool is_wrong = false;
                std::string desc;
                if (exp_ch && got_ch && exp_upper != got_upper) {
                    is_wrong = true;
                    desc = std::string(1, COL_LABELS[c]) + std::to_string(r+1)
                         + ": exp=" + std::string(1, exp_ch) + " got=" + std::string(1, got_ch);
                } else if (exp_ch && !got_ch) {
                    is_wrong = true;
                    desc = std::string(1, COL_LABELS[c]) + std::to_string(r+1)
                         + ": exp=" + std::string(1, exp_ch) + " got=.";
                } else if (!exp_ch && got_ch) {
                    is_wrong = true;
                    desc = std::string(1, COL_LABELS[c]) + std::to_string(r+1)
                         + ": exp=. got=" + std::string(1, got_ch);
                }
                if (is_wrong)
                    out << "<span class=\"err\">" << desc << "</span>";
            }
        }
        out << "</div>\n";

        out << "<div class=\"layout\">\n";

        // Original image
        out << "<div class=\"img-col\">\n";
        out << "<div class=\"img-label\">Original</div>\n";
        out << "<img src=\"data:image/png;base64,"
            << base64_encode(fc.orig_png) << "\">\n";
        out << "</div>\n";

        // Debug image (green rect + grid)
        if (!fc.debug_png.empty()) {
            out << "<div class=\"img-col\">\n";
            out << "<div class=\"img-label\">Detected board (green rect + grid)</div>\n";
            out << "<img src=\"data:image/png;base64,"
                << base64_encode(fc.debug_png) << "\">\n";
            out << "</div>\n";
        }

        // Board comparison table
        out << "<div>\n";
        out << "<div class=\"img-label\">Board comparison (hover for details)</div>\n";
        out << "<table class=\"board\">\n";
        out << "<tr><th></th>";
        for (int c = 0; c < 15; c++)
            out << "<th>" << COL_LABELS[c] << "</th>";
        out << "</tr>\n";

        for (int r = 0; r < 15; r++) {
            out << "<tr><th>" << (r+1) << "</th>";
            for (int c = 0; c < 15; c++) {
                char exp_ch = fc.gt[r][c];
                char got_ch = fc.cells[r][c].letter;
                float conf = fc.cells[r][c].confidence;
                char exp_upper = exp_ch ? static_cast<char>(
                    std::toupper(static_cast<unsigned char>(exp_ch))) : 0;
                char got_upper = got_ch ? static_cast<char>(
                    std::toupper(static_cast<unsigned char>(got_ch))) : 0;

                std::string cls;
                std::string display;
                std::string extra;
                std::string tip;

                std::string pos = std::string(1, COL_LABELS[c]) + std::to_string(r+1);

                // Build candidate list HTML for tooltip
                auto cands_html = [&]() -> std::string {
                    std::string s;
                    bool any = false;
                    for (int k = 0; k < 5 && fc.cells[r][c].cand_letters[k]; k++) {
                        if (!any) { s += "<br><span class=tip-cands>"; any = true; }
                        else s += " &nbsp;";
                        s += "<span class=tip-cand-letter>";
                        s += fc.cells[r][c].cand_letters[k];
                        s += "</span><span class=tip-cand-pct>(";
                        s += std::to_string(static_cast<int>(
                                 fc.cells[r][c].cand_scores[k] * 100));
                        s += "%)</span>";
                    }
                    if (any) s += "</span>";
                    return s;
                };

                // Display char: lowercase for blanks, uppercase for normal tiles
                char got_display = got_ch;  // preserves case (a-z = blank)
                char exp_display = exp_ch;

                if (exp_ch && got_ch && exp_upper == got_upper) {
                    // Correct tile — use ground truth char to show blank case
                    cls = "tile ok";
                    display = std::string(1, exp_display);
                    tip = "<span class=tip-pos>" + pos + "</span> " +
                          display + " <span class=tip-conf>" +
                          std::to_string(static_cast<int>(conf * 100)) +
                          "%</span>" + cands_html();
                } else if (exp_ch && got_ch && exp_upper != got_upper) {
                    // Wrong letter
                    cls = "tile wrong";
                    display = std::string(1, got_display);
                    extra = "<span class=\"exp\">" + std::string(1, exp_display) + "</span>";
                    tip = "<span class=tip-pos>" + pos + "</span> " +
                          "<span class=tip-err>exp=" + std::string(1, exp_display) +
                          " got=" + std::string(1, got_display) + " " +
                          std::to_string(static_cast<int>(conf * 100)) +
                          "%</span>" + cands_html();
                } else if (exp_ch && !got_ch) {
                    // False negative (missed tile)
                    cls = "occ-fn";
                    extra = "<span class=\"exp\">" + std::string(1, exp_display) + "</span>";
                    tip = "<span class=tip-pos>" + pos + "</span> " +
                          "<span class=tip-err>MISSED exp=" +
                          std::string(1, exp_display) + "</span>" + cands_html();
                } else if (!exp_ch && got_ch) {
                    // False positive (phantom tile)
                    cls = "tile occ-fp";
                    display = std::string(1, got_display);
                    tip = "<span class=tip-pos>" + pos + "</span> " +
                          "<span class=tip-err>PHANTOM got=" +
                          std::string(1, got_display) + " " +
                          std::to_string(static_cast<int>(conf * 100)) +
                          "%</span>" + cands_html();
                } else {
                    // Empty, correct
                    cls = "";
                    tip = "<span class=tip-pos>" + pos + "</span> empty";
                }

                std::string bg = premium_bg(PREMIUM[r][c]);
                out << "<td class=\"" << cls << "\" style=\"background:" << bg
                    << "\" data-tip=\"" << tip << "\">"
                    << display << extra;
                // Show confidence for wrong cells
                if ((exp_ch && got_ch && exp_upper != got_upper) ||
                    (!exp_ch && got_ch)) {
                    out << "<span class=\"conf\">"
                        << static_cast<int>(conf * 100) << "</span>";
                }
                out << "</td>";
            }
            out << "</tr>\n";
        }
        out << "</table>\n";
        out << "</div>\n"; // close board div

        out << "</div>\n"; // close layout

        // Collapsible log
        out << "<div class=\"log-section\">\n";
        out << "<span class=\"log-toggle\" onclick=\""
            << "var el=this.nextElementSibling;"
            << "el.style.display=el.style.display==='block'?'none':'block';"
            << "\">Show/hide debug log</span>\n";
        out << "<div class=\"log-content\">";
        html_escape(out, fc.log);
        out << "</div>\n";
        out << "</div>\n";

        out << "</div>\n"; // close case
    }

    out << "</body></html>\n";
    std::fprintf(stderr, "HTML report written: %s (%zu failing cases)\n",
                 path.c_str(), cases.size());
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::setbuf(stdout, nullptr);
    if (argc < 2) {
        std::cerr << "Usage: eval_local <testdata_dir> [--html <output.html>]\n";
        return 1;
    }
    std::string dir = argv[1];
    std::string html_path;
    for (int i = 2; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--html") {
            html_path = argv[i+1];
            i++;
        }
    }

    int n_files = 0;
    int total_tiles = 0, total_correct = 0, total_occ_errors = 0;
    int perfect_cases = 0;
    double total_ms = 0;

    // Per-letter confusion tracking
    int per_letter_total[26] = {};
    int per_letter_correct[26] = {};

    std::vector<FailCase> fail_cases;

    std::vector<std::string> files;
    for (auto& entry : fs::directory_iterator(dir)) {
        std::string ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpg") continue;
        std::string name = entry.path().stem().string();
        if (!fs::exists(dir + "/" + name + ".cgp")) continue;
        files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());

    std::printf("%-50s %5s %5s %5s %5s %7s %7s\n",
                "Case", "Tiles", "Cor", "Wrong", "Occ", "Acc%", "ms");
    std::printf("%s\n", std::string(90, '-').c_str());

    for (auto& path : files) {
        std::string name = fs::path(path).stem().string();
        std::string cgp_path = dir + "/" + name + ".cgp";

        std::ifstream cgp_ifs(cgp_path);
        std::string cgp_line;
        std::getline(cgp_ifs, cgp_line);
        char gt[15][15];
        parse_cgp_letters(cgp_line, gt);

        std::ifstream ifs(path, std::ios::binary);
        std::vector<uint8_t> imgdata(std::istreambuf_iterator<char>(ifs), {});

        auto t0 = std::chrono::high_resolution_clock::now();
        auto dr = process_board_image_debug(imgdata);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;

        // Compare per-cell
        int tiles = 0, correct = 0, occ_err = 0;
        for (int r = 0; r < 15; r++) {
            for (int c = 0; c < 15; c++) {
                char exp_ch = gt[r][c];
                char got_ch = dr.cells[r][c].letter;
                bool exp_tile = (exp_ch != 0);

                if (exp_tile != (got_ch != 0)) {
                    occ_err++;
                }

                if (exp_tile) {
                    tiles++;
                    char exp_upper = static_cast<char>(
                        std::toupper(static_cast<unsigned char>(exp_ch)));
                    char got_upper = static_cast<char>(
                        std::toupper(static_cast<unsigned char>(got_ch)));
                    int li = exp_upper - 'A';
                    if (li >= 0 && li < 26) per_letter_total[li]++;
                    if (exp_upper == got_upper) {
                        correct++;
                        if (li >= 0 && li < 26) per_letter_correct[li]++;
                    }
                }
            }
        }

        int wrong = tiles - correct;
        double pct = tiles > 0 ? 100.0 * correct / tiles : 100.0;

        std::printf("%-50s %5d %5d %5d %5d %6.1f%% %6.0f\n",
                    name.c_str(), tiles, correct, wrong, occ_err, pct, ms);

        // Collect failing case for HTML report
        if (!html_path.empty() && (wrong > 0 || occ_err > 0)) {
            FailCase fc;
            fc.name = name;
            fc.orig_png = imgdata;
            fc.debug_png = dr.debug_png;
            std::memcpy(fc.gt, gt, sizeof(gt));
            std::memcpy(fc.cells, dr.cells, sizeof(fc.cells));
            fc.log = dr.log;
            fc.got_cgp = dr.cgp;
            fc.tiles = tiles;
            fc.correct = correct;
            fc.occ_err = occ_err;
            fc.ms = ms;
            fail_cases.push_back(std::move(fc));
        }

        total_tiles += tiles;
        total_correct += correct;
        total_occ_errors += occ_err;
        if (wrong == 0 && occ_err == 0) perfect_cases++;
        n_files++;
    }

    std::printf("%s\n", std::string(90, '=').c_str());
    double total_pct = total_tiles > 0 ? 100.0 * total_correct / total_tiles : 0;
    std::printf("Total: %d files, %d/%d tiles correct (%.2f%%), %d occ errors\n",
                n_files, total_correct, total_tiles, total_pct, total_occ_errors);
    std::printf("Perfect cases: %d/%d\n", perfect_cases, n_files);
    std::printf("Total time: %.0fms (%.1fms/case)\n", total_ms, total_ms / n_files);

    std::printf("\nPer-letter accuracy:\n");
    for (int i = 0; i < 26; i++) {
        if (per_letter_total[i] > 0) {
            double acc = 100.0 * per_letter_correct[i] / per_letter_total[i];
            std::printf("  %c: %d/%d = %.1f%%\n",
                        'A' + i, per_letter_correct[i], per_letter_total[i], acc);
        }
    }

    // Generate HTML report if requested
    if (!html_path.empty()) {
        write_html_report(html_path, fail_cases,
                          n_files, total_tiles, total_correct,
                          total_occ_errors, perfect_cases);
    }
}
