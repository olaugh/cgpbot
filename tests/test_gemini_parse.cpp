#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../src/gemini_parse.h"

// Source directory set by CMake
#ifndef SOURCE_DIR
#define SOURCE_DIR "."
#endif

static std::string read_file(const std::string& path) {
    // Try relative path first, then prepend SOURCE_DIR
    std::ifstream f(path);
    if (!f) {
        std::string full = std::string(SOURCE_DIR) + "/" + path;
        f.open(full);
    }
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string grid_to_string(const CellResult cells[15][15]) {
    std::string result;
    for (int r = 0; r < 15; r++) {
        for (int c = 0; c < 15; c++) {
            result += cells[r][c].letter ? cells[r][c].letter : '.';
        }
        result += '\n';
    }
    return result;
}

static int count_cells(const CellResult cells[15][15]) {
    int n = 0;
    for (int r = 0; r < 15; r++)
        for (int c = 0; c < 15; c++)
            if (cells[r][c].letter != 0) n++;
    return n;
}

// --- Tests ---

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    static void test_##name(); \
    static struct Register_##name { \
        Register_##name() { test_##name(); } \
    } register_##name; \
    static void test_##name()

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << msg << "\n"; \
        return; \
    } \
} while(0)

#define PASS(name) do { \
    tests_passed++; \
    std::cout << "  PASS: " << name << "\n"; \
} while(0)

// --- extract_gemini_text ---

TEST(extract_text_basic) {
    tests_run++;
    std::string json = R"({"candidates":[{"content":{"parts":[{"text":"hello world"}]}}]})";
    auto text = extract_gemini_text(json);
    ASSERT(text == "hello world", "expected 'hello world', got '" + text + "'");
    PASS("extract_text_basic");
}

TEST(extract_text_escapes) {
    tests_run++;
    std::string json = R"({"parts":[{"text":"line1\nline2\ttab"}]})";
    auto text = extract_gemini_text(json);
    ASSERT(text == "line1\nline2\ttab",
           "expected newline+tab, got '" + text + "'");
    PASS("extract_text_escapes");
}

TEST(extract_text_last_key) {
    tests_run++;
    // Gemini 2.5 thinking: multiple "text" keys; we want the last one
    std::string json = R"({"parts":[{"text":"thinking..."},{"text":"actual answer"}]})";
    auto text = extract_gemini_text(json);
    ASSERT(text == "actual answer",
           "expected last text, got '" + text + "'");
    PASS("extract_text_last_key");
}

TEST(extract_text_empty) {
    tests_run++;
    std::string json = R"({"error":"something bad"})";
    auto text = extract_gemini_text(json);
    ASSERT(text.empty(), "expected empty, got '" + text + "'");
    PASS("extract_text_empty");
}

TEST(extract_text_escaped_quotes) {
    tests_run++;
    std::string json = R"({"text":"say \"hello\""})";
    auto text = extract_gemini_text(json);
    ASSERT(text == "say \"hello\"",
           "expected escaped quotes, got '" + text + "'");
    PASS("extract_text_escaped_quotes");
}

// --- parse_gemini_board ---

TEST(parse_board_normal_15x15) {
    tests_run++;
    std::string text = R"({"board":[
        ["A",null,null,null,null,null,null,null,null,null,null,null,null,null,"Z"],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,"H",null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,"B"]
    ]})";
    CellResult cells[15][15];
    bool ok = parse_gemini_board(text, cells);
    ASSERT(ok, "parse failed");
    ASSERT(cells[0][0].letter == 'A', "expected A at [0][0]");
    ASSERT(cells[0][14].letter == 'Z', "expected Z at [0][14]");
    ASSERT(cells[7][7].letter == 'H', "expected H at [7][7]");
    ASSERT(cells[14][14].letter == 'B', "expected B at [14][14]");
    ASSERT(cells[1][0].letter == 0, "expected empty at [1][0]");
    ASSERT(count_cells(cells) == 4, "expected 4 cells filled");
    PASS("parse_board_normal_15x15");
}

TEST(parse_board_with_code_fence) {
    tests_run++;
    std::string text = "```json\n"
        "{\"board\":[\n"
        "[\"X\",null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n"
        "[null,null,null,null,null,null,null,null,null,null,null,null,null,null,\"Y\"]\n"
        "]}\n```";
    CellResult cells[15][15];
    bool ok = parse_gemini_board(text, cells);
    ASSERT(ok, "parse failed with code fence");
    ASSERT(cells[0][0].letter == 'X', "expected X at [0][0]");
    ASSERT(cells[14][14].letter == 'Y', "expected Y at [14][14]");
    PASS("parse_board_with_code_fence");
}

TEST(parse_board_blank_tiles) {
    tests_run++;
    std::string text = R"({"board":[
        ["a","B",null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null]
    ]})";
    CellResult cells[15][15];
    bool ok = parse_gemini_board(text, cells);
    ASSERT(ok, "parse failed");
    ASSERT(cells[0][0].letter == 'a', "expected 'a' at [0][0]");
    ASSERT(cells[0][0].is_blank == true, "expected blank at [0][0]");
    ASSERT(cells[0][1].letter == 'B', "expected 'B' at [0][1]");
    ASSERT(cells[0][1].is_blank == false, "expected not blank at [0][1]");
    PASS("parse_board_blank_tiles");
}

TEST(parse_board_14_element_rows) {
    tests_run++;
    // Test with actual Gemini response that has rows with 14 elements
    std::string text = read_file("testcases/gemini_wilders_rows10_11_14elems.txt");
    ASSERT(!text.empty(), "could not read test fixture wilders_rows10_11_14elems");
    CellResult cells[15][15];
    bool ok = parse_gemini_board(text, cells);
    ASSERT(ok, "parse failed for 14-element rows");

    // Row 10 (0-indexed 9) has 14 elements: BOAS should be at positions 9-12
    // Verify key cells that should be present
    ASSERT(cells[3][12].letter == 'W', "expected W at [3][12]");
    ASSERT(cells[4][12].letter == 'I', "expected I at [4][12]");
    ASSERT(cells[5][5].letter == 'K', "expected K at [5][5]");
    ASSERT(cells[6][4].letter == 'J', "expected J at [6][4]");
    ASSERT(cells[7][3].letter == 'B', "expected B at [7][3]");

    // Check that the short rows (10, 11) still parsed correctly
    // Row 10 has BOAS starting from position 9 (14 elements = positions 0-13)
    ASSERT(cells[9][9].letter == 'B', "expected B at [9][9] (short row)");
    ASSERT(cells[9][10].letter == 'O', "expected O at [9][10] (short row)");
    ASSERT(cells[9][11].letter == 'A', "expected A at [9][11] (short row)");
    ASSERT(cells[9][12].letter == 'S', "expected S at [9][12] (short row)");

    // Row 11 has DRYSTONE (14 elements = positions 0-13)
    ASSERT(cells[10][3].letter == 'D', "expected D at [10][3] (short row)");
    ASSERT(cells[10][4].letter == 'R', "expected R at [10][4]");
    ASSERT(cells[10][10].letter == 'E', "expected E at [10][10] (last of DRYSTONE)");

    // Check subsequent rows still parse correctly after short rows
    ASSERT(cells[11][7].letter == 'H', "expected H at [11][7] (after short rows)");
    ASSERT(cells[12][6].letter == 'M', "expected M at [12][6]");
    ASSERT(cells[13][3].letter == 'T', "expected T at [13][3]");
    ASSERT(cells[14][7].letter == 'N', "expected N at [14][7]");

    PASS("parse_board_14_element_rows");
}

TEST(parse_board_14_element_rows_frowie) {
    tests_run++;
    // Frowie with rows 7 and 8 having 14 elements
    std::string text = read_file("testcases/gemini_frowie_rows7_8_14elems.txt");
    ASSERT(!text.empty(), "could not read test fixture frowie_rows7_8_14elems");
    CellResult cells[15][15];
    bool ok = parse_gemini_board(text, cells);
    ASSERT(ok, "parse failed for frowie 14-element rows");

    // Row 1: PHAGES at positions 7, 9-14
    ASSERT(cells[0][7].letter == 'F', "expected F at [0][7]");
    ASSERT(cells[0][9].letter == 'P', "expected P at [0][9]");

    // Row 7 (0-indexed 6) has 14 elements: OM..IN.EEL
    ASSERT(cells[6][4].letter == 'O', "expected O at [6][4]");
    ASSERT(cells[6][5].letter == 'M', "expected M at [6][5]");

    // Row 8 (0-indexed 7) has 14 elements: PRINTED..AY
    ASSERT(cells[7][3].letter == 'P', "expected P at [7][3]");
    ASSERT(cells[7][9].letter == 'D', "expected D at [7][9]");

    // Check rows after the short rows still parse correctly
    ASSERT(cells[8][4].letter == 'E', "expected E at [8][4]");
    ASSERT(cells[14][0].letter == 'B', "expected B at [14][0]");
    ASSERT(cells[14][3].letter == 'D', "expected D at [14][3]");

    PASS("parse_board_14_element_rows_frowie");
}

TEST(parse_board_16_element_row) {
    tests_run++;
    // Rage with row 4 having 16 elements
    std::string text = read_file("testcases/gemini_rage_row4_16elems.txt");
    ASSERT(!text.empty(), "could not read test fixture rage_row4_16elems");
    CellResult cells[15][15];
    bool ok = parse_gemini_board(text, cells);
    ASSERT(ok, "parse failed for 16-element row");

    // Row 1: RAGE..FILL
    ASSERT(cells[0][5].letter == 'R', "expected R at [0][5]");
    ASSERT(cells[0][8].letter == 'E', "expected E at [0][8]");
    ASSERT(cells[0][11].letter == 'F', "expected F at [0][11]");

    // Row 4 (0-indexed 3) has 16 elements: F.....EVICT
    // With 16 elements, only the first 15 are stored
    ASSERT(cells[3][5].letter == 'F', "expected F at [3][5]");
    ASSERT(cells[3][11].letter == 'E', "expected E at [3][11]");
    ASSERT(cells[3][14].letter == 'C', "expected C at [3][14]");
    // The 16th element (T) is truncated

    // Check rows after the 16-element row still parse correctly
    ASSERT(cells[4][4].letter == 'D', "expected D at [4][4]");
    ASSERT(cells[4][5].letter == 'A', "expected A at [4][5]");
    ASSERT(cells[5][3].letter == 'V', "expected V at [5][3]");
    ASSERT(cells[14][3].letter == 'G', "expected G at [14][3]");

    PASS("parse_board_16_element_row");
}

TEST(parse_board_raw_array_no_key) {
    tests_run++;
    // Some Gemini responses return just the array without "board" key
    std::string text = R"([
        ["A","B","C",null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
        [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null]
    ])";
    CellResult cells[15][15];
    bool ok = parse_gemini_board(text, cells);
    ASSERT(ok, "parse failed for raw array");
    ASSERT(cells[0][0].letter == 'A', "expected A at [0][0]");
    ASSERT(cells[0][1].letter == 'B', "expected B at [0][1]");
    ASSERT(cells[0][2].letter == 'C', "expected C at [0][2]");
    PASS("parse_board_raw_array_no_key");
}

TEST(parse_board_garbage_returns_false) {
    tests_run++;
    CellResult cells[15][15];

    ASSERT(!parse_gemini_board("not json at all", cells),
           "should fail on garbage input");
    ASSERT(!parse_gemini_board("", cells),
           "should fail on empty input");
    ASSERT(!parse_gemini_board("{\"error\": \"rate limited\"}", cells),
           "should fail on error response");
    PASS("parse_board_garbage_returns_false");
}

TEST(cells_to_cgp_basic) {
    tests_run++;
    CellResult cells[15][15] = {};
    cells[0][0].letter = 'A';
    cells[0][1].letter = 'B';
    cells[0][5].letter = 'X';
    cells[14][14].letter = 'Z';
    auto cgp = cells_to_cgp(cells);
    // Row 0: AB3X9
    // Rows 1-13: 15 each
    // Row 14: 14Z
    ASSERT(cgp.find("AB3X9") == 0, "expected AB3X9 at start, got: " + cgp);
    ASSERT(cgp.find("14Z") != std::string::npos,
           "expected 14Z at end, got: " + cgp);
    PASS("cells_to_cgp_basic");
}

TEST(cells_to_cgp_blanks) {
    tests_run++;
    CellResult cells[15][15] = {};
    cells[0][0].letter = 'a';  // blank tile
    cells[0][0].is_blank = true;
    cells[0][1].letter = 'B';  // regular tile
    auto cgp = cells_to_cgp(cells);
    ASSERT(cgp.substr(0, 4) == "aB13", "expected aB13 start, got: " + cgp);
    PASS("cells_to_cgp_blanks");
}

// --- All fixtures regression test ---
// Parses ALL saved Gemini response fixtures to catch regressions.

TEST(parse_all_fixtures) {
    tests_run++;
    struct Fixture {
        const char* path;
        int min_cells; // minimum expected non-empty cells
    };
    Fixture fixtures[] = {
        {"testcases/gemini_flies_79cells_ok.txt", 70},
        {"testcases/gemini_flies_78cells_debug.txt", 70},
        {"testcases/gemini_flies_79cells_v2.txt", 70},
        {"testcases/gemini_frowie_83cells_ok.txt", 75},
        {"testcases/gemini_frowie_83cells_v2.txt", 75},
        {"testcases/gemini_frowie_rows7_8_14elems.txt", 70},
        {"testcases/gemini_rage_85cells_ok.txt", 75},
        {"testcases/gemini_rage_85cells_v2.txt", 75},
        {"testcases/gemini_rage_row4_16elems.txt", 75},
        {"testcases/gemini_vamose_85cells_ok.txt", 70},
        {"testcases/gemini_vamose_85cells_v2.txt", 70},
        {"testcases/gemini_vom_85cells_ok.txt", 75},
        {"testcases/gemini_vom_85cells_v2.txt", 75},
        {"testcases/gemini_vom_88cells_row10_14elems.txt", 75},
        {"testcases/gemini_wilders_40cells_ok.txt", 35},
        {"testcases/gemini_wilders_40cells_row10_14elems.txt", 35},
        {"testcases/gemini_wilders_rows10_11_14elems.txt", 35},
    };
    const int n_fixtures = sizeof(fixtures) / sizeof(fixtures[0]);

    int passed = 0;
    for (const auto& fix : fixtures) {
        std::string text = read_file(fix.path);
        if (text.empty()) {
            std::cerr << "  SKIP: " << fix.path << " (file not found)\n";
            continue;
        }
        // Try parsing directly; if that fails, try extracting from API JSON first
        CellResult cells[15][15];
        bool ok = parse_gemini_board(text, cells);
        if (!ok) {
            std::string extracted = extract_gemini_text(text);
            if (!extracted.empty())
                ok = parse_gemini_board(extracted, cells);
        }
        if (!ok) {
            std::cerr << "  FAIL: " << fix.path << " (parse returned false)\n";
            return;
        }
        int n = count_cells(cells);
        if (n < fix.min_cells) {
            std::cerr << "  FAIL: " << fix.path << " (only " << n
                      << " cells, expected >= " << fix.min_cells << ")\n";
            return;
        }
        passed++;
    }
    ASSERT(passed == n_fixtures,
           "not all fixtures loaded (" + std::to_string(passed)
           + "/" + std::to_string(n_fixtures) + ")");
    PASS("parse_all_fixtures (" + std::to_string(n_fixtures) + " responses)");
}

// --- Vom response with 14-element row 10 ---

TEST(parse_vom_14_element_row) {
    tests_run++;
    std::string text = read_file("testcases/gemini_vom_88cells_row10_14elems.txt");
    ASSERT(!text.empty(), "could not read vom fixture");
    CellResult cells[15][15];
    bool ok = parse_gemini_board(text, cells);
    ASSERT(ok, "parse failed for vom 14-element row");

    // Row 10 (0-indexed 9) has 14 elements
    // Expected: I...YIKED.A...
    ASSERT(cells[9][0].letter == 'I', "expected I at [9][0]");
    ASSERT(cells[9][4].letter == 'Y', "expected Y at [9][4]");

    // Rows after the short row should parse correctly
    ASSERT(cells[10][0].letter == 'N', "expected N at [10][0]");
    ASSERT(cells[13][1].letter == 'S', "expected S at [13][1]");
    ASSERT(cells[14][8].letter == 'A', "expected A at [14][8]");

    PASS("parse_vom_14_element_row");
}

// --- Real Gemini response integration test ---

TEST(parse_full_api_response) {
    tests_run++;
    // Simulate a full Gemini API response (as returned by curl)
    std::string api_response = R"({
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "```json\n{\n  \"board\": [\n    [\"F\",\"L\",\"I\",\"E\",\"S\",null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],\n    [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null]\n  ]\n}\n```"
          }
        ],
        "role": "model"
      }
    }
  ]
})";
    auto text = extract_gemini_text(api_response);
    ASSERT(!text.empty(), "extract_gemini_text should find text in API response");

    CellResult cells[15][15];
    bool ok = parse_gemini_board(text, cells);
    ASSERT(ok, "parse should succeed on API response");
    ASSERT(cells[0][0].letter == 'F', "expected F at [0][0]");
    ASSERT(cells[0][4].letter == 'S', "expected S at [0][4]");
    ASSERT(count_cells(cells) == 5, "expected 5 cells");
    PASS("parse_full_api_response");
}

// --- parse_retry_response (cropped cell revalidation) ---

TEST(retry_basic_letters) {
    tests_run++;
    std::vector<RetryCell> positions = {{0,0,false}, {0,1,false}, {0,2,false}};
    auto results = parse_retry_response(R"(["A", "B", "C"])", positions);
    ASSERT(results.size() == 3, "expected 3 results");
    ASSERT(results[0].letter == 'A', "expected A");
    ASSERT(results[1].letter == 'B', "expected B");
    ASSERT(results[2].letter == 'C', "expected C");
    ASSERT(results[0].confidence == 0.8f, "missing cell should have confidence 0.8");
    PASS("retry_basic_letters");
}

TEST(retry_with_nulls) {
    tests_run++;
    std::vector<RetryCell> positions = {
        {0,0,false}, {0,1,false}, {0,2,false}, {0,3,false}
    };
    auto results = parse_retry_response(R"(["F", null, "s", "A"])", positions);
    ASSERT(results.size() == 4, "expected 4 results");
    ASSERT(results[0].letter == 'F', "expected F");
    ASSERT(results[1].letter == 0, "expected null (empty)");
    ASSERT(results[2].letter == 's', "expected s (blank)");
    ASSERT(results[2].is_blank == true, "lowercase should be blank");
    ASSERT(results[3].letter == 'A', "expected A");
    PASS("retry_with_nulls");
}

TEST(retry_disputed_confidence) {
    tests_run++;
    std::vector<RetryCell> positions = {
        {0,0,true},   // disputed
        {0,1,false},  // missing
        {0,2,true},   // disputed
    };
    auto results = parse_retry_response(R"(["X", "Y", "Z"])", positions);
    ASSERT(results[0].confidence == 0.75f, "disputed should have confidence 0.75");
    ASSERT(results[1].confidence == 0.8f, "missing should have confidence 0.8");
    ASSERT(results[2].confidence == 0.75f, "disputed should have confidence 0.75");
    PASS("retry_disputed_confidence");
}

TEST(retry_blank_tiles) {
    tests_run++;
    std::vector<RetryCell> positions = {{0,0,false}, {0,1,false}};
    auto results = parse_retry_response(R"(["a", "B"])", positions);
    ASSERT(results[0].letter == 'a', "expected lowercase a");
    ASSERT(results[0].is_blank == true, "lowercase = blank");
    ASSERT(results[1].letter == 'B', "expected uppercase B");
    ASSERT(results[1].is_blank == false, "uppercase = not blank");
    PASS("retry_blank_tiles");
}

TEST(retry_with_code_fence) {
    tests_run++;
    std::vector<RetryCell> positions = {{0,0,false}, {0,1,false}};
    std::string text = "```json\n[\"M\", \"N\"]\n```";
    auto results = parse_retry_response(text, positions);
    ASSERT(results[0].letter == 'M', "expected M through code fence");
    ASSERT(results[1].letter == 'N', "expected N through code fence");
    PASS("retry_with_code_fence");
}

TEST(retry_all_nulls) {
    tests_run++;
    std::vector<RetryCell> positions = {{0,0,true}, {0,1,true}, {0,2,true}};
    auto results = parse_retry_response(R"([null, null, null])", positions);
    ASSERT(results.size() == 3, "expected 3 results");
    ASSERT(results[0].letter == 0, "expected null");
    ASSERT(results[1].letter == 0, "expected null");
    ASSERT(results[2].letter == 0, "expected null");
    PASS("retry_all_nulls");
}

TEST(retry_empty_array) {
    tests_run++;
    std::vector<RetryCell> positions = {{0,0,false}};
    auto results = parse_retry_response(R"([])", positions);
    ASSERT(results.size() == 1, "expected 1 result");
    ASSERT(results[0].letter == 0, "unparsed element should be empty");
    PASS("retry_empty_array");
}

TEST(retry_no_array) {
    tests_run++;
    std::vector<RetryCell> positions = {{0,0,false}};
    auto results = parse_retry_response("just some garbage text", positions);
    ASSERT(results.size() == 1, "expected 1 result");
    ASSERT(results[0].letter == 0, "no array = all empty");
    PASS("retry_no_array");
}

TEST(retry_multiline_whitespace) {
    tests_run++;
    std::vector<RetryCell> positions = {{0,0,false}, {0,1,false}, {0,2,false}};
    std::string text = "[\n  \"A\",\n  null,\n  \"Z\"\n]";
    auto results = parse_retry_response(text, positions);
    ASSERT(results[0].letter == 'A', "expected A");
    ASSERT(results[1].letter == 0, "expected null");
    ASSERT(results[2].letter == 'Z', "expected Z");
    PASS("retry_multiline_whitespace");
}

TEST(retry_fewer_results_than_positions) {
    tests_run++;
    // Gemini returns fewer elements than we asked for
    std::vector<RetryCell> positions = {
        {0,0,false}, {0,1,false}, {0,2,false}, {0,3,false}
    };
    auto results = parse_retry_response(R"(["A", "B"])", positions);
    ASSERT(results.size() == 4, "should have 4 results");
    ASSERT(results[0].letter == 'A', "expected A");
    ASSERT(results[1].letter == 'B', "expected B");
    ASSERT(results[2].letter == 0, "unparsed should be empty");
    ASSERT(results[3].letter == 0, "unparsed should be empty");
    PASS("retry_fewer_results_than_positions");
}

TEST(retry_real_gemini_response) {
    tests_run++;
    // Simulate a real Gemini retry response wrapped in API JSON
    std::string api_json = R"({
        "candidates": [{"content": {"parts": [
            {"text": "[\"E\", null, \"r\", \"T\"]"}
        ]}}]
    })";
    auto text = extract_gemini_text(api_json);
    ASSERT(!text.empty(), "should extract text from API response");

    std::vector<RetryCell> positions = {
        {3,5,false}, {3,6,true}, {7,2,true}, {10,10,false}
    };
    auto results = parse_retry_response(text, positions);
    ASSERT(results[0].letter == 'E', "expected E");
    ASSERT(results[0].confidence == 0.8f, "missing cell confidence");
    ASSERT(results[1].letter == 0, "expected null for disputed empty");
    ASSERT(results[2].letter == 'r', "expected blank r");
    ASSERT(results[2].is_blank == true, "lowercase = blank");
    ASSERT(results[2].confidence == 0.75f, "disputed cell confidence");
    ASSERT(results[3].letter == 'T', "expected T");
    PASS("retry_real_gemini_response");
}

// --- check_board_connectivity ---

// Helper to set a cell's letter
static void set_cell(CellResult cells[15][15], int r, int c, char letter) {
    cells[r][c].letter = letter;
    cells[r][c].confidence = 1.0f;
    cells[r][c].is_blank = (letter >= 'a' && letter <= 'z');
}

TEST(connectivity_empty_board) {
    tests_run++;
    CellResult cells[15][15] = {};
    auto result = check_board_connectivity(cells);
    ASSERT(!result.center_empty, "empty board should not flag center_empty");
    ASSERT(result.main_component.empty(), "no main component on empty board");
    ASSERT(result.islands.empty(), "no islands on empty board");
    ASSERT(result.bridge_candidates.empty(), "no bridges on empty board");
    PASS("connectivity_empty_board");
}

TEST(connectivity_single_tile_at_center) {
    tests_run++;
    CellResult cells[15][15] = {};
    set_cell(cells, 7, 7, 'A');
    auto result = check_board_connectivity(cells);
    ASSERT(!result.center_empty, "center has a tile");
    ASSERT(result.main_component.size() == 1, "one tile in main component");
    ASSERT(result.islands.empty(), "no islands");
    ASSERT(result.bridge_candidates.empty(), "no bridges needed");
    PASS("connectivity_single_tile_at_center");
}

TEST(connectivity_center_empty) {
    tests_run++;
    CellResult cells[15][15] = {};
    // Tiles exist but not at center
    set_cell(cells, 0, 0, 'A');
    set_cell(cells, 0, 1, 'B');
    auto result = check_board_connectivity(cells);
    ASSERT(result.center_empty, "center should be flagged as empty");
    ASSERT(result.main_component.empty(), "no main component when center empty");
    ASSERT(result.islands.size() == 1, "tiles form one island");
    ASSERT(result.islands[0].size() == 2, "island has 2 tiles");
    // (7,7) should be a bridge candidate
    bool found_center = false;
    for (const auto& [r, c] : result.bridge_candidates)
        if (r == 7 && c == 7) found_center = true;
    ASSERT(found_center, "(7,7) should be a bridge candidate");
    PASS("connectivity_center_empty");
}

TEST(connectivity_fully_connected) {
    tests_run++;
    CellResult cells[15][15] = {};
    // Cross through center
    for (int i = 5; i <= 9; i++) set_cell(cells, 7, i, 'A');  // horizontal
    for (int i = 5; i <= 9; i++) set_cell(cells, i, 7, 'B');  // vertical
    cells[7][7].letter = 'C';  // center overlap
    auto result = check_board_connectivity(cells);
    ASSERT(!result.center_empty, "center has tile");
    ASSERT(result.main_component.size() == 9, "9 unique tiles in cross");
    ASSERT(result.islands.empty(), "fully connected, no islands");
    ASSERT(result.bridge_candidates.empty(), "no bridges needed");
    PASS("connectivity_fully_connected");
}

TEST(connectivity_one_island) {
    tests_run++;
    CellResult cells[15][15] = {};
    // Main group connected to center
    set_cell(cells, 7, 7, 'A');
    set_cell(cells, 7, 8, 'B');
    set_cell(cells, 7, 9, 'C');
    // Isolated island
    set_cell(cells, 0, 0, 'X');
    set_cell(cells, 0, 1, 'Y');
    auto result = check_board_connectivity(cells);
    ASSERT(!result.center_empty, "center has tile");
    ASSERT(result.main_component.size() == 3, "3 tiles in main component");
    ASSERT(result.islands.size() == 1, "1 island");
    ASSERT(result.islands[0].size() == 2, "island has 2 tiles");
    PASS("connectivity_one_island");
}

TEST(connectivity_bridge_candidates) {
    tests_run++;
    CellResult cells[15][15] = {};
    // Main component at center
    set_cell(cells, 7, 7, 'A');
    // Island at (3,3)
    set_cell(cells, 3, 3, 'Z');
    auto result = check_board_connectivity(cells);
    ASSERT(result.islands.size() == 1, "1 island");
    // Bridge candidates should be the 4 neighbors of (3,3)
    ASSERT(result.bridge_candidates.size() == 4,
           "expected 4 bridge candidates around (3,3), got "
           + std::to_string(result.bridge_candidates.size()));
    // Verify all are neighbors of (3,3)
    for (const auto& [r, c] : result.bridge_candidates) {
        bool is_neighbor = (r == 2 && c == 3) || (r == 4 && c == 3)
                        || (r == 3 && c == 2) || (r == 3 && c == 4);
        ASSERT(is_neighbor, "bridge candidate should be neighbor of (3,3)");
    }
    PASS("connectivity_bridge_candidates");
}

// --- find_gap_candidates ---

TEST(gap_single_horizontal) {
    tests_run++;
    CellResult cells[15][15] = {};
    // AB.DE on row 0 (2+ on each side)
    set_cell(cells, 0, 0, 'A');
    set_cell(cells, 0, 1, 'B');
    // gap at col 2
    set_cell(cells, 0, 3, 'D');
    set_cell(cells, 0, 4, 'E');
    auto gaps = find_gap_candidates(cells);
    ASSERT(gaps.size() == 1, "expected 1 gap, got " + std::to_string(gaps.size()));
    ASSERT(gaps[0].r == 0 && gaps[0].c == 2, "gap should be at (0,2)");
    ASSERT(gaps[0].horizontal == true, "gap should be horizontal");
    PASS("gap_single_horizontal");
}

TEST(gap_single_vertical) {
    tests_run++;
    CellResult cells[15][15] = {};
    // AB.DE vertically in col 0 (2+ on each side)
    set_cell(cells, 0, 0, 'A');
    set_cell(cells, 1, 0, 'B');
    // gap at row 2
    set_cell(cells, 3, 0, 'D');
    set_cell(cells, 4, 0, 'E');
    auto gaps = find_gap_candidates(cells);
    ASSERT(gaps.size() == 1, "expected 1 gap, got " + std::to_string(gaps.size()));
    ASSERT(gaps[0].r == 2 && gaps[0].c == 0, "gap should be at (2,0)");
    ASSERT(gaps[0].horizontal == false, "gap should be vertical");
    PASS("gap_single_vertical");
}

TEST(gap_two_empty_no_match) {
    tests_run++;
    CellResult cells[15][15] = {};
    // AB..DE — 2-cell gap, should NOT match
    set_cell(cells, 0, 0, 'A');
    set_cell(cells, 0, 1, 'B');
    // 2 empty cells at 2,3
    set_cell(cells, 0, 4, 'D');
    set_cell(cells, 0, 5, 'E');
    auto gaps = find_gap_candidates(cells);
    ASSERT(gaps.empty(), "2-cell gap should not match");
    PASS("gap_two_empty_no_match");
}

TEST(gap_single_tiles_no_match) {
    tests_run++;
    CellResult cells[15][15] = {};
    // A.B.C — single tiles separated by empties, should NOT match
    // (requires 2+ letters on each side)
    set_cell(cells, 0, 0, 'A');
    set_cell(cells, 0, 2, 'B');
    set_cell(cells, 0, 4, 'C');
    auto gaps = find_gap_candidates(cells);
    ASSERT(gaps.empty(), "single tiles separated by empties should not match");
    PASS("gap_single_tiles_no_match");
}

TEST(gap_no_gaps) {
    tests_run++;
    CellResult cells[15][15] = {};
    // ABCDE — no gaps
    for (int c = 0; c < 5; c++) set_cell(cells, 0, c, 'A' + c);
    auto gaps = find_gap_candidates(cells);
    ASSERT(gaps.empty(), "continuous word should have no gaps");
    PASS("gap_no_gaps");
}

TEST(gap_edge_position) {
    tests_run++;
    CellResult cells[15][15] = {};
    // Gap at col 0 with only right side — should NOT match
    // .AB at start of row
    set_cell(cells, 0, 1, 'A');
    set_cell(cells, 0, 2, 'B');
    auto gaps = find_gap_candidates(cells);
    ASSERT(gaps.empty(), "gap at edge with only one side should not match");
    PASS("gap_edge_position");
}

int main() {
    std::cout << "Running Gemini parse tests...\n";
    // All tests auto-register via static constructors
    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed.\n";
    return tests_passed == tests_run ? 0 : 1;
}
