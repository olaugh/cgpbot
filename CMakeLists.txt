cmake_minimum_required(VERSION 3.22)
project(cgpbot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include(FetchContent)

# ── DPP (Discord++) ──────────────────────────────────────────────────────────

FetchContent_Declare(
    dpp
    GIT_REPOSITORY https://github.com/brainboxdotcc/DPP.git
    GIT_TAG        v10.0.35
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(dpp)

# ── cpp-httplib (test server) ────────────────────────────────────────────────

set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(httplib)

# ── OpenCV (system) ──────────────────────────────────────────────────────────

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ── FreeType2 (for template rendering) ──────────────────────────────────────

find_package(PkgConfig REQUIRED)
pkg_check_modules(FREETYPE2 REQUIRED IMPORTED_TARGET freetype2)

# ── Tesseract (optional, system) ─────────────────────────────────────────────

pkg_check_modules(TESSERACT IMPORTED_TARGET tesseract lept)

# ── Board processing library (shared) ────────────────────────────────────────

add_library(board_lib STATIC src/board.cpp)
target_include_directories(board_lib PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(board_lib PUBLIC ${OpenCV_LIBS} PkgConfig::FREETYPE2)

# Pass font path as compile-time define
target_compile_definitions(board_lib PUBLIC
    FONT_PATH="${CMAKE_SOURCE_DIR}/fonts/RobotoMono-Bold.ttf")

if(TESSERACT_FOUND)
    target_compile_definitions(board_lib PUBLIC HAS_TESSERACT=1)
    target_link_libraries(board_lib PUBLIC PkgConfig::TESSERACT)
endif()

# ── Discord bot ──────────────────────────────────────────────────────────────

add_executable(cgpbot src/main.cpp)
target_link_libraries(cgpbot PRIVATE board_lib dpp)

# ── Test bench (local web UI) ────────────────────────────────────────────────

add_executable(cgptest src/testapp.cpp)
target_link_libraries(cgptest PRIVATE board_lib httplib::httplib)

# ── Occupancy test / diagnostics ─────────────────────────────────────────────

add_executable(occ_test src/occ_test.cpp)
target_link_libraries(occ_test PRIVATE board_lib)

add_executable(occ_diag src/occ_diag.cpp)
target_link_libraries(occ_diag PRIVATE board_lib)

add_executable(color_survey src/color_survey.cpp)
target_link_libraries(color_survey PRIVATE board_lib)

add_executable(cell_hsv src/cell_hsv.cpp)
target_link_libraries(cell_hsv PRIVATE board_lib)

# ── Gemini parse unit tests ────────────────────────────────────────────────

add_executable(test_gemini_parse tests/test_gemini_parse.cpp)
target_include_directories(test_gemini_parse PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(test_gemini_parse PRIVATE
    SOURCE_DIR="${CMAKE_SOURCE_DIR}")
