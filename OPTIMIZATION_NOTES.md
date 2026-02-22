# Gemini OCR Pipeline — Speed & Accuracy Improvement Ideas

Notes on potential improvements to the Scrabble board screenshot analysis
pipeline in `src/testapp.cpp`.

## Current Pipeline (sequential)

1. **OpenCV board detection** — find board rect, cell size
2. **Color-based occupancy detection** — which cells have tiles
3. **Gemini Flash main OCR** — full screenshot + occupancy grid prompt
4. **Enforce occupancy mask** — clear cells Gemini hallucinated
5. **Retry missing cells** — crop + re-query cells the mask says are occupied but Gemini returned null
6. **Bag-math verification** — re-query cells with over-counted letters
7. **Rack verification** — re-query rack tile crops if count mismatch or blanks
8. **Dictionary validation** — re-query cells forming invalid words

Steps 5–8 are conditional and currently run **sequentially**. Each is an
independent Gemini API call blocked on the previous one finishing.

---

## Speed Ideas

### 1. Parallelize follow-up Gemini calls

Steps 5–8 are mostly independent of each other. After the main OCR result is
parsed and the occupancy mask is enforced (step 4), we know:
- Which cells are missing (step 5 input)
- Which letters are over-counted via bag math (step 6 input)
- Whether the rack count mismatches (step 7 input)

Steps 5, 6, and 7 can fire **concurrently** using `std::async` or threads.
Step 8 (dictionary validation) depends on the corrected board state from 5+6,
so it must wait — but it could overlap with step 7 (rack verification).

**Expected savings**: Each follow-up call takes ~1–3s. Running 3 in parallel
instead of serial could cut ~2–6s off total time.

### 2. Use libcurl instead of popen("curl ...")

Every Gemini call currently spawns a child process via `popen()`:
```cpp
std::string cmd = "curl -s --max-time 60 -X POST ...";
FILE* pipe = popen(cmd.c_str(), "r");
```

Using libcurl directly (or cpp-httplib's client) avoids:
- Process fork overhead (~5–10ms per call)
- Shell argument serialization
- Temp file writes for payloads (the base64 image is written to `/tmp/gemini_XXXXXX`)

With libcurl you can POST directly from memory and reuse a connection (HTTP
keep-alive) across the follow-up calls, saving TLS handshake time (~50–100ms
per call).

### 3. Downscale screenshots before sending

The main OCR call sends the full screenshot PNG as base64. Gemini bills by
image tokens which scale with resolution. A typical Woogles.io screenshot
might be 1000x850+ pixels, but the 15x15 grid doesn't need full resolution
for letter OCR.

Downscaling to ~768px width (or even 512px if quality holds up) would:
- Reduce base64 payload size → faster upload
- Reduce Gemini input tokens → lower cost
- Possibly reduce Gemini processing time

**Test this carefully** — too much downscaling will hurt accuracy on small
subscripts and blank tile detection.

### 4. Send cropped board region only

Instead of sending the full screenshot (which includes UI chrome, chat,
timers, etc.), crop to just the board + rack + score area after OpenCV
detects the board rect. This reduces image size and removes distracting
visual noise that Gemini must parse.

The occupancy detection (step 2) already knows `bx, by, cell_sz`, so the
crop coordinates are readily available.

### 5. Pre-crop individual cells during OpenCV step

Currently the retry/verify calls (steps 5–8) crop cells on-demand after
Gemini responds. Since OpenCV already knows the board grid geometry, all 225
cell crops could be pre-computed during step 1–2 and stored in memory. This
removes the redundant `cv::imdecode` calls in each follow-up step (the full
image is decoded from the PNG buffer 4+ separate times).

### 6. Batch all follow-up cells into a single API call

Steps 5, 6, and 8 each send cropped cell images to Gemini separately. These
could potentially be merged into a **single call** with all suspect cells,
tagged by reason:
```
"Cell A3: possibly missing. Cell H7: letter count issue (too many E's).
Cell K12: forms invalid word QOPH."
```

One call with 15 crops is faster than three calls with 5 crops each, since
the per-request overhead (TLS, cold start, scheduling) dominates for small
payloads.

---

## Accuracy Ideas

### 7. Two-pass OCR with targeted re-prompting

After the main OCR, identify low-confidence areas (cells where occupancy
mask and Gemini disagree, or letters that are commonly confused) and send
**higher-resolution crops** of just those cells with a more targeted prompt
that lists the likely candidates.

The current retry prompts are generic ("identify the letter"). A targeted
prompt like "Is this an H, N, or I?" would constrain Gemini's output space
and improve accuracy on confusable letters.

### 8. Confidence-weighted dictionary correction

The current dictionary validation (step 8) re-queries all cells in invalid
words equally. Instead, assign confidence scores based on:
- Whether the cell was already corrected in a previous step
- Whether the letter is in a commonly-confused pair (H/N, O/Q/D, E/F, S/Z)
- Whether changing that one letter would make the word valid

Only re-query the **lowest-confidence cell** in each invalid word first. If
that fixes the word, skip the rest.

### 9. Use Gemini's structured output mode

Instead of asking for free-form JSON and parsing it manually with string
searches, use Gemini's structured output / JSON mode
(`response_mime_type: "application/json"` with a response schema). This
eliminates parsing failures from markdown fences, extra text, or malformed
JSON — which the code currently handles defensively in multiple places.

### 10. Cross-reference rack + bag + board for blank detection

The code already does bag-math validation, but blank tile detection remains
tricky (blanks look like empty squares on the rack). A stronger constraint:
if the bag is visible and we can count total tiles (board + bag + rack = 100),
then we know exactly how many blanks should exist. If the board has N
lowercase letters (placed blanks) and total blank count is 2, then
`2 - N` blanks must be on the rack or in the bag. This is already partially
implemented but could be made more aggressive — e.g., if we know 1 blank
must be on the rack, force one rack tile to `?` even if Gemini reads a letter
on it.

### 11. Ensemble with a second model or a second prompt

For high-stakes cells (e.g., triple-word-score positions, or cells that
change the validity of multiple words), run the same crop through Gemini
**twice with different prompts** and take the consensus. This catches
non-deterministic errors without needing a different model.

### 12. Leverage the font rendering for template matching

The app already has `fonts/RobotoMono-Bold.ttf`. Scrabble tiles on
Woogles.io use known fonts. For each cropped cell, render all 26 letters at
the matching size and do template matching (normalized cross-correlation)
as a fast local check before calling Gemini. This could resolve
high-confidence cells without any API call and reserve Gemini only for
ambiguous ones.

---

## Cost Ideas

### 13. Skip Gemini entirely for high-confidence cells

If template matching (idea 12) or a lightweight local model can identify
cells with >95% confidence, exclude those from the Gemini prompt and
include only the ambiguous ones. This reduces image payload and token count.

### 14. Use Gemini 2.5 Flash-Lite for follow-up calls

The retry/verify calls (steps 5–8) send small, tightly-cropped cell images
with simple prompts ("what letter is this?"). These don't need the full
Flash model. Gemini 2.5 Flash-Lite costs $0.10/$0.40 per 1M tokens
(3x cheaper than Flash) and may be sufficient for single-letter OCR on
clean crops.

### 15. Implicit caching across screenshots

The prompt templates are identical across screenshots. Google's implicit
context caching (enabled by default) should already discount repeated prompt
text. The savings are small since images dominate the token count, but it's
free.
