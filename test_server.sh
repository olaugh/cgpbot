#!/usr/bin/env bash
# Run all testdata images through the Gemini OCR server and report accuracy.
# Usage: ./test_server.sh [host]
# Default host: http://localhost:8080

HOST="${1:-http://localhost:8080}"
TESTDATA="testdata"
TMPDIR_BASE=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BASE"' EXIT

# Submit one image in the background, save raw NDJSON to tmpfile
submit() {
    local name="$1" img="$2"
    curl -s --max-time 180 -X POST "$HOST/analyze-gemini" \
        -F "image=@$img" > "$TMPDIR_BASE/$name.ndjson" 2>/dev/null &
}

# Discover test cases and launch in parallel
names=()
for cgp_file in "$TESTDATA"/*.cgp; do
    name=$(basename "$cgp_file" .cgp)
    img=""
    for ext in png jpg jpeg; do
        [[ -f "$TESTDATA/$name.$ext" ]] && { img="$TESTDATA/$name.$ext"; break; }
    done
    [[ -z "$img" ]] && continue
    names+=("$name")
    submit "$name" "$img"
done

echo "Waiting for ${#names[@]} requests against $HOST ..."
wait

# Compare all results with Python
python3 - "$TESTDATA" "$TMPDIR_BASE" "${names[@]}" <<'PYEOF'
import sys, re, json, os

testdata, tmpdir = sys.argv[1], sys.argv[2]
names = sorted(sys.argv[3:])

def parse_board(cgp):
    board_str = cgp.split()[0]
    cells = []
    for row in board_str.split('/'):
        i, row_cells = 0, []
        while i < len(row):
            if row[i].isdigit():
                n = 0
                while i < len(row) and row[i].isdigit():
                    n = n * 10 + int(row[i]); i += 1
                row_cells.extend([' '] * n)
            else:
                row_cells.append(row[i]); i += 1
        cells.extend((row_cells + [' '] * 15)[:15])
    return (cells + [' '] * 225)[:225]

def parse_scores(cgp):
    m = re.search(r'/\s*(\d+)\s+(\d+)', cgp)
    return (int(m.group(1)), int(m.group(2))) if m else None

def extract_cgp(ndjson_path):
    try:
        with open(ndjson_path) as f:
            for line in reversed(f.readlines()):
                line = line.strip()
                if '"cgp"' in line:
                    m = re.search(r'"cgp":"([^"]+)"', line)
                    if m:
                        return m.group(1).replace('\\/', '/')
    except Exception:
        pass
    return None

total_cells = total_correct = total_wrong = 0
scores_correct = scores_total = 0

header = f"{'Case':<14} {'Cells':>6} {'Correct':>7} {'Wrong':>6} {'Board%':>7}  {'Exp scores':<13} {'Got scores':<13} Scores"
print()
print(header)
print('-' * len(header))

for name in names:
    exp_cgp_path = os.path.join(testdata, name + '.cgp')
    ndjson_path  = os.path.join(tmpdir,   name + '.ndjson')

    with open(exp_cgp_path) as f:
        exp_cgp = f.readline().strip()

    got_cgp = extract_cgp(ndjson_path)
    if not got_cgp:
        print(f"{'  '+name:<14}  ERROR — no response")
        continue

    # Board accuracy
    exp_cells = parse_board(exp_cgp)
    got_cells = parse_board(got_cgp)
    case_total = case_correct = case_wrong = 0
    wrong_cells = []
    for i, (e, g) in enumerate(zip(exp_cells, got_cells)):
        if e != ' ' or g != ' ':
            case_total += 1
            if e == g:
                case_correct += 1
            else:
                case_wrong += 1
                col = chr(ord('A') + i % 15)
                row = i // 15 + 1
                wrong_cells.append(f"{col}{row}:{e or '.'}→{g or '.'}")
    pct = f"{case_correct*100/case_total:.1f}%" if case_total else "n/a"

    # Score accuracy
    exp_sc = parse_scores(exp_cgp)
    got_sc = parse_scores(got_cgp)
    if exp_sc:
        scores_total += 1
        sc_mark = '✓' if exp_sc == got_sc else '✗'
        if exp_sc == got_sc:
            scores_correct += 1
        exp_sc_str = f"{exp_sc[0]} {exp_sc[1]}"
        got_sc_str = f"{got_sc[0]} {got_sc[1]}" if got_sc else "?"
    else:
        sc_mark = '-'
        exp_sc_str = got_sc_str = '—'

    print(f"{name:<14} {case_total:>6} {case_correct:>7} {case_wrong:>6} {pct:>7}  {exp_sc_str:<13} {got_sc_str:<13} {sc_mark}")
    if wrong_cells:
        print(f"  {'  '.join(wrong_cells)}")

    total_cells   += case_total
    total_correct += case_correct
    total_wrong   += case_wrong

overall = f"{total_correct*100/total_cells:.1f}%" if total_cells else "n/a"
print('-' * len(header))
print(f"{'TOTAL':<14} {total_cells:>6} {total_correct:>7} {total_wrong:>6} {overall:>7}  {scores_correct}/{scores_total} scores correct")
PYEOF
