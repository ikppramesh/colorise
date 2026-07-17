#!/usr/bin/env bash
# ==============================================================================
#  B&W TO COLOUR VIDEO CONVERTER
#
#  AI-powered colorisation using Zhang et al. (2016)
#  "Colorful Image Colorization" — deep learning via OpenCV DNN
#
#  HOW IT WORKS:
#    1. Converts each frame to LAB colour space
#    2. Feeds the L (lightness) channel into a Caffe neural network
#    3. Network predicts the A and B (colour) channels
#    4. Merges L + predicted AB → full colour frame
#    5. FFmpeg reassembles frames + original audio into final MP4
#
#  USAGE:
#    ./colorise.sh                  Convert all files in Source/
#    ./colorise.sh -f movie.mp4     Convert a single file
#    ./colorise.sh -h               Show help
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/Source"
OUTPUT_DIR="$SCRIPT_DIR/Output"
VENV_DIR="$SCRIPT_DIR/.venv"
ENGINE="$SCRIPT_DIR/colorise_engine.py"

EXTENSIONS=("mp4" "mkv" "avi" "mov" "wmv" "m4v" "flv" "webm" "mpg" "mpeg" "ts" "ogv" "3gp")

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()     { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[DONE]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

print_banner() {
    echo -e "${BOLD}"
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║          B&W → COLOUR  AI VIDEO COLORISER               ║"
    echo "  ║     Zhang et al. 2016 · OpenCV DNN · Auto-colour        ║"
    echo "  ╚══════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
}

usage() {
    echo ""
    echo -e "${BOLD}USAGE:${RESET}  ./colorise.sh [OPTIONS]"
    echo ""
    echo -e "${BOLD}OPTIONS:${RESET}"
    echo "  -f <file>    Colorise a single file (from Source/ or full path)"
    echo "  -h, --help   Show this help"
    echo ""
    echo -e "${BOLD}EXAMPLES:${RESET}"
    echo "  ./colorise.sh                     # All files in Source/"
    echo "  ./colorise.sh -f old_film.mp4     # Single file"
    echo ""
    exit 0
}

# ── Args ───────────────────────────────────────────────────────────────────────
SINGLE_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f)         SINGLE_FILE="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *)          warn "Unknown option: $1"; shift ;;
    esac
done

# ── Checks ─────────────────────────────────────────────────────────────────────
command -v ffmpeg  &>/dev/null || { error "FFmpeg not found. brew install ffmpeg"; exit 1; }
command -v python3 &>/dev/null || { error "Python 3 not found. brew install python"; exit 1; }
[[ -f "$ENGINE" ]] || { error "Missing: $ENGINE"; exit 1; }
mkdir -p "$OUTPUT_DIR"

# ── Python virtual env + dependencies ──────────────────────────────────────────
setup_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    # Check if opencv is already installed
    if ! python3 -c "import cv2, numpy, torch, skimage, IPython" &>/dev/null 2>&1; then
        log "Installing dependencies (opencv-python, torch, scikit-image, ipython)..."
        pip install --quiet --upgrade pip
        pip install --quiet opencv-python numpy Pillow scikit-image ipython
        pip install --quiet torch torchvision
        ok "Dependencies installed"
    else
        log "Dependencies: already installed"
    fi

    # Download colorizers source from GitHub (master branch) if not present
    COLORIZERS_PKG="$SCRIPT_DIR/colorizers"
    if [[ ! -f "$COLORIZERS_PKG/eccv16.py" ]]; then
        log "Downloading colorizers source from GitHub..."
        mkdir -p "$COLORIZERS_PKG"
        BASE_URL="https://raw.githubusercontent.com/richzhang/colorization/master/colorizers"
        for f in __init__.py base_color.py eccv16.py siggraph17.py util.py; do
            curl -fsSL "${BASE_URL}/${f}" -o "${COLORIZERS_PKG}/${f}"
        done
        ok "Colorizers source ready"
    fi

    # Make local colorizers package importable
    export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
}

# ── Build file list ────────────────────────────────────────────────────────────
declare -a FILES=()
if [[ -n "$SINGLE_FILE" ]]; then
    if   [[ -f "$SINGLE_FILE" ]];             then FILES=("$SINGLE_FILE")
    elif [[ -f "$SOURCE_DIR/$SINGLE_FILE" ]]; then FILES=("$SOURCE_DIR/$SINGLE_FILE")
    else error "File not found: $SINGLE_FILE"; exit 1
    fi
else
    for ext in "${EXTENSIONS[@]}"; do
        while IFS= read -r -d '' f; do
            FILES+=("$f")
        done < <(find "$SOURCE_DIR" -maxdepth 1 -iname "*.${ext}" -print0 2>/dev/null)
    done
fi

[[ ${#FILES[@]} -eq 0 ]] && {
    warn "No video files found in: $SOURCE_DIR"
    echo -e "  Drop B&W videos into: ${CYAN}$SOURCE_DIR${RESET}"
    exit 0
}

human_size() {
    local b="$1"
    (( b >= 1073741824 )) && { printf "%.2f GB" "$(echo "scale=2;$b/1073741824"|bc)"; return; }
    (( b >= 1048576    )) && { printf "%.2f MB" "$(echo "scale=2;$b/1048576"   |bc)"; return; }
    printf "%d KB" $(( b / 1024 ))
}

# ── Main ───────────────────────────────────────────────────────────────────────
print_banner
setup_venv
source "$VENV_DIR/bin/activate"

log "Source  : $SOURCE_DIR"
log "Output  : $OUTPUT_DIR"
log "Engine  : Zhang et al. 2016 (OpenCV DNN + Caffe)"
echo ""
log "Found ${#FILES[@]} file(s)"
echo ""

PASS=0; FAIL=0; SKIP=0
START_TOTAL=$SECONDS

for INPUT in "${FILES[@]}"; do
    BASENAME=$(basename "$INPUT")
    STEM="${BASENAME%.*}"
    OUTPUT="$OUTPUT_DIR/${STEM}_colourised.mp4"
    TEMP_VIDEO="/tmp/colorise_silent_$$.mp4"

    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    log "Input   : $BASENAME"
    log "Output  : ${STEM}_colourised.mp4"

    if [[ -f "$OUTPUT" ]]; then
        warn "Already exists — skipping (delete to re-colorise)"
        (( SKIP++ )); continue
    fi

    SRC_SIZE=$(stat -f%z "$INPUT" 2>/dev/null || echo 0)
    log "Size    : $(human_size $SRC_SIZE)"

    # Check if source has audio
    HAS_AUDIO=$(ffprobe -v error -select_streams a:0 \
        -show_entries stream=codec_name -of csv=p=0 "$INPUT" 2>/dev/null || true)

    log "Step 1/2 — AI Colorisation (frame by frame)..."
    START=$SECONDS

    if PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" python3 "$ENGINE" "$INPUT" "$TEMP_VIDEO"; then
        ELAPSED=$(( SECONDS - START ))
        ok "Colorisation done in ${ELAPSED}s"

        log "Step 2/2 — Merging audio + encoding final MP4..."
        if [[ -n "$HAS_AUDIO" ]]; then
            # Merge silent colorised video with original audio
            ffmpeg -hide_banner -loglevel warning -y \
                -i "$TEMP_VIDEO" \
                -i "$INPUT" \
                -map 0:v:0 \
                -map 1:a:0 \
                -c:v libx264 -crf 18 -preset fast \
                -c:a aac -b:a 256k \
                -movflags +faststart \
                -metadata title="${STEM} [Colourised — Zhang et al. 2016]" \
                "$OUTPUT"
        else
            # No audio — just re-encode video
            ffmpeg -hide_banner -loglevel warning -y \
                -i "$TEMP_VIDEO" \
                -c:v libx264 -crf 18 -preset fast \
                -movflags +faststart \
                -metadata title="${STEM} [Colourised — Zhang et al. 2016]" \
                "$OUTPUT"
        fi

        rm -f "$TEMP_VIDEO"

        OUT_SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || echo 0)
        ok "Output  : $(human_size $OUT_SIZE)"
        ok "Saved   : $OUTPUT"
        (( PASS++ ))
    else
        error "Colorisation failed for: $BASENAME"
        rm -f "$TEMP_VIDEO"
        (( FAIL++ ))
    fi
    echo ""
done

deactivate 2>/dev/null || true

TOTAL=$(( SECONDS - START_TOTAL ))
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}SUMMARY${RESET}"
echo -e "  ${GREEN}Colourised : $PASS${RESET}"
[[ $SKIP -gt 0 ]] && echo -e "  ${YELLOW}Skipped    : $SKIP${RESET}"
[[ $FAIL -gt 0 ]] && echo -e "  ${RED}Failed     : $FAIL${RESET}"
echo    "  Time       : ${TOTAL}s"
echo -e "  Output in  : ${CYAN}$OUTPUT_DIR${RESET}"
echo ""
