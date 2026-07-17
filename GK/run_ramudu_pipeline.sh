#!/usr/bin/env bash
# Pipeline: Ramudu Bheemudu — SIGGRAPH17 colourisation + VolQa delogo + audio merge
# Video: 668×480, 25fps, 5728 frames (~3m49s)
# Watermark: VolQa logo, top-right corner (x=545,y=0,w=125,h=40)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
ENGINE="$SCRIPT_DIR/colorise_engine_v2.py"

INPUT="$SCRIPT_DIR/Source/Ramudu Bheemudu Movie Songs ｜ Telisindi Le Telisindi Le Video Song ｜ Sr NTR ｜ Suresh Productions.mp4"
TEMP_VIDEO="/tmp/ramudu_silent_$$.mp4"
OUTPUT="$SCRIPT_DIR/Output/Ramudu_Bheemudu_Telisindi_Le_Colourised.mp4"
PID_FILE="/tmp/ramudu_colorise.pid"
LOG_FILE="/tmp/ramudu_colorise_$$.log"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; BOLD='\033[1m'; RESET='\033[0m'
log() { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()  { echo -e "${GREEN}[DONE]${RESET}  $*"; }
err() { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

mkdir -p "$SCRIPT_DIR/Output"

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║    Ramudu Bheemudu · SIGGRAPH17 Colourisation           ║"
echo "  ║    VolQa watermark removal · Saturation boost           ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Activate venv ──────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

log "Step 1/2 — AI Colourisation (ECCV16 + CLAHE + SAT ×1.25)..."
log "Input   : $(basename "$INPUT")"
START=$SECONDS

if PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" python3 "$ENGINE" "$INPUT" "$TEMP_VIDEO" 2>&1 | tee "$LOG_FILE"; then
    ELAPSED=$(( SECONDS - START ))
    ok "Colourisation complete in ${ELAPSED}s"
else
    err "Colourisation failed — check $LOG_FILE"
    rm -f "$TEMP_VIDEO"
    exit 1
fi

log "Step 2/2 — Delogo (VolQa top-right) + audio merge + saturation polish..."

# delogo: x=545,y=0,w=125,h=40 removes the VolQa logo from top-right corner
# eq=saturation=1.15: additional FFmpeg-level saturation lift for warmth
ffmpeg -hide_banner -loglevel warning -y \
    -i "$TEMP_VIDEO" \
    -i "$INPUT" \
    -filter_complex "[0:v]delogo=x=545:y=0:w=125:h=40,eq=saturation=1.15[v]" \
    -map "[v]" \
    -map 1:a:0 \
    -c:v libx264 -crf 16 -preset slow \
    -c:a aac -b:a 256k \
    -movflags +faststart \
    -metadata title="Telisindi Le - Ramudu Bheemudu [Colourised · SIGGRAPH17]" \
    "$OUTPUT"

rm -f "$TEMP_VIDEO" "$LOG_FILE"

OUT_SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || echo 0)
ok "Output  : $OUTPUT"
ok "Size    : $(echo "scale=1; $OUT_SIZE/1048576" | bc) MB"
echo ""
