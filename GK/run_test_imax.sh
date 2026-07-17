#!/usr/bin/env bash
# =============================================================================
#  GUNDAMMA KATHA — IMAX Colourisation Test (first 5 minutes)
#
#  Pipeline:
#    1. FFmpeg  — extract 5 min + crop black bars (watermark handled in Python)
#    2. Python  — inpaint watermark · SIGGRAPH17 AI colour · IMAX 1544×1080
#    3. FFmpeg  — merge original audio + final H.264 encode
#
#  Output: Output/Gundamma_Katha_TEST_5min_v4_IMAX143.mp4
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOURCE="$SCRIPT_DIR/Source/Gundamma Katha Telugu Full Length Movie ｜｜ గుండమ్మ కథ సినిమా ｜｜ SVR, NTR, ANR, Savitri ,Jamuna.mp4"
OUTPUT_DIR="$SCRIPT_DIR/Output"
ENGINE="$SCRIPT_DIR/colorise_imax_v4.py"
VENV="$SCRIPT_DIR/.venv"

TEMP_PRE="/tmp/gundamma_pre_$$.mp4"
TEMP_COL="/tmp/gundamma_col_$$.mp4"
OUTPUT="$OUTPUT_DIR/Gundamma_Katha_TEST_5min_v4_IMAX190.mp4"

DURATION=300   # 5 minutes

# Black bar crop only — watermark removal is done in Python (cv2.inpaint)
CROP="crop=1280:560:0:80"

# ── Colours ───────────────────────────────────────────────────────────────────
G='\033[0;32m'; C='\033[0;36m'; Y='\033[1;33m'
R='\033[0;31m'; B='\033[1m';    RS='\033[0m'

log()  { echo -e "${C}[INFO]${RS}  $*"; }
ok()   { echo -e "${G}[DONE]${RS}  $*"; }
warn() { echo -e "${Y}[WARN]${RS}  $*"; }
err()  { echo -e "${R}[ERROR]${RS} $*" >&2; }

cleanup() { rm -f "$TEMP_PRE" "$TEMP_COL"; }
trap cleanup EXIT

# ── Sanity checks ─────────────────────────────────────────────────────────────
[[ -f "$SOURCE"  ]] || { err "Source not found: $SOURCE";  exit 1; }
[[ -f "$ENGINE"  ]] || { err "Engine not found: $ENGINE";  exit 1; }
[[ -d "$VENV"    ]] || { err "Venv not found:   $VENV";    exit 1; }
command -v ffmpeg &>/dev/null || { err "ffmpeg not found"; exit 1; }
mkdir -p "$OUTPUT_DIR"

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${B}  ╔══════════════════════════════════════════════════════════════╗${RS}"
echo -e "${B}  ║      GUNDAMMA KATHA — IMAX COLOURISATION  [TEST 5 min]      ║${RS}"
echo -e "${B}  ║  Python inpaint WM · SIGGRAPH17 · LAB 2× · IMAX 1544×1080  ║${RS}"
echo -e "${B}  ╚══════════════════════════════════════════════════════════════╝${RS}"
echo ""
log "Source  : $(basename "$SOURCE")"
log "Output  : $OUTPUT"
log "Steps   : crop bars → [Python: inpaint WM + SIGGRAPH17 + vivid colours + IMAX] → merge audio"
echo ""

# ── Step 1: Extract 5 min + crop black bars ───────────────────────────────────
log "STEP 1/3 — Extract 5 min · crop black bars (watermark handled in Python)"
log "  crop   : 1280×560 (removes 80px top+bottom bars)"

ffmpeg -hide_banner -loglevel warning -y \
    -i "$SOURCE" \
    -t "$DURATION" \
    -vf "$CROP" \
    -c:v libx264 -preset ultrafast -crf 18 \
    -an \
    "$TEMP_PRE"

PRE_SIZE=$(du -h "$TEMP_PRE" | cut -f1)
ok "Preprocessed clip ready  (${PRE_SIZE})  →  $TEMP_PRE"
echo ""

# ── Step 2: AI Colorisation ───────────────────────────────────────────────────
log "STEP 2/3 — AI Colourisation (v4 engine)"
log "  WM     : cv2.inpaint TELEA on Shalimar mask (before AI sees frame)"
log "  Model  : SIGGRAPH17  (vivid, scene-aware — major upgrade over ECCV16)"
log "  Colour : LAB AB ×2.0  +  HSV Vibrance  (no more brown cast)"
log "  Output : 1544×1080  IMAX 1.43:1  (no black bars)"
echo ""

source "$VENV/bin/activate"
PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" python3 "$ENGINE" "$TEMP_PRE" "$TEMP_COL"
deactivate 2>/dev/null || true

echo ""
COL_SIZE=$(du -h "$TEMP_COL" | cut -f1)
ok "Colourisation complete  (${COL_SIZE})  →  $TEMP_COL"
echo ""

# ── Step 3: Merge audio + final encode ───────────────────────────────────────
log "STEP 3/3 — Merge original audio + final H.264 encode"

ffmpeg -hide_banner -loglevel warning -y \
    -i "$TEMP_COL" \
    -ss 0 -t "$DURATION" -i "$SOURCE" \
    -map 0:v:0 \
    -map 1:a:0 \
    -c:v libx264 -crf 16 -preset medium \
    -c:a aac -b:a 256k \
    -movflags +faststart \
    -metadata title="Gundamma Katha [Colourised IMAX 1.43:1 v4 — SIGGRAPH17]" \
    "$OUTPUT"

# ── Summary ───────────────────────────────────────────────────────────────────
OUT_SIZE=$(du -h "$OUTPUT" | cut -f1)
echo ""
echo -e "${B}  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RS}"
ok "TEST COMPLETE"
echo -e "  ${C}File  :${RS} $OUTPUT"
echo -e "  ${C}Size  :${RS} $OUT_SIZE"
echo -e "  ${C}Res   :${RS} 1544×1080  |  IMAX 1.43:1  |  no black bars"
echo -e "  ${C}Clip  :${RS} First 5 minutes"
echo -e "  ${C}Model :${RS} SIGGRAPH17 + LAB×2 + Vibrance  |  WM: cv2.inpaint"
echo ""
echo -e "  Once happy, run the full movie with:  ${Y}./run_full_imax.sh${RS}"
echo ""
