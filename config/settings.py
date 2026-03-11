# ── Distraction threshold ──────────────────────────────────────────
GADGET_ALLOWED_DURATION = 1.0

# ── YOLO detection settings ────────────────────────────────────────
GADGET_CONFIDENCE_THRESHOLD = 0.45

YOLO_MODEL = "yolov8m.pt"

# ── Objects that count as distractions ────────────────────────────
GADGET_CLASSES = [
    "cell phone"
]

# ── Pilot detection settings ───────────────────────────────────────
PILOT_CONFIDENCE_THRESHOLD = 0.40
MAX_PILOTS = 2

# ── Re-log suppression ─────────────────────────────────────────────
RELOG_INTERVAL = 5.0

# ── Video output ───────────────────────────────────────────────────
OUTPUT_PATH = "outputs/gadget_detection_output.mp4"
LOG_PATH    = "logs/distraction_log.txt"

# ── Display ────────────────────────────────────────────────────────
WINDOW_NAME   = "Loco Pilot - Gadget Detection"
DISPLAY_SCALE = 1.0