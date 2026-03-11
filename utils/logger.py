# utils/logger.py

import logging
import os
from config.settings import LOG_PATH


def setup_logger(name: str = "gadget_monitor") -> logging.Logger:
    """
    Returns a logger writing to BOTH console and log file.

    BUG FIXED 1: os.path.dirname("logs/file.txt") can return ""
                 when the script is run from outside the project dir.
                 Using os.path.abspath() ensures the full path is resolved
                 so makedirs always succeeds.
    """
    # Always resolve to absolute path before creating directories
    abs_log_path = os.path.abspath(LOG_PATH)
    log_dir      = os.path.dirname(abs_log_path)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # BUG FIXED 2: if logger.handlers check was blocking re-init after
    # a crash/restart in the same process — clear stale handlers first.
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    # ── File handler — writes every event permanently ─────────────
    fh = logging.FileHandler(abs_log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(message)s"))

    # ── Console handler ────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    # BUG FIXED 3: propagate=False stops Python's root logger from
    # swallowing or double-printing the log messages, which was
    # causing file writes to appear to be missing.
    logger.propagate = False

    return logger


def video_timestamp(seconds: float) -> str:
    """Convert float seconds → [HH:MM:SS]"""
    t  = int(seconds)
    hh = t // 3600
    mm = (t % 3600) // 60
    ss = t % 60
    return f"[{hh:02d}:{mm:02d}:{ss:02d}]"


def log_distraction(
    logger:     logging.Logger,
    video_time: float,
    pilot_id:   int,
    event:      str,
    gadget:     str = "",
    severity:   str = "CRITICAL",
) -> None:
    """
    Write one distraction event line to console + file.

    Example output:
        [00:02:10] Pilot 2 - Gadget Usage Detected (cell phone)  [CRITICAL]
    """
    ts      = video_timestamp(video_time)
    detail  = f" ({gadget})" if gadget else ""
    msg     = f"{ts} Pilot {pilot_id} - {event}{detail}  [{severity}]"

    logger.info(msg)

    # Force flush so the line appears in the file immediately,
    # even if the process is killed mid-run.
    for handler in logger.handlers:
        handler.flush()