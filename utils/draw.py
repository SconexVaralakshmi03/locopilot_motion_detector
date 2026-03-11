# utils/draw.py
# ──────────────────────────────────────────────────────────────────
# All OpenCV drawing helpers used to annotate the video frames
# ──────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import List, Tuple

import cv2
import numpy as np

# ── Colour palette (BGR) ───────────────────────────────────────────
CLR_PILOT_1   = (0,   200, 255)   # cyan
CLR_PILOT_2   = (255, 165,   0)   # amber
CLR_ALERT     = (0,     0, 255)   # red   – distraction active
CLR_SAFE      = (0,   220,   0)   # green – no distraction
CLR_GADGET    = (0,    50, 255)   # bright red for gadget box
CLR_TEXT_BG   = (20,   20,  20)   # near-black background for text


def put_text(
    frame:  np.ndarray,
    text:   str,
    pos:    Tuple[int, int],
    colour: Tuple[int, int, int] = (220, 220, 220),
    scale:  float = 0.52,
    thick:  int   = 1,
) -> None:
    """Draw text with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(frame,
                  (x - 2,      y - th - 3),
                  (x + tw + 2, y + bl + 1),
                  CLR_TEXT_BG, cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, scale, colour, thick, cv2.LINE_AA)


def draw_pilot_box(
    frame:       np.ndarray,
    bbox:        Tuple[int, int, int, int],
    pilot_id:    int,
    distracted:  bool,
    gadgets:     List[str],
) -> None:
    """
    Draw the pilot bounding box + status label.
    Box turns RED when a gadget is detected, GREEN when clean.
    """
    x1, y1, x2, y2 = bbox
    base_colour = CLR_PILOT_1 if pilot_id == 1 else CLR_PILOT_2
    box_colour  = CLR_ALERT if distracted else base_colour

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)

    # ── Pilot ID label above the box ──────────────────────────────
    status = "DISTRACTED" if distracted else "OK"
    label  = f"Pilot {pilot_id}  [{status}]"
    put_text(frame, label, (x1, y1 - 8), box_colour, scale=0.55)

    # ── Gadget names stacked below the box ────────────────────────
    for i, g in enumerate(gadgets):
        put_text(
            frame,
            f"  Gadget: {g}",
            (x1, y2 + 18 + i * 18),
            CLR_ALERT,
            scale=0.46,
        )


def draw_gadget_box(
    frame: np.ndarray,
    bbox:  Tuple[int, int, int, int],
    label: str,
    conf:  float,
) -> None:
    """Draw bounding box around the detected gadget object."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_GADGET, 2)
    put_text(
        frame,
        f"{label} {conf:.0%}",
        (x1, y1 - 6),
        CLR_GADGET,
        scale=0.45,
    )


def draw_hud(
    frame:      np.ndarray,
    video_time: float,
    frame_no:   int,
    pilot_count: int,
) -> None:
    """Top-left heads-up display: time, frame, pilot count."""
    hh = int(video_time) // 3600
    mm = (int(video_time) % 3600) // 60
    ss = int(video_time) % 60
    put_text(
        frame,
        f"Time {hh:02d}:{mm:02d}:{ss:02d}  |  Frame {frame_no}  |  Pilots detected: {pilot_count}",
        (10, 22),
        (200, 200, 200),
        scale=0.50,
    )


def draw_alert_banner(frame: np.ndarray, pilot_id: int, gadget: str) -> None:
    """Full-width red banner at the bottom when distraction is critical."""
    h, w = frame.shape[:2]
    banner_h = 36
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 180), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    msg = f"  !! CRITICAL - Pilot {pilot_id} using {gadget} !!"
    put_text(frame, msg, (8, h - 10), (255, 255, 255), scale=0.58, thick=1)