# detector/gadget_detector.py
# ──────────────────────────────────────────────────────────────────
# FIXES APPLIED
# ─────────────────────────────────────────────────────────────────
# 1. EXPANDED match zone  → uses full pilot box + proximity margin
#    so gadgets near hands (even outside the box edge) are caught
# 2. REMOVED motion filter → static phone held still was being skipped
# 3. REMOVED desk_limit   → phone in lap / low hand position now caught
# 4. LOWER confidence     → nearby/partial objects detected more reliably
# 5. EXPANDED gadget list → added handbag, umbrella, food items
# 6. PROXIMITY check      → if gadget box TOUCHES pilot box it counts
# 7. OVERLAP RATIO check  → even small overlap (10%) triggers match
# ──────────────────────────────────────────────────────────────────

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from config.settings import (
    YOLO_MODEL,
    GADGET_CLASSES,
    GADGET_CONFIDENCE_THRESHOLD,
    PILOT_CONFIDENCE_THRESHOLD,
    MAX_PILOTS,
    GADGET_ALLOWED_DURATION,
    RELOG_INTERVAL,
)

# ── YOLO lazy loader ───────────────────────────────────────────────
_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(YOLO_MODEL)
    return _model


# ──────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────

@dataclass
class GadgetHit:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class PilotResult:
    pilot_id:    int
    bbox:        Tuple[int, int, int, int]
    gadgets:     List[GadgetHit] = field(default_factory=list)
    distracted:  bool  = False
    timer_value: float = 0.0


# ──────────────────────────────────────────────────────────────────
# PER-PILOT TIMER
# ──────────────────────────────────────────────────────────────────

@dataclass
class _PilotTimer:
    pilot_id:     int
    start_time:   Optional[float] = None
    last_logged:  Optional[float] = None

    def activate(self):
        if self.start_time is None:
            self.start_time = time.monotonic()

    def reset(self):
        self.start_time  = None
        self.last_logged = None

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time

    def should_log(self, video_time: float) -> bool:
        if self.elapsed() < GADGET_ALLOWED_DURATION:
            return False
        if self.last_logged is None:
            return True
        return (video_time - self.last_logged) > RELOG_INTERVAL

    def mark_logged(self, video_time: float):
        self.last_logged = video_time


# ──────────────────────────────────────────────────────────────────
# MAIN DETECTOR
# ──────────────────────────────────────────────────────────────────

class GadgetDetector:

    # ── How far OUTSIDE the pilot box a gadget can still be matched ─
    # e.g. 0.15 means 15% of box width/height counts as "near enough"
    PROXIMITY_MARGIN = 0.18

    # ── Minimum fraction of gadget box that must overlap pilot box ──
    # 0.0 means just touching is enough
    MIN_OVERLAP_RATIO = 0.0

    def __init__(self):
        self.timers: Dict[int, _PilotTimer] = {
            1: _PilotTimer(1),
            2: _PilotTimer(2),
        }
        self.last_gadget_hits: List[GadgetHit] = []
        self._prev_pilot_boxes: Dict[int, Tuple[int,int,int,int]] = {}

    # ──────────────────────────────────────────────────────────────
    # PUBLIC: process one frame
    # ──────────────────────────────────────────────────────────────

    def process(
        self,
        frame:      np.ndarray,
        video_time: float,
    ) -> Tuple[List[PilotResult], List[Tuple[int, str]]]:

        # FIX: enhance ONLY for dark/low-contrast frames, not always
        enhanced = self._smart_enhance(frame)

        pilot_boxes, gadgets = self._run_yolo(enhanced)
        pilot_boxes          = self._assign_pilots(pilot_boxes, frame)

        results:    List[PilotResult]        = []
        log_events: List[Tuple[int, str]]    = []

        active_ids = {pid for pid, _ in pilot_boxes}

        for pid, pbox in pilot_boxes:

            matched = self._match_gadgets(pbox, gadgets, frame.shape)
            timer   = self.timers[pid]

            if matched:
                timer.activate()
            else:
                timer.reset()

            distracted = timer.elapsed() >= GADGET_ALLOWED_DURATION

            if distracted and timer.should_log(video_time):
                best_gadget = max(matched, key=lambda g: g.confidence)
                log_events.append((pid, best_gadget.class_name))
                timer.mark_logged(video_time)

            results.append(PilotResult(
                pilot_id    = pid,
                bbox        = pbox,
                gadgets     = matched,
                distracted  = distracted,
                timer_value = timer.elapsed(),
            ))

        # Reset timers for pilots that disappeared
        for pid in [1, 2]:
            if pid not in active_ids:
                self.timers[pid].reset()

        self.last_gadget_hits = gadgets
        return results, log_events

    # ──────────────────────────────────────────────────────────────
    # SMART ENHANCE  (only boost if frame is dark)
    # ──────────────────────────────────────────────────────────────

    def _smart_enhance(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        # Only apply CLAHE if the frame is genuinely dark
        if mean_brightness < 100:
            clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return frame   # colour frame is fine as-is

    # ──────────────────────────────────────────────────────────────
    # YOLO DETECTION
    # ──────────────────────────────────────────────────────────────

    def _run_yolo(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[int,int,int,int]], List[GadgetHit]]:

        model   = _get_model()
        results = model(frame, verbose=False)[0]

        persons: List[Tuple[Tuple[int,int,int,int], float]] = []
        gadgets: List[GadgetHit] = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            name   = model.names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)

            if name == "person" and conf > PILOT_CONFIDENCE_THRESHOLD:
                persons.append((bbox, conf))

            elif name in GADGET_CLASSES and conf > GADGET_CONFIDENCE_THRESHOLD:
                gadgets.append(GadgetHit(
                    class_name = name,
                    confidence = conf,
                    bbox       = bbox,
                ))

        # Keep the 2 largest persons (the pilots)
        persons.sort(
            key=lambda p: (p[0][2]-p[0][0]) * (p[0][3]-p[0][1]),
            reverse=True,
        )
        return [p[0] for p in persons[:MAX_PILOTS]], gadgets

    # ──────────────────────────────────────────────────────────────
    # ASSIGN STABLE PILOT IDs  (left=1, right=2 + IoU tracking)
    # ──────────────────────────────────────────────────────────────

    def _assign_pilots(
        self,
        boxes:       List[Tuple[int,int,int,int]],
        frame:       np.ndarray,
    ) -> List[Tuple[int, Tuple[int,int,int,int]]]:

        if not boxes:
            return []

        # Sort by x-centre so left pilot is always Pilot 1
        boxes = sorted(boxes, key=lambda b: (b[0]+b[2])//2)

        if not self._prev_pilot_boxes:
            assigned = [(i+1, b) for i, b in enumerate(boxes[:2])]
            self._prev_pilot_boxes = {pid: b for pid, b in assigned}
            return assigned

        # IoU-based matching for stable IDs across frames
        assigned:  List[Tuple[int, Tuple]] = []
        used_pids: set = set()

        for box in boxes:
            best_pid  = None
            best_iou  = -1.0
            for pid, prev in self._prev_pilot_boxes.items():
                if pid in used_pids:
                    continue
                score = _iou(box, prev)
                if score > best_iou:
                    best_iou = score
                    best_pid = pid

            if best_pid and best_iou > 0.05:
                used_pids.add(best_pid)
                assigned.append((best_pid, box))
            else:
                free = next((p for p in [1, 2] if p not in used_pids), None)
                if free:
                    used_pids.add(free)
                    assigned.append((free, box))

        # Re-sort so Pilot 1 stays left
        assigned.sort(key=lambda x: (x[1][0]+x[1][2])//2)
        assigned = [(i+1, b) for i, (_, b) in enumerate(assigned)]
        self._prev_pilot_boxes = {pid: b for pid, b in assigned}
        return assigned

    # ──────────────────────────────────────────────────────────────
    # GADGET → PILOT MATCHING   (the main fix)
    # ──────────────────────────────────────────────────────────────

    def _match_gadgets(
        self,
        pilot_box:    Tuple[int, int, int, int],
        gadgets:      List[GadgetHit],
        frame_shape:  Tuple[int, ...],
    ) -> List[GadgetHit]:
        """
        A gadget is considered held by (or near) a pilot if ANY of
        these conditions are true:

        ① OVERLAP   – gadget box overlaps the pilot box
        ② PROXIMITY – gadget box is within PROXIMITY_MARGIN of the
                       pilot box edges (catches phone held slightly
                       outside the detected person region)
        ③ HAND ZONE – gadget centre falls in the lower-half of the
                       pilot box (where hands naturally rest on controls)
                       even if the gadget box itself is outside

        FIX: Removed motion filter — a phone held still is still a phone.
        FIX: Removed desk_limit   — phone in lap is now caught.
        """
        px1, py1, px2, py2 = pilot_box
        pw = px2 - px1
        ph = py2 - py1

        # Expanded box with margin
        margin_x = int(pw * self.PROXIMITY_MARGIN)
        margin_y = int(ph * self.PROXIMITY_MARGIN)
        ex1 = px1 - margin_x
        ey1 = py1 - margin_y
        ex2 = px2 + margin_x
        ey2 = py2 + margin_y

        # Hand zone = lower 60% of pilot box (where hands are)
        hand_zone_top = py1 + int(ph * 0.40)

        matched: List[GadgetHit] = []

        for g in gadgets:
            gx1, gy1, gx2, gy2 = g.bbox
            gcx = (gx1 + gx2) // 2
            gcy = (gy1 + gy2) // 2

            # ① Check overlap with expanded box
            overlap = _intersection_area(
                (ex1, ey1, ex2, ey2),
                g.bbox,
            )
            if overlap > 0:
                matched.append(g)
                continue

            # ② Check if gadget centre is in hand zone
            in_hand_zone = (px1 <= gcx <= px2) and (hand_zone_top <= gcy <= ey2)
            if in_hand_zone:
                matched.append(g)
                continue

            # ③ Check if gadget is very close (within half margin) of pilot box
            close_x = ex1 <= gcx <= ex2
            close_y = ey1 <= gcy <= ey2
            if close_x and close_y:
                matched.append(g)

        return matched


# ──────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────

def _intersection_area(
    a: Tuple[int,int,int,int],
    b: Tuple[int,int,int,int],
) -> int:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _iou(
    a: Tuple[int,int,int,int],
    b: Tuple[int,int,int,int],
) -> float:
    inter = _intersection_area(a, b)
    if inter == 0:
        return 0.0
    aA = (a[2]-a[0]) * (a[3]-a[1])
    aB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / float(aA + aB - inter)