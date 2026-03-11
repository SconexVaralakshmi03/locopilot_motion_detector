#!/usr/bin/env python3
# main.py
# ──────────────────────────────────────────────────────────────────
# Loco Pilot Distraction Detection – PHASE 1: Gadget / Mobile Usage
#
# Run on webcam:
#   python main.py
#
# Run on a video file:
#   python main.py --source data/your_video.mp4
#
# Run headless (no display window, e.g. on a server):
#   python main.py --source data/your_video.mp4 --no-display
#
# Skip saving output video:
#   python main.py --no-save
# ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Optional

import cv2
import numpy as np

# ── project imports ────────────────────────────────────────────────
from config.settings  import (
    OUTPUT_PATH, WINDOW_NAME, DISPLAY_SCALE
)
from utils.logger     import setup_logger, log_distraction
from utils.draw       import (
    draw_pilot_box, draw_gadget_box, draw_hud, draw_alert_banner
)
from detector.gadget_detector import GadgetDetector


# ──────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────

class GadgetDetectionPipeline:
    """
    Full processing loop for gadget / mobile usage detection.

    Steps per frame
    ───────────────
    1. Read frame from source
    2. Run GadgetDetector  (YOLO person + object detection)
    3. Match gadgets → pilots
    4. Apply time threshold (GADGET_ALLOWED_DURATION)
    5. Log critical events
    6. Annotate frame
    7. Write to output video (optional)
    8. Display (optional)
    """

    def __init__(
        self,
        source:   str | int,
        save:     bool = True,
        display:  bool = True,
    ) -> None:
        self.source  = source
        self.save    = save
        self.display = display

        self.logger   = setup_logger()
        self.detector = GadgetDetector()
        self._writer: Optional[cv2.VideoWriter] = None

    # ── Entry point ───────────────────────────────────────────────

    def run(self) -> None:
        # ── Open video source ─────────────────────────────────────
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.logger.error(f"Cannot open source: {self.source!r}")
            sys.exit(1)

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._print_banner(fps, width, height, total)

        # ── Optional video writer ──────────────────────────────────
        if self.save:
            os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
            fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                OUTPUT_PATH, fourcc, fps, (width, height)
            )

        frame_no = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_no  += 1
                video_time = frame_no / fps

                # ── Core processing ────────────────────────────────
                annotated = self._process_frame(frame, video_time, frame_no)

                # ── Write to output video ──────────────────────────
                if self._writer:
                    self._writer.write(annotated)

                # ── Display ────────────────────────────────────────
                if self.display:
                    show = annotated
                    if DISPLAY_SCALE != 1.0:
                        nw = int(width  * DISPLAY_SCALE)
                        nh = int(height * DISPLAY_SCALE)
                        show = cv2.resize(annotated, (nw, nh))
                    cv2.imshow(WINDOW_NAME, show)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:   # q or ESC to quit
                        self.logger.info("Quit by user.")
                        break

        except KeyboardInterrupt:
            self.logger.info("\nInterrupted by user.")
        except Exception:
            self.logger.error("Unexpected error:\n" + traceback.format_exc())
        finally:
            cap.release()
            if self._writer:
                self._writer.release()
            if self.display:
                cv2.destroyAllWindows()
            self._print_summary(frame_no)

    # ── Per-frame logic ───────────────────────────────────────────

    def _process_frame(
        self,
        frame:      np.ndarray,
        video_time: float,
        frame_no:   int,
    ) -> np.ndarray:
        annotated = frame.copy()

        # ── Run detector ───────────────────────────────────────────
        results, log_events = [], []
        try:
            results, log_events = self.detector.process(frame, video_time)
        except Exception as exc:
            # Never crash the pipeline on a single bad frame
            self.logger.debug(f"Detector error frame {frame_no}: {exc}")

        # ── Log critical events ────────────────────────────────────
        for pilot_id, gadget_name in log_events:
            log_distraction(
                
    self.logger,
    video_time,
    pilot_id,
    "Gadget Usage Detected",
    gadget=gadget_name,
    severity="CRITICAL",

            )

        # ── Draw all gadget bounding boxes ─────────────────────────
        for g in self.detector.last_gadget_hits:
            draw_gadget_box(annotated, g.bbox, g.class_name, g.confidence)

        # ── Draw per-pilot boxes and status ────────────────────────
        any_distracted = False
        last_distracted_pilot  = None
        last_distracted_gadget = ""

        for r in results:
            gadget_names = [g.class_name for g in r.gadgets]
            draw_pilot_box(
                annotated,
                r.bbox,
                r.pilot_id,
                r.distracted,
                gadget_names,
            )
            if r.distracted:
                any_distracted = True
                last_distracted_pilot  = r.pilot_id
                last_distracted_gadget = gadget_names[0] if gadget_names else "gadget"

        # ── Alert banner when any pilot is distracted ──────────────
        if any_distracted and last_distracted_pilot is not None:
            draw_alert_banner(annotated, last_distracted_pilot, last_distracted_gadget)

        # ── HUD overlay ────────────────────────────────────────────
        draw_hud(annotated, video_time, frame_no, len(results))

        return annotated

    # ── Helpers ───────────────────────────────────────────────────

    def _print_banner(
        self, fps: float, w: int, h: int, total: int
    ) -> None:
        self.logger.info(
            f"\n{'='*60}\n"
            f"  LOCO PILOT DISTRACTION DETECTION\n"
            f"  Phase 1 : Gadget / Mobile Usage\n"
            f"  Source  : {self.source}\n"
            f"  Video   : {w}×{h}  @  {fps:.1f} fps"
            + (f"  ({total} frames)" if total > 0 else "") + "\n"
            f"  Output  : {OUTPUT_PATH if self.save else 'disabled'}\n"
            f"  Log     : logs/distraction_log.txt\n"
            f"  Press Q / ESC to stop early\n"
            f"{'='*60}\n"
        )

    def _print_summary(self, frame_no: int) -> None:
        self.logger.info(
            f"\n{'='*60}\n"
            f"  Processing complete\n"
            f"  Frames processed : {frame_no}\n"
            f"  Output video     : {OUTPUT_PATH if self.save else 'N/A'}\n"
            f"  Distraction log  : logs/distraction_log.txt\n"
            f"{'='*60}\n"
        )


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 – Loco Pilot Gadget / Mobile Usage Detection"
    )
    p.add_argument(
        "--source",
        default=0,
        help="Video file path or camera index (default: 0 = webcam)",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Disable real-time preview window",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write output video",
    )
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    source = args.source

    # Convert "0", "1" etc. to integer camera index
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    pipeline = GadgetDetectionPipeline(
        source  = source,
        save    = not args.no_save,
        display = not args.no_display,
    )
    pipeline.run()