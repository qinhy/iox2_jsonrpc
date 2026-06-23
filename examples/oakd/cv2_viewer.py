from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List
import cv2

import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(Path(__file__).absolute().parent)))
from dai_camera import DepthAICamera


CloseCallback = Callable[[], None]
OverlayProvider = Callable[[], list[str]]


@dataclass
class Cv2Preview:
    """OpenCV preview window and preview thread management."""

    config: Any
    camera: DepthAICamera
    on_close_requested: CloseCallback | None = None
    overlay_provider: OverlayProvider | None = None

    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _close_requested_by_viewer: bool = field(default=False, init=False, repr=False)

    @property
    def stop_event(self) -> threading.Event:
        return self._stop

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        from rpc_server import CameraConfig
        self.config:CameraConfig = self.config
        if not self.config.debug_preview:
            return

        if self.is_running:
            return

        self._close_requested_by_viewer = False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="serverCamPreview",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, join: bool = True) -> None:
        self._stop.set()

        if (
            join
            and self._thread is not None
            and self._thread.is_alive()
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout=self.config.close_join_timeout_s)

        if not self.is_running:
            self._thread = None

    def destroy_window(self) -> None:
        try:
            cv2.destroyWindow(self.config.debug_window_name)
        except Exception:
            pass

    def _run(self) -> None:
        window_created = False

        try:
            cv2.namedWindow(self.config.debug_window_name, cv2.WINDOW_NORMAL)
            window_created = True

            while not self._stop.is_set() and self.camera.is_open:
                frame = self.camera.read_frame(timeout_s=0.5, stop_event=self._stop)
                if frame is None:
                    continue

                if type(frame) is list:
                    frames = frame
                else:
                    frames = [frame]

                for frame in frames:
                    cv2.resizeWindow(
                        self.config.debug_window_name,
                        min(self.camera.width, 1280),
                        min(self.camera.height, 720),
                    )

                    text_lines = (
                        self.overlay_provider()
                        if self.overlay_provider is not None
                        else [
                            f"Width: {self.camera.width}",
                            f"Height: {self.camera.height}",
                            f"FPS: {self.config.fps}",
                        ]
                    )

                    frame = draw_overlay(frame, text_lines)
                    cv2.imshow(self.config.debug_window_name, frame)

                    # Required for OpenCV GUI events. q or Esc closes the camera.
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        logging.info("Preview close requested by keyboard")
                        self._close_requested_by_viewer = True
                        self._stop.set()
                        break

        except Exception:
            logging.exception("Camera preview loop stopped because of an error")
            self._close_requested_by_viewer = True
            self._stop.set()

        finally:
            if window_created:
                self.destroy_window()

            if self._close_requested_by_viewer and self.on_close_requested is not None:
                self.on_close_requested()


def draw_overlay(frame, text_lines: list[str]):
    """Draw multiple overlay text lines with dynamic font scale."""

    if not text_lines:
        return frame

    frame_height, frame_width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    longest_line = max(text_lines, key=len)
    target_text_width = int(frame_width * 0.45)
    base_scale = 1.0
    base_thickness = 2

    (base_text_width, _), _ = cv2.getTextSize(
        longest_line,
        font,
        base_scale,
        base_thickness,
    )

    font_scale = target_text_width / max(base_text_width, 1)
    font_scale = max(0.4, min(font_scale, 2.0))
    thickness = max(1, int(font_scale * 2))

    (_, text_height), _ = cv2.getTextSize(
        longest_line,
        font,
        font_scale,
        thickness,
    )

    x = int(frame_width * 0.03)
    y = int(frame_height * 0.08)
    line_gap = int(text_height * 1.5)

    for index, line in enumerate(text_lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * line_gap),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

    return frame
