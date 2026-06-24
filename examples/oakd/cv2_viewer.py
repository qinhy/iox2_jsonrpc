from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2

import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(Path(__file__).absolute().parent)))
from dai_camera import DepthAICamera


CloseCallback = Callable[[], None]
OverlayProvider = Callable[[], list[str]]
EncodedPacketConsumer = Callable[[dict[str, list[Any]]], None]


@dataclass
class Cv2Preview:
    """OpenCV preview window plus an independent full-res H264 drain thread.

    v4/v5 DepthAICamera has two independent paths:
      * read_frame(): small preview mosaic for GUI only
      * read_encoded_packets(): true full-resolution RGB/left/right H264 packets

    Do not drain the full-res packets from the GUI loop. cv2.imshow(), overlay
    drawing, and OS window events can easily run at only 5-10 Hz. If the full-res
    queues have maxSize=1, a slow GUI loop will count only 5-10 packets/s even if
    the device is producing 15 FPS and dropping old packets. This class drains the
    full-res queues in a dedicated thread.
    """

    config: Any
    camera: DepthAICamera = None
    on_close_requested: CloseCallback | None = None
    overlay_provider: OverlayProvider | None = None
    encoded_packet_consumer: EncodedPacketConsumer | None = None

    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _encoded_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _close_requested_by_viewer: bool = field(default=False, init=False, repr=False)

    _encoded_counts: dict[str, int] = field(
        default_factory=lambda: {"rgb": 0, "left": 0, "right": 0}, init=False, repr=False)
    _encoded_fps: dict[str, float] = field(
        default_factory=lambda: {"rgb": 0.0, "left": 0.0, "right": 0.0}, init=False, repr=False)
    # Estimated producer FPS from message sequence-number deltas. This helps
    # distinguish "device only produces 8 FPS" from "host receives 8 FPS because
    # the tiny queue drops frames before we drain it".
    _encoded_seq_fps: dict[str, float] = field(
        default_factory=lambda: {"rgb": 0.0, "left": 0.0, "right": 0.0}, init=False, repr=False)
    _encoded_seq_counts: dict[str, int] = field(
        default_factory=lambda: {"rgb": 0, "left": 0, "right": 0}, init=False, repr=False)
    _last_encoded_seq: dict[str, int | None] = field(
        default_factory=lambda: {"rgb": None, "left": None, "right": None}, init=False, repr=False)
    _encoded_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _fps_window_start: float = field(default_factory=time.monotonic, init=False, repr=False)

    @property
    def stop_event(self) -> threading.Event:
        return self._stop

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, camera: DepthAICamera) -> None:
        from rpc_server_camera import CameraConfig
        self.config: CameraConfig = self.config
        self.camera = camera

        if not self.config.debug_preview:
            return

        if self.is_running:
            return

        self._close_requested_by_viewer = False
        self._stop.clear()

        # Start the encoded drain first so full-res queues do not fill while the
        # GUI window is being created.
        if bool(getattr(self.config, "debug_preview_drain_encoded", True)):
            self._encoded_thread = threading.Thread(
                target=self._run_encoded_drain,
                name="serverCamEncodedDrain",
                daemon=True,
            )
            self._encoded_thread.start()

        self._thread = threading.Thread(
            target=self._run,
            name="serverCamPreview",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, join: bool = True) -> None:
        self._stop.set()

        timeout = float(getattr(self.config, "close_join_timeout_s", 2.0))
        if join:
            for thread in (self._thread, self._encoded_thread):
                if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                    thread.join(timeout=timeout)

        if not self.is_running:
            self._thread = None
        if self._encoded_thread is not None and not self._encoded_thread.is_alive():
            self._encoded_thread = None

    def destroy_window(self) -> None:
        try:
            cv2.destroyWindow(self.config.debug_window_name)
        except Exception:
            pass

    def _run_encoded_drain(self) -> None:
        """Fast loop that drains true full-resolution H264 queues.

        This loop should stay very cheap. If encoded_packet_consumer writes to disk,
        network, or RPC and becomes slow, move that work to another queue/thread.
        """
        sleep_s = float(getattr(self.config, "encoded_drain_sleep_s", 0.001))
        max_packets = int(getattr(self.config, "debug_preview_max_encoded_packets", 64))

        while not self._stop.is_set() and self.camera.is_open:
            try:
                packets = self.camera.read_encoded_packets(max_packets=max_packets)
                if packets:
                    self._update_encoded_stats(packets)
                    if self.encoded_packet_consumer is not None:
                        self.encoded_packet_consumer(packets)
                else:
                    # Tiny sleep only when queues are empty. When packets are available,
                    # immediately drain again to catch up.
                    time.sleep(sleep_s)
            except Exception:
                logging.exception("Encoded packet drain loop stopped because of an error")
                self._stop.set()
                return

    def _update_encoded_stats(self, packets: dict[str, list[Any]]) -> None:
        now = time.monotonic()
        with self._encoded_lock:
            for key in ("rgb", "left", "right"):
                msgs = packets.get(key, [])
                self._encoded_counts[key] += len(msgs)

                for msg in msgs:
                    seq = None
                    try:
                        seq = int(msg.getSequenceNum())
                    except Exception:
                        seq = None

                    if seq is None:
                        # Fall back to observed packets if this message type does not
                        # expose sequence numbers.
                        self._encoded_seq_counts[key] += 1
                        continue

                    last_seq = self._last_encoded_seq[key]
                    if last_seq is None:
                        self._encoded_seq_counts[key] += 1
                    elif seq >= last_seq:
                        # If queue_max_size=1 and the host misses frames, the sequence
                        # jump still tells us approximately how many frames the device
                        # produced during the interval.
                        self._encoded_seq_counts[key] += max(1, seq - last_seq)
                    else:
                        # Sequence wrapped/reset; count the observed message.
                        self._encoded_seq_counts[key] += 1
                    self._last_encoded_seq[key] = seq

            elapsed = now - self._fps_window_start
            if elapsed >= 1.0:
                self._encoded_fps = {
                    key: self._encoded_counts[key] / elapsed for key in self._encoded_counts
                }
                self._encoded_seq_fps = {
                    key: self._encoded_seq_counts[key] / elapsed for key in self._encoded_seq_counts
                }
                self._encoded_counts = {"rgb": 0, "left": 0, "right": 0}
                self._encoded_seq_counts = {"rgb": 0, "left": 0, "right": 0}
                self._fps_window_start = now

    def _run(self) -> None:
        window_created = False

        try:
            cv2.namedWindow(self.config.debug_window_name, cv2.WINDOW_NORMAL)
            window_created = True

            while not self._stop.is_set() and self.camera.is_open:
                # Preview only. Full-res H264 is drained by _run_encoded_drain().
                preview_timeout_s = float(getattr(self.config, "debug_preview_timeout_s", 0.01))
                frame = self.camera.read_frame(timeout_s=preview_timeout_s, stop_event=self._stop)
                if frame is None:
                    cv2.waitKey(1)
                    continue

                frames = frame if isinstance(frame, list) else [frame]

                # Show only the newest preview frame. Displaying all returned frames
                # is another source of GUI lag.
                for frame in frames[-1:]:
                    cv2.resizeWindow(
                        self.config.debug_window_name,
                        min(self.camera.width, 1280),
                        min(self.camera.height, 720),
                    )

                    frame = draw_overlay(frame, self._overlay_lines())
                    cv2.imshow(self.config.debug_window_name, frame)

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

    def _overlay_lines(self) -> list[str]:
        if self.overlay_provider is not None:
            lines = list(self.overlay_provider())
        else:
            lines = [
                f"Preview Width: {self.camera.width}",
                f"Preview Height: {self.camera.height}",
                f"Target FPS: {self.config.fps}",
            ]

        if bool(getattr(self.config, "debug_preview_show_encoded_fps", True)):
            with self._encoded_lock:
                fps = dict(self._encoded_fps)
                seq_fps = dict(self._encoded_seq_fps)
            lines.extend([
                f"Recv RGB/Left/Right pkt/s: {fps['rgb']:.1f} / {fps['left']:.1f} / {fps['right']:.1f}",
                f"Seq  RGB/Left/Right fps: {seq_fps['rgb']:.1f} / {seq_fps['left']:.1f} / {seq_fps['right']:.1f}",
            ])
        return lines


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
