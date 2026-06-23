from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import cv2
import depthai as dai
import numpy as np

logger = logging.getLogger(__name__)
REQUIRED_CONFIG_FIELDS = ("width", "height", "fps", "queue_max_size", "frame_poll_sleep_s", "device_id")


def _private(default: Any = None, *, factory: Any | None = None):
    kwargs = {"init": False, "repr": False}
    kwargs["default_factory" if factory is not None else "default"] = factory or default
    return field(**kwargs)


def _is_stopped(stop_event: threading.Event | None) -> bool:
    return stop_event is not None and stop_event.is_set()


def _drain_queue(queue: Any | None, *, max_packets: int = 128) -> list[Any]:
    """Return every packet currently waiting in a DepthAI queue."""
    if queue is None:
        return []
    packets: list[Any] = []
    for _ in range(int(max_packets)):
        packet = queue.tryGet()
        if packet is None:
            break
        packets.append(packet)
    return packets


@runtime_checkable
class CameraConfigProtocol(Protocol):
    """Minimal config contract used by DepthAICamera."""

    width: int
    height: int
    fps: int | float
    queue_max_size: int
    frame_poll_sleep_s: float
    device_id: str | None


@dataclass
class DepthAICamera:
    """DepthAI adapter optimized for full-res encoded streams plus low-latency preview.

    The previous implementation decoded full 4000x3000 RGB H264 plus stereo H264 and
    created a huge NumPy mosaic in read_frame().  That path is CPU/memory-bound and
    normally cannot hold 15 FPS in Python.

    This version starts a pipeline with:
      * full-resolution RGB/left/right H264 queues, available via read_encoded_packets()
      * three small preview queues composed on the host by read_frame()
    """

    config: CameraConfigProtocol
    width: int = _private()
    height: int = _private()

    _device: dai.Device | None = _private()
    _pipeline: dai.Pipeline | None = _private()
    _rgb_q: dai.MessageQueue | None = _private()          # preview RGB queue in realtime mode
    _left_q: dai.MessageQueue | None = _private()         # preview LEFT queue in realtime mode / legacy decoded mode
    _right_q: dai.MessageQueue | None = _private()        # preview RIGHT queue in realtime mode / legacy decoded mode
    _full_rgb_q: dai.MessageQueue | None = _private()     # full-res encoded H264
    _full_left_q: dai.MessageQueue | None = _private()    # full-res encoded H264
    _full_right_q: dai.MessageQueue | None = _private()   # full-res encoded H264
    _control_q: Any | None = _private()
    _decoder: Any | None = _private()

    _preview_rgb_frame: np.ndarray | None = _private()
    _preview_left_frame: np.ndarray | None = _private()
    _preview_right_frame: np.ndarray | None = _private()

    _latest_frame: np.ndarray | None = _private()
    _last_frame_time: float = _private(0.0)
    _frame_lock: threading.Lock = _private(factory=threading.Lock)
    _state_lock: threading.RLock = _private(factory=threading.RLock)

    def __post_init__(self) -> None:
        self._validate_config()
        self.width = int(self.config.width)
        self.height = int(self.config.height)

    def __enter__(self) -> DepthAICamera:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        with self._state_lock:
            return all((self._device, self._pipeline, self._rgb_q))

    @property
    def last_frame_age_s(self) -> float | None:
        with self._frame_lock:
            if self._latest_frame is None or self._last_frame_time <= 0.0:
                return None
            return time.monotonic() - self._last_frame_time

    def open(self) -> None:
        from utils import DepthAIH264Decoder, FullRGBStereoH264Reader

        with self._state_lock:
            if self.is_open:
                return
            self._clear_latest_frame()
            try:
                self._device = self._create_device()
                self._bind_pipeline_result(self._make_pipeline(), DepthAIH264Decoder, FullRGBStereoH264Reader)
                self._pipeline.start()
            except Exception:
                logger.exception("Failed to open DepthAI camera")
                self.close()
                raise

    def close(self) -> None:
        with self._state_lock:
            pipeline, device = self._pipeline, self._device
            self._pipeline = self._device = None
            self._rgb_q = self._left_q = self._right_q = None
            self._full_rgb_q = self._full_left_q = self._full_right_q = None
            self._control_q = self._decoder = None
            self._preview_rgb_frame = self._preview_left_frame = self._preview_right_frame = None
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    logger.exception("Failed to stop DepthAI pipeline")
            if device is not None:
                try:
                    device.close()
                except Exception:
                    logger.exception("Failed to close DepthAI device")
            self._clear_latest_frame()

    def read_frame(self, *, timeout_s: float = 1.0,
                   stop_event: threading.Event | None = None) -> list[np.ndarray] | None:
        """Return low-latency preview frames.

        In realtime mode this intentionally does NOT decode the full-resolution H264
        streams.  Use read_encoded_packets() to consume/record/forward those streams.
        """
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = time.monotonic() + timeout_s

        while not _is_stopped(stop_event):
            with self._state_lock:
                preview_q = self._rgb_q
                left_q, right_q = self._left_q, self._right_q
                realtime_mode = self._full_rgb_q is not None
            if preview_q is None:
                return None

            if realtime_mode:
                frames = self._compose_latest_preview(
                    _drain_queue(preview_q, max_packets=8),
                    _drain_queue(left_q, max_packets=8),
                    _drain_queue(right_q, max_packets=8),
                )
                if frames:
                    for frame in frames:
                        self._cache_frame(frame)
                    return frames
            else:
                frames = self._decode_available(_drain_queue(preview_q), _drain_queue(left_q), _drain_queue(right_q))
                if frames:
                    for frame in frames:
                        self._cache_frame(frame)
                    return frames

            if time.monotonic() >= deadline:
                return None
            time.sleep(self._poll_sleep_s())
        return None

    def read_encoded_packets(self, *, max_packets: int = 128) -> dict[str, list[Any]]:
        """Drain full-resolution encoded H264 packets without decoding them.

        Call this in the code path that records, forwards, muxes, or sends the full data.
        Keeping these packets encoded is the only practical way to sustain full 12MP RGB
        plus stereo at 15 FPS in Python.
        """
        with self._state_lock:
            rgb_q, left_q, right_q = self._full_rgb_q, self._full_left_q, self._full_right_q
        return {
            "rgb": _drain_queue(rgb_q, max_packets=max_packets),
            "left": _drain_queue(left_q, max_packets=max_packets),
            "right": _drain_queue(right_q, max_packets=max_packets),
        }

    def latest_frame_copy(self, *, wait_s: float = 2.0, pull_if_missing: bool = False,
                          stop_event: threading.Event | None = None) -> np.ndarray:
        if wait_s < 0:
            raise ValueError("wait_s must be non-negative")
        deadline = time.monotonic() + wait_s

        while not _is_stopped(stop_event):
            frame = self._latest_frame_copy_or_none()
            if frame is not None:
                return frame
            if time.monotonic() >= deadline:
                break
            if pull_if_missing:
                self.read_frame(timeout_s=min(0.2, max(0.0, deadline - time.monotonic())), stop_event=stop_event)
            else:
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        raise RuntimeError("No camera frame available")

    def set_manual_exposure(self, *, exposure_ms: int, iso: int) -> bool:
        if exposure_ms <= 0:
            raise ValueError("exposure_ms must be positive")
        if iso <= 0:
            raise ValueError("iso must be positive")
        with self._state_lock:
            control_q = self._control_q
        if control_q is None:
            logger.debug("DepthAI inputControl queue is unavailable; exposure not changed")
            return False
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(int(exposure_ms * 1000), int(iso))
        control_q.send(ctrl)
        return True

    def _make_pipeline(self) -> tuple[Any, ...]:
        from utils import make_full_rgb_stereo_h264_plus_preview_pipeline

        return make_full_rgb_stereo_h264_plus_preview_pipeline(
            device=self._device,
            fps=self.config.fps,
            queue_max_size=getattr(self.config, "queue_max_size", 1),
            rgb_bitrate_kbps=getattr(self.config, "rgb_bitrate_kbps", 40000),
            mono_bitrate_kbps=getattr(self.config, "mono_bitrate_kbps", 4000),
            preview_width=getattr(self.config, "preview_width", min(960, int(self.config.width))),
            preview_height=getattr(self.config, "preview_height", min(720, int(self.config.height))),
            preview_stereo_height=getattr(self.config, "preview_stereo_height", None),
            rgb_encoder_pool_frames=getattr(self.config, "rgb_encoder_pool_frames", 1),
            mono_encoder_pool_frames=getattr(self.config, "mono_encoder_pool_frames", 1),
            rgb_full_width=getattr(self.config, "rgb_full_width", None),
            rgb_full_height=getattr(self.config, "rgb_full_height", None),
        )

    def _bind_pipeline_result(self, res: tuple[Any, ...], h264_decoder: Any, stereo_reader: Any) -> None:
        if len(res) == 8:
            (self._pipeline, self._rgb_q, self._left_q, self._right_q,
             self._full_rgb_q, self._full_left_q, self._full_right_q,
             self._control_q) = res
            self._decoder = None
            self._preview_rgb_frame = self._preview_left_frame = self._preview_right_frame = None
        elif len(res) == 6:
            # Backward compatibility with the v3 pipeline that returned a single
            # composed preview queue. Prefer the v4 8-value pipeline to avoid the
            # HostNode/Sync timestamp warnings.
            (self._pipeline, self._rgb_q, self._full_rgb_q, self._full_left_q,
             self._full_right_q, self._control_q) = res
            self._left_q = self._right_q = None
            self._decoder = None
        elif len(res) == 3:
            self._pipeline, self._rgb_q, self._control_q = res
            self._left_q = self._right_q = None
            self._full_rgb_q = self._full_left_q = self._full_right_q = None
            self._decoder = h264_decoder(output_format="bgr24")
        elif len(res) == 5:
            self._pipeline, self._rgb_q, self._left_q, self._right_q, self._control_q = res
            self._full_rgb_q = self._full_left_q = self._full_right_q = None
            self._decoder = stereo_reader(output_format="bgr24")
        else:
            raise RuntimeError(f"Unsupported pipeline return shape: {len(res)} values")

    def _compose_latest_preview(self, rgb_packets: list[Any], left_packets: list[Any],
                                right_packets: list[Any]) -> list[np.ndarray]:
        """Compose a low-latency preview from the newest frames in each queue.

        This avoids DepthAI HostNode.link_args()/Sync for previews. At full 12MP RGB
        load, RGB and mono timestamps may not line up within a few milliseconds; strict
        on-device sync only builds latency. For display we prefer newest available
        frames and drop stale preview frames.
        """
        if rgb_packets:
            self._preview_rgb_frame = rgb_packets[-1].getCvFrame()
        if left_packets:
            self._preview_left_frame = left_packets[-1].getCvFrame()
        if right_packets:
            self._preview_right_frame = right_packets[-1].getCvFrame()

        if self._preview_rgb_frame is None or self._preview_left_frame is None or self._preview_right_frame is None:
            return []

        rgb = self._preview_rgb_frame
        left = self._preview_left_frame
        right = self._preview_right_frame

        if left.ndim == 2:
            left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        if right.ndim == 2:
            right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

        # Make the bottom row exactly as wide as the RGB preview without resizing up.
        rgb_h, rgb_w = rgb.shape[:2]
        stereo_h = max(left.shape[0], right.shape[0])
        bottom = np.zeros((stereo_h, rgb_w, 3), dtype=np.uint8)
        x = 0
        bottom[:left.shape[0], x:x + min(left.shape[1], rgb_w)] = left[:, :min(left.shape[1], rgb_w)]
        x += left.shape[1]
        if x < rgb_w:
            bottom[:right.shape[0], x:x + min(right.shape[1], rgb_w - x)] = right[:, :min(right.shape[1], rgb_w - x)]

        return [np.ascontiguousarray(np.vstack([rgb, bottom]))]

    def _decode_available(self, rgb_packets: list[Any], left_packets: list[Any],
                          right_packets: list[Any]) -> list[np.ndarray]:
        if not rgb_packets and not left_packets and not right_packets:
            return []
        if self._left_q is not None or self._right_q is not None:
            self._decoder.decode_packet_batches(
                rgb_packets=rgb_packets, left_packets=left_packets, right_packets=right_packets)
            return self._decoder.compose_latest_rgb_with_fresh_stereo()

        frames: list[np.ndarray] = []
        for packet in rgb_packets:
            if hasattr(packet, "getCvFrame"):
                frames.append(packet.getCvFrame())
            elif self._decoder is not None:
                frames.extend(self._decoder.decode(packet))
        return frames

    def _create_device(self) -> dai.Device:
        device_id = getattr(self.config, "device_id", None)
        return dai.Device(dai.DeviceInfo(str(device_id))) if device_id else dai.Device()

    def _cache_frame(self, frame: np.ndarray) -> None:
        self.width, self.height = int(frame.shape[1]), int(frame.shape[0])
        with self._frame_lock:
            self._latest_frame = frame
            self._last_frame_time = time.monotonic()

    def _latest_frame_copy_or_none(self) -> np.ndarray | None:
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def _clear_latest_frame(self) -> None:
        with self._frame_lock:
            self._latest_frame = None
            self._last_frame_time = 0.0
        self._preview_rgb_frame = None
        self._preview_left_frame = None
        self._preview_right_frame = None

    def _poll_sleep_s(self) -> float:
        return max(0.001, float(self.config.frame_poll_sleep_s))

    def _validate_config(self) -> None:
        missing = [name for name in REQUIRED_CONFIG_FIELDS if not hasattr(self.config, name)]
        if missing:
            raise TypeError(f"Camera config is missing required fields: {', '.join(missing)}")
        checks = (
            (int(self.config.width) > 0, "config.width must be positive"),
            (int(self.config.height) > 0, "config.height must be positive"),
            (float(self.config.fps) > 0, "config.fps must be positive"),
            (int(self.config.queue_max_size) > 0, "config.queue_max_size must be positive"),
            (float(self.config.frame_poll_sleep_s) >= 0, "config.frame_poll_sleep_s must be non-negative"),
        )
        for ok, message in checks:
            if not ok:
                raise ValueError(message)
