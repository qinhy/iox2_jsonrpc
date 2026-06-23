from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

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


def _try_get(queue: Any | None) -> Any | None:
    return queue.tryGet() if queue is not None else None


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
    """Compact adapter that owns the DepthAI device, pipeline, queues, and latest frame."""

    config: CameraConfigProtocol
    width: int = _private()
    height: int = _private()

    _device: dai.Device | None = _private()
    _pipeline: dai.Pipeline | None = _private()
    _rgb_q: dai.MessageQueue | None = _private()
    _left_q: dai.MessageQueue | None = _private()
    _right_q: dai.MessageQueue | None = _private()
    _control_q: Any | None = _private()
    _decoder: Any | None = _private()

    _latest_frame: np.ndarray | None = _private()
    _last_frame_time: float = _private(0.0)
    _frame_lock: threading.Lock = _private(factory=threading.Lock)
    _state_lock: threading.RLock = _private(factory=threading.RLock)

    def __post_init__(self) -> None:
        self._validate_config(); self.width = int(self.config.width); self.height = int(self.config.height)

    def __enter__(self) -> DepthAICamera:
        self.open(); return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        """Whether DepthAI resources are initialized."""
        with self._state_lock: return all((self._device, self._pipeline, self._rgb_q))

    @property
    def last_frame_age_s(self) -> float | None:
        """Age of the cached frame, or None before the first frame."""
        with self._frame_lock:
            if self._latest_frame is None or self._last_frame_time <= 0.0: return None
            return time.monotonic() - self._last_frame_time

    def open(self) -> None:
        """Open the DepthAI device and start the configured pipeline."""
        from utils import DepthAIH264Decoder, FullRGBStereoH264Reader

        with self._state_lock:
            if self.is_open: return
            self._clear_latest_frame()
            try:
                self._device = self._create_device()
                self._bind_pipeline_result(self._make_pipeline(), DepthAIH264Decoder, FullRGBStereoH264Reader)
                self._pipeline.start()
            except Exception:
                logger.exception("Failed to open DepthAI camera"); self.close(); raise

    def close(self) -> None:
        """Close DepthAI resources. Safe to call repeatedly."""
        with self._state_lock:
            pipeline, device = self._pipeline, self._device
            self._pipeline = self._device = self._rgb_q = self._left_q = self._right_q = self._control_q = self._decoder = None
            if pipeline is not None:
                try: pipeline.stop()
                except Exception: logger.exception("Failed to stop DepthAI pipeline")
            if device is not None:
                try: device.close()
                except Exception: logger.exception("Failed to close DepthAI device")
            self._clear_latest_frame()

    def read_frame(self, *, timeout_s: float = 1.0,
                   stop_event: threading.Event | None = None) -> list[np.ndarray] | None:
        """Read available frames, cache the latest one, and return frames or None on timeout."""
        if timeout_s < 0: raise ValueError("timeout_s must be non-negative")
        deadline = time.monotonic() + timeout_s

        while not _is_stopped(stop_event):
            with self._state_lock: rgb_q, left_q, right_q = self._rgb_q, self._left_q, self._right_q
            if rgb_q is None: return None

            frames = self._decode_available(_try_get(rgb_q), _try_get(left_q), _try_get(right_q))
            if frames:
                for frame in frames: self._cache_frame(frame)
                return frames
            if time.monotonic() >= deadline: return None
            time.sleep(self._poll_sleep_s())
        return None

    def latest_frame_copy(self, *, wait_s: float = 2.0, pull_if_missing: bool = False,
                          stop_event: threading.Event | None = None) -> np.ndarray:
        """Return a copy of the latest frame, optionally polling until one is available."""
        if wait_s < 0: raise ValueError("wait_s must be non-negative")
        deadline = time.monotonic() + wait_s

        while not _is_stopped(stop_event):
            frame = self._latest_frame_copy_or_none()
            if frame is not None: return frame
            if time.monotonic() >= deadline: break
            if pull_if_missing: self.read_frame(timeout_s=min(0.2, max(0.0, deadline - time.monotonic())), stop_event=stop_event)
            else: time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        raise RuntimeError("No camera frame available")

    def set_manual_exposure(self, *, exposure_ms: int, iso: int) -> bool:
        """Set manual exposure if the DepthAI control queue exists."""
        if exposure_ms <= 0: raise ValueError("exposure_ms must be positive")
        if iso <= 0: raise ValueError("iso must be positive")
        with self._state_lock: control_q = self._control_q
        if control_q is None:
            logger.debug("DepthAI inputControl queue is unavailable; exposure not changed"); return False
        ctrl = dai.CameraControl(); ctrl.setManualExposure(int(exposure_ms * 1000), int(iso)); control_q.send(ctrl)
        return True

    def _make_pipeline(self) -> tuple[dai.Pipeline, Any, Any | None] | tuple[dai.Pipeline, Any, Any, Any, Any | None]:
        from utils import make_full_rgb_stereo_h264_synced_pipeline

        return make_full_rgb_stereo_h264_synced_pipeline(
            device=self._device, fps=self.config.fps, queue_max_size=self.config.queue_max_size)

    def _bind_pipeline_result(self, res: tuple[Any, ...], h264_decoder: Any, stereo_reader: Any) -> None:
        if len(res) == 3:
            self._pipeline, self._rgb_q, self._control_q = res
            self._left_q = self._right_q = None; self._decoder = h264_decoder(output_format="bgr24")
        elif len(res) == 5:
            self._pipeline, self._rgb_q, self._left_q, self._right_q, self._control_q = res
            self._decoder = stereo_reader(output_format="bgr24")
        else: raise RuntimeError(f"Unsupported pipeline return shape: {len(res)} values")

    def _decode_available(self, rgb_packet: Any | None, left_packet: Any | None,
                          right_packet: Any | None) -> list[np.ndarray]:
        if rgb_packet is None and left_packet is None and right_packet is None: return []
        if self._left_q is not None or self._right_q is not None:
            self._decoder.decode_packets(rgb_packet=rgb_packet, left_packet=left_packet, right_packet=right_packet)
            return self._decoder.compose_rgb_with_latest_stereo()
        return self._decode_single_rgb(rgb_packet) if rgb_packet is not None else []

    def _decode_single_rgb(self, packet: Any) -> list[np.ndarray]:
        if hasattr(packet, "getCvFrame"): return [packet.getCvFrame()]
        return self._decoder.decode(packet) if self._decoder is not None else []

    def _create_device(self) -> dai.Device:
        device_id = getattr(self.config, "device_id", None)
        return dai.Device(dai.DeviceInfo(str(device_id))) if device_id else dai.Device()

    def _cache_frame(self, frame: np.ndarray) -> None:
        self.width, self.height = int(frame.shape[1]), int(frame.shape[0])
        with self._frame_lock: self._latest_frame = frame; self._last_frame_time = time.monotonic()

    def _latest_frame_copy_or_none(self) -> np.ndarray | None:
        with self._frame_lock: return None if self._latest_frame is None else self._latest_frame.copy()

    def _clear_latest_frame(self) -> None:
        with self._frame_lock: self._latest_frame = None; self._last_frame_time = 0.0

    def _poll_sleep_s(self) -> float:
        return max(0.001, float(self.config.frame_poll_sleep_s))

    def _validate_config(self) -> None:
        missing = [name for name in REQUIRED_CONFIG_FIELDS if not hasattr(self.config, name)]
        if missing: raise TypeError(f"Camera config is missing required fields: {', '.join(missing)}")
        checks = (
            (int(self.config.width) > 0, "config.width must be positive"),
            (int(self.config.height) > 0, "config.height must be positive"),
            (float(self.config.fps) > 0, "config.fps must be positive"),
            (int(self.config.queue_max_size) > 0, "config.queue_max_size must be positive"),
            (float(self.config.frame_poll_sleep_s) >= 0, "config.frame_poll_sleep_s must be non-negative"),
        )
        for ok, message in checks:
            if not ok: raise ValueError(message)
