from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from iox2_jsonrpc import EmptyParams, RpcModel

import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(Path(__file__).absolute().parent)))
from cv2_viewer import Cv2Preview
from dai_camera import DepthAICamera
from utils import encode_frame_as_jpeg_base64


class CameraBaseModel(RpcModel):
    """Base model for camera-specific JSON-RPC params/results."""

    service: Literal["serverCam"] = "serverCam"


class CameraConfig(CameraBaseModel):
    """Runtime settings for the server camera service.

    This model is intentionally usable as the JSON-RPC params object for
    ``camera.open`` so callers can choose the camera/session config at runtime.
    """

    device_id: str = "169.254.1.221"
    width: int = 640
    height: int = 480
    fps: int = 15

    debug_preview: bool = True
    debug_window_name: str = "serverCam live preview"

    queue_max_size: int = 4
    frame_poll_sleep_s: float = 0.005
    close_join_timeout_s: float = 2.0


class CaptureParams(CameraBaseModel):
    exposure_ms: int = Field(default=10, ge=1, le=33)
    iso: int = Field(default=800, ge=100, le=1600)
    jpeg_quality: int = Field(default=85, ge=1, le=100)


class CameraStatusResult(CameraBaseModel):
    opened: bool
    captures: int
    device_id: str
    width: int
    height: int
    preview_running: bool = False


class CaptureResult(CameraStatusResult):
    frame_id: int
    exposure_ms: int
    iso: int
    jpeg_base64: str


@dataclass
class CameraController:
    """
    JSON-RPC controller for the DepthAI camera.

    Public RPC methods intentionally keep the original names/behavior:
    - camera.open starts the camera and optional OpenCV preview using the
      CameraConfig supplied by the JSON-RPC request.
    - camera.capture returns latest frame as JPEG base64.
    - camera.close can be called at any time.
    - camera.status reports camera/preview state.
    """

    service_name: str = "serverCam"
    controller_name: str = "camera"

    config: CameraConfig = None
    opened: bool = False
    captures: int = 0

    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _camera: DepthAICamera | None = field(default=None, init=False, repr=False)
    _preview: Cv2Preview | None = field(default=None, init=False, repr=False)

    def _build_session(self) -> None:
        """Create camera/preview objects lazily for the current config."""

        self._camera = DepthAICamera(self.config)
        self._preview = Cv2Preview(
            self.config,
            self._camera,
            on_close_requested=lambda: self._close_depthai(join_preview_thread=False),
            overlay_provider=self._preview_overlay_lines,
        )

    def _ensure_session(self) -> tuple[DepthAICamera, Cv2Preview]:
        if self._camera is None or self._preview is None:
            self._build_session()

        assert self._camera is not None
        assert self._preview is not None
        return self._camera, self._preview

    def _replace_config(self, config: CameraConfig) -> None:
        """Switch to a new config before opening the camera session."""

        self.config = config
        self._camera = None
        self._preview = None

    def _preview_overlay_lines(self) -> list[str]:
        if self._camera is None:
            return [
                "Camera not initialized",
                f"FPS: {self.config.fps}",
                f"Captures: {self.captures}",
            ]

        return [
            f"Width: {self._camera.width}",
            f"Height: {self._camera.height}",
            f"FPS: {self.config.fps}",
            f"Captures: {self.captures}",
        ]

    def _open_depthai(self, config: CameraConfig | None = None) -> None:
        if config is not None and config != self.config and self.opened:
            # Re-opening with a different runtime config means the current
            # hardware session must be closed before a new DepthAICamera is built.
            self._close_depthai(join_preview_thread=True)

        with self._state_lock:
            if config is not None and config != self.config:
                if self._preview is not None:
                    self._preview.destroy_window()
                if self._camera is not None:
                    self._camera.close()
                self.opened = False
                self._replace_config(config)

            camera, preview = self._ensure_session()
            preview.stop_event.clear()

            if self.opened:
                preview.start()
                return

            camera.open()
            self.opened = True
            preview.start()

            # If preview is disabled, pull one frame so width/height are updated.
            if not self.config.debug_preview:
                camera.read_frame(timeout_s=2.0, stop_event=preview.stop_event)

    def _close_depthai(self, *, join_preview_thread: bool = True) -> None:
        # Do not hold _state_lock while joining the preview thread. The preview
        # thread can also request closure after q/Esc.
        preview = self._preview
        if preview is not None:
            preview.stop(join=join_preview_thread)

        with self._state_lock:
            if self._preview is not None:
                self._preview.destroy_window()
            if self._camera is not None:
                self._camera.close()
            self.opened = False

    def _status(self) -> CameraStatusResult:
        camera = self._camera
        preview = self._preview

        return CameraStatusResult(
            opened=self.opened,
            captures=self.captures,
            device_id=self.config.device_id,
            width=camera.width if camera is not None else self.config.width,
            height=camera.height if camera is not None else self.config.height,
            preview_running=preview.is_running if preview is not None else False,
        )

    def open(self, params: CameraConfig) -> CameraStatusResult:
        self._open_depthai(params)
        return self._status()

    def close(self, params: EmptyParams) -> CameraStatusResult:
        self._close_depthai(join_preview_thread=True)
        return self._status()

    def status(self, params: EmptyParams) -> CameraStatusResult:
        return self._status()

    def capture(self, params: CaptureParams) -> CaptureResult:
        self._open_depthai()

        camera, preview = self._ensure_session()
        camera.set_manual_exposure(
            exposure_ms=params.exposure_ms,
            iso=params.iso,
        )

        frame = camera.latest_frame_copy(
            wait_s=2.0,
            pull_if_missing=not self.config.debug_preview,
            stop_event=preview.stop_event,
        )

        self.captures += 1
        jpeg_base64 = encode_frame_as_jpeg_base64(frame, params.jpeg_quality)

        return CaptureResult(
            opened=self.opened,
            captures=self.captures,
            device_id=self.config.device_id,
            width=camera.width,
            height=camera.height,
            preview_running=preview.is_running,
            frame_id=self.captures,
            exposure_ms=params.exposure_ms,
            iso=params.iso,
            jpeg_base64=jpeg_base64,
        )


def run_server() -> None:
    """Run the camera controller as an iceoryx2 JSON-RPC service.

    ``config`` is only the initial/default config. The actual DepthAI camera
    session is created lazily from the CameraConfig passed to ``camera.open``.
    """

    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    server = Iox2JsonRpcServer(CameraController())
    server.run_forever()
