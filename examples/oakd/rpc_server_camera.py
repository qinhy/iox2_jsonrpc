from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

Resolution = tuple[int, int]
Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Matrix4x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
DistortionCoeffs = tuple[float, ...]

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

    # OpenCV preview only. read_frame() returns a small preview mosaic, not the
    # full-resolution encoded streams.
    debug_preview: bool = True
    debug_window_name: str = "serverCam live preview"
    debug_preview_timeout_s: float = 0.01
    debug_preview_drain_encoded: bool = True
    debug_preview_show_encoded_fps: bool = True
    debug_preview_max_encoded_packets: int = 256
    encoded_drain_sleep_s: float = 0.001

    # Keep queues small for low latency. The full-resolution H264 packets must be
    # drained continuously by Cv2Preview's encoded-drain thread or by your own
    # recorder/sender thread.
    queue_max_size: int = 1
    frame_poll_sleep_s: float = 0.001
    close_join_timeout_s: float = 2.0

    # Full-resolution encoded path. Leave rgb_full_width/height as None for true
    # highest-resolution RGB, or set 3840x2160 for a practical 15 FPS mode.
    rgb_bitrate_kbps: int = 40000
    mono_bitrate_kbps: int = 4000
    rgb_encoder_pool_frames: int = 1
    mono_encoder_pool_frames: int = 1
    rgb_full_width: int | None = None
    rgb_full_height: int | None = None

    # Preview path dimensions. These do not affect the full H264 streams.
    preview_width: int = 640
    preview_height: int = 480
    preview_stereo_height: int | None = 160


class CaptureParams(CameraBaseModel):
    exposure_ms: int = Field(default=10, ge=1, le=33)
    iso: int = Field(default=800, ge=100, le=1600)
    jpeg_quality: int = Field(default=85, ge=1, le=100)


class CameraCalibrationParams(CameraBaseModel):
    rgb_resolution: Resolution = Field(default=(4056, 3040))
    left_resolution: Resolution = Field(default=(1280, 800))
    right_resolution: Resolution = Field(default=(1280, 800))

    stereo_translation_units_hint: str = Field(
        default="Calibration extrinsics units from device; Luxonis stereo baseline override API uses centimeters."
    )

    rgb_intrinsics: Matrix3x3 = Field(
        default=(
            (2430.31884765625, 0.0, 2063.196044921875),
            (0.0, 2429.41748046875, 1490.1956787109375),
            (0.0, 0.0, 1.0),
        )
    )

    left_intrinsics: Matrix3x3 = Field(
        default=(
            (570.8507690429688, 0.0, 653.754150390625),
            (0.0, 570.580810546875, 390.99169921875),
            (0.0, 0.0, 1.0),
        )
    )

    right_intrinsics: Matrix3x3 = Field(
        default=(
            (567.8758544921875, 0.0, 655.560546875),
            (0.0, 567.7424926757812, 393.97039794921875),
            (0.0, 0.0, 1.0),
        )
    )

    left_to_right_extrinsics: Matrix4x4 = Field(
        default=(
            (0.9998766183853149, 0.0023975009098649025, -0.015519456937909126, -7.537897109985352),
            (-0.0024368134327232838, 0.9999938607215881, -0.002514647087082267, 0.09707357734441757),
            (0.01551333349198103, 0.0025521547067910433, 0.9998764395713806, -0.08006280660629272),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    left_to_rgb_extrinsics: Matrix4x4 = Field(
        default=(
            (0.999745786190033, -0.010174884460866451, -0.020119857043027878, -3.7557337284088135),
            (0.010090984404087067, 0.9999399781227112, -0.004267154261469841, -0.004705727566033602),
            (0.02016206830739975, 0.004063040018081665, 0.9997884631156921, -0.04603101313114166),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    rgb_distortion: DistortionCoeffs = Field(
        default=(
            11.808209419250488,
            11.02328872680664,
            0.0005683265044353902,
            -0.0014364976668730378,
            -1.831695795059204,
            11.769153594970703,
            14.672675132751465,
            -1.088363766670227,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.00932407472282648,
            -0.015433108434081078,
        )
    )

    left_distortion: DistortionCoeffs = Field(
        default=(
            5.454817771911621,
            1.694711446762085,
            8.319105836562812e-05,
            -5.4938958783168346e-05,
            0.029059873893857002,
            5.82321310043335,
            3.369436502456665,
            0.23804199695587158,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.004699581768363714,
            -0.0014164879685267806,
        )
    )

    right_distortion: DistortionCoeffs = Field(
        default=(
            5.091114521026611,
            1.5919005870819092,
            -7.720104804320727e-06,
            2.0027317077619955e-05,
            0.029687780886888504,
            5.4577412605285645,
            3.150493621826172,
            0.22900323569774628,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.0026084419805556536,
            -0.002354657743126154,
        )
    )

    stereo_baseline_cm: float = Field(default=7.537897109985352)

class CameraStatusResult(CameraBaseModel):
    opened: bool
    captures: int
    device_id: str
    width: int
    height: int
    preview_running: bool = False
    camera_calib: CameraCalibrationParams | None = None

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

    config: CameraConfig = field(default_factory=CameraConfig)
    opened: bool = False
    captures: int = 0

    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _camera: DepthAICamera | None = field(default=None, init=False, repr=False)
    _preview: Cv2Preview | None = field(default=None, init=False, repr=False)

    def _ensure_session(self) -> tuple[DepthAICamera, Cv2Preview]:
        if self._camera is None or self._preview is None:
            """Create camera/preview objects lazily for the current config."""

            self._camera = DepthAICamera(self.config)
            self._preview = Cv2Preview(
                self.config,
                on_close_requested=lambda: self._close_depthai(join_preview_thread=False),
                overlay_provider=self._preview_overlay_lines,
            )

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
        config = params
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
                preview.start(camera)
                return

            camera.open()
            self.opened = True
            preview.start(camera)

            # If preview is disabled, pull one frame so width/height are updated.
            if not self.config.debug_preview:
                camera.read_frame(timeout_s=2.0, stop_event=preview.stop_event)

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


def run_server(controller_name="camera") -> None:
    """Run the camera controller as an iceoryx2 JSON-RPC service.

    ``config`` is only the initial/default config. The actual DepthAI camera
    session is created lazily from the CameraConfig passed to ``camera.open``.
    """

    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    server = Iox2JsonRpcServer(CameraController(controller_name=controller_name))
    server.run_forever()
