from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from iox2_jsonrpc import (
    EmptyParams,
    RpcModel,
)


class CameraBaseModel(RpcModel):
    """Base model for camera-specific params/results."""

    service: Literal["serverCam"] = "serverCam"


class CaptureParams(CameraBaseModel):
    exposure_ms: int = Field(default=10, ge=1, le=10_000)


class CameraStatusResult(CameraBaseModel):
    opened: bool
    captures: int


class CaptureResult(CameraStatusResult):
    frame_id: int
    exposure_ms: int


@dataclass
class CameraController:
    """Example controller. The library discovers public typed methods below."""

    opened: bool = False
    captures: int = 0

    service_name: str = "serverCam"
    controller_name: str = "camera"

    def open(self, params: EmptyParams) -> CameraStatusResult:
        self.opened = True
        return CameraStatusResult(opened=self.opened, captures=self.captures)

    def close(self, params: EmptyParams) -> CameraStatusResult:
        self.opened = False
        return CameraStatusResult(opened=self.opened, captures=self.captures)

    def status(self, params: EmptyParams) -> CameraStatusResult:
        return CameraStatusResult(opened=self.opened, captures=self.captures)

    def capture(self, params: CaptureParams) -> CaptureResult:
        if not self.opened:
            self.opened = True

        self.captures += 1
        return CaptureResult(
            opened=self.opened,
            captures=self.captures,
            frame_id=self.captures,
            exposure_ms=params.exposure_ms,
        )
    

def run_server() -> None:
    """Run the camera controller as a real iceoryx2 JSON-RPC service."""

    # Import here so `python camera_all_in_one.py local` works without iceoryx2.
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    server = Iox2JsonRpcServer(CameraController())
    server.run_forever()


if __name__ == "__main__":
    run_server()