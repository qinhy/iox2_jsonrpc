from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from iox2_jsonrpc.iox2_transport import Iceoryx2JsonRpcServer
from common import EmptyParams, add_config_arg, build_processor, load_service_name


class CameraBaseModel(BaseModel):
    service: Literal["serverB"] = "serverB"

    def model_post_init(self, context):
        print(f"[{self.service} {self.__class__.__name__}]")
        return super().model_post_init(context)
    
class CaptureParams(CameraBaseModel):
    exposure_ms: int = Field(default=10, ge=1, le=10000)

class CameraStatusResult(CameraBaseModel):
    opened: bool
    captures: int

class CaptureResult(CameraStatusResult):
    frame_id: int
    exposure_ms: int


@dataclass
class CameraController:
    opened: bool = False
    captures: int = 0

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


def main() -> None:
    parser = argparse.ArgumentParser(description="serverB: camera JSON-RPC service over iceoryx2")
    add_config_arg(parser)
    parser.add_argument("--service-key", default="serverB")
    args = parser.parse_args()

    iox_service_name = load_service_name(args.config, args.service_key)
    
    processor=build_processor(CameraController(),prefix="camera")

    server = Iceoryx2JsonRpcServer(service_name=iox_service_name, processor=processor)
    server.serve_forever()


if __name__ == "__main__":
    main()
