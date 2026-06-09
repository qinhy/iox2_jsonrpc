from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from iox2_jsonrpc import (
    ControllerRpcEndpoint,
    EmptyParams,
    JsonRpcRequest,
    RpcModel,
    describe_controller,
)

from webapi import create_controller_fastapi_app

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


def print_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    return json.dumps(value, indent=2)


def run_local() -> None:
    """Run the controller directly through the local JSON-RPC endpoint."""

    controller = CameraController()
    endpoint = ControllerRpcEndpoint(controller)

    logging.basicConfig(
        filename="local.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("\n=== Descriptor ===")
    logging.info(describe_controller(controller))

    calls = [
        JsonRpcRequest(id=1, method="camera.status"),
        JsonRpcRequest(id=2, method="camera.capture", params={"exposure_ms": 25}),
        JsonRpcRequest(id=3, method="camera.capture"),
        JsonRpcRequest(id=4, method="camera.close"),
    ]

    logging.info("\n=== Local JSON-RPC calls ===")
    for request in calls:
        response = endpoint.handle(request)
        logging.info(f"\n> {request.method}")
        logging.info(print_json(response))


def create_fastapi_app():
    """Create the FastAPI app for this camera example.

    The HTTP adapter itself is generic and lives in iox2_jsonrpc.fastapi.
    This function only chooses which controller instance the example serves.
    """

    return create_controller_fastapi_app(
        CameraController(),
        title="Camera controller API",
        description="FastAPI example using the generic iox2_jsonrpc controller adapter.",
    )


def run_api(host: str, port: int) -> None:
    """Run the camera controller through the generic FastAPI adapter."""

    # Import here so existing modes work without uvicorn installed.
    import uvicorn

    uvicorn.run(create_fastapi_app(), host=host, port=port)


def run_server() -> None:
    """Run the camera controller as a real iceoryx2 JSON-RPC service."""

    # Import here so `python camera.py local` works without iceoryx2.
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    server = Iox2JsonRpcServer(CameraController())
    server.run_forever()


def run_client() -> None:
    """Discover an iceoryx2 JSON-RPC service and call the camera methods."""

    # Import here so `python camera.py local` works without iceoryx2.
    from iox2_jsonrpc.iceoryx import Iox2RpcRegistry

    registry = Iox2RpcRegistry.discover_all()

    logging.basicConfig(
        filename="client.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Program started")
    logging.warning("Something might be wrong")
    logging.error("Something failed")

    logging.info("\n=== Discovered JSON-RPC catalog ===")
    logging.info(json.dumps(registry.catalog(), indent=2))

    logging.info("\n=== camera.status ===")
    logging.info(registry.call_unique("camera.status"))

    logging.info("\n=== camera.capture exposure_ms=25 ===")
    logging.info(registry.call_unique("camera.capture", {"exposure_ms": 25}))

    logging.info("\n=== camera.capture default exposure ===")
    logging.info(registry.call_unique("camera.capture"))

    logging.info("\n=== camera.close ===")
    logging.info(registry.call_unique("camera.close"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-in-one camera example for the iox2_jsonrpc library."
    )
    parser.add_argument(
        "mode",
        choices=["local", "server", "client", "api"],
        help=(
            "local = no iceoryx2 needed; "
            "server/client = real iceoryx2 request-response transport; "
            "api = FastAPI HTTP service"
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for FastAPI mode. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for FastAPI mode. Default: 8000",
    )

    args = parser.parse_args()

    if args.mode == "local":
        run_local()
    elif args.mode == "server":
        run_server()
    elif args.mode == "client":
        run_client()
    elif args.mode == "api":
        run_api(host=args.host, port=args.port)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()