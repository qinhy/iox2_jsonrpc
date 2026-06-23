from __future__ import annotations

import argparse
from pathlib import Path


import logging
from pathlib import Path

import json
import logging
from pathlib import Path
from typing import Any

import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(Path(__file__).absolute().parent)))
from rpc_server import CameraConfig, run_server
from utils import print_json_result, shorten_capture_result, configure_file_logging


class CameraRpcApi:
    """Thin JSON-RPC client wrapper for the camera methods."""

    def __init__(self, registry: Any, *, controller_name: str = "camera") -> None:
        from iox2_jsonrpc.iceoryx import Iox2RpcRegistry
        self.registry:Iox2RpcRegistry = registry
        self.controller_name = controller_name

    @classmethod
    def discover(cls, *, controller_name: str = "camera") -> "CameraRpcApi":
        from iox2_jsonrpc.iceoryx import Iox2RpcRegistry

        registry = Iox2RpcRegistry.discover_all()
        logging.info("\n=== Discovered JSON-RPC catalog ===")
        logging.info(json.dumps(registry.catalog(), indent=2, default=str))
        return cls(registry, controller_name=controller_name)

    def method(self, name: str) -> str:
        return f"{self.controller_name}.{name}"

    def call(self, name: str, params: dict[str, Any] | None = None, *, timeout_s: float = 2.0) -> Any:
        method = self.method(name)
        logging.info("\n=== %s ===", method)

        if params is None:
            return self.registry.call_unique(method, timeout_s=timeout_s)

        return self.registry.call_unique(method, params, timeout_s=timeout_s)

    def status(self, *, timeout_s: float = 2.0) -> Any:
        return self.call("status", timeout_s=timeout_s)

    def open(self, *, timeout_s: float = 10.0) -> Any:
        return self.call("open", timeout_s=timeout_s)

    def close(self, *, timeout_s: float = 5.0) -> Any:
        return self.call("close", timeout_s=timeout_s)

    def capture(
        self,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float = 2.0,
    ) -> Any:
        return self.call("capture", params=params, timeout_s=timeout_s)

    def call_and_print(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float = 2.0,
        output_dir: str | Path = ".",
    ) -> Any:
        result = self.call(name, params=params, timeout_s=timeout_s)
        compact = shorten_capture_result(result, output_dir=output_dir)
        print_json_result(self.method(name), compact)
        return compact


def run_client(*, output_dir: str | Path = ".") -> None:
    """Call status, open, capture twice, and close, like the original test client."""

    configure_file_logging()
    logging.info("Program started")

    api = CameraRpcApi.discover()
    api.call_and_print("status", output_dir=output_dir)
    api.call_and_print("open", timeout_s=10.0, output_dir=output_dir)
    api.call_and_print("status", output_dir=output_dir)
    api.call_and_print(
        "capture",
        {
            "exposure_ms": 25,
            "iso": 800,
            "jpeg_quality": 85,
        },
        output_dir=output_dir,
    )
    api.call_and_print("capture", output_dir=output_dir)
    api.call_and_print("close", timeout_s=5.0, output_dir=output_dir)

    logging.info("Program finished")


def run_live_client() -> None:
    """
    Open the camera and keep the server-side preview running until the user closes it.

    The OpenCV preview window is created by the server process, so q/Esc must be
    pressed in the server's preview window. Enter/Ctrl+C in this client calls
    camera.close over RPC.
    """

    configure_file_logging()

    api = CameraRpcApi.discover()
    api.call_and_print("open", timeout_s=10.0)

    print("\nLive camera preview is running.")
    print("Close options:")
    print("  - Press q or Esc in the OpenCV preview window")
    print("  - Press Enter here")
    print("  - Press Ctrl+C here")

    try:
        input("\nPress Enter to close camera... ")
    except KeyboardInterrupt:
        print("\nCtrl+C received. Closing camera...")
    finally:
        try:
            api.call_and_print("close", timeout_s=5.0)
        except Exception:
            logging.exception("Failed to close camera from live client")
            raise

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DepthAI camera JSON-RPC demo")
    parser.add_argument(
        "mode",
        choices=["server", "serve", "client", "live", "preview", "stream"],
        help="Run the RPC server, demo client, or live-preview client.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the server-side OpenCV preview window.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where client capture images are saved.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mode = args.mode.lower()

    if mode in {"server", "serve"}:
        run_server()
    elif mode == "client":
        run_client(output_dir=args.output_dir)
    elif mode in {"live", "preview", "stream"}:
        run_live_client()
    else:
        raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
