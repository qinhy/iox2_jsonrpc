from __future__ import annotations

import argparse

import uvicorn

from iox2_jsonrpc.gateway import FastApiJsonRpcGateway
from common import add_config_arg
from iox2_jsonrpc.services import JsonRpcServiceRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="FastAPI gateway for JSON-RPC over iceoryx2 services")
    add_config_arg(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    registry = JsonRpcServiceRegistry.from_toml(args.config)
    app = FastApiJsonRpcGateway(registry).create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
