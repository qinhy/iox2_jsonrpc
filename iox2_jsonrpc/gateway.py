from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .iox2_transport import Iceoryx2JsonRpcClientPool
from .services import JsonRpcServiceRegistry


JsonRpcId = int | str | None
JsonRpcParams = dict[str, Any] | list[Any] | None


class JsonRpcHttpRequest(BaseModel):
    """OpenAPI-friendly JSON-RPC request body.

    This model exists only so FastAPI /docs shows a request-body editor.
    The gateway still forwards JSON bytes to the selected iceoryx2 service.
    """

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = Field(default="2.0", description="JSON-RPC protocol version")
    method: str = Field(..., examples=["pipeline.status"])
    params: JsonRpcParams = Field(default=None, examples=[{}])
    id: JsonRpcId = Field(default=None, examples=[1])


JsonRpcHttpBody = Annotated[
    JsonRpcHttpRequest | list[JsonRpcHttpRequest],
    Body(
        openapi_examples={
            "serverA_pipeline_status": {
                "summary": "serverA: pipeline.status",
                "description": "Use service_name = serverA with /{service_name}/rpc.",
                "value": {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "pipeline.status",
                    "params": {},
                },
            },
            "serverA_pipeline_start": {
                "summary": "serverA: pipeline.start",
                "value": {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "pipeline.start",
                    "params": {"profile": "demo"},
                },
            },
            "serverB_camera_capture": {
                "summary": "serverB: camera.capture",
                "description": "Use service_name = serverB with /{service_name}/rpc.",
                "value": {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "camera.capture",
                    "params": {"path": "capture.jpg"},
                },
            },
            "batch": {
                "summary": "Batch request",
                "value": [
                    {
                        "jsonrpc": "2.0",
                        "id": 10,
                        "method": "pipeline.status",
                        "params": {},
                    },
                    {
                        "jsonrpc": "2.0",
                        "id": 11,
                        "method": "pipeline.stop",
                        "params": {},
                    },
                ],
            },
            "notification": {
                "summary": "Notification: no response body",
                "value": {
                    "jsonrpc": "2.0",
                    "method": "pipeline.stop",
                    "params": {},
                },
            },
        }
    ),
]


class FastApiJsonRpcGateway:
    def __init__(self, registry: JsonRpcServiceRegistry, client_pool: Iceoryx2JsonRpcClientPool | None = None) -> None:
        self.registry = registry
        self.client_pool = client_pool or Iceoryx2JsonRpcClientPool()

    def create_app(self) -> FastAPI:
        app = FastAPI(title="JSON-RPC over iceoryx2 Gateway")

        @app.get("/services")
        async def list_services() -> dict[str, object]:
            return {"services": [service.model_dump() for service in self.registry.list()]}

        @app.get("/services/{service_name}/health")
        async def service_health(service_name: str) -> dict[str, object]:
            service = self._get_service_or_404(service_name)
            return {
                "name": service.name,
                "iceoryx2_service": service.iceoryx2_service,
                "configured": True,
                "note": "This endpoint checks gateway config, not remote liveness.",
            }

        @app.post("/{service_name}/rpc", summary="Forward JSON-RPC to one iceoryx2 service")
        async def rpc_by_prefix(service_name: str, body: JsonRpcHttpBody) -> Response:
            return await self._forward_body(service_name, body)

        @app.post("/rpc/{service_name}", summary="Forward JSON-RPC to one iceoryx2 service")
        async def rpc_by_namespace(service_name: str, body: JsonRpcHttpBody) -> Response:
            return await self._forward_body(service_name, body)

        return app

    async def _forward_body(self, service_name: str, body: JsonRpcHttpRequest | list[JsonRpcHttpRequest]) -> Response:
        service = self._get_service_or_404(service_name)
        payload = self._body_to_json_bytes(body)

        client = self.client_pool.get(service)
        try:
            response_payload = await run_in_threadpool(client.call, payload)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - gateway turns transport exceptions into HTTP 502.
            raise HTTPException(status_code=502, detail=f"iceoryx2 transport error: {exc}") from exc

        if not response_payload:
            return Response(status_code=204)
        return Response(content=response_payload, media_type="application/json")

    def _body_to_json_bytes(self, body: JsonRpcHttpRequest | list[JsonRpcHttpRequest]) -> bytes:
        if isinstance(body, list):
            value = [item.model_dump(mode="json", exclude_unset=True) for item in body]
        else:
            value = body.model_dump(mode="json", exclude_unset=True)
        return json.dumps(value, separators=(",", ":")).encode("utf-8")

    def _get_service_or_404(self, service_name: str):
        try:
            return self.registry.get(service_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown service: {service_name}") from exc
