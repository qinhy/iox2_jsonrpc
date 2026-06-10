from __future__ import annotations

import importlib
import json
from typing import Any

import pytest

from iox2_jsonrpc import JsonRpcResponse


@pytest.fixture()
def registry_module(install_fake_iox2: Any) -> Any:
    install_fake_iox2()
    import iox2_jsonrpc.iceoryx_runtime as runtime

    importlib.reload(runtime)

    import iox2_jsonrpc.iceoryx_registry as registry

    return importlib.reload(registry)


class _FakeNode:
    def __init__(self) -> None:
        self.waits: list[int] = []

    def wait(self, duration: int) -> None:
        self.waits.append(duration)


def test_catalog_returns_json_ready_service_entries(registry_module: Any) -> None:
    service = registry_module.DiscoveredRpcService(
        service_name="mathService",
        controller_name="calculator",
        rpc_endpoint="mathService/calculator/rpc",
        schema_endpoint="mathService/calculator/schema",
        methods=["calculator.add"],
        attributes={"rpc.protocol": "jsonrpc-2.0"},
    )
    registry = registry_module.Iox2RpcRegistry(node=_FakeNode(), services=[service])

    assert registry.catalog() == [
        {
            "service": "mathService",
            "controller": "calculator",
            "rpc_endpoint": "mathService/calculator/rpc",
            "schema_endpoint": "mathService/calculator/schema",
            "methods": ["calculator.add"],
            "attributes": {"rpc.protocol": "jsonrpc-2.0"},
        }
    ]


def test_call_builds_json_rpc_request_and_increments_ids(
    registry_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payloads: list[dict[str, Any]] = []
    registry = registry_module.Iox2RpcRegistry(node=_FakeNode(), services=[])

    def fake_request_bytes(*, client: object, payload: bytes, timeout_s: float) -> bytes:
        assert client == "client"
        assert timeout_s == 3.5
        captured_payloads.append(json.loads(payload))
        return (
            JsonRpcResponse.ok(id=captured_payloads[-1]["id"], result={"value": 9})
            .model_dump_json()
            .encode()
        )

    monkeypatch.setattr(registry, "_request_bytes", fake_request_bytes)

    service = registry_module.DiscoveredRpcService(
        service_name="mathService",
        controller_name="calculator",
        rpc_endpoint="mathService/calculator/rpc",
        schema_endpoint="mathService/calculator/schema",
        methods=["calculator.add"],
        attributes={},
        rpc_client="client",
    )

    first = registry.call(service, "calculator.add", {"left": 4, "right": 5}, timeout_s=3.5)
    second = registry.call(service, "calculator.add", None, timeout_s=3.5)

    assert first.result == {"value": 9}
    assert second.id == 2
    assert captured_payloads == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "calculator.add",
            "params": {"left": 4, "right": 5},
        },
        {"jsonrpc": "2.0", "id": 2, "method": "calculator.add", "params": {}},
    ]


def test_call_requires_opened_rpc_client(registry_module: Any) -> None:
    service = registry_module.DiscoveredRpcService(
        service_name="mathService",
        controller_name="calculator",
        rpc_endpoint="mathService/calculator/rpc",
        schema_endpoint="mathService/calculator/schema",
        methods=["calculator.add"],
        attributes={},
    )
    registry = registry_module.Iox2RpcRegistry(node=_FakeNode(), services=[service])

    with pytest.raises(RuntimeError, match="has no opened RPC client"):
        registry.call(service, "calculator.add")


def test_call_unique_reports_missing_and_ambiguous_methods(registry_module: Any) -> None:
    first = registry_module.DiscoveredRpcService(
        service_name="svc1",
        controller_name="calculator",
        rpc_endpoint="svc1/calculator/rpc",
        schema_endpoint="svc1/calculator/schema",
        methods=["calculator.add"],
        attributes={},
        rpc_client="client-1",
    )
    second = registry_module.DiscoveredRpcService(
        service_name="svc2",
        controller_name="calculator",
        rpc_endpoint="svc2/calculator/rpc",
        schema_endpoint="svc2/calculator/schema",
        methods=["calculator.add"],
        attributes={},
        rpc_client="client-2",
    )
    registry = registry_module.Iox2RpcRegistry(node=_FakeNode(), services=[first, second])

    with pytest.raises(RuntimeError, match="No discovered service provides method"):
        registry.call_unique("calculator.subtract")

    with pytest.raises(RuntimeError, match="ambiguous"):
        registry.call_unique("calculator.add")


def test_call_endpoint_uses_exact_rpc_endpoint(
    registry_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = registry_module.DiscoveredRpcService(
        service_name="mathService",
        controller_name="calculator",
        rpc_endpoint="mathService/calculator/rpc",
        schema_endpoint="mathService/calculator/schema",
        methods=["calculator.add"],
        attributes={},
        rpc_client="client",
    )
    registry = registry_module.Iox2RpcRegistry(node=_FakeNode(), services=[service])
    calls: list[tuple[str, str, dict[str, int] | None, float]] = []

    def fake_call(
        found_service: object,
        method: str,
        params: dict[str, int] | None = None,
        *,
        timeout_s: float = 5.0,
    ) -> JsonRpcResponse:
        assert found_service is service
        calls.append((service.rpc_endpoint, method, params, timeout_s))
        return JsonRpcResponse.ok(id=1, result={"ok": True})

    monkeypatch.setattr(registry, "call", fake_call)

    response = registry.call_endpoint(
        "mathService/calculator/rpc",
        "calculator.add",
        {"left": 1},
        timeout_s=0.25,
    )

    assert response.result == {"ok": True}
    assert calls == [("mathService/calculator/rpc", "calculator.add", {"left": 1}, 0.25)]


def test_find_by_endpoint_reports_missing_endpoint(registry_module: Any) -> None:
    registry = registry_module.Iox2RpcRegistry(node=_FakeNode(), services=[])

    with pytest.raises(RuntimeError, match="No RPC service found at endpoint"):
        registry.find_by_endpoint("missing/rpc")
