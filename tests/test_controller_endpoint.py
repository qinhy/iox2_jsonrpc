from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from iox2_jsonrpc import (
    ControllerRpcEndpoint,
    EmptyParams,
    JsonRpcRequest,
    JsonRpcResponse,
    RpcModel,
    describe_controller,
)


class AddParams(RpcModel):
    left: int = Field(ge=0)
    right: int = Field(ge=0)


class ValueResult(RpcModel):
    value: int


@dataclass
class CalculatorController:
    service_name: str = "mathService"
    controller_name: str = "calculator"
    calls: int = 0

    def add(self, params: AddParams) -> ValueResult:
        self.calls += 1
        return ValueResult(value=params.left + params.right)

    def calls_count(self, params: EmptyParams) -> ValueResult:
        return ValueResult(value=self.calls)

    def helper_not_exposed(self, value: int) -> int:
        return value


def test_describe_controller_only_includes_typed_rpc_methods() -> None:
    descriptor = describe_controller(CalculatorController())

    assert descriptor.service == "mathService"
    assert descriptor.controller == "calculator"
    assert descriptor.rpc_endpoint == "mathService/calculator/rpc"
    assert descriptor.schema_endpoint == "mathService/calculator/schema"
    assert [method.jsonrpc_method for method in descriptor.methods] == [
        "calculator.add",
        "calculator.calls_count",
    ]


def test_endpoint_dispatches_valid_request() -> None:
    endpoint = ControllerRpcEndpoint(CalculatorController())

    response = endpoint.handle(
        JsonRpcRequest(
            id=123,
            method="calculator.add",
            params={"left": 2, "right": 5},
        )
    )

    assert response.error is None
    assert response.result == {"value": 7}


def test_endpoint_returns_invalid_params() -> None:
    endpoint = ControllerRpcEndpoint(CalculatorController())

    response = endpoint.handle(
        JsonRpcRequest(
            id=123,
            method="calculator.add",
            params={"left": -1, "right": 5},
        )
    )

    assert response.error is not None
    assert response.error.code == -32602
    assert response.error.message == "Invalid params"


def test_endpoint_handles_json_bytes() -> None:
    endpoint = ControllerRpcEndpoint(CalculatorController())
    raw = b'{"jsonrpc":"2.0","id":1,"method":"calculator.calls_count"}'

    response = JsonRpcResponse.model_validate_json(endpoint.handle_bytes(raw))

    assert response.error is None
    assert response.result == {"value": 0}


def test_endpoint_handles_parse_error() -> None:
    endpoint = ControllerRpcEndpoint(CalculatorController())

    response = JsonRpcResponse.model_validate_json(endpoint.handle_bytes(b"not-json"))

    assert response.error is not None
    assert response.error.code == -32700
