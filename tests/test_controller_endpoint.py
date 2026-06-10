from __future__ import annotations

from dataclasses import dataclass

import pytest
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


class TextResult(RpcModel):
    text: str


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

    def fail(self, params: EmptyParams) -> ValueResult:
        raise RuntimeError("boom")

    def echo_default(self, params: EmptyParams = EmptyParams()) -> TextResult:
        return TextResult(text="defaulted")

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
        "calculator.echo_default",
        "calculator.fail",
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


def test_iter_rpc_methods_requires_controller_identity() -> None:
    with pytest.raises(TypeError, match="service_name"):
        describe_controller(object())


def test_endpoint_rejects_unknown_controller_prefix() -> None:
    endpoint = ControllerRpcEndpoint(CalculatorController())

    response = endpoint.handle(
        JsonRpcRequest(id="abc", method="other.add", params={"left": 1, "right": 2})
    )

    assert response.result is None
    assert response.error is not None
    assert response.error.code == -32601
    assert response.error.message == "Unknown controller for method: other.add"
    assert response.id == "abc"


def test_endpoint_rejects_unknown_method_after_valid_prefix() -> None:
    endpoint = ControllerRpcEndpoint(CalculatorController())

    response = endpoint.handle(JsonRpcRequest(id=99, method="calculator.subtract"))

    assert response.error is not None
    assert response.error.code == -32601
    assert response.error.message == "Method not found: calculator.subtract"


def test_endpoint_returns_validation_details_for_extra_params() -> None:
    endpoint = ControllerRpcEndpoint(CalculatorController())

    response = endpoint.handle(
        JsonRpcRequest(
            id=123,
            method="calculator.add",
            params={"left": 2, "right": 5, "extra": 7},
        )
    )

    assert response.error is not None
    assert response.error.code == -32602
    assert response.error.data[0]["type"] == "extra_forbidden"
    assert response.error.data[0]["loc"] == ("extra",)


def test_endpoint_converts_controller_exceptions_to_internal_error() -> None:
    endpoint = ControllerRpcEndpoint(CalculatorController())

    response = endpoint.handle(JsonRpcRequest(id=123, method="calculator.fail"))

    assert response.error is not None
    assert response.error.code == -32603
    assert response.error.message == "Internal error"
    assert response.error.data == "boom"


def test_endpoint_treats_null_params_as_empty_object() -> None:
    endpoint = ControllerRpcEndpoint(CalculatorController())

    response = endpoint.handle(
        JsonRpcRequest(id=123, method="calculator.echo_default", params=None)
    )

    assert response.error is None
    assert response.result == {"text": "defaulted"}
