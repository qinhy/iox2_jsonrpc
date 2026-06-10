from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import Field

from iox2_jsonrpc import EmptyParams, JsonRpcRequest, RpcModel, describe_controller


class AddParams(RpcModel):
    left: int = Field(ge=0)
    right: int = Field(ge=0)


class ValueResult(RpcModel):
    value: int


@dataclass
class CalculatorController:
    service_name: str = "mathService"
    controller_name: str = "calculator"

    def add(self, params: AddParams) -> ValueResult:
        return ValueResult(value=params.left + params.right)

    def calls_count(self, params: EmptyParams) -> ValueResult:
        return ValueResult(value=0)


def test_readme_controller_descriptor_shape() -> None:
    descriptor = describe_controller(CalculatorController())

    assert descriptor.model_dump(exclude={"methods": {"__all__": {"params_schema", "result_schema"}}}) == {
        "protocol": "jsonrpc-2.0",
        "service": "mathService",
        "controller": "calculator",
        "rpc_endpoint": "mathService/calculator/rpc",
        "schema_endpoint": "mathService/calculator/schema",
        "methods": [
            {
                "name": "add",
                "jsonrpc_method": "calculator.add",
                "params_model": "AddParams",
                "result_model": "ValueResult",
            },
            {
                "name": "calls_count",
                "jsonrpc_method": "calculator.calls_count",
                "params_model": "EmptyParams",
                "result_model": "ValueResult",
            },
        ],
    }


def test_readme_json_rpc_request_bytes_are_stable() -> None:
    request = JsonRpcRequest(
        id=1,
        method="calculator.add",
        params={"left": 2, "right": 5},
    )

    assert json.loads(request.model_dump_json()) == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "calculator.add",
        "params": {"left": 2, "right": 5},
    }
