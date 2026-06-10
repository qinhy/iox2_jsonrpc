from __future__ import annotations

import pytest
from pydantic import ValidationError

from iox2_jsonrpc import JsonRpcErrorObject, JsonRpcRequest, JsonRpcResponse, RpcModel


class NestedResult(RpcModel):
    value: int


def test_json_rpc_response_rejects_result_and_error_together() -> None:
    with pytest.raises(ValidationError, match="cannot contain both result and error"):
        JsonRpcResponse(
            id=1,
            result={"value": 1},
            error=JsonRpcErrorObject(code=-32603, message="Internal error"),
        )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (NestedResult(value=42), {"value": 42}),
        ({"value": 7}, {"value": 7}),
        ([1, 2, 3], [1, 2, 3]),
    ],
)
def test_json_rpc_response_ok_serializes_supported_results(
    result: object,
    expected: object,
) -> None:
    response = JsonRpcResponse.ok(id="request-1", result=result)

    assert response.model_dump(exclude_none=True) == {
        "jsonrpc": "2.0",
        "id": "request-1",
        "result": expected,
    }


def test_json_rpc_response_fail_builds_error_object_with_data() -> None:
    response = JsonRpcResponse.fail(
        id=None,
        code=-32602,
        message="Invalid params",
        data=[{"loc": ["left"], "msg": "Input should be greater than or equal to 0"}],
    )

    assert response.result is None
    assert response.error is not None
    assert response.error.code == -32602
    assert response.error.data == [
        {"loc": ["left"], "msg": "Input should be greater than or equal to 0"}
    ]


def test_rpc_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        JsonRpcRequest(method="controller.method", unexpected=True)  # type: ignore[call-arg]


def test_rpc_models_validate_assignment() -> None:
    request = JsonRpcRequest(id=1, method="controller.method")

    with pytest.raises(ValidationError):
        request.jsonrpc = "1.0"  # type: ignore[assignment]
