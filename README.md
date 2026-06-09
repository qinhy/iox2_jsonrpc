# iox2-jsonrpc

Typed JSON-RPC 2.0 helpers for Python controllers, with optional iceoryx2
request-response transport.

The core package lets you expose public methods on a controller as JSON-RPC
methods when those methods use Pydantic models for both parameters and results.
The iceoryx2 integration publishes the controller over IPC and exposes a schema
endpoint that clients can discover at runtime.

## Features

- JSON-RPC 2.0 request and response models
- Pydantic validation for method parameters and result schemas
- Controller introspection for typed public methods
- In-process endpoint for local dispatch and testing
- Optional iceoryx2 server, service discovery, schema loading, and client calls

## Requirements

- Python 3.11 or newer
- `uv` for the commands below
- `iceoryx2` available in your environment when using IPC transport

The in-process endpoint only needs the package runtime dependencies. The
iceoryx2 transport imports `iceoryx2` lazily from `iox2_jsonrpc.iceoryx`.

## Installation

For local development:

```bash
uv sync
```

Run the tests:

```bash
uv run pytest
```

## Controller Basics

A controller must define `service_name` and `controller_name`. Public methods
are exposed when they accept one Pydantic model argument named `params` and
return a Pydantic model.

```python
from dataclasses import dataclass

from pydantic import Field

from iox2_jsonrpc import EmptyParams, RpcModel, describe_controller


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


descriptor = describe_controller(CalculatorController())
print(descriptor.model_dump_json(indent=2))
```

This exposes:

- `calculator.add`
- `calculator.calls_count`

The default endpoint names are derived from the service and controller names:

- RPC endpoint: `mathService/calculator/rpc`
- Schema endpoint: `mathService/calculator/schema`

## In-Process JSON-RPC

Use `ControllerRpcEndpoint` to dispatch JSON-RPC requests directly without
iceoryx2. This is useful for tests and local controller development.

```python
from iox2_jsonrpc import ControllerRpcEndpoint, JsonRpcRequest

endpoint = ControllerRpcEndpoint(CalculatorController())

response = endpoint.handle(
    JsonRpcRequest(
        id=1,
        method="calculator.add",
        params={"left": 2, "right": 5},
    )
)

print(response.model_dump_json(indent=2))
```

For byte-oriented integrations:

```python
raw = b'{"jsonrpc":"2.0","id":1,"method":"calculator.calls_count"}'
response_bytes = endpoint.handle_bytes(raw)
```

## iceoryx2 Transport

Start a controller as an iceoryx2 JSON-RPC service:

```python
from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

server = Iox2JsonRpcServer(CalculatorController())
server.run_forever()
```

Discover available services and call a unique method:

```python
from iox2_jsonrpc.iceoryx import Iox2RpcRegistry

registry = Iox2RpcRegistry.discover_all()
registry.print_catalog()

response = registry.call_unique(
    "calculator.add",
    {"left": 2, "right": 5},
)
print(response.model_dump_json(indent=2))
```

If multiple services expose the same JSON-RPC method, call by endpoint instead:

```python
response = registry.call_endpoint(
    "mathService/calculator/rpc",
    "calculator.add",
    {"left": 2, "right": 5},
)
```

## Example

The repository includes an all-in-one camera example:

```bash
uv run python examples/camera.py local
```

To use the real iceoryx2 transport, run the server and client in separate
terminals:

```bash
uv run python examples/camera.py server
```

```bash
uv run python examples/camera.py client
```

The example controller exposes:

- `camera.open`
- `camera.close`
- `camera.status`
- `camera.capture`

## Package Map

```text
iox2_jsonrpc/models.py
  JSON-RPC and service descriptor Pydantic models

iox2_jsonrpc/controller.py
  Controller validation, method introspection, and schema descriptors

iox2_jsonrpc/endpoint.py
  In-process JSON-RPC request dispatch

iox2_jsonrpc/iceoryx.py
  iceoryx2 server, discovery registry, and client calls

examples/camera.py
  Local and iceoryx2 camera controller example

tests/test_controller_endpoint.py
  Unit tests for descriptors and in-process dispatch
```

## License

MIT
