# iox2-jsonrpc-simple

Tiny JSON-RPC 2.0 over **iceoryx2**, with a central FastAPI control gateway.

This repo is intentionally simple:

- `uv` project
- OOP-style Python code
- no pytest dependency by default
- no TCP fallback
- no in-memory fallback
- no bundled iceoryx2 build
- internal communication is only iceoryx2 request-response
- FastAPI is only the external HTTP control plane

Important: this repo does **not** install or build `iceoryx2` for you. It expects this to work in your environment:

```bash
python -c "import iceoryx2; print('iceoryx2 OK')"
```

## Install runtime dependencies

```bash
uv sync
```

This installs only the Python runtime dependencies from `pyproject.toml`:

- `fastapi`
- `pydantic`
- `uvicorn`

## Configure services

Edit `config/services.toml`:

```toml
[services.serverA]
iceoryx2_service = "jsonrpc/serverA"
timeout_seconds = 5.0

[services.serverB]
iceoryx2_service = "jsonrpc/serverB"
timeout_seconds = 5.0
```

The FastAPI gateway uses these names to route HTTP requests to iceoryx2 services.

## Run the real demo

Use three terminals.

### Terminal 1: serverA

```bash
uv run iox2-server-a
```

Methods:

- `pipeline.start`
- `pipeline.stop`
- `pipeline.status`
- `rpc.health`

### Terminal 2: serverB

```bash
uv run iox2-server-b
```

Methods:

- `camera.open`
- `camera.close`
- `camera.status`
- `camera.capture`
- `rpc.health`

### Terminal 3: FastAPI gateway

```bash
uv run iox2-gateway --host 127.0.0.1 --port 8000
```

## Call serverA through FastAPI

```bash
curl -X POST http://127.0.0.1:8000/serverA/rpc \
  -H "content-type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"pipeline.status","params":{}}'
```

```bash
curl -X POST http://127.0.0.1:8000/serverA/rpc \
  -H "content-type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"pipeline.start","params":{"profile":"demo"}}'
```

## Call serverB through FastAPI

```bash
curl -X POST http://127.0.0.1:8000/serverB/rpc \
  -H "content-type: application/json" \
  -d '{"jsonrpc":"2.0","id":3,"method":"camera.capture","params":{"exposure_ms":25}}'
```

Alternative route style:

```bash
curl -X POST http://127.0.0.1:8000/rpc/serverB \
  -H "content-type: application/json" \
  -d '{"jsonrpc":"2.0","id":4,"method":"camera.status","params":{}}'
```

## Gateway metadata

```bash
curl http://127.0.0.1:8000/services
curl http://127.0.0.1:8000/services/serverA/health
```

## Architecture

```text
HTTP client
   |
   v
FastAPI gateway
   |
   | iceoryx2 request-response only
   v
serverA / serverB
   |
   v
JSON-RPC method registry
```

## Code map

```text
src/iox2_jsonrpc/core.py
  JSON-RPC parser, response builder, method registry, Pydantic validation

src/iox2_jsonrpc/packet.py
  JsonRpcPacket ctypes.Structure and encode/decode helpers

src/iox2_jsonrpc/iox2_transport.py
  Iceoryx2JsonRpcServer, Iceoryx2JsonRpcClient, client pool

src/iox2_jsonrpc/gateway.py
  FastAPI gateway routes

src/iox2_jsonrpc/programs/server_a.py
  pipeline microservice

src/iox2_jsonrpc/programs/server_b.py
  camera microservice

src/iox2_jsonrpc/programs/gateway.py
  central FastAPI gateway program
```

## Why the packet is fixed-size

The JSON-RPC payload is encoded as UTF-8 JSON bytes inside a fixed-size `ctypes.Structure`:

```python
class JsonRpcPacket(ctypes.Structure):
    _fields_ = [
        ("correlation_id", ctypes.c_uint64),
        ("payload_len", ctypes.c_uint32),
        ("payload", ctypes.c_uint8 * 65536),
    ]
```

That keeps the type shared-memory-friendly and avoids passing Python dict/list/string objects directly through iceoryx2.

## Add a new service

1. Add it to `config/services.toml`.
2. Create a new `programs/server_x.py`.
3. Register methods with `MethodRegistry`.
4. Run that service in its own terminal.
5. Call it through `POST /serviceName/rpc`.
