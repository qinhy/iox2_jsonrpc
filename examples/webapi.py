from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, Signature
from threading import RLock
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from iox2_jsonrpc.controller import describe_controller, iter_rpc_methods
from iox2_jsonrpc.endpoint import (
    ControllerRpcEndpoint,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
)
from iox2_jsonrpc.models import JsonRpcRequest, JsonRpcResponse, RpcMethodDescriptor, RpcModel

if TYPE_CHECKING:  # pragma: no cover - imported only by type checkers
    from fastapi import FastAPI
    from fastapi.routing import APIRouter


class ControllerRouteInfo(RpcModel):
    """HTTP routes exposed for one registered controller."""

    service: str
    controller: str
    rpc_endpoint: str
    schema_endpoint: str
    method_routes: list[str]


class ControllerApiInfo(RpcModel):
    """Summary returned by the generic controller HTTP API root."""

    controllers: list[ControllerRouteInfo]
    routes: list[str]


@dataclass
class RegisteredController:
    """Runtime registration for a typed JSON-RPC controller."""

    controller: Any
    service_name: str
    controller_name: str
    endpoint: ControllerRpcEndpoint
    lock: RLock


def _as_controller_list(controllers: Any | list[Any] | tuple[Any, ...]) -> list[Any]:
    if isinstance(controllers, tuple):
        return list(controllers)
    if isinstance(controllers, list):
        return controllers
    return [controllers]


def _controller_name(controller: Any) -> str:
    return str(getattr(controller, "controller_name"))


def _service_name(controller: Any) -> str:
    return str(getattr(controller, "service_name"))


def _register_controllers(
    controllers: Any | list[Any] | tuple[Any, ...],
    *,
    thread_safe: bool,
) -> dict[str, RegisteredController]:
    controller_list = _as_controller_list(controllers)

    if not controller_list:
        raise ValueError("At least one controller is required")

    registered: dict[str, RegisteredController] = {}

    for controller in controller_list:
        # ControllerRpcEndpoint and iter_rpc_methods already validate the controller shape.
        endpoint = ControllerRpcEndpoint(controller)
        controller_name = _controller_name(controller)

        if controller_name in registered:
            raise ValueError(f"Duplicate controller_name: {controller_name!r}")

        registered[controller_name] = RegisteredController(
            controller=controller,
            service_name=_service_name(controller),
            controller_name=controller_name,
            endpoint=endpoint,
            lock=RLock() if thread_safe else _NullLock(),
        )

    return registered


class _NullLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


def _model_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _descriptor_for_http(entry: RegisteredController) -> Any:
    controller = entry.controller_name
    return describe_controller(
        entry.controller,
        rpc_endpoint=f"POST /controllers/{controller}/rpc",
        schema_endpoint=f"GET /controllers/{controller}/schema",
    )


def _route_info(entry: RegisteredController) -> ControllerRouteInfo:
    controller = entry.controller_name
    method_routes: list[str] = []

    for binding in iter_rpc_methods(entry.controller):
        method_routes.extend(
            [
                f"GET /controllers/{controller}/methods/{binding.name}",
                f"POST /controllers/{controller}/methods/{binding.name}",
            ]
        )

    return ControllerRouteInfo(
        service=entry.service_name,
        controller=controller,
        rpc_endpoint=f"POST /controllers/{controller}/rpc",
        schema_endpoint=f"GET /controllers/{controller}/schema",
        method_routes=method_routes,
    )


def _normalized_request(
    request: JsonRpcRequest,
    *,
    entry: RegisteredController,
    allow_unprefixed_method: bool,
) -> JsonRpcRequest | JsonRpcResponse:
    expected_prefix = f"{entry.controller_name}."

    if request.method.startswith(expected_prefix):
        return request

    if "." not in request.method and allow_unprefixed_method:
        payload = request.model_dump(mode="json")
        payload["method"] = f"{entry.controller_name}.{request.method}"
        return JsonRpcRequest.model_validate(payload)

    return JsonRpcResponse.fail(
        id=request.id,
        code=METHOD_NOT_FOUND,
        message=f"Method {request.method!r} does not belong to controller {entry.controller_name!r}",
    )


def _jsonrpc_for_entry(
    entry: RegisteredController,
    request: JsonRpcRequest,
    *,
    allow_unprefixed_method: bool = False,
) -> JsonRpcResponse:
    normalized = _normalized_request(
        request,
        entry=entry,
        allow_unprefixed_method=allow_unprefixed_method,
    )

    if isinstance(normalized, JsonRpcResponse):
        return normalized

    with entry.lock:
        return entry.endpoint.handle(normalized)


def _jsonrpc_for_registered(
    registered: dict[str, RegisteredController],
    request: JsonRpcRequest,
) -> JsonRpcResponse:
    if "." in request.method:
        controller_name = request.method.split(".", 1)[0]
        entry = registered.get(controller_name)

        if entry is None:
            return JsonRpcResponse.fail(
                id=request.id,
                code=METHOD_NOT_FOUND,
                message=f"Unknown controller for method: {request.method}",
            )

        return _jsonrpc_for_entry(entry, request)

    if len(registered) == 1:
        entry = next(iter(registered.values()))
        return _jsonrpc_for_entry(entry, request, allow_unprefixed_method=True)

    return JsonRpcResponse.fail(
        id=request.id,
        code=METHOD_NOT_FOUND,
        message="Method must include a controller prefix, for example 'camera.status'",
    )


def _raise_for_jsonrpc_error(response: JsonRpcResponse) -> None:
    if response.error is None:
        return

    status_code = 400
    if response.error.code == METHOD_NOT_FOUND:
        status_code = 404
    elif response.error.code == INVALID_PARAMS:
        status_code = 422
    elif response.error.code == INTERNAL_ERROR:
        status_code = 500
    elif response.error.code == PARSE_ERROR:
        status_code = 400

    # FastAPI/Starlette must be imported lazily so this module stays optional.
    from fastapi import HTTPException

    raise HTTPException(
        status_code=status_code,
        detail=response.error.model_dump(mode="json"),
    )


def _call_direct_method(
    entry: RegisteredController,
    method_name: str,
    params: BaseModel,
) -> Any:
    request = JsonRpcRequest(
        id=1,
        method=f"{entry.controller_name}.{method_name}",
        params=params.model_dump(mode="json"),
    )
    response = _jsonrpc_for_entry(entry, request)
    _raise_for_jsonrpc_error(response)
    return response.result


def _make_post_method_endpoint(entry: RegisteredController, binding: Any) -> Any:
    from fastapi import Body

    async def endpoint(params: BaseModel) -> Any:
        return _call_direct_method(entry, binding.name, params)

    endpoint.__name__ = f"{entry.controller_name}_{binding.name}_post"
    endpoint.__doc__ = f"Call JSON-RPC method {entry.controller_name}.{binding.name}."
    endpoint.__signature__ = Signature(
        parameters=[
            Parameter(
                "params",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=binding.params_model,
                default=Body(default_factory=binding.params_model),
            )
        ],
        return_annotation=binding.result_model,
    )
    return endpoint


def _make_get_method_endpoint(entry: RegisteredController, binding: Any) -> Any:
    from fastapi import HTTPException, Request

    async def endpoint(request: Any) -> Any:
        try:
            params = binding.params_model.model_validate(dict(request.query_params))
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

        return _call_direct_method(entry, binding.name, params)

    endpoint.__name__ = f"{entry.controller_name}_{binding.name}_get"
    endpoint.__doc__ = f"Call JSON-RPC method {entry.controller_name}.{binding.name} with query parameters."
    endpoint.__signature__ = Signature(
        parameters=[
            Parameter(
                "request",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Request,
            )
        ],
        return_annotation=binding.result_model,
    )
    return endpoint


def _add_controller_routes(
    router: "APIRouter",
    entry: RegisteredController,
    *,
    include_short_routes: bool,
    tags: list[str],
) -> None:
    from fastapi import HTTPException

    controller = entry.controller_name

    def controller_schema() -> Any:
        return _descriptor_for_http(entry)

    def controller_methods() -> list[RpcMethodDescriptor]:
        return _descriptor_for_http(entry).methods

    def controller_rpc(request: JsonRpcRequest) -> JsonRpcResponse:
        return _jsonrpc_for_entry(entry, request, allow_unprefixed_method=True)

    router.add_api_route(
        f"/controllers/{controller}/schema",
        controller_schema,
        methods=["GET"],
        response_model_exclude_none=True,
        tags=tags,
        summary=f"Describe {controller}",
    )
    router.add_api_route(
        f"/controllers/{controller}/methods",
        controller_methods,
        methods=["GET"],
        response_model=list[RpcMethodDescriptor],
        response_model_exclude_none=True,
        tags=tags,
        summary=f"List {controller} methods",
    )
    router.add_api_route(
        f"/controllers/{controller}/rpc",
        controller_rpc,
        methods=["POST"],
        response_model=JsonRpcResponse,
        response_model_exclude_none=True,
        tags=tags,
        summary=f"JSON-RPC bridge for {controller}",
    )

    if include_short_routes:
        router.add_api_route(
            f"/{controller}/schema",
            controller_schema,
            methods=["GET"],
            include_in_schema=False,
        )
        router.add_api_route(
            f"/{controller}/rpc",
            controller_rpc,
            methods=["POST"],
            include_in_schema=False,
        )

    for binding in iter_rpc_methods(entry.controller):
        post_endpoint = _make_post_method_endpoint(entry, binding)
        get_endpoint = _make_get_method_endpoint(entry, binding)

        method_path = f"/controllers/{controller}/methods/{binding.name}"
        router.add_api_route(
            method_path,
            post_endpoint,
            methods=["POST"],
            response_model=binding.result_model,
            response_model_exclude_none=True,
            tags=tags,
            summary=f"Call {controller}.{binding.name}",
        )
        router.add_api_route(
            method_path,
            get_endpoint,
            methods=["GET"],
            response_model=binding.result_model,
            response_model_exclude_none=True,
            tags=tags,
            summary=f"Call {controller}.{binding.name} from query parameters",
        )

        if include_short_routes:
            short_path = f"/{controller}/{binding.name}"
            router.add_api_route(
                short_path,
                post_endpoint,
                methods=["POST"],
                response_model=binding.result_model,
                response_model_exclude_none=True,
                tags=tags,
                include_in_schema=False,
            )
            router.add_api_route(
                short_path,
                get_endpoint,
                methods=["GET"],
                response_model=binding.result_model,
                response_model_exclude_none=True,
                tags=tags,
                include_in_schema=False,
            )


def create_controller_fastapi_router(
    controllers: Any | list[Any] | tuple[Any, ...],
    *,
    include_short_routes: bool = True,
    thread_safe: bool = True,
    tags: list[str] | None = None,
) -> "APIRouter":
    """Create a reusable FastAPI router for one or more typed RPC controllers.

    Any object accepted by ControllerRpcEndpoint is accepted here. The router exposes:

    - POST /rpc for JSON-RPC requests, dispatched by the method prefix.
    - GET /controllers for the registered controller list.
    - GET /controllers/{controller}/schema for a controller descriptor.
    - POST /controllers/{controller}/rpc for controller-scoped JSON-RPC.
    - GET/POST /controllers/{controller}/methods/{method} for direct HTTP control.

    When include_short_routes is true, compatibility aliases such as /camera/status
    are also registered but hidden from OpenAPI.
    """

    from fastapi import APIRouter

    registered = _register_controllers(controllers, thread_safe=thread_safe)
    router = APIRouter()
    route_tags = tags or ["controllers"]

    def root() -> ControllerApiInfo:
        return ControllerApiInfo(
            controllers=[_route_info(entry) for entry in registered.values()],
            routes=[
                "GET /controllers",
                "GET /describe",
                "POST /rpc",
                "GET /controllers/{controller}/schema",
                "GET /controllers/{controller}/methods",
                "POST /controllers/{controller}/rpc",
                "GET /controllers/{controller}/methods/{method}",
                "POST /controllers/{controller}/methods/{method}",
            ],
        )

    def controllers_list() -> list[ControllerRouteInfo]:
        return [_route_info(entry) for entry in registered.values()]

    def describe_all() -> dict[str, Any]:
        return {
            name: _model_json(_descriptor_for_http(entry))
            for name, entry in registered.items()
        }

    def rpc(request: JsonRpcRequest) -> JsonRpcResponse:
        return _jsonrpc_for_registered(registered, request)

    router.add_api_route(
        "/",
        root,
        methods=["GET"],
        response_model=ControllerApiInfo,
        response_model_exclude_none=True,
        tags=route_tags,
        summary="Controller API info",
    )
    router.add_api_route(
        "/controllers",
        controllers_list,
        methods=["GET"],
        response_model=list[ControllerRouteInfo],
        response_model_exclude_none=True,
        tags=route_tags,
        summary="List controllers",
    )
    router.add_api_route(
        "/describe",
        describe_all,
        methods=["GET"],
        response_model_exclude_none=True,
        tags=route_tags,
        summary="Describe all controllers",
    )
    router.add_api_route(
        "/rpc",
        rpc,
        methods=["POST"],
        response_model=JsonRpcResponse,
        response_model_exclude_none=True,
        tags=route_tags,
        summary="JSON-RPC bridge",
    )

    for entry in registered.values():
        _add_controller_routes(
            router,
            entry,
            include_short_routes=include_short_routes,
            tags=route_tags,
        )

    return router


def create_controller_fastapi_app(
    controllers: Any | list[Any] | tuple[Any, ...],
    *,
    title: str = "Controller JSON-RPC API",
    version: str = "1.0.0",
    description: str = "Generic FastAPI adapter for typed JSON-RPC controllers.",
    include_short_routes: bool = True,
    thread_safe: bool = True,
    tags: list[str] | None = None,
) -> "FastAPI":
    """Create a complete FastAPI app for one or more typed RPC controllers."""

    from fastapi import FastAPI

    app = FastAPI(title=title, version=version, description=description)

    @app.get("/health", tags=["health"])
    def health() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(
        create_controller_fastapi_router(
            controllers,
            include_short_routes=include_short_routes,
            thread_safe=thread_safe,
            tags=tags,
        )
    )
    return app
