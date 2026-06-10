from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

import pytest


@pytest.fixture()
def helpers(install_fake_iox2: Any) -> Any:
    install_fake_iox2()
    import iox2_jsonrpc.iceoryx_runtime as runtime

    importlib.reload(runtime)

    import iox2_jsonrpc.iceoryx_helpers as helper_module

    return importlib.reload(helper_module)


class _FakeSlice:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __getitem__(self, index: int) -> int:
        return self.values[index]

    def len(self) -> int:
        return len(self.values)


class _FakeUninitSlice:
    def __init__(self, length: int) -> None:
        self.buffer = [0] * length
        self.initialized = object()

    def payload(self) -> list[int]:
        return self.buffer

    def assume_init(self) -> object:
        return self.initialized


def test_slice_to_bytes_reads_each_payload_byte(helpers: Any) -> None:
    assert helpers.slice_to_bytes(_FakeSlice([65, 66, 255])) == b"AB\xff"


def test_write_bytes_to_uninit_slice_copies_payload_and_assumes_init(helpers: Any) -> None:
    uninit = _FakeUninitSlice(3)

    initialized = helpers.write_bytes_to_uninit_slice(uninit, b"rpc")

    assert initialized is uninit.initialized
    assert uninit.buffer == [114, 112, 99]


def test_to_str_prefers_to_string_then_as_str(helpers: Any) -> None:
    class WithToString:
        def to_string(self) -> str:
            return "from-to-string"

    class WithCallableAsStr:
        def as_str(self) -> str:
            return "from-as-str"

    class WithValueAsStr:
        as_str = "value-as-str"

    assert helpers.to_str(lambda: WithToString()) == "from-to-string"
    assert helpers.to_str(WithCallableAsStr()) == "from-as-str"
    assert helpers.to_str(WithValueAsStr()) == "value-as-str"
    assert helpers.to_str(123) == "123"


def test_make_attributes_and_verifier_record_all_pairs(helpers: Any) -> None:
    attributes = helpers.make_attributes({"rpc.protocol": "jsonrpc-2.0", "rpc.kind": "rpc"})
    verifier = helpers.make_attribute_verifier({"rpc.service": "svc"})

    assert attributes.values == [("rpc.protocol", "jsonrpc-2.0"), ("rpc.kind", "rpc")]
    assert verifier.values == [("rpc.service", "svc")]


def test_attribute_set_to_dict_handles_callable_accessors(helpers: Any) -> None:
    @dataclass
    class Attribute:
        key: Any
        value: Any

    @dataclass
    class AttributeSet:
        items: list[Attribute]

        def values(self) -> list[Attribute]:
            return self.items

    attrs = AttributeSet(
        [
            Attribute(key=lambda: "rpc.service", value=lambda: "camera"),
            Attribute(key="rpc.controller", value="device"),
        ]
    )

    assert helpers.attribute_set_to_dict(attrs) == {
        "rpc.service": "camera",
        "rpc.controller": "device",
    }


def test_best_effort_cleanup_uses_node_fallback_signature(helpers: Any) -> None:
    @dataclass
    class Node:
        config: str = "node-config"
        cleanup_calls: list[tuple[Any, ...]] = field(default_factory=list)

        def try_cleanup_dead_nodes(self, *args: Any) -> None:
            self.cleanup_calls.append(args)
            if not args:
                raise TypeError("alternate signature required")

    node = Node()

    helpers.best_effort_cleanup_dead_nodes(node=node)

    assert node.cleanup_calls == [(), (helpers.iox2.ServiceType.Ipc, "node-config")]


def test_create_server_retries_after_dead_server_limit(helpers: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers.time, "sleep", lambda _seconds: None)

    @dataclass
    class Service:
        create_calls: int = 0
        cleanup_calls: int = 0

        def server_builder(self) -> Service:
            return self

        def initial_max_slice_len(self, _length: int) -> Service:
            return self

        def allocation_strategy(self, _strategy: object) -> Service:
            return self

        def create(self) -> object:
            self.create_calls += 1
            if self.create_calls == 1:
                raise RuntimeError("ExceedsMaxSupportedServers")
            return "server"

        def try_cleanup_dead_nodes(self) -> None:
            self.cleanup_calls += 1

    service = Service()

    server = helpers.create_server_with_dead_node_cleanup(
        service,
        initial_max_slice_len=128,
        service_name="svc/rpc",
        max_attempts=2,
    )

    assert server == "server"
    assert service.create_calls == 2
    assert service.cleanup_calls == 1
