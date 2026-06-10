from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class FakeDuration:
    @staticmethod
    def from_millis(value: int) -> int:
        return value


class FakeAllocationStrategy:
    PowerOfTwo = object()


class FakeServiceType:
    Ipc = object()


class FakeMessagingPattern:
    RequestResponse = object()


class FakeLogLevel:
    Info = object()


class FakeSliceType:
    def __class_getitem__(cls, _item: object) -> type[FakeSliceType]:
        return cls


class FakeNamedValue:
    def __init__(self, value: str) -> None:
        self.value = value

    @classmethod
    def new(cls, value: str) -> FakeNamedValue:
        return cls(value)


class FakeAttributeSpecifier:
    def __init__(self) -> None:
        self.values: list[tuple[str, str]] = []

    @classmethod
    def new(cls) -> FakeAttributeSpecifier:
        return cls()

    def define(self, key: FakeNamedValue, value: FakeNamedValue) -> FakeAttributeSpecifier:
        self.values.append((key.value, value.value))
        return self


class FakeAttributeVerifier:
    def __init__(self) -> None:
        self.values: list[tuple[str, str]] = []

    @classmethod
    def new(cls) -> FakeAttributeVerifier:
        return cls()

    def require(self, key: FakeNamedValue, value: FakeNamedValue) -> FakeAttributeVerifier:
        self.values.append((key.value, value.value))
        return self


@pytest.fixture()
def install_fake_iox2(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install() -> types.SimpleNamespace:
        fake_iox2 = types.SimpleNamespace(
            AllocationStrategy=FakeAllocationStrategy,
            AttributeKey=FakeNamedValue,
            AttributeSpecifier=FakeAttributeSpecifier,
            AttributeValue=FakeNamedValue,
            AttributeVerifier=FakeAttributeVerifier,
            Duration=FakeDuration,
            LogLevel=FakeLogLevel,
            MessagingPattern=FakeMessagingPattern,
            NodeBuilder=types.SimpleNamespace(new=lambda: None),
            Service=types.SimpleNamespace(list=lambda *_args: []),
            ServiceName=FakeNamedValue,
            ServiceType=FakeServiceType,
            Slice=FakeSliceType,
            config=types.SimpleNamespace(global_config=lambda: object()),
            set_log_level_from_env_or=lambda _level: None,
        )
        monkeypatch.setitem(sys.modules, "iceoryx2", fake_iox2)
        return fake_iox2

    return install
