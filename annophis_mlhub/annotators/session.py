from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from annophis_mlhub.models import WsInputUnit, WsOutputUnit


@runtime_checkable
class StreamSession(Protocol):
    """Protocol for annotator streaming sessions."""

    async def send(self, unit: WsInputUnit) -> None: ...

    def __aiter__(self) -> AsyncIterator[WsOutputUnit]: ...
