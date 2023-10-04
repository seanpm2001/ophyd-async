import asyncio
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    Union,
)

from bluesky.protocols import (
    Asset,
    Configurable,
    Descriptor,
    Readable,
    Reading,
    Stageable,
    Triggerable,
    WritesExternalAssets,
)
from event_model import StreamDatum, StreamResource

from ophyd_async.core import Device
from ophyd_async.core._device._signal.signal import SignalR
from ophyd_async.core.async_status import AsyncStatus
from ophyd_async.core.utils import merge_gathered_dicts


class DetectorTrigger(Enum):
    #: Detector generates internal trigger for given rate
    internal = ()
    #: Expect a series of constant width external gate signals
    constant_gate = ()
    #: Expect a series of variable width external gate signals
    variable_gate = ()


class DetectorLogic(ABC):
    @abstractmethod
    def get_deadtime(self, exposure: float) -> float:
        """For a given exposure, how long should the time between exposures be"""

    @abstractmethod
    async def arm(
        self,
        trigger: DetectorTrigger = DetectorTrigger.internal,
        num: int = 0,
    ) -> AsyncStatus:
        """Arm the detector and return AsyncStatus that waits for num frames to be written"""

    @abstractmethod
    async def disarm(self):
        """Disarm the detector"""


class WriterLogic(ABC):
    @abstractmethod
    async def open(self, multiplier: int = 1) -> Dict[str, Descriptor]:
        """Open writer and wait for it to be ready for data.

        Args:
            multiplier: Each StreamDatum index corresponds to this many
                written exposures

        Returns:
            Output for ``describe()``
        """

    @abstractmethod
    async def wait_for_index(self, index: int):
        """Wait until a specific index is ready to be collected"""

    @abstractmethod
    async def get_indices_written(self) -> int:
        """Get the number of indices written"""

    @abstractmethod
    async def collect_stream_docs(
        self, indices_written: int
    ) -> AsyncIterator[Union[StreamResource, StreamDatum]]:
        """Create Stream docs up to given number written"""

    @abstractmethod
    async def close(self):
        """Close writer and wait for it to be finished"""


@dataclass
class DirectoryInfo:
    directory_path: str
    filename_prefix: str


class DirectoryProvider(Protocol):
    @abstractmethod
    def __call__(self) -> DirectoryInfo:
        """Get the current directory to write files into"""


class StaticDirectoryProvider(DirectoryProvider):
    def __init__(self, directory_path: str, filename_prefix: str) -> None:
        self._directory_info = DirectoryInfo(directory_path, filename_prefix)

    def __call__(self) -> DirectoryInfo:
        return self._directory_info


class NameProvider(Protocol):
    @abstractmethod
    def __call__(self) -> str:
        ...


class ShapeProvider(Protocol):
    @abstractmethod
    async def __call__(self) -> Sequence[int]:
        ...


class StreamingDetector(
    Device, Stageable, Configurable, Readable, Triggerable, WritesExternalAssets
):
    def __init__(
        self,
        detector_logic: DetectorLogic,
        writer_logic: WriterLogic,
        settings: Dict[str, Any],
        config_sigs: Sequence[SignalR],
        name: str = "",
        **plugins: Device,
    ) -> None:
        self.detector_logic = detector_logic
        self.writer_logic = writer_logic
        self._settings = settings
        self._config_sigs = config_sigs
        self._describe: Dict[str, Descriptor] = {}
        self.__dict__.update(plugins)
        super().__init__(name)

    @AsyncStatus.wrap
    async def stage(self) -> None:
        # Stop everything, get into a known state and open the stream"""
        await asyncio.gather(self.writer_logic.close(), self.detector_logic.disarm())
        await load_settings(self, self._settings)
        self._describe = await self.writer_logic.open()

    async def describe_configuration(self) -> Dict[str, Descriptor]:
        return await merge_gathered_dicts(sig.describe() for sig in self._config_sigs)

    async def read_configuration(self) -> Dict[str, Reading]:
        return await merge_gathered_dicts(sig.read() for sig in self._config_sigs)

    def describe(self) -> Dict[str, Descriptor]:
        return self._describe

    @AsyncStatus.wrap
    async def trigger(self) -> None:
        # Arm the detector, then wait for it to finish
        written_status = await self.detector_logic.arm(DetectorTrigger.internal, num=1)
        await written_status

    async def read(self) -> Dict[str, Reading]:
        # All data is in StreamResources, not Events, so nothing to output here
        return {}

    async def collect_asset_docs(self) -> Iterator[Asset]:
        indices_written = await self.writer_logic.get_indices_written()
        async for doc in self.writer_logic.collect_stream_docs(indices_written):
            yield doc

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        await self.writer_logic.close()

    async def pause(self) -> None:
        await self.detector_logic.disarm()

    async def resume(self) -> None:
        await self.writer_logic.reset_index()
