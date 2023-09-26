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
    async def get_shape(self) -> Sequence[int]:
        """Get the shape of the detector, slowest moving dim first"""

    @abstractmethod
    async def get_deadtime(self, exposure: float) -> float:
        """For a given exposure, how long should the time between frames be

        This may set PVs and will only be used when disarmed
        """

    @abstractmethod
    async def arm(
        self,
        trigger: DetectorTrigger = DetectorTrigger.internal,
        num: int = 0,
        exposure: Optional[float] = None,
    ) -> AsyncStatus:
        """Arm the detector and return AsyncStatus that waits for num frames to be written"""

    @abstractmethod
    async def disarm(self):
        """Disarm the detector"""


class StreamLogic(ABC):
    @abstractmethod
    async def open(self):
        """Open stream and wait for it to be ready for frames"""

    @abstractmethod
    async def describe_datasets(
        self, name: str, detector_shape: Sequence[int], outer_shape: Sequence[int] = ()
    ) -> Dict[str, Descriptor]:
        """Describe the datasets this will create"""

    @abstractmethod
    async def frames_written(self) -> int:
        """Get the number of frames written"""

    @abstractmethod
    async def collect_stream_docs(
        self, frames_written: int
    ) -> AsyncIterator[Union[StreamResource, StreamDatum]]:
        """Create Stream docs up to given number written"""

    @abstractmethod
    async def close(self):
        """Close stream and wait for it to be finished"""


@dataclass
class DirectoryInfo:
    directory_path: str
    filename_prefix: str


class DirectoryProvider(Protocol):
    @abstractproperty
    def directory_info(self) -> DirectoryInfo:
        """Get the current directory to write files into"""


class StaticDirectoryProvider(DirectoryProvider):
    def __init__(self, directory_path: str, filename_prefix: str) -> None:
        self.directory_info = DirectoryInfo(directory_path, filename_prefix)


class StreamDetector(
    Device, Stageable, Configurable, Readable, Triggerable, WritesExternalAssets
):
    def __init__(
        self,
        detector_logic: DetectorLogic,
        stream_logic: StreamLogic,
        settings: Dict[str, Any],
        config_sigs: Sequence[SignalR],
        name: str = "",
        **plugins: Device,
    ) -> None:
        self.detector_logic = detector_logic
        self.stream_logic = stream_logic
        self._settings = settings
        self._config_sigs = config_sigs
        self.__dict__.update(plugins)
        super().__init__(name)

    @AsyncStatus.wrap
    async def stage(self) -> None:
        # Stop everything, get into a known state and open the stream"""
        await asyncio.gather(self.stream_logic.close(), self.detector_logic.disarm())
        await load_settings(self, self._settings)
        await self.stream_logic.open()

    async def describe_configuration(self) -> Dict[str, Descriptor]:
        return await merge_gathered_dicts(sig.describe() for sig in self._config_sigs)

    async def read_configuration(self) -> Dict[str, Reading]:
        return await merge_gathered_dicts(sig.read() for sig in self._config_sigs)

    async def describe(self) -> Dict[str, Descriptor]:
        # Describe the datasets that will be written
        detector_shape = await self.detector_logic.get_shape()
        return await self.stream_logic.describe_datasets(self.name, detector_shape)

    @AsyncStatus.wrap
    async def trigger(self) -> None:
        # Arm the detector, then wait for it to finish
        written_status = await self.detector_logic.arm(DetectorTrigger.internal, num=1)
        await written_status

    async def read(self) -> Dict[str, Reading]:
        # All data is in StreamResources, not Events, so nothing to output here
        return {}

    async def collect_asset_docs(self) -> Iterator[Asset]:
        frames_written = await self.stream_logic.frames_written()
        async for doc in self.stream_logic.collect_stream_docs(frames_written):
            yield doc

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        await self.stream_logic.close()
