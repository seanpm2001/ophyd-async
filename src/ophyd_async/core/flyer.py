import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
from bluesky.protocols import (
    Asset,
    Collectable,
    Descriptor,
    Flyable,
    Movable,
    Pausable,
    Reading,
    Stageable,
    WritesExternalAssets,
)
from scanspec.core import Frames, Path
from scanspec.specs import DURATION, Spec

from .async_status import AsyncStatus
from .detector import DetectorControl, DetectorTrigger, DetectorWriter
from .device import Device
from .signal import SignalR
from .utils import gather_list, merge_gathered_dicts

T = TypeVar("T")


@dataclass(frozen=True)
class TriggerInfo:
    #: Number of triggers that will be sent
    num: int
    #: Sort of triggers that will be sent
    trigger: DetectorTrigger
    #: What is the minimum deadtime between triggers
    deadtime: float
    #: What is the maximum high time of the triggers
    livetime: float


class DetectorGroupLogic(ABC):
    # Read multipliers here, exposure is set in the plan
    @abstractmethod
    async def open(self) -> Dict[str, Descriptor]:
        """Open all writers, wait for them to be open and return their descriptors"""

    @abstractmethod
    async def ensure_armed(self, trigger_info: TriggerInfo) -> AsyncStatus:
        """Ensure the detectors are armed, return AsyncStatus that waits for disarm."""

    @abstractmethod
    async def collect_asset_docs(self) -> AsyncIterator[Asset]:
        """Collect asset docs from all writers"""

    @abstractmethod
    async def wait_for_index(self, index: int):
        """Wait until a specific index is ready to be collected"""

    @abstractmethod
    async def disarm(self):
        """Disarm detectors"""

    @abstractmethod
    async def close(self):
        """Close all writers and wait for them to be closed"""


class SameTriggerDetectorGroupLogic(DetectorGroupLogic):
    def __init__(
        self,
        detector_logics: Sequence[DetectorControl],
        writer_logics: Sequence[DetectorWriter],
    ) -> None:
        self.detector_logics = detector_logics
        self.writer_logics = writer_logics
        self._arm_status: Optional[AsyncStatus] = None
        self._trigger_info: Optional[TriggerInfo] = None

    async def open(self) -> Dict[str, Descriptor]:
        return merge_gathered_dicts(wl.open() for wl in self.writer_logics)

    async def ensure_armed(self, trigger_info: TriggerInfo) -> AsyncStatus:
        if (
            not self._arm_status
            or self._arm_status.done
            or trigger_info != self._trigger_info
        ):
            # We need to re-arm
            await gather_list(dl.disarm() for dl in self.detector_logics)
            for dl in self.detector_logics:
                required = dl.get_deadtime(trigger_info.livetime)
                assert required > trigger_info.deadtime, (
                    f"Detector {dl} needs at least {required}s deadtime, "
                    "but trigger logic provides only {trigger_info.deadtime}s"
                )
            statuses = await gather_list(
                dl.arm(trigger=trigger_info.trigger, exposure=trigger_info.livetime)
                for dl in self.detector_logics
            )
            self._arm_status = AsyncStatus(gather_list(statuses))
            self._trigger_info = trigger_info
        return self._arm_status

    async def collect_asset_docs(self) -> AsyncIterator[Asset]:
        indices_written = min(
            gather_list(wl.get_indices_written() for wl in self.writer_logics)
        )
        for wl in self.writer_logics:
            async for doc in wl.collect_stream_docs(indices_written):
                yield doc

    @abstractmethod
    async def wait_for_index(self, index: int):
        await gather_list(wl.wait_for_index(index) for wl in self.writer_logics)

    @abstractmethod
    async def disarm(self):
        await gather_list(dl.disarm() for dl in self.detector_logics)

    @abstractmethod
    async def close(self):
        await gather_list(wl.close() for wl in self.writer_logics)


class TriggerLogic(ABC, Generic[T]):
    @abstractmethod
    def trigger_info(self, value: T) -> TriggerInfo:
        """Return info about triggers that will be produced for a given value"""

    @abstractmethod
    async def prepare(self, value: T):
        """Move to the start of the flyscan"""

    @abstractmethod
    async def start(self):
        """Start the flyscan"""

    @abstractmethod
    async def stop(self):
        """Stop flying and wait everything to be stopped"""


class HardwareTriggeredFlyable(
    Device, Movable, Stageable, Flyable, Collectable, WritesExternalAssets, Generic[T]
):
    def __init__(
        self,
        detector_group_logic: DetectorGroupLogic,
        trigger_logic: TriggerLogic[T],
        configuration_signals: Sequence[SignalR],
        name: str = "",
    ):
        self._detector_group_logic = detector_group_logic
        self._trigger_logic = trigger_logic
        self._configuration_signals = tuple(configuration_signals)
        self._describe: Dict[str, Descriptor] = {}
        self._det_status: Optional[AsyncStatus] = None
        self._watchers: List[Callable] = []
        self._fly_status: Optional[AsyncStatus] = None
        self._fly_start = 0.0
        self._offset = 0  # Add this to index to get frame number
        self._current_frame = 0  # The current frame we are on
        self._last_frame = 0  # The last frame that will be emitted
        super().__init__(name=name)

    @AsyncStatus.wrap
    async def stage(self) -> None:
        await self.unstage()
        await self._detector_group_logic.open()
        self._offset = 0
        self._current_frame = 0

    @AsyncStatus.wrap
    async def set(self, value: T):
        """Arm detectors and setup trajectories"""
        # index + offset = current_frame, but starting a new scan so want it to be 0
        # so subtract current_frame from both sides
        self._offset -= self._current_frame
        self._current_frame = 0
        await self._prepare(value)

    async def _prepare(self, value: T):
        trigger_info = self._trigger_logic.trigger_info(value)
        # Move to start and setup the flyscan, and arm dets in parallel
        self._det_status, num_frames = await asyncio.gather(
            self._detector_group_logic.ensure_armed(trigger_info),
            self._trigger_logic.prepare(value),
        )
        self._last_frame = self._current_frame + num_frames

    async def describe_configuration(self) -> Dict[str, Descriptor]:
        return await merge_gathered_dicts(
            [sig.describe() for sig in self._configuration_signals]
        )

    async def read_configuration(self) -> Dict[str, Reading]:
        return await merge_gathered_dicts(
            [sig.read() for sig in self._configuration_signals]
        )

    async def describe_collect(self) -> Dict[str, Descriptor]:
        return self._describe

    @AsyncStatus.wrap
    async def kickoff(self) -> None:
        self._watchers = []
        self._fly_status = AsyncStatus(self._fly(), self._watchers)
        self._fly_start = time.monotonic()

    async def _fly(self) -> None:
        await self._trigger_logic.start()
        # Wait for all detectors to have written up to a particular frame
        await self._detector_group_logic.wait_for_index(self._last_frame + self._offset)

    async def collect_asset_docs(self) -> AsyncIterator[Asset]:
        current_frame = self._current_frame
        async for name, doc in self._detector_group_logic.collect_asset_docs():
            if name == "stream_datum":
                current_frame = doc["indices"]["stop"] + self._offset
            yield name, doc
        if current_frame != self._current_frame:
            self._current_frame = current_frame
            for watcher in self._watchers:
                watcher(
                    name=self.name,
                    current=current_frame,
                    initial=0,
                    target=self._last_frame,
                    unit="",
                    precision=0,
                    time_elapsed=time.monotonic() - self._fly_start,
                )

    def complete(self) -> AsyncStatus:
        assert self._fly_status, "Kickoff not run"
        return self._fly_status

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        await asyncio.gather(
            self._trigger_logic.stop(),
            self._detector_group_logic.close(),
            self._detector_group_logic.disarm(),
        )


ScanAxis = Union[Device, Literal["DURATION"]]


class ScanSpecFlyable(HardwareTriggeredFlyable[Path[ScanAxis]], Pausable):
    _spec: Optional[Spec] = None
    _frames: Sequence[Frames] = ()

    @AsyncStatus.wrap
    async def set(self, value: Spec[ScanAxis]):
        """Arm detectors and setup trajectories"""
        if value != self._spec:
            self._spec = value
            self._frames = value.calculate()
        super().set(Path(self._frames))

    async def pause(self):
        assert self._fly_status, "Kickoff not run"
        self._fly_status.task.cancel()
        await self.unstage()
        await self._detector_group_logic.open()
        # Next frame will have index 0, but will be self._current_frame
        self._offset = self._current_frame
        await self._prepare(Path(self._frames, start=self._current_frame))

    async def resume(self):
        assert self._fly_status, "Kickoff not run"
        assert self._fly_status.task.cancelled(), "You didn't call pause"
        self._fly_status.task = asyncio.create_task(self._fly())


def get_duration(frames: List[Frames]) -> Optional[float]:
    for fs in frames:
        if DURATION in fs.axes():
            durations = fs.midpoints[DURATION]
            first_duration = durations[0]
            if np.all(durations == first_duration):
                # Constant duration, return it
                return first_duration
            else:
                return None
    raise ValueError("Duration not specified in Spec")
