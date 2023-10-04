import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np
from annotated_types import T
from bluesky import Msg
from bluesky.protocols import (
    Asset,
    Collectable,
    Descriptor,
    Flyable,
    Movable,
    PartialEvent,
    Reading,
    Stageable,
    SyncOrAsync,
    WritesExternalAssets,
)
from scanspec.core import Frames, Path
from scanspec.specs import DURATION, Line, Repeat, Spec, Static

from ophyd_async.core import Device, DeviceCollector, StandardReadable, T
from ophyd_async.core._device._signal.signal import SignalR, wait_for_value
from ophyd_async.core.async_status import AsyncStatus
from ophyd_async.core.utils import gather_list, merge_gathered_dicts
from ophyd_async.detector import (
    DetectorLogic,
    DetectorTrigger,
    StaticDirectoryProvider,
    StreamingDetector,
    WriterLogic,
)
from ophyd_async.epics.areadetector import (
    NDFileHDF,
    NDPluginStats,
    ad_driver,
    hdf_logic,
    pilatus,
)
from ophyd_async.epics.signal import epics_signal_r, epics_signal_rw
from ophyd_async.panda.panda import PandA, PcapBlock, SeqBlock, build_seq_table

dp = StaticDirectoryProvider("/dls/p45/data/cmxxx/i22-yyy", "i22-yyy-")


def hdf_stats_pilatus(prefix: str, settings_path: str):
    drv = pilatus.ADPilatus(prefix + "DRV:")
    hdf = NDFileHDF(prefix + "HDF:")
    det = StreamingDetector(
        detector_logic=pilatus.PilatusLogic(drv),
        writer_logic=hdf_logic.ADHDFLogic(
            hdf,
            directory_provider=dp,
            name_provider=lambda: det.name,
            shape_provider=ad_driver.ADDriverShapeProvider(drv),
            sum="NDStatsSum",
        ),
        settings=settings_from_yaml(settings_path),
        config_sigs=[drv.acquire_time, drv.acquire_period],
        drv=drv,
        stats=NDPluginStats(prefix + "STATS:"),
        hdf=hdf,
    )
    return det


class Linkam(StandardReadable):
    def __init__(self, prefix: str, name: str = "") -> None:
        self.setpoint = epics_signal_rw(float, prefix + "RAMP:LIMIT:SET")
        self.readback = epics_signal_r(float, prefix + "TEMP")
        self.ramp_rate = epics_signal_rw(float, prefix + "RAMP:RATE:SET")
        self.set_readable_signals(config=[self.ramp_rate], read=[self.readback])
        super().__init__(name)

    def set_name(self, name: str):
        super().set_name(name)
        # Readback should be named the same as its parent in read()
        self.readback.set_name(name)

    # TODO: add set logic


with DeviceCollector():
    saxs = hdf_stats_pilatus("BL22I-EA-DET-01:", "/path/to/saxs_settings.yaml")
    waxs = hdf_stats_pilatus("BL22I-EA-DET-01:", "/path/to/waxs_settings.yaml")
    panda = PandA("BL22I-MO-PANDA-01:")
    linkam = Linkam("BL22I-EA-TEMPC-01:")


def linkam_fly_plan(
    target_temp: float,
    ramp_rate: float,
    num_frames: int,
    exposure: float,
    period: float,
    md: Optional[Dict[str, Any]] = None,
):
    """Fly the linkam, taking timed frames perodically.

    Ramp to target_temp at ramp_rate, then every period while not at target,
    expose num_frames.
    """
    _md = {
        "hw_spec": hw_spec.serialize(),
        "hw_dets": [det.name for det in flyer.dets],
        "sw_spec": sw_spec.serialize(),
        "sw_dets": [det.name for det in sw_dets],
        "hints": {},
    }
    _md.update(md or {})

    @bpp.stage_decorator([flyer] + list(sw_dets))
    @bpp.run_decorator(md=_md)
    def inner_linkam_fly_plan():
        yield from bps.abs_set(linkam.ramp_rate, ramp_rate)
        linkam_group = group_uuid("linkam")
        yield from bps.abs_set(linkam, target_temp, group=linkam_group, wait=False)
        done = False
        while not done:
            yield from take_timed_frames(exposure, num_frames)
            try:
                yield from bps.wait(group=linkam_group, timeout=period)
            except TimeoutError:
                pass
            else:
                done = True


def fast_freeze_plan(
    start_temp: float,
    freeze_temp: float,
):
    pass


def generic_thing():
    rs_uid = yield from scanspec_fly(
        flyer=ScanspecFlyer(
            dets=[saxs, waxs],
            settings={panda: settings_from_yaml("/path/to/panda_settings.yaml")},
            panda_streams=[PandAHDFLogic(panda, {"COUNTER1.VAL": "I0"})],
            fly_logic=PandATimeSeqLogic(panda),
        ),
        sw_spec=Line(linkam, start_temp, stop_temp, num_temps),
        sw_dets=[linkam],
        hw_spec=Static.duration(exposure, num_frames),
    )
    return rs_uid


def group_uuid(name: str) -> str:
    # Unique but readable
    return f"{name}-{str(uuid.uuid4())[:6]}"


class DetectorGroupLogic(ABC):
    # Read multipliers here, exposure is set in the plan
    @abstractmethod
    async def open(self) -> Dict[str, Descriptor]:
        """Open all writers, wait for them to be open and return their descriptors"""

    @abstractmethod
    async def ensure_armed(self) -> AsyncStatus:
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


class TriggeredDetectorsLogic(DetectorGroupLogic):
    def __init__(
        self,
        detector_logics: Sequence[DetectorLogic],
        writer_logics: Sequence[WriterLogic],
    ) -> None:
        self.detector_logics = detector_logics
        self.writer_logics = writer_logics
        self._arm_status: Optional[AsyncStatus] = None

    async def open(self) -> Dict[str, Descriptor]:
        return merge_gathered_dicts(wl.open() for wl in self.writer_logics)

    async def ensure_armed(self) -> AsyncStatus:
        if not self._arm_status or self._arm_status.done:
            # We need to re-arm
            # TODO: how to do variable gate here?
            statuses = await gather_list(
                dl.arm(DetectorTrigger.constant_gate) for dl in self.detector_logics
            )
            self._arm_status = AsyncStatus(gather_list(statuses))
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


class FlyerLogic(ABC, Generic[T]):
    @abstractmethod
    async def prepare(self, value: T):
        """Move to the start of the flyscan"""

    @abstractmethod
    async def fly(self):
        """Start the flyscan"""

    @abstractmethod
    async def stop(self):
        """Stop flying and wait everything to be stopped"""


def in_micros(t: float):
    return np.ceil(t / 1e6)


class PandARepeatedSeqLogic(FlyerLogic):
    def __init__(self, srgate: SrgateBlock, seq: SeqBlock) -> None:
        self.srgate = srgate
        self.seq = seq

    async def prepare(self, path: Path):
        # Only use path for length
        assert path.axes() == [], "Not expecting to move anything"
        await self.seq.repeats.set(len(path))

    async def fly(self):
        await self.srgate.force_set.execute()

    async def stop(self):
        await self.srgate.force_rst.execute()
        await wait_for_value(self.srgate.out, 0, timeout=1)


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


ScanAxis = Union[Device, Literal["DURATION"]]


class ScanspecFlyer(
    Device, Movable, Stageable, Flyable, Collectable, WritesExternalAssets
):
    def __init__(
        self,
        detector_group_logic: DetectorGroupLogic,
        flyer_logic: FlyerLogic[Path[ScanAxis]],
        settings: Dict[Device, Dict[str, Any]],
        configuration_signals: Sequence[SignalR],
    ):
        self._detector_group_logic = detector_group_logic
        self._flyer_logic = flyer_logic
        self._settings = settings
        self._configuration_signals = tuple(configuration_signals)
        self._spec: Optional[Spec] = None
        self._frames: List[Frames] = []
        self._describe: Dict[str, Descriptor] = {}
        self._det_status: Optional[AsyncStatus] = None
        self._watchers: List[Callable] = []
        self._fly_status: Optional[AsyncStatus] = None
        self._fly_start = 0.0
        self._offset = 0  # Add this to index to get frame number
        self._current_frame = 0  # The current frame we are on
        super().__init__(name="hw")

    @AsyncStatus.wrap
    async def stage(self) -> None:
        await self.unstage()
        await asyncio.gather(
            *[
                load_settings(device, settings)
                for device, settings in self._settings.items()
            ]
        )
        await self._detector_group_logic.open()
        self._offset = 0
        self._current_frame = 0

    @AsyncStatus.wrap
    async def set(self, spec: Spec[ScanAxis]):
        """Arm detectors and setup trajectories"""
        if spec != self._spec:
            self._spec = spec
            self._frames = spec.calculate()
        # index + offset = current_frame, but starting a new scan so want it to be 0
        # so subtract current_frame from both sides
        self._offset -= self._current_frame
        self._current_frame = 0
        await self._prepare(Path(self._frames))

    async def _prepare(self, path: Path[ScanAxis]):
        # Move to start and setup the flyscan, and arm dets in parallel
        self._det_status, _ = await asyncio.gather(
            self._detector_group_logic.ensure_armed(),
            self._flyer_logic.prepare(path),
        )

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
        await self._flyer_logic.fly()
        # Wait for all detectors to have written up to a particular frame
        await self._detector_group_logic.wait_for_index(len(self._spec) - self._offset)

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
                    target=len(self._spec),
                    unit="",
                    precision=0,
                    time_elapsed=time.monotonic() - self._fly_start,
                )

    def complete(self) -> AsyncStatus:
        assert self._fly_status, "Kickoff not run"
        return self._fly_status

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

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        await asyncio.gather(
            self._flyer_logic.stop(),
            self._detector_group_logic.close(),
            self._detector_group_logic.disarm(),
        )


def scanspec_fly(
    flyer: ScanspecFlyer,
    hw_spec: Spec[Device],
    setup_detectors: Iterator[Msg] = iter([]),
    sw_spec: Spec[Device] = Repeat(1),
    sw_dets: Sequence[Device] = (),
    flush_period=0.5,
    md: Optional[Dict[str, Any]] = None,
):
    _md = {
        "hw_spec": hw_spec.serialize(),
        "hw_dets": [det.name for det in flyer.dets],
        "sw_spec": sw_spec.serialize(),
        "sw_dets": [det.name for det in sw_dets],
        "hints": {},
    }
    _md.update(md or {})

    @bpp.stage_decorator([flyer] + list(sw_dets))
    @bpp.run_decorator(md=_md)
    def hw_scanspec_fly():
        yield from setup_detectors()
        yield from bps.declare_stream(*sw_dets, name="sw")
        yield from bps.declare_stream(flyer, name="hw")
        for point in sw_spec.midpoints():
            # Move flyer to start too
            point[flyer] = hw_spec
            # TODO: need to make pos_cache optional in this func
            yield from bps.move_per_step(point)
            yield from bps.trigger_and_read(sw_dets)
            yield from bps.checkpoint()
            yield from bps.kickoff(flyer)
            complete_group = group_uuid("complete")
            yield from bps.complete(flyer, group=complete_group)
            done = False
            while not done:
                try:
                    yield from bps.wait(group=complete_group, timeout=flush_period)
                except TimeoutError:
                    pass
                else:
                    done = True
                yield from bps.collect(flyer, stream=True, return_payload=False)
                yield from bps.checkpoint()

    rs_uid = yield from hw_scanspec_fly()
    return rs_uid
