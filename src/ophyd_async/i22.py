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
    Tuple,
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
    Pausable,
    Reading,
    Stageable,
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
from ophyd_async.panda.panda import (
    PandA,
    SeqBlock,
    SeqTable,
    SeqTableRow,
    seq_table_from_rows,
    seq_table_row,
)

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


def group_uuid(name: str) -> str:
    # Unique but readable
    return f"{name}-{str(uuid.uuid4())[:6]}"


def fly_and_collect(flyer, flush_period=0.5, checkpoint_every_collect=False):
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
        if checkpoint_every_collect:
            yield from bps.checkpoint()


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
        detector_logics: Sequence[DetectorLogic],
        writer_logics: Sequence[WriterLogic],
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


def in_micros(t: float):
    return np.ceil(t / 1e6)


@dataclass
class RepeatedTrigger:
    num: int
    width: float
    deadtime: float
    repeats: int = 1
    period: float = 0.0


class PandARepeatedTriggerLogic(TriggerLogic[RepeatedTrigger]):
    trigger = DetectorTrigger.constant_gate

    def __init__(self, seq: SeqBlock, shutter_time: float = 0) -> None:
        self.seq = seq
        self.shutter_time = shutter_time

    def trigger_info(self, value: RepeatedTrigger) -> TriggerInfo:
        return TriggerInfo(
            num=value.num * value.repeats,
            trigger=DetectorTrigger.constant_gate,
            deadtime=value.deadtime,
            livetime=value.width,
        )

    async def prepare(self, value: RepeatedTrigger):
        trigger_time = value.num * (value.width + value.deadtime)
        pre_delay = max(value.period - 2 * self.shutter_time - trigger_time, 0)
        table = seq_table_from_rows(
            # Wait for pre-delay then open shutter
            SeqTableRow(
                time1=in_micros(pre_delay),
                time2=in_micros(self.shutter_time),
                outa2=True,
            ),
            # Keeping shutter open, do N triggers
            SeqTableRow(
                repeats=value.num,
                time1=in_micros(value.width),
                outa1=True,
                outb1=True,
                time2=in_micros(value.deadtime),
                outa2=True,
            ),
            # Add the shutter close
            SeqTableRow(time2=in_micros(self.shutter_time)),
        )
        await asyncio.gather(
            self.seq.prescale_units.set("us"),
            self.seq.enable.set("ZERO"),
        )
        await asyncio.gather(
            self.seq.prescale.set(1),
            self.seq.repeats.set(value.repeats),
            self.seq.table.set(table),
        )

    async def start(self):
        await self.seq.enable.set("ONE")
        await wait_for_value(self.seq.active, 1, timeout=1)
        await wait_for_value(self.seq.active, 0, timeout=1)

    async def stop(self):
        await self.seq.enable.set("ZERO")
        await wait_for_value(self.seq.active, 0, timeout=1)


class HardwareTriggeredFlyable(
    Device, Movable, Stageable, Flyable, Collectable, WritesExternalAssets, Generic[T]
):
    def __init__(
        self,
        detector_group_logic: DetectorGroupLogic,
        trigger_logic: TriggerLogic[T],
        settings: Dict[Device, Dict[str, Any]],
        configuration_signals: Sequence[SignalR],
        name: str = "",
    ):
        self._detector_group_logic = detector_group_logic
        self._trigger_logic = trigger_logic
        self._settings = settings
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


def scanspec_fly(
    flyer: ScanSpecFlyable,
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
        yield from setup_detectors
        yield from bps.declare_stream(*sw_dets, name="sw")
        yield from bps.declare_stream(flyer, name="hw")
        for point in sw_spec.midpoints():
            # Move flyer to start too
            point[flyer] = hw_spec
            # TODO: need to make pos_cache optional in this func
            yield from bps.move_per_step(point)
            yield from bps.trigger_and_read(sw_dets)
            yield from bps.checkpoint()
            yield from fly_and_collect(
                flyer, flush_period, checkpoint_every_collect=True
            )

    rs_uid = yield from hw_scanspec_fly()
    return rs_uid


def step_to_num(start: float, stop: float, step: float) -> Tuple[float, float, int]:
    # Make step be the right direction
    step = abs(step) if stop > start else -abs(step)
    # If stop is within 1% of a step then include it
    num = int((stop - start) / step + 0.01)
    return start, start + num * step, num


def scan_linkam(
    flyer: HardwareTriggeredFlyable,
    start: float,
    stop: float,
    step: float,
    rate: float,
    exposure: float,
    deadtime: float,
    num_frames: int,
    fly=False,
):
    one_batch = RepeatedTrigger(num=num_frames, width=exposure, deadtime=deadtime)
    start, stop, num = step_to_num(start, stop, step)
    yield from bps.mv(linkam.ramp_rate, rate)
    if fly:
        # Do a single batch to start
        yield from bps.mv(linkam, start, flyer, one_batch)
        yield from fly_and_collect(flyer)
        # Setup for many batches
        many_batches = RepeatedTrigger(
            num=num_frames,
            width=exposure,
            deadtime=deadtime,
            repeats=num,
            period=abs((stop - start) / rate) / num,
        )
        yield from bps.mv(flyer, many_batches)
        # Then start flying, collecting roughly every step
        linkam_group = group_uuid("linkam")
        yield from bps.abs_set(linkam, stop, group=linkam_group, wait=False)
        # Collect constantly
        yield from fly_and_collect(flyer)
        # Make sure linkam has finished
        yield from bps.wait(group=linkam_group)
    else:
        temps = np.linspace(start, stop, num)
        for temp in temps:
            yield from bps.mv(linkam, temp, flyer, one_batch)
            # Collect at each step
            yield from fly_and_collect(flyer)


def linkam_plan(
    start_temp: float,
    cool_temp: float,
    cool_step: float,
    cool_rate: float,
    heat_temp: float,
    heat_step: float,
    heat_rate: float,
    num_frames: int,
    exposure: float,
    md: Optional[Dict[str, Any]] = None,
):
    """Cool in steps, then heat constantly, taking collections of num_frames each time::

                      _             __ heat_temp
                     / \           /
        cool_interval___\__       /
                           \     /
                  cool_temp \__ /
        exposures        xx  xx   xx    num_frames=2 each time

    Fast shutter will be opened for each group of exposures
    """
    dets = [saxs, waxs]  # and tetramm
    flyer = HardwareTriggeredFlyable(
        SameTriggerDetectorGroupLogic(
            [det.detector_logic for det in dets],
            [det.writer_logic for det in dets],
        ),
        PandARepeatedTriggerLogic(panda.seq[1], shutter_time=0.004),
        settings={saxs: config_with_temperature_stamping},
        # Or maybe a different object?
    )
    deadtime = max(det.detector_logic.get_deadtime(exposure) for det in dets)
    _md = {
        "dets": [det.name for det in dets],
    }
    _md.update(md or {})

    @bpp.stage_decorator([flyer])
    @bpp.run_decorator(md=_md)
    def inner_linkam_plan():
        # Step down at the cool rate
        yield from scan_linkam(
            flyer=flyer,
            start=start_temp,
            stop=cool_temp,
            step=cool_step,
            rate=cool_rate,
            exposure=exposure,
            deadtime=deadtime,
            num_frames=num_frames,
            fly=False,
        )
        # Fly up at the heat rate
        yield from scan_linkam(
            flyer=flyer,
            start=cool_temp,
            stop=heat_temp,
            step=heat_step,
            rate=heat_rate,
            exposure=exposure,
            deadtime=deadtime,
            num_frames=num_frames,
            fly=True,
        )

    rs_uid = yield from inner_linkam_plan()
    return rs_uid
