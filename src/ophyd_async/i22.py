import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np
from bluesky.protocols import (
    Collectable,
    Descriptor,
    Flyable,
    Movable,
    Reading,
    Stageable,
    SyncOrAsync,
    WritesExternalAssets,
)
from scanspec.core import Frames, Path
from scanspec.specs import DURATION, Line, Repeat, Spec, Static

from ophyd_async.core import Device, DeviceCollector, StandardReadable
from ophyd_async.core._device._signal.signal import wait_for_value
from ophyd_async.core.async_status import AsyncStatus
from ophyd_async.core.utils import merge_gathered_dicts
from ophyd_async.detector import (
    DetectorTrigger,
    StaticDirectoryProvider,
    StreamDetector,
    StreamLogic,
)
from ophyd_async.epics.areadetector import NDFileHDF, NDPluginStats, hdf_logic, pilatus
from ophyd_async.epics.signal import epics_signal_r, epics_signal_rw
from ophyd_async.panda.panda import PandA, PcapBlock, SeqBlock, build_seq_table

dp = StaticDirectoryProvider("/dls/p45/data/cmxxx/i22-yyy", "i22-yyy-")


def hdf_stats_pilatus(prefix: str, settings_path: str):
    drv = pilatus.ADPilatus(prefix + "DRV:")
    hdf = NDFileHDF(prefix + "HDF:")
    det = StreamDetector(
        detector_logic=pilatus.PilatusLogic(drv),
        stream_logic=hdf_logic.HDFLogic(hdf, dp, sum="NDStatsSum"),
        # settings=settings_from_yaml(settings_path),
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


def linkam_plan(
    start_temp: float,
    stop_temp: float,
    num_temps: int,
    num_frames: int,
    exposure: float,
):
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


class FlyerLogic(ABC):
    @abstractmethod
    async def setup(self, path: Path, exposure: Optional[float], deadtime: float):
        ...

    @abstractmethod
    async def fly(self):
        ...

    @abstractmethod
    async def stop(self):
        ...


def in_micros(t: float):
    return int(t / 1e6)


class PandATimeSeqLogic(FlyerLogic):
    def __init__(self, pcap: PcapBlock, seq: SeqBlock) -> None:
        self.pcap = pcap
        self.seq = seq

    async def setup(self, path: Path, exposure: Optional[float], deadtime: float):
        await self.stop()
        assert exposure, "Can only do a fixed exposure at the moment"
        # TODO: consider what happens if the timebase changes
        await self.seq.prescale_units.set("us")
        table = build_seq_table(
            time1=[in_micros(exposure)], time2=[in_micros(deadtime)], outa1=[1]
        )
        await asyncio.gather(
            self.seq.prescale.set(1),
            self.seq.repeats.set(len(path)),
            self.seq.table.set(table),
        )

    async def fly(self):
        await self.pcap.arm.set(1)

    async def stop(self):
        await self.pcap.arm.set(0, wait=True)
        await wait_for_value(self.pcap.arm, 0, timeout=1)


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


class ScanspecFlyer(
    Device, Movable, Stageable, Flyable, Collectable, WritesExternalAssets
):
    def __init__(
        self,
        dets: Sequence[StreamDetector],
        panda_streams: Sequence[StreamLogic],
        settings: Dict[Device, Dict[str, Any]],
        fly_logic: FlyerLogic,
    ):
        self.dets = list(dets)
        self.panda_streams = list(panda_streams)
        self.fly_logic = fly_logic
        self.settings = settings
        self._frames: List[Frames] = []
        self._det_statuses: List[AsyncStatus] = []
        self._num_frames: int = 0
        self._watchers: List[Callable] = []
        self._fly_status: Optional[AsyncStatus] = None
        self._fly_start: float = 0.0
        super().__init__(name="hw")

    @AsyncStatus.wrap
    async def stage(self) -> None:
        # Stop everything, get into a known state and open the stream"""
        stage_statuses = [det.stage() for det in self.dets]
        await asyncio.gather(
            *[self.fly_logic.stop()] + [stream.close() for stream in self.panda_streams]
        )
        await asyncio.gather(
            *[
                load_settings(device, settings)
                for device, settings in self.settings.items()
            ]
        )
        await asyncio.gather(*[stream.open() for stream in self.panda_streams])
        await asyncio.gather(*stage_statuses)

    @AsyncStatus.wrap
    async def set(self, frames: List[Frames]):
        """Arm detectors and setup trajectories"""
        self._frames = frames
        self._num_frames = len(Path(self._frames))
        exposure = get_duration(self._frames)
        if exposure is None:
            det_coros = [
                det.detector_logic.arm(
                    trigger=DetectorTrigger.variable_gate,
                    num=self._num_frames,
                )
                for det in self.dets
            ]
        else:
            det_coros = [
                det.detector_logic.arm(
                    trigger=DetectorTrigger.constant_gate,
                    num=self._num_frames,
                    exposure=exposure,
                )
                for det in self.dets
            ]
        # Start arming the detectors
        det_future = asyncio.gather(*det_coros)
        deadtime = max(det.detector_logic.get_deadtime(exposure) for det in self.dets)
        # Move to start and setup the flyscan
        await self.fly_logic.setup(Path(self._frames), exposure, deadtime)
        # Wait for detectors to be armed
        self._det_statuses = await det_future

    async def describe_configuration(self) -> Dict[str, Descriptor]:
        return await merge_gathered_dicts(
            det.describe_configuration() for det in self.dets
        )

    async def read_configuration(self) -> Dict[str, Reading]:
        return await merge_gathered_dicts(det.read_configuration() for det in self.dets)

    async def describe_collect(self) -> Dict[str, Descriptor]:
        shapes = asyncio.gather(*[det.detector_logic.get_shape() for det in self.dets])
        # TODO: add sw_shape here when detectors are multiplied up
        coros = [
            det.stream_logic.describe_datasets(det.name, shape, sw_shape=())
            for det, shape in zip(self.dets, shapes)
        ] + [
            stream.describe_datasets("", (), sw_shape=())
            for stream in self.panda_streams
        ]
        return await merge_gathered_dicts(coros)

    @AsyncStatus.wrap
    async def kickoff(self) -> None:
        self._watchers = []
        self._fly_status = AsyncStatus(self.fly_logic.fly(), self._watchers)
        self._fly_start = time.monotonic()

    async def collect_asset_docs(self) -> AsyncIterator[Asset]:
        stream_logics = [det.stream_logic for det in self.dets] + self.panda_streams
        frames_written = min(await stream.frames_written() for stream in stream_logics)
        for stream in stream_logics:
            async for doc in stream.collect_stream_docs(frames_written):
                yield doc
        for watcher in self._watchers:
            watcher(
                name=self.name,
                current=frames_written,
                initial=0,
                target=self._num_frames,
                units="",
                precision=0,
                time_elapsed=time.monotonic() - self._fly_start,
            )

    def complete(self) -> AsyncStatus:
        assert self._fly_status, "Kickoff not run"
        return self._fly_status

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        coros = [det.stage() for det in self.dets] + [
            stream.close() for stream in self.panda_streams
        ]
        await asyncio.gather(*coros)


def scanspec_fly(
    flyer: ScanspecFlyer,
    hw_spec: Spec[Device],
    sw_spec: Spec[Device] = Repeat(1),
    sw_dets: Sequence[Device] = (),
    flush_period=0.5,
    md: Optional[Dict[str, Any]] = None,
):
    # TODO: hw and sw instead of hw and sw
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
        yield from bps.declare_stream(*sw_dets, name="sw")
        yield from bps.declare_stream(flyer, name="hw")
        hw_frames = hw_spec.calculate()
        for point in sw_spec.midpoints():
            # Move flyer to start too
            point[flyer] = hw_frames
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
