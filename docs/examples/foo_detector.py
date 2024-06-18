import asyncio
from typing import Optional

from bluesky.protocols import HasHints, Hints

from ophyd_async.core import (
    AsyncStatus,
    DetectorControl,
    DetectorTrigger,
    DirectoryProvider,
    StandardDetector,
)
from ophyd_async.epics import ImageMode, ad_rw, stop_busy_record
from ophyd_async.epics.adcore import (
    ADBase,
    ADBaseShapeProvider,
    HDFWriter,
    NDFileHDF,
    start_acquiring_driver_and_ensure_status,
)


class FooDriver(ADBase):
    def __init__(self, prefix: str, name: str = "") -> None:
        self.trigger_mode = ad_rw(str, prefix + "TriggerMode")
        super().__init__(prefix, name)


class FooController(DetectorControl):
    def __init__(self, driver: FooDriver) -> None:
        self._drv = driver

    def get_deadtime(self, exposure: float) -> float:
        # FooDetector deadtime handling
        return 0.001

    async def arm(
        self,
        num: int,
        trigger: DetectorTrigger = DetectorTrigger.internal,
        exposure: Optional[float] = None,
    ) -> AsyncStatus:
        await asyncio.gather(
            self._drv.num_images.set(num),
            self._drv.image_mode.set(ImageMode.multiple),
            self._drv.trigger_mode.set(f"FOO{trigger}"),
        )
        if exposure is not None:
            await self._drv.acquire_time.set(exposure)
        return await start_acquiring_driver_and_ensure_status(self._drv)

    async def disarm(self):
        await stop_busy_record(self._drv.acquire, False, timeout=1)


class FooDetector(StandardDetector, HasHints):
    _controller: FooController
    _writer: HDFWriter

    def __init__(
        self,
        prefix: str,
        directory_provider: DirectoryProvider,
        drv_suffix="cam1:",
        hdf_suffix="HDF1:",
        name="",
    ):
        # Must be children to pick up connect
        self.drv = FooDriver(prefix + drv_suffix)
        self.hdf = NDFileHDF(prefix + hdf_suffix)

        super().__init__(
            FooController(self.drv),
            HDFWriter(
                self.hdf,
                directory_provider,
                lambda: self.name,
                ADBaseShapeProvider(self.drv),
            ),
            config_sigs=(self.drv.acquire_time,),
            name=name,
        )

    @property
    def hints(self) -> Hints:
        return self._writer.hints
