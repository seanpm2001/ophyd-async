import asyncio
from typing import Literal, Tuple

from ophyd_async.core import (
    DetectorControl,
    DetectorTrigger,
    TriggerInfo,
)
from ophyd_async.core._status import AsyncStatus
from ophyd_async.epics import adcore
from ophyd_async.epics.adcore._core_logic import (
    start_acquiring_driver_and_ensure_status,
)

from ._aravis_io import AravisDriverIO, AravisTriggerMode, AravisTriggerSource

# The deadtime of an ADaravis controller varies depending on the exact model of camera.
# Ideally we would maximize performance by dynamically retrieving the deadtime at
# runtime. See https://github.com/bluesky/ophyd-async/issues/308
_HIGHEST_POSSIBLE_DEADTIME = 1961e-6


class AravisController(DetectorControl):
    GPIO_NUMBER = Literal[1, 2, 3, 4]

    def __init__(self, driver: AravisDriverIO, gpio_number: GPIO_NUMBER) -> None:
        self._drv = driver
        self.gpio_number = gpio_number
        self._arm_status: AsyncStatus | None = None

    def get_deadtime(self, exposure: float) -> float:
        return _HIGHEST_POSSIBLE_DEADTIME

    async def prepare(self, trigger_info: TriggerInfo):
        if (num := trigger_info.number) == 0:
            image_mode = adcore.ImageMode.continuous
        else:
            image_mode = adcore.ImageMode.multiple
        if (exposure := trigger_info.livetime) is not None:
            await self._drv.acquire_time.set(exposure)

        trigger_mode, trigger_source = self._get_trigger_info(trigger_info.trigger)
        # trigger mode must be set first and on it's own!
        await self._drv.trigger_mode.set(trigger_mode)

        await asyncio.gather(
            self._drv.trigger_source.set(trigger_source),
            self._drv.num_images.set(num),
            self._drv.image_mode.set(image_mode),
        )

    async def arm(self):
        self._arm_status = await start_acquiring_driver_and_ensure_status(self._drv)

    async def wait_for_idle(self):
        if self._arm_status:
            await self._arm_status

    def _get_trigger_info(
        self, trigger: DetectorTrigger
    ) -> Tuple[AravisTriggerMode, AravisTriggerSource]:
        supported_trigger_types = (
            DetectorTrigger.constant_gate,
            DetectorTrigger.edge_trigger,
            DetectorTrigger.internal,
        )
        if trigger not in supported_trigger_types:
            raise ValueError(
                f"{self.__class__.__name__} only supports the following trigger "
                f"types: {supported_trigger_types} but was asked to "
                f"use {trigger}"
            )
        if trigger == DetectorTrigger.internal:
            return AravisTriggerMode.off, "Freerun"
        else:
            return (AravisTriggerMode.on, f"Line{self.gpio_number}")

    async def disarm(self):
        await adcore.stop_busy_record(self._drv.acquire, False, timeout=1)
