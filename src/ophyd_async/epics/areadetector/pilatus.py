import asyncio
from enum import Enum
from typing import Sequence

from ophyd_async.core._device._signal.signal import (
    set_and_wait_for_value,
    wait_for_value,
)
from ophyd_async.core.async_status import AsyncStatus
from ophyd_async.detector import ArmMode, DetectorLogic
from ophyd_async.epics.areadetector.utils import ImageMode, ad_rw

from .ad_driver import ADDriver


class TriggerMode(Enum):
    internal = "Internal"
    ext_enable = "Ext. Enable"
    ext_trigger = "Ext. Trigger"
    mult_trigger = "Mult. Trigger"
    alignment = "Alignment"


class ADPilatus(ADDriver):
    def __init__(self, prefix: str) -> None:
        self.trigger_mode = ad_rw(TriggerMode, prefix + "TriggerMode")
        super().__init__(prefix)


TRIGGER_MODE = {
    ArmMode.internal: TriggerMode.internal,
    ArmMode.gated: TriggerMode.ext_enable,
    ArmMode.triggered: TriggerMode.ext_trigger,
}


class PilatusLogic(DetectorLogic):
    def __init__(self, driver: ADPilatus) -> None:
        self._driver = driver

    async def get_deadtime(self, exposure: float) -> float:
        return 0.002

    async def arm(self, mode: ArmMode = ArmMode.internal, num: int = 0) -> AsyncStatus:
        await asyncio.gather(
            self._driver.trigger_mode.set(TRIGGER_MODE[mode]),
            self._driver.num_images.set(num),
            self._driver.image_mode.set(ImageMode.multiple),
        )
        return await set_and_wait_for_value(self._driver.acquire, True)

    async def disarm(self):
        await self._driver.acquire.set(1, wait=False)
        await wait_for_value(self._driver.acquire, False, timeout=1)
