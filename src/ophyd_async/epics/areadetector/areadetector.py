from typing import Sequence, Tuple, cast

from ophyd_async.core import DirectoryProvider, SignalR, StandardDetector

from .controllers import ADController
from .drivers import ADDriver, ADDriverShapeProvider
from .writers import HDFWriter, NDFileHDF


class AreaDetector(StandardDetector):
    _controller: ADController
    _writer: HDFWriter

    def __init__(
        self,
        prefix: str,
        directory_provider: DirectoryProvider,
        name: str = "",
        config_sigs: Sequence[Tuple[str, SignalR]] = (),
    ):
        self.drv = ADDriver(prefix + "DRV:")
        self.hdf = NDFileHDF(prefix + "HDF:")

        super().__init__(
            ADController(self.drv),
            HDFWriter(
                self.hdf,
                directory_provider,
                lambda: self.name,
                ADDriverShapeProvider(self.drv),
            ),
            config_sigs=config_sigs,
            name=name,
        )

    async def connect(self, sim: bool = False):
        await super().connect(sim=sim)
        driver = self._controller.driver
        self._config_sigs = [
            *self._config_sigs,
            *[
                (getattr(signal, "name"), cast(SignalR, signal))
                for signal in [driver.acquire_time, driver.acquire]
            ],
        ]
