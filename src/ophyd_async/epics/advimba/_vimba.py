from bluesky.protocols import HasHints, Hints

from ophyd_async.core import DirectoryProvider, StandardDetector
from ophyd_async.epics.adcore import ADBaseShapeProvider, HDFWriter, NDFileHDF

from ._vimba_controller import VimbaController
from ._vimba_io import VimbaDriverIO


class VimbaDetector(StandardDetector, HasHints):
    """
    Ophyd-async implementation of an ADVimba Detector.
    """

    _controller: VimbaController
    _writer: HDFWriter

    def __init__(
        self,
        prefix: str,
        directory_provider: DirectoryProvider,
        drv_suffix="cam1:",
        hdf_suffix="HDF1:",
        name="",
    ):
        self.drv = VimbaDriverIO(prefix + drv_suffix)
        self.hdf = NDFileHDF(prefix + hdf_suffix)

        super().__init__(
            VimbaController(self.drv),
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
