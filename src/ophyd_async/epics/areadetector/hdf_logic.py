import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterator, List, Optional, Sequence, Union

from bluesky.protocols import Descriptor
from event_model import StreamDatum, StreamResource, compose_stream_resource

from ophyd_async.core._device._signal.signal import set_and_wait_for_value
from ophyd_async.core.async_status import AsyncStatus
from ophyd_async.detector import DirectoryProvider, StreamLogic

from .nd_file_hdf import FileWriteMode, NDFileHDF

# How long to wait for new frames before timing out
FRAME_TIMEOUT = 120


@dataclass
class _HDFDataset:
    name: str
    path: str
    frame_shape: Sequence[int]


class _HDFFile:
    def __init__(self, full_file_name: str, datasets: List[_HDFDataset]) -> None:
        self._last_emitted = 0
        self._last_flush = time.monotonic()
        self._bundles = [
            compose_stream_resource(
                spec="AD_HDF5_SWMR_SLICE",
                root="/",
                data_key=ds.name,
                resource_path=full_file_name,
                resource_kwargs={"path": ds.path, "frame_shape": ds.frame_shape},
            )
            for ds in datasets
        ]

    def stream_resources(self) -> Iterator[StreamResource]:
        for bundle in self._bundles:
            yield bundle.stream_resource_doc

    def stream_data(self, frames_written: int) -> Iterator[StreamDatum]:
        if frames_written >= self._last_emitted:
            indices = dict(start=self._last_emitted, stop=frames_written)
            self._last_emitted = frames_written
            self._last_flush = time.monotonic()
            for bundle in self._bundles:
                yield bundle.compose_stream_datum(indices)
        if time.monotonic() - self._last_flush > FRAME_TIMEOUT:
            raise TimeoutError(f"Writing stalled on frame {frames_written}")
        return None


class ADHDFLogic(StreamLogic):
    def __init__(
        self,
        plugin: NDFileHDF,
        directory_provider: DirectoryProvider,
        **scalar_datasets_paths: str,
    ) -> None:
        self._plugin = plugin
        self._directory_provider = directory_provider
        self._scalar_datasets_paths = scalar_datasets_paths
        self._capture_status: Optional[AsyncStatus] = None
        self._datasets: List[_HDFDataset] = []
        self._file: Optional[_HDFFile] = None

    async def open(self):
        self._file = None
        info = self._directory_provider.directory_info
        await asyncio.gather(
            self._plugin.lazy_open.set(True),
            self._plugin.swmr_mode.set(True),
            self._plugin.file_path.set(info.directory_path),
            self._plugin.file_name.set(f"{info.filename_prefix}{self._plugin.name}"),
            self._plugin.file_template.set("%s/%s.h5"),
            # Go forever
            self._plugin.num_capture.set(0),
            self._plugin.file_write_mode.set(FileWriteMode.stream),
        )
        # Wait for it to start, stashing the status that tells us when it finishes
        self._capture_status = await set_and_wait_for_value(self._plugin.capture, True)

    async def describe_datasets(
        self, name: str, detector_shape: Sequence[int], outer_shape: Sequence[int] = ()
    ) -> Dict[str, Descriptor]:
        # Add the main data
        self._datasets = [
            _HDFDataset(name, "/entry/data", outer_shape + detector_shape)
        ]
        # And all the scalar datasets
        for ds_name, ds_path in self._scalar_datasets_paths.items():
            self._datasets.append(
                _HDFDataset(f"{name}.{ds_name}", f"/entry/{ds_path}", outer_shape)
            )
        describe = {
            ds.name: Descriptor(
                source=self._plugin.full_file_name.source,
                shape=ds.frame_shape,
                dtype="array",
                external="STREAM:",
            )
            for ds in self._datasets
        }
        return describe

    async def frames_written(self) -> int:
        return await self._plugin.num_captured.get_value()

    async def collect_stream_docs(
        self, frames_written: int
    ) -> AsyncIterator[Union[StreamResource, StreamDatum]]:
        if frames_written and not self._file:
            self._file = _HDFFile(
                await self._plugin.full_file_name.get_value(), self._datasets
            )
            for doc in self._file.stream_resources():
                yield doc
        for doc in self._file.stream_data(frames_written):
            yield doc

    async def close(self):
        assert self._capture_status, "Open not run"
        # Already done a caput callback in _capture_status, so can't do one here
        await self._plugin.capture.set(0, wait=False)
        await self._capture_status
