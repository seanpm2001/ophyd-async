import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterator, List, Optional, Sequence, Union

from bluesky.protocols import Descriptor
from event_model import StreamDatum, StreamResource, compose_stream_resource

from ophyd_async.core._device._signal.signal import (
    set_and_wait_for_value,
    wait_for_value,
)
from ophyd_async.core.async_status import AsyncStatus
from ophyd_async.detector import (
    DirectoryProvider,
    NameProvider,
    ShapeProvider,
    WriterLogic,
)

from .nd_file_hdf import FileWriteMode, NDFileHDF

# How long to wait for new frames before timing out
FRAME_TIMEOUT = 120


@dataclass
class _HDFDataset:
    name: str
    path: str
    shape: Sequence[int]
    multiplier: int


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
                resource_kwargs={
                    "path": ds.path,
                    "multiplier": ds.multiplier,
                },
            )
            for ds in datasets
        ]

    def stream_resources(self) -> Iterator[StreamResource]:
        for bundle in self._bundles:
            yield bundle.stream_resource_doc

    def stream_data(self, indices_written: int) -> Iterator[StreamDatum]:
        # Indices are relative to resource
        if indices_written >= self._last_emitted:
            indices = dict(
                start=self._last_emitted,
                stop=indices_written,
            )
            self._last_emitted = indices_written
            self._last_flush = time.monotonic()
            for bundle in self._bundles:
                yield bundle.compose_stream_datum(indices)
        if time.monotonic() - self._last_flush > FRAME_TIMEOUT:
            raise TimeoutError(f"Writing stalled on frame {indices_written}")
        return None


class ADHDFLogic(WriterLogic):
    def __init__(
        self,
        plugin: NDFileHDF,
        directory_provider: DirectoryProvider,
        name_provider: NameProvider,
        shape_provider: ShapeProvider,
        **scalar_datasets_paths: str,
    ) -> None:
        self._plugin = plugin
        self._directory_provider = directory_provider
        self._name_provider = name_provider
        self._shape_provider = shape_provider
        self._scalar_datasets_paths = scalar_datasets_paths
        self._capture_status: Optional[AsyncStatus] = None
        self._datasets: List[_HDFDataset] = []
        self._file: Optional[_HDFFile] = None
        self._multiplier = 1

    async def open(self, multiplier: int = 1) -> Dict[str, Descriptor]:
        self._file = None
        info = self._directory_provider()
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
        name = self._name_provider()
        detector_shape = tuple(await self._shape_provider())
        self._multiplier = multiplier
        if multiplier > 1:
            outer_shape = (multiplier,)
        else:
            outer_shape = ()
        # Add the main data
        self._datasets = [_HDFDataset(name, "/entry/data", detector_shape, multiplier)]
        # And all the scalar datasets
        for ds_name, ds_path in self._scalar_datasets_paths.items():
            self._datasets.append(
                _HDFDataset(f"{name}.{ds_name}", f"/entry/{ds_path}", (), multiplier)
            )
        describe = {
            ds.name: Descriptor(
                source=self._plugin.full_file_name.source,
                shape=outer_shape + ds.shape,
                dtype="array",
                external="STREAM:",
            )
            for ds in self._datasets
        }
        return describe

    async def wait_for_index(self, index: int):
        def matcher(value: int) -> bool:
            return value // self._multiplier >= index

        matcher.__name__ = f"index_at_least_{index}"
        await wait_for_value(self._plugin.num_captured, matcher, timeout=None)

    async def get_indices_written(self) -> int:
        num_captured = await self._plugin.num_captured.get_value()
        return num_captured // self._multiplier

    async def collect_stream_docs(
        self, indices_written: int
    ) -> AsyncIterator[Union[StreamResource, StreamDatum]]:
        # TODO: fail if we get dropped frames
        if indices_written and not self._file:
            self._file = _HDFFile(
                await self._plugin.full_file_name.get_value(), self._datasets
            )
            for doc in self._file.stream_resources():
                yield doc
        for doc in self._file.stream_data(indices_written):
            yield doc

    async def close(self):
        assert self._capture_status, "Open not run"
        # Already done a caput callback in _capture_status, so can't do one here
        await self._plugin.capture.set(0, wait=False)
        await self._capture_status
