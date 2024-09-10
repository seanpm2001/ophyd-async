import asyncio
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from scanspec.specs import Frames, Path, Spec

from ophyd_async.core import (
    AsyncStatus,
    TriggerLogic,
    WatchableAsyncStatus,
    wait_for_value,
)
from ophyd_async.epics import motor

from ._block import PcompBlock, PcompDirectionOptions, SeqBlock, TimeUnits
from ._table import SeqTable, SeqTableRow, SeqTrigger, seq_table_from_rows


class SeqTableInfo(BaseModel):
    sequence_table: SeqTable = Field(strict=True)
    repeats: int = Field(ge=0)
    prescale_as_us: float = Field(default=1, ge=0)  # microseconds


class ScanSpecInfo(BaseModel):
    spec: Spec[motor.Motor] = Field()
    deadtime: float = Field()


class StaticSeqTableTriggerLogic(TriggerLogic[SeqTableInfo]):
    def __init__(self, seq: SeqBlock) -> None:
        self.seq = seq

    async def prepare(self, value: SeqTableInfo):
        await asyncio.gather(
            self.seq.prescale_units.set(TimeUnits.us),
            self.seq.enable.set("ZERO"),
        )
        await asyncio.gather(
            self.seq.prescale.set(value.prescale_as_us),
            self.seq.repeats.set(value.repeats),
            self.seq.table.set(value.sequence_table),
        )

    async def kickoff(self) -> None:
        await self.seq.enable.set("ONE")
        await wait_for_value(self.seq.active, True, timeout=1)

    async def complete(self) -> None:
        await wait_for_value(self.seq.active, False, timeout=None)

    async def stop(self):
        await self.seq.enable.set("ZERO")
        await wait_for_value(self.seq.active, False, timeout=1)


class ScanSpecSeqTableTriggerLogic(TriggerLogic[ScanSpecInfo]):
    def __init__(self, seq: SeqBlock, name="") -> None:
        self.seq = seq
        self.name = name

    @AsyncStatus.wrap
    async def prepare(self, value: ScanSpecInfo):
        await asyncio.gather(
            self.seq.prescale_units.set(TimeUnits.us),
            self.seq.enable.set("ZERO"),
        )
        path = Path(value.spec.calculate())
        chunk = path.consume()
        gaps = self._calculate_gaps(chunk)
        if gaps[0] == 0:
            gaps = np.delete(gaps, 0)
        scan_size = len(chunk)

        fast_axis = chunk.axes()[len(chunk.axes()) - 2]
        gaps = np.append(gaps, scan_size)
        start = 0
        rows: SeqTableRow = ()
        for gap in gaps:
            # Wait for GPIO to go low
            rows += (SeqTableRow(trigger=SeqTrigger.BITA_0),)
            # Wait for GPIO to go high
            rows += (
                SeqTableRow(
                    trigger=SeqTrigger.BITA_1,
                ),
            )
            # Wait for position
            if chunk.midpoints[fast_axis][gap - 1] > chunk.midpoints[fast_axis][start]:
                trig = SeqTrigger.POSA_GT
                dir = False

            else:
                trig = SeqTrigger.POSA_LT
                dir = True
            rows += (
                SeqTableRow(
                    trigger=trig,
                    position=chunk.lower[fast_axis][start]
                    / await fast_axis.encoder_res.get_value(),
                ),
            )

            # Time based triggers
            rows += (
                SeqTableRow(
                    repeats=gap - start,
                    trigger=SeqTrigger.IMMEDIATE,
                    time1=(chunk.midpoints["DURATION"][0] - value.deadtime) * 10**6,
                    time2=value.deadtime * 10**6,
                    outa1=True,
                    outb1=dir,
                    outa2=False,
                    outb2=False,
                ),
            )

            start = gap
        table: SeqTable = seq_table_from_rows(rows)
        await asyncio.gather(
            self.seq.prescale.set(0),
            self.seq.repeats.set(1),
            self.seq.table.set(table),
        )

    @AsyncStatus.wrap
    async def kickoff(self) -> None:
        await self.seq.enable.set("ONE")
        await wait_for_value(self.seq.active, True, timeout=1)

    @WatchableAsyncStatus.wrap
    async def complete(self) -> None:
        await wait_for_value(self.seq.active, False, timeout=None)

    async def stop(self):
        await self.seq.enable.set("ZERO")
        await wait_for_value(self.seq.active, False, timeout=1)

    def _calculate_gaps(self, chunk: Frames[motor.Motor]):
        inds = np.argwhere(chunk.gap)
        if len(inds) == 0:
            return len(chunk)
        else:
            return inds


class PcompInfo(BaseModel):
    start_postion: int = Field(description="start position in counts")
    pulse_width: int = Field(description="width of a single pulse in counts", gt=0)
    rising_edge_step: int = Field(
        description="step between rising edges of pulses in counts", gt=0
    )  #
    number_of_pulses: int = Field(
        description=(
            "Number of pulses to send before the PCOMP block is disarmed. "
            "0 means infinite."
        ),
        ge=0,
    )
    direction: PcompDirectionOptions = Field(
        description=(
            "Specifies which direction the motor counts should be "
            "moving. Pulses won't be sent unless the values are moving in "
            "this direction."
        )
    )


class StaticPcompTriggerLogic(TriggerLogic[PcompInfo]):
    def __init__(self, pcomp: PcompBlock) -> None:
        self.pcomp = pcomp

    async def prepare(self, value: PcompInfo):
        await self.pcomp.enable.set("ZERO")
        await asyncio.gather(
            self.pcomp.start.set(value.start_postion),
            self.pcomp.width.set(value.pulse_width),
            self.pcomp.step.set(value.rising_edge_step),
            self.pcomp.pulses.set(value.number_of_pulses),
            self.pcomp.dir.set(value.direction),
        )

    async def kickoff(self) -> None:
        await self.pcomp.enable.set("ONE")
        await wait_for_value(self.pcomp.active, True, timeout=1)

    async def complete(self, timeout: Optional[float] = None) -> None:
        await wait_for_value(self.pcomp.active, False, timeout=timeout)

    async def stop(self):
        await self.pcomp.enable.set("ZERO")
        await wait_for_value(self.pcomp.active, False, timeout=1)
