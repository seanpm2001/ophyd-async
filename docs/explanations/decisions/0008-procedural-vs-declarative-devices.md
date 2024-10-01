# 8. Procedural vs Declarative Devices

Date: 01/10/24

## Status

Accepted

## Context

In [](./0006-procedural-device-definitions.rst) we decided we preferred the procedural approach to devices, because of the issue of applying structure like `DeviceVector`. Since then we have `FastCS` and `Tango` support which use a declarative approach. We need to decide whether we are happy with this situation, or whether we should go all in one way or the other. A suitable test Device would be:

```python
class EpicsProceduralDevice(StandardReadable):
    def __init__(self, prefix: str, num_values: int, name="") -> None:
        with self.add_children_as_readables():
            self.value = DeviceVector(
                {
                    i: epics_signal_r(float, f"{prefix}Value{i}")
                    for i in range(1, num_values + 1)
                }
            )
        with self.add_children_as_readables(ConfigSignal):
            self.mode = epics_signal_rw(EnergyMode, prefix + "Mode")
        super().__init__(name=name)
```

and a Tango/FastCS procedural equivalent would be (if we add support to StandardReadable for HINTED and CONFIG annotations):
```python
class TangoDeclarativeDevice(StandardReadable, TangoDevice):
    value: Annotated[DeviceVector[SignalR[float]], HINTED]
    mode: Annotated[SignalRW[EnergyMode], CONFIG]
```

But we could specify the Tango one procedurally (with some slight ugliness around the DeviceVector):
```python
class TangoProceduralDevice(StandardReadable):
    def __init__(self, prefix: str, name="") -> None:
        with self.add_children_as_readables():
            self.value = DeviceVector({0: tango_signal_r(float)})
        with self.add_children_as_readables(ConfigSignal):
            self.mode = tango_signal_rw(EnergyMode)
        super().__init__(name=name, connector=TangoConnector(prefix))
```

or the EPICS one could be declarative:
```python
class EpicsDeclarativeDevice(StandardReadable, EpicsDevice):
    value: Annotated[
        DeviceVector[SignalR[float]], HINTED, EpicsSuffix("Mode%d", "num_values")
    ]
    mode: Annotated[SignalRW[EnergyMode], CONFIG, EpicsSuffix("Mode")]
```

Which do we prefer?

## Decision

What decision we made.

## Consequences

What we will do as a result of this decision.
