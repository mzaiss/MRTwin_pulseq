from __future__ import annotations
from warnings import warn
from typing import Optional, Callable, Union
from enum import Enum
import numpy as np
from .system import system

# NOTE: All properties should be in SI units, only converted when exporting
# time: [s], grad: [Hz/m], rf amp: [Hz], phase: [rad]


class ShapeType(Enum):
    RF = 0
    GRAD = 1


class Shape:
    """Shape usable for gradients and RF magnitude & phase, not timing."""

    def __init__(self, shape: np.ndarray) -> None:
        if shape.ndim != 1:
            raise ValueError(
                f"Argument 'shape' must have dim=1, given: dim={shape.ndim}."
            )
        if shape.size < 1:
            raise ValueError("Argument 'shape' must have at least 1 element.")
        if shape.min() < -1 or 1 < shape.max():
            raise ValueError(
                f"Input array 'shape' must be in the range [-1, 1], but "
                f"contains values in the range [{shape.min()}, {shape.max()}]."
            )

        self.shape = shape
        self.size = shape.size

        self.grad_mom = float(shape.sum() * system.grad_raster_time)
        self.flip_angle = float(shape.sum() * system.rf_raster_time * 2*np.pi)

        # Max. <amp> for use as gradient based on max_grad & max_slew
        slew = np.max(np.abs(np.diff(np.concatenate(
            [np.zeros(1), self.shape, np.zeros(1)]
        ))))
        slew /= system.grad_raster_time  # Relative to gradient <amp> [Hz/m]

        self.amp_max_grad = float('inf')  # Shape is all zeros
        self.amp_max_slew = float('inf')  # Shape is constant

        if np.any(self.shape != 0):
            self.amp_max_grad = system.max_grad / np.abs(self.shape).max()
            if slew > 0:
                self.amp_max_slew = system.max_slew / slew

    @classmethod
    def from_func(cls, func: Callable[[np.ndarray], np.ndarray],
                  duration: float, typ: ShapeType) -> Shape:
        if typ == ShapeType.GRAD:
            raster_time = system.grad_raster_time
        else:
            raster_time = system.rf_raster_time
        t = np.linspace(0, 1, round(duration / raster_time))
        return cls(func(t))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Shape):
            return False
        else:
            return np.array_equal(self.shape, o.shape)


class RfEvent:
    def __init__(self, mag_shape: Shape, phase_shape: Shape,
                 flip_angle: float, phase: float, delay: float) -> None:
        if mag_shape.size != phase_shape.size:
            raise ValueError(
                f"The magnitude and phase shape of a RF event must have the "
                f"same size, but have sizes mag_shape.size={mag_shape.size} "
                f"and phase_shape.size={phase_shape.size}."
            )
        self.amp = flip_angle / mag_shape.flip_angle
        self.mag_shape: Union[Shape, int] = mag_shape
        self.phase_shape: Union[Shape, int] = phase_shape
        # <time_id> = 0: default time raster
        self.delay = delay
        # <freq> = 0: no off-resonance pulses
        self.phase = phase

        self.duration = self.delay + self.mag_shape.size*system.rf_raster_time

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, RfEvent):
            return False
        else:
            return (
                self.amp == o.amp and self.delay == o.delay and
                self.phase == o.phase and
                self.mag_shape is o.mag_shape and  # Same instance
                self.phase_shape == o.phase_shape  # Same instance
            )


class ShapeGradEvent:
    def __init__(self, shape: Shape, amplitude: float, delay: float) -> None:
        if amplitude > shape.amp_max_grad:
            raise ValueError(
                f"The given amplitude of {amplitude} violates the scanner's "
                f"maximum gradient amplitude of {system.max_grad}, which "
                f"is an amplitude of {shape.amp_max_grad} at the given shape."
            )
        if amplitude > shape.amp_max_slew:
            raise ValueError(
                f"The given amplitude of {amplitude} violates the scanner's "
                f"maximum gradient slew rate of {system.max_slew}, which "
                f"is an amplitude of {shape.amp_max_slew} at the given shape."
            )
        self.shape = shape
        self.amp = amplitude
        # <time_id> = 0: default time raster
        self.delay = delay

        self.duration = self.delay + self.shape.size*system.grad_raster_time
    
    @classmethod
    def from_grad_mom(cls, shape: Shape, grad_mom: float,
                      delay: float) -> ShapeGradEvent:
        if shape.grad_mom == 0:
            raise ValueError(
                "Can't create a ShapeGradEvent with from_grad_mom based on a "
                "shape with a total gradient moment of 0."
            )
        return cls(shape, grad_mom / shape.grad_mom, delay)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ShapeGradEvent):
            return False
        else:
            return (
                self.amp == o.amp and self.delay == o.delay and
                self.shape is o.shape  # Same instance
            )


class TrapGradEvent:
    def __init__(self, amplitude: float, delay: float, rise: float,
                 flat: float, fall: float) -> None:
        if amplitude > system.max_slew:
            raise ValueError(
                f"The given amplitude of {amplitude} violates the scanner's "
                f"maximum gradient amplitude of {system.max_grad}."
            )
        if amplitude / rise > system.max_slew:
            raise ValueError(
                f"The given rise time of {rise} is too short and violates the "
                f"scanner's maximum gradient slew rate of {system.max_slew}. "
                f"Increase the rise time to {amplitude / system.max_slew} or "
                f"decrease the amplitude to {rise * system.max_slew}."
            )
        if amplitude / fall > system.max_slew:
            raise ValueError(
                f"The given fall time of {fall} is too short and violates the "
                f"scanner's maximum gradient slew rate of {system.max_slew}. "
                f"Increase the fall time to {amplitude / system.max_slew} or "
                f"decrease the amplitude to {fall * system.max_slew}."
            )
        self.amp = amplitude
        self.delay = delay
        self.rise = rise
        self.flat = flat
        self.fall = fall

        self.duration = self.delay + self.rise + self.flat + self.fall
        self.area = self.amp * (self.flat + self.rise/2 + self.fall/2)

    @classmethod
    def from_amplitude(cls, amplitude: float,
                       delay: float, flat: float) -> TrapGradEvent:
        rise = fall = amplitude / system.max_slew
        return cls(amplitude, delay, rise, flat, fall)

    @classmethod
    def from_area(cls, area: float, delay: float) -> TrapGradEvent:
        # Assume triangle gradient:
        amp = np.sqrt(area * system.max_slew)
        if amp <= system.max_grad:
            rise = fall = amp / system.max_slew
            return cls(amp, delay, rise, 0.0, fall)
        else:
            # Make trapezoidal if amplitude is too large
            amp = system.max_grad
            rise = fall = amp / system.max_slew
            flat_area = area - amp * rise
            flat = flat_area / amp
            return cls(amp, delay, rise, flat, fall)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TrapGradEvent):
            return False
        else:
            return (
                self.amp == o.amp and self.delay == o.delay and
                self.rise == o.rise and self.flat == o.flat and
                self.fall == o.fall
            )


class AdcEvent:
    def __init__(self, sample_count: int, dwell_time: float,
                 delay: float, phase: float) -> None:
        if np.modf(dwell_time / system.adc_raster_time + 1e-4)[0] > 2e-4:
            warn(
                f"The chosen ADC dwell time ({dwell_time}) should be a "
                f"multiple of the AdcRasterTime ({system.adc_raster_time}). "
                f"In pulseq 1.4.0 they are defined independent, this relation "
                f"seems to be ignored by the scanner."
            )
        self.num = sample_count
        self.dwell = dwell_time
        self.delay = delay
        # <freq> = 0: no frequency offset
        self.phase = phase

        self.duration = self.delay + self.num * self.dwell

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, AdcEvent):
            return False
        else:
            return (
                self.num == o.num and self.dwell == o.dwell and
                self.delay == o.delay and self.phase == o.phase
            )


PulseType = Optional[RfEvent]
GradType = Optional[Union[ShapeGradEvent, TrapGradEvent]]
AdcType = Optional[AdcEvent]


class Block:
    def __init__(self, rf: PulseType = None,
                 gx: GradType = None, gy: GradType = None, gz: GradType = None,
                 adc: AdcType = None,
                 duration: Optional[float] = None) -> None:
        # The blocks duration must be at least as long as its longest event!
        min_duration = max([
            rf.duration if rf else 0.0,
            gx.duration if gx else 0.0,
            gy.duration if gy else 0.0,
            gz.duration if gz else 0.0,
            adc.duration if adc else 0.0,
        ])
        if duration:
            if duration < min_duration:
                raise ValueError(
                    f"The given duration of {duration} is shorter than the "
                    f"longest event, increase to at least {min_duration}."
                )
            self.duration = duration
        else:
            self.duration = min_duration
        self.rf = rf
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.adc = adc
        # <ext> = 0: no support for extensions
