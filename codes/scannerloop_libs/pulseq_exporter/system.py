"""This module contains a global system object describing physical limits."""


class System:
    def __init__(self) -> None:
        # Durations are defined as multiples of these values
        # Siemens default values, according to pulseq 1.4.0 interpreter
        self.grad_raster_time = 10e-6  # Unit: s
        self.rf_raster_time = 1e-6  # Unit: s
        self.adc_raster_time = 0.1e-6  # Unit: s
        self.block_raster_time = 10e-6  # Unit: s

        # Used for raise errors on scanner limit violations
        self.max_grad = float('inf')  # Unit: Hz / m
        self.max_slew = float('inf')  # Unit: Hz / m / s

        # Not used (yet?)
        self.adc_dead_time = 0.0  # Unit: s
        self.rf_dead_time = 0.0  # Unit: s
        self.rf_ringdown_time = 0.0  # Unit: s

    def set_max_grad(self, max_grad, unit, gamma=42.576e6):
        if unit != "mT/m":
            raise NotImplementedError("Can't use units other than 'mT/m'")
        self.max_grad = max_grad * 1e-3 * gamma

    def set_max_slew(self, max_slew, unit, gamma=42.576e6):
        if unit != "T/m/s":
            raise NotImplementedError("Can't use units other than 'T/m/s'")
        self.max_slew = max_slew * gamma


# This class will be used as "global" (module-level) variable, so instanciate
# one and export that. Normally global variables are bad, but since the system
# is set once and used anywhere, it's ok. If this usecase changes, change the
# code to always expect system as a parameter
system = System()
