from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt
from .sequence import Sequence

import sys
sys.path.append("../scannerloop_libs")
import pulseq_exporter as pe


def export(seq_in: Sequence, file_name: str, slice_thickness: float,
           FOV: float, name: str) -> None:
    # Maybe better use a tensor for FOV?
    fov_tuple = (int(FOV), int(FOV), int(slice_thickness))
    FOV /= 1000  # FOV argument is in mm
    seq_in = seq_in.scale_gradients(1 / (FOV * 2*np.pi))

    pe.system.rf_ringdown_time = 20e-6
    pe.system.rf_dead_time = 100e-6
    pe.system.adc_dead_time = 20e-6
    pe.system.grad_raster_time = 10e-6
    pe.system.rf_raster_time = 1e-6
    pe.system.adc_raster_time = 0.1e-6
    pe.system.block_raster_time = 10e-6
    pe.system.set_max_grad(36, unit='mT/m')
    pe.system.set_max_slew(140, unit='T/m/s')

    seq_out = []

    # Wait 5 seconds before starting
    seq_out.append(pe.Block(duration=5.0))

    # Prepare pulses
    pulse_duration = 1e-3
    time_bw_product = 4
    bandwidth = time_bw_product / pulse_duration
    slice_sel_grad_amp = bandwidth / slice_thickness

    # Pulses have no phase profile
    rf_phase_shape = pe.Shape.from_func(
        pe.shapes.constant(0.0), pulse_duration, pe.ShapeType.RF
    )

    # Prepare a non-selective block pulse
    rf_global_mag_shape = pe.Shape.from_func(
        pe.shapes.constant(1.0), pulse_duration, pe.ShapeType.RF
    )

    def pulse_global(flip, phase, duration):
        pulse = pe.RfEvent(
            rf_global_mag_shape, rf_phase_shape, flip, phase, 0.0
        )
        return [pe.Block(rf=pulse, duration=duration)]

    # Prepare a slice selective sinc pulse
    rf_selective_mag_shape = pe.Shape.from_func(
        pe.shapes.windowed_sinc(1), pulse_duration, pe.ShapeType.RF
    )

    def pulse_selective(flip, phase, duration):
        sel_grad = pe.TrapGradEvent.from_amplitude(
            slice_sel_grad_amp, 0.0, pulse_duration
        )
        sel_grad_rewinder = pe.TrapGradEvent.from_area(sel_grad.area, 0.0)
        pulse = pe.RfEvent(
            rf_selective_mag_shape, rf_phase_shape, flip, phase, sel_grad.rise
        )
        # FIXME: The original code doesn't apply the rewinder gradient for
        # refocussing pulses. Maybe make Pulse.selective an enum with options
        # GLOBAL, SELECTIVE, SELECTIVE_REWOUND
        return [
            pe.Block(rf=pulse, gz=sel_grad),
            pe.Block(gz=sel_grad_rewinder, duration=duration-sel_grad.duration)
        ]

    for rep in seq_in:
        # Gradients and adc during pulse is not supported
        assert torch.all(rep.gradm[0, :] == 0)
        assert rep.adc[0] == 0

        if rep.pulse.selective:
            seq_out += pulse_selective(
                rep.pulse.angle, rep.pulse.phase, rep.event_time[0]
            )
        else:
            seq_out += pulse_global(
                rep.pulse.angle, rep.pulse.phase, rep.event_time[0]
            )

        # Skip rest of loop if the repetition contains just the pulse
        if rep.event_count == 1:
            continue

        # Splits is True where event_time or adc changes
        splits = (
            (torch.diff(rep.event_time) > 0) |
            (torch.diff(rep.adc).abs() > 0)
        )
        # Index at which the next split begins
        split_indices = (torch.nonzero(splits, as_tuple=True)[0] + 1).tolist()
        if len(split_indices) == 0 or split_indices[0] != 1:
            # first section always starts directly after pulse
            split_indices = [1] + split_indices
        # End of repetition as additional index
        split_indices.append(rep.event_count)

        # Split repetition into sections with identical event_time and adc
        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i+1]

            event_time = rep.event_time[start].item()
            has_adc = rep.adc[start].abs() > 0
            adc_phase = rep.adc[start].angle()

            gradm_events = [None, None, None]
            for d in range(3):
                # TODO: use TrapGrad if sufficient
                gradm = rep.gradm[start:end, d].detach().cpu().numpy()
                if np.all(gradm == 0):
                    continue

                gradm = resample_grad(gradm, event_time)
                gradm_amp = np.max(np.abs(gradm))
                gradm_shape = pe.Shape(gradm / gradm_amp)
                gradm_events[d] = pe.ShapeGradEvent(
                    gradm_shape, gradm_amp / pe.system.grad_raster_time, 0.0
                )

            # Specification: n-th sample of RF, Gradient or ADC is located at
            # t_n = t_start + t_raster * (n + 0.5)
            # NOTE: it does not actually seem like the pulseq interpreter
            # adheres to this centering (... + 0.5), so it's assumed that
            # everything is on-grid

            # Delay by a whole event to measure after gradients were applied
            adc_event = pe.AdcEvent(
                end-start, event_time, event_time, adc_phase
            ) if has_adc else None

            seq_out.append(pe.Block(
                gx=gradm_events[0], gy=gradm_events[1], gz=gradm_events[2],
                adc=adc_event
            ))

    pe.write_sequence(seq_out, file_name, fov_tuple, name)


def resample_grad(grad_moms: np.ndarray, event_time: float):
    # This method interpolates the gradient in a way that reduces slew rates
    # but still guarantees to produce the correct gradient moments in every
    # event (at least if event_time is a multiple of GradRasterTime).
    # For a visualisation, see https://www.desmos.com/calculator/fhn2ypie4k

    # Adjust gradient moments for different event time
    grads = grad_moms / event_time * pe.system.grad_raster_time

    # Gradients *between* events, average of neighbour gradients
    edge_points = np.zeros(grads.size + 2)
    edge_points[1:-1] = grads
    edge_points = 0.5 * (edge_points[1:] + edge_points[:-1])
    # Begin and end event at zero
    edge_points[0] = edge_points[-1] = 0

    peak_points = np.zeros((grads.size, 2))

    for i in range(grads.size):
        peak_points[i, :] = calc_peak_pos(
            edge_points[i], edge_points[i+1], grads[i]
        )

    # Resample the shape consisting of edge_points and peak_points
    total_duration = grads.size * event_time
    time = np.arange(0, total_duration, pe.system.grad_raster_time)
    event_array, dt_array = np.divmod(time, event_time)
    dt_array = (dt_array + 0.5*pe.system.grad_raster_time) / event_time
    shape = np.zeros(time.size)

    for i, (dt, event) in enumerate(zip(dt_array, event_array)):
        event = int(event)
        t_peak = peak_points[event, 0]
        if dt < t_peak:
            a = edge_points[event]
            b = peak_points[event, 1]
            t = dt / t_peak
        else:
            a = peak_points[event, 1]
            b = edge_points[event+1]
            t = (dt - t_peak) / (1 - t_peak)

        shape[i] = a + (b - a) * t

    return shape


def calc_peak_pos(h1: float, h2: float, A: float) -> tuple[float, float]:
    # Calculates solution for two constraints:
    # - Total area must be equal to A
    # - Slew must be minimized -> identical for rise and fall
    # selects the legal solution (which is in [0, 1])

    # Trivial solution
    if h1 == h2:
        return 0.5, 2*A - h1

    # Discriminant (will always be > 0)
    D = np.sqrt(0.5*(h1*h1 + h2*h2) + A**2 - A*(h1+h2))

    t1 = (A - h2 + D) / (h1 - h2)
    t2 = (A - h2 - D) / (h1 - h2)

    t = t1 if 0 < t1 < 1 else t2
    h = 2*A - t*h1 - (1-t)*h2

    return t, h


def plot_resampled(grads, shape, edge_points, peak_points, event_time):
    # This function can be used to debug the gradient resampling algorithm
    plot = np.zeros((grads.size*2+1, 2))

    for i in range(grads.size):
        plot[2*i+0, 0] = i
        plot[2*i+0, 1] = edge_points[i]
        plot[2*i+1, 0] = i + peak_points[i, 0]
        plot[2*i+1, 1] = peak_points[i, 1]

    plot[-1, 0] = grads.size
    plot[-1, 1] = edge_points[grads.size]

    plt.figure()
    plt.plot(plot[:, 0] * event_time, plot[:, 1], label="Calculated shape")
    # NOTE: last step is not plotted (plot ends on the last point)
    plt.step(np.arange(grads.size) * event_time, grads,
             where='post', label="Original Gradients")
    plt.step(np.arange(shape.size) * pe.system.grad_raster_time, shape,
             where='post', label="Resampled Gradients")
    plt.grid()
    plt.legend()
    plt.show()
