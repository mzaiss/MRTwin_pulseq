from __future__ import annotations
import os
from typing import Optional
import numpy as np
from . import system
from .events import Block, TrapGradEvent


def write_sequence(seq: list[Block], file_name: str,
                   FOV: Optional[tuple[int, int, int]] = None,
                   name: Optional[str] = None) -> None:
    # Create directory if it doesn't exist yet
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    file = open(file_name, 'w')

    file.write(
        "[VERSION]\n"
        "major 1\n"
        "minor 4\n"
        "revision 0\n"
        "\n"
    )
    file.write(
        f"[DEFINITIONS]\n"
        f"GradientRasterTime {system.grad_raster_time:.7g}\n"
        f"RadiofrequencyRasterTime {system.rf_raster_time:.7g}\n"
        f"AdcRasterTime {system.adc_raster_time:.7g}\n"
        f"BlockDurationRaster {system.block_raster_time:.7g}\n"
        f"TotalDuration {sum(block.duration for block in seq):.7g}\n"
    )
    if FOV:
        file.write(f"FOV {FOV[0]} {FOV[1]} {FOV[2]}\n")
    if name:
        file.write(f"Name {name}\n")

    file.write("\n")

    rf_list = []
    grad_list = []
    adc_list = []
    shape_list = []

    # Returns a unique ID for unique obj's, removing duplicates
    # Slower than dict / set but only requires implementing __eq__
    def get_id(obj, obj_list):
        if obj is None:
            return 0
        for i, elem in enumerate(obj_list):
            if elem == obj:
                return i+1
        obj_list.append(obj)
        return len(obj_list)

    file.write("[BLOCKS]\n")
    file.write("# dur. RF GX GY GZ ADC EXT\n")
    for i, block in enumerate(seq):
        if block.rf:
            block.rf.mag_shape = get_id(block.rf.mag_shape, shape_list)
            block.rf.phase_shape = get_id(block.rf.phase_shape, shape_list)
        rf_id = get_id(block.rf, rf_list)
        gx_id = get_id(block.gx, grad_list)
        gy_id = get_id(block.gy, grad_list)
        gz_id = get_id(block.gz, grad_list)
        adc_id = get_id(block.adc, adc_list)

        file.write(
            f"{i+1} "
            f"{int(np.ceil(block.duration / system.block_raster_time)):4d} "
            f"{rf_id:2d} {gx_id:2d} {gy_id:2d} {gz_id:2d} {adc_id:3d}   0\n"
        )
    file.write("\n")

    file.write("[RF]\n")
    for i, rf in enumerate(rf_list):
        file.write(
            f"{i+1} {rf.amp:.7g} {rf.mag_shape} {rf.phase_shape} "
            f"0 {int(np.round(rf.delay*1e6))} 0 {rf.phase:.7g}\n"
        )
    file.write("\n")

    # Separate gradients into TrapGradients and ShapeGradients
    trap_grads = []
    shape_grads = []
    for i, grad in enumerate(grad_list):
        if isinstance(grad, TrapGradEvent):
            trap_grads.append((i, grad))
        else:
            shape_grads.append((i, grad))

    file.write("[GRADIENT]\n")
    for i, grad in shape_grads:
        file.write(
            f"{i+1} {grad.amp:.7g} {get_id(grad.shape, shape_list)} 0 "
            f"{int(np.round(grad.delay*1e6))}\n"
        )
    file.write("\n")

    file.write("[TRAP]\n")
    for i, grad in trap_grads:
        file.write(
            f"{i+1} {grad.amp:.7g} "
            f"{int(np.round(grad.rise*1e6))} {int(np.round(grad.flat*1e6))} "
            f"{int(np.round(grad.fall*1e6))} {int(np.round(grad.delay*1e6))}\n"
        )
    file.write("\n")

    file.write("[ADC]\n")
    for i, adc in enumerate(adc_list):
        # NOTE: dwell time is in [ns], spec says float but interpreter assumes
        # integer, which is also more consistent with other parameters
        dwell_time = int(np.round(adc.dwell * 1e9))
        file.write(
            f"{i+1} {adc.num} {dwell_time} {int(np.round(adc.delay*1e6))} "
            f"0 {adc.phase:.7g}\n"
        )
    file.write("\n")

    file.write("[SHAPES]\n")
    for i, shape in enumerate(shape_list):
        file.write(f"\nshape_id {i+1}\n")
        write_shape(shape.shape, file)
    file.write("\n")


def write_shape(shape: np.ndarray, file) -> None:
    compressed, RLE = compress_shape(shape)

    file.write(f"num_samples {len(shape)}\n")

    if len(shape) <= len(compressed):
        for sample in shape:
            file.write(f"{sample:.7g}\n")
    else:
        for sample, is_run_length in zip(compressed, RLE):
            if is_run_length:
                file.write(f"{int(sample)}\n")
            else:
                file.write(f"{sample:.7g}\n")


# Ripped from pypulseq
def compress_shape(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a run-length encoded compressed shape.

    Parameters
    ----------
    decompressed_shape : numpy.ndarray
        Decompressed shape.

    Returns
    -------
    compressed_shape : np.ndarray
        Run-length encoded deriviate of the ``shape``.
    """
    quant_factor = 1e-7
    decompressed_shape_scaled = shape / quant_factor
    datq = np.round(np.insert(np.diff(decompressed_shape_scaled), 0, decompressed_shape_scaled[0]))
    qerr = decompressed_shape_scaled - np.cumsum(datq)
    qcor = np.insert(np.diff(np.round(qerr)), 0, 0)
    datd = datq + qcor
    mask_changes = np.insert(np.asarray(np.diff(datd) != 0, dtype=np.int), 0, 1)
    vals = datd[mask_changes.nonzero()[0]] * quant_factor

    k = np.append(mask_changes, 1).nonzero()[0]
    n = np.diff(k)

    n_extra = (n - 2).astype(np.float32)  # Cast as float for nan assignment to work
    vals2 = np.copy(vals)
    vals2[n_extra < 0] = np.nan
    n_extra[n_extra < 0] = np.nan
    v = np.stack((vals, vals2, n_extra))

    # vals: quantised deriviate of shape, duplicates removed
    # n: run length of quantised deriviate
    # n_extra: run_length-2 where run_length >= 2, NaN otherwise
    # vals2: same as vals, but NaN where run_length < 1

    # Basically insert duplicated vals and run_length if run_length > 1

    v = v.T[np.isfinite(v).T]  # Use transposes to match Matlab's Fortran indexing order
    v[abs(v) < 1e-10] = 0

    # To get RLE indices we encode normal values with 0 and RLE values with 1
    vals[:] = 0
    vals2[np.isfinite(vals2)] = 0
    n_extra[np.isfinite(n_extra)] = 1

    v2 = np.stack((vals, vals2, n_extra))
    RLE = v2.T[np.isfinite(v2).T]

    return v, RLE
