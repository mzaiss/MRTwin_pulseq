# %% S0. SETUP env
import MRzeroCore as mr0
import pypulseq as pp
import numpy as np
import torch
from matplotlib import pyplot as plt

# makes the ex folder your working directory
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))


# %% LOAD and PLOT a sequence   .seq
seq = pp.Sequence()

seq.read('out/exB02_SE_to_RARE_2D.seq')

seq.plot()


# %% GENERATE and WRITE a sequence   .seq
# %% S1. SETUP sys

## choose the scanner limits
system = pp.Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s', rf_ringdown_time=20e-6,
                 rf_dead_time=100e-6, adc_dead_time=20e-6,grad_raster_time=50*10e-6)

# %% S2. DEFINE the sequence 
seq = pp.Sequence()

# Define rf events
rf, _,_ = pp.make_sinc_pulse(flip_angle=90.0 * np.pi / 180, duration=1e-3,slice_thickness=8e-3, apodization=0.5, time_bw_product=4, system=system, return_gz=True)
# rf1, _= pp.make_block_pulse(flip_angle=90 * np.pi / 180, duration=1e-3, system=system, return_gz=True)

# Define other gradients and ADC events
gx = pp.make_trapezoid(channel='y', area=8, duration=1e-3, system=system)
adc = pp.make_adc(num_samples=128, duration=20e-3, phase_offset=0*np.pi/180, system=system)
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=5e-3, system=system)
del15=pp.make_delay(0.015)

seq.add_block(del15)
seq.add_block(rf)
seq.add_block(gx_pre)
seq.add_block(adc,gx)
seq.add_block(del15)

 
seq.plot()
# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

seq.plot()

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [1, 1, 8e-3]) # in m
seq.set_definition('Name', 'test')
seq.write('out/external.seq')



