# %% S0. SETUP env
import sys,os
os.chdir(os.path.abspath(os.path.dirname(__file__)))  #  makes the ex folder your working directory
sys.path.append(os.path.dirname(os.getcwd()))         #  add required folders to path
mpath=os.path.dirname(os.getcwd())
c1=r'codes'; c2=r'codes\GradOpt_python'; c3=r'codes\scannerloop_libs' #  add required folders to path
sys.path += [rf'{mpath}\{c1}',rf'{mpath}\{c2}',rf'{mpath}\{c3}']

import math
import numpy as np
import torch
from matplotlib import pyplot as plt

## imports for pypulseq
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.opts import Opts

# %% LOAD and PLOT a sequence   .seq
seq = Sequence()

seq.read('out/exB02_SE_to_RARE_2D.seq')
 
seq.plot()



# %% GENERATE and WRITE a sequence   .seq
# %% S1. SETUP sys

## choose the scanner limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s', rf_ringdown_time=20e-6,
                 rf_dead_time=100e-6, adc_dead_time=20e-6,grad_raster_time=50*10e-6)

# %% S2. DEFINE the sequence 
seq = Sequence()

# Define rf events
rf, _,_ = make_sinc_pulse(flip_angle=90.0 * math.pi / 180, duration=1e-3,slice_thickness=8e-3, apodization=0.5, time_bw_product=4, system=system)
# rf1, _= make_block_pulse(flip_angle=90 * math.pi / 180, duration=1e-3, system=system)

# Define other gradients and ADC events
gx = make_trapezoid(channel='y', area=8, duration=1e-3, system=system)
adc = make_adc(num_samples=128, duration=20e-3, phase_offset=0*np.pi/180, system=system)
gx_pre = make_trapezoid(channel='x', area=-gx.area / 2, duration=5e-3, system=system)
del15=make_delay(0.015)

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
seq.set_definition('FOV', [1000, 1000, 8]) # in mm
seq.set_definition('Name', 'test')
seq.write('out/external.seq')



