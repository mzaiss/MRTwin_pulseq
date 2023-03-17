# %% S0. SETUP env
import pypulseq as pp
import numpy as np

# makes the ex folder your working directory
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import util

# %% GENERATE and WRITE a sequence   .seq
# %% S1. SETUP sys

# choose the scanner limits
system = pp.Opts(
    max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6,
    grad_raster_time=50 * 10e-6
)

# %% S2. DEFINE the sequence
seq = pp.Sequence()

# Define rf events
rf, _, _ = pp.make_sinc_pulse(
    flip_angle=90.0 * np.pi / 180, duration=2e-3, slice_thickness=8e-3,
    apodization=0.5, time_bw_product=4, system=system, return_gz=True
)
rf2, _, _ = pp.make_sinc_pulse(
    flip_angle=120.0 * np.pi / 180, duration=2e-3, slice_thickness=8e-3,
    apodization=0.5, time_bw_product=4, system=system, return_gz=True
)
# rf1, _, _ = pp.make_block_pulse(
#     flip_angle=90 * np.pi / 180, duration=1e-3, system=system, return_gz=True
# )

# Define other gradients and ADC events
gx = pp.make_trapezoid(channel='y', area=80, duration=2e-3, system=system)
adc = pp.make_adc(num_samples=128, duration=10e-3, phase_offset=0 * np.pi / 180, system=system)
gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3, system=system)
del15 = pp.make_delay(0.0015)

seq.add_block(del15)
seq.add_block(rf)
seq.add_block(gx_pre)
seq.add_block(gx)
seq.add_block(gx)
seq.add_block(gx_pre)
seq.add_block(rf2)
seq.add_block(gx,gx_pre,adc)

# PLOT sequence
util.pulseq_plot(seq)


# %% S3. CHECK, PLOT and WRITE the sequence  as .seq
# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

seq.plot()

# Prepare the sequence output for the scanner
seq.set_definition('FOV', [1, 1, 8e-3])  # in m
seq.set_definition('Name', 'test')
seq.write('out/external.seq')



# %% LOAD and PLOT a sequence   .seq
seq = pp.Sequence()

seq.read('out/external.seq')

seq.plot()



# %% Exact pulse timing
#S1. SETUP sys and DEFINE the sequence

# choose the scanner limits
system = pp.Opts(
    max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6, rf_dead_time=100e-6,
    adc_dead_time=20e-6, grad_raster_time=50 * 10e-6
)

seq = pp.Sequence()

# Define FOV and resolution
fov = 1000e-3
Nread = 128
Nphase = 1
slice_thickness = 8e-3  # slice

# Define rf events
rf, _, _ = pp.make_sinc_pulse(
    flip_angle=1 * np.pi / 180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    delay=0.0, system=system, return_gz=True
)

rf2, _, _ = pp.make_sinc_pulse(
    flip_angle=1 * np.pi / 180, duration=2e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    delay=0.0, system=system, return_gz=True
)

# what i want: an RF pulse at exactly 0.1

rf.delay                    # delay before the start of the pulse due to dead time or gradient rise time
ct=pp.calc_rf_center(rf)    # rf center time returns time and index of the center of the pulse
ct[0]                       # this is the rf center time
rf.ringdown_time            # this is the rf ringdown time after the rf pulse
pp.calc_duration(rf)        # this is the rf duration including delay and ringdown time


print("for a pulse with requested duration 1 ms the total duration is")
print("%0.5f" %(pp.calc_duration(rf) )  )     # this is the rf duration including delay and ringdown time
print("consiting of ")
print("%0.5f" %(1e-3), "s requested duration")                # requested duration
print("%0.5f" %(rf.delay), "s rf  delay"   )               # delay before the start of the pulse due to dead time or gradient rise time
print("%0.5f" %(rf.ringdown_time), "s ringdown_time")        # this is the rf ringdown time after the rf pulse

ct=pp.calc_rf_center(rf)    # rf center time returns time and index of the center of the pulse
print("%0.5f" %(ct[0]), "s after the delay, we have the center of the pulse, thus at"  )                # this is the rf center time
print("%0.5f" %(ct[0]+rf.delay), "s after the start of the block"  )                # this is the rf center time


# delay1=pp.make_delay(.10)
# delay1=pp.make_delay(.10 - rf.delay)
delay1=pp.make_delay(.10 - rf.delay - ct[0])

# we know the pulse ends at  0.1 + pp.calc_duration(rf) -rf.delay - ct[0]
# for the second pulse we have again to remove -rf.delay - ct[0]  to center it at teh additional 0.1
# for the same pulse (same duration) this is aleady done when we just remove the first pp.calc_duration(rf)
# thus
delay2=pp.make_delay(.10 - pp.calc_duration(rf) )

# or detailed ( and for a different rf pulse)
ct2=pp.calc_rf_center(rf2)    # rf center time returns time and index of the center of the pulse
delay2=pp.make_delay(.10 - pp.calc_duration(rf) +rf.delay + ct[0]    -rf2.delay - ct2[0])


# ======
# CONSTRUCT SEQUENCE
# ======
seq.add_block(delay1)
seq.add_block(rf)
seq.add_block(delay2)
seq.add_block(rf2)


# # Bug: pypulseq 1.3.1post1 write() crashes when there is no gradient event
seq.add_block(pp.make_trapezoid('x', duration=20e-3, area=10))

# % S3. CHECK, PLOT and WRITE the sequence  as .seq
# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed. Error listing follows:')
    [print(e) for e in error_report]

# PLOT sequence
sp_adc, t_adc = util.pulseq_plot(seq)